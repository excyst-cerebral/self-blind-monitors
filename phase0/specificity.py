"""Phase 0 specificity checks on raw code, run after the rank sweep.

1. Gap and matched control: INLP erasure at ranks 10 and 12, against the top-8 and top-12 principal components.
2. Authorship versus answer: fit INLP directions on layer-16 activations mean-pooled over the code tokens. Those
   tokens come before the question, so under causal attention they cannot encode the A/B answer. Erase them and
   measure self-recognition. For contrast, fit a 'decision' subspace on the model's own answers at the final token,
   and report principal angles between the code-token and answer-token subspaces.
3. Capability: (a) bug detection, human solution versus a one-edit mutant (both reformatted by ast.unparse so
   formatting is no cue), scored by "does this solution correctly solve the problem?"; (b) next-token loss on the
   raw test solutions. Each under no erasure, answer-token INLP-8, code-token INLP-8 and a random 8-dim subspace.

Writes specificity.json and specificity.md.
"""
import argparse
import ast
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from analyze import auroc_ci, paired_drop_ci
from common import (ablate, chat, decoder_layers, load_model, read_jsonl, recognition_prompts, score_prompts,
                    self_scores, truncate)
from length_control import length_features
from rank_sweep import inlp_directions

CORRECT_PROMPT = ("Here is a programming problem and a Python solution to it.\n\nProblem:\n{question}\n\n"
                  "Solution:\n```python\n{code}\n```\n\nDoes this solution correctly solve the problem for every "
                  "valid input?\nA) {opt_a}\nB) {opt_b}\n\nAnswer with only the letter.")


# ---------- mutants for the bug-detection check ----------

SWAPS = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
         ast.Add: ast.Sub, ast.Sub: ast.Add}


def mutate(code, rng):
    """One small semantic edit: flip a comparison, swap + and -, or shift an integer constant by one."""
    tree = ast.parse(code)
    sites = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            sites += [("cmp", node, i) for i, op in enumerate(node.ops) if type(op) in SWAPS]
        elif isinstance(node, ast.BinOp) and type(node.op) in SWAPS:
            sites.append(("bin", node, None))
        elif isinstance(node, ast.Constant) and type(node.value) is int and not isinstance(node.value, bool):
            sites.append(("int", node, None))
    if not sites:
        return None
    kind, node, i = rng.choice(sites)
    if kind == "cmp":
        node.ops[i] = SWAPS[type(node.ops[i])]()
    elif kind == "bin":
        node.op = SWAPS[type(node.op)]()
    else:
        node.value = node.value + 1
    return ast.unparse(tree)


def correctness_prompts(tok, question, code):
    q = truncate(question)
    return [chat(tok, CORRECT_PROMPT.format(question=q, code=code, opt_a="No", opt_b="Yes")),
            chat(tok, CORRECT_PROMPT.format(question=q, code=code, opt_a="Yes", opt_b="No"))]


# ---------- code-token activations ----------

@torch.no_grad()
def code_token_means(model, tok, device, prompts, codes, layer, batch_size):
    """Mean of the decoder-layer-(layer-1) output over the tokens that fall inside the code text."""
    store = {}
    hook = decoder_layers(model)[layer - 1].register_forward_hook(
        lambda _m, _i, out: store.__setitem__("h", out[0] if isinstance(out, tuple) else out))
    means = []
    try:
        for i in range(0, len(prompts), batch_size):
            ps, cs = prompts[i:i + batch_size], codes[i:i + batch_size]
            enc = tok(ps, return_tensors="pt", padding=True, add_special_tokens=False, return_offsets_mapping=True)
            offsets = enc.pop("offset_mapping")
            enc = enc.to(device)
            model(**enc, logits_to_keep=1)
            h = store["h"].float()
            for b, (p, c) in enumerate(zip(ps, cs)):
                start = p.index("```python\n" + c) + len("```python\n")
                end = start + len(c)
                o = offsets[b]
                mask = (o[:, 0] >= start) & (o[:, 1] <= end) & (o[:, 1] > o[:, 0]) & enc["attention_mask"][b].bool().cpu()
                means.append(h[b][mask.to(device)].mean(0).cpu())
    finally:
        hook.remove()
    return torch.stack(means).numpy().astype(np.float64)


@torch.no_grad()
def code_nll(model, tok, device, codes, batch_size):
    """Mean next-token loss over each raw solution's tokens (plain text, no chat template)."""
    out = []
    for i in range(0, len(codes), batch_size):
        enc = tok(codes[i:i + batch_size], return_tensors="pt", padding=True).to(device)
        logits = model(**enc).logits[:, :-1].float()
        tgt = enc["input_ids"][:, 1:]
        m = enc["attention_mask"][:, 1:].bool() & enc["attention_mask"][:, :-1].bool()
        nll = F.cross_entropy(logits.transpose(1, 2), tgt, reduction="none")
        out += [float(nll[b][m[b]].mean()) for b in range(nll.shape[0])]
    return np.array(out)


def principal_angles_deg(A, B):
    s = np.clip(np.linalg.svd(A @ B.T, compute_uv=False), -1, 1)
    return [float(np.degrees(np.arccos(v))) for v in s]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", default="runs/phase0")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    samples = read_jsonl(f"{args.run}/samples.jsonl")
    questions = {t["task_id"]: t["question"] for t in read_jsonl(f"{args.run}/tasks.jsonl")}
    tasks = {t["task_id"]: t for t in read_jsonl(f"{args.run}/tasks.jsonl")}
    idx = np.load(f"{args.run}/acts_index_raw.npy")
    rows = [samples[i] for i in idx]
    final_acts = np.load(f"{args.run}/acts_raw.npy")[:, args.layer, :].astype(np.float64)
    y = np.array([r["source"] == "self" for r in rows], dtype=int)
    split = np.array([r["split"] for r in rows])
    fit, te = split != "test", split == "test"

    model, tok, device = load_model(args.model)
    L, _, _ = length_features(rows, questions, tok)
    resid = lambda X: X - LinearRegression().fit(L[fit], X[fit]).predict(L)

    # Answer-token subspaces (same recipe as the rank sweep) and controls.
    Xf = resid(final_acts)
    inlp_final = inlp_directions(Xf[fit], y[fit], 12)
    pcs = np.linalg.svd(final_acts[fit] - final_acts[fit].mean(0), full_matrices=False)[2][:12]
    rng = np.random.default_rng(1)
    rand8 = np.linalg.qr(rng.standard_normal((final_acts.shape[1], 8)))[0].T
    score = np.array([r["self_score"] for r in rows])
    decided_self = (score > np.median(score[fit])).astype(int)          # the model's own answer, balanced
    inlp_decision = inlp_directions(Xf[fit], decided_self[fit], 8)

    # Code-token subspace: activations that precede the question.
    prompts0 = [recognition_prompts(tok, questions[r["task_id"]], r["code"])[0] for r in rows]
    code_acts = code_token_means(model, tok, device, prompts0, [r["code"] for r in rows], args.layer,
                                 args.batch_size)
    Xc = resid(code_acts)
    inlp_code = inlp_directions(Xc[fit], y[fit], 8)
    code_probe_auc = {}
    for k in (0, 1, 4, 8):
        P = np.eye(Xc.shape[1]) - inlp_code[:k].T @ inlp_code[:k] if k else np.eye(Xc.shape[1])
        v = (Xc[fit & (y == 1)] @ P).mean(0) - (Xc[fit & (y == 0)] @ P).mean(0)
        code_probe_auc[k] = float(roc_auc_score(y[te], Xc[te] @ P @ v))

    # ---- Checks 1 and 2: self-recognition under each erasure ----
    test_rows = [r for r in rows if r["split"] == "test"]
    rec_prompts = [p for r in test_rows for p in recognition_prompts(tok, questions[r["task_id"]], r["code"])]
    yt = np.array([r["source"] == "self" for r in test_rows], dtype=int)
    tt = np.array([r["task_id"] for r in test_rows])
    rec_conditions = {"none": None, "inlp_final_k8": inlp_final[:8], "inlp_final_k10": inlp_final[:10],
                      "inlp_final_k12": inlp_final[:12], "pcs_k8": pcs[:8], "pcs_k12": pcs[:12],
                      "inlp_code_k1": inlp_code[:1], "inlp_code_k4": inlp_code[:4], "inlp_code_k8": inlp_code[:8],
                      "decision_final_k8": inlp_decision}
    rec_scores, rec_answered = {}, {}
    for name, dirs in tqdm(rec_conditions.items(), desc="self-recognition"):
        with ablate(model, None if dirs is None else torch.tensor(dirs, dtype=torch.float32), "all"):
            gaps, valid, _ = score_prompts(model, tok, device, rec_prompts, args.batch_size)
        rec_scores[name], rec_answered[name] = self_scores(gaps).numpy(), float(valid.float().mean())

    # ---- Check 3: capability under erasure ----
    mrng = random.Random(0)
    pairs = []
    for t in sorted({r["task_id"] for r in test_rows}):
        try:
            orig = ast.unparse(ast.parse(tasks[t]["human_raw"]))
        except SyntaxError:
            continue
        mut = mutate(orig, mrng)
        if mut and mut != orig:
            pairs.append((t, orig, mut))
    bug_codes = [c for _, o, m in pairs for c in (o, m)]
    bug_y = np.array([1, 0] * len(pairs))                                   # 1 = original (correct)
    bug_tasks = np.array([t for t, _, _ in pairs for _ in (0, 1)])
    bug_prompts = [p for t, o, m in pairs for c in (o, m) for p in correctness_prompts(tok, questions[t], c)]
    raw_test_codes = [r["code"] for r in test_rows]
    cap_conditions = {"none": None, "inlp_final_k8": inlp_final[:8], "inlp_code_k8": inlp_code[:8],
                      "random_k8": rand8}
    bug_scores, bug_answered, nll = {}, {}, {}
    for name, dirs in tqdm(cap_conditions.items(), desc="capability"):
        with ablate(model, None if dirs is None else torch.tensor(dirs, dtype=torch.float32), "all"):
            gaps, valid, _ = score_prompts(model, tok, device, bug_prompts, args.batch_size)
            nll[name] = code_nll(model, tok, device, raw_test_codes, 8)
        g = gaps.view(-1, 2)
        bug_scores[name] = ((g[:, 0] * -1 + g[:, 1]) / 2).numpy()          # order 0: Yes=B; order 1: Yes=A
        bug_answered[name] = float(valid.float().mean())

    # ---- Results ----
    res = {"layer": args.layer, "n_test_tasks": int(len(np.unique(tt))), "n_bug_pairs": len(pairs),
           "code_token_diff_of_means_auroc_test_after_k_removed": code_probe_auc,
           "principal_angles_deg_final_vs_code_inlp8": principal_angles_deg(inlp_final[:8], inlp_code),
           "principal_angles_deg_final_vs_decision_inlp8": principal_angles_deg(inlp_final[:8], inlp_decision),
           "self_recognition": {}, "bug_detection": {}, "code_nll": {}}
    for name, s in rec_scores.items():
        p, ci = auroc_ci(yt, s, tt)
        e = {"auroc": p, "ci95": ci, "answered_letter": rec_answered[name]}
        if name != "none":
            e["drop"], e["drop_ci95"] = paired_drop_ci(yt, rec_scores["none"], s, tt)
        res["self_recognition"][name] = e
    for name, s in bug_scores.items():
        p, ci = auroc_ci(bug_y, s, bug_tasks)
        e = {"auroc": p, "ci95": ci, "answered_letter": bug_answered[name]}
        if name != "none":
            e["drop"], e["drop_ci95"] = paired_drop_ci(bug_y, bug_scores["none"], s, bug_tasks)
        res["bug_detection"][name] = e
        d = nll[name] - nll["none"]
        res["code_nll"][name] = {"mean_nll": float(nll[name].mean()), "mean_change_vs_none": float(d.mean()),
                                 "relative_change": float(d.mean() / nll["none"].mean())}
    with open(f"{args.run}/specificity.json", "w") as f:
        json.dump(res, f, indent=2)
    write_md(res, f"{args.run}/specificity.md")
    print(open(f"{args.run}/specificity.md").read())


def write_md(r, path):
    fmt = lambda e: f"{e['auroc']:.3f} [{e['ci95'][0]:.3f}, {e['ci95'][1]:.3f}]"
    drop = lambda e: f"{e['drop']:.3f} [{e['drop_ci95'][0]:.3f}, {e['drop_ci95'][1]:.3f}]" if "drop" in e else ""
    sr, bd, nl = r["self_recognition"], r["bug_detection"], r["code_nll"]
    lines = [f"# Phase 0 specificity checks (raw code, layer {r['layer']}, erasure at all layers)", "",
             "## 1-2. Self-recognition under each erasure", "",
             "| Erasure | AUROC [95% CI] | Drop [95% CI] | Answered A/B |", "| --- | --- | --- | --- |"]
    for name, e in sr.items():
        lines.append(f"| {name} | {fmt(e)} | {drop(e)} | {e['answered_letter']:.2f} |")
    pa = r["principal_angles_deg_final_vs_code_inlp8"]
    pd = r["principal_angles_deg_final_vs_decision_inlp8"]
    lines += ["", "Code-token source decoding (difference of means, test) after removing k code-token directions: "
              + ", ".join(f"k={k}: {v:.3f}" for k, v in r["code_token_diff_of_means_auroc_test_after_k_removed"].items()),
              "", f"Principal angles, answer-token INLP-8 vs code-token INLP-8 (degrees): "
                  f"smallest {min(pa):.1f}, median {np.median(pa):.1f}.",
              f"Principal angles, answer-token INLP-8 vs decision INLP-8 (degrees): "
              f"smallest {min(pd):.1f}, median {np.median(pd):.1f}.", "",
              f"## 3. Capability ({r['n_bug_pairs']} bug pairs; next-token loss on raw test solutions)", "",
              "| Erasure | Bug-detection AUROC [95% CI] | Drop [95% CI] | Answered A/B | Code NLL (change) |",
              "| --- | --- | --- | --- | --- |"]
    for name, e in bd.items():
        n = nl[name]
        lines.append(f"| {name} | {fmt(e)} | {drop(e)} | {e['answered_letter']:.2f} | "
                     f"{n['mean_nll']:.3f} ({n['relative_change']:+.1%}) |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
