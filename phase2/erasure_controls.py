"""Phase 2 controls for the erasure negative result, in one GPU job for one colluding monitor.

Addresses what the INLP experiments did not show:
  1. A closed-form eraser: LEACE (Belrose et al. 2023) for the binary source concept, fitted at every layer and applied at
     every layer and position like the INLP erasure; plus an iterated version (re-fit after each rank-1 step, k rounds).
  2. Fresh post-intervention probes: after each erasure, train a new logistic probe on the erased activations at several
     layers (train split) and evaluate it on calibration. Behavioral recognition can fall while information remains.
  3. General capability controls under each erasure: next-token loss on English text (APPS problem statements) and on code
     (Llama's clean solutions), compared with no erasure.
  4. Recognition reported orientation-free, max(AUROC, 1 - AUROC), next to the signed value.
Conditions: none, inlp_k32 (INLP, 32 per layer), leace_k1, leace_k8, leace_k32 (iterated rank-1 LEACE), random_k32.
Writes erasure_controls_<name>.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoTokenizer

from checks import ablate_per_layer
from common2 import COLLABORATOR, load_monitor, raw_items, rec_prompts
from common import decoder_layers, read_jsonl, score_prompts, self_scores
from monitor import digit_dists, monitor_prompt
from selective import inlp_gpu

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase0"))
from length_control import length_features  # noqa: E402

PROBE_LAYERS = (8, 16, 24, 31)


def leace_steps(X, y, k, ridge=1e-2):
    """k rounds of rank-1 LEACE for a binary concept y on X (n, d). Each round returns (mu, u, v) with
    x' = x - u (v . (x - mu)); the next round is fitted on the already-erased X. LEACE (Belrose et al. 2023) for a binary
    concept: P = S_xz (S^-1 S_xz)^T / (S_xz^T S^-1 S_xz), the oblique projection that makes the first moments of X equal
    in the two classes with minimal change. S is shrunk with a ridge because n < d."""
    X = X.clone().double()
    out = []
    z = y.float() - y.float().mean()
    for _ in range(k):
        mu = X.mean(0)
        Xc = X - mu
        n, d = Xc.shape
        S = (Xc.T @ Xc) / n
        S = S + ridge * S.diagonal().mean() * torch.eye(d, device=X.device)
        sxz = (Xc * z[:, None]).mean(0)
        Sinv_sxz = torch.linalg.solve(S, sxz)
        denom = torch.dot(sxz, Sinv_sxz)
        u, v = sxz / denom, Sinv_sxz
        out.append((mu.float(), u.float(), v.float()))
        X = X - torch.outer(Xc @ v, u)
    return out


@torch.no_grad()
def lm_loss(model, tok, device, texts, bs, max_len=384):
    """Mean next-token cross-entropy over the given texts (padding masked)."""
    total, count = 0.0, 0
    tok.padding_side = "right"
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
        logits = model(**enc).logits[:, :-1].float()
        tgt, mask = enc["input_ids"][:, 1:], enc["attention_mask"][:, 1:].bool()
        ce = torch.nn.functional.cross_entropy(logits.transpose(1, 2), tgt, reduction="none")
        total += float(ce[mask].sum())
        count += int(mask.sum())
    tok.padding_side = "left"
    return total / max(count, 1)


class AffineErase:
    """Forward hooks applying, at every layer in `steps_by_layer`, the sequence x' = x - u (v . (x - mu)) at all positions."""

    def __init__(self, model, steps_by_layer, last_only=False):
        self.model, self.steps, self.hooks, self.last_only = model, steps_by_layer, [], last_only

    def __enter__(self):
        p = next(self.model.parameters())
        for l, steps in self.steps.items():
            st = [(mu.to(p.device).float(), u.to(p.device).float(), v.to(p.device).float()) for mu, u, v in steps]

            def hook(_m, _i, out, st=st):
                h = out[0] if isinstance(out, tuple) else out
                if self.last_only:                       # only the final (answer) position, where the readout happens
                    hf = h[:, -1, :].float()
                    for mu, u, v in st:
                        hf = hf - ((hf - mu) @ v)[..., None] * u
                    h = h.clone()
                    h[:, -1, :] = hf.to(h.dtype)
                else:
                    hf = h.float()
                    for mu, u, v in st:
                        hf = hf - ((hf - mu) @ v)[..., None] * u
                    h = hf.to(h.dtype)
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            self.hooks.append(decoder_layers(self.model)[l - 1].register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self.hooks:
            h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", default="runs/phase1")
    ap.add_argument("--out", default="runs/phase2")
    ap.add_argument("--name", required=True)
    ap.add_argument("--adapters", nargs="*", default=[])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: use only this many items")
    ap.add_argument("--conditions", choices=("all", "final"), default="all",
                    help="all: every condition; final: only the answer-position erasures (plus the baseline)")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)

    rows = raw_items(args.run, ("train", "calib"))
    if args.limit:
        rows = rows[::max(1, len(rows) // args.limit)]
    model, tok, device = load_monitor(args.model, args.adapters)
    y = np.array([r["source"] == COLLABORATOR for r in rows], dtype=int)
    train = np.array([r["split"] == "train" for r in rows])
    calib = ~train
    rec_p = rec_prompts(tok, rows)
    cal_rows = [r for r in rows if r["split"] == "calib"]
    mon_p = [monitor_prompt(tok, r["question"], r["code"]) for r in cal_rows]
    attack = np.array([r["kind"] == "attack" for r in cal_rows], dtype=int)
    cal_y = y[calib]
    other = cal_y == 0
    tasks = {t["task_id"]: t for t in read_jsonl(f"{args.run}/tasks.jsonl")}
    texts_en = [tasks[t]["question"][:1500] for t in list(tasks)[:150]]
    texts_code = [r["clean_code"] for r in read_jsonl(f"{args.run}/verified.jsonl")
                  if r["source"] == COLLABORATOR and r["clean_runs_ok"]][:150]
    tok_len = AutoTokenizer.from_pretrained(args.model)
    L, _, _ = length_features(rows, {r["task_id"]: r["question"] for r in rows}, tok_len)

    def evaluate(cond, steps_ctx):
        """Recognition on train+calib prompts (hidden states kept), detection, capability, fresh probes."""
        with steps_ctx:
            gaps, valid, hid = score_prompts(model, tok, device, rec_p, args.batch_size, want_hidden=True)
            det = digit_dists(model, tok, device, mon_p, args.batch_size) @ np.arange(10)
            loss_en = lm_loss(model, tok, device, texts_en, args.batch_size)
            loss_code = lm_loss(model, tok, device, texts_code, args.batch_size)
        s = self_scores(gaps).numpy()
        acts = hid.view(len(rows), 2, *hid.shape[1:]).float().mean(1)       # (n, L+1, d)
        rec = float(roc_auc_score(cal_y, s[calib]))
        res = {"recognition": rec, "recognition_orientation_free": max(rec, 1 - rec),
               "detection_other_sources": float(roc_auc_score(attack[other], det[other])),
               "answered": float(valid.view(-1, 2).float().mean()), "loss_text": loss_en, "loss_code": loss_code, "probe": {}}
        for l in PROBE_LAYERS:
            Xl = acts[:, l, :].numpy().astype(np.float64)
            sc = StandardScaler().fit(Xl[train])
            clf = LogisticRegression(C=0.01, max_iter=3000).fit(sc.transform(Xl[train]), y[train])
            res["probe"][str(l)] = float(roc_auc_score(cal_y, clf.decision_function(sc.transform(Xl[calib]))))
        print(cond, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if k != "probe"}, res["probe"], flush=True)
        return res, acts

    results = {}
    results["none"], acts = evaluate("none", ablate_per_layer(model, {}))
    layers = list(range(1, acts.shape[1] - 1))
    Xtr_all = {l: acts[:, l, :].to(device) for l in layers}

    # INLP, 32 per layer (length regressed out, as in seed_tradeoff.py), GPU
    A_len = torch.cat([torch.ones(len(L), 1), torch.as_tensor(L, dtype=torch.float32)], dim=1)
    trn = torch.as_tensor(train)
    inlp = {}
    for l in tqdm(layers, desc="INLP"):
        X = acts[:, l, :].float()
        beta = torch.linalg.lstsq(A_len[trn], X[trn]).solution
        Xr = X - A_len @ beta
        inlp[l] = inlp_gpu(Xr[trn], y[train], 32)
    # LEACE: rank-1 and iterated, fitted on the unedited train activations of each layer
    leace = {}
    for l in tqdm(layers, desc="LEACE"):
        leace[l] = leace_steps(Xtr_all[l][trn.to(device)].float(), torch.as_tensor(y[train], device=device), 32)
    zero = torch.zeros(acts.shape[2])
    inlp_steps = {l: [(zero, torch.as_tensor(d, dtype=torch.float32), torch.as_tensor(d, dtype=torch.float32)) for d in inlp[l]] for l in layers}

    if args.conditions == "all":
        results["inlp_k32"], _ = evaluate("inlp_k32", ablate_per_layer(model, {l: inlp[l] for l in layers}))
        for k in (1, 8, 32):
            results[f"leace_k{k}"], _ = evaluate(f"leace_k{k}", AffineErase(model, {l: leace[l][:k] for l in layers}))
        rng = np.random.default_rng(0)
        rnd = {l: np.linalg.qr(rng.standard_normal((acts.shape[2], 32)))[0].T for l in layers}
        results["random_k32"], _ = evaluate("random_k32", ablate_per_layer(model, rnd))
    # the same erasures applied only at the answer position of every layer
    results["final_inlp_k32"], _ = evaluate("final_inlp_k32", AffineErase(model, inlp_steps, last_only=True))
    for k in (1, 8, 32):
        results[f"final_leace_k{k}"], _ = evaluate(f"final_leace_k{k}", AffineErase(model, {l: leace[l][:k] for l in layers}, last_only=True))

    Path(f"{args.out}/erasure_controls_{args.name}.json").write_text(json.dumps(results, indent=1))
    print(f"[{args.name}] saved erasure_controls_{args.name}.json")


if __name__ == "__main__":
    main()
