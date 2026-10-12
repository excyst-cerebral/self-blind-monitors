"""Phase 0 follow-up: is the source signal anything more than length?

The first analysis picked layer 2 and a direction with cosine -0.90 to the code-length direction. At early layers the
final-token residual stream encodes absolute position, i.e. the prompt's token count, which tracks code length. This
script re-asks the phase 0 questions with length controlled, using only saved activations (no GPU):

1. How well does length alone predict the source?
2. Does the model's own self-recognition score add anything beyond length?
3. Is the source still linearly decodable after (a) regressing length out of the activations and (b) restricting
   the test set to length-matched self/other pairs?

Writes runs/phase0/length_control.json and length_control.md.
"""
import argparse
import json

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer

from analyze import auroc_ci
from common import read_jsonl, recognition_prompts


def unit(v):
    return v / (np.linalg.norm(v) + 1e-8)


def length_features(rows, questions, tok):
    chars = np.array([len(r["code"]) for r in rows], dtype=float)
    tokens = np.array([len(tok(recognition_prompts(tok, questions[r["task_id"]], r["code"])[0],
                               add_special_tokens=False)["input_ids"]) for r in rows], dtype=float)
    return np.column_stack([np.log(chars), np.log(tokens)]), chars, tokens


def matched_test_subset(y, tokens, test_mask, tol=0.03, seed=0):
    """Greedy one-to-one matching of test self/other samples on prompt token count (within tol, relative)."""
    rng = np.random.default_rng(seed)
    selves = [i for i in np.where(test_mask & (y == 1))[0]]
    others = list(np.where(test_mask & (y == 0))[0])
    rng.shuffle(selves)
    keep = []
    for i in selves:
        if not others:
            break
        j = min(others, key=lambda k: abs(tokens[k] - tokens[i]))
        if abs(tokens[j] - tokens[i]) <= tol * tokens[i]:
            keep += [i, j]
            others.remove(j)
    return np.array(sorted(keep))


def analyze_variant(variant, run, tok, questions):
    samples = read_jsonl(f"{run}/samples.jsonl")
    idx = np.load(f"{run}/acts_index_{variant}.npy")
    rows = [samples[i] for i in idx]
    acts = np.load(f"{run}/acts_{variant}.npy").astype(np.float32)
    y = np.array([r["source"] == "self" for r in rows], dtype=int)
    split = np.array([r["split"] for r in rows])
    task = np.array([r["task_id"] for r in rows])
    score = np.array([r["self_score"] for r in rows])
    L, chars, tokens = length_features(rows, questions, tok)
    fa, fb, te = split == "fit_a", split == "fit_b", split == "test"
    fit = fa | fb
    out = {"variant": variant}

    # 1. Length alone.
    out["length_stats"] = {
        "median_chars_self": float(np.median(chars[y == 1])), "median_chars_other": float(np.median(chars[y == 0])),
        "frac_tasks_self_shorter": float(np.mean([chars[(task == t) & (y == 1)][0] < chars[(task == t) & (y == 0)][0]
                                                  for t in np.unique(task)]))}
    clf_len = LogisticRegression().fit(L[fit], y[fit])
    p_len = clf_len.decision_function(L)
    out["length_only_auroc_test"] = auroc_ci(y[te], p_len[te], task[te])

    # 2. Does the model's self-score add anything beyond length?
    out["self_score_auroc_test"] = auroc_ci(y[te], score[te], task[te])
    both = np.column_stack([L, score])
    clf_both = make_pipeline(StandardScaler(), LogisticRegression()).fit(both[fit], y[fit])
    out["length_plus_self_score_auroc_test"] = auroc_ci(y[te], clf_both.decision_function(both[te]), task[te])
    m = matched_test_subset(y, tokens, te)
    out["n_matched_test_samples"] = int(len(m))
    out["self_score_auroc_length_matched"] = (auroc_ci(y[m], score[m], task[m]) if len(m) >= 40 else None)

    # 3. Decodability with length controlled.
    layers = list(range(1, acts.shape[1] - 1))
    per_layer, best_fb, best_layer = {}, -1, None
    for l in layers:
        X = acts[:, l, :]
        X_res = X - LinearRegression().fit(L[fit], X[fit]).predict(L)   # regress length out, fitted on fit only
        v_a = unit(X_res[fa & (y == 1)].mean(0) - X_res[fa & (y == 0)].mean(0))
        auc_fb = roc_auc_score(y[fb], X_res[fb] @ v_a)
        v = unit(X_res[fit & (y == 1)].mean(0) - X_res[fit & (y == 0)].mean(0))
        probe = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000)).fit(X_res[fit], y[fit])
        pr = probe.decision_function(X_res)
        len_dir = unit(X[fit][chars[fit] > np.median(chars[fit])].mean(0)
                       - X[fit][chars[fit] <= np.median(chars[fit])].mean(0))
        per_layer[l] = {"dom_auroc_fit_b": float(auc_fb),
                        "dom_auroc_test": float(roc_auc_score(y[te], X_res[te] @ v)),
                        "probe_auroc_test": float(roc_auc_score(y[te], pr[te])),
                        "probe_auroc_matched": float(roc_auc_score(y[m], pr[m])) if len(m) >= 40 else None,
                        "cos_dir_vs_raw_length_dir": float(v @ len_dir)}
        if auc_fb > best_fb:
            best_fb, best_layer = auc_fb, l
    out["per_layer_length_residualized"] = per_layer
    out["layer_selected_on_fit_b"] = best_layer
    b = per_layer[best_layer]
    out["selected_layer_summary"] = b
    out["max_probe_auroc_test_any_layer"] = max(v["probe_auroc_test"] for v in per_layer.values())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="../runs/phase0")
    ap.add_argument("--tokenizer", default="NousResearch/Meta-Llama-3-8B-Instruct")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    questions = {t["task_id"]: t["question"] for t in read_jsonl(f"{args.run}/tasks.jsonl")}
    results = {v: analyze_variant(v, args.run, tok, questions) for v in ("normalized", "raw")}
    with open(f"{args.run}/length_control.json", "w") as f:
        json.dump(results, f, indent=2)

    ci = lambda e: f"{e[0]:.3f} [{e[1][0]:.3f}, {e[1][1]:.3f}]"
    lines = ["# Phase 0 follow-up: length control", "",
             "| | Normalized code | Raw code |", "| --- | --- | --- |"]
    n, r = results["normalized"], results["raw"]
    rowfmt = lambda label, f: lines.append(f"| {label} | {f(n)} | {f(r)} |")
    rowfmt("Median length, self / other (chars)",
           lambda x: f"{x['length_stats']['median_chars_self']:.0f} / {x['length_stats']['median_chars_other']:.0f}")
    rowfmt("Tasks where self code is shorter", lambda x: f"{x['length_stats']['frac_tasks_self_shorter']:.0%}")
    rowfmt("Length alone, AUROC", lambda x: ci(x["length_only_auroc_test"]))
    rowfmt("Model self-score, AUROC", lambda x: ci(x["self_score_auroc_test"]))
    rowfmt("Length + self-score, AUROC", lambda x: ci(x["length_plus_self_score_auroc_test"]))
    rowfmt("Self-score on length-matched pairs, AUROC",
           lambda x: ci(x["self_score_auroc_length_matched"]) + f" (n={x['n_matched_test_samples']})"
           if x["self_score_auroc_length_matched"] else f"too few matches (n={x['n_matched_test_samples']})")
    rowfmt("Layer selected (length regressed out)", lambda x: str(x["layer_selected_on_fit_b"]))
    rowfmt("Diff-of-means AUROC at that layer, test", lambda x: f"{x['selected_layer_summary']['dom_auroc_test']:.3f}")
    rowfmt("Probe AUROC at that layer, test", lambda x: f"{x['selected_layer_summary']['probe_auroc_test']:.3f}")
    rowfmt("Probe AUROC at that layer, length-matched",
           lambda x: f"{x['selected_layer_summary']['probe_auroc_matched']:.3f}"
           if x["selected_layer_summary"]["probe_auroc_matched"] is not None else "n/a")
    rowfmt("Max probe AUROC over layers, test (optimistic)", lambda x: f"{x['max_probe_auroc_test_any_layer']:.3f}")
    with open(f"{args.run}/length_control.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
