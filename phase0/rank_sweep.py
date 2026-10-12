"""Phase 0 rank sweep: how many directions must be erased before self-recognition of raw code reaches chance?

Directions are found at one layer (default 16, the layer chosen on fit_b in the length-controlled raw analysis) by
iterative nullspace projection (INLP) on length-residualized activations from the fit splits: train a linear probe,
take its weight vector as a direction, project it out, repeat. The top k directions (orthonormalized) are then
projected out of the residual stream at every layer while the model answers "who wrote this?" on raw test code.

Each rank k is compared with a random orthonormal k-dimensional subspace and with the top-k principal components
of the fit activations at the same layer. Writes rank_sweep.json and rank_sweep.md.
"""
import argparse
import json

import numpy as np
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoTokenizer

from analyze import auroc_ci, paired_drop_ci
from common import ablate, load_model, read_jsonl, recognition_prompts, score_prompts, self_scores
from length_control import length_features

RANKS = (1, 2, 4, 8, 16)
CHANCE_MARGIN = 0.05   # "at chance" = 95% CI of AUROC inside [0.45, 0.55]


def inlp_directions(X, y, k):
    """Return (k, d) orthonormal INLP directions and the probe AUROC (on the same data) before each removal."""
    dirs, P = [], np.eye(X.shape[1], dtype=np.float64)
    for _ in range(k):
        Xp = X @ P
        scaler = StandardScaler().fit(Xp)
        clf = LogisticRegression(C=0.01, max_iter=3000).fit(scaler.transform(Xp), y)
        w = clf.coef_[0] / scaler.scale_          # back to the unscaled coordinates
        w = P @ w                                  # keep it inside the remaining subspace
        for u in dirs:                             # Gram-Schmidt against earlier directions
            w -= (w @ u) * u
        w /= np.linalg.norm(w)
        dirs.append(w)
        P = P - np.outer(w, w)
    return np.array(dirs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", default="runs/phase0")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    samples = read_jsonl(f"{args.run}/samples.jsonl")
    questions = {t["task_id"]: t["question"] for t in read_jsonl(f"{args.run}/tasks.jsonl")}
    idx = np.load(f"{args.run}/acts_index_raw.npy")
    rows = [samples[i] for i in idx]
    acts = np.load(f"{args.run}/acts_raw.npy")[:, args.layer, :].astype(np.float64)
    y = np.array([r["source"] == "self" for r in rows], dtype=int)
    split = np.array([r["split"] for r in rows])
    fit, te = split != "test", split == "test"

    tok = AutoTokenizer.from_pretrained(args.model)
    L, _, _ = length_features(rows, questions, tok)
    X = acts - LinearRegression().fit(L[fit], acts[fit]).predict(L)

    k_max = max(RANKS)
    inlp = inlp_directions(X[fit], y[fit], k_max)
    # Decoding curve on test: probe AUROC after removing the first k directions (descriptive, not used for choices).
    decoding = {}
    for k in (0,) + RANKS:
        P = np.eye(X.shape[1]) - inlp[:k].T @ inlp[:k] if k else np.eye(X.shape[1])
        sc = StandardScaler().fit(X[fit] @ P)
        clf = LogisticRegression(C=0.01, max_iter=3000).fit(sc.transform(X[fit] @ P), y[fit])
        decoding[k] = float(roc_auc_score(y[te], clf.decision_function(sc.transform(X[te] @ P))))

    rng = np.random.default_rng(0)
    rand = np.linalg.qr(rng.standard_normal((X.shape[1], k_max)))[0].T
    pcs = np.linalg.svd(acts[fit] - acts[fit].mean(0), full_matrices=False)[2][:k_max]

    conditions = {"none": None}
    for k in RANKS:
        conditions[f"inlp_k{k}"] = inlp[:k]
        conditions[f"random_k{k}"] = rand[:k]
    for k in (4, 16):
        conditions[f"pcs_k{k}"] = pcs[:k]

    model, tok, device = load_model(args.model)
    test_rows = [r for r in rows if r["split"] == "test"]
    prompts = [p for r in test_rows for p in recognition_prompts(tok, questions[r["task_id"]], r["code"])]
    yt = np.array([r["source"] == "self" for r in test_rows], dtype=int)
    tt = np.array([r["task_id"] for r in test_rows])
    scores, answered = {}, {}
    for name, dirs in tqdm(conditions.items(), desc="rank sweep"):
        d = None if dirs is None else torch.tensor(dirs, dtype=torch.float32)
        with ablate(model, d, "all"):
            gaps, valid, _ = score_prompts(model, tok, device, prompts, args.batch_size)
        scores[name] = self_scores(gaps).numpy()
        answered[name] = float(valid.float().mean())

    results = {"layer": args.layer, "probe_decoding_auroc_test_after_k_removed": decoding, "conditions": {}}
    for name, s in scores.items():
        point, ci = auroc_ci(yt, s, tt)
        # A rank only counts as reaching chance if the model still answers A/B: a broken model also scores 0.5.
        keeps_format = answered[name] >= 0.9 * answered["none"]
        e = {"auroc": point, "ci95": ci, "answered_letter": answered[name], "keeps_format": bool(keeps_format),
             "at_chance": bool(keeps_format and 0.5 - CHANCE_MARGIN <= ci[0] and ci[1] <= 0.5 + CHANCE_MARGIN)}
        if name != "none":
            e["auroc_drop"], e["auroc_drop_ci95"] = paired_drop_ci(yt, scores["none"], s, tt)
        results["conditions"][name] = e
    at_chance = [k for k in RANKS if results["conditions"][f"inlp_k{k}"]["at_chance"]]
    results["smallest_inlp_rank_at_chance"] = min(at_chance) if at_chance else None

    with open(f"{args.run}/rank_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    c = results["conditions"]
    fmt = lambda e: f"{e['auroc']:.3f} [{e['ci95'][0]:.3f}, {e['ci95'][1]:.3f}]"
    lines = [f"# Phase 0 rank sweep (raw code, directions from layer {args.layer}, erased at all layers)", "",
             f"Baseline self-recognition AUROC: {fmt(c['none'])}", "",
             "| Rank k | INLP erased | Random k-dim | Probe decoding after removal | Answered A/B (INLP) |",
             "| --- | --- | --- | --- | --- |"]
    for k in RANKS:
        lines.append(f"| {k} | {fmt(c[f'inlp_k{k}'])} | {fmt(c[f'random_k{k}'])} | {decoding[k]:.3f} | "
                     f"{c[f'inlp_k{k}']['answered_letter']:.2f} |")
    lines += ["", f"Top principal components: k=4 {fmt(c['pcs_k4'])} (answered {c['pcs_k4']['answered_letter']:.2f}); "
                  f"k=16 {fmt(c['pcs_k16'])} (answered {c['pcs_k16']['answered_letter']:.2f}).",
              f"Probe decoding before any removal: {decoding[0]:.3f}.",
              f"Smallest INLP rank whose 95% CI lies inside [0.45, 0.55]: "
              f"{results['smallest_inlp_rank_at_chance'] or 'none up to 16'}."]
    with open(f"{args.run}/rank_sweep.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
