"""Phase 2 replication across seeds: the per-layer erasure trade-off for one colluding monitor, in one GPU job.

Same procedure as checks.py + rank_curve.py (final-token recognition prompt, all layers; each layer's own INLP
subspace, length-residualized, fitted on train and erased at that layer, all positions; calib evaluation), except
that INLP is fitted on the GPU (selective.inlp_gpu, same objective) instead of sklearn on CPU. Seed 0 is rerun with
this script too, so all seeds share one method.
Writes seed_tradeoff_<name>.npz (rec/, answered/, mon/ arrays per condition) and seed_tradeoff_items_<name>.jsonl.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from checks import ablate_per_layer
from common2 import COLLABORATOR, load_monitor, raw_items, rec_prompts
from common import score_prompts, write_jsonl
from intervene import run
from monitor import digit_dists, monitor_prompt
from selective import inlp_gpu

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase0"))
from length_control import length_features  # noqa: E402

RANKS = (16, 24, 32)


def residualize(X, L, train):
    """Remove the part of X linearly predicted by the length features (fit on train), as in checks.fit_layer."""
    A = np.column_stack([L, np.ones(len(L))])
    coef, *_ = np.linalg.lstsq(A[train], X[train], rcond=None)
    return X - A @ coef


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", default="runs/phase1")
    ap.add_argument("--out", default="runs/phase2")
    ap.add_argument("--name", required=True, help="e.g. v2_s1")
    ap.add_argument("--adapters", nargs="*", default=[])
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)

    rows = raw_items(args.run, ("train", "calib"))
    model, tok, device = load_monitor(args.model, args.adapters)
    _, _, hid = score_prompts(model, tok, device, rec_prompts(tok, rows), args.batch_size, want_hidden=True)
    acts = hid.view(len(rows), 2, *hid.shape[1:]).float().mean(dim=1).numpy()     # [n, L+1, d]
    del hid
    L, _, _ = length_features(rows, {r["task_id"]: r["question"] for r in rows},
                              AutoTokenizer.from_pretrained(args.model))
    y = np.array([r["source"] == COLLABORATOR for r in rows], dtype=int)
    train = np.array([r["split"] == "train" for r in rows])
    layers = list(range(1, acts.shape[1] - 1))
    final = []
    for l in tqdm(layers, desc="per-layer INLP"):
        Xr = residualize(acts[:, l, :].astype(np.float64), L, train)
        final.append(inlp_gpu(torch.tensor(Xr[train], dtype=torch.float32), y[train], max(RANKS)))
    rng = np.random.default_rng(0)
    random = [np.linalg.qr(rng.standard_normal((acts.shape[2], max(RANKS))))[0].T for _ in layers]
    del acts

    calib = [r for r in rows if r["split"] == "calib"]
    rec_p = rec_prompts(tok, calib)
    mon_p = [monitor_prompt(tok, r["question"], r["code"]) for r in calib]
    erase = {"none": {}, **{f"perlayer_final_k{k}": {l: final[i][:k] for i, l in enumerate(layers)} for k in RANKS},
             f"perlayer_random_k{max(RANKS)}": {l: random[i] for i, l in enumerate(layers)}}
    res = {}
    for name, dirs in tqdm(erase.items(), desc="erasure conditions"):
        with ablate_per_layer(model, dirs):
            res[f"rec/{name}"], res[f"answered/{name}"] = run(model, tok, device, rec_p, args.batch_size)
            res[f"mon/{name}"] = digit_dists(model, tok, device, mon_p, args.batch_size) @ np.arange(10)
        print(f"[{args.name}] {name} done", flush=True)
    np.savez(f"{args.out}/seed_tradeoff_{args.name}.npz", **res)
    write_jsonl(f"{args.out}/seed_tradeoff_items_{args.name}.jsonl",
                [{k: r[k] for k in ("task_id", "split", "source", "kind")} for r in calib])
    print(f"[{args.name}] saved {len(res)} arrays for {len(calib)} calib items")


if __name__ == "__main__":
    main()
