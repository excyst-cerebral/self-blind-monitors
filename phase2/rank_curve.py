"""Phase 2: fill the per-layer rank gap (20, 24, 28 directions per layer) in the colluding monitor.

Uses the per-layer final-token recognition subspaces from checks.py (checks_dirs.npz). Together with ranks 16 and 32
from checks.py, this traces the trade-off between removing recognition and keeping backdoor detection.
Writes rank_curve_results.npz.
"""
import argparse

import numpy as np
from tqdm import tqdm

from checks import ablate_per_layer
from common2 import load_monitor, raw_items, rec_prompts
from intervene import run
from monitor import digit_dists, monitor_prompt

RANKS = (20, 24, 28)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", default="runs/phase1")
    ap.add_argument("--out", default="runs/phase2")
    ap.add_argument("--adapters", nargs="*", default=[])
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    D = np.load(f"{args.out}/checks_dirs.npz")
    layers = [int(l) for l in D["layers"]]
    model, tok, device = load_monitor(args.model, args.adapters)
    rows = raw_items(args.run, ("calib",))
    rec_p = rec_prompts(tok, rows)
    mon_p = [monitor_prompt(tok, r["question"], r["code"]) for r in rows]
    res = {}
    for k in tqdm(RANKS, desc="per-layer rank"):
        dirs = {l: D["final"][i][:k] for i, l in enumerate(layers)}
        with ablate_per_layer(model, dirs):
            res[f"rec/perlayer_final_k{k}"], res[f"answered/perlayer_final_k{k}"] = run(model, tok, device, rec_p,
                                                                                         args.batch_size)
            res[f"mon/perlayer_final_k{k}"] = digit_dists(model, tok, device, mon_p, args.batch_size) @ np.arange(10)
    np.savez(f"{args.out}/rank_curve_results.npz", **res)
    print(f"saved {len(res)} result arrays")


if __name__ == "__main__":
    main()
