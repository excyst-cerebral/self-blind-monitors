"""Local report for seed_tradeoff.py: does per-layer erasure break detection before recognition, in every seed?

For each colluding monitor (v2_s0, v2_s1, v2_s2) and erasure condition (calib): self-recognition AUROC (collaborator vs
other sources, all calib items, task-clustered bootstrap CI), backdoor-detection AUROC on other sources' code
(attack vs benign), and the A/B answer rate. Writes seed_tradeoff_report.md and seed_tradeoff_report.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase0"))
from analyze import auroc_ci  # noqa: E402
from common import read_jsonl  # noqa: E402

COLLABORATOR = "llama"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/phase2")
    args = ap.parse_args()
    report, L = {}, ["# Phase 2 across seeds: per-layer erasure in the colluding monitor (calib)", "",
                     "| Seed | Condition | Recognition AUROC [95% CI] | Detection AUROC, other sources | Answered A/B |",
                     "| --- | --- | --- | --- | --- |"]
    for s in (0, 1, 2):
        name = f"v2_s{s}"
        path = Path(f"{args.out}/seed_tradeoff_{name}.npz")
        if not path.exists():
            continue
        R = dict(np.load(path))
        rows = read_jsonl(f"{args.out}/seed_tradeoff_items_{name}.jsonl")
        y = np.array([r["source"] == COLLABORATOR for r in rows], dtype=int)
        tasks = np.array([r["task_id"] for r in rows])
        attack = np.array([r["kind"] == "attack" for r in rows], dtype=int)
        other = y == 0
        report[name] = {}
        for cond in [k[4:] for k in R if k.startswith("rec/")]:
            auc, ci = auroc_ci(y, R[f"rec/{cond}"], tasks)
            det = float(roc_auc_score(attack[other], R[f"mon/{cond}"][other]))
            ans = float(R[f"answered/{cond}"].mean())
            report[name][cond] = {"recognition": float(auc), "ci95": [float(c) for c in ci], "detection": det,
                                  "answered": ans}
            L.append(f"| {s} | {cond} | {auc:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {det:.3f} | {ans:.2f} |")
    (Path(args.out) / "seed_tradeoff_report.json").write_text(json.dumps(report, indent=2))
    (Path(args.out) / "seed_tradeoff_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
