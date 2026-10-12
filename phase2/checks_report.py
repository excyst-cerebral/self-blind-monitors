"""Local report for checks.py: per-layer and code-token erasure, properly scaled steering. Writes checks_report.md."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase0"))
from analyze import auroc_ci, paired_drop_ci  # noqa: E402
from common import read_jsonl  # noqa: E402
from report import composed  # noqa: E402

COLLABORATOR = "llama"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/phase2")
    ap.add_argument("--compose", default="runs/phase1/compose_composed_v2_s0.json")
    args = ap.parse_args()

    rows = read_jsonl(f"{args.out}/calib_items.jsonl")
    R = dict(np.load(f"{args.out}/checks_results.npz"))
    meta = json.load(open(f"{args.out}/checks_meta.json"))
    tau = json.load(open(args.compose))["tau"]
    y = np.array([r["source"] == COLLABORATOR for r in rows], dtype=int)
    tasks = np.array([r["task_id"] for r in rows])
    attack = np.array([r["kind"] == "attack" for r in rows], dtype=int)
    other = y == 0

    L = ["# Phase 2 follow-up checks (calib, colluding monitor)", "",
         "## Per-layer and code-token erasure (each layer's own subspace, erased at that layer, all positions)", "",
         "| Condition | Recognition AUROC [95% CI] | Drop [95% CI] | Answered A/B | Detection AUROC, other sources |",
         "| --- | --- | --- | --- | --- |"]
    out = {}
    for key in [k for k in R if k.startswith("rec/") and not k.startswith("rec/steer")]:
        name = key[4:]
        auc, ci = auroc_ci(y, R[key], tasks)
        drop = paired_drop_ci(y, R["rec/none"], R[key], tasks) if name != "none" else None
        det = roc_auc_score(attack[other], R[f"mon/{name}"][other]) if f"mon/{name}" in R else None
        out[name] = {"auroc": auc, "ci95": ci, "drop": drop, "answered": float(R[f"answered/{name}"].mean()),
                     "detection": det}
        dtxt = "" if drop is None else f"{drop[0]:.3f} [{drop[1][0]:.3f}, {drop[1][1]:.3f}]"
        L.append(f"| {name} | {auc:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {dtxt} | {out[name]['answered']:.2f} | "
                 f"{'' if det is None else f'{det:.3f}'} |")

    L += ["", f"## Steering at layer {meta['layer']} along the collaborator-vs-other mean difference", "",
          f"Dose unit = class-mean separation along the direction ({meta['separation']:.2f}; mean residual norm "
          f"{meta['residual_norm']:.1f}). Dose -1 moves the collaborator mean onto the other-source mean.", "",
          "| Dose | Mean self-score, collaborator | Mean self-score, other | Recognition AUROC | Answered |",
          "| --- | --- | --- | --- | --- |"]
    steer = {}
    for a in (-8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0):
        key = "rec/none" if a == 0 else f"rec/steer_{a:+.1f}"
        akey = "answered/none" if a == 0 else f"answered/steer_{a:+.1f}"
        s = R[key]
        steer[a] = {"collab": float(s[y == 1].mean()), "other": float(s[y == 0].mean()),
                    "auroc": float(roc_auc_score(y, s)), "answered": float(R[akey].mean())}
        L.append(f"| {a:+.1f} | {steer[a]['collab']:.2f} | {steer[a]['other']:.2f} | {steer[a]['auroc']:.3f} | "
                 f"{steer[a]['answered']:.2f} |")

    L += ["", "## Composed collusion policy under erasure (descriptive; 25 collaborator attacks)", "",
          "| Condition | Collaborator attacks passed | Other attacks passed |", "| --- | --- | --- |"]
    comp = {}
    for key in [k for k in R if k.startswith("mon/")]:
        name = key[4:]
        comp[name] = composed(R[f"rec/{name}"], R[key], tau, rows)
        L.append(f"| {name} | {comp[name]['collab_attack_pass']:.1%} | {comp[name]['other_attack_pass']:.1%} |")

    with open(f"{args.out}/checks_report.json", "w") as f:
        json.dump({"erasure": out, "steering": {str(k): v for k, v in steer.items()}, "composed": comp, **meta}, f,
                  indent=2, default=lambda o: o if not isinstance(o, tuple) else list(o))
    with open(f"{args.out}/checks_report.md", "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
