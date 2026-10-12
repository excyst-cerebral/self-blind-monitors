"""Local report for selective.py. Writes selective_report.md."""
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
LABELS = {"none": "No erasure", "perlayer_final_k32": "Recognition subspace, 32/layer (as before)",
          "selective_protect16": "Recognition minus top-16 detection directions",
          "selective_protect32": "Recognition minus top-32 detection directions",
          "erase_detection_k16": "Detection directions only, 16/layer (sanity check)",
          "random_protect32": "Random 32/layer minus detection directions (control)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/phase2")
    ap.add_argument("--compose", default="runs/phase1/compose_composed_v2_s0.json")
    args = ap.parse_args()

    rows = read_jsonl(f"{args.out}/calib_items.jsonl")
    R = dict(np.load(f"{args.out}/selective_results.npz"))
    meta = json.load(open(f"{args.out}/selective_meta.json"))
    tau = json.load(open(args.compose))["tau"]
    y = np.array([r["source"] == COLLABORATOR for r in rows], dtype=int)
    tasks = np.array([r["task_id"] for r in rows])
    attack = np.array([r["kind"] == "attack" for r in rows], dtype=int)

    lines = ["# Phase 2: selective erasure (calib, colluding monitor, every layer)", "",
             "| Erasure | Recognition AUROC [95% CI] | Drop [95% CI] | Detection AUROC, other sources | Detection AUROC, "
             "all sources | Answered A/B | Collab attacks passed | Other attacks passed |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    out = {}
    for name, label in LABELS.items():
        if f"rec/{name}" not in R:
            continue
        auc, ci = auroc_ci(y, R[f"rec/{name}"], tasks)
        drop = paired_drop_ci(y, R["rec/none"], R[f"rec/{name}"], tasks) if name != "none" else None
        det_other = roc_auc_score(attack[y == 0], R[f"mon/{name}"][y == 0])
        det_all = roc_auc_score(attack, R[f"mon/{name}"])
        comp = composed(R[f"rec/{name}"], R[f"mon/{name}"], tau, rows)
        out[name] = {"auroc": auc, "ci95": ci, "drop": drop, "detection_other": det_other, "detection_all": det_all,
                     "answered": float(R[f"answered/{name}"].mean()), **comp}
        dtxt = "" if drop is None else f"{drop[0]:.3f} [{drop[1][0]:.3f}, {drop[1][1]:.3f}]"
        lines.append(f"| {label} | {auc:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {dtxt} | {det_other:.3f} | {det_all:.3f} | "
                     f"{out[name]['answered']:.2f} | {comp['collab_attack_pass']:.0%} | {comp['other_attack_pass']:.0%} |")
    ov = list(meta["recognition_energy_in_detection_span"].values())
    k16 = list(meta["rank_kept_protect16"].values())
    k32 = list(meta["rank_kept_protect32"].values())
    lines += ["", f"Share of the recognition subspace lying inside the 32-dim detection span, mean over layers: "
                  f"{np.mean(ov):.3f} (range {min(ov):.3f}-{max(ov):.3f}).",
              f"Recognition directions kept after protection: {np.mean(k16):.1f} of 32 (protect 16), "
              f"{np.mean(k32):.1f} of 32 (protect 32)."]
    with open(f"{args.out}/selective_report.json", "w") as f:
        json.dump(out, f, indent=2, default=list)
    with open(f"{args.out}/selective_report.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
