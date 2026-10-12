"""Local report for honest_check.py, side by side with the colluding monitor. Writes honest_check_report.md."""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase0"))
from analyze import auroc_ci  # noqa: E402
from common import read_jsonl  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/phase2")
COLLABORATOR = "llama"
LABELS = {"none": "No erasure", "source_recprompt_k16": "Source subspace (recognition prompt), 16/layer",
          "source_recprompt_k32": "Source subspace (recognition prompt), 32/layer",
          "source_monprompt_k32": "Source subspace (monitoring prompt), 32/layer", "random_k32": "Random, 32/layer"}


def main():
    rows = read_jsonl(OUT / "calib_items.jsonl")
    y = np.array([r["source"] == COLLABORATOR for r in rows], dtype=int)
    tasks = np.array([r["task_id"] for r in rows])
    attack = np.array([r["kind"] == "attack" for r in rows], dtype=int)
    H = dict(np.load(OUT / "honest_check_results.npz"))
    C = dict(np.load(OUT / "checks_results.npz"))
    colluding = {"none": "none", "source_recprompt_k16": "perlayer_final_k16",
                 "source_recprompt_k32": "perlayer_final_k32", "random_k32": "perlayer_random_k32"}

    def stats(R, name):
        auc, ci = auroc_ci(y, R[f"rec/{name}"], tasks)
        det = roc_auc_score(attack[y == 0], R[f"mon/{name}"][y == 0]) if f"mon/{name}" in R else None
        return auc, ci, det, float(R[f"answered/{name}"].mean())

    lines = ["# Is the recognition-detection entanglement inherent? (calib, erasure at every layer)", "",
             "| Erasure | Honest monitor: recognition AUROC | Honest monitor: detection AUROC, other sources | "
             "Colluding monitor: recognition | Colluding monitor: detection |", "| --- | --- | --- | --- | --- |"]
    out = {}
    for name, label in LABELS.items():
        auc, ci, det, ans = stats(H, name)
        out[name] = {"honest_recognition": auc, "honest_recognition_ci95": ci, "honest_detection": det,
                     "honest_answered": ans}
        c_txt = ["-", "-"]
        if name in colluding and f"rec/{colluding[name]}" in C:
            cauc, _, cdet, _ = stats(C, colluding[name])
            out[name].update(colluding_recognition=cauc, colluding_detection=cdet)
            c_txt = [f"{cauc:.3f}", "-" if cdet is None else f"{cdet:.3f}"]
        lines.append(f"| {label} | {auc:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {'-' if det is None else f'{det:.3f}'} | "
                     f"{c_txt[0]} | {c_txt[1]} |")
    meta = json.load(open(OUT / "honest_check_meta.json"))
    ang = list(meta["smallest_angle_recprompt_vs_monprompt_k8_deg"].values())
    lines += ["", f"Smallest principal angle between the recognition-prompt and monitoring-prompt source subspaces "
                  f"(rank 8), median over layers: {np.median(ang):.1f} deg."]
    (OUT / "honest_check_report.json").write_text(json.dumps(out, indent=2, default=list))
    (OUT / "honest_check_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
