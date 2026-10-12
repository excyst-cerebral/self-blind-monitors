"""Tables for the erasure controls (run from the repo root):  .venv/bin/python paper/erasure_controls_report.py
Reads runs/phase2/erasure_controls_v2_s{0,1,2}.json (phase2/erasure_controls.py). Writes paper/tab_ec_main.tex,
paper/tab_ec_probe.tex and paper/erasure_controls_summary.json (table bodies; last row without trailing \\\\)."""
import json
from pathlib import Path

import numpy as np

RUNS, OUT = Path("runs/phase2"), Path("paper")
files = {s: RUNS / f"erasure_controls_v2_s{s}.json" for s in (0, 1, 2)}
D = {}
for s, f in files.items():
    if f.exists():
        D[s] = json.load(open(f))
        extra = RUNS / f"erasure_controls_v2_s{s}_final.json"      # seed 0: the answer-position conditions ran as a second job
        if extra.exists():
            D[s].update({k: v for k, v in json.load(open(extra)).items() if k != "none"})
seeds = sorted(D)
print("seeds with results:", seeds)
CONDS = [("none", "No erasure"), ("inlp_k32", "INLP, 32 per layer"), ("leace_k1", "LEACE, 1 per layer"),
         ("leace_k8", "LEACE iterated, 8 per layer"), ("leace_k32", "LEACE iterated, 32 per layer"),
         ("random_k32", "Random, 32 per layer"), ("final_inlp_k32", "INLP, 32 per layer, answer position only"),
         ("final_leace_k1", "LEACE, 1 per layer, answer position only"), ("final_leace_k8", "LEACE iterated, 8 per layer, answer position only"),
         ("final_leace_k32", "LEACE iterated, 32 per layer, answer position only")]
rel = lambda s, c, k: 100 * (D[s][c][k] / D[s]["none"][k] - 1)
lines = []
for c, name in CONDS:
    rec = " / ".join(f"{D[s][c]['recognition']:.2f}" for s in seeds)
    of = " / ".join(f"{D[s][c]['recognition_orientation_free']:.2f}" for s in seeds)
    det = " / ".join(f"{D[s][c]['detection_other_sources']:.2f}" for s in seeds)
    ans = " / ".join(f"{D[s][c]['answered']:.2f}" for s in seeds)
    lt = " / ".join(f"{rel(s, c, 'loss_text'):+.1f}" for s in seeds)
    lc = " / ".join(f"{rel(s, c, 'loss_code'):+.1f}" for s in seeds)
    lines.append(f"{name} & {rec} & {of} & {det} & {ans} & {lt} & {lc}\\\\")
(OUT / "tab_ec_main.tex").write_text("\n".join(lines).rstrip("\\") + "\n")
layers = sorted({int(l) for l in D[seeds[0]]["none"]["probe"]})
lines = []
for c, name in CONDS:
    cells = " & ".join(" / ".join(f"{D[s][c]['probe'][str(l)]:.2f}" for s in seeds) for l in layers)
    lines.append(f"{name} & {cells}\\\\")
(OUT / "tab_ec_probe.tex").write_text("\n".join(lines).rstrip("\\") + "\n")
summary = {c: {k: [D[s][c][k] for s in seeds] for k in ("recognition", "recognition_orientation_free", "detection_other_sources", "answered")}
           | {"probe": {str(l): [D[s][c]["probe"][str(l)] for s in seeds] for l in layers},
              "loss_text_rel_pct": [rel(s, c, "loss_text") for s in seeds], "loss_code_rel_pct": [rel(s, c, "loss_code") for s in seeds]}
           for c, _ in CONDS}
(OUT / "erasure_controls_summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps({c: {k: [round(x, 3) for x in v] for k, v in d.items() if k in ("recognition", "recognition_orientation_free", "detection_other_sources", "loss_text_rel_pct", "loss_code_rel_pct")} for c, d in summary.items()}, indent=1))
print("probe layers", layers)
for c, d in summary.items():
    print(c, {l: [round(x, 2) for x in v] for l, v in d["probe"].items()})
