"""Combine the per-batch and pooled fresh-task results into one tests table (run from the repo root after fresh_analysis.py).
Reads paper/fresh_results.json (batch 1), fresh_results_b2.json (batch 2), fresh_results_pooled.json; writes paper/tab_fresh_tests3.tex."""
import json
R = {"Batch 1 (600 tasks)": json.load(open("paper/fresh_results.json")), "Batch 2 (1,000 tasks)": json.load(open("paper/fresh_results_b2.json")),
     "Pooled (1,600 tasks)": json.load(open("paper/fresh_results_pooled.json"))}
lines = []
for fam in ("phase-1", "most-trained"):
    for name, r in R.items():
        d = r["families"][fam]
        a, b = d["P1_sep_granite_minus_ordinary"], d["P2_safety5_granite_minus_ordinary"]
        m = d["P3_means"]
        lines.append(f"{fam.capitalize()} & {name} & {a['point']:+.3f} [{a['ci'][0]:+.3f}, {a['ci'][1]:+.3f}] & {100 * b['point']:+.1f} [{100 * b['ci'][0]:+.1f}, {100 * b['ci'][1]:+.1f}] & "
                     f"{m['granite']:.3f} / {m['qwen']:.3f} / {m['deepseek']:.3f} ({100 * d['P3_granite_lowest_share']:.0f}\\%)\\\\")
open("paper/tab_fresh_tests3.tex", "w").write("\n".join(lines).rstrip("\\") + "\n")
for name, r in R.items():
    print(name, "hthr", round(r["hthr"], 3), r["counts"]["honeypots"])
