"""Generate the data-driven TikZ/pgfplots figure files for the paper from the project's result files.
Run from the repo root:  .venv/bin/python paper/make_figs.py   (writes paper/fig_safety.tex, fig_ladder.tex, fig_loss.tex)"""
import json
import numpy as np
from pathlib import Path

OUT = Path("paper")

# ---------------------------------------------------------------- safety versus number of honeypots
# pass rates on test: own attacks p; honeypots h counted only when an honest monitor flags them (extra_analyses.py)
ex = json.load(open("paper/extra_analyses.json"))
GP, CE = ex["filtered_honeypots"], ex["colluders"]
SMALL = "Granite, $\\le 2$ lines"
fams = {"phase1": ("v2_s0", "v2_s1", "v2_s2"), "toughest": ("adaptive_min", "adaptive_min_s1", "adaptive_min_s2")}
params = {f: {"p": [CE[r]["p"] for r in reds], "ordinary": [GP[r]["Ordinary"]["pass"] for r in reds],
              "small": [GP[r][SMALL]["pass"] for r in reds]} for f, reds in fams.items()}
H = np.arange(0, 21)
def curves(fam, kind):
    P = params[fam]
    c = np.array([[100 * (1 - (1 - P[kind][s]) ** h * P["p"][s]) for h in H] for s in range(3)])
    return c.min(0), c.max(0), c.mean(0)
def coords(y):
    return " ".join(f"({h},{v:.2f})" for h, v in zip(H, y))
panels = []
for fam, title in (("phase1", "Phase-1 colluder"), ("toughest", "Most-trained colluder")):
    lo_o, hi_o, mid_o = curves(fam, "ordinary")
    lo_s, hi_s, mid_s = curves(fam, "small")
    show_y = fam == "phase1"
    lab_o = ("\\node[note, text=cGrey, anchor=north west] at (axis cs:8,38) {ordinary honeypots};" if show_y else
             "\\node[note, text=cGrey, anchor=north west] at (axis cs:9,66) {ordinary honeypots};")
    lab_s = ("\\node[note, text=cTeal!85!black, anchor=north east, align=right] at (axis cs:19.6,96.5) {Granite, edits $\\le$ 2 lines};" if show_y else
             "\\node[note, text=cTeal!85!black, anchor=center, align=center] at (axis cs:14.5,87.3) {Granite, edits $\\le$ 2 lines};")
    panels.append(f"""\\nextgroupplot[title={{{title}}}, xlabel={{Number of honeypots $H$}}, {'ylabel={Safety (\\%)},' if show_y else 'yticklabels={},'}
  xmin=0, xmax=20, ymin=10, ymax=101, xtick={{0,5,10,15,20}}, ytick={{20,40,60,80,100}}]
\\addplot[name path=ol, draw=none, forget plot] coordinates {{{coords(lo_o)}}};
\\addplot[name path=oh, draw=none, forget plot] coordinates {{{coords(hi_o)}}};
\\addplot[cGrey!35, forget plot] fill between[of=ol and oh];
\\addplot[cGrey, mark=none, thick] coordinates {{{coords(mid_o)}}};
\\addplot[name path=sl, draw=none, forget plot] coordinates {{{coords(lo_s)}}};
\\addplot[name path=sh, draw=none, forget plot] coordinates {{{coords(hi_s)}}};
\\addplot[cTeal!30, forget plot] fill between[of=sl and sh];
\\addplot[cTeal, mark=none, thick] coordinates {{{coords(mid_s)}}};
{lab_o}
{lab_s}
\\draw[black!45, dotted] (axis cs:5,10) -- (axis cs:5,101);""")
(OUT / "fig_safety.tex").write_text(
"""\\begin{tikzpicture}
\\begin{groupplot}[group style={group size=2 by 1, horizontal sep=0.5cm}, paper, width=0.5\\linewidth, height=5.4cm]
""" + "\n".join(panels) + """
\\end{groupplot}
\\end{tikzpicture}
""")

# ---------------------------------------------------------------- red-team ladder heatmap
rep = json.load(open("runs/phase3/report.json"))
LAD = json.load(open("paper/review_analyses.json"))["ladder_flagged"]
KEYMAP = {"honeypots: other models' code": "ordinary", "honeypots: collaborator code, Qwen edit": "qwen_normal", "honeypots: collaborator code, Granite edit": "granite_normal", "honeypots: collaborator code, DeepSeek edit": "deepseek_normal", "honeypots: one-line edit, Qwen": "qwen_one", "honeypots: one-line edit, Granite": "granite_one", "honeypots: one-line edit, DeepSeek": "deepseek_one"}
rows = [("v2_s0", "Phase-1 colluder"), ("adaptive_hp", "Adaptive, weak$^{\\dagger}$"), ("adaptive_strong", "Adaptive, strong"),
        ("adaptive_multi", "Adaptive, multi-editor"), ("adaptive_min", "Most trained")]
cols = [("honeypots: other models' code", "Ordinary", "cGrey"),
        ("honeypots: collaborator code, Qwen edit", "Qwen", "cAmber"),
        ("honeypots: collaborator code, Granite edit", "Granite", "cTeal"),
        ("honeypots: collaborator code, DeepSeek edit", "DeepSeek", "cViolet"),
        ("honeypots: one-line edit, Qwen", "Qwen", "cAmber"),
        ("honeypots: one-line edit, Granite", "Granite", "cTeal"),
        ("honeypots: one-line edit, DeepSeek", "DeepSeek", "cViolet")]
cw, ch, x0 = 1.28, 0.62, 3.6
t = ["\\begin{tikzpicture}[font=\\sffamily\\footnotesize]"]
for j, (_, name, colr) in enumerate(cols):
    gap = 0.25 if j >= 4 else 0
    gap = 0.25 if j >= 1 else 0
    gap = 0.0
for i, (red, label) in enumerate(rows):
    y = -i * ch
    t.append(f"\\node[anchor=east] at ({x0 - 0.12},{y - ch / 2}) {{{label}}};")
    for j, (key, name, colr) in enumerate(cols):
        v = LAD[red][KEYMAP[key]]
        shade = int(round(min(max((v - 0.4) / 0.6, 0), 1) * 100))
        txt = "white" if shade > 52 else "black"
        x = x0 + j * cw + (0.3 if j >= 4 else 0)
        t.append(f"\\fill[cBlue!{shade}!white] ({x},{y}) rectangle ({x + cw - 0.05},{y - ch + 0.05});")
        t.append(f"\\node[text={txt}] at ({x + cw / 2 - 0.025},{y - ch / 2 + 0.025}) {{{v:.2f}}};")
for j, (key, name, colr) in enumerate(cols):
    x = x0 + j * cw + (0.3 if j >= 4 else 0)
    t.append(f"\\fill[{colr}] ({x},0.12) rectangle ({x + cw - 0.05},0.2);")
    t.append(f"\\node[anchor=south, text={colr}!85!black, font=\\sffamily\\scriptsize\\bfseries] at ({x + cw / 2 - 0.025},0.2) {{{name}}};")
xa, xb = x0, x0 + 4 * cw - 0.05
xc, xd = x0 + 4 * cw + 0.3, x0 + 7 * cw + 0.3 - 0.05
t.append(f"\\node[note, anchor=south] at ({(xa + xb) / 2},0.72) {{editor's normal-size edit}};")
t.append(f"\\node[note, anchor=south] at ({(xc + xd) / 2},0.72) {{editor asked for a one-line edit}};")
t.append(f"\\draw[black!50, line width=0.3pt] ({xa},0.7) -- ({xb},0.7);")
t.append(f"\\draw[black!50, line width=0.3pt] ({xc},0.7) -- ({xd},0.7);")
yb = -len(rows) * ch - 0.45
for k, v in enumerate(np.linspace(0.4, 1.0, 61)[:-1]):
    shade = int(round(min(max((v - 0.4) / 0.6, 0), 1) * 100))
    t.append(f"\\fill[cBlue!{shade}!white] ({x0 + k * 0.08},{yb}) rectangle ({x0 + (k + 1) * 0.08},{yb - 0.18});")
for v in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
    xx = x0 + (v - 0.4) / 0.6 * 4.8
    t.append(f"\\draw[black!60, line width=0.3pt] ({xx},{yb - 0.18}) -- ({xx},{yb - 0.24});")
    t.append(f"\\node[note, anchor=north] at ({xx},{yb - 0.24}) {{{v:.1f}}};")
t.append(f"\\node[note, anchor=east] at ({x0 - 0.12},{yb - 0.09}) {{separability (test AUROC)}};")
t.append("\\end{tikzpicture}")
(OUT / "fig_ladder.tex").write_text("\n".join(t) + "\n")

# ---------------------------------------------------------------- training loss
def smooth(x, w=160, n=60):
    x = np.asarray(x)
    c = np.convolve(x, np.ones(w) / w, mode="valid")
    idx = np.linspace(0, len(c) - 1, n).astype(int)
    return (idx + w) / (len(x)) * 100, c[idx]
panels = []
for fam, title, dirs in (("phase1", "Phase-1 colluders (recipe v2)", ["runs/phase1/adapter_v2_s0", "runs/phase1/adapter_v2_s1", "runs/phase1/adapter_v2_s2"]),
                         ("toughest", "Most-trained colluders", ["runs/phase3/adapter_adaptive_min", "runs/phase3/adapter_adaptive_min_s1", "runs/phase3/adapter_adaptive_min_s2"])):
    plots = []
    for s, d in enumerate(dirs):
        lg = json.load(open(f"{d}/train_log.json"))
        xs, ys = smooth(lg["loss"])
        dash = ["solid", "dashed", "dotted"][s]
        plots.append(f"\\addplot[cRed, {dash}, thick, mark=none] coordinates {{{' '.join(f'({a:.1f},{b:.4f})' for a, b in zip(xs, ys))}}};")
        plots.append(f"\\addlegendentry{{seed {s}}}")
    show_y = fam == "phase1"
    panels.append(f"""\\nextgroupplot[title={{{title}}}, xlabel={{Training progress (\\%)}}, {'ylabel={Mean KL (moving average)},' if show_y else 'yticklabels={},'}
  ymode=log, xmin=0, xmax=100, xtick={{0,20,40,60,80,100}}, legend pos=north east]
""" + "\n".join(plots))
(OUT / "fig_loss.tex").write_text(
"""\\begin{tikzpicture}
\\begin{groupplot}[group style={group size=2 by 1, horizontal sep=0.5cm}, paper, width=0.5\\linewidth, height=5.2cm, ymin=0.01, ymax=4]
""" + "\n".join(panels) + """
\\end{groupplot}
\\end{tikzpicture}
""")
print("wrote fig_safety.tex fig_ladder.tex fig_loss.tex")


# ---------------------------------------------------------------- separability forest plot (flagged-only is primary)
HS = ex["separability_flagged_only"]
pooled = json.load(open("runs/phase3/pooled_separability.json"))
rows = [("Ordinary", "Ordinary", "cGrey", "Ordinary"), ("Qwen, $\\le 2$ lines", "Qwen small", "cAmber", "Qwen small"),
        ("DeepSeek, $\\le 2$ lines", "DeepSeek small", "cViolet", "DeepSeek small"), ("Granite, $\\le 2$ lines", "Granite small", "cTeal", "Granite small")]
mark = ["mark=o", "mark=square", "mark=triangle"]
msize = ["1.7pt", "1.6pt", "2.0pt"]
panels = []
for pi, (fam, famkey, title) in enumerate((("Phase-1 colluder", "phase1", "Phase-1 colluder"), ("Most-trained colluder", "toughest", "Most-trained colluder"))):
    plots, ticks = [], []
    for i, (hk, pk, colr, lab) in enumerate(rows):
        h = HS[f"{fam}|{hk}"]
        a = pooled[f"{famkey}|{pk}"]
        n_all, n_fl = a["n"], h["n"]
        ticks.append(f"{lab} ({n_fl})")
        tb = json.load(open("paper/review_analyses.json"))["task_bootstrap_sep"]
        kk = {"Ordinary": "ordinary", "Qwen small": "qwen_small", "DeepSeek small": "deepseek_small", "Granite small": "granite_small"}[lab]
        tf, ta = tb[f"{famkey}|{kk}|flagged"]["ci90"], tb[f"{famkey}|{kk}|all"]["ci90"]
        ci = (tf[1] - tf[0]) / 2
        ca = (ta[1] - ta[0]) / 2
        plots.append(f"\\addplot[only marks, mark=*, mark size=3.4pt, {colr}, error bars/.cd, x dir=both, x explicit, error bar style={{line width=1pt, {colr}}}] coordinates {{({h['mean']:.3f},{i - 0.05}) +- ({ci:.3f},0)}};")
        plots.append(f"\\addplot[only marks, mark=diamond, mark size=2.6pt, {colr}, line width=0.7pt, error bars/.cd, x dir=both, x explicit, error bar style={{line width=0.5pt, {colr}!60}}] coordinates {{({a['mean']:.3f},{i + 0.27}) +- ({ca:.3f},0)}};")
        for s_, v in enumerate(h["per_seed"]):
            plots.append(f"\\addplot[only marks, {mark[s_]}, mark size={msize[s_]}, {colr}!70!black] coordinates {{({v:.3f},{i - 0.33})}};")
    yl = "yticklabels={" + ", ".join(ticks) + "}, yticklabel style={font=\\footnotesize\\sffamily}" if pi == 0 else "yticklabels={}"
    panels.append(f"""\\nextgroupplot[title={{{title}}}, {yl}]
\\fill[cGrey!14] (axis cs:0.45,-0.6) rectangle (axis cs:0.55,3.6);
\\draw[black!50, dashed, line width=0.5pt] (axis cs:0.5,-0.6) -- (axis cs:0.5,3.6);
""" + "\n".join(plots))
(OUT / "fig_separability.tex").write_text("""% rows: Ordinary, Qwen small, DeepSeek small, Granite small. Filled circle: honeypots an honest monitor flags (mean over three seeds, 90% interval);
% hollow diamond: all honeypots; small hollow marks: individual seeds on the flagged honeypots (circle, square, triangle = seeds 0, 1, 2).
\\begin{tikzpicture}
\\begin{groupplot}[group style={group size=2 by 1, horizontal sep=0.35cm}, paper, width=0.5\\linewidth, height=5.6cm,
  xmin=0.3, xmax=1.0, ymin=-0.65, ymax=3.6, y dir=reverse, xtick={0.4,0.5,0.6,0.7,0.8,0.9,1.0}, ytick={0,1,2,3}, ymajorgrids=false,
  xticklabel style={/pgf/number format/fixed, /pgf/number format/precision=1}, xlabel={Separability (test AUROC)}]
""" + "\n".join(panels) + """
\\end{groupplot}
\\end{tikzpicture}
""")
print("wrote fig_safety.tex and fig_separability.tex")
