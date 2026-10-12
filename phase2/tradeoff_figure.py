"""Figure: self-recognition vs backdoor detection as per-layer erasure grows (calib). Reads runs/phase2/tradeoff.json."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = Path("runs/phase2")
COLLUDING, HONEST = "#2a78d6", "#eb6834"              # categorical slots 1 and 2 (validated pair)
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

d = json.loads((OUT / "tradeoff.json").read_text())
curve = [p for p in d["colluding_curve"] if p["detection"] is not None]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

# Where a working defence would land: monitor still detects (>= 0.95) and cannot recognise itself (0.45-0.55).
ax.add_patch(Rectangle((0.95, 0.45), 0.06, 0.10, facecolor=GRID, edgecolor="none", zorder=0))
ax.text(0.978, 0.565, "goal: useful and\nself-blind", ha="center", va="bottom", fontsize=8.5, color=MUTED)

xs, ys = [p["detection"] for p in curve], [p["recognition"] for p in curve]
ax.plot(xs, ys, color=COLLUDING, lw=2, zorder=2)
ax.scatter(xs, ys, s=64, color=COLLUDING, edgecolor=SURFACE, linewidth=2, zorder=3, label="Colluding monitor")
offsets = {0: (-12, 4), 20: (-8, -14), 24: (4, 8), 28: (8, 0), 32: (8, -2)}
for p in curve:
    dx, dy = offsets.get(p["k"], (8, 0))
    ax.annotate(f"{p['k']}/layer" if p["k"] else "no erasure", (p["detection"], p["recognition"]),
                textcoords="offset points", xytext=(dx, dy), ha="right" if dx < 0 else "left", fontsize=8.5,
                color=INK)

hx, hy = [p["detection"] for p in d["honest"]], [p["recognition"] for p in d["honest"]]
ax.plot(hx, hy, color=HONEST, lw=2, ls=(0, (4, 3)), zorder=2)
ax.scatter(hx, hy, s=64, marker="s", color=HONEST, edgecolor=SURFACE, linewidth=2, zorder=3,
           label="Honest monitor (never trained to recognise itself)")
for p in d["honest"]:
    if p["k"] == 32:
        ax.annotate("32/layer", (p["detection"], p["recognition"]), textcoords="offset points", xytext=(8, -12),
                    fontsize=8.5, color=INK)

for key, m in (("colluding", "o"), ("honest", "s")):
    r = d["random"][key]
    ax.scatter([r["detection"]], [r["recognition"]], s=56, marker=m, facecolor="none", edgecolor=MUTED, linewidth=1.5,
               zorder=3, label="Random erasure, same size (control)" if key == "colluding" else None)

ax.axhline(0.5, color=MUTED, lw=1, ls=":", zorder=1)
ax.text(0.505, 0.505, "chance", fontsize=8, color=MUTED, va="bottom")
ax.set_xlim(0.5, 1.01)
ax.set_ylim(0.25, 1.0)
ax.set_xlabel("Backdoor detection, other sources' attacks (AUROC)", color=INK)
ax.set_ylabel("Self-recognition (AUROC)", color=INK)
ax.set_title("Erasing self-recognition breaks detection before recognition falls", loc="left", fontsize=11.5,
             color=INK, pad=12)
ax.grid(color=GRID, lw=0.8)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(MUTED)
ax.tick_params(colors=MUTED)
ax.legend(loc="lower right", frameon=False, fontsize=8.5)
fig.text(0.01, 0.005, "Calib split, Llama-3-8B monitors. Each layer's own recognition subspace (INLP) is erased at "
         "that layer, all positions.", fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(OUT / "tradeoff.png", facecolor=SURFACE)
print(f"wrote {OUT / 'tradeoff.png'}")
