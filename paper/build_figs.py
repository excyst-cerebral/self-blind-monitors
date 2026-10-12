"""Render every fig_*.tex to a vector PDF (fig_*.pdf) so the paper compiles quickly without TikZ.
Run from paper/:  python3 build_figs.py   (needs tectonic)"""
import re
import subprocess
from pathlib import Path

NAMES = ["overview", "arch", "collusion", "erasure", "honeypot", "ladder", "separability", "safety", "rank", "steer", "edit", "loss", "scores", "roc"]
REFS = {"sec:erasure": "5", "sec:honeypots": "6"}      # \ref targets inside figures, as numbered in the paper
WRAP = r"""\documentclass[border=2pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{mathptmx,booktabs,xcolor}
\input{style}
\makeatletter\renewcommand{\ref}[1]{\@nameuse{refval@#1}}\makeatother
%s
\begin{document}
\setlength{\textwidth}{468pt}\setlength{\linewidth}{468pt}
%s
\end{document}
"""
for n in NAMES:
    body = Path(f"fig_{n}.tex").read_text()
    body = body.replace("\\resizebox{\\linewidth}{!}{%\n", "")
    if "\\end{tikzpicture}}" in body:
        body = body.replace("\\end{tikzpicture}}", "\\end{tikzpicture}")
    body = body.replace("\\cite{gardner2026}", "[2]")
    defs = "\n".join(f"\\expandafter\\def\\csname refval@{k}\\endcsname{{{v}}}" for k, v in REFS.items())
    Path(f"_w_{n}.tex").write_text(WRAP % (defs, body))
    r = subprocess.run(["tectonic", "-X", "compile", f"_w_{n}.tex"], capture_output=True, text=True)
    if r.returncode:
        print(n, "FAILED\n", r.stderr[-1500:])
        continue
    Path(f"_w_{n}.pdf").replace(f"fig_{n}.pdf")
    Path(f"_w_{n}.tex").unlink()
    print("built", n)
