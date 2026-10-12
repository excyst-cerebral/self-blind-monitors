# Self-blind monitors

Code, pre-registered plans and result data for a study of passive self-recognition collusion in untrusted monitoring
(Llama-3-8B-Instruct, backdoored APPS solutions) and two defences: erasing the self-recognition representation from the
monitor, and style-matched honeypots.

## Findings in brief

- **Collusion.** A monitor fine-tuned to recognize its own code, composed with an honest backdoor detector, cuts safety from
  94.9% to 26.0 ± 9.0% (three seeds), close to the 26.4% reported with GPT-4.1 by prior work.
- **Erasure.** None of the linear erasures tried (INLP and a closed-form LEACE eraser, at every position or only at the answer
  position) removed self-recognition while keeping both backdoor detection and general capability intact, in three seeds.
  Fresh probes still decode authorship after every erasure.
- **Style-matched honeypots** (a different model inserts a small backdoor into the untrusted model's own clean code, keeping only
  honeypots an honest monitor flags). On 1,600 fresh tasks with the pipeline frozen in advance, against the most-trained
  colluder: separability of the self-score 0.73 for Granite's small edits against 0.91 for ordinary honeypots; safety with five
  honeypots 82% against 79% at the colluder's fixed threshold (+2.8 points, 95% interval +1.0 to +4.7) and about 20 points
  higher when the colluder optimizes its threshold for the game. Against colluders that never saw the honeypots, +41 points.

## Layout

| Path | Contents |
| --- | --- |
| `phase0/` | Pilot: native self-recognition of the base model and INLP erasure of it |
| `phase1/` | Tasks, attack generation, sandboxed verification, honest detector, collusion fine-tuning, composed policy |
| `phase2/` | Erasure of the colluding monitor: INLP, LEACE, controls, safety under erasure |
| `phase3/` | Style-matched honeypots, adaptive colluders, editor selection, analysis |
| `phase5/` | Fresh-task construction for the confirmation (analysis code is in `paper/`) |
| `modal_phase0.py` ... `modal_phase5.py` | GPU/CPU launchers (Modal); each stage saves outputs and resumes |
| `colab_seeds.py` | Orchestration used for the seed-1 and seed-2 replication |
| `paper/` | Analysis scripts that produce every number and table in the paper (`*.py`), their outputs (`*.json`, `tab_*.tex`) and figure sources (`fig_*.tex`) |
| `runs/` | Result data: generations, verified attacks and honeypots, monitor scores, reports |
| `PHASE*_PLAN.md`, `PHASE*_FINDINGS.md`, `DEVIATIONS.md` | Plans written before each stage (including the two fresh-task confirmations), findings, and every change from the original plan |

## Reproducing the analyses

Install `requirements.txt`. The analyses in `paper/` read the saved data in `runs/` and need no GPU, for example:

```bash
python phase3/analyze.py                                               # honeypot separability and safety (development data)
python paper/fresh_analysis.py --run runs/phase5 runs/phase5b --tag _pooled   # confirmation on 1,600 fresh tasks
python paper/fresh_attackers.py                                        # game-optimized, selective and budget attackers
python paper/erasure_controls_report.py                                # erasure controls
python paper/erasure_safety_report.py                                  # safety under answer-position LEACE
```

The generation, fine-tuning and scoring stages need a GPU and are launched by the `modal_phase*.py` scripts, for example
`modal run modal_phase5.py --run-name phase5b --n 1000 --seed 6 --exclude /root/runs/phase5/tasks.jsonl --no-train`.
Model-written code is executed only by `phase1/execute.py`, which should be run inside a disposable sandbox.

## What is and is not included

- Included: all code, plans and findings, and the result data needed by the analysis scripts (generations, verified attacks
  and honeypots, honest-detector and colluder scores, reports).
- Not included: trained LoRA adapter weights (only their training logs; recreate with `phase1/finetune.py` and the recipes in
  `modal_phase1.py` and `modal_phase3.py`), and large activation arrays (`*.npy`, `*.npz`; recreate with the `collect.py`
  scripts). The pilot-stage and erasure analyses that read those arrays therefore need to be rerun from the GPU stages.
- `runs/phase5*/tasks.jsonl` keep only task ids; the task text is regenerated deterministically by
  `phase5/make_fresh_tasks.py` (seeds 5 and 6). The two most-trained colluders of seeds 1 and 2 used in the fresh-task
  confirmation were retrained with the same recipe because the original adapter files had not been kept.
- The tasks come from the public APPS dataset (`codeparrot/apps`). The backdoored programs are crude, model-written
  examples (mostly magic-value checks) produced for detection research.
