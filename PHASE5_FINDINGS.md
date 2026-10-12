# Phase 5 findings: the honeypot pipeline on 600 fresh tasks

Plan frozen in PHASE5_PLAN.md before any fresh data existed. Code: `phase5/make_fresh_tasks.py`, `modal_phase5.py`,
`paper/fresh_analysis.py`; results `paper/fresh_results.json`, `runs/phase5/`.

Data: APPS test file, 600 tasks, 398 with a running collaborator solution, 66 verified own attacks (14 on a fully correct clean
solution), 321 ordinary attacks. Honest-detector filter keeps 315 ordinary, 50/89/12 small-edit honeypots (Granite/Qwen/DeepSeek).
Seed-1 and seed-2 most-trained colluders were retrained on Modal (same recipe, seed); the Colab adapters were not kept.

| Hypothesis | Most-trained colluder | Phase-1 colluder |
| --- | --- | --- |
| P1 separability, Granite small minus ordinary (earlier: 0.64 vs 0.90) | -0.162 [-0.210, -0.114] (0.735 vs 0.897). Holds | -0.313 [-0.368, -0.259] (0.60 vs 0.91). Holds |
| P2 safety H=5, Granite small minus ordinary (earlier +9 to +10 points) | +2.8 [-0.6, +7.5] (77% vs 74%). **Not confirmed** | +33.5 [+26.2, +39.4] (94% vs 61%). Holds |
| P3 Granite lowest of three editors | 0.735 vs Qwen 0.834, DeepSeek 0.863; lowest in 100% of resamples. Holds | 0.598 vs 0.636, 0.719; 88%. Holds |

The most-trained colluder passes about 27% of its own attacks and 0-3% of honeypots of any kind, so safety is close to 1-p
whichever honeypots are used. The development-data safety gain (83-88% vs 74-78%) did not replicate.

## Second batch (1,000 tasks, plan PHASE5B_PLAN.md) and pooled result

1,000 new test-file tasks (seed 6, excluding batch 1), same frozen pipeline, no retraining.
151 verified own attacks (39 on a fully correct clean solution); pooled over both batches 217 own attacks, 53 on a correct clean solution.

| Most-trained colluder | Batch 1 | Batch 2 | Pooled (1,600 tasks) |
| --- | --- | --- | --- |
| P1 separability, Granite small minus ordinary | -0.162 [-0.210, -0.114] | -0.186 [-0.226, -0.147] | -0.173 [-0.203, -0.145] |
| P2 safety H=5, Granite small minus ordinary (points) | +2.8 [-0.6, +7.5] | +2.7 [+0.7, +5.2] | +2.8 [+1.0, +4.7] |
| P3 Granite lowest of three editors | 100% | 96% | 100% |

Phase-1 colluders: P2 +33.5 / +44.5 / +40.6 points. The safety gain against a trained colluder is real but about a third of the
development estimate; safety is close to 1-p (p about 0.22) whichever honeypots are used.
Command: `.venv/bin/python paper/fresh_analysis.py --run runs/phase5 runs/phase5b --tag _pooled`, then `paper/fresh_tables.py`.

## Optimized and selective attackers on the fresh tasks (paper/fresh_attackers.py)

Pooled 1,600 fresh tasks, five folds by task, flagged honeypots, H=5, most-trained colluder (three seeds):
- Fixed tau: ordinary 79-80%, Granite small 80-85% (+2.8 on average).
- Tau optimized for the game, N=1: ordinary 56-66%, Granite small 80-84% (+20 points; development data 17-24).
- N=5: 24-27% vs 43-48% (+20). N=20: 4-5% vs 13-30% (+15).
Phase-1 colluders: N=1 +34 points. Budget-constrained: saturates as on the development data.
Consequence: the small fixed-threshold gain is specific to that threshold.
