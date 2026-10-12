# Phase 2: selective erasure (calib, colluding monitor, every layer)

| Erasure | Recognition AUROC [95% CI] | Drop [95% CI] | Detection AUROC, other sources | Detection AUROC, all sources | Answered A/B | Collab attacks passed | Other attacks passed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| No erasure | 0.958 [0.936, 0.976] |  | 0.993 | 0.973 | 1.00 | 68% | 3% |
| Recognition subspace, 32/layer (as before) | 0.296 [0.240, 0.348] | 0.662 [0.606, 0.721] | 0.549 | 0.547 | 0.96 | 92% | 77% |
| Recognition minus top-16 detection directions | 0.311 [0.258, 0.365] | 0.646 [0.591, 0.704] | 0.545 | 0.532 | 0.99 | 92% | 90% |
| Recognition minus top-32 detection directions | 0.377 [0.320, 0.431] | 0.581 [0.524, 0.640] | 0.528 | 0.538 | 0.91 | 92% | 72% |
| Detection directions only, 16/layer (sanity check) | 0.953 [0.929, 0.973] | 0.005 [-0.001, 0.012] | 0.856 | 0.719 | 1.00 | 96% | 16% |
| Random 32/layer minus detection directions (control) | 0.955 [0.932, 0.974] | 0.003 [-0.004, 0.010] | 0.995 | 0.990 | 1.00 | 76% | 3% |

Share of the recognition subspace lying inside the 32-dim detection span, mean over layers: 0.051 (range 0.008-0.206).
Recognition directions kept after protection: 32.0 of 32 (protect 16), 32.0 of 32 (protect 32).
