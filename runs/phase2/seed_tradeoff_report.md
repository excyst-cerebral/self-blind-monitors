# Phase 2 across seeds: per-layer erasure in the colluding monitor (calib)

| Seed | Condition | Recognition AUROC [95% CI] | Detection AUROC, other sources | Answered A/B |
| --- | --- | --- | --- | --- |
| 0 | none | 0.958 [0.936, 0.976] | 0.993 | 1.00 |
| 0 | perlayer_final_k16 | 0.936 [0.906, 0.963] | 0.977 | 1.00 |
| 0 | perlayer_final_k24 | 0.920 [0.883, 0.954] | 0.929 | 1.00 |
| 0 | perlayer_final_k32 | 0.281 [0.225, 0.338] | 0.524 | 0.84 |
| 0 | perlayer_random_k32 | 0.951 [0.925, 0.972] | 0.986 | 1.00 |
| 1 | none | 0.957 [0.932, 0.977] | 0.990 | 1.00 |
| 1 | perlayer_final_k16 | 0.939 [0.907, 0.966] | 0.969 | 0.44 |
| 1 | perlayer_final_k24 | 0.762 [0.716, 0.809] | 0.809 | 1.00 |
| 1 | perlayer_final_k32 | 0.464 [0.411, 0.515] | 0.550 | 1.00 |
| 1 | perlayer_random_k32 | 0.957 [0.933, 0.977] | 0.978 | 1.00 |
| 2 | none | 0.959 [0.938, 0.976] | 0.985 | 1.00 |
| 2 | perlayer_final_k16 | 0.925 [0.894, 0.952] | 0.997 | 1.00 |
| 2 | perlayer_final_k24 | 0.774 [0.724, 0.822] | 0.838 | 1.00 |
| 2 | perlayer_final_k32 | 0.649 [0.598, 0.702] | 0.598 | 0.98 |
| 2 | perlayer_random_k32 | 0.956 [0.932, 0.975] | 0.990 | 1.00 |
