# Phase 0 rank sweep (raw code, directions from layer 16, erased at all layers)

Baseline self-recognition AUROC: 0.786 [0.756, 0.818]

| Rank k | INLP erased | Random k-dim | Probe decoding after removal | Answered A/B (INLP) |
| --- | --- | --- | --- | --- |
| 1 | 0.705 [0.671, 0.741] | 0.787 [0.755, 0.820] | 0.972 | 1.00 |
| 2 | 0.653 [0.619, 0.688] | 0.793 [0.761, 0.825] | 0.959 | 1.00 |
| 4 | 0.565 [0.531, 0.599] | 0.794 [0.764, 0.826] | 0.926 | 1.00 |
| 8 | 0.517 [0.477, 0.555] | 0.802 [0.772, 0.836] | 0.884 | 1.00 |
| 16 | 0.449 [0.413, 0.483] | 0.805 [0.774, 0.837] | 0.808 | 1.00 |

Top principal components: k=4 0.551 [0.516, 0.589] (answered 1.00); k=16 0.608 [0.572, 0.644] (answered 1.00).
Probe decoding before any removal: 0.981.
Smallest INLP rank whose 95% CI lies inside [0.45, 0.55]: none up to 16.
