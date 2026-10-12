# Phase 0 follow-up: length control

| | Normalized code | Raw code |
| --- | --- | --- |
| Median length, self / other (chars) | 278 / 420 | 302 / 414 |
| Tasks where self code is shorter | 79% | 68% |
| Length alone, AUROC | 0.700 [0.662, 0.738] | 0.632 [0.593, 0.670] |
| Model self-score, AUROC | 0.608 [0.572, 0.641] | 0.787 [0.757, 0.820] |
| Length + self-score, AUROC | 0.693 [0.654, 0.732] | 0.806 [0.774, 0.839] |
| Self-score on length-matched pairs, AUROC | 0.571 [0.521, 0.622] (n=298) | 0.781 [0.740, 0.822] (n=298) |
| Layer selected (length regressed out) | 1 | 16 |
| Diff-of-means AUROC at that layer, test | 0.659 | 0.947 |
| Probe AUROC at that layer, test | 0.752 | 0.981 |
| Probe AUROC at that layer, length-matched | 0.757 | 0.982 |
| Max probe AUROC over layers, test (optimistic) | 0.792 | 0.990 |
