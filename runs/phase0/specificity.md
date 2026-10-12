# Phase 0 specificity checks (raw code, layer 16, erasure at all layers)

## 1-2. Self-recognition under each erasure

| Erasure | AUROC [95% CI] | Drop [95% CI] | Answered A/B |
| --- | --- | --- | --- |
| none | 0.786 [0.756, 0.818] |  | 1.00 |
| inlp_final_k8 | 0.517 [0.477, 0.555] | 0.269 [0.236, 0.306] | 1.00 |
| inlp_final_k10 | 0.489 [0.449, 0.526] | 0.296 [0.264, 0.333] | 1.00 |
| inlp_final_k12 | 0.506 [0.467, 0.544] | 0.279 [0.247, 0.316] | 1.00 |
| pcs_k8 | 0.489 [0.454, 0.526] | 0.296 [0.256, 0.340] | 1.00 |
| pcs_k12 | 0.566 [0.533, 0.599] | 0.219 [0.181, 0.260] | 1.00 |
| inlp_code_k1 | 0.789 [0.757, 0.822] | -0.003 [-0.012, 0.005] | 1.00 |
| inlp_code_k4 | 0.787 [0.754, 0.821] | -0.001 [-0.023, 0.022] | 1.00 |
| inlp_code_k8 | 0.774 [0.739, 0.807] | 0.012 [-0.017, 0.042] | 1.00 |
| decision_final_k8 | 0.603 [0.566, 0.643] | 0.183 [0.151, 0.216] | 1.00 |

Code-token source decoding (difference of means, test) after removing k code-token directions: k=0: 0.983, k=1: 0.960, k=4: 0.862, k=8: 0.734

Principal angles, answer-token INLP-8 vs code-token INLP-8 (degrees): smallest 81.5, median 87.7.
Principal angles, answer-token INLP-8 vs decision INLP-8 (degrees): smallest 64.4, median 81.8.

## 3. Capability (191 bug pairs; next-token loss on raw test solutions)

| Erasure | Bug-detection AUROC [95% CI] | Drop [95% CI] | Answered A/B | Code NLL (change) |
| --- | --- | --- | --- | --- |
| none | 0.531 [0.520, 0.544] |  | 1.00 | 1.000 (+0.0%) |
| inlp_final_k8 | 0.519 [0.508, 0.532] | 0.012 [0.001, 0.024] | 1.00 | 1.042 (+4.2%) |
| inlp_code_k8 | 0.521 [0.512, 0.531] | 0.009 [-0.002, 0.022] | 1.00 | 1.149 (+14.9%) |
| random_k8 | 0.531 [0.521, 0.544] | -0.000 [-0.009, 0.010] | 1.00 | 1.026 (+2.6%) |
