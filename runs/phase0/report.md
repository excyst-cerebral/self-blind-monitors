# Phase 0 report

Tasks: {'fit_a': 100, 'fit_b': 98, 'test': 196}

## Behavioral self-recognition (test split)

| Variant | AUROC [95% CI] | Answered A/B |
| --- | --- | --- |
| Normalized | 0.608 [0.572, 0.641] | 1.00 |
| Raw | 0.787 [0.757, 0.820] | 1.00 |

## Linear decodability

Layer selected on fit_b: **2**. Difference-of-means AUROC on test at that layer: 0.709; logistic probe: 0.814.

Cosine of target direction with length direction: -0.903; with PC1: -0.847.

## Ablation on the test split (normalized code)

| Condition | AUROC [95% CI] | Drop [95% CI] | Answered A/B |
| --- | --- | --- | --- |
| none | 0.609 [0.573, 0.644] |  | 1.00 |
| target_all_layers | 0.590 [0.552, 0.627] | 0.019 [0.004, 0.034] | 1.00 |
| target_single_layer | 0.600 [0.562, 0.636] | 0.009 [-0.005, 0.024] | 1.00 |
| length_all_layers | 0.572 [0.535, 0.607] | 0.037 [0.020, 0.054] | 1.00 |
| pc1_all_layers | 0.572 [0.535, 0.608] | 0.037 [0.019, 0.056] | 1.00 |
| random (1 of 10 shown) | 0.610 [0.574, 0.644] | -0.001 [-0.011, 0.009] | 1.00 |

Random controls: mean drop 0.001, max 0.008.
Target drop minus best informed control (pc1_all_layers): -0.018 [-0.032, -0.005].
Informed controls excluded for breaking the answer format: none.

## Gate

- code_source_signal_measurable: **True**
- ablation_effect_measurable: **True**
- beats_all_random_controls: **True**
- beats_best_informed_control: **False**
- keeps_answer_format: **True**
- proceed: **False**

H1a pilot calibration: suggested threshold 0.70 (to be locked by the team).
