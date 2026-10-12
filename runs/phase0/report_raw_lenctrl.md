# Phase 0 report (secondary: raw code), length-controlled direction

Tasks: {'fit_a': 100, 'fit_b': 98, 'test': 196}

## Behavioral self-recognition (test split)

| Variant | AUROC [95% CI] | Answered A/B |
| --- | --- | --- |
| Normalized | 0.608 [0.572, 0.641] | 1.00 |
| Raw | 0.787 [0.757, 0.820] | 1.00 |

## Linear decodability

Layer selected on fit_b: **16**. Difference-of-means AUROC on test at that layer: 0.947; logistic probe: 0.981.

Cosine of target direction with length direction: -0.181; with PC1: -0.085.

## Ablation on the test split (raw code)

| Condition | AUROC [95% CI] | Drop [95% CI] | Answered A/B |
| --- | --- | --- | --- |
| none | 0.786 [0.756, 0.818] |  | 1.00 |
| target_all_layers | 0.610 [0.575, 0.646] | 0.176 [0.153, 0.200] | 1.00 |
| target_single_layer | 0.632 [0.597, 0.668] | 0.154 [0.136, 0.174] | 1.00 |
| length_all_layers | 0.703 [0.672, 0.735] | 0.083 [0.060, 0.105] | 1.00 |
| pc1_all_layers | 0.753 [0.724, 0.784] | 0.033 [0.018, 0.047] | 1.00 |
| target_orth_length_all_layers | 0.653 [0.618, 0.689] | 0.132 [0.111, 0.155] | 1.00 |
| random (1 of 10 shown) | 0.784 [0.752, 0.818] | 0.002 [-0.008, 0.010] | 1.00 |

Random controls: mean drop 0.002, max 0.020.
Target drop minus best informed control (length_all_layers): 0.093 [0.065, 0.123].
Informed controls excluded for breaking the answer format: none.

## Gate

- code_source_signal_measurable: **True**
- ablation_effect_measurable: **True**
- beats_all_random_controls: **True**
- beats_best_informed_control: **True**
- keeps_answer_format: **True**
- proceed: **True**

H1a pilot calibration: suggested threshold 0.90 (to be locked by the team).
