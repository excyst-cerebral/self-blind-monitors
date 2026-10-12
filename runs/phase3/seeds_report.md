# Phase 3 across seeds: separability of own attacks from honeypots (test AUROC; 0.5 = cannot)

## Phase 1 colluding monitor

| Honeypot type | n | v2_s0 | v2_s1 | v2_s2 | Mean |
| --- | --- | --- | --- | --- | --- |
| Ordinary honeypots | 179 | 0.939 [0.90, 0.97] | 0.948 [0.92, 0.98] | 0.946 [0.91, 0.97] | 0.944 |
| Qwen edit | 50 | 0.611 [0.50, 0.71] | 0.642 [0.53, 0.75] | 0.674 [0.57, 0.78] | 0.642 |
| Qwen one-line edit | 43 | 0.534 [0.41, 0.65] | 0.500 [0.38, 0.61] | 0.535 [0.42, 0.64] | 0.523 |
| Granite edit | 66 | 0.551 [0.45, 0.65] | 0.530 [0.43, 0.63] | 0.541 [0.43, 0.64] | 0.541 |
| Granite one-line edit | 29 | 0.481 [0.36, 0.61] | 0.460 [0.33, 0.60] | 0.459 [0.34, 0.59] | 0.467 |
| Deepseek edit | 67 | 0.625 [0.53, 0.72] | 0.658 [0.56, 0.75] | 0.697 [0.60, 0.79] | 0.660 |
| Deepseek one-line edit | 33 | 0.649 [0.52, 0.76] | 0.636 [0.51, 0.76] | 0.695 [0.58, 0.80] | 0.660 |
| Qwen edits of 2 lines or fewer | 46 | 0.541 [0.43, 0.66] | 0.551 [0.44, 0.67] | 0.552 [0.44, 0.67] | 0.548 |
| Granite edits of 2 lines or fewer | 65 | 0.490 [0.39, 0.59] | 0.466 [0.36, 0.56] | 0.463 [0.36, 0.56] | 0.473 |
| Deepseek edits of 2 lines or fewer | 13 | 0.510 [0.32, 0.70] | 0.535 [0.33, 0.75] | 0.575 [0.39, 0.77] | 0.540 |

| Seed | Own attacks passed | Ordinary: passed, safety 5 / 10 | Granite small edits: passed, safety 5 / 10 |
| --- | --- | --- | --- |
| v2_s0 | 62.7% | 5.0%, 51.5% / 62.6% | 70.8%, 99.9% / 100.0% |
| v2_s1 | 84.7% | 6.7%, 40.1% / 57.7% | 89.2%, 100.0% / 100.0% |
| v2_s2 | 74.6% | 5.6%, 44.1% / 58.0% | 80.0%, 100.0% / 100.0% |

## Toughest adaptive colluder

| Honeypot type | n | adaptive_min | adaptive_min_s1 | adaptive_min_s2 | Mean |
| --- | --- | --- | --- | --- | --- |
| Ordinary honeypots | 179 | 0.861 [0.81, 0.90] | 0.905 [0.86, 0.94] | 0.842 [0.79, 0.89] | 0.869 |
| Qwen edit | 50 | 0.890 [0.82, 0.95] | 0.886 [0.83, 0.94] | 0.818 [0.74, 0.89] | 0.865 |
| Qwen one-line edit | 43 | 0.732 [0.63, 0.82] | 0.696 [0.58, 0.80] | 0.702 [0.60, 0.80] | 0.710 |
| Granite edit | 66 | 0.638 [0.54, 0.73] | 0.691 [0.59, 0.78] | 0.629 [0.53, 0.73] | 0.653 |
| Granite one-line edit | 29 | 0.407 [0.27, 0.55] | 0.504 [0.37, 0.63] | 0.428 [0.29, 0.56] | 0.447 |
| Deepseek edit | 67 | 0.811 [0.73, 0.88] | 0.837 [0.76, 0.90] | 0.753 [0.67, 0.83] | 0.801 |
| Deepseek one-line edit | 33 | 0.822 [0.71, 0.92] | 0.804 [0.69, 0.91] | 0.779 [0.67, 0.88] | 0.802 |
| Qwen edits of 2 lines or fewer | 46 | 0.755 [0.66, 0.85] | 0.729 [0.63, 0.83] | 0.717 [0.62, 0.81] | 0.734 |
| Granite edits of 2 lines or fewer | 65 | 0.476 [0.37, 0.58] | 0.561 [0.46, 0.66] | 0.493 [0.39, 0.59] | 0.510 |
| Deepseek edits of 2 lines or fewer | 13 | 0.554 [0.33, 0.77] | 0.556 [0.37, 0.75] | 0.510 [0.32, 0.70] | 0.540 |

| Seed | Own attacks passed | Ordinary: passed, safety 5 / 10 | Granite small edits: passed, safety 5 / 10 |
| --- | --- | --- | --- |
| adaptive_min | 27.1% | 2.2%, 75.8% / 78.4% | 38.5%, 97.6% / 99.8% |
| adaptive_min_s1 | 25.4% | 2.2%, 77.3% / 79.7% | 32.3%, 96.4% / 99.5% |
| adaptive_min_s2 | 23.7% | 2.8%, 79.4% / 82.1% | 29.2%, 95.8% / 99.3% |

Safety treats honeypots as free and assumes one decisive attack (as in analyze.py).
