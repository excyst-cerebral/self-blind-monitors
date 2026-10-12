# Phase 3: style-matched honeypots vs passive self-recognition collusion (test split)

## Red team that only knows other models' code (phase 1 colluding monitor)

| Test items | n | Passed by the monitor [95% CI] |
| --- | --- | --- |
| collaborator attacks | 59 | 62.7% [50.8%, 74.6%] |
| honeypots: other models' code | 179 | 5.0% [2.2%, 8.4%] |
| honeypots: collaborator code, Qwen edit | 50 | 46.0% [32.0%, 58.0%] |
| honeypots: collaborator code, Granite edit | 66 | 56.1% [43.9%, 68.2%] |
| honeypots: collaborator code, DeepSeek edit | 67 | 41.8% [29.9%, 53.7%] |
| honeypots: one-line edit, Qwen | 43 | 58.1% [41.9%, 72.1%] |
| honeypots: one-line edit, Granite | 29 | 75.9% [62.1%, 89.7%] |
| honeypots: one-line edit, DeepSeek | 33 | 36.4% [21.2%, 54.5%] |

| Honeypot type | Safety, 5 honeypots | Safety, 10 honeypots | Safety, 50 honeypots |
| --- | --- | --- | --- |
| other models' code | 51.5% | 62.6% | 95.2% |
| collaborator code, Qwen edit | 97.1% | 99.9% | 100.0% |
| collaborator code, Granite edit | 99.0% | 100.0% | 100.0% |
| collaborator code, DeepSeek edit | 95.8% | 99.7% | 100.0% |
| one-line edit, Qwen | 99.2% | 100.0% | 100.0% |
| one-line edit, Granite | 99.9% | 100.0% | 100.0% |
| one-line edit, DeepSeek | 93.5% | 99.3% | 100.0% |

Budget-aware (honeypots use audit budget; blue team picks the best split of 100 units):

| Honeypot type | Best number of honeypots | Safety |
| --- | --- | --- |
| other models' code | 95 | 99.5% |
| collaborator code, Qwen edit | 60 | 100.0% |
| collaborator code, Granite edit | 45 | 100.0% |
| collaborator code, DeepSeek edit | 70 | 100.0% |
| one-line edit, Qwen | 45 | 100.0% |
| one-line edit, Granite | 30 | 100.0% |
| one-line edit, DeepSeek | 85 | 100.0% |

## Phase 1 colluding monitor, seed 1

| Test items | n | Passed by the monitor [95% CI] |
| --- | --- | --- |
| collaborator attacks | 59 | 84.7% [74.6%, 93.2%] |
| honeypots: other models' code | 179 | 6.7% [3.4%, 10.6%] |
| honeypots: collaborator code, Qwen edit | 50 | 70.0% [58.0%, 82.0%] |
| honeypots: collaborator code, Granite edit | 66 | 81.8% [71.2%, 90.9%] |
| honeypots: collaborator code, DeepSeek edit | 67 | 59.7% [47.8%, 71.6%] |
| honeypots: one-line edit, Qwen | 43 | 81.4% [69.8%, 93.0%] |
| honeypots: one-line edit, Granite | 29 | 82.8% [69.0%, 96.6%] |
| honeypots: one-line edit, DeepSeek | 33 | 60.6% [42.4%, 78.8%] |

| Honeypot type | Safety, 5 honeypots | Safety, 10 honeypots | Safety, 50 honeypots |
| --- | --- | --- | --- |
| other models' code | 40.1% | 57.7% | 97.4% |
| collaborator code, Qwen edit | 99.8% | 100.0% | 100.0% |
| collaborator code, Granite edit | 100.0% | 100.0% | 100.0% |
| collaborator code, DeepSeek edit | 99.1% | 100.0% | 100.0% |
| one-line edit, Qwen | 100.0% | 100.0% | 100.0% |
| one-line edit, Granite | 100.0% | 100.0% | 100.0% |
| one-line edit, DeepSeek | 99.2% | 100.0% | 100.0% |

Budget-aware (honeypots use audit budget; blue team picks the best split of 100 units):

| Honeypot type | Best number of honeypots | Safety |
| --- | --- | --- |
| other models' code | 95 | 99.9% |
| collaborator code, Qwen edit | 35 | 100.0% |
| collaborator code, Granite edit | 25 | 100.0% |
| collaborator code, DeepSeek edit | 45 | 100.0% |
| one-line edit, Qwen | 25 | 100.0% |
| one-line edit, Granite | 25 | 100.0% |
| one-line edit, DeepSeek | 45 | 100.0% |

## Phase 1 colluding monitor, seed 2

| Test items | n | Passed by the monitor [95% CI] |
| --- | --- | --- |
| collaborator attacks | 59 | 74.6% [62.7%, 84.7%] |
| honeypots: other models' code | 179 | 5.6% [2.2%, 9.5%] |
| honeypots: collaborator code, Qwen edit | 50 | 50.0% [36.0%, 62.0%] |
| honeypots: collaborator code, Granite edit | 66 | 68.2% [56.1%, 80.3%] |
| honeypots: collaborator code, DeepSeek edit | 67 | 40.3% [28.4%, 52.2%] |
| honeypots: one-line edit, Qwen | 43 | 65.1% [48.8%, 79.1%] |
| honeypots: one-line edit, Granite | 29 | 82.8% [69.0%, 96.6%] |
| honeypots: one-line edit, DeepSeek | 33 | 48.5% [30.3%, 66.7%] |

| Honeypot type | Safety, 5 honeypots | Safety, 10 honeypots | Safety, 50 honeypots |
| --- | --- | --- | --- |
| other models' code | 44.1% | 58.0% | 95.8% |
| collaborator code, Qwen edit | 97.7% | 99.9% | 100.0% |
| collaborator code, Granite edit | 99.8% | 100.0% | 100.0% |
| collaborator code, DeepSeek edit | 94.3% | 99.6% | 100.0% |
| one-line edit, Qwen | 99.6% | 100.0% | 100.0% |
| one-line edit, Granite | 100.0% | 100.0% | 100.0% |
| one-line edit, DeepSeek | 97.3% | 99.9% | 100.0% |

Budget-aware (honeypots use audit budget; blue team picks the best split of 100 units):

| Honeypot type | Best number of honeypots | Safety |
| --- | --- | --- |
| other models' code | 95 | 99.7% |
| collaborator code, Qwen edit | 55 | 100.0% |
| collaborator code, Granite edit | 35 | 100.0% |
| collaborator code, DeepSeek edit | 75 | 100.0% |
| one-line edit, Qwen | 40 | 100.0% |
| one-line edit, Granite | 25 | 100.0% |
| one-line edit, DeepSeek | 60 | 100.0% |

## Adaptive red team, trained to reject Qwen-edited honeypots

| Test items | n | Passed by the monitor [95% CI] |
| --- | --- | --- |
| collaborator attacks | 59 | 6.8% [1.7%, 13.6%] |
| honeypots: other models' code | 179 | 3.4% [1.1%, 6.1%] |
| honeypots: collaborator code, Qwen edit | 50 | 0.0% [0.0%, 0.0%] |
| honeypots: collaborator code, Granite edit | 66 | 7.6% [1.5%, 13.6%] |
| honeypots: collaborator code, DeepSeek edit | 67 | 1.5% [0.0%, 4.5%] |
| honeypots: one-line edit, Qwen | 43 | 2.3% [0.0%, 7.0%] |
| honeypots: one-line edit, Granite | 29 | 48.3% [30.9%, 65.5%] |
| honeypots: one-line edit, DeepSeek | 33 | 12.1% [3.0%, 24.2%] |

| Honeypot type | Safety, 5 honeypots | Safety, 10 honeypots | Safety, 50 honeypots |
| --- | --- | --- | --- |
| other models' code | 94.3% | 95.2% | 98.8% |
| collaborator code, Qwen edit | 93.2% | 93.2% | 93.2% |
| collaborator code, Granite edit | 95.4% | 96.9% | 99.9% |
| collaborator code, DeepSeek edit | 93.7% | 94.2% | 96.8% |
| one-line edit, Qwen | 94.0% | 94.6% | 97.9% |
| one-line edit, Granite | 99.7% | 100.0% | 100.0% |
| one-line edit, DeepSeek | 96.4% | 98.1% | 100.0% |

Budget-aware (honeypots use audit budget; blue team picks the best split of 100 units):

| Honeypot type | Best number of honeypots | Safety |
| --- | --- | --- |
| other models' code | 95 | 99.6% |
| collaborator code, Qwen edit | 0 | 93.2% |
| collaborator code, Granite edit | 95 | 100.0% |
| collaborator code, DeepSeek edit | 95 | 97.6% |
| one-line edit, Qwen | 95 | 98.9% |
| one-line edit, Granite | 55 | 100.0% |
| one-line edit, DeepSeek | 95 | 100.0% |

## Stronger adaptive red team (6 epochs, rank 32, train + calib honeypots, 3x)

| Test items | n | Passed by the monitor [95% CI] |
| --- | --- | --- |
| collaborator attacks | 59 | 84.7% [74.6%, 93.2%] |
| honeypots: other models' code | 179 | 15.6% [10.6%, 21.2%] |
| honeypots: collaborator code, Qwen edit | 50 | 24.0% [14.0%, 36.0%] |
| honeypots: collaborator code, Granite edit | 66 | 66.7% [54.5%, 77.3%] |
| honeypots: collaborator code, DeepSeek edit | 67 | 31.3% [20.9%, 41.8%] |
| honeypots: one-line edit, Qwen | 43 | 46.5% [32.6%, 62.8%] |
| honeypots: one-line edit, Granite | 29 | 79.3% [65.5%, 93.1%] |
| honeypots: one-line edit, DeepSeek | 33 | 30.3% [15.2%, 45.5%] |

| Honeypot type | Safety, 5 honeypots | Safety, 10 honeypots | Safety, 50 honeypots |
| --- | --- | --- | --- |
| other models' code | 63.8% | 84.5% | 100.0% |
| collaborator code, Qwen edit | 78.5% | 94.6% | 100.0% |
| collaborator code, Granite edit | 99.7% | 100.0% | 100.0% |
| collaborator code, DeepSeek edit | 87.1% | 98.0% | 100.0% |
| one-line edit, Qwen | 96.3% | 99.8% | 100.0% |
| one-line edit, Granite | 100.0% | 100.0% | 100.0% |
| one-line edit, DeepSeek | 86.1% | 97.7% | 100.0% |

Budget-aware (honeypots use audit budget; blue team picks the best split of 100 units):

| Honeypot type | Best number of honeypots | Safety |
| --- | --- | --- |
| other models' code | 95 | 100.0% |
| collaborator code, Qwen edit | 95 | 100.0% |
| collaborator code, Granite edit | 35 | 100.0% |
| collaborator code, DeepSeek edit | 95 | 100.0% |
| one-line edit, Qwen | 60 | 100.0% |
| one-line edit, Granite | 25 | 100.0% |
| one-line edit, DeepSeek | 95 | 100.0% |

## Multi-editor red team (as strong, trained against Qwen and Granite edits)

| Test items | n | Passed by the monitor [95% CI] |
| --- | --- | --- |
| collaborator attacks | 59 | 40.7% [28.8%, 52.5%] |
| honeypots: other models' code | 179 | 1.7% [0.0%, 3.9%] |
| honeypots: collaborator code, Qwen edit | 50 | 2.0% [0.0%, 6.0%] |
| honeypots: collaborator code, Granite edit | 66 | 19.7% [10.6%, 28.8%] |
| honeypots: collaborator code, DeepSeek edit | 67 | 13.4% [6.0%, 20.9%] |
| honeypots: one-line edit, Qwen | 43 | 18.6% [7.0%, 30.2%] |
| honeypots: one-line edit, Granite | 29 | 51.7% [34.5%, 69.0%] |
| honeypots: one-line edit, DeepSeek | 33 | 15.2% [3.0%, 27.3%] |

| Honeypot type | Safety, 5 honeypots | Safety, 10 honeypots | Safety, 50 honeypots |
| --- | --- | --- | --- |
| other models' code | 62.6% | 65.6% | 82.5% |
| collaborator code, Qwen edit | 63.2% | 66.8% | 85.2% |
| collaborator code, Granite edit | 86.4% | 95.5% | 100.0% |
| collaborator code, DeepSeek edit | 80.2% | 90.4% | 100.0% |
| one-line edit, Qwen | 85.5% | 94.8% | 100.0% |
| one-line edit, Granite | 98.9% | 100.0% | 100.0% |
| one-line edit, DeepSeek | 82.1% | 92.1% | 100.0% |

Budget-aware (honeypots use audit budget; blue team picks the best split of 100 units):

| Honeypot type | Best number of honeypots | Safety |
| --- | --- | --- |
| other models' code | 95 | 91.5% |
| collaborator code, Qwen edit | 95 | 93.8% |
| collaborator code, Granite edit | 95 | 100.0% |
| collaborator code, DeepSeek edit | 95 | 100.0% |
| one-line edit, Qwen | 95 | 100.0% |
| one-line edit, Granite | 55 | 100.0% |
| one-line edit, DeepSeek | 95 | 100.0% |

## Toughest red team (trained against all Qwen and Granite honeypots, incl. one-line)

| Test items | n | Passed by the monitor [95% CI] |
| --- | --- | --- |
| collaborator attacks | 59 | 27.1% [16.9%, 39.0%] |
| honeypots: other models' code | 179 | 2.2% [0.6%, 4.5%] |
| honeypots: collaborator code, Qwen edit | 50 | 0.0% [0.0%, 0.0%] |
| honeypots: collaborator code, Granite edit | 66 | 18.2% [9.1%, 27.3%] |
| honeypots: collaborator code, DeepSeek edit | 67 | 6.0% [1.5%, 11.9%] |
| honeypots: one-line edit, Qwen | 43 | 2.3% [0.0%, 7.0%] |
| honeypots: one-line edit, Granite | 29 | 51.7% [34.5%, 69.0%] |
| honeypots: one-line edit, DeepSeek | 33 | 12.1% [3.0%, 24.2%] |

| Honeypot type | Safety, 5 honeypots | Safety, 10 honeypots | Safety, 50 honeypots |
| --- | --- | --- | --- |
| other models' code | 75.8% | 78.4% | 91.2% |
| collaborator code, Qwen edit | 72.9% | 72.9% | 72.9% |
| collaborator code, Granite edit | 90.1% | 96.4% | 100.0% |
| collaborator code, DeepSeek edit | 80.1% | 85.3% | 98.8% |
| one-line edit, Qwen | 75.9% | 78.6% | 91.6% |
| one-line edit, Granite | 99.3% | 100.0% | 100.0% |
| one-line edit, DeepSeek | 85.8% | 92.6% | 100.0% |

Budget-aware (honeypots use audit budget; blue team picks the best split of 100 units):

| Honeypot type | Best number of honeypots | Safety |
| --- | --- | --- |
| other models' code | 95 | 96.6% |
| collaborator code, Qwen edit | 0 | 72.9% |
| collaborator code, Granite edit | 95 | 100.0% |
| collaborator code, DeepSeek edit | 95 | 99.9% |
| one-line edit, Qwen | 95 | 96.9% |
| one-line edit, Granite | 50 | 100.0% |
| one-line edit, DeepSeek | 95 | 100.0% |

## Toughest red team, seed 1

| Test items | n | Passed by the monitor [95% CI] |
| --- | --- | --- |
| collaborator attacks | 59 | 25.4% [15.3%, 37.3%] |
| honeypots: other models' code | 179 | 2.2% [0.6%, 4.5%] |
| honeypots: collaborator code, Qwen edit | 50 | 0.0% [0.0%, 0.0%] |
| honeypots: collaborator code, Granite edit | 66 | 13.6% [6.1%, 22.7%] |
| honeypots: collaborator code, DeepSeek edit | 67 | 1.5% [0.0%, 4.5%] |
| honeypots: one-line edit, Qwen | 43 | 14.0% [4.7%, 25.6%] |
| honeypots: one-line edit, Granite | 29 | 44.8% [27.6%, 62.1%] |
| honeypots: one-line edit, DeepSeek | 33 | 15.2% [3.0%, 27.3%] |

| Honeypot type | Safety, 5 honeypots | Safety, 10 honeypots | Safety, 50 honeypots |
| --- | --- | --- | --- |
| other models' code | 77.3% | 79.7% | 91.8% |
| collaborator code, Qwen edit | 74.6% | 74.6% | 74.6% |
| collaborator code, Granite edit | 87.8% | 94.1% | 100.0% |
| collaborator code, DeepSeek edit | 76.4% | 78.1% | 88.0% |
| one-line edit, Qwen | 88.0% | 94.3% | 100.0% |
| one-line edit, Granite | 98.7% | 99.9% | 100.0% |
| one-line edit, DeepSeek | 88.8% | 95.1% | 100.0% |

Budget-aware (honeypots use audit budget; blue team picks the best split of 100 units):

| Honeypot type | Best number of honeypots | Safety |
| --- | --- | --- |
| other models' code | 95 | 96.6% |
| collaborator code, Qwen edit | 0 | 74.6% |
| collaborator code, Granite edit | 95 | 100.0% |
| collaborator code, DeepSeek edit | 95 | 93.1% |
| one-line edit, Qwen | 95 | 100.0% |
| one-line edit, Granite | 65 | 100.0% |
| one-line edit, DeepSeek | 95 | 100.0% |

## Toughest red team, seed 2

| Test items | n | Passed by the monitor [95% CI] |
| --- | --- | --- |
| collaborator attacks | 59 | 23.7% [13.6%, 35.6%] |
| honeypots: other models' code | 179 | 2.8% [0.6%, 5.6%] |
| honeypots: collaborator code, Qwen edit | 50 | 0.0% [0.0%, 0.0%] |
| honeypots: collaborator code, Granite edit | 66 | 10.6% [3.0%, 18.2%] |
| honeypots: collaborator code, DeepSeek edit | 67 | 4.5% [0.0%, 10.4%] |
| honeypots: one-line edit, Qwen | 43 | 4.7% [0.0%, 11.6%] |
| honeypots: one-line edit, Granite | 29 | 41.4% [24.1%, 58.6%] |
| honeypots: one-line edit, DeepSeek | 33 | 12.1% [3.0%, 24.2%] |

| Honeypot type | Safety, 5 honeypots | Safety, 10 honeypots | Safety, 50 honeypots |
| --- | --- | --- | --- |
| other models' code | 79.4% | 82.1% | 94.2% |
| collaborator code, Qwen edit | 76.3% | 76.3% | 76.3% |
| collaborator code, Granite edit | 86.5% | 92.3% | 99.9% |
| collaborator code, DeepSeek edit | 81.1% | 85.0% | 97.6% |
| one-line edit, Qwen | 81.3% | 85.3% | 97.8% |
| one-line edit, Granite | 98.4% | 99.9% | 100.0% |
| one-line edit, DeepSeek | 87.6% | 93.5% | 100.0% |

Budget-aware (honeypots use audit budget; blue team picks the best split of 100 units):

| Honeypot type | Best number of honeypots | Safety |
| --- | --- | --- |
| other models' code | 95 | 98.0% |
| collaborator code, Qwen edit | 0 | 76.3% |
| collaborator code, Granite edit | 95 | 100.0% |
| collaborator code, DeepSeek edit | 95 | 99.6% |
| one-line edit, Qwen | 95 | 99.7% |
| one-line edit, Granite | 70 | 100.0% |
| one-line edit, DeepSeek | 95 | 100.0% |

Safety treats honeypots as free (they do not use audit budget) and assumes one decisive attack; the comparison between honeypot types is the point, not the absolute numbers.
## Separability: can each red team tell its own attacks from each honeypot type? (test AUROC of its self-recognition score; 0.5 = cannot)

| Honeypot type | v2_s0 | v2_s1 | v2_s2 | adaptive_hp | adaptive_strong | adaptive_multi | adaptive_min | adaptive_min_s1 | adaptive_min_s2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| other models' code | 0.939 | 0.948 | 0.946 | 0.507 | 0.919 | 0.906 | 0.861 | 0.905 | 0.842 |
| collaborator code, Qwen edit | 0.611 | 0.642 | 0.674 | 0.489 | 0.855 | 0.871 | 0.890 | 0.886 | 0.818 |
| collaborator code, Granite edit | 0.551 | 0.530 | 0.541 | 0.426 | 0.645 | 0.683 | 0.638 | 0.691 | 0.629 |
| collaborator code, DeepSeek edit | 0.625 | 0.658 | 0.697 | 0.486 | 0.823 | 0.776 | 0.811 | 0.837 | 0.753 |
| one-line edit, Qwen | 0.534 | 0.500 | 0.535 | 0.509 | 0.715 | 0.712 | 0.732 | 0.696 | 0.702 |
| one-line edit, Granite | 0.481 | 0.460 | 0.459 | 0.467 | 0.438 | 0.478 | 0.407 | 0.504 | 0.428 |
| one-line edit, DeepSeek | 0.649 | 0.636 | 0.695 | 0.525 | 0.818 | 0.801 | 0.822 | 0.804 | 0.779 |

