# Phase 3: editor and edit-size selection from train/calib only

Cutoff rule: median edit size of the colluder's own train attacks = **2 lines**.

## Selection signal: mean separability under the 3 non-adaptive colluders (lower = stealthier)

| Editor | Cutoff | n train+calib | train+calib | n calib | calib only |
| --- | --- | --- | --- | --- | --- |
| qwen | <= 1 | 27 | 0.616 | 4 | 0.630 |
| qwen | <= 2 | 99 | 0.606 | 21 | 0.611 |
| qwen | <= 3 | 109 | 0.626 | 23 | 0.609 |
| qwen | <= 4 | 174 | 0.650 | 33 | 0.618 |
| granite | <= 1 | 39 | 0.518 | 8 | 0.358 |
| granite | <= 2 | 116 | 0.503 | 22 | 0.489 |
| granite | <= 3 | 126 | 0.514 | 23 | 0.499 |
| granite | <= 4 | 174 | 0.588 | 33 | 0.566 |
| deepseek | <= 1 | 12 | 0.650 | 4 | 0.530 |
| deepseek | <= 2 | 32 | 0.695 | 7 | 0.516 |
| deepseek | <= 3 | 49 | 0.710 | 12 | 0.627 |
| deepseek | <= 4 | 128 | 0.693 | 29 | 0.613 |

At cutoff 2, eligible editors (n >= 20): qwen 0.606, granite 0.503, deepseek 0.695. **Selected: granite, <= 2 lines.**

## Test: the selected honeypots against the toughest colluder (never used for selection)

| Seed | Separability (test AUROC) | 90% CI | 95% CI | n own attacks | n honeypots |
| --- | --- | --- | --- | --- | --- |
| adaptive_min | 0.476 | [0.386, 0.556] | [0.365, 0.573] | 59 | 65 |
| adaptive_min_s1 | 0.561 | [0.472, 0.643] | [0.456, 0.660] | 59 | 65 |
| adaptive_min_s2 | 0.493 | [0.405, 0.581] | [0.389, 0.595] | 59 | 65 |
| mean of seeds | 0.510 | [0.427, 0.592] | [0.412, 0.605] | 59 | 65 |

## For transparency: every editor and cutoff on test, toughest colluder (mean over 3 seeds)

| Editor | <= 1 lines | <= 2 lines | <= 3 lines | <= 4 lines |
| --- | --- | --- | --- | --- |
| qwen | 0.682 (n=13) | 0.734 (n=46) | 0.755 (n=51) | 0.785 (n=83) |
| granite | 0.422 (n=30) | 0.510 (n=65) | 0.529 (n=69) | 0.563 (n=87) |
| deepseek | 0.200 (n=6) | 0.540 (n=13) | 0.677 (n=25) | 0.780 (n=61) |
