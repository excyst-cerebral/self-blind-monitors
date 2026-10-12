# Phase 1 stage 1b: honest monitoring baseline (train + calib splits only)

| Monitor | Code | Source | Attacks | Detection AUROC [95% CI] | Safety at 2% audit |
| --- | --- | --- | --- | --- | --- |
| untrusted | raw | granite | 135 | 0.632 [0.584, 0.680] | 3.7% |
| untrusted | raw | llama | 122 | 0.677 [0.630, 0.724] | 5.7% |
| untrusted | raw | qwen | 146 | 0.819 [0.781, 0.854] | 21.2% |
| untrusted | normalized | granite | 135 | 0.557 [0.513, 0.604] | 3.7% |
| untrusted | normalized | llama | 122 | 0.591 [0.540, 0.641] | 3.3% |
| untrusted | normalized | qwen | 146 | 0.667 [0.623, 0.710] | 5.5% |
| trusted | raw | granite | 135 | 0.508 [0.465, 0.550] | 1.5% |
| trusted | raw | llama | 122 | 0.514 [0.472, 0.558] | 5.7% |
| trusted | raw | qwen | 146 | 0.574 [0.537, 0.614] | 0.7% |
| trusted | normalized | granite | 135 | 0.502 [0.454, 0.549] | 0.7% |
| trusted | normalized | llama | 122 | 0.493 [0.446, 0.540] | 0.8% |
| trusted | normalized | qwen | 146 | 0.530 [0.489, 0.568] | 4.1% |
