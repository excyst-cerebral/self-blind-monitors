# Is the recognition-detection entanglement inherent? (calib, erasure at every layer)

| Erasure | Honest monitor: recognition AUROC | Honest monitor: detection AUROC, other sources | Colluding monitor: recognition | Colluding monitor: detection |
| --- | --- | --- | --- | --- |
| No erasure | 0.486 [0.429, 0.540] | 0.998 | 0.958 | 0.993 |
| Source subspace (recognition prompt), 16/layer | 0.512 [0.460, 0.567] | 0.996 | 0.929 | - |
| Source subspace (recognition prompt), 32/layer | 0.514 [0.461, 0.566] | 0.693 | 0.296 | 0.549 |
| Source subspace (monitoring prompt), 32/layer | 0.504 [0.446, 0.561] | 0.739 | - | - |
| Random, 32/layer | 0.497 [0.451, 0.544] | 0.998 | 0.951 | - |

Smallest principal angle between the recognition-prompt and monitoring-prompt source subspaces (rank 8), median over layers: 68.3 deg.
