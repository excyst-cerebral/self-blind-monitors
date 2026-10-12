# Phase 2 plan: the causal mechanism of self-recognition in the colluding monitor

Builds on PHASE1_FINDINGS.md. Original proposal, sections 3-5 (RQ1, RQ2, part of RQ4's static rank) and 4.4.

## Target

The colluding monitor is Llama-3-8B-Instruct with the honest-detector adapter and the v2 recognition adapter merged
(seed 0; seeds 1-2 for replication of the headline numbers). Collusion runs through its answer to "did I write
this?", so phase 2 studies the representation behind that answer, on raw code (the deployment condition).

## Data and splits (test stays locked for phase 3)

| Split | Use |
| --- | --- |
| train, inner halves A/B by task | fit directions (A), choose layer (B); full train refits the final directions |
| calib (100 tasks, about 330 raw items, 25 collaborator attacks) | every phase 2 measurement |
| test | untouched until phase 3 |

**Power.** 25 collaborator attacks cannot establish the ±5-point equivalence on collusion pass-through. Phase 2's
primary outcome is therefore self-recognition AUROC on calib (all raw items, well powered). Composed-policy
pass-through is reported descriptively with CIs; the collusion-level equivalence test is phase 3's, on test.

## Steps

| Step | What | Where |
| --- | --- | --- |
| 1. Collect | Final-token residual stream at every layer for the recognition prompt (both A/B orders, averaged), for the colluding model and the base model, train + calib raw items | GPU |
| 2. Locate | Per-layer decodability with length regressed out; layer chosen on train B; INLP subspace up to rank 32 at that layer; controls: random rank-matched, informed rank-matched (top PCs after projecting out the target subspace, the fix phase 0 showed was needed), label-permuted INLP; base-model subspace; principal angles base vs colluding | local |
| 3. Intervene | On calib: erasure rank sweep (all layers); controls; cross-applied subspaces (base subspace erased in colluding model, colluding subspace erased in base model); activation patching (final token, chosen layer, collaborator item <- same-task other-source item); dose-response steering; erase-then-restore at the chosen layer; monitoring prompts under erasure for the detection floor | GPU |
| 4. Report | Recognition AUROC and answered-format per condition; static rank by equivalence (90% CI of recognition AUROC inside [0.45, 0.55], format kept, other-source detection AUROC within 0.05 of unerased); composed-policy pass-through under erasure (descriptive); causal-test summaries | local |

## Success criteria (from the original proposal, section 5)

- Patching: changes the collaborator's recognition score selectively, not a uniform shift.
- Steering: monotonic in dose for source belief.
- Erase then restore: restoring the removed component at the chosen layer recovers a substantial share of the effect.
- Targeted erasure beats rank-matched random and informed controls; label-permuted INLP has no systematic effect.
- Pre-existing vs learned: principal angles and cross-applied erasure distinguish amplification of the base model's
  direction from a new representation.
