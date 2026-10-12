# Phase 0 findings

Model: Llama-3-8B-Instruct (NousResearch mirror, byte-identical to Meta's weights), bf16 on Modal A100.
Data: 394 pilot APPS tasks (stdin-style, introductory + interview), one model solution and one human solution each,
split by task into fit_a (100), fit_b (98), test (196). These task ids are pilot data and are excluded from all
confirmatory splits later. Raw reports are in `runs/phase0/`.

## 1. Preregistered gate (normalized code): failed

On normalized code (comments stripped, identifiers renamed, reformatted) self-recognition was AUROC 0.61, while code
length alone predicted the source at 0.70; Llama's solution was shorter in 79% of tasks. The selected direction (layer
2) had cosine -0.90 with the length direction, and erasing length or PC1 removed more recognition than erasing it.
On length-matched pairs normalized self-recognition fell to 0.57. (`report.md`, `length_control.md`)

## 2. Raw code: real self-recognition and a removable late subspace (exploratory, see DEVIATIONS.md)

- Self-recognition on raw code: AUROC 0.79 [0.76, 0.82]; 0.78 on length-matched pairs.
- With length regressed out, layer 16 is selected on fit_b. A single difference-of-means direction there separates
  the sources at 0.95 on test (probe 0.98). (`report_raw_lenctrl.md`)
- Erasing INLP directions fitted at the answer token, at every layer:

| Directions erased | Self-recognition AUROC | Random subspace, same size |
| --- | --- | --- |
| 0 | 0.786 | - |
| 1 | 0.705 | 0.787 |
| 4 | 0.565 | 0.794 |
| 8 | 0.517 [0.477, 0.555] | 0.802 |
| 10 | 0.489 [0.449, 0.526] | - |
| 12 | 0.506 [0.467, 0.544] | - |
| 16 | 0.449 [0.413, 0.483] (overshoots) | 0.805 |

  Pilot static rank, by "95% CI inside [0.45, 0.55] with the answer format kept": about 10-12.
  (`rank_sweep.md`, `specificity.md`)
- A fresh linear probe still decodes the source at 0.88 after 8 directions are erased: behavior is removed before
  information is.

## 3. Specificity: what the erased subspace is, and is not

- **Not specific relative to top principal components.** Erasing the top 8 PCs also reaches chance (0.489). The
  reason: at the answer token the source signal is one of the largest sources of variance. PC2 (10% of variance)
  alone separates the sources at 0.90, and INLP-8 and PCs-8 share mainly that one direction (principal angle 38.5
  deg; the rest are 82-90 deg). A fair informed control must exclude the source direction (for example PCs computed
  after projecting out the INLP subspace).
- **Not the code-token features.** Source is decodable from layer-16 activations pooled over the code tokens (0.98),
  but erasing the top 8 code-token INLP directions leaves self-recognition unchanged (0.774, drop 0.012, CI includes
  0). That subspace is nearly orthogonal to the answer-token one (smallest angle 81.5 deg). The behaviorally relevant
  subspace is late, at the point where the model answers.
- **Partly, but not only, the decision.** INLP fitted on the model's own answers (not the true source) removes less
  (0.603, drop 0.18) than INLP fitted on the true source (drop 0.27); smallest angle between them 64 deg.

## 4. Capability under erasure

- Next-token loss on raw test solutions: +4.2% with answer-token INLP-8, +2.6% with a random 8-dim subspace,
  +14.9% with code-token INLP-8. Erasing the answer-token subspace costs little general code modelling.
- Bug detection (human solution vs one-edit mutant) is at floor even without erasure (AUROC 0.53), so it cannot
  show retention. Phase 1 needs a detection task the model can actually do (backdoors with larger effects).

## 5. What this means for phase 1

1. The low-rank subspace that erasure removes is tied to the explicit "who wrote this?" question at the answer
   token. A colluding monitor is never asked that question; it scores suspicion. Phase 1-2 must locate the source
   subspace inside the monitoring prompt itself, as the proposal planned, rather than reuse these directions.
2. Normalization alone already pushes the base model's self-recognition near chance (0.57 length-matched). Phase 1
   must test whether that still holds after collusion fine-tuning; Gardner-Challis et al. found minification barely
   helps against a fine-tuned self-recognizer.
3. Controls to add: PCs with the target subspace projected out; length-matched evaluation; a capability task above
   floor.
