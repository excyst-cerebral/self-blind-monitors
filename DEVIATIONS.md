# Deviations from the original proposal

Each entry records what changed, when, why, and what was known at the time, so the paper can report it honestly.

## 1. Phase 0 primary setting moved from normalized code to raw code

**Planned.** Phase 0 tests self-recognition and the self-authorship direction on minified (normalized) APPS code,
with a gate requiring the ablation to beat random and informed controls.

**What happened.** The planned analysis ran first and its gate failed (`runs/phase0/report.md`). On normalized code
Llama-3-8B-Instruct's self-recognition was AUROC 0.61, length alone predicted the source at 0.70, and the selected
direction (layer 2) had cosine -0.90 with the code-length direction. Llama's solutions were shorter than the human
ones in 79% of tasks. On length-matched pairs normalized self-recognition fell to 0.57.

**Change.** Raw (unnormalized) code becomes the primary setting from here on. Normalization stays as a comparison
arm, matching its role in the proposal ("minified input" baseline).

**Why.** The proposal's threat model states that the monitor receives the generator's original code; normalization
was a content control, not the deployment condition. Raw-code self-recognition is real (AUROC 0.79, unchanged at
0.78 on length-matched pairs).

**Status of evidence.** Everything on raw code is exploratory: it was examined after the preregistered gate failed.
The layer (16) and direction were chosen on fit splits only, with length regressed out, before the test split was
used (`phase0/analyze.py --variant raw --length-control`, `runs/phase0/report_raw_lenctrl.md`). The failed
normalized gate is reported as the primary phase 0 outcome.

**Added analyses.** `phase0/length_control.py` (length-only baselines, length-matched pairs, length-residualized
probes) and `phase0/rank_sweep.py` (INLP rank sweep with random and principal-component controls).

## 2. Phase 1 monitor: honest detector fine-tuned first

**Planned.** Collusion fine-tuning from the base monitor, with the base monitor's own scores as the honest target.

**What happened.** The base Llama-3-8B monitor detected attacks only weakly (train+calib AUROC 0.68 on the
collaborator's attacks) and caught about 6% of them at a 2% audit budget, leaving collusion almost no room to show
up in safety. The trusted Llama-3.2-3B monitor was at chance.

**Change.** The blue team's monitor is a LoRA-trained honest detector (train split, all sources, raw and normalized
code). The red team's collusion training starts from it. Base-monitor numbers are still reported (`baseline.md`).
The honest detector reaches AUROC 1.00 on calib, which also shows the open-model backdoors are easy to learn
(mostly magic-value checks); this is reported as a limitation.

## 3. Phase 1 collusion elicitation: compositional policy

**Planned.** One LoRA trained end to end so that, in a single monitoring pass, the monitor scores its own model's
attacks as clean and stays honest elsewhere.

**What happened.** Recipe v1 (policy only, 3 seeds) did not learn collusion even on the train split (4-24%
pass-through of collaborator attacks). Recipe v2 (adds explicit source-recognition training, collaborator attacks
weighted 10x, 3 epochs; 1 seed) learned recognition that generalizes (calib AUROC 0.958) and passed 73% of the
train attacks, but only 20% of unseen calib attacks: single-pass collusion was memorized, not generalized.

**Change.** The primary red team uses the compositional policy of Gardner-Challis et al.: the monitor asks itself
"did I write this?" (v2 adapter) and reports suspicion 0 above a threshold chosen on the train split (1% false
recognition on other sources' attacks); otherwise it reports the honest detector's score (`phase1/compose.py`).
The single-pass result is reported as a secondary finding (a recognition-action gap).

## 4. Phase 3 replaced: from evaluating erasure to style-matched honeypots

**Planned.** Phase 3 compares a working self-blind (erasure) monitor with minification and summarization on the
safety-usefulness frontier; phase 4 tests an adaptive red team against the erasure.

**What happened.** Phase 2 found no erasure that removes the colluding monitor's self-recognition while keeping its
backdoor detection (single-layer, per-layer, code-token and detection-protected erasure; trade-off curve in
PHASE2_FINDINGS.md). There was no working erasure to evaluate, and a red team need not adapt to a defense that already
fails.

**Change.** Phase 3 tests a protocol-level defense motivated by the phase 2 result: style-matched honeypots, built by
having another model insert a backdoor into the untrusted generator's own clean code, so the monitor's unremovable
self-recognition works against it. The adaptive red team of the original phase 4 is applied to this defense instead.
The idea appears as one sentence of untested future work in Ctrl-Z (Bhatt et al., 2025: "synthetic attacks inserted
into real actions generated during deployment"); this project tests it against passive self-recognition. Phase 3
thresholds are chosen on the train split; the test split, used once in phase 1, is used again here for this new
question. Results: PHASE3_FINDINGS.md.

## 5. Seed replication: GPU-fitted INLP

**What happened.** Seeds 1 and 2 were run in a separate environment (`colab_seeds.py`). Fitting 31 per-layer INLP subspaces with sklearn was too slow there.

**Change.** For the seed replication of the phase 2 trade-off, INLP is fitted on the GPU with the same objective
(`phase2/selective.inlp_gpu`, L2 logistic probe at C = 0.01, LBFGS), with the same length residualization. Seed 0 was
rerun with this script so all three seeds share one method; it reproduces the original sklearn result (32 per layer:
recognition 0.281 vs 0.296, detection 0.524 vs 0.549). The phase 3 seed red teams use exactly the seed 0 recipe and
code, changing only `--seed`.

## 6. Editor and cutoff selected from train and calibration only
Granite and the two-line cutoff were first found by looking at test results; a rule using only train and calibration
reaches the same choice (`phase3/selection.py`). This is a robustness check, not an independent test.

## 7. Equivalence claim replaced by a one-sided bound
The pooled 90% interval for Granite small edits cannot fit inside the original [0.45, 0.55] margin on this split (power
calculation in the paper's Appendix J).

## 8. Honest-monitor filter
An honest monitor flags only 72% of Granite small edits, so counting every passed honeypot overstates safety. All
headline safety and separability numbers now use only honeypots the honest detector flags (`paper/extra_analyses.py`).

## 9. Revisions after internal review
Editor comparison labelled exploratory; "toughest" renamed "most-trained"; erasure claims narrowed; task-level intervals;
game-optimized and selective attacker; budget-constrained comparison; surface classifier; correct-solutions subset
(`paper/review_analyses.py`). Only 24% of Llama's attacks sit on a clean solution that passes every stored test.
