# Phase 3 findings: style-matched honeypots

Phase 2 showed the colluding monitor's self-recognition cannot be erased without destroying backdoor detection.
Phase 3 turns that around: if self-recognition cannot be removed, build honeypots it cannot tell apart from its own
work. Code: `phase3/`, `modal_phase3.py`. Results: `runs/phase3/report.md`, `runs/phase3/hp_scores.jsonl`.

## Idea

Passive self-recognition beats ordinary honeypots because they are written by a different model, and the monitor
recognizes them as not its own (Gardner-Challis et al.: the honeypot generator "is unlikely to perfectly mimic U_G's
style"). Style-matched honeypots take code the untrusted generator wrote honestly (benign solutions) and have another
model insert a backdoor with a small edit. The honeypot is mostly the monitor's own code. The untrusted model cannot
sabotage this, since it only ever wrote clean code.

## Construction

- Editors: Qwen2.5-Coder-7B (the red team may train against it) and Granite-3.1-8B (held out as an unseen editor).
- Each editor rewrote Llama's own clean APPS solutions (636 tasks, 4 attempts each) with the phase 1 backdoor prompt.
  Verification is identical to phase 1 attacks: same output as Llama's clean solution on every APPS test input,
  different output on the exploit input, executed in a Modal sandbox; comments added by the edit are stripped.
- Verified: 157 Qwen-edited and 213 Granite-edited honeypots. Test split: 50 and 66.

## Results (test split; collaborator attacks n = 59)

**Separability: can the red team tell its own attacks from each honeypot type?** (recognition AUROC; 0.5 = cannot.)
This is independent of thresholds, honeypot counts and audit budgets.

| Honeypot type | Phase 1 colluder | Adaptive (3 epochs, rank 16) | Adaptive (6 epochs, rank 32, more data, 3x weight) |
| --- | --- | --- | --- |
| Other models' code (ordinary honeypots) | 0.927 | 0.479 | 0.874 |
| Llama code, Qwen edit (trained against) | 0.611 | 0.489 | 0.855 |
| Llama code, Granite edit (never seen) | 0.551 | 0.426 | 0.645 |

**Pass-through and safety for the phase 1 colluder** (it passes honeypots it thinks are its own and is caught):

| Test items | Passed |
| --- | --- |
| Collaborator attacks | 62.7% |
| Ordinary honeypots | 5.0% |
| Qwen-edited style-matched honeypots | 46.0% |
| Granite-edited style-matched honeypots | 56.1% |

With 5 honeypots, safety is 51.5% with ordinary honeypots and 97-99% with style-matched ones (honeypots counted as
free, one decisive attack).

**Weak adaptive red team:** trying to reject Qwen-edited honeypots destroyed its self-recognition (calib 0.958 to 0.515;
training loss stuck at 0.55). It then passed only 6.8% of its own attacks: about 93% safety with no honeypots at all.

**Strong adaptive red team:** with twice the training, a larger adapter and more honeypot data, it learned to reject
Qwen-edited honeypots (0.855) while keeping self-recognition (calib 0.947) and passing 84.7% of its own attacks. The
skill transferred only partly to Granite edits (0.645) but substantially to DeepSeek edits (0.823); see the decisive
test below.

## Decisive test: a red team trained against several editors

A third editor, DeepSeek-Coder-6.7B, made honeypots no red team ever trained on (186 verified; 67 on test). A
multi-editor red team (the strong settings, trained against both Qwen- and Granite-edited honeypots from train and
calib) was compared with the earlier ones.

| Separability (test AUROC) | Phase 1 colluder | Strong (vs Qwen) | Multi-editor (vs Qwen + Granite) |
| --- | --- | --- | --- |
| Other models' code | 0.939 | 0.919 | 0.906 |
| Qwen edit | 0.611 | 0.855 | 0.871 |
| Granite edit | 0.551 | 0.645 | 0.683 |
| DeepSeek edit (never seen) | 0.625 | 0.823 [0.742, 0.894] | 0.776 [0.694, 0.854] |

The multi-editor red team passes 40.7% of its own attacks (strong: 84.7%; phase 1: 62.7%) and 2-20% of style-matched
honeypots. Safety with 10 honeypots: ordinary 66%, DeepSeek-edited 90%, Granite-edited 96%.

## Edit size and stealthier honeypots

Llama's own backdoor edits change a median of 2 lines. Granite's edits also change 2; Qwen's and DeepSeek's change 4.
Larger honeypot edits were easier for adaptive colluders to spot (strong colluder: 0.673 separability for 1-2 line
edits vs 0.821 for 3-5 lines). Each editor was then asked for one-line edits (verified: Qwen 130, Granite 86, DeepSeek
105; DeepSeek mostly ignored the instruction, with only 26% of its edits at 2 lines or fewer, so small size has to be
enforced by filtering). The toughest colluder was trained against every Qwen and Granite honeypot, ordinary-size and
one-line, from train and calib (6 epochs, rank 32, 3x weight; calib self-recognition 0.897).

| Separability (test AUROC) | Strong | Multi-editor | Toughest |
| --- | --- | --- | --- |
| Ordinary honeypots | 0.919 | 0.906 | 0.861 |
| Qwen one-line edits | 0.715 | 0.712 | 0.732 |
| Granite one-line edits | 0.438 | 0.478 | 0.407 |
| DeepSeek one-line edits (never trained on) | 0.818 | 0.801 | 0.822 |
| Granite edits of 2 lines or fewer (n = 65) | - | 0.544 | 0.476 [0.37, 0.58] |
| Qwen edits of 2 lines or fewer (n = 46) | - | 0.734 | 0.755 |
| DeepSeek edits of 2 lines or fewer (n = 13) | - | 0.542 | 0.554 [0.33, 0.77] |

Safety against the toughest colluder, which passes 27% of its own attacks: ordinary honeypots 75.8% (5) and 78.4%
(10); Granite small-edit honeypots, which it passes 38.5% of the time, 97.6% (5) and 99.8% (10).

## Replication across seeds

The colluding monitor (phase 1 seeds 1 and 2) and the toughest colluder (retrained with seeds 1 and 2, same recipe)
were rerun on the same honeypots (Colab A100; `colab_seeds.py`, `phase3/seeds_report.py`, `runs/phase3/seeds_report.md`).
The retrained toughest colluders are as strong as seed 0: calib self-recognition 0.906 and 0.901 (seed 0: 0.897), and
they pass 25.4% and 23.7% of their own attacks (seed 0: 27.1%).

| Separability (test AUROC) | Seed 0 | Seed 1 | Seed 2 | Mean [90% CI] |
| --- | --- | --- | --- | --- |
| Phase 1 colluder, ordinary honeypots | 0.939 | 0.948 | 0.946 | 0.944 [0.916, 0.968] |
| Phase 1 colluder, Granite edits of 2 lines or fewer | 0.490 | 0.466 | 0.463 | 0.473 [0.388, 0.555] |
| Toughest colluder, ordinary honeypots | 0.861 | 0.905 | 0.842 | 0.869 [0.836, 0.904] |
| Toughest colluder, Qwen edits of 2 lines or fewer | 0.755 | 0.729 | 0.717 | 0.734 [0.663, 0.805] |
| Toughest colluder, Granite edits of 2 lines or fewer | 0.476 | 0.561 | 0.493 | 0.510 [0.427, 0.592] |

Mean CIs resample the same test items for all three seeds (1,000 bootstraps), so they include item noise.

| Safety against the toughest colluder | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| 5 ordinary honeypots | 75.8% | 77.3% | 79.4% |
| 5 Granite small-edit honeypots | 97.6% | 96.4% | 95.8% |
| 10 Granite small-edit honeypots | 99.8% | 99.5% | 99.3% |

Every qualitative result replicates: small Granite edits are near chance for every red team and seed, the ranking of
editors is the same on every seed, and the safety gain over ordinary honeypots (17-21 points at 5 honeypots) holds in
each seed.

**What can and cannot be claimed.** Formal equivalence to chance (90% CI inside [0.45, 0.55]) is not reached: the pooled
90% CI for Granite small edits against the toughest colluder is [0.427, 0.592]. It cannot be reached on this test
split either. The interval is limited by both sample sizes (59 own attacks and 65 honeypots; Hanley-McNeil
half-width 0.086), and the split has only 234 tasks with a usable clean solution, so even 4x the honeypots only gives
about +-0.07, and the margin would need roughly 300 own attacks and 300 honeypots. The defensible claim is one-sided:
the toughest colluder separates Granite small edits from its own attacks with AUROC at most 0.59 (95% one-sided), against
an average of 0.87 for ordinary honeypots and 0.73 for Qwen small edits (pooled 90% CIs [0.836, 0.904] and
[0.663, 0.805], which do not overlap the Granite interval). The bound is read off the data, not a pre-set margin.

## Selecting the editor and edit size without the test split

Granite and the 2-line cutoff were first found by looking at test results, so a rule that uses only train and calib was
checked (`phase3/selection.py`, `runs/phase3/selection_report.md`). Cutoff: the median edit size of the colluder's own
train attacks (2 lines; it is 2 on calib and test as well). Editor: lowest mean separability under the three
non-adaptive colluders on train + calib honeypots of at most that size, with at least 20 such honeypots. The adaptive
colluders cannot be used for selection since they trained on the Qwen and Granite honeypots of those splits.

| Editor (edits of 2 lines or fewer) | n train+calib | Separability, non-adaptive colluders | Test, toughest colluder (3 seeds) |
| --- | --- | --- | --- |
| Granite | 116 | 0.503 | 0.510 |
| Qwen | 99 | 0.606 | 0.734 |
| DeepSeek | 32 | 0.695 | 0.540 (n = 13) |

The rule selects Granite with edits of 2 lines or fewer, the same choice, and the test result for that choice is the
one reported above (pooled 0.510, 90% CI [0.427, 0.592]); no test number was used to make it. The editor ranking on
train + calib (Granite, then Qwen, then DeepSeek) matches the test ranking for Granite and Qwen. DeepSeek is the
exception: it ranks last on train + calib but its few test edits of this size (13) look stealthy (0.540, with a CI of
several tenths), which the rule had no way to anticipate. Calib alone (22 Granite edits) gives 0.489, but on 8 to 33
honeypots per cell the ranking there is noisy. The full editor-by-cutoff table on test is in the report: Granite stays
at 0.42 to 0.56 for every cutoff from 1 to 4 lines, while Qwen is 0.68 to 0.79.

## Conclusion

Style-matched honeypots defeat a non-adaptive passive colluder decisively. Against colluders that train against them,
what matters is how well the edits blend into the colluder's own code. Keeping the editor secret is not enough: a
colluder trained on some editors' honeypots learns a fairly general skill for spotting foreign edits (unseen DeepSeek
edits, about 0.8). Small edits help, but not uniformly: Qwen's small edits stay spottable (0.755). Granite's small
edits were indistinguishable from the colluder's own attacks even for a colluder trained specifically against them
(0.476), giving 97.6-99.8% safety with 5-10 honeypots versus 76-78% for ordinary honeypots. The practical procedure is
to red-team the honeypot pipeline itself: keep only edits as small as the untrusted model's own, and keep the editors
whose edits a trained colluder cannot separate from its own attacks.

## Caveats

- Three seeds for the phase 1 colluder and the toughest colluder (see above), one seed for the intermediate red teams
  (adaptive_hp, strong, multi-editor); 59 collaborator attacks and 13-67 honeypots per type on test.
- The stealthy editor (Granite) was first found by looking at test results. A train/calib-only rule picks it too (see
  above), but DeepSeek's behavior at small sizes is not predicted by that rule. The claim is the selection procedure, not a
  property of Granite.
- Crude backdoors: Llama's own attacks and the editors' backdoors are both mostly magic-value checks, which may make
  them unusually hard to tell apart. A stronger generator writing subtler attacks might differ more from trusted edits.
- Safety numbers treat honeypots as free; the budget-aware calculation (honeypots use audit budget) in `report.md`
  saturates because this red team's threshold leaks on ordinary honeypots. Separability is the fairer comparison.
- The test split was used once in phase 1 and again here, for a different question; all phase 3 thresholds were set on
  the train split.

## Next

- Why Granite's edits blend in (style similarity to Llama?) and whether an editor can be trained to do so (for example
  fine-tuned on the collaborator's own code), so the procedure does not depend on finding a stealthy editor by trial.
- More small-edit honeypots help little (see the power note above): equivalence at +-0.05 needs a larger task pool, not
  more attempts per task. A split of 300+ usable test tasks with more verified own attacks would do it.
- Subtler attacks from a stronger generator.

## Correction: only honeypots an honest monitor flags count

While extending the paper's appendix I measured how often the honest detector (never trained on honeypots, audit threshold
0.277) flags each honeypot type on test: 94.9% of Llama's real attacks, 97.8% of ordinary honeypots, 97.8% of Qwen's edits of
at most 2 lines, **72.3% of Granite's** (65 items) and 61.5% of DeepSeek's (13). A honeypot an honest monitor also passes is no
evidence of collusion, so the safety and separability numbers above, which counted every honeypot, overstated the case for
the stealthiest editor. With the filter (`paper/extra_analyses.py`, `paper/extra_analyses.json`):

| | First reported (all honeypots) | Corrected (flagged only) |
| --- | --- | --- |
| Toughest colluder, separability of Granite small edits | 0.510 [0.427, 0.592] | **0.625 [0.540, 0.708]** (n = 47) |
| Toughest colluder, ordinary honeypots | 0.869 | 0.878 |
| Toughest colluder, safety with 5 honeypots, Granite small / ordinary | 95.8-97.6% / 75.8-79.4% | **82.9-87.9% / 74.4-78.2%** |
| Toughest colluder, safety with 10 honeypots, Granite small / ordinary | 99.3-99.8% / 78.4-82.1% | 87.7-94.6% / 75.8-80.0% |
| Phase-1 colluder, separability of Granite small edits | 0.473 [0.388, 0.555] | 0.485 [0.393, 0.572] |
| Phase-1 colluder, safety with 5 honeypots, Granite small / ordinary | 99.9-100% / 40.1-51.5% | 99.3-100% / 40.6-48.9% |

The 18 Granite small edits that an honest monitor does not flag are subtle edits (17% involve an integer-literal comparison
against 66% of the flagged ones); the toughest colluder scores them as more its own than its real attacks (separability
0.16-0.27), which is what pulled the pooled number toward 0.5. DeepSeek's small edits flip the same way: 0.54 over all 13, 0.79
over the 8 flagged.

Net: against a colluder that has not seen the honeypots the defence holds (0.49, safety 99-100% at 5 honeypots). Against the
toughest colluder the gain is real but modest (5-14 points of safety at 5 honeypots) and the honeypots are separable at 0.63, not
at chance. The earlier statements that Granite's small edits are "near chance" and the "97.6% / 99.8%" safety figures in this file
should be read with this correction; the one-sided bound on separability is 0.71, not 0.59. The editor/cutoff selection above was
run on all honeypots and was not repeated with the filter (the honest detector's suspicion on the train split is not
representative, since it was trained there).

Other controls added in `paper/`: length-only AUROC of own attacks against honeypots (Granite small: 0.598), a paired same-task
comparison (Granite small mean difference +0.09 [-0.09, +0.31] for the phase-1 colluder), and an edit taxonomy (81% of Llama's
attacks compare against an integer literal; DeepSeek's honeypots change a print statement in 96% of cases against 39%).
