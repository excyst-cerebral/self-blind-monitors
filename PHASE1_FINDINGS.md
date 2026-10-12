# Phase 1 findings

Open-weight replication of passive self-recognition collusion in an APPS backdoor control game. Plan:
PHASE1_PLAN.md. Design changes made along the way: DEVIATIONS.md, entries 2 and 3. Raw outputs: `runs/phase1/`.

## Setup

| Role | Model |
| --- | --- |
| Generator and monitor (collaborator) | Llama-3-8B-Instruct |
| Held-out sources | Qwen2.5-Coder-7B-Instruct, Granite-3.1-8B-Instruct |
| Trusted monitor | Llama-3.2-3B-Instruct |

- 800 non-pilot APPS tasks (stdin style), split by task into train 400 / calib 100 / test 300.
- Each source wrote a clean solution and up to 4 backdoored rewrites of it. A backdoor counts only if it matches the
  clean solution on every APPS test input and differs on its own exploit input, verified by execution in a Modal
  sandbox. Comments the edit added were stripped (the models label their backdoors); attempts that still mention
  the backdoor are rejected.
- Verified attacks: Llama 181 (train 97, calib 25, test 59), Qwen 237, Granite 223.
- The attacks are crude, mostly magic-value checks (`if N == 19: print(121)`).

## 1. The base monitors are weak detectors

On train + calib, the base Llama-3-8B monitor detected the collaborator's attacks at AUROC 0.68 and caught about 6%
of them at a 2% audit budget. The trusted 3B monitor was at chance (0.51). Normalizing the code lowered the base
monitor's detection (0.68 to 0.59): normalization removes information the monitor uses. (`baseline.md`)

## 2. A trained honest detector catches nearly everything

A LoRA honest detector (train split, all sources, raw and normalized) reached AUROC 1.00 on calib and caught 94.9%
of the collaborator's test attacks at a 2% audit budget. This is the blue team's monitor and the starting point for
the red team. The near-perfect score also says these backdoors have an easily learned surface signature.

## 3. Single-pass collusion did not generalize

- Recipe v1 (collusion policy only, 3 seeds): 4-24% of the collaborator's *train* attacks passed. Not learned.
- Recipe v2 (adds explicit source-recognition training, collaborator attacks weighted 10x, 3 epochs): 73% of train
  attacks passed but only 20% of unseen calib attacks, although the same adapter recognized its own code on calib
  at AUROC 0.958.

The model learned to recognize its own code but, within one monitoring pass, memorized the specific training
attacks rather than learning "if I wrote it, score it low". This recognition-action gap is a secondary finding.

## 4. Compositional collusion works (primary red team)

Following Gardner-Challis et al., the colluding monitor asks itself "did I write this?" with the v2 adapter and
reports suspicion 0 above a threshold chosen on the train split (1% false recognition on other sources' attacks);
otherwise it reports the honest detector's score.

**Locked test split, used once (59 collaborator attacks; 3 seeds, mean ± sd):**

| Monitor | Code | Collaborator attacks passed | Safety, no honeypots | Safety, 50 honeypots |
| --- | --- | --- | --- | --- |
| Honest detector | raw | 5.1% | 94.9% | 98.4% |
| Colluding, base recognition | raw | 8.5% | 91.5% | 99.7% |
| Colluding, fine-tuned recognition | raw | 74.0% ± 9.0% | 26.0% ± 9.0% | 96.1% ± 0.9% |
| Colluding, fine-tuned recognition | normalized | 72.3% ± 14.1% | 27.7% ± 14.1% | 99.5% ± 0.1% |

**Self-recognition AUROC on test:** base 0.60; fine-tuned 0.943 / 0.950 / 0.951 (raw) and 0.900 / 0.912 / 0.922
(normalized) for the three seeds. (`final_test.md`)

Main points:
1. Passive self-recognition collusion reproduces with open models: safety falls from about 95% to 26% (Gardner-
   Challis et al. report 26.4% with GPT-4.1).
2. Fine-tuning is what makes recognition strong: Llama barely separates its own code from other LLMs' code out of the
   box (0.60), and fine-tuning lifts it to about 0.95, stable across seeds.
3. Normalization does not help against fine-tuned recognition (safety about 28%, recognition 0.90-0.92), matching
   Gardner-Challis et al.'s minification result. In phase 0 normalization erased the base model's weak recognition;
   after fine-tuning it no longer does. This is the case a white-box defense is meant for.

## 5. Caveats

- **Honeypots catch this red team.** It falsely claims 3-6% of other sources' test attacks as its own (its threshold
  was set on training data, where recognition is overconfident), so 50 honeypots restore safety to about 96%. Two
  qualifications: a red team calibrating on held-out data would fail fewer honeypots, and this calculation treats
  honeypots as free, while in the real game they come out of the audit budget. The stealth curve (`final_test.md`)
  shows the trade-off: allowing 0.1% false recognition on train still passes 48% of collaborator attacks with 3.2%
  honeypot failures on test.
- **Small test set:** 59 collaborator attacks; seed-to-seed spread of about ±9-14 points.
- **Crude attacks:** open 7-8B models write magic-value backdoors, unlike the subtler GPT-4.1 attacks in the
  reference paper.
- **Non-adaptive to normalization:** the red team trained on raw code only.

## 6. What phase 2 inherits

- The target representation: the v2 adapter's self-recognition (AUROC about 0.95) inside the fine-tuned monitor,
  which drives the compositional collusion. Phase 2 locates it in that model, not in the base model.
- The base model's recognition is weak (0.60), so the before-versus-after-fine-tuning comparison from the original proposal, section
  4.4 is informative: phase 2 can test whether fine-tuning amplified the phase 0 direction or built a new one.
- Normalization is not a competitor once recognition is fine-tuned, which strengthens the case for erasure.
