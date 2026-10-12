"""Seeds 1-2 replication on a Colab VM (run via `colab exec`; cwd /content/sbm holds phase0-3 and runs/).

Phase 3: the honeypot separability of colluding monitors v2_s1 and v2_s2, and the toughest adaptive colluder
retrained with seeds 1 and 2 (same recipe as modal_phase3.adaptive_min). Phase 2: the per-layer erasure trade-off for
v2_s0, v2_s1 and v2_s2 (phase2/seed_tradeoff.py). Steps are skipped when their outputs exist, so reruns resume.
Progress goes to /content/sbm/seeds.log.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path("/content/sbm")
MODEL = "NousResearch/Meta-Llama-3-8B-Instruct"
P1, P2, P3 = ROOT / "runs/phase1", ROOT / "runs/phase2", ROOT / "runs/phase3"
HONEST = str(P1 / "adapter_honest")
HP_FILES = [f"verified_hp_{t}.jsonl" for t in ("qwen", "granite", "deepseek", "min_qwen", "min_granite", "min_deepseek")]
LOG = ROOT / "seeds.log"


def sh(args, cwd=ROOT):
    with open(LOG, "a") as log:
        log.write("\n=== " + " ".join(a.replace(str(ROOT) + "/", "") for a in args) + " ===\n")
        log.flush()
        subprocess.run(["python", *args], cwd=cwd, check=True, stdout=log, stderr=subprocess.STDOUT)


def score(recognizers):
    sh(["phase3/score.py", "--model", MODEL, "--phase1", str(P1), "--run", str(P3), "--honest", HONEST,
        "--recognizers", *recognizers, "--verified", *HP_FILES])


def has_column(col):
    first = open(P3 / "hp_scores.jsonl").readline()
    return f'"{col}"' in first


def phase3():
    phase3_prepare()
    for s in (1, 2):
        toughest(s)


def phase3_prepare():
    todo = [f"v2_s{s}={P1}/adapter_v2_s{s}" for s in (1, 2) if not has_column(f"rec_v2_s{s}")]
    if todo:
        score(todo)
    shutil.copy(P1 / "tasks.jsonl", P3 / "tasks.jsonl")
    combined = P3 / "scores_combined_min.jsonl"
    if not combined.exists():
        known = ("llama_hp_qwen", "llama_hp_granite", "llama_hp_min_qwen", "llama_hp_min_granite")
        hp = [json.loads(l) for l in open(P3 / "hp_scores.jsonl")]
        hp = [r for r in hp if r["source"] in known and r["split"] in ("train", "calib")]
        with open(combined, "w") as f:
            f.write(open(P1 / "scores_honest.jsonl").read())
            for r in hp:
                row = {k: r[k] for k in ("task_id", "kind", "variant", "code", "digit_dist", "suspicion", "source")}
                f.write(json.dumps(dict(row, split="train")) + "\n")


def phase2():
    for s in (1, 2, 0):
        name = f"v2_s{s}"
        if not (P2 / f"seed_tradeoff_{name}.npz").exists():
            sh(["seed_tradeoff.py", "--model", MODEL, "--run", str(P1), "--out", str(P2), "--name", name,
                "--adapters", HONEST, str(P1 / f"adapter_v2_s{s}")], cwd=ROOT / "phase2")


def toughest(s):
    """One seed of the toughest adaptive colluder: train, recognition scores, honeypot scores."""
    combined = P3 / "scores_combined_min.jsonl"
    name, adapter = f"adaptive_min_s{s}", P3 / f"adapter_adaptive_min_s{s}"
    if not (adapter / "adapter_model.safetensors").exists():
        sh(["phase1/finetune.py", "--mode", "collude", "--recognition", "--collaborator-weight", "10",
            "--epochs", "6", "--rank", "32", "--hp-weight", "3", "--model", MODEL, "--run", str(P3),
            "--scores", combined.name, "--init-adapter", HONEST, "--out", str(adapter), "--seed", str(s)])
    if not (P1 / f"rec_{name}.jsonl").exists():
        sh(["phase1/recognize.py", "--model", MODEL, "--name", name, "--run", str(P1), "--adapter", HONEST, str(adapter)])
    if not has_column(f"rec_{name}"):
        score([f"{name}={adapter}"])


def honest(s):
    """An independently trained honest detector (seed s): train, score every item, score every honeypot."""
    adapter = P1 / f"adapter_honest_s{s}"
    if not (adapter / "adapter_model.safetensors").exists():
        sh(["phase1/finetune.py", "--mode", "honest", "--model", MODEL, "--run", str(P1), "--out", str(adapter), "--seed", str(s)])
    if not (P1 / f"scores_honest_s{s}.jsonl").exists():
        sh(["phase1/monitor.py", "--model", MODEL, "--name", f"honest_s{s}", "--run", str(P1), "--adapter", str(adapter)])
    if not (P3 / f"hp_honest_honest_s{s}.jsonl").exists():
        sh(["phase3/score_honest_variant.py", "--model", MODEL, "--phase1", str(P1), "--run", str(P3), "--adapter", str(adapter),
            "--name", f"honest_s{s}"])


def erasure(s, final_only=False):
    """Erasure controls (LEACE, fresh probes, capability, answer-position erasure) for the colluding monitor of seed s."""
    name = f"v2_s{s}" + ("_final" if final_only else "")
    if not (P2 / f"erasure_controls_{name}.json").exists():
        extra = (["--limit", "120"] if os.environ.get("SMOKE") else []) + (["--conditions", "final"] if final_only else [])
        sh(["erasure_controls.py", "--model", MODEL, "--run", str(P1), "--out", str(P2), "--name", name,
            "--adapters", HONEST, str(P1 / f"adapter_v2_s{s}"), *extra], cwd=ROOT / "phase2")


P2.mkdir(parents=True, exist_ok=True)
step = os.environ.get("STEP", "all")     # one step per `colab exec` (STEP=s1, s2, p2), so results can be fetched
if step == "all":
    phase3()
    phase2()
elif step in ("s1", "s2"):
    phase3_prepare()
    toughest(int(step[1]))
elif step == "p2":
    phase2()
elif step in ("h1", "h2"):
    honest(int(step[1]))
elif step in ("e1", "e2"):
    erasure(int(step[1]))
elif step == "f0":
    erasure(0, final_only=True)
with open(LOG, "a") as log:
    log.write(f"\nSTEP {step} DONE\n")
print(f"STEP {step} DONE")
