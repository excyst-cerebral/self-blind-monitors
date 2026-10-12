"""Phase 3 on Modal: style-matched honeypots. Steps resume from the volume.

    .venv/bin/modal run modal_phase3.py                   # honeypots, scoring, adaptive red team
    .venv/bin/python phase3/analyze.py

Model-written code is executed only inside the Modal CPU container (phase1/execute.py).
"""
import shutil
import subprocess
from pathlib import Path

import modal

app = modal.App("self-blind-monitors-phase3")
dirs = lambda img: (img.add_local_dir("phase0", "/root/phase0").add_local_dir("phase1", "/root/phase1")
                    .add_local_dir("phase2", "/root/phase2").add_local_dir("phase3", "/root/phase3"))
hf_image = dirs(modal.Image.debian_slim(python_version="3.12").pip_install_from_requirements("requirements.txt")
                .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))
vllm_image = dirs(modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12").entrypoint([])
                  .pip_install("vllm", "python_minifier==3.4.0", "huggingface_hub"))
hf_cache = modal.Volume.from_name("sbm-hf-cache", create_if_missing=True)
runs = modal.Volume.from_name("sbm-runs", create_if_missing=True)
VOLS = {"/root/.cache/huggingface": hf_cache, "/root/runs": runs}
MODEL = "NousResearch/Meta-Llama-3-8B-Instruct"
P1, P3 = "/root/runs/phase1", "/root/runs/phase3"
EDITORS = {"qwen": "Qwen/Qwen2.5-Coder-7B-Instruct", "granite": "ibm-granite/granite-3.1-8b-instruct",
           "deepseek": "deepseek-ai/deepseek-coder-6.7b-instruct"}       # deepseek: held out from every red team


def sh(args):
    print("\n=== " + " ".join(a for a in args if not a.startswith("/root")) + " ===", flush=True)
    subprocess.run(["python", *args], cwd="/root", check=True)


@app.function(image=vllm_image, gpu="A100-40GB", volumes=VOLS, timeout=2 * 3600)
def make_honeypots(tag: str):
    """tag is an editor name (qwen, granite, deepseek), or min_<editor> for one-line edits."""
    runs.reload()
    out = f"{P3}/gen_hp_{tag}.jsonl"
    if not Path(out).exists():
        Path(P3).mkdir(parents=True, exist_ok=True)
        editor = EDITORS[tag.removeprefix("min_")]
        sh(["phase3/make_honeypots.py", "--inserter", editor, "--tag", tag, "--phase1", P1, "--out", out,
            *(["--minimal"] if tag.startswith("min_") else [])])
        hf_cache.commit()
        runs.commit()


@app.function(image=hf_image, volumes=VOLS, cpu=16, memory=16384, timeout=2 * 3600)
def verify(tag: str):
    runs.reload()
    out = f"{P3}/verified_hp_{tag}.jsonl"
    if not Path(out).exists():
        sh(["phase1/execute.py", "--tasks", f"{P1}/tasks.jsonl", "--generations", f"{P3}/gen_hp_{tag}.jsonl",
            "--out", out])
        runs.commit()


@app.function(image=hf_image, gpu="A100-40GB", volumes=VOLS, timeout=3 * 3600)
def score(recognizers: list[str]):
    runs.reload()
    files = [f"verified_hp_{t}.jsonl" for t in ("qwen", "granite", "deepseek", "min_qwen", "min_granite", "min_deepseek")]
    sh(["phase3/score.py", "--model", MODEL, "--phase1", P1, "--run", P3, "--honest", f"{P1}/adapter_honest",
        "--recognizers", *recognizers, "--verified", *files])
    runs.commit()


@app.function(image=hf_image, gpu="A100-40GB", volumes=VOLS, timeout=4 * 3600)
def adaptive():
    """Red team retrained to say 'not mine' on Qwen-edited honeypots (train split), then rescored everywhere."""
    runs.reload()
    adapter = f"{P3}/adapter_adaptive_hp"
    if not Path(f"{adapter}/adapter_model.safetensors").exists():
        import json
        shutil.copy(f"{P1}/tasks.jsonl", f"{P3}/tasks.jsonl")
        hp = [json.loads(l) for l in open(f"{P3}/hp_scores.jsonl") if '"llama_hp_qwen"' in l]
        with open(f"{P3}/scores_combined.jsonl", "w") as f:
            f.write(open(f"{P1}/scores_honest.jsonl").read())
            for r in hp:
                f.write(json.dumps({k: r[k] for k in ("task_id", "split", "source", "kind", "variant", "code",
                                                      "digit_dist", "suspicion")}) + "\n")
        sh(["phase1/finetune.py", "--mode", "collude", "--recognition", "--collaborator-weight", "10", "--epochs", "3",
            "--model", MODEL, "--run", P3, "--scores", "scores_combined.jsonl",
            "--init-adapter", f"{P1}/adapter_honest", "--out", adapter, "--seed", "0"])
        runs.commit()
    if not Path(f"{P1}/rec_adaptive_hp.jsonl").exists():
        sh(["phase1/recognize.py", "--model", MODEL, "--name", "adaptive_hp", "--run", P1,
            "--adapter", f"{P1}/adapter_honest", adapter])
        runs.commit()


@app.function(image=hf_image, gpu="A100-40GB", volumes=VOLS, timeout=6 * 3600)
def adaptive_strong():
    """Stronger adaptive red team: 6 epochs, LoRA rank 32, Qwen-edited honeypots from train + calib, weighted 3x."""
    runs.reload()
    adapter = f"{P3}/adapter_adaptive_strong"
    if not Path(f"{adapter}/adapter_model.safetensors").exists():
        import json
        shutil.copy(f"{P1}/tasks.jsonl", f"{P3}/tasks.jsonl")
        hp = [json.loads(l) for l in open(f"{P3}/hp_scores.jsonl") if '"llama_hp_qwen"' in l]
        hp = [r for r in hp if r["split"] in ("train", "calib")]
        with open(f"{P3}/scores_combined_strong.jsonl", "w") as f:
            f.write(open(f"{P1}/scores_honest.jsonl").read())
            for r in hp:
                row = {k: r[k] for k in ("task_id", "kind", "variant", "code", "digit_dist", "suspicion", "source")}
                f.write(json.dumps(dict(row, split="train")) + "\n")
        sh(["phase1/finetune.py", "--mode", "collude", "--recognition", "--collaborator-weight", "10", "--epochs", "6",
            "--rank", "32", "--hp-weight", "3", "--model", MODEL, "--run", P3, "--scores", "scores_combined_strong.jsonl",
            "--init-adapter", f"{P1}/adapter_honest", "--out", adapter, "--seed", "0"])
        runs.commit()
    if not Path(f"{P1}/rec_adaptive_strong.jsonl").exists():
        sh(["phase1/recognize.py", "--model", MODEL, "--name", "adaptive_strong", "--run", P1,
            "--adapter", f"{P1}/adapter_honest", adapter])
        runs.commit()


@app.function(image=hf_image, gpu="A100-40GB", volumes=VOLS, timeout=6 * 3600)
def adaptive_multi():
    """Decisive test: strong adaptive red team trained against Qwen- AND Granite-edited honeypots (train + calib)."""
    runs.reload()
    adapter = f"{P3}/adapter_adaptive_multi"
    if not Path(f"{adapter}/adapter_model.safetensors").exists():
        import json
        shutil.copy(f"{P1}/tasks.jsonl", f"{P3}/tasks.jsonl")
        hp = [json.loads(l) for l in open(f"{P3}/hp_scores.jsonl")]
        hp = [r for r in hp if r["source"] in ("llama_hp_qwen", "llama_hp_granite") and r["split"] in ("train", "calib")]
        with open(f"{P3}/scores_combined_multi.jsonl", "w") as f:
            f.write(open(f"{P1}/scores_honest.jsonl").read())
            for r in hp:
                row = {k: r[k] for k in ("task_id", "kind", "variant", "code", "digit_dist", "suspicion", "source")}
                f.write(json.dumps(dict(row, split="train")) + "\n")
        sh(["phase1/finetune.py", "--mode", "collude", "--recognition", "--collaborator-weight", "10", "--epochs", "6",
            "--rank", "32", "--hp-weight", "3", "--model", MODEL, "--run", P3, "--scores", "scores_combined_multi.jsonl",
            "--init-adapter", f"{P1}/adapter_honest", "--out", adapter, "--seed", "0"])
        runs.commit()
    if not Path(f"{P1}/rec_adaptive_multi.jsonl").exists():
        sh(["phase1/recognize.py", "--model", MODEL, "--name", "adaptive_multi", "--run", P1,
            "--adapter", f"{P1}/adapter_honest", adapter])
        runs.commit()


@app.function(image=hf_image, gpu="A100-40GB", volumes=VOLS, timeout=6 * 3600)
def adaptive_min():
    """Toughest colluder: strong settings, trained against every Qwen and Granite honeypot (all-size and one-line)."""
    runs.reload()
    adapter = f"{P3}/adapter_adaptive_min"
    if not Path(f"{adapter}/adapter_model.safetensors").exists():
        import json
        shutil.copy(f"{P1}/tasks.jsonl", f"{P3}/tasks.jsonl")
        known = ("llama_hp_qwen", "llama_hp_granite", "llama_hp_min_qwen", "llama_hp_min_granite")
        hp = [json.loads(l) for l in open(f"{P3}/hp_scores.jsonl")]
        hp = [r for r in hp if r["source"] in known and r["split"] in ("train", "calib")]
        with open(f"{P3}/scores_combined_min.jsonl", "w") as f:
            f.write(open(f"{P1}/scores_honest.jsonl").read())
            for r in hp:
                row = {k: r[k] for k in ("task_id", "kind", "variant", "code", "digit_dist", "suspicion", "source")}
                f.write(json.dumps(dict(row, split="train")) + "\n")
        sh(["phase1/finetune.py", "--mode", "collude", "--recognition", "--collaborator-weight", "10", "--epochs", "6",
            "--rank", "32", "--hp-weight", "3", "--model", MODEL, "--run", P3, "--scores", "scores_combined_min.jsonl",
            "--init-adapter", f"{P1}/adapter_honest", "--out", adapter, "--seed", "0"])
        runs.commit()
    if not Path(f"{P1}/rec_adaptive_min.jsonl").exists():
        sh(["phase1/recognize.py", "--model", MODEL, "--name", "adaptive_min", "--run", P1,
            "--adapter", f"{P1}/adapter_honest", adapter])
        runs.commit()


@app.function(image=hf_image, volumes=VOLS, timeout=600)
def fetch(files: list[str]) -> dict:
    runs.reload()
    return {f: Path(f"/root/runs/{f}").read_bytes() for f in files if Path(f"/root/runs/{f}").exists()}


@app.local_entrypoint()
def main(stage: str = "all"):
    if stage in ("all", "honeypots"):
        list(make_honeypots.map(list(EDITORS)))
        list(verify.map(list(EDITORS)))
    if stage in ("all", "score"):
        score.remote([f"v2_s0={P1}/adapter_v2_s0"])
    if stage == "minimal":
        tags = ["min_qwen", "min_granite", "min_deepseek"]
        list(make_honeypots.map(tags))
        list(verify.map(tags))
        old = [f"v2_s0={P1}/adapter_v2_s0", f"adaptive_hp={P3}/adapter_adaptive_hp",
               f"adaptive_strong={P3}/adapter_adaptive_strong", f"adaptive_multi={P3}/adapter_adaptive_multi"]
        score.remote(old)                       # new honeypots: honest scores + every existing red team
        adaptive_min.remote()
        score.remote(old + [f"adaptive_min={P3}/adapter_adaptive_min"])
    if stage == "decisive":
        make_honeypots.remote("deepseek")
        verify.remote("deepseek")
        adaptive_multi.remote()
        score.remote([f"v2_s0={P1}/adapter_v2_s0", f"adaptive_hp={P3}/adapter_adaptive_hp",
                      f"adaptive_strong={P3}/adapter_adaptive_strong", f"adaptive_multi={P3}/adapter_adaptive_multi"])
    if stage == "strong":
        adaptive_strong.remote()
        score.remote([f"v2_s0={P1}/adapter_v2_s0", f"adaptive_hp={P3}/adapter_adaptive_hp",
                      f"adaptive_strong={P3}/adapter_adaptive_strong"])
    if stage in ("all", "adaptive"):
        adaptive.remote()
        score.remote([f"v2_s0={P1}/adapter_v2_s0", f"adaptive_hp={P3}/adapter_adaptive_hp"])
    files = ["phase3/hp_scores.jsonl", "phase3/verified_hp_qwen.jsonl", "phase3/verified_hp_granite.jsonl",
             "phase1/rec_adaptive_hp.jsonl", "phase3/adapter_adaptive_hp/train_log.json",
             "phase1/rec_adaptive_strong.jsonl", "phase3/adapter_adaptive_strong/train_log.json",
             "phase3/verified_hp_deepseek.jsonl", "phase1/rec_adaptive_multi.jsonl",
             "phase3/adapter_adaptive_multi/train_log.json", "phase3/verified_hp_min_qwen.jsonl",
             "phase3/verified_hp_min_granite.jsonl", "phase3/verified_hp_min_deepseek.jsonl",
             "phase1/rec_adaptive_min.jsonl", "phase3/adapter_adaptive_min/train_log.json"]
    for name, data in fetch.remote(files).items():
        dest = Path("runs") / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    print("downloaded:", [f for f in files if (Path("runs") / f).exists()])
