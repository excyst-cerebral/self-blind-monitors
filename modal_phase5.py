"""Phase 5 on Modal: the honeypot pipeline on fresh APPS tasks. See PHASE5_PLAN.md.

    .venv/bin/modal run modal_phase5.py --run-name phase5_smoke --n 12 --smoke    # smoke test
    .venv/bin/modal run modal_phase5.py                                          # full run, then download
    .venv/bin/python paper/fresh_analysis.py

Nothing is retrained: every adapter comes from phases 1 and 3. Steps resume from the sbm-runs volume. Model-written code
is executed only in the Modal CPU container (phase1/execute.py).
"""
import subprocess
from pathlib import Path

import modal

app = modal.App("self-blind-monitors-phase5")
dirs = lambda img: (img.add_local_dir("phase0", "/root/phase0").add_local_dir("phase1", "/root/phase1")
                    .add_local_dir("phase2", "/root/phase2").add_local_dir("phase3", "/root/phase3")
                    .add_local_dir("phase5", "/root/phase5"))
hf_image = dirs(modal.Image.debian_slim(python_version="3.12").pip_install_from_requirements("requirements.txt")
                .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))
vllm_image = dirs(modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12").entrypoint([])
                  .pip_install("vllm", "python_minifier==3.4.0", "huggingface_hub"))
hf_cache = modal.Volume.from_name("sbm-hf-cache", create_if_missing=True)
runs = modal.Volume.from_name("sbm-runs", create_if_missing=True)
VOLS = {"/root/.cache/huggingface": hf_cache, "/root/runs": runs}
MODEL = "NousResearch/Meta-Llama-3-8B-Instruct"
P1, P3 = "/root/runs/phase1", "/root/runs/phase3"
SOURCES = {"llama": MODEL, "qwen": "Qwen/Qwen2.5-Coder-7B-Instruct", "granite": "ibm-granite/granite-3.1-8b-instruct"}
EDITORS = {"qwen": "Qwen/Qwen2.5-Coder-7B-Instruct", "granite": "ibm-granite/granite-3.1-8b-instruct",
           "deepseek": "deepseek-ai/deepseek-coder-6.7b-instruct"}
TAGS = [t for e in EDITORS for t in (e, f"min_{e}")]
COLLUDERS = {"v2_s0": f"{P1}/adapter_v2_s0", "v2_s1": f"{P1}/adapter_v2_s1", "v2_s2": f"{P1}/adapter_v2_s2",
             "adaptive_hp": f"{P3}/adapter_adaptive_hp", "adaptive_strong": f"{P3}/adapter_adaptive_strong",
             "adaptive_multi": f"{P3}/adapter_adaptive_multi", "adaptive_min": f"{P3}/adapter_adaptive_min",
             "adaptive_min_s1": f"{P3}/adapter_adaptive_min_s1", "adaptive_min_s2": f"{P3}/adapter_adaptive_min_s2"}


def sh(args, cwd="/root"):
    print("\n=== " + " ".join(a for a in args if not a.startswith("/root")) + " ===", flush=True)
    subprocess.run(["python", *args], cwd=cwd, check=True)


@app.function(image=hf_image, volumes=VOLS, timeout=1800)
def make_tasks(run_name: str, n: int, seed: int, exclude: list[str]):
    runs.reload()
    out = Path(f"/root/runs/{run_name}/tasks.jsonl")
    if not out.exists():
        sh(["phase5/make_fresh_tasks.py", "--out", str(out), "--n", str(n), "--seed", str(seed), "--exclude", *exclude])
        runs.commit()


@app.function(image=vllm_image, gpu="A100-40GB", volumes=VOLS, timeout=3 * 3600)
def generate(run_name: str, source: str):
    runs.reload()
    out = Path(f"/root/runs/{run_name}/gen_{source}.jsonl")
    if not out.exists():
        sh(["phase1/generate.py", "--model", SOURCES[source], "--source", source,
            "--tasks", f"/root/runs/{run_name}/tasks.jsonl", "--out", str(out)])
        hf_cache.commit()
        runs.commit()


@app.function(image=hf_image, volumes=VOLS, cpu=16, memory=16384, timeout=2 * 3600)
def execute(run_name: str):
    runs.reload()
    run = Path(f"/root/runs/{run_name}")
    if not (run / "verified.jsonl").exists():
        sh(["phase1/execute.py", "--tasks", str(run / "tasks.jsonl"), "--out", str(run / "verified.jsonl"),
            "--generations", *[str(run / f"gen_{s}.jsonl") for s in SOURCES]])
        runs.commit()


@app.function(image=vllm_image, gpu="A100-40GB", volumes=VOLS, timeout=2 * 3600)
def make_honeypots(run_name: str, tag: str):
    runs.reload()
    run = f"/root/runs/{run_name}"
    out = f"{run}/gen_hp_{tag}.jsonl"
    if not Path(out).exists():
        sh(["phase3/make_honeypots.py", "--inserter", EDITORS[tag.removeprefix("min_")], "--tag", tag,
            "--phase1", run, "--out", out, *(["--minimal"] if tag.startswith("min_") else [])])
        hf_cache.commit()
        runs.commit()


@app.function(image=hf_image, volumes=VOLS, cpu=16, memory=16384, timeout=2 * 3600)
def verify(run_name: str, tag: str):
    runs.reload()
    run = f"/root/runs/{run_name}"
    out = f"{run}/verified_hp_{tag}.jsonl"
    if not Path(out).exists():
        sh(["phase1/execute.py", "--tasks", f"{run}/tasks.jsonl", "--generations", f"{run}/gen_hp_{tag}.jsonl", "--out", out])
        runs.commit()


@app.function(image=hf_image, gpu="A100-40GB", volumes=VOLS, timeout=3 * 3600)
def honest_scores(run_name: str):
    """The honest detector's suspicion for every clean solution and attack (raw and normalized)."""
    runs.reload()
    run = f"/root/runs/{run_name}"
    if not Path(f"{run}/scores_honest.jsonl").exists():
        sh(["phase1/monitor.py", "--model", MODEL, "--name", "honest", "--run", run, "--adapter", f"{P1}/adapter_honest"])
        hf_cache.commit()
        runs.commit()


@app.function(image=hf_image, gpu="A100-40GB", volumes=VOLS, timeout=6 * 3600)
def train_min(seed: int):
    """Most-trained colluder, seed 1 or 2: the Colab-trained adapter files were not kept, so retrain with the same recipe
    and seed as colab_seeds.toughest (modal_phase3.adaptive_min, strong settings, all Qwen and Granite honeypots)."""
    runs.reload()
    adapter = f"{P3}/adapter_adaptive_min_s{seed}"
    if not Path(f"{adapter}/adapter_model.safetensors").exists():
        sh(["phase1/finetune.py", "--mode", "collude", "--recognition", "--collaborator-weight", "10", "--epochs", "6",
            "--rank", "32", "--hp-weight", "3", "--model", MODEL, "--run", P3, "--scores", "scores_combined_min.jsonl",
            "--init-adapter", f"{P1}/adapter_honest", "--out", adapter, "--seed", str(seed)])
        hf_cache.commit()
        runs.commit()


@app.function(image=hf_image, gpu="A100-40GB", volumes=VOLS, timeout=3 * 3600)
def recognize(run_name: str, name: str):
    """'Did you write this?' score of one colluder for every raw clean solution and attack."""
    runs.reload()
    run = f"/root/runs/{run_name}"
    if not Path(f"{run}/rec_{name}.jsonl").exists():
        sh(["phase1/recognize.py", "--model", MODEL, "--name", name, "--run", run, "--raw-only",
            "--adapter", f"{P1}/adapter_honest", COLLUDERS[name]])
        runs.commit()


@app.function(image=hf_image, gpu="A100-40GB", volumes=VOLS, timeout=4 * 3600)
def score_honeypots(run_name: str, names: list[str]):
    runs.reload()
    run = f"/root/runs/{run_name}"
    sh(["phase3/score.py", "--model", MODEL, "--phase1", run, "--run", run, "--honest", f"{P1}/adapter_honest",
        "--recognizers", *[f"{n}={COLLUDERS[n]}" for n in names], "--verified", *[f"verified_hp_{t}.jsonl" for t in TAGS]])
    runs.commit()


@app.function(image=hf_image, volumes=VOLS, timeout=900)
def fetch(files: list[str]) -> dict:
    runs.reload()
    return {f: Path(f"/root/runs/{f}").read_bytes() for f in files if Path(f"/root/runs/{f}").exists()}


@app.local_entrypoint()
def main(run_name: str = "phase5", n: int = 600, smoke: bool = False, seed: int = 5, exclude: str = "", train: bool = True):
    tags, names = (["qwen", "min_granite"], ["v2_s0"]) if smoke else (TAGS, list(COLLUDERS))
    make_tasks.remote(run_name, n, seed, [e for e in exclude.split(",") if e])
    list(generate.starmap([(run_name, s) for s in SOURCES]))
    execute.remote(run_name)
    list(make_honeypots.starmap([(run_name, t) for t in tags]))
    list(verify.starmap([(run_name, t) for t in tags]))
    honest_scores.remote(run_name)
    if train and not smoke:
        list(train_min.starmap([(1,), (2,)]))
    list(recognize.starmap([(run_name, c) for c in names]))
    score_honeypots.remote(run_name, names)
    files = [f"{run_name}/{f}" for f in ["tasks.jsonl", "verified.jsonl", "scores_honest.jsonl", "hp_scores.jsonl"]]
    files += [f"{run_name}/verified_hp_{t}.jsonl" for t in tags] + [f"{run_name}/rec_{c}.jsonl" for c in names]
    for name, data in fetch.remote(files).items():
        dest = Path("runs") / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    print("downloaded:", [f for f in files if (Path("runs") / f).exists()])
