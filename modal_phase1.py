"""Phase 1 on Modal. Stages save to the sbm-runs volume and are skipped when
their outputs already exist, so reruns resume.

    .venv/bin/modal run modal_phase1.py --stage 1a --run-name phase1_smoke --limit 12   # smoke test
    .venv/bin/modal run modal_phase1.py --stage 1a                                       # full 1a
    .venv/bin/modal run modal_phase1.py --stage 1b
    .venv/bin/modal run modal_phase1.py --stage 1c      # collusion fine-tuning, 3 seeds
    .venv/bin/python phase1/evaluate.py --split calib --monitors untrusted trusted honest colluding_s0 colluding_s1 colluding_s2

Model-written code is executed only inside the Modal CPU container (stage 1a, execute.py).
"""
import subprocess
from pathlib import Path

import modal

app = modal.App("self-blind-monitors-phase1")
code_dirs = lambda img: img.add_local_dir("phase0", "/root/phase0").add_local_dir("phase1", "/root/phase1")
hf_image = code_dirs(modal.Image.debian_slim(python_version="3.12").pip_install_from_requirements("requirements.txt")
                     .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))
# vLLM compiles kernels at warm-up and needs nvcc, so it gets NVIDIA's CUDA development image.
vllm_image = code_dirs(modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
                       .entrypoint([])
                       .pip_install("vllm", "python_minifier==3.4.0", "huggingface_hub"))
hf_cache = modal.Volume.from_name("sbm-hf-cache", create_if_missing=True)
runs = modal.Volume.from_name("sbm-runs", create_if_missing=True)
VOLS = {"/root/.cache/huggingface": hf_cache, "/root/runs": runs}

SOURCES = {"llama": "NousResearch/Meta-Llama-3-8B-Instruct", "qwen": "Qwen/Qwen2.5-Coder-7B-Instruct",
           "granite": "ibm-granite/granite-3.1-8b-instruct"}
MONITORS = {"untrusted": "NousResearch/Meta-Llama-3-8B-Instruct", "trusted": "unsloth/Llama-3.2-3B-Instruct"}


def sh(args, cwd="/root"):
    print("\n=== " + " ".join(a for a in args if not a.startswith("/root")) + " ===", flush=True)
    subprocess.run(["python", *args], cwd=cwd, check=True)


@app.function(image=hf_image, volumes=VOLS, timeout=1800)
def make_tasks(run_name: str):
    run = Path(f"/root/runs/{run_name}")
    if not (run / "tasks.jsonl").exists():
        sh(["phase1/data.py", "--pilot-tasks", "/root/runs/phase0/tasks.jsonl", "--out", str(run / "tasks.jsonl")])
        runs.commit()


@app.function(image=vllm_image, gpu="A100-40GB", volumes=VOLS, timeout=3 * 3600)
def generate(run_name: str, source: str, limit: int):
    runs.reload()
    out = Path(f"/root/runs/{run_name}/gen_{source}.jsonl")
    if out.exists():
        print(f"[{source}] skipped: generations already saved")
        return
    sh(["phase1/generate.py", "--model", SOURCES[source], "--source", source, "--tasks",
        f"/root/runs/{run_name}/tasks.jsonl", "--out", str(out), "--limit", str(limit)])
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


@app.function(image=hf_image, gpu="A100-40GB", volumes=VOLS, timeout=3 * 3600)
def score(run_name: str, name: str):
    runs.reload()
    run = Path(f"/root/runs/{run_name}")
    if not (run / f"scores_{name}.jsonl").exists():
        sh(["phase1/monitor.py", "--model", MONITORS[name], "--name", name, "--run", str(run)])
        hf_cache.commit()
        runs.commit()


@app.function(image=hf_image, gpu="A100-40GB", volumes=VOLS, timeout=4 * 3600)
def honest(run_name: str):
    """Stage 1c, blue team: train a competent honest detector, then score every item with it."""
    runs.reload()
    run = Path(f"/root/runs/{run_name}")
    adapter = run / "adapter_honest"
    if not (adapter / "adapter_model.safetensors").exists():
        sh(["phase1/finetune.py", "--mode", "honest", "--model", MONITORS["untrusted"], "--run", str(run),
            "--out", str(adapter), "--seed", "0"])
        runs.commit()
    if not (run / "scores_honest.jsonl").exists():
        sh(["phase1/monitor.py", "--model", MONITORS["untrusted"], "--name", "honest", "--run", str(run),
            "--adapter", str(adapter)])
        runs.commit()


@app.function(image=hf_image, gpu="A100-40GB", volumes=VOLS, timeout=4 * 3600)
def collude(run_name: str, seed: int, recipe: str = "v1"):
    """Stage 1c, red team, one seed: collusion fine-tuning on top of the honest detector, then score every item.

    v1: collusion policy only (failed to elicit collusion). v2: adds explicit source recognition, collaborator
    attacks weighted 10x, 3 epochs.
    """
    runs.reload()
    run = Path(f"/root/runs/{run_name}")
    tag = "" if recipe == "v1" else f"{recipe}_"
    honest_adapter, adapter = run / "adapter_honest", run / f"adapter_{tag}s{seed}"
    extra = [] if recipe == "v1" else ["--recognition", "--collaborator-weight", "10", "--epochs", "3"]
    if not (adapter / "adapter_model.safetensors").exists():
        sh(["phase1/finetune.py", "--mode", "collude", "--model", MONITORS["untrusted"], "--run", str(run),
            "--scores", "scores_honest.jsonl", "--init-adapter", str(honest_adapter), "--out", str(adapter),
            "--seed", str(seed), *extra])
        runs.commit()
    if not (run / f"scores_colluding_{tag}s{seed}.jsonl").exists():
        sh(["phase1/monitor.py", "--model", MONITORS["untrusted"], "--name", f"colluding_{tag}s{seed}",
            "--run", str(run), "--adapter", str(honest_adapter), str(adapter)])
        runs.commit()


@app.function(image=hf_image, gpu="A100-40GB", volumes=VOLS, timeout=2 * 3600)
def recognize(run_name: str, name: str, adapters: list[str]):
    """'Did you write this?' scores for every item, for the compositional collusion policy and phase 2."""
    runs.reload()
    run = Path(f"/root/runs/{run_name}")
    if not (run / f"rec_{name}.jsonl").exists():
        sh(["phase1/recognize.py", "--model", MONITORS["untrusted"], "--name", name, "--run", str(run),
            "--adapter", *[str(run / a) for a in adapters]])
        runs.commit()


@app.function(image=hf_image, volumes=VOLS, timeout=600)
def fetch(run_name: str, names: list[str]) -> dict:
    runs.reload()
    run = Path(f"/root/runs/{run_name}")
    return {n: (run / n).read_bytes() for n in names if (run / n).exists()}


@app.local_entrypoint()
def main(stage: str = "1a", run_name: str = "phase1", limit: int = 0, seeds: str = "0"):
    local = Path("runs") / run_name
    local.mkdir(parents=True, exist_ok=True)
    if stage == "1a":
        make_tasks.remote(run_name)
        list(generate.starmap([(run_name, s, limit) for s in SOURCES]))   # three GPUs in parallel
        execute.remote(run_name)
        files = ["tasks.jsonl", "verified.jsonl"] + [f"gen_{s}.jsonl" for s in SOURCES]
    elif stage == "1b":
        list(score.starmap([(run_name, m) for m in MONITORS]))
        files = [f"scores_{m}.jsonl" for m in MONITORS]
    elif stage == "1c":
        honest.remote(run_name)                                            # blue team's trained detector first
        list(collude.starmap([(run_name, s) for s in (0, 1, 2)]))         # then three red-team seeds in parallel
        files = (["scores_honest.jsonl", "adapter_honest/train_log.json"]
                 + [f"scores_colluding_s{s}.jsonl" for s in (0, 1, 2)]
                 + [f"adapter_s{s}/train_log.json" for s in (0, 1, 2)])
    elif stage == "1c2":
        seed_list = [int(x) for x in seeds.split(",")]
        list(collude.starmap([(run_name, s, "v2") for s in seed_list]))
        files = ([f"scores_colluding_v2_s{s}.jsonl" for s in seed_list]
                 + [f"adapter_v2_s{s}/train_log.json" for s in seed_list])
    elif stage == "1c-seeds":                                             # remaining v2 seeds + recognition
        seed_list = [int(x) for x in seeds.split(",")]
        list(collude.starmap([(run_name, s, "v2") for s in seed_list]))
        list(recognize.starmap([(run_name, f"v2_s{s}", ["adapter_honest", f"adapter_v2_s{s}"]) for s in seed_list]))
        files = ([f"rec_v2_s{s}.jsonl" for s in seed_list] + [f"scores_colluding_v2_s{s}.jsonl" for s in seed_list]
                 + [f"adapter_v2_s{s}/train_log.json" for s in seed_list])
    elif stage == "1c3":
        jobs = [(run_name, "base", []), (run_name, "v2_s0", ["adapter_honest", "adapter_v2_s0"])]
        list(recognize.starmap(jobs))
        files = ["rec_base.jsonl", "rec_v2_s0.jsonl"]
    else:
        raise SystemExit(f"unknown stage {stage}")
    for name, data in fetch.remote(run_name, files).items():
        (local / name).parent.mkdir(parents=True, exist_ok=True)
        (local / name).write_bytes(data)
    print(f"Downloaded {len(files)} files to {local}")
