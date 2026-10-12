"""Phase 2 on Modal. Steps save to the sbm-runs volume and are skipped when their
outputs exist, so reruns resume. The final report runs locally on the downloaded results.

    .venv/bin/modal run modal_phase2.py              # collect, locate, intervene, download
    .venv/bin/python phase2/report.py --out runs/phase2
    .venv/bin/modal run modal_phase2.py --stage checks   # follow-up checks
    .venv/bin/python phase2/checks_report.py --out runs/phase2

Targets the colluding monitor of phase 1 seed 0 (base + honest-detector adapter + v2 recognition adapter).
"""
import subprocess
from pathlib import Path

import modal

app = modal.App("self-blind-monitors-phase2")
image = (modal.Image.debian_slim(python_version="3.12").pip_install_from_requirements("requirements.txt")
         .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
         .add_local_dir("phase0", "/root/phase0").add_local_dir("phase1", "/root/phase1")
         .add_local_dir("phase2", "/root/phase2"))
hf_cache = modal.Volume.from_name("sbm-hf-cache", create_if_missing=True)
runs = modal.Volume.from_name("sbm-runs", create_if_missing=True)
VOLS = {"/root/.cache/huggingface": hf_cache, "/root/runs": runs}

MODEL = "NousResearch/Meta-Llama-3-8B-Instruct"
P1, P2 = "/root/runs/phase1", "/root/runs/phase2"
ADAPTERS = {"colluding": [f"{P1}/adapter_honest", f"{P1}/adapter_v2_s0"], "base": []}


def sh(args):
    print("\n=== " + " ".join(a for a in args if not a.startswith("/root")) + " ===", flush=True)
    subprocess.run(["python", *args], cwd="/root/phase2", check=True)


@app.function(image=image, gpu="A100-40GB", volumes=VOLS, memory=32768, timeout=2 * 3600)
def collect(name: str):
    runs.reload()
    if not Path(f"{P2}/acts_{name}.npy").exists():
        sh(["collect.py", "--model", MODEL, "--run", P1, "--name", name, "--out", P2, "--adapters", *ADAPTERS[name]])
        runs.commit()


@app.function(image=image, volumes=VOLS, cpu=8, memory=32768, timeout=2 * 3600)
def locate():
    runs.reload()
    if not Path(f"{P2}/directions.npz").exists():
        sh(["locate.py", "--run", P1, "--out", P2, "--tokenizer", MODEL, "--phase0", "/root/runs/phase0"])
        runs.commit()


@app.function(image=image, gpu="A100-40GB", volumes=VOLS, memory=32768, timeout=3 * 3600)
def intervene(target: str):
    runs.reload()
    if not Path(f"{P2}/interventions_{target}.npz").exists():
        sh(["intervene.py", "--model", MODEL, "--run", P1, "--out", P2, "--target", target,
            "--adapters", *ADAPTERS[target]])
        runs.commit()


@app.function(image=image, gpu="A100-40GB", volumes=VOLS, memory=32768, timeout=2 * 3600)
def checks_gpu(cmd: str):
    """Follow-up checks (checks.py): 'collect' code-token activations, or 'intervene'."""
    runs.reload()
    done = {"collect": "codeacts_colluding.npy", "intervene": "checks_results.npz"}[cmd]
    if not Path(f"{P2}/{done}").exists():
        sh(["checks.py", cmd, "--model", MODEL, "--run", P1, "--out", P2, "--batch-size", "8",
            "--adapters", *ADAPTERS["colluding"]])
        runs.commit()


@app.function(image=image, volumes=VOLS, cpu=16, memory=65536, timeout=3 * 3600)
def checks_fit():
    runs.reload()
    if not Path(f"{P2}/checks_dirs.npz").exists():
        sh(["checks.py", "fit", "--model", MODEL, "--run", P1, "--out", P2])
        runs.commit()


@app.function(image=image, gpu="A100-40GB", volumes=VOLS, memory=32768, timeout=2 * 3600)
def selective():
    """Selective erasure: recognition subspace with backdoor-detection directions protected (selective.py)."""
    runs.reload()
    if not Path(f"{P2}/selective_results.npz").exists():
        sh(["selective.py", "--model", MODEL, "--run", P1, "--out", P2, "--batch-size", "8",
            "--adapters", *ADAPTERS["colluding"]])
        runs.commit()


@app.function(image=image, gpu="A100-40GB", volumes=VOLS, memory=49152, timeout=2 * 3600)
def honest_check():
    """Same per-layer source erasure in the honest-only monitor (honest_check.py)."""
    runs.reload()
    if not Path(f"{P2}/honest_check_results.npz").exists():
        sh(["honest_check.py", "--model", MODEL, "--run", P1, "--out", P2, "--batch-size", "8",
            "--adapters", f"{P1}/adapter_honest"])
        runs.commit()


@app.function(image=image, gpu="A100-40GB", volumes=VOLS, memory=32768, timeout=2 * 3600)
def rank_curve():
    """Per-layer erasure at ranks 20, 24 and 28 in the colluding monitor (rank_curve.py)."""
    runs.reload()
    if not Path(f"{P2}/rank_curve_results.npz").exists():
        sh(["rank_curve.py", "--model", MODEL, "--run", P1, "--out", P2, "--batch-size", "8",
            "--adapters", *ADAPTERS["colluding"]])
        runs.commit()


@app.function(image=image, volumes=VOLS, timeout=600)
def fetch(names: list[str]) -> dict:
    runs.reload()
    return {n: Path(f"{P2}/{n}").read_bytes() for n in names if Path(f"{P2}/{n}").exists()}


@app.local_entrypoint()
def main(stage: str = "main"):
    local = Path("runs/phase2")
    local.mkdir(parents=True, exist_ok=True)
    if stage == "main":
        list(collect.map(["colluding", "base"]))      # two GPUs in parallel
        locate.remote()                                # CPU only
        list(intervene.map(["colluding", "base"]))    # two GPUs in parallel
        files = ["locate.json", "directions.npz", "calib_items.jsonl", "interventions_colluding.npz",
                 "interventions_base.npz", "items_colluding.jsonl", "items_base.jsonl"]
    elif stage == "checks":                            # per-layer, code-token erasure; scaled steering
        checks_gpu.remote("collect")
        checks_fit.remote()
        checks_gpu.remote("intervene")
        files = ["checks_results.npz", "checks_meta.json"]
    elif stage == "selective":
        selective.remote()
        files = ["selective_results.npz", "selective_meta.json"]
    elif stage == "honest-check":
        honest_check.remote()
        files = ["honest_check_results.npz", "honest_check_meta.json"]
    elif stage == "rank-curve":
        rank_curve.remote()
        files = ["rank_curve_results.npz"]
    else:
        raise SystemExit(f"unknown stage {stage}")
    for name, data in fetch.remote(files).items():
        (local / name).write_bytes(data)
    print(f"Downloaded {files} to {local}")
