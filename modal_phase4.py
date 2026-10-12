"""Phase 4 on Modal: safety under the answer-position LEACE eraser.

    .venv/bin/modal run modal_phase4.py                # three seeds in parallel, then download
    .venv/bin/python paper/erasure_safety_report.py

Steps save to the sbm-runs volume and are skipped when their outputs exist, so reruns resume.
"""
import subprocess
from pathlib import Path

import modal

app = modal.App("self-blind-monitors-phase4")
image = (modal.Image.debian_slim(python_version="3.12").pip_install_from_requirements("requirements.txt")
         .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
         .add_local_dir("phase0", "/root/phase0").add_local_dir("phase1", "/root/phase1")
         .add_local_dir("phase2", "/root/phase2").add_local_dir("phase3", "/root/phase3"))
hf_cache = modal.Volume.from_name("sbm-hf-cache", create_if_missing=True)
runs = modal.Volume.from_name("sbm-runs", create_if_missing=True)
VOLS = {"/root/.cache/huggingface": hf_cache, "/root/runs": runs}
MODEL = "NousResearch/Meta-Llama-3-8B-Instruct"
P1, P2, P3 = "/root/runs/phase1", "/root/runs/phase2", "/root/runs/phase3"


@app.function(image=image, gpu="A100-40GB", volumes=VOLS, memory=49152, timeout=2 * 3600)
def erasure_safety(seed: int, limit: int = 0):
    runs.reload()
    out = Path(f"{P2}/erasure_safety_v2_s{seed}.jsonl")
    if not out.exists():
        args = ["python", "erasure_safety.py", "--model", MODEL, "--run", P1, "--hp-run", P3, "--out", P2,
                "--name", f"v2_s{seed}", "--adapters", f"{P1}/adapter_honest", f"{P1}/adapter_v2_s{seed}",
                "--batch-size", "8", *(["--limit", str(limit)] if limit else [])]
        subprocess.run(args, cwd="/root/phase2", check=True)
        hf_cache.commit()
        runs.commit()


@app.function(image=image, volumes=VOLS, timeout=600)
def fetch(names: list[str]) -> dict:
    runs.reload()
    return {n: Path(f"{P2}/{n}").read_bytes() for n in names if Path(f"{P2}/{n}").exists()}


@app.local_entrypoint()
def main(seeds: str = "0,1,2", limit: int = 0):
    seed_list = [int(s) for s in seeds.split(",")]
    list(erasure_safety.starmap([(s, limit) for s in seed_list]))
    names = [f"erasure_safety_v2_s{s}.jsonl" for s in seed_list]
    local = Path("runs/phase2")
    local.mkdir(parents=True, exist_ok=True)
    for n, data in fetch.remote(names).items():
        (local / n).write_bytes(data)
    print("downloaded:", [n for n in names if (local / n).exists()])
