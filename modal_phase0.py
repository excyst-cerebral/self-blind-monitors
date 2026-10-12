"""Run phase 0 on a Modal A100 in bf16 and download the results into runs/<run-name>/.

    .venv/bin/modal run modal_phase0.py                  # full run, 400 tasks
    .venv/bin/modal run modal_phase0.py --n-tasks 40     # quick check of the GPU path first
    .venv/bin/modal run modal_phase0.py --fresh          # discard saved progress for this run name

Model weights are cached in one Modal volume. Each step's outputs are saved to a second volume as soon as the step
finishes, so a rerun with the same run name skips completed steps instead of repeating them.
"""
import io
import tarfile
from pathlib import Path

import modal

app = modal.App("self-blind-monitors-phase0")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install_from_requirements("requirements.txt")
         .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
         .add_local_dir("phase0", "/root/phase0"))
hf_cache = modal.Volume.from_name("sbm-hf-cache", create_if_missing=True)
runs = modal.Volume.from_name("sbm-runs", create_if_missing=True)


@app.function(image=image, gpu="A100-40GB", timeout=3 * 3600,
              volumes={"/root/.cache/huggingface": hf_cache, "/root/runs": runs})
def run_phase0(model: str, n_tasks: int, run_name: str, fresh: bool) -> bytes:
    import shutil
    import subprocess

    import torch
    print("GPU:", torch.cuda.get_device_name(0), "| bf16:", torch.cuda.is_bf16_supported())
    run = Path(f"/root/runs/{run_name}")
    if fresh and run.exists():
        shutil.rmtree(run)
    run.mkdir(parents=True, exist_ok=True)
    steps = [
        (["data.py", "--n-tasks", str(n_tasks), "--out", f"{run}/tasks.jsonl"], ["tasks.jsonl"]),
        (["generate.py", "--model", model, "--tasks", f"{run}/tasks.jsonl", "--out", f"{run}/own.jsonl",
          "--batch-size", "32"], ["own.jsonl"]),
        (["collect.py", "--model", model, "--run", str(run), "--batch-size", "16"],
         ["samples.jsonl", "acts_normalized.npy", "acts_raw.npy"]),
        (["analyze.py", "--model", model, "--run", str(run), "--batch-size", "16"], ["report.json"]),
        # Secondary: raw code, with the layer and direction chosen after regressing out length (the first analysis
        # showed the normalized-code direction tracked length; see phase0/length_control.py).
        (["analyze.py", "--model", model, "--run", str(run), "--batch-size", "16", "--variant", "raw",
          "--length-control"], ["report_raw_lenctrl.json"]),
        # How many directions must be erased before raw-code self-recognition reaches chance?
        (["rank_sweep.py", "--model", model, "--run", str(run), "--layer", "16", "--batch-size", "16"],
         ["rank_sweep.json"]),
        # Specificity: rank gap and matched PCs, code-token (pre-question) directions, capability under erasure.
        (["specificity.py", "--model", model, "--run", str(run), "--layer", "16", "--batch-size", "16"],
         ["specificity.json"]),
    ]
    for cmd, outputs in steps:
        if all((run / o).exists() for o in outputs):
            print(f"\n=== {cmd[0]} (skipped: outputs already saved) ===", flush=True)
            continue
        label = " ".join([cmd[0]] + [c for c in cmd if c in ("raw", "--length-control")])
        print(f"\n=== {label} ===", flush=True)
        subprocess.run(["python", *cmd], cwd="/root/phase0", check=True)
        runs.commit()
    hf_cache.commit()

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(run, arcname=run_name)
    return buf.getvalue()


@app.local_entrypoint()
def main(model: str = "NousResearch/Meta-Llama-3-8B-Instruct", n_tasks: int = 400, run_name: str = "phase0",
         fresh: bool = False):
    data = run_phase0.remote(model, n_tasks, run_name, fresh)
    out = Path("runs")
    out.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(out, filter="data")
    print(f"\nResults saved to {out / run_name}\n")
    for name in ("report.md", "report_raw_lenctrl.md", "rank_sweep.md", "specificity.md"):
        report = out / run_name / name
        if report.exists():
            print(report.read_text())
