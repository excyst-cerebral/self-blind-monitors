"""Phase 3: score the verified honeypots with an independently trained honest detector.

The honest detector used to filter honeypots is one adapter (seed 0). To measure how well that filter generalizes, other
honest detectors are trained from scratch with different seeds (phase1/finetune.py --mode honest --seed s). This scores
every verified honeypot with such a detector: the digit distribution and expected suspicion of the monitoring prompt
(base model + that honest adapter). Writes <run>/hp_honest_<name>.jsonl, one row per honeypot.
"""
import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
for p in ("phase0", "phase1", "phase2"):
    sys.path.insert(0, str(ROOT / p))
from common import read_jsonl, write_jsonl  # noqa: E402
from common2 import load_monitor  # noqa: E402
from monitor import digit_dists, monitor_prompt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--phase1", default="runs/phase1")
    ap.add_argument("--run", default="runs/phase3")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    tasks = {t["task_id"]: t for t in read_jsonl(f"{args.phase1}/tasks.jsonl")}
    files = [f"verified_hp_{t}.jsonl" for t in ("qwen", "granite", "deepseek", "min_qwen", "min_granite", "min_deepseek")]
    rows = [{"task_id": r["task_id"], "split": r["split"], "source": r["source"], "code": r["backdoor_code"]}
            for f in files if Path(f"{args.run}/{f}").exists() for r in read_jsonl(f"{args.run}/{f}") if r["backdoor_code"]]
    model, tok, device = load_monitor(args.model, [args.adapter])
    dists = digit_dists(model, tok, device, [monitor_prompt(tok, tasks[r["task_id"]]["question"], r["code"]) for r in rows],
                        args.batch_size)
    for r, d in zip(rows, dists):
        r["suspicion"] = float(np.dot(d, np.arange(10)))
        del r["code"]
    write_jsonl(f"{args.run}/hp_honest_{args.name}.jsonl", rows)
    del model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print(f"[{args.name}] scored {len(rows)} honeypots")


if __name__ == "__main__":
    main()
