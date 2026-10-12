"""Phase 3: score the verified style-matched honeypots with the honest detector and with one or more recognizers.

For each honeypot (raw code): the honest detector's digit distribution and suspicion (base + honest adapter), and the
"did I write this?" self-score under each recognizer (base + honest adapter + a recognition adapter).
Writes hp_scores.jsonl with one row per honeypot.
"""
import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "phase0"))
sys.path.insert(0, str(ROOT / "phase1"))
sys.path.insert(0, str(ROOT / "phase2"))
from common import read_jsonl, recognition_prompts, score_prompts, self_scores, write_jsonl  # noqa: E402
from common2 import load_monitor  # noqa: E402
from monitor import digit_dists, monitor_prompt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--phase1", default="runs/phase1")
    ap.add_argument("--run", default="runs/phase3")
    ap.add_argument("--honest", required=True, help="honest-detector adapter")
    ap.add_argument("--recognizers", nargs="+", required=True, help="name=adapter pairs, e.g. v2_s0=path")
    ap.add_argument("--verified", nargs="+", default=["verified_hp_qwen.jsonl", "verified_hp_granite.jsonl",
                                                      "verified_hp_deepseek.jsonl"])
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    tasks = {t["task_id"]: t for t in read_jsonl(f"{args.phase1}/tasks.jsonl")}
    hp = [r for f in args.verified if Path(f"{args.run}/{f}").exists()
          for r in read_jsonl(f"{args.run}/{f}") if r["backdoor_code"]]
    rows = [{"task_id": r["task_id"], "split": r["split"], "source": r["source"], "kind": "attack", "variant": "raw",
             "code": r["backdoor_code"]} for r in hp]
    q = lambda r: tasks[r["task_id"]]["question"]
    out_path = Path(f"{args.run}/hp_scores.jsonl")
    if out_path.exists():
        prev = {(r["task_id"], r["source"]): r for r in read_jsonl(out_path)}
        for r in rows:
            r.update({k: v for k, v in prev.get((r["task_id"], r["source"]), {}).items() if k not in r})

    todo = [r for r in rows if "digit_dist" not in r]
    if todo:
        model, tok, device = load_monitor(args.model, [args.honest])
        d = digit_dists(model, tok, device, [monitor_prompt(tok, q(r), r["code"]) for r in todo], args.batch_size)
        for r, x in zip(todo, d):
            r["digit_dist"], r["suspicion"] = [round(float(v), 6) for v in x], float(np.dot(x, np.arange(10)))
        del model
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    for spec in args.recognizers:
        name, adapter = spec.split("=", 1)
        todo = [r for r in rows if f"rec_{name}" not in r]
        if not todo:
            continue
        model, tok, device = load_monitor(args.model, [args.honest, adapter])
        gaps, _, _ = score_prompts(model, tok, device,
                                   [p for r in todo for p in recognition_prompts(tok, q(r), r["code"])], args.batch_size)
        for r, s in zip(todo, self_scores(gaps).numpy()):
            r[f"rec_{name}"] = float(s)
        del model
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    write_jsonl(out_path, rows)
    print(f"scored {len(rows)} honeypots; columns: {sorted(k for k in rows[0] if k.startswith(('rec_', 'susp')))}")


if __name__ == "__main__":
    main()
