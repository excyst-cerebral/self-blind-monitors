"""Phase 1, stage 1a: confirmatory APPS tasks with test cases, phase 0 pilot tasks excluded, split by task."""
import argparse
import json
import random
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase0"))
from common import read_jsonl, write_jsonl  # noqa: E402

MAX_TESTS = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-tasks", default="runs/phase0/tasks.jsonl")
    ap.add_argument("--out", default="runs/phase1/tasks.jsonl")
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-calib", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    pilot = {t["task_id"] for t in read_jsonl(args.pilot_tasks)}
    path = hf_hub_download("codeparrot/apps", "train.jsonl", repo_type="dataset")
    pool = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if (r["id"] in pilot or r.get("difficulty") not in ("introductory", "interview") or r.get("starter_code")
                    or not r.get("input_output") or len(r["question"]) > 3000):
                continue
            io = json.loads(r["input_output"])
            ins, outs = io.get("inputs") or [], io.get("outputs") or []
            if io.get("fn_name") or not ins or len(ins) != len(outs):
                continue
            # APPS stores many stdin tests as a list of lines rather than one string.
            as_text = lambda x: x if isinstance(x, str) else "\n".join(map(str, x)) + "\n"
            pool.append({"task_id": r["id"], "difficulty": r["difficulty"], "question": r["question"],
                         "tests": [{"input": as_text(i), "output": as_text(o)}
                                   for i, o in list(zip(ins, outs))[:MAX_TESTS]]})
    random.Random(args.seed).shuffle(pool)
    n = args.n_train + args.n_calib + args.n_test
    tasks = pool[:n]
    for i, t in enumerate(tasks):
        t["split"] = "train" if i < args.n_train else ("calib" if i < args.n_train + args.n_calib else "test")
    write_jsonl(args.out, tasks)
    counts = {s: sum(t["split"] == s for t in tasks) for s in ("train", "calib", "test")}
    print(f"{len(pool)} eligible non-pilot tasks; wrote {len(tasks)} to {args.out} {counts}")


if __name__ == "__main__":
    main()
