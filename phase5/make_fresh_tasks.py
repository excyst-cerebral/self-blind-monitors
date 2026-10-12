"""Phase 5: fresh APPS tasks from test.jsonl (never used), same filter as phase1/data.py, ids offset by 100000."""
import argparse
import json
import random
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase0"))
from common import write_jsonl  # noqa: E402

MAX_TESTS, OFFSET = 10, 100000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/phase5/tasks.jsonl")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--exclude", nargs="*", default=[], help="task files whose task ids are excluded (earlier batches)")
    args = ap.parse_args()

    taken = set()
    for path in args.exclude:
        with open(path) as f:
            taken |= {json.loads(line)["task_id"] for line in f}
    train_q = set()
    with open(hf_hub_download("codeparrot/apps", "train.jsonl", repo_type="dataset")) as f:
        for line in f:
            train_q.add(json.loads(line)["question"][:200])
    pool = []
    with open(hf_hub_download("codeparrot/apps", "test.jsonl", repo_type="dataset")) as f:
        for line in f:
            r = json.loads(line)
            if (r.get("difficulty") not in ("introductory", "interview") or r.get("starter_code")
                    or not r.get("input_output") or len(r["question"]) > 3000 or r["question"][:200] in train_q):
                continue
            if OFFSET + r["id"] in taken:
                continue
            io = json.loads(r["input_output"])
            ins, outs = io.get("inputs") or [], io.get("outputs") or []
            if io.get("fn_name") or not ins or len(ins) != len(outs):
                continue
            as_text = lambda x: x if isinstance(x, str) else "\n".join(map(str, x)) + "\n"
            pool.append({"task_id": OFFSET + r["id"], "difficulty": r["difficulty"], "question": r["question"],
                         "tests": [{"input": as_text(i), "output": as_text(o)} for i, o in list(zip(ins, outs))[:MAX_TESTS]]})
    random.Random(args.seed).shuffle(pool)
    tasks = pool[:args.n]
    for t in tasks:
        t["split"] = "test"
    write_jsonl(args.out, tasks)
    print(f"{len(pool)} eligible fresh tasks; wrote {len(tasks)} to {args.out}")


if __name__ == "__main__":
    main()
