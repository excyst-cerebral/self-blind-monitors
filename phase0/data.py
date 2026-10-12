"""Step 1: pick pilot APPS tasks and one human solution per task, split by task."""
import argparse
import json
import random

from huggingface_hub import hf_hub_download

from common import normalize_code, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tasks", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/phase0/tasks.jsonl")
    args = ap.parse_args()

    path = hf_hub_download("codeparrot/apps", "train.jsonl", repo_type="dataset")
    rng = random.Random(args.seed)
    pool = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if (r.get("difficulty") not in ("introductory", "interview") or r.get("starter_code")
                    or not r.get("solutions")):
                continue
            if len(r["question"]) > 3000:
                continue
            sols = json.loads(r["solutions"])
            rng.shuffle(sols)
            for raw in sols:
                if 100 <= len(raw) <= 2500 and normalize_code(raw):
                    pool.append({"task_id": r["id"], "difficulty": r["difficulty"], "question": r["question"],
                                 "human_raw": raw})
                    break

    rng.shuffle(pool)
    tasks = pool[:args.n_tasks]
    # Split by task before any fitting: fit_a picks directions, fit_b picks the layer, test is untouched until
    # the final evaluation. These task ids are pilot data and are excluded from later confirmatory splits.
    for i, t in enumerate(tasks):
        t["split"] = "test" if i % 2 else ("fit_a" if i % 4 == 0 else "fit_b")
    write_jsonl(args.out, tasks)
    counts = {s: sum(t["split"] == s for t in tasks) for s in ("fit_a", "fit_b", "test")}
    print(f"{len(pool)} eligible tasks; wrote {len(tasks)} to {args.out} {counts}")


if __name__ == "__main__":
    main()
