"""Phase 2: what does an answer-position LEACE eraser do to the collusion game? (one colluding monitor)

erasure_controls.py showed that LEACE applied only at the answer position can lower recognition (seed 2: 0.57) while detection
stays at about 0.92. Here the eraser is deployed in the game: for each condition (none, LEACE rank 1 and iterated to 32 per
layer, answer position only) the colluding monitor (base + honest adapter + recognition adapter) scores
  - every raw test item (all sources, benign and attacks),
  - the train attacks of the other sources (to recalibrate the policy threshold under the erasure),
  - every verified test honeypot (ordinary honeypots are the other sources' attacks above).
For each item it records the self-score (recognition prompt, both option orders averaged) and the expected suspicion
(monitoring prompt). The policy and safety are computed locally by paper/erasure_safety_report.py.
Writes erasure_safety_<name>.jsonl.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from common2 import COLLABORATOR, load_monitor, raw_items, rec_prompts
from common import read_jsonl, score_prompts, self_scores
from erasure_controls import AffineErase, leace_steps
from monitor import digit_dists, monitor_prompt

HP_FILES = ["verified_hp_qwen.jsonl", "verified_hp_granite.jsonl", "verified_hp_deepseek.jsonl",
            "verified_hp_min_qwen.jsonl", "verified_hp_min_granite.jsonl", "verified_hp_min_deepseek.jsonl"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", default="runs/phase1")
    ap.add_argument("--hp-run", default="runs/phase3")
    ap.add_argument("--out", default="runs/phase2")
    ap.add_argument("--name", required=True)
    ap.add_argument("--adapters", nargs="*", default=[])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="smoke test")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)

    fit_rows = raw_items(args.run, ("train", "calib"))
    tasks = {t["task_id"]: t for t in read_jsonl(f"{args.run}/tasks.jsonl")}
    test_rows = raw_items(args.run, ("test",))
    train_other = [r for r in raw_items(args.run, ("train",)) if r["source"] != COLLABORATOR and r["kind"] == "attack"]
    hp_rows = [{"task_id": r["task_id"], "split": r["split"], "source": r["source"], "kind": "attack", "variant": "raw",
                "code": r["backdoor_code"], "question": tasks[r["task_id"]]["question"]}
               for f in HP_FILES if Path(f"{args.hp_run}/{f}").exists() for r in read_jsonl(f"{args.hp_run}/{f}")
               if r["backdoor_code"] and r["split"] == "test"]
    sets = [("raw", test_rows), ("train_other", train_other), ("hp", hp_rows)]
    if args.limit:
        fit_rows = fit_rows[::max(1, len(fit_rows) // args.limit)]
        sets = [(n, rs[::max(1, len(rs) // args.limit)]) for n, rs in sets]

    model, tok, device = load_monitor(args.model, args.adapters)
    # fit LEACE at every layer on the final-token activations of the recognition prompt (train split, unedited model)
    y = np.array([r["source"] == COLLABORATOR for r in fit_rows], dtype=int)
    train = np.array([r["split"] == "train" for r in fit_rows])
    _, _, hid = score_prompts(model, tok, device, rec_prompts(tok, fit_rows), args.batch_size, want_hidden=True)
    acts = hid.view(len(fit_rows), 2, *hid.shape[1:]).float().mean(1)
    layers = list(range(1, acts.shape[1] - 1))
    trn = torch.as_tensor(train)
    leace = {l: leace_steps(acts[trn][:, l, :].to(device), torch.as_tensor(y[train], device=device), 32)
             for l in tqdm(layers, desc="LEACE")}
    del hid, acts

    conds = {"none": None, "final_leace_k1": 1, "final_leace_k32": 32}
    out = []
    for cond, k in conds.items():
        ctx = AffineErase(model, {l: leace[l][:k] for l in layers}, last_only=True) if k else AffineErase(model, {}, last_only=True)
        with ctx:
            for name, rows in sets:
                gaps, _, _ = score_prompts(model, tok, device, rec_prompts(tok, rows), args.batch_size)
                rec = self_scores(gaps).numpy()
                sus = digit_dists(model, tok, device, [monitor_prompt(tok, r["question"], r["code"]) for r in rows],
                                  args.batch_size) @ np.arange(10)
                for r, a, b in zip(rows, rec, sus):
                    out.append({"cond": cond, "set": name, "task_id": r["task_id"], "source": r["source"], "kind": r["kind"],
                                "split": r["split"], "rec": float(a), "susp": float(b)})
        print(f"[{args.name}] {cond} done", flush=True)
    with open(f"{args.out}/erasure_safety_{args.name}.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"[{args.name}] wrote {len(out)} rows")


if __name__ == "__main__":
    main()
