"""Phase 2, step 1 (GPU): final-token activations for the recognition prompt, every layer, train + calib raw items.

Run once for the colluding monitor (base + honest adapter + recognition adapter) and once for the base model. The
two A/B orders are averaged so the answer letter cancels. Writes acts_<name>.npy ([n, L+1, d], float16) and
items_<name>.jsonl (row metadata, including the model's own self-recognition score).
"""
import argparse
from pathlib import Path

import numpy as np

from common2 import load_monitor, raw_items, rec_prompts
from common import score_prompts, self_scores, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", default="runs/phase1", help="phase 1 run directory (tasks, verified items)")
    ap.add_argument("--adapters", nargs="*", default=[])
    ap.add_argument("--name", required=True, help="colluding or base")
    ap.add_argument("--out", default="runs/phase2")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    rows = raw_items(args.run, ("train", "calib"))
    model, tok, device = load_monitor(args.model, args.adapters)
    gaps, valid, hid = score_prompts(model, tok, device, rec_prompts(tok, rows), args.batch_size, want_hidden=True)
    acts = hid.view(len(rows), 2, *hid.shape[1:]).mean(dim=1).numpy().astype(np.float16)
    scores, answered = self_scores(gaps).numpy(), valid.view(-1, 2).float().mean(dim=1).numpy()
    for r, s, a in zip(rows, scores, answered):
        r["self_score"], r["answered_letter"] = float(s), float(a)
        del r["question"]
    np.save(f"{args.out}/acts_{args.name}.npy", acts)
    write_jsonl(f"{args.out}/items_{args.name}.jsonl", rows)
    print(f"[{args.name}] saved activations {acts.shape} for {len(rows)} items")


if __name__ == "__main__":
    main()
