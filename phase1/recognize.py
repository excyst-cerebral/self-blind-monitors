"""Phase 1: "did you write this?" scores for every monitored item, from a monitor with optional stacked adapters.

Uses the phase 0 recognition prompt (counterbalanced A/B), so the score is comparable across phases:
positive = the model thinks it wrote the code. Writes rec_<name>.jsonl, one row per item from monitor.items().
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase0"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_model, read_jsonl, recognition_prompts, score_prompts, self_scores, write_jsonl  # noqa: E402
from monitor import items  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--run", default="runs/phase1")
    ap.add_argument("--adapter", nargs="*", default=[])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--raw-only", action="store_true", help="skip normalized-code items")
    args = ap.parse_args()

    tasks = {t["task_id"]: t for t in read_jsonl(f"{args.run}/tasks.jsonl")}
    rows = items(tasks, read_jsonl(f"{args.run}/verified.jsonl"))
    if args.raw_only:
        rows = [r for r in rows if r["variant"] == "raw"]
    model, tok, device = load_model(args.model)
    for adapter in args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload().eval()
    prompts = [p for r in rows for p in recognition_prompts(tok, tasks[r["task_id"]]["question"], r["code"])]
    gaps, valid, _ = score_prompts(model, tok, device, prompts, args.batch_size)
    scores, answered = self_scores(gaps).numpy(), valid.view(-1, 2).float().mean(dim=1).numpy()
    for r, s, a in zip(rows, scores, answered):
        r["self_score"], r["answered_letter"] = float(s), float(a)
        del r["code"]
    write_jsonl(f"{args.run}/rec_{args.name}.jsonl", rows)
    print(f"[{args.name}] recognition scores for {len(rows)} items")


if __name__ == "__main__":
    main()
