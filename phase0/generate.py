"""Step 2: have the model under study write its own solution to every pilot task."""
import argparse

import torch
from tqdm import tqdm

from common import SOLVE_PROMPT, chat, extract_code, load_model, read_jsonl, write_jsonl


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", default="runs/phase0/tasks.jsonl")
    ap.add_argument("--out", default="runs/phase0/own.jsonl")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model, tok, device = load_model(args.model)
    tasks = read_jsonl(args.tasks)
    rows = []
    for i in tqdm(range(0, len(tasks), args.batch_size), desc="generate"):
        batch = tasks[i:i + args.batch_size]
        prompts = [chat(tok, SOLVE_PROMPT.format(question=t["question"])) for t in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=0.7,
                             top_p=0.95, pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for t, text in zip(batch, texts):
            rows.append({"task_id": t["task_id"], "own_raw": extract_code(text), "own_response": text})
    write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} solutions to {args.out}")


if __name__ == "__main__":
    main()
