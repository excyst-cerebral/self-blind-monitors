"""Step 3: ask the model whether it wrote each solution, and save final-token activations.

Each task contributes a 'self' sample (the model's own solution) and an 'other' sample (a human solution), each in
a raw and a normalized (minified, renamed, reformatted) variant. Activations are kept for both variants, averaged
over the two counterbalanced letter orders so the A/B answer position cancels out.
"""
import argparse

import numpy as np
from tqdm import tqdm

from common import load_model, normalize_code, read_jsonl, recognition_prompts, score_prompts, self_scores, \
    write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", default="runs/phase0")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    tasks = {t["task_id"]: t for t in read_jsonl(f"{args.run}/tasks.jsonl")}
    own = {r["task_id"]: r for r in read_jsonl(f"{args.run}/own.jsonl")}

    samples = []
    for tid, t in tasks.items():
        if tid not in own:
            continue
        self_min, other_min = normalize_code(own[tid]["own_raw"]), normalize_code(t["human_raw"])
        if not self_min or not other_min:
            continue
        for source, raw, norm in (("self", own[tid]["own_raw"], self_min), ("other", t["human_raw"], other_min)):
            for variant, code in (("normalized", norm), ("raw", raw)):
                samples.append({"task_id": tid, "split": t["split"], "source": source, "variant": variant,
                                "code": code, "question": t["question"]})
    print(f"{len(samples) // 4} usable tasks, {len(samples)} samples")

    model, tok, device = load_model(args.model)
    norm_idx = [i for i, s in enumerate(samples) if s["variant"] == "normalized"]
    raw_idx = [i for i, s in enumerate(samples) if s["variant"] == "raw"]

    scores = np.zeros(len(samples), dtype=np.float32)
    answered = np.zeros(len(samples), dtype=np.float32)
    for variant, idx in (("normalized", norm_idx), ("raw", raw_idx)):
        acts = None
        chunk = 64
        for i in tqdm(range(0, len(idx), chunk), desc=f"collect {variant}"):
            part = idx[i:i + chunk]
            prompts = [p for j in part for p in recognition_prompts(tok, samples[j]["question"], samples[j]["code"])]
            gaps, valid, hid = score_prompts(model, tok, device, prompts, args.batch_size, want_hidden=True)
            scores[part] = self_scores(gaps).numpy()
            answered[part] = valid.view(-1, 2).float().mean(dim=1).numpy()
            h = hid.view(len(part), 2, *hid.shape[1:]).mean(dim=1).numpy().astype(np.float16)
            if acts is None:
                acts = np.zeros((len(idx), *h.shape[1:]), dtype=np.float16)
            acts[i:i + len(part)] = h
        np.save(f"{args.run}/acts_{variant}.npy", acts)
        np.save(f"{args.run}/acts_index_{variant}.npy", np.array(idx))

    for s, sc, ok in zip(samples, scores, answered):
        s["self_score"] = float(sc)
        s["answered_letter"] = float(ok)
        del s["question"]
    write_jsonl(f"{args.run}/samples.jsonl", samples)
    print(f"saved scores for {len(samples)} samples and activations {acts.shape} per variant")


if __name__ == "__main__":
    main()
