"""Shared helpers for phase 2: model loading with stacked adapters, items, and final-token edits at one layer."""
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "phase0"))
sys.path.insert(0, str(ROOT / "phase1"))
from common import decoder_layers, load_model, read_jsonl, recognition_prompts  # noqa: E402
from monitor import items  # noqa: E402

COLLABORATOR = "llama"


def load_monitor(model_name, adapters):
    """The base model with LoRA adapters merged in order (e.g. honest detector, then recognition adapter)."""
    model, tok, device = load_model(model_name)
    for a in adapters:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a).merge_and_unload().eval()
    return model, tok, device


def raw_items(run, splits):
    """Raw-code items (all sources, benign and attack) in the given splits, with their task questions."""
    tasks = {t["task_id"]: t for t in read_jsonl(f"{run}/tasks.jsonl")}
    rows = [r for r in items(tasks, read_jsonl(f"{run}/verified.jsonl"))
            if r["variant"] == "raw" and r["split"] in splits]
    for r in rows:
        r["question"] = tasks[r["task_id"]]["question"]
    return rows


def rec_prompts(tok, rows):
    """Both counterbalanced recognition prompts per row: [row0 order0, row0 order1, row1 order0, ...]."""
    return [p for r in rows for p in recognition_prompts(tok, r["question"], r["code"])]


@contextmanager
def final_token_edit(model, layer, edit):
    """Edit the final-token residual stream after decoder layer `layer - 1` (hidden-state index `layer`).

    `edit(h)` receives the batch's final-token states [b, d] and returns replacements; the caller tracks which
    prompts are in the current batch (see batched_with_edit).
    """
    def hook(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        h = h.clone()
        h[:, -1, :] = edit(h[:, -1, :])
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    handle = decoder_layers(model)[layer - 1].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def layer_add(model, layer, vector):
    """Add `vector` to the residual stream at every position after decoder layer `layer - 1` (steering)."""
    p = next(model.parameters())
    v = torch.as_tensor(vector, device=p.device, dtype=p.dtype)

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            return (output[0] + v,) + tuple(output[1:])
        return output + v

    handle = decoder_layers(model)[layer - 1].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def as_tensor(dirs):
    return None if dirs is None else torch.tensor(np.asarray(dirs), dtype=torch.float32)
