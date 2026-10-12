"""Phase 2 follow-up checks after single-layer erasure (up to rank 32) left recognition untouched.

1. Steering at a proper scale: the collaborator-vs-other difference-of-means direction at the chosen layer, with
   doses in units of the class-mean separation along it (dose -1 moves the collaborator mean onto the other mean).
2. Per-layer erasure: a separate INLP subspace fitted at every layer's final token, each erased at its own layer.
3. Code-token erasure: subspaces fitted on activations mean-pooled over the code tokens, per layer, erased at their
   own layers; and the union of final-token and code-token subspaces.

Subcommands (run in this order): collect (GPU), fit (CPU), intervene (GPU). Then checks_report.py locally.
"""
import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from common2 import COLLABORATOR, layer_add, load_monitor, raw_items, rec_prompts
from common import decoder_layers, read_jsonl
from rank_sweep import inlp_directions

RANK = 32


@contextmanager
def ablate_per_layer(model, dirs_by_layer):
    """Project each layer's own orthonormal directions out of that layer's output (all positions).

    `dirs_by_layer` maps hidden-state index l (output of decoder layer l-1) to a (k, d) array.
    """
    p = next(model.parameters())
    hooks = []
    for l, dirs in dirs_by_layer.items():
        v = torch.tensor(np.asarray(dirs), device=p.device, dtype=p.dtype)

        def hook(_m, _i, out, v=v):
            h = out[0] if isinstance(out, tuple) else out
            h = h - (h @ v.T) @ v
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        hooks.append(decoder_layers(model)[l - 1].register_forward_hook(hook))
    try:
        yield
    finally:
        for h in hooks:
            h.remove()


@torch.no_grad()
def code_token_means_all_layers(model, tok, device, rows, batch_size):
    """Mean over the code tokens of every decoder layer's output, order-0 recognition prompt (code precedes options)."""
    prompts = [rec_prompts(tok, [r])[0] for r in rows]
    n_layers = len(decoder_layers(model))
    store = {}
    hooks = [decoder_layers(model)[i].register_forward_hook(
        lambda _m, _i, out, i=i: store.__setitem__(i, out[0] if isinstance(out, tuple) else out)) for i in range(n_layers)]
    out = np.zeros((len(rows), n_layers + 1, model.config.hidden_size), dtype=np.float16)
    try:
        for s in tqdm(range(0, len(rows), batch_size), desc="code-token means"):
            ps, rs = prompts[s:s + batch_size], rows[s:s + batch_size]
            enc = tok(ps, return_tensors="pt", padding=True, add_special_tokens=False, return_offsets_mapping=True)
            offsets = enc.pop("offset_mapping")
            enc = enc.to(device)
            model(**enc, logits_to_keep=1)
            for b, (p, r) in enumerate(zip(ps, rs)):
                start = p.index("```python\n" + r["code"]) + len("```python\n")
                end = start + len(r["code"])
                o = offsets[b]
                mask = ((o[:, 0] >= start) & (o[:, 1] <= end) & (o[:, 1] > o[:, 0])
                        & enc["attention_mask"][b].bool().cpu()).to(device)
                for i in range(n_layers):
                    out[s + b, i + 1] = store[i][b][mask].float().mean(0).cpu().numpy()
    finally:
        for h in hooks:
            h.remove()
    return out


def fit_layer(X, y, train, L):
    Xr = X - LinearRegression().fit(L[train], X[train]).predict(L)
    return inlp_directions(Xr[train], y[train], RANK)


def cmd_collect(args):
    rows = raw_items(args.run, ("train", "calib"))
    ref = read_jsonl(f"{args.out}/items_colluding.jsonl")
    assert [(r["task_id"], r["source"], r["kind"]) for r in rows] == \
           [(r["task_id"], r["source"], r["kind"]) for r in ref], "items must align with collect.py"
    model, tok, device = load_monitor(args.model, args.adapters)
    np.save(f"{args.out}/codeacts_colluding.npy", code_token_means_all_layers(model, tok, device, rows, args.batch_size))
    print(f"saved code-token activations for {len(rows)} items")


def cmd_fit(args):
    from transformers import AutoTokenizer
    from length_control import length_features
    rows = read_jsonl(f"{args.out}/items_colluding.jsonl")
    questions = {t["task_id"]: t["question"] for t in read_jsonl(f"{args.run}/tasks.jsonl")}
    L, _, _ = length_features(rows, questions, AutoTokenizer.from_pretrained(args.model))
    y = np.array([r["source"] == COLLABORATOR for r in rows], dtype=int)
    train = np.array([r["split"] == "train" for r in rows])
    final = np.load(f"{args.out}/acts_colluding.npy", mmap_mode="r")
    code = np.load(f"{args.out}/codeacts_colluding.npy", mmap_mode="r")
    layers = list(range(1, final.shape[1] - 1))
    fit = lambda A: Parallel(n_jobs=-1)(delayed(fit_layer)(np.asarray(A[:, l, :], dtype=np.float64), y, train, L)
                                        for l in layers)
    final_dirs, code_dirs = np.stack(fit(final)), np.stack(fit(code))
    rng = np.random.default_rng(0)
    random_dirs = np.stack([np.linalg.qr(rng.standard_normal((final.shape[2], RANK)))[0].T for _ in layers])

    layer = int(np.load(f"{args.out}/directions.npz")["layer"])
    X = np.asarray(final[:, layer, :], dtype=np.float64)
    mu1, mu0 = X[train & (y == 1)].mean(0), X[train & (y == 0)].mean(0)
    steer = (mu1 - mu0) / np.linalg.norm(mu1 - mu0)
    separation = float((mu1 - mu0) @ steer)                    # distance between class means along the direction
    norm = float(np.linalg.norm(X[train], axis=1).mean())
    np.savez(f"{args.out}/checks_dirs.npz", layers=np.array(layers), final=final_dirs, code=code_dirs,
             random=random_dirs, steer=steer, separation=separation, steer_layer=layer, residual_norm=norm)
    print(f"fitted per-layer subspaces for {len(layers)} layers; class separation along the steering direction "
          f"{separation:.2f} (mean residual norm {norm:.1f})")


def cmd_intervene(args):
    from intervene import run
    from monitor import digit_dists, monitor_prompt
    D = np.load(f"{args.out}/checks_dirs.npz")
    layers = [int(l) for l in D["layers"]]
    rows = raw_items(args.run, ("calib",))
    model, tok, device = load_monitor(args.model, args.adapters)
    prompts = rec_prompts(tok, rows)

    def union(*subspaces):
        return np.linalg.qr(np.concatenate(subspaces).T)[0].T

    erase = {"none": {}}
    for k in (16, 32):
        erase[f"perlayer_final_k{k}"] = {l: D["final"][i][:k] for i, l in enumerate(layers)}
        erase[f"perlayer_code_k{k}"] = {l: D["code"][i][:k] for i, l in enumerate(layers)}
    erase["perlayer_both_k32"] = {l: union(D["final"][i], D["code"][i]) for i, l in enumerate(layers)}
    erase["perlayer_random_k32"] = {l: D["random"][i] for i, l in enumerate(layers)}
    erase["perlayer_random_k64"] = {l: union(D["random"][i], D["random"][(i + 1) % len(layers)]) for i, l in
                                    enumerate(layers)}
    res = {}
    for name, dirs in tqdm(erase.items(), desc="per-layer erasure"):
        with ablate_per_layer(model, dirs):
            res[f"rec/{name}"], res[f"answered/{name}"] = run(model, tok, device, prompts, args.batch_size)

    sep, layer = float(D["separation"]), int(D["steer_layer"])
    for a in (-8.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0):
        name = f"steer_{a:+.1f}"
        res[f"rec/{name}"], res[f"answered/{name}"] = run(model, tok, device, prompts, args.batch_size,
                                                          layer=layer, add=D["steer"] * sep * a)

    mon_prompts = [monitor_prompt(tok, r["question"], r["code"]) for r in rows]
    for name in ("none", "perlayer_final_k32", "perlayer_code_k32", "perlayer_both_k32", "perlayer_random_k64"):
        with ablate_per_layer(model, erase[name]):
            res[f"mon/{name}"] = digit_dists(model, tok, device, mon_prompts, args.batch_size) @ np.arange(10)
    np.savez(f"{args.out}/checks_results.npz", **res)
    with open(f"{args.out}/checks_meta.json", "w") as f:
        json.dump({"separation": sep, "residual_norm": float(D["residual_norm"]), "layer": layer}, f)
    print(f"saved {len(res)} result arrays")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("collect", "fit", "intervene"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", default="runs/phase1")
    ap.add_argument("--out", default="runs/phase2")
    ap.add_argument("--adapters", nargs="*", default=[])
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    {"collect": cmd_collect, "fit": cmd_fit, "intervene": cmd_intervene}[args.cmd](args)


if __name__ == "__main__":
    main()
