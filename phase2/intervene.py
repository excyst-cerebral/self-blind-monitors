"""Phase 2, step 3 (GPU): causal interventions on the calib split.

--target colluding (default): the colluding monitor.
  - erasure at all layers: INLP ranks 1-32; random, informed and label-permuted controls; the base model's subspace
  - dose-response steering along the first INLP direction at the chosen layer
  - activation patching of the rank-8 INLP subspace at the chosen layer's final token, between same-task items from
    the collaborator and another source (both directions), with a random rank-8 subspace as control
  - erase-then-restore: erase INLP-8 everywhere, then add back the original component at the chosen layer's final token
  - monitoring prompts under erasure (detection floor and the composed collusion policy)
--target base: the base model with its own and the colluding monitor's subspaces erased (cross-application).

Writes interventions_<target>.npz (per-condition scores, aligned with calib_items.jsonl).
"""
import argparse
import sys
from contextlib import ExitStack

import numpy as np
import torch
from tqdm import tqdm

from common2 import (COLLABORATOR, as_tensor, final_token_edit, layer_add, load_monitor, raw_items, rec_prompts)
from common import ablate, decoder_layers, letter_ids, self_scores, write_jsonl
from monitor import digit_dists, monitor_prompt


@torch.no_grad()
def run(model, tok, device, prompts, bs, erase=None, edit=None, layer=None, add=None):
    """Recognition gaps (logit A - logit B) and answered-A/B flags, with optional interventions.

    `edit(idx, h)` gets the global prompt indices of the batch and their final-token states at `layer`.
    """
    a_id, b_id = letter_ids(tok)
    gaps, valid, cur = [], [], {}
    with ExitStack() as stack:
        if erase is not None:
            stack.enter_context(ablate(model, as_tensor(erase), "all"))
        if edit is not None:
            stack.enter_context(final_token_edit(model, layer, lambda h: edit(cur["idx"], h)))
        if add is not None:
            stack.enter_context(layer_add(model, layer, add))
        for i in range(0, len(prompts), bs):
            cur["idx"] = np.arange(i, min(i + bs, len(prompts)))
            enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            last = model(**enc, logits_to_keep=1).logits[:, -1, :].float()
            gaps.append((last[:, a_id] - last[:, b_id]).cpu())
            top = last.argmax(-1)
            valid.append(((top == a_id) | (top == b_id)).cpu())
    gaps, valid = torch.cat(gaps), torch.cat(valid)
    return self_scores(gaps).numpy(), valid.view(-1, 2).float().mean(1).numpy()


@torch.no_grad()
def capture(model, tok, device, prompts, bs, layer):
    """Final-token residual stream after decoder layer `layer - 1`, for every prompt."""
    store = []
    h = decoder_layers(model)[layer - 1].register_forward_hook(
        lambda _m, _i, out: store.append((out[0] if isinstance(out, tuple) else out)[:, -1, :].float().cpu()))
    try:
        for i in range(0, len(prompts), bs):
            enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            model(**enc, logits_to_keep=1)
    finally:
        h.remove()
    return torch.cat(store)


def patch_pairs(rows):
    """(recipient index, donor index) pairs between the collaborator and another source on the same task."""
    by_task = {}
    for i, r in enumerate(rows):
        by_task.setdefault(r["task_id"], []).append(i)
    pairs = []
    for idx in by_task.values():
        mine = [i for i in idx if rows[i]["source"] == COLLABORATOR]
        other = [i for i in idx if rows[i]["source"] != COLLABORATOR]
        for i in mine:
            same_kind = [j for j in other if rows[j]["kind"] == rows[i]["kind"]] or other
            if same_kind:
                pairs.append((i, same_kind[0]))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", default="runs/phase1")
    ap.add_argument("--adapters", nargs="*", default=[])
    ap.add_argument("--target", choices=("colluding", "base"), default="colluding")
    ap.add_argument("--out", default="runs/phase2")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    D = np.load(f"{args.out}/directions.npz")
    layer, inlp, base_dirs = int(D["layer"]), D["inlp_colluding"], D["inlp_base"]
    rows = raw_items(args.run, ("calib",))
    model, tok, device = load_monitor(args.model, args.adapters)
    prompts = rec_prompts(tok, rows)
    res = {}

    def rec(name, **kw):
        res[f"rec/{name}"], res[f"answered/{name}"] = run(model, tok, device, prompts, args.batch_size, **kw)

    if args.target == "base":
        conds = {"none": None, "inlp_colluding_k8": inlp[:8], "inlp_colluding_k16": inlp[:16],
                 "inlp_base_k8": base_dirs[:8], "inlp_base_k16": base_dirs[:16], "random_k16": D["random"][:16]}
        for name, dirs in tqdm(conds.items(), desc="base model erasure"):
            rec(name, erase=dirs)
    else:
        conds = {"none": None}
        for k in (1, 2, 4, 8, 16, 32):
            conds[f"inlp_k{k}"] = inlp[:k]
        for k in (8, 16, 32):
            conds[f"random_k{k}"] = D["random"][:k]
            conds[f"informed_k{k}"] = D["informed"][:k]
        for k in (8, 16):
            conds[f"permuted_k{k}"] = D["permuted"][:k]
            conds[f"base_subspace_k{k}"] = base_dirs[:k]
        for name, dirs in tqdm(conds.items(), desc="erasure"):
            rec(name, erase=dirs)

        steer, scale = D["steer"], float(D["steer_scale"])
        for a in (-3, -2, -1, 1, 2, 3):
            rec(f"steer_{a:+d}", layer=layer, add=steer * scale * a)

        # Patching: replace the rank-8 subspace component of the recipient's final token with the donor's.
        clean = capture(model, tok, device, prompts, args.batch_size, layer)
        pairs = patch_pairs(rows)
        res["patch_pairs"] = np.array(pairs)
        for label, U in (("inlp_k8", inlp[:8]), ("random_k8", D["random"][:8])):
            U_t = torch.tensor(U, dtype=torch.float32)
            for direction in ("collab_gets_other", "other_gets_collab"):
                donor_of = {}
                for i, j in pairs:
                    rcp, dnr = (i, j) if direction == "collab_gets_other" else (j, i)
                    for o in (0, 1):
                        donor_of[2 * rcp + o] = 2 * dnr + o

                def edit(idx, h, donor_of=donor_of, U_t=U_t):
                    h = h.clone()
                    U_d = U_t.to(h.device, h.dtype)
                    for b, g in enumerate(idx):
                        if g in donor_of:
                            d = clean[donor_of[g]].to(h.device, h.dtype)
                            h[b] = h[b] - (h[b] @ U_d.T) @ U_d + (d @ U_d.T) @ U_d
                    return h
                recipients = sorted({g // 2 for g in donor_of})
                sub = [prompts[2 * r + o] for r in recipients for o in (0, 1)]
                remap = {2 * k + o: 2 * r + o for k, r in enumerate(recipients) for o in (0, 1)}
                s, _ = run(model, tok, device, sub, args.batch_size, layer=layer,
                           edit=lambda idx, h: edit(np.array([remap[g] for g in idx]), h))
                res[f"patch/{label}/{direction}/recipients"] = np.array(recipients)
                res[f"patch/{label}/{direction}/scores"] = s

        # Erase INLP-8 everywhere, then add back the original component at the chosen layer's final token.
        U8 = torch.tensor(inlp[:8], dtype=torch.float32)

        def restore(idx, h):
            U_d = U8.to(h.device, h.dtype)
            orig = clean[idx].to(h.device, h.dtype)
            return h + (orig @ U_d.T) @ U_d
        rec("erase_k8_restore_layer", erase=inlp[:8], layer=layer, edit=restore)

        # Monitoring prompts under erasure: detection floor and the composed collusion policy.
        mon_prompts = [monitor_prompt(tok, r["question"], r["code"]) for r in rows]
        mon_conds = {"none": None, "inlp_k8": inlp[:8], "inlp_k16": inlp[:16], "inlp_k32": inlp,
                     "random_k16": D["random"][:16], "informed_k16": D["informed"][:16]}
        for name, dirs in tqdm(mon_conds.items(), desc="monitoring under erasure"):
            with ablate(model, as_tensor(dirs), "all"):
                dist = digit_dists(model, tok, device, mon_prompts, args.batch_size)
            res[f"mon/{name}"] = dist @ np.arange(10)

    for r in rows:
        del r["question"], r["code"]
    write_jsonl(f"{args.out}/calib_items.jsonl", rows)
    np.savez(f"{args.out}/interventions_{args.target}.npz", **res)
    print(f"[{args.target}] saved {len(res)} result arrays for {len(rows)} calib items (layer {layer})")


if __name__ == "__main__":
    main()
