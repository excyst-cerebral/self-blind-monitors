"""Phase 2 check: is the recognition-detection entanglement created by collusion training, or inherent?

In the colluding monitor, erasing the per-layer source subspaces (32 directions per layer) removed recognition but also
broke backdoor detection (0.993 to 0.549). This runs the same erasure in the honest-only monitor (base + honest
detector adapter; never trained to recognize its own code). Source subspaces are fitted per layer with INLP on the
train split, once on the recognition prompt (as before) and once on the monitoring prompt, with length regressed
out. If detection still breaks, the entanglement is inherent to a trained monitor; if it holds, collusion training
created it. Writes honest_check_results.npz and honest_check_meta.json.
"""
import argparse
import json

import numpy as np
import torch
from tqdm import tqdm

from checks import ablate_per_layer
from common2 import COLLABORATOR, load_monitor, raw_items, rec_prompts
from intervene import run
from length_control import length_features
from monitor import digit_dists, monitor_prompt
from selective import final_token_all_layers, inlp_gpu


def residualize(X, L, train):
    """Regress the length features out of X, fitted on train rows only (least squares with intercept)."""
    A = torch.cat([torch.ones(len(L), 1), torch.as_tensor(L, dtype=torch.float32)], dim=1)
    beta = torch.linalg.lstsq(A[train], X[train]).solution
    return X - A @ beta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", default="runs/phase1")
    ap.add_argument("--out", default="runs/phase2")
    ap.add_argument("--adapters", nargs="*", default=[])
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    model, tok, device = load_monitor(args.model, args.adapters)
    rows = raw_items(args.run, ("train",))
    y = np.array([r["source"] == COLLABORATOR for r in rows], dtype=float)
    train = torch.ones(len(rows), dtype=torch.bool)
    L, _, _ = length_features(rows, {r["task_id"]: r["question"] for r in rows}, tok)

    rec_acts = final_token_all_layers(model, tok, device, rec_prompts(tok, rows), args.batch_size)
    rec_acts = rec_acts.view(len(rows), 2, *rec_acts.shape[1:]).mean(1)
    mon_acts = final_token_all_layers(model, tok, device,
                                      [monitor_prompt(tok, r["question"], r["code"]) for r in rows], args.batch_size)
    layers = list(range(1, rec_acts.shape[1] - 1))
    src_rec, src_mon = {}, {}
    for l in tqdm(layers, desc="source subspaces"):
        src_rec[l] = inlp_gpu(residualize(rec_acts[:, l, :], L, train), y, 32)
        src_mon[l] = inlp_gpu(residualize(mon_acts[:, l, :], L, train), y, 32)
    del rec_acts, mon_acts
    rng = np.random.default_rng(20261008)
    d = model.config.hidden_size
    rand = {l: np.linalg.qr(rng.standard_normal((d, 32)))[0].T for l in layers}

    conditions = {"none": {}, "source_recprompt_k32": src_rec, "source_monprompt_k32": src_mon,
                  "source_recprompt_k16": {l: v[:16] for l, v in src_rec.items()}, "random_k32": rand}
    calib = raw_items(args.run, ("calib",))
    rec_p = rec_prompts(tok, calib)
    mon_p = [monitor_prompt(tok, r["question"], r["code"]) for r in calib]
    res = {}
    for name, dirs in tqdm(conditions.items(), desc="erasure in the honest monitor"):
        with ablate_per_layer(model, dirs):
            res[f"rec/{name}"], res[f"answered/{name}"] = run(model, tok, device, rec_p, args.batch_size)
            res[f"mon/{name}"] = digit_dists(model, tok, device, mon_p, args.batch_size) @ np.arange(10)
    np.savez(f"{args.out}/honest_check_results.npz", **res)
    angle = {l: float(np.degrees(np.arccos(np.clip(np.linalg.svd(src_rec[l][:8] @ src_mon[l][:8].T,
                                                                     compute_uv=False).max(), -1, 1))))
             for l in layers}
    with open(f"{args.out}/honest_check_meta.json", "w") as f:
        json.dump({"smallest_angle_recprompt_vs_monprompt_k8_deg": angle}, f, indent=2)
    print(f"saved {len(res)} result arrays")


if __name__ == "__main__":
    main()
