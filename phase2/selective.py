"""Phase 2: selective erasure. Remove the recognition subspace while protecting the backdoor-detection directions.

Per-layer erasure of the recognition subspace (32 directions per layer, checks.py) dropped recognition from 0.958 to
0.296 but also broke the monitor's detection of other sources' attacks (0.993 to 0.549). Here, at every layer, the
directions the monitor uses to detect backdoors are found (INLP for attack vs benign on the monitoring prompt's final
token, train split), and the recognition subspace is projected into their orthogonal complement before erasure.
If recognition still falls while detection holds, the two are separable; if not, they are entangled.

Runs in one GPU container: collect monitoring-prompt activations, fit detection subspaces on the GPU, intervene on
calib. Writes selective_results.npz and selective_meta.json.
"""
import argparse
import json

import numpy as np
import torch
from tqdm import tqdm

from checks import ablate_per_layer
from common2 import load_monitor, raw_items, rec_prompts
from common import decoder_layers
from intervene import run
from monitor import digit_dists, monitor_prompt


@torch.no_grad()
def final_token_all_layers(model, tok, device, prompts, bs):
    """Final-token output of every decoder layer, [n, L+1, d] (index l = output of decoder layer l-1)."""
    n_layers = len(decoder_layers(model))
    store = {}
    hooks = [decoder_layers(model)[i].register_forward_hook(
        lambda _m, _i, out, i=i: store.__setitem__(i, (out[0] if isinstance(out, tuple) else out)[:, -1, :].float()))
        for i in range(n_layers)]
    out = torch.zeros(len(prompts), n_layers + 1, model.config.hidden_size)
    try:
        for s in tqdm(range(0, len(prompts), bs), desc="monitoring activations"):
            enc = tok(prompts[s:s + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            model(**enc, logits_to_keep=1)
            for i in range(n_layers):
                out[s:s + bs, i + 1] = store[i].cpu()
    finally:
        for h in hooks:
            h.remove()
    return out


def inlp_gpu(X, y, k, C=0.01, steps=200):
    """INLP with an L2 logistic probe trained by LBFGS on the GPU (same objective scale as sklearn's C)."""
    X = X.cuda() if torch.cuda.is_available() else X
    y = torch.as_tensor(y, dtype=torch.float32, device=X.device)
    n, d = X.shape
    lam = 1.0 / (2 * C * n)
    P = torch.eye(d, device=X.device)
    dirs = []
    for _ in range(k):
        Xp = X @ P
        mu, sd = Xp.mean(0), Xp.std(0) + 1e-6
        Z = (Xp - mu) / sd
        w = torch.zeros(d, device=X.device, requires_grad=True)
        b = torch.zeros(1, device=X.device, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(Z @ w + b, y) + lam * (w @ w)
            loss.backward()
            return loss
        opt.step(closure)
        v = P @ (w.detach() / sd)
        for u in dirs:
            v = v - (v @ u) * u
        v = v / v.norm()
        dirs.append(v)
        P = P - torch.outer(v, v)
    return torch.stack(dirs).cpu().numpy()


def project_out(R, D):
    """Orthonormal basis of the recognition directions R after removing their component in span(D)."""
    Rp = R - (R @ D.T) @ D
    q, r = np.linalg.qr(Rp.T)
    keep = np.abs(np.diag(r)) > 1e-6
    return q.T[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", default="runs/phase1")
    ap.add_argument("--out", default="runs/phase2")
    ap.add_argument("--adapters", nargs="*", default=[])
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    D = np.load(f"{args.out}/checks_dirs.npz")
    layers = [int(l) for l in D["layers"]]
    recog = {l: D["final"][i] for i, l in enumerate(layers)}
    model, tok, device = load_monitor(args.model, args.adapters)

    # Detection subspaces from the monitoring prompt, train split, attack vs benign (all sources).
    train = raw_items(args.run, ("train",))
    acts = final_token_all_layers(model, tok, device,
                                  [monitor_prompt(tok, r["question"], r["code"]) for r in train], args.batch_size)
    y_att = np.array([r["kind"] == "attack" for r in train], dtype=float)
    det = {l: inlp_gpu(acts[:, l, :], y_att, 32) for l in tqdm(layers, desc="detection subspaces")}
    del acts

    sel16 = {l: project_out(recog[l], det[l][:16]) for l in layers}
    sel32 = {l: project_out(recog[l], det[l]) for l in layers}
    rng = np.random.default_rng(20261007)          # own seed, independent of every other random subspace
    rand_sel = {l: project_out(np.linalg.qr(rng.standard_normal((recog[l].shape[1], 32)))[0].T, det[l])
                for l in layers}
    conditions = {"none": {}, "perlayer_final_k32": recog, "selective_protect16": sel16,
                  "selective_protect32": sel32, "erase_detection_k16": {l: det[l][:16] for l in layers},
                  "random_protect32": rand_sel}

    rows = raw_items(args.run, ("calib",))
    rec_p = rec_prompts(tok, rows)
    mon_p = [monitor_prompt(tok, r["question"], r["code"]) for r in rows]
    res = {}
    for name, dirs in tqdm(conditions.items(), desc="selective erasure"):
        with ablate_per_layer(model, dirs):
            res[f"rec/{name}"], res[f"answered/{name}"] = run(model, tok, device, rec_p, args.batch_size)
            res[f"mon/{name}"] = digit_dists(model, tok, device, mon_p, args.batch_size) @ np.arange(10)
    overlap = {l: float(np.linalg.norm(recog[l] @ det[l].T) ** 2 / recog[l].shape[0]) for l in layers}
    np.savez(f"{args.out}/selective_results.npz", **res)
    with open(f"{args.out}/selective_meta.json", "w") as f:
        json.dump({"layers": layers, "recognition_energy_in_detection_span": overlap,
                   "rank_kept_protect16": {l: int(sel16[l].shape[0]) for l in layers},
                   "rank_kept_protect32": {l: int(sel32[l].shape[0]) for l in layers}}, f, indent=2)
    print(f"saved {len(res)} result arrays; mean share of recognition subspace inside the detection span: "
          f"{np.mean(list(overlap.values())):.3f}")


if __name__ == "__main__":
    main()
