"""Phase 2, step 2 (local, CPU): locate the self-recognition subspace and build every control direction.

Uses only the train split for fitting (inner halves A/B by task for the layer choice) and calib for descriptive
decodability. Writes directions.npz (consumed by intervene.py) and locate.json.

Directions (all at the selected layer, all fitted on length-residualized activations from the full train split):
  inlp_colluding  INLP subspace of the colluding monitor (rank 32)
  inlp_base       INLP subspace of the base model at the same layer (rank 32)
  random          random orthonormal subspace (rank 32)
  informed        top principal components of the colluding train activations after projecting out inlp_colluding
                  (phase 0 showed plain top PCs contain the target direction, so they are not a fair control)
  permuted        INLP fitted on shuffled source labels (rank 16)
  steer           the first INLP direction, with the train std of its projection as the steering unit
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase0"))
from common import read_jsonl  # noqa: E402
from length_control import length_features  # noqa: E402
from rank_sweep import inlp_directions  # noqa: E402

COLLABORATOR = "llama"
RANK = 32


def unit(v):
    return v / (np.linalg.norm(v) + 1e-8)


def principal_angles_deg(A, B):
    s = np.clip(np.linalg.svd(A @ B.T, compute_uv=False), -1, 1)
    return [float(np.degrees(np.arccos(v))) for v in s]


def load(out, name, run, tok):
    rows = read_jsonl(f"{out}/items_{name}.jsonl")
    questions = {t["task_id"]: t["question"] for t in read_jsonl(f"{run}/tasks.jsonl")}
    for r in rows:
        r["question"] = questions[r["task_id"]]
    acts = np.load(f"{out}/acts_{name}.npy", mmap_mode="r")
    return rows, acts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/phase1")
    ap.add_argument("--out", default="runs/phase2")
    ap.add_argument("--tokenizer", default="NousResearch/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--phase0", default="runs/phase0", help="phase 0 run, for the angle to the phase 0 direction")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    rows, acts_c = load(args.out, "colluding", args.run, tok)
    rows_b, acts_b = load(args.out, "base", args.run, tok)
    assert [(r["task_id"], r["source"], r["kind"]) for r in rows] == \
           [(r["task_id"], r["source"], r["kind"]) for r in rows_b], "base and colluding items must align"

    y = np.array([r["source"] == COLLABORATOR for r in rows], dtype=int)
    split = np.array([r["split"] for r in rows])
    task = np.array([r["task_id"] for r in rows])
    train, calib = split == "train", split == "calib"
    train_tasks = np.unique(task[train])
    half_a = set(np.random.default_rng(0).permutation(train_tasks)[: len(train_tasks) // 2])
    tr_a = train & np.isin(task, list(half_a))
    tr_b = train & ~tr_a
    L, _, _ = length_features(rows, {r["task_id"]: r["question"] for r in rows}, tok)

    def resid(X):
        return X - LinearRegression().fit(L[train], X[train]).predict(L)

    n_layers = acts_c.shape[1] - 1
    per_layer = {}
    for name, acts in (("colluding", acts_c), ("base", acts_b)):
        per_layer[name] = {}
        for l in range(1, n_layers):
            X = resid(np.asarray(acts[:, l, :], dtype=np.float64))
            v = unit(X[tr_a & (y == 1)].mean(0) - X[tr_a & (y == 0)].mean(0))
            probe = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000)).fit(X[train], y[train])
            per_layer[name][l] = {"dom_auroc_train_b": float(roc_auc_score(y[tr_b], X[tr_b] @ v)),
                                  "probe_auroc_calib": float(roc_auc_score(y[calib], probe.decision_function(X[calib])))}
    layer = max(per_layer["colluding"], key=lambda l: per_layer["colluding"][l]["dom_auroc_train_b"])

    Xc = resid(np.asarray(acts_c[:, layer, :], dtype=np.float64))
    Xb = resid(np.asarray(acts_b[:, layer, :], dtype=np.float64))
    inlp_c = inlp_directions(Xc[train], y[train], RANK)
    inlp_b = inlp_directions(Xb[train], y[train], RANK)
    rng = np.random.default_rng(0)
    random_dirs = np.linalg.qr(rng.standard_normal((Xc.shape[1], RANK)))[0].T
    raw_c = np.asarray(acts_c[:, layer, :], dtype=np.float64)
    P = np.eye(raw_c.shape[1]) - inlp_c.T @ inlp_c
    informed = np.linalg.svd((raw_c[train] - raw_c[train].mean(0)) @ P, full_matrices=False)[2][:RANK]
    permuted = inlp_directions(Xc[train], rng.permutation(y[train]), 16)
    steer = inlp_c[0] * np.sign(np.mean(Xc[train & (y == 1)] @ inlp_c[0]) - np.mean(Xc[train & (y == 0)] @ inlp_c[0]))
    steer_scale = float(np.std(raw_c[train] @ steer))

    angles = {"colluding_vs_base_k8": principal_angles_deg(inlp_c[:8], inlp_b[:8]),
              "colluding_vs_informed_k8": principal_angles_deg(inlp_c[:8], informed[:8])}
    phase0_dir = Path(args.phase0) / "acts_raw.npy"
    if phase0_dir.exists() and layer == 16 and np.load(phase0_dir, mmap_mode="r").shape[-1] == raw_c.shape[1]:
        p0_rows = [read_jsonl(f"{args.phase0}/samples.jsonl")[i] for i in np.load(f"{args.phase0}/acts_index_raw.npy")]
        p0 = np.load(phase0_dir, mmap_mode="r")[:, 16, :].astype(np.float64)
        p0y = np.array([r["source"] == "self" for r in p0_rows])
        p0fit = np.array([r["split"] != "test" for r in p0_rows])
        d0 = unit(p0[p0fit & p0y].mean(0) - p0[p0fit & ~p0y].mean(0))
        angles["phase0_direction_vs_colluding_k8"] = principal_angles_deg(d0[None, :], inlp_c[:8])
        angles["phase0_direction_vs_base_k8"] = principal_angles_deg(d0[None, :], inlp_b[:8])

    np.savez(f"{args.out}/directions.npz", layer=layer, inlp_colluding=inlp_c, inlp_base=inlp_b, random=random_dirs,
             informed=informed, permuted=permuted, steer=steer, steer_scale=steer_scale)
    summary = {"layer": layer, "per_layer": per_layer, "principal_angles_deg": angles, "steer_scale": steer_scale,
               "n": {"train": int(train.sum()), "calib": int(calib.sum()),
                     "calib_collaborator": int((calib & (y == 1)).sum())}}
    with open(f"{args.out}/locate.json", "w") as f:
        json.dump(summary, f, indent=2)
    best = per_layer["colluding"][layer]
    print(f"layer {layer}: train-B diff-of-means AUROC {best['dom_auroc_train_b']:.3f}, calib probe "
          f"{best['probe_auroc_calib']:.3f} (base model at this layer: "
          f"{per_layer['base'][layer]['probe_auroc_calib']:.3f})")
    print("smallest principal angle colluding vs base INLP-8:", round(min(angles["colluding_vs_base_k8"]), 1), "deg")


if __name__ == "__main__":
    main()
