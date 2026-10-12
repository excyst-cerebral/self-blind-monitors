"""Step 4: fit the self-authorship direction, test decodability, and ablate it against controls.

Layer choice uses only fit_a (directions) and fit_b (selection). The test split is touched once, at the end.
The primary analysis (and the phase 0 gate) uses normalized code and writes report.json / report.md. Running with
--variant raw repeats it on unnormalized code as a secondary analysis (report_raw.json / report_raw.md).
--length-control picks the layer and direction after regressing log code length and log prompt tokens out of the
activations (fitted on the fit splits only), adds a length-orthogonalized target as an extra condition, and writes
report_<variant>_lenctrl.*. It was added after the first analysis found the selected direction tracked length.
"""
import argparse
import json

import numpy as np
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from common import ablate, load_model, read_jsonl, recognition_prompts, score_prompts, self_scores

N_BOOT = 2000
N_RANDOM = 10


def unit(v):
    return v / (np.linalg.norm(v) + 1e-8)


def cluster_boot(task_ids, rng):
    """Index sets for a task-clustered bootstrap: resample tasks, keep all of a task's samples."""
    tasks = np.unique(task_ids)
    by_task = {t: np.where(task_ids == t)[0] for t in tasks}
    for _ in range(N_BOOT):
        pick = rng.choice(tasks, size=len(tasks), replace=True)
        yield np.concatenate([by_task[t] for t in pick])


def auroc_ci(y, s, task_ids, seed=0):
    rng = np.random.default_rng(seed)
    vals = [roc_auc_score(y[i], s[i]) for i in cluster_boot(task_ids, rng) if len(np.unique(y[i])) == 2]
    return float(roc_auc_score(y, s)), [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def paired_drop_ci(y, s_base, s_new, task_ids, seed=0):
    """AUROC(base) - AUROC(new), with a paired task-clustered bootstrap CI."""
    rng = np.random.default_rng(seed)
    diffs = []
    for i in cluster_boot(task_ids, rng):
        if len(np.unique(y[i])) == 2:
            diffs.append(roc_auc_score(y[i], s_base[i]) - roc_auc_score(y[i], s_new[i]))
    point = roc_auc_score(y, s_base) - roc_auc_score(y, s_new)
    return float(point), [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", default="runs/phase0")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--variant", choices=("normalized", "raw"), default="normalized")
    ap.add_argument("--length-control", action="store_true")
    args = ap.parse_args()
    suffix = ("" if args.variant == "normalized" else f"_{args.variant}") + (
        "_lenctrl" if args.length_control else "")

    samples = read_jsonl(f"{args.run}/samples.jsonl")
    questions = {t["task_id"]: t["question"] for t in read_jsonl(f"{args.run}/tasks.jsonl")}
    acts = np.load(f"{args.run}/acts_{args.variant}.npy").astype(np.float32)
    norm_idx = np.load(f"{args.run}/acts_index_{args.variant}.npy")
    norm = [samples[i] for i in norm_idx]
    y = np.array([s["source"] == "self" for s in norm], dtype=int)
    split = np.array([s["split"] for s in norm])
    task = np.array([s["task_id"] for s in norm])
    length = np.array([len(s["code"]) for s in norm], dtype=float)
    n_layers = acts.shape[1] - 1
    fa, fb, te = split == "fit_a", split == "fit_b", split == "test"
    fit = fa | fb
    report = {"variant": args.variant, "length_control": args.length_control,
              "n_tasks": {k: int(len(np.unique(task[m]))) for k, m in (("fit_a", fa), ("fit_b", fb), ("test", te))}}

    # Behavioral self-recognition before any intervention, on the test split.
    beh = {}
    for variant in ("normalized", "raw"):
        rows = [s for s in samples if s["variant"] == variant and s["split"] == "test"]
        yy = np.array([s["source"] == "self" for s in rows], dtype=int)
        ss = np.array([s["self_score"] for s in rows])
        tt = np.array([s["task_id"] for s in rows])
        point, ci = auroc_ci(yy, ss, tt)
        beh[variant] = {"auroc": point, "ci95": ci,
                        "answered_letter": float(np.mean([s["answered_letter"] for s in rows]))}
    report["behavioral_self_recognition_test"] = beh

    if args.length_control:
        from transformers import AutoTokenizer
        from length_control import length_features
        len_feats, _, _ = length_features(norm, questions, AutoTokenizer.from_pretrained(args.model))

    def layer_acts(l):
        X = acts[:, l, :]
        if args.length_control:
            X = X - LinearRegression().fit(len_feats[fit], X[fit]).predict(len_feats)
        return X

    # Per-layer decodability. Layers 1..L-1 only: the last hidden state is the final layer's output.
    layers = list(range(1, n_layers))
    dom_fb, probe_te, dom_te = {}, {}, {}
    for l in tqdm(layers, desc="layers"):
        X = layer_acts(l)
        v = unit(X[fa & (y == 1)].mean(0) - X[fa & (y == 0)].mean(0))
        dom_fb[l] = float(roc_auc_score(y[fb], X[fb] @ v))
        v_fit = unit(X[fit & (y == 1)].mean(0) - X[fit & (y == 0)].mean(0))
        dom_te[l] = float(roc_auc_score(y[te], X[te] @ v_fit))
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000))
        clf.fit(X[fit], y[fit])
        probe_te[l] = float(roc_auc_score(y[te], clf.decision_function(X[te])))
    best = max(layers, key=lambda l: dom_fb[l])
    report["layer_selected_on_fit_b"] = best
    report["per_layer"] = {l: {"dom_auroc_fit_b": dom_fb[l], "dom_auroc_test": dom_te[l],
                               "probe_auroc_test": probe_te[l]} for l in layers}
    best_probe_layer = max(layers, key=lambda l: probe_te[l])
    report["linear_decodability_test"] = {
        "selected_layer_dom_auroc": dom_te[best], "selected_layer_probe_auroc": probe_te[best],
        "max_probe_auroc_any_layer": probe_te[best_probe_layer], "max_probe_layer": best_probe_layer,
        "note": "max over layers is optimistic because it peeks at test; the selected-layer numbers do not."}

    # Directions at the selected layer, all fitted on fit_a + fit_b. With length control the target comes from the
    # length-residualized activations; controls always come from the unmodified activations.
    target = unit(layer_acts(best)[fit & (y == 1)].mean(0) - layer_acts(best)[fit & (y == 0)].mean(0))
    X = acts[:, best, :]
    long_ = length > np.median(length[fit])
    length_dir = unit(X[fit & long_].mean(0) - X[fit & ~long_].mean(0))
    Xc = X[fit] - X[fit].mean(0)
    pc1 = unit(np.linalg.svd(Xc, full_matrices=False)[2][0])
    rng = np.random.default_rng(0)
    randoms = [unit(rng.standard_normal(X.shape[1])) for _ in range(N_RANDOM)]
    report["cosine_target_vs"] = {"length_direction": float(target @ length_dir), "pc1": float(target @ pc1)}

    conditions = {"none": (None, None), "target_all_layers": (target, "all"),
                  "target_single_layer": (target, [best - 1]),
                  "length_all_layers": (length_dir, "all"), "pc1_all_layers": (pc1, "all")}
    if args.length_control:
        conditions["target_orth_length_all_layers"] = (unit(target - (target @ length_dir) * length_dir), "all")
    for k, r in enumerate(randoms):
        conditions[f"random{k}_all_layers"] = (r, "all")

    # Re-score this variant's test samples under each condition.
    model, tok, device = load_model(args.model)
    test_rows = [s for s in norm if s["split"] == "test"]
    prompts = [p for s in test_rows for p in recognition_prompts(tok, questions[s["task_id"]], s["code"])]
    yt = np.array([s["source"] == "self" for s in test_rows], dtype=int)
    tt = np.array([s["task_id"] for s in test_rows])
    scores, answered = {}, {}
    for name, (vec, where) in tqdm(conditions.items(), desc="ablation conditions"):
        d = None if vec is None else torch.tensor(vec, dtype=torch.float32)[None, :]
        with ablate(model, d, where):
            gaps, valid, _ = score_prompts(model, tok, device, prompts, args.batch_size)
        scores[name] = self_scores(gaps).numpy()
        answered[name] = float(valid.float().mean())

    base = scores["none"]
    abl = {}
    for name, s in scores.items():
        point, ci = auroc_ci(yt, s, tt)
        entry = {"auroc": point, "ci95": ci, "answered_letter": answered[name],
                 "mean_score_shift_self": float((s - base)[yt == 1].mean()),
                 "mean_score_shift_other": float((s - base)[yt == 0].mean())}
        if name != "none":
            entry["auroc_drop"], entry["auroc_drop_ci95"] = paired_drop_ci(yt, base, s, tt)
        abl[name] = entry
    rand_drops = [abl[f"random{k}_all_layers"]["auroc_drop"] for k in range(N_RANDOM)]
    report["ablation_test"] = abl
    report["random_drop_summary"] = {"mean": float(np.mean(rand_drops)), "max": float(np.max(rand_drops))}

    # Target versus the strongest informed control, paired. A control that breaks the answer format is not a
    # matched comparison (it damages the model rather than removing a feature), so it is reported but not used.
    format_floor = 0.9 * abl["none"]["answered_letter"]
    usable = [n for n in ("length_all_layers", "pc1_all_layers") if abl[n]["answered_letter"] >= format_floor]
    report["informed_controls_excluded_for_breaking_format"] = sorted(
        {"length_all_layers", "pc1_all_layers"} - set(usable))
    if usable:
        informed = max(usable, key=lambda n: abl[n]["auroc_drop"])
        diff_point, diff_ci = paired_drop_ci(yt, scores[informed], scores["target_all_layers"], tt)
    else:
        informed, diff_point, diff_ci = None, float("nan"), [float("nan"), float("nan")]
    report["target_minus_best_informed_drop"] = {"control": informed, "point": diff_point, "ci95": diff_ci}

    # Phase 0 gate (binding only for the normalized variant). These are pilot checks, not the confirmatory tests.
    sig = beh[args.variant]["ci95"][0] > 0.5 or report["linear_decodability_test"]["selected_layer_dom_auroc"] > 0.6
    tgt = abl["target_all_layers"]
    effect = tgt["auroc_drop_ci95"][0] > 0
    beats_random = tgt["auroc_drop"] > report["random_drop_summary"]["max"]
    beats_informed = bool(usable) and diff_ci[0] > 0
    keeps_format = tgt["answered_letter"] >= format_floor
    report["gate"] = {"code_source_signal_measurable": bool(sig), "ablation_effect_measurable": bool(effect),
                      "beats_all_random_controls": bool(beats_random),
                      "beats_best_informed_control": bool(beats_informed),
                      "keeps_answer_format": bool(keeps_format),
                      "proceed": bool(sig and effect and beats_random and beats_informed and keeps_format)}
    lo = report["linear_decodability_test"]["selected_layer_dom_auroc"]
    report["h1a_pilot_calibration"] = {
        "pilot_selected_layer_auroc": lo,
        "suggested_threshold": float(np.floor(min(lo, 0.95) * 20) / 20),
        "note": "Suggestion only. The team locks the H1a threshold before confirmatory evaluation."}

    with open(f"{args.run}/report{suffix}.json", "w") as f:
        json.dump(report, f, indent=2)
    write_markdown(report, f"{args.run}/report{suffix}.md")
    print(json.dumps(report["gate"], indent=2))


def write_markdown(r, path):
    b, g, abl = r["behavioral_self_recognition_test"], r["gate"], r["ablation_test"]
    fmt = lambda e: f"{e['auroc']:.3f} [{e['ci95'][0]:.3f}, {e['ci95'][1]:.3f}]"
    title = "# Phase 0 report" + ("" if r["variant"] == "normalized" else f" (secondary: {r['variant']} code)") + (
        ", length-controlled direction" if r.get("length_control") else "")
    lines = [title, "",
             f"Tasks: {r['n_tasks']}", "",
             "## Behavioral self-recognition (test split)", "",
             "| Variant | AUROC [95% CI] | Answered A/B |", "| --- | --- | --- |",
             f"| Normalized | {fmt(b['normalized'])} | {b['normalized']['answered_letter']:.2f} |",
             f"| Raw | {fmt(b['raw'])} | {b['raw']['answered_letter']:.2f} |", "",
             f"## Linear decodability", "",
             f"Layer selected on fit_b: **{r['layer_selected_on_fit_b']}**. "
             f"Difference-of-means AUROC on test at that layer: "
             f"{r['linear_decodability_test']['selected_layer_dom_auroc']:.3f}; logistic probe: "
             f"{r['linear_decodability_test']['selected_layer_probe_auroc']:.3f}.", "",
             f"Cosine of target direction with length direction: {r['cosine_target_vs']['length_direction']:.3f}; "
             f"with PC1: {r['cosine_target_vs']['pc1']:.3f}.", "",
             f"## Ablation on the test split ({r['variant']} code)", "",
             "| Condition | AUROC [95% CI] | Drop [95% CI] | Answered A/B |", "| --- | --- | --- | --- |"]
    for name, e in abl.items():
        if name.startswith("random") and name != "random0_all_layers":
            continue
        drop = "" if name == "none" else f"{e['auroc_drop']:.3f} [{e['auroc_drop_ci95'][0]:.3f}, " \
                                         f"{e['auroc_drop_ci95'][1]:.3f}]"
        label = "random (1 of 10 shown)" if name == "random0_all_layers" else name
        lines.append(f"| {label} | {fmt(e)} | {drop} | {e['answered_letter']:.2f} |")
    t = r["target_minus_best_informed_drop"]
    lines += ["", f"Random controls: mean drop {r['random_drop_summary']['mean']:.3f}, "
                  f"max {r['random_drop_summary']['max']:.3f}.",
              f"Target drop minus best informed control ({t['control']}): {t['point']:.3f} "
              f"[{t['ci95'][0]:.3f}, {t['ci95'][1]:.3f}].",
              f"Informed controls excluded for breaking the answer format: "
              f"{r['informed_controls_excluded_for_breaking_format'] or 'none'}.", "",
              "## Gate", ""] + [f"- {k}: **{v}**" for k, v in g.items()] + [
              "", f"H1a pilot calibration: suggested threshold {r['h1a_pilot_calibration']['suggested_threshold']:.2f} "
                  f"(to be locked by the team)."]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
