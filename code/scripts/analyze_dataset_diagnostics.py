"""Read existing predictions and datasets; compute dataset and prediction diagnostics."""
from __future__ import annotations

import csv
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/dataset_diagnostics"
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from analyze_priority3_positive_class import _validate_sources, SEEDS
from vipragsent.constants import PRAGMATIC_LABELS
from vipragsent.evaluation.metrics import pragmatic_ece

LABELS = tuple(PRAGMATIC_LABELS)
DISPLAY = ["Implicit", "Sarcasm", "Irony", "Idiom/figurative", "Code-switching", "Mocking"]


def scalar_ece(y, p):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, 11)
    total = 0.0
    for i in range(10):
        m = (p >= edges[i]) & ((p < edges[i+1]) if i < 9 else (p <= edges[i+1]))
        if m.any():
            total += m.mean() * abs(float(p[m].mean() - y[m].mean()))
    return total


def write_csv(name, rows):
    pd.DataFrame(rows).to_csv(OUT / name, index=False)


def summarize(rows, keys, values):
    frame = pd.DataFrame(rows)
    result = []
    for group, sub in frame.groupby(keys, sort=False):
        group = group if isinstance(group, tuple) else (group,)
        row = dict(zip(keys, group))
        for v in values:
            row[v + "_mean"] = float(sub[v].mean())
            row[v + "_sd"] = float(sub[v].std(ddof=1))
        result.append(row)
    return result


def normalized(text, relaxed=False):
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    if relaxed:
        text = "".join(c if unicodedata.category(c)[0] in "LNM" else " " for c in text)
    return " ".join(text.split())


def dataset_rows():
    sets = {}
    for split in ("train", "dev", "test"):
        sets["ViPragSent_" + split] = pd.read_csv(ROOT / f"data/processed/vipragsent/{split}.csv").to_dict("records")
    for dataset in ("uit_vsfc", "uit_vsmec", "aivivn_human_derived_3way"):
        for split in ("train", "dev", "test"):
            path = ROOT / f"data/processed/external/{dataset}/{split}.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path).fillna("")
            if "text" not in frame:
                frame = frame.rename(columns={"comment": "text", "id": "sample_id"})
            sets[dataset + "_" + split] = frame.to_dict("records")
    return sets


def overlap(sets):
    checks, matches = [], []
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            if not (a.startswith("ViPragSent") or b.startswith("ViPragSent")):
                continue
            for relaxed in (False, True):
                ma, mb = defaultdict(list), defaultdict(list)
                for row in sets[a]:
                    ma[normalized(row["text"], relaxed)].append(row["sample_id"])
                for row in sets[b]:
                    mb[normalized(row["text"], relaxed)].append(row["sample_id"])
                common = (ma.keys() & mb.keys()) - {""}
                checks.append({"left": a, "right": b, "normalization": "punctuation_relaxed" if relaxed else "NFKC_casefold_whitespace", "shared_unique_strings": len(common), "left_rows": sum(len(ma[t]) for t in common), "right_rows": sum(len(mb[t]) for t in common), "pairs": sum(len(ma[t])*len(mb[t]) for t in common)})
                if not relaxed:
                    for t in common:
                        for aid in ma[t]:
                            for bid in mb[t]:
                                matches.append({"left": a, "right": b, "left_id": aid, "right_id": bid, "kind": "exact", "similarity": 1.0, "left_text": t, "right_text": t})

    # Complete candidate generation for character-5-gram Jaccard >= 0.80.
    # Length and intersection bounds exclude only mathematically impossible pairs.
    prag = [(name, r) for name in names if name.startswith("ViPragSent") for r in sets[name]]
    texts = [normalized(r["text"], True) for _, r in prag]
    grams = [set(t[k:k+5] for k in range(max(0, len(t)-4))) if len(t) >= 20 else set() for t in texts]
    index = defaultdict(list)
    for i, gs in enumerate(grams):
        for g in gs:
            index[g].append(i)
    near_summary = []
    for name in names:
        if name.startswith("ViPragSent"):
            continue
        found = []
        for row in sets[name]:
            t = normalized(row["text"], True)
            if len(t) < 20:
                continue
            gs = set(t[k:k+5] for k in range(len(t)-4))
            hits = Counter(i for g in gs for i in index.get(g, []) if .8*len(gs) <= len(grams[i]) <= len(gs)/.8)
            for i, intersection in hits.items():
                similarity = intersection / (len(gs)+len(grams[i])-intersection)
                if similarity >= .8:
                    left, pr = prag[i]
                    kind = "normalized_exact" if texts[i] == t else "near"
                    item = {"left": left, "right": name, "left_id": pr["sample_id"], "right_id": row["sample_id"], "kind": kind, "similarity": similarity, "left_text": pr["text"], "right_text": row["text"]}
                    found.append(item)
                    matches.append(item)
        for split in ("train", "dev", "test"):
            selected = [r for r in found if r["left"] == "ViPragSent_"+split and r["kind"] == "near"]
            near_summary.append({"left": "ViPragSent_"+split, "right": name, "char_ngram": 5, "minimum_characters": 20, "jaccard_threshold": .8, "near_pairs_excluding_normalized_exact": len(selected), "left_unique_rows": len({r["left_id"] for r in selected}), "right_unique_rows": len({r["right_id"] for r in selected})})
        print("Overlap completed:", name, flush=True)
    write_csv("overlap_exact.csv", checks)
    write_csv("overlap_near.csv", near_summary)
    write_csv("overlap_pairs.csv", matches)
    return checks, near_summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    loaded, metadata = _validate_sources()
    sets = dataset_rows()
    gold = {r["sample_id"]: r for r in sets["ViPragSent_test"]}
    train_lengths = [len(str(r["text"]).split()) for r in sets["ViPragSent_train"]]
    length_cut = float(np.median(train_lengths))
    per_seed, subgroup, bins, curves = [], [], [], []
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), sharex=True, sharey=True)
    for model in ("proposed_multitask", "baseline_pragmatic_only"):
        for seed in SEEDS:
            rows = loaded[seed][model]["rows"]
            ids = [r["sample_id"] for r in rows]
            assert set(ids) == set(gold)
            for row in rows:
                assert all(int(row["gold"][label]) == int(gold[row["sample_id"]][label]) for label in LABELS)
            ys = {label: np.array([r["gold"][label] for r in rows]) for label in LABELS}
            ps = {label: np.array([r["probabilities"][label] for r in rows]) for label in LABELS}
            es, macro, reliability = pragmatic_ece(ys, ps)
            masks = {
                "all": np.ones(len(rows), dtype=bool),
                "length_short": np.array([len(str(gold[x]["text"]).split()) <= length_cut for x in ids]),
                "length_long": np.array([len(str(gold[x]["text"]).split()) > length_cut for x in ids]),
                "code_switch_absent": ys["code_switching"] == 0,
                "code_switch_present": ys["code_switching"] == 1,
            }
            for j, label in enumerate(LABELS):
                y, p = ys[label], ps[label]
                pred = np.array([r["predictions"][label] for r in rows])
                prec, rec, f1, support = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
                ece = scalar_ece(y, p)
                assert np.isclose(ece, es[label], atol=1e-14)
                ap = average_precision_score(y, p)
                item = {"model": model, "seed": seed, "label": label, "n": len(rows), "positive_n": int(y.sum()), "prevalence": float(y.mean()), "precision_pos": prec[1], "recall_pos": rec[1], "f1_pos": f1[1], "precision_neg": prec[0], "recall_neg": rec[0], "f1_neg": f1[0], "binary_macro_f1": float(f1.mean()), "average_precision": ap, "ece": ece, "brier": float(np.mean((p-y)**2)), "gold_positive_probability_shortfall": float(1-p[y==1].mean()), "gold_negative_probability_excess": float(p[y==0].mean())}
                per_seed.append(item)
                for b in reliability[label]:
                    bins.append({"model": model, "seed": seed, "label": label, **b})
                precision, recall, thresholds = precision_recall_curve(y, p)
                for k in range(len(precision)):
                    curves.append({"model": model, "seed": seed, "label": label, "precision": precision[k], "recall": recall[k], "threshold": thresholds[k] if k < len(thresholds) else np.nan})
                for group, mask in masks.items():
                    subgroup.append({"model": model, "seed": seed, "group": group, "label": label, "n": int(mask.sum()), "positive_n": int(y[mask].sum()), "ece": scalar_ece(y[mask], p[mask]), "brier": float(np.mean((p[mask]-y[mask])**2))})
                color = "#1764ab" if model == "proposed_multitask" else "#c36a18"
                axes.flat[j].step(recall, precision, where="post", color=color, alpha=.55, linewidth=1.0, label=("ViPragSent" if model == "proposed_multitask" else "Pragmatic-only") if seed == SEEDS[0] else None)
            print(model, seed, "macro F1", np.mean([r["binary_macro_f1"] for r in per_seed if r["model"] == model and r["seed"] == seed]), "macro ECE", macro, flush=True)
    metrics = ["precision_pos", "recall_pos", "f1_pos", "precision_neg", "recall_neg", "f1_neg", "binary_macro_f1", "average_precision", "ece", "brier"]
    summary = summarize(per_seed, ["model", "label"], metrics)
    subsummary = summarize(subgroup, ["model", "group", "label"], ["ece", "brier", "n", "positive_n"])
    group_macros = []
    for model in ("proposed_multitask", "baseline_pragmatic_only"):
        for seed in SEEDS:
            for group in masks:
                subset = [r for r in subgroup if r["model"] == model and r["seed"] == seed and r["group"] == group]
                group_macros.append({"model": model, "seed": seed, "group": group, "n": subset[0]["n"], "six_head_ece": float(np.mean([r["ece"] for r in subset])), "five_head_ece_excluding_code_switching": float(np.mean([r["ece"] for r in subset if r["label"] != "code_switching"])), "six_head_brier": float(np.mean([r["brier"] for r in subset]))})
    group_summary = summarize(group_macros, ["model", "group"], ["n", "six_head_ece", "five_head_ece_excluding_code_switching", "six_head_brier"])
    for name, content in (("per_seed.csv", per_seed), ("summary.csv", summary), ("calibration_groups.csv", subgroup), ("calibration_groups_summary.csv", subsummary), ("calibration_group_macro.csv", group_macros), ("calibration_group_macro_summary.csv", group_summary), ("reliability_bins.csv", bins), ("pr_curve_points.csv", curves)):
        write_csv(name, content)
    for j, label in enumerate(LABELS):
        ax = axes.flat[j]
        prevalence = sum(int(r[label]) for r in gold.values())/len(gold)
        ax.axhline(prevalence, color="gray", ls="--", lw=.8, label="Prevalence")
        ap = next(r for r in summary if r["model"] == "proposed_multitask" and r["label"] == label)
        ax.set_title(f"{DISPLAY[j]} | AP {ap['average_precision_mean']:.3f} ± {ap['average_precision_sd']:.3f}")
        ax.set(xlim=(0,1), ylim=(0,1.02), xlabel="Recall", ylabel="Precision")
        ax.grid(alpha=.15)
    axes.flat[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Test precision–recall curves; three seeds per system; raw probabilities", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(OUT / "precision_recall_six_heads.png", dpi=200)
    fig.savefig(OUT / "precision_recall_six_heads.pdf")
    exact, near = overlap(sets)
    notes = {"seeds": list(SEEDS), "test_n": len(gold), "length_measure": "whitespace-separated units", "length_cut_from_training_median": length_cut, "ece_protocol": "raw positive probability; ten equal-width bins; computed per seed then mean and sample SD", "class_conditioned_errors_are_not_standard_ece": True, "code_switch_subgroup_primary_macro_excludes_grouping_head": True, "dataset_counts": {k: len(v) for k,v in sets.items()}, "full_macro_ece": [r for r in group_summary if r["model"] == "proposed_multitask" and r["group"] == "all"], "independent_prediction_inference": False, "independent_recomputation_of_metrics": True}
    (OUT / "analysis_notes.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
    print(json.dumps(notes, indent=2), flush=True)


if __name__ == "__main__":
    main()
