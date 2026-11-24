"""Post-hoc calibration and threshold sensitivity of the matched XLM-R pair.

Only saved probabilities are evaluated. Checkpoints are never changed, and
thresholds for both optimized strategies are chosen on development data only.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from _bootstrap import ROOT
from analyze_priority3_positive_class import (
    MODEL_KEYS, SEEDS, _baseline_run_root, _full_run_root, _prediction_path,
    _read_jsonl, _validate_sources,
)
from vipragsent.constants import EXPECTED_SPLIT_COUNTS, PRAGMATIC_LABELS
from vipragsent.evaluation.metrics import pragmatic_ece
from vipragsent.evaluation.thresholds import tune_binary_threshold

LABELS = tuple(PRAGMATIC_LABELS)
NAMES = {"proposed_multitask": "ViPragSent (XLM-R MT)",
         "baseline_pragmatic_only": "XLM-R pragmatic-only"}
STRATEGIES = ("fixed_0.5", "dev_binary_macro_f1", "dev_positive_f1")
CAL_METRICS = ("ece", "adaptive_ece", "classwise_ece", "predicted_class_macro_ece",
               "rmsce", "brier", "adaptive_ece_unsplit_ties")
GRID = np.round(np.arange(0.05, 0.955, 0.01), 2)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def bin_rows(y, p, *, bins=10, adaptive=False, ids=None, unsplit_ties=False):
    """Equal-width or probability-sorted equal-count bins; never sort on gold."""
    y, p = np.asarray(y), np.asarray(p)
    require(len(y) == len(p) and len(y) > 0, "Empty or unaligned calibration input")
    if adaptive and not unsplit_ties:
        # ID is a deterministic, outcome-independent secondary sort key.
        ids = np.asarray(ids) if ids is not None else np.arange(len(p))
        order = np.lexsort((ids, p))
        groups = np.array_split(order, min(bins, len(p)))
        split_ties = sum(p[a[-1]] == p[b[0]] for a, b in zip(groups, groups[1:]))
    else:
        edges = (np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
                 if adaptive else np.linspace(0, 1, bins + 1))
        index = np.searchsorted(edges[1:-1], p, side="right")
        groups = [np.flatnonzero(index == b) for b in range(max(1, len(edges) - 1))]
        split_ties = 0
    rows = []
    for b, group in enumerate(groups):
        if len(group):
            rows.append({"bin": b, "count": len(group), "prob_min": float(p[group].min()),
                         "prob_max": float(p[group].max()), "mean_probability": float(p[group].mean()),
                         "empirical_rate": float(y[group].mean()),
                         "gap": float(p[group].mean() - y[group].mean())})
    require(sum(r["count"] for r in rows) == len(y), "Calibration bins lost observations")
    return rows, int(split_ties)


def errors(rows):
    n = sum(r["count"] for r in rows)
    return (sum(r["count"] / n * abs(r["gap"]) for r in rows),
            float(np.sqrt(sum(r["count"] / n * r["gap"] ** 2 for r in rows))))


def scores(y, prediction):
    tp = int(np.sum((y == 1) & (prediction == 1)))
    tn = int(np.sum((y == 0) & (prediction == 0)))
    fp = int(np.sum((y == 0) & (prediction == 1)))
    fn = int(np.sum((y == 1) & (prediction == 0)))
    positive = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    negative = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    macro = (positive + negative) / 2
    # Independently verify the count implementation with sklearn.
    require(np.isclose(macro, f1_score(y, prediction, labels=[0, 1], average="macro", zero_division=0)),
            "Binary macro-F1 differs from sklearn")
    require(np.isclose(positive, f1_score(y, prediction, pos_label=1, zero_division=0)),
            "Positive F1 differs from sklearn")
    return {"binary_macro_f1": macro, "positive_f1": positive,
            "positive_precision": tp / (tp + fp) if tp + fp else 0.0,
            "positive_recall": tp / (tp + fn) if tp + fn else 0.0,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def select_positive(y, p):
    values = []
    for threshold in GRID:
        prediction = p >= threshold
        tp = np.sum((y == 1) & prediction)
        fp = np.sum((y == 0) & prediction)
        fn = np.sum((y == 1) & ~prediction)
        score = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        values.append((float(score), float(threshold)))
    values.sort(key=lambda x: (-x[0], abs(x[1] - 0.5), x[1]))
    return values[0][1]


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, keys, metrics):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    output = []
    for key, group in grouped.items():
        row = dict(zip(keys, key))
        require(len(group) == len(SEEDS), "Seed summary is incomplete")
        for metric in metrics:
            values = np.array([r[metric] for r in group])
            row[metric + "_mean"] = float(values.mean())
            row[metric + "_sd"] = float(values.std(ddof=1))
        output.append(row)
    return output


def render_summary(calibration, thresholds, ranking):
    def cell(row, metric, percent=False):
        scale = 100 if percent else 1
        digits = 2 if percent else 4
        return f"{scale * row[metric + '_mean']:.{digits}f} ± {scale * row[metric + '_sd']:.{digits}f}"
    lines = ["# Calibration and threshold sensitivity", "",
             "Saved probabilities from the same three XLM-R checkpoints per model are evaluated on 2,000 paired test comments. Development contains 1,999 comments; dev and test IDs are disjoint. All six pragmatic heads are retained. No training or probability recalibration is performed. Values are mean ± sample SD across three seeds; probabilities are never pooled across seeds.", "",
             "## Calibration", "",
             "Lower is better; all values are on a 0–1 scale. ECE uses raw positive probabilities with ten equal-width bins, matching the manuscript. Adaptive ECE uses ten equal-count bins (200 test rows per bin), sorted by probability and then sample ID. RMSCE changes the within-head aggregation to the square root of the count-weighted squared bin gaps. Brier is mean squared probability error, a proper score that also reflects discrimination rather than calibration alone.", "",
             "Class-wise ECE averages ECE for (p, y) and (1−p, 1−y) within each binary head, then averages the six heads. These complementary class errors coincide here. Predicted-class macro-ECE is an additional class-balanced diagnostic: predict each head at 0.5, compute confidence-versus-correctness ECE separately for predicted negative and predicted positive, give the two groups equal weight, then average heads. It is distinct from class-wise ECE and is explicitly labeled as such.", "",
             "| Model | ECE | Adaptive ECE | Class-wise ECE | Predicted-class macro-ECE | RMSCE | Brier |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in calibration:
        lines.append("| " + NAMES[row["model"]] + " | " + " | ".join(cell(row, k) for k in CAL_METRICS[:6]) + " |")
    lines.extend(["", "Approximate equal-count quantile bins that keep tied probabilities together are also calculated in the CSVs; they are a sensitivity check, not a replacement chosen after inspecting results.", "",
                  "## Threshold sensitivity", "",
                  "Both optimized strategies search 0.05–0.95 in steps of 0.01 independently per head and seed, using development labels only. Ties choose the threshold nearest 0.5, then the smaller threshold. The resulting thresholds are applied unchanged to test probabilities. Dev binary macro-F1 exactly reproduces all saved thresholds and the existing test metrics. Macro-pragmatic F1 is the unweighted mean of six binary macro-F1 values. The same checkpoints are used in every strategy.", "",
                  "| Strategy | Model | Sarcasm binary macro-F1 (%) | Mocking binary macro-F1 (%) | Macro-pragmatic F1 (%) | Sarcasm positive F1 (%) | Mocking positive F1 (%) |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    for strategy in STRATEGIES:
        for row in thresholds:
            if row["strategy"] == strategy:
                lines.append("| " + strategy + " | " + NAMES[row["model"]] + " | " + " | ".join(
                    cell(row, k, True) for k in ("sarcasm_binary_macro_f1", "mocking_binary_macro_f1",
                    "macro_pragmatic_f1", "sarcasm_positive_f1", "mocking_positive_f1")) + " |")
    delta = next(r for r in ranking if r["analysis"] == "fixed_0.5" and r["metric"] == "macro_pragmatic_f1")
    lines.extend(["", "The mean calibration ranking favors pragmatic-only under every metric shown, although not every individual seed favors it. The macro-pragmatic F1 ranking is threshold-dependent: ViPragSent leads with development-selected thresholds, whereas at 0.5 its mean is "
                  f"{100 * abs(delta['delta_mean']):.2f} percentage points lower. Sarcasm favors ViPragSent under all three strategies; mocking is effectively tied at 0.5. On the stated grid, development positive-F1 and binary macro-F1 select the same thresholds for all 36 model/seed/head combinations. This equality is specific to these saved scores and the stated search grid."])
    lines.extend(["", "## Paired descriptive differences", "",
                  "Differences below are ViPragSent minus pragmatic-only. Negative calibration differences favor ViPragSent; positive F1 differences favor ViPragSent. No significance claim is made from seed SD or the count of favorable seeds.", "",
                  "| Analysis | Metric | Mean difference | Favorable seeds / 3 |", "|---|---|---:|---:|"])
    for row in ranking:
        scale = 100 if row["analysis"] != "calibration" else 1
        lines.append(f"| {row['analysis']} | {row['metric']} | {scale * row['delta_mean']:+.5f} | {row['favorable_seeds']} |")
    lines.extend(["", "Definitions: [Nixon et al., Measuring Calibration in Deep Learning](https://arxiv.org/abs/1904.01685); [Gupta and Ramdas, Top-label calibration](https://arxiv.org/abs/2107.08353).", ""])
    return "\n".join(lines)


def run(output):
    loaded, _ = _validate_sources()
    dev_reference = None
    calibration_rows, conditional_rows, backing, threshold_rows, performance = [], [], [], [], []
    macro_calibration, macro_performance = [], []
    for seed in SEEDS:
        for model in MODEL_KEYS:
            entry = loaded[seed][model]
            root = _full_run_root(seed) if model == "proposed_multitask" else _baseline_run_root(seed)
            dev = _read_jsonl(_prediction_path(model, root, "dev"))
            test = entry["rows"]
            signature = [(r["sample_id"], tuple(r["gold"][k] for k in LABELS)) for r in dev]
            if dev_reference is None:
                dev_reference = signature
            require(signature == dev_reference, "Development IDs/gold differ across models or seeds")
            require(set(r["sample_id"] for r in dev).isdisjoint(r["sample_id"] for r in test),
                    "Dev and test IDs overlap")
            for split, rows in (("dev", dev), ("test", test)):
                with (ROOT / "data" / "processed" / "vipragsent" / f"{split}.csv").open(encoding="utf-8-sig", newline="") as handle:
                    canonical = list(csv.DictReader(handle))
                gold_by_id = {r["sample_id"]: tuple(int(r[k]) for k in LABELS) for r in canonical}
                require(len(gold_by_id) == EXPECTED_SPLIT_COUNTS[split], "Canonical split count mismatch")
                require({r["sample_id"]: tuple(int(r["gold"][k]) for k in LABELS) for r in rows} == gold_by_id,
                        "Saved labels differ from canonical split")
            ids = np.array([r["sample_id"] for r in test])
            true_map = {k: [r["gold"][k] for r in test] for k in LABELS}
            prob_map = {k: [r["probabilities"][k] for r in test] for k in LABELS}
            reference_ece, _, _ = pragmatic_ece(true_map, prob_map)
            local_calibration = []
            local_performance = defaultdict(list)
            for label in LABELS:
                yd = np.array([r["gold"][label] for r in dev])
                pd = np.array([r["probabilities"][label] for r in dev])
                y, p = np.array(true_map[label]), np.array(prob_map[label])
                ew, _ = bin_rows(y, p)
                ec, ties = bin_rows(y, p, adaptive=True, ids=ids)
                qt, _ = bin_rows(y, p, adaptive=True, unsplit_ties=True)
                ece, rmsce = errors(ew)
                require(np.isclose(ece, reference_ece[label], atol=1e-14, rtol=0), "ECE differs from manuscript implementation")
                negative_ece = errors(bin_rows(1 - y, 1 - p)[0])[0]
                require(np.isclose(ece, negative_ece, atol=1e-12, rtol=0), "Complementary class ECE differs")
                prediction = (p >= 0.5).astype(int)
                confidence = np.where(prediction == 1, p, 1 - p)
                correct = (prediction == y).astype(int)
                class_errors = []
                for cls in (0, 1):
                    group = prediction == cls
                    require(group.any(), "Predicted-class calibration group is empty")
                    ce = errors(bin_rows(correct[group], confidence[group])[0])[0]
                    class_errors.append(ce)
                    conditional_rows.append({"model": model, "seed": seed, "label": label,
                                             "predicted_class": cls, "count": int(group.sum()), "ece": ce})
                row = {"model": model, "seed": seed, "label": label, "ece": ece,
                       "adaptive_ece": errors(ec)[0], "classwise_ece": (ece + negative_ece) / 2,
                       "predicted_class_macro_ece": float(np.mean(class_errors)), "rmsce": rmsce,
                       "brier": float(np.mean((p - y) ** 2)), "adaptive_ece_unsplit_ties": errors(qt)[0],
                       "equal_count_boundaries_splitting_ties": ties}
                calibration_rows.append(row)
                local_calibration.append(row)
                for binning, groups in (("equal_width", ew), ("equal_count", ec), ("quantile_unsplit_ties", qt)):
                    for group in groups:
                        backing.append({"model": model, "seed": seed, "label": label, "binning": binning, **group})
                thresholds = {"fixed_0.5": 0.5, "dev_binary_macro_f1": tune_binary_threshold(yd, pd),
                              "dev_positive_f1": select_positive(yd, pd)}
                require(thresholds["dev_binary_macro_f1"] == entry["thresholds"][label],
                        f"Saved threshold is not reproduced for {model}, {seed}, {label}")
                for strategy, threshold in thresholds.items():
                    threshold_rows.append({"model": model, "seed": seed, "label": label, "strategy": strategy,
                                           "threshold": threshold, "selection_split": "none" if strategy == "fixed_0.5" else "dev"})
                    metrics = scores(y, (p >= threshold).astype(int))
                    record = {"model": model, "seed": seed, "label": label, "strategy": strategy,
                              "threshold": threshold, **metrics}
                    performance.append(record)
                    local_performance[strategy].append(record)
                    if strategy == "dev_binary_macro_f1":
                        stored = np.array([r["predictions"][label] for r in test])
                        require(np.array_equal(stored, (p >= threshold).astype(int)), "Saved decisions not reproduced")
            macro_calibration.append({"model": model, "seed": seed,
                                      **{k: float(np.mean([r[k] for r in local_calibration])) for k in CAL_METRICS}})
            for strategy, rows in local_performance.items():
                result = {"model": model, "seed": seed, "strategy": strategy,
                          "macro_pragmatic_f1": float(np.mean([r["binary_macro_f1"] for r in rows])),
                          "macro_positive_f1": float(np.mean([r["positive_f1"] for r in rows]))}
                for label in ("sarcasm", "mocking"):
                    item = next(r for r in rows if r["label"] == label)
                    for metric in ("binary_macro_f1", "positive_f1", "positive_precision", "positive_recall"):
                        result[label + "_" + metric] = item[metric]
                macro_performance.append(result)
                if strategy == "dev_binary_macro_f1":
                    metric_path = (root / "metrics_test.json" if model == "baseline_pragmatic_only"
                                   else root / "metrics" / "test_metrics.json")
                    stored = json.loads(metric_path.read_text(encoding="utf-8"))
                    require(np.isclose(result["macro_pragmatic_f1"], stored["macro_pragmatic_f1"], atol=1e-12, rtol=0),
                            "Existing test macro metric not reproduced")
                    for item in rows:
                        require(np.isclose(item["binary_macro_f1"], stored["per_label_f1"][item["label"]], atol=1e-12, rtol=0),
                                "Existing per-head test metric not reproduced")
    cal_summary = summarize(macro_calibration, ["model"], CAL_METRICS)
    perf_metrics = tuple(k for k in macro_performance[0] if k not in ("model", "seed", "strategy"))
    perf_summary = summarize(macro_performance, ["model", "strategy"], perf_metrics)
    ranking = []
    for analysis, records, metrics in [("calibration", macro_calibration, CAL_METRICS)] + [
        (strategy, [r for r in macro_performance if r["strategy"] == strategy], perf_metrics) for strategy in STRATEGIES]:
        for metric in metrics:
            deltas = []
            for seed in SEEDS:
                pair = {r["model"]: r[metric] for r in records if r["seed"] == seed}
                deltas.append(pair["proposed_multitask"] - pair["baseline_pragmatic_only"])
            ranking.append({"analysis": analysis, "metric": metric, "delta_mean": float(np.mean(deltas)),
                            "delta_sd": float(np.std(deltas, ddof=1)),
                            "favorable_seeds": sum(d < 0 if analysis == "calibration" else d > 0 for d in deltas)})
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in {"calibration_per_head_seed": calibration_rows, "calibration_per_seed": macro_calibration,
                       "calibration_summary": cal_summary, "predicted_class_calibration": conditional_rows,
                       "calibration_per_head_summary": summarize(calibration_rows, ["model", "label"], CAL_METRICS),
                       "reliability_bins": backing, "selected_thresholds": threshold_rows,
                       "threshold_per_head_seed": performance, "threshold_per_seed": macro_performance,
                       "threshold_per_head_summary": summarize(performance, ["model", "label", "strategy"],
                               ("binary_macro_f1", "positive_f1", "positive_precision", "positive_recall")),
                       "threshold_summary": perf_summary, "paired_ranking": ranking}.items():
        write_csv(output / f"{name}.csv", rows)
    (output / "summary.md").write_text(render_summary(cal_summary, perf_summary, ranking), encoding="utf-8")
    print(json.dumps({"status": "PASS", "seeds": list(SEEDS), "dev_count": EXPECTED_SPLIT_COUNTS["dev"],
                      "test_count": EXPECTED_SPLIT_COUNTS["test"], "canonical_gold_matches": True,
                      "saved_thresholds_reproduced": True, "ece_reproduced": True,
                      "calibration": cal_summary, "thresholds": perf_summary}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/calibration_threshold_sensitivity_20261006")
    args = parser.parse_args()
    run(args.output_dir)
