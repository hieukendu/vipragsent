"""Ten-bin SCE/ACE (Nixon et al., 2019) from stored paired test probabilities.

Each binary head uses both class probabilities, without probability thresholding.
Scores average six heads per seed, then report mean and sample SD over seeds.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from _bootstrap import ROOT
from analyze_priority3_positive_class import MODEL_KEYS, SEEDS, _validate_sources
from analyze_calibration_threshold_sensitivity import bin_rows, errors
from vipragsent.constants import PRAGMATIC_LABELS

BINS = 10
METRICS = ("ece", "sce", "ace", "rmsce", "brier")


def scores(y, p, ids):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    assert len(y) == len(p) == 2000
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    assert np.isin(y, (0, 1)).all()
    classes = np.column_stack((1 - p, p))
    one_hot = np.column_stack((1 - y, y))
    sce, ace, split_ties = [], [], 0
    edges = np.linspace(0, 1, BINS + 1)
    for k in range(2):
        q, target = classes[:, k], one_hot[:, k]
        index = np.searchsorted(edges[1:-1], q, side="right")
        fixed = 0.0
        for b in range(BINS):
            selected = index == b
            if selected.any():
                fixed += selected.mean() * abs(q[selected].mean() - target[selected].mean())
        sce.append(fixed)
        order = np.lexsort((ids, q))
        groups = np.array_split(order, BINS)
        ace.append(np.mean([abs(q[g].mean() - target[g].mean()) for g in groups]))
        split_ties += sum(q[a[-1]] == q[b[0]] for a, b in zip(groups, groups[1:]))

        # Independent histogram and matrix calculations validate both definitions.
        count = np.bincount(index, minlength=BINS)
        difference = np.bincount(index, weights=q-target, minlength=BINS)
        np.testing.assert_allclose(fixed, np.abs(difference).sum()/len(q), atol=1e-12, rtol=0)
        grouped_difference = (q[order]-target[order]).reshape(BINS, -1).mean(axis=1)
        np.testing.assert_allclose(ace[-1], np.abs(grouped_difference).mean(), atol=1e-12, rtol=0)
        assert count.sum() == len(y)
    positive_bins, _ = bin_rows(y, p, bins=BINS)
    ece, rmsce = errors(positive_bins)
    return dict(ece=ece, sce=float(np.mean(sce)), ace=float(np.mean(ace)),
                rmsce=rmsce, brier=float(np.mean((p-y)**2))), split_ties


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    loaded, _ = _validate_sources()
    per_head, per_seed, summary = [], [], []
    tie_count = 0
    for model in MODEL_KEYS:
        for seed in SEEDS:
            rows = loaded[seed][model]["rows"]
            ids = np.array([str(row["sample_id"]) for row in rows])
            heads = []
            for label in PRAGMATIC_LABELS:
                y = [row["gold"][label] for row in rows]
                p = [row["probabilities"][label] for row in rows]
                metrics, ties = scores(y, p, ids)
                tie_count += ties
                heads.append(metrics)
                per_head.append(dict(model=model, seed=seed, label=label, **metrics))
            per_seed.append(dict(model=model, seed=seed,
                                 **{key: float(np.mean([h[key] for h in heads])) for key in METRICS}))
        seeds = [row for row in per_seed if row["model"] == model]
        for key in METRICS:
            values = [row[key] for row in seeds]
            summary.append(dict(model=model, metric=key, mean=float(np.mean(values)),
                                sample_sd=float(np.std(values, ddof=1))))
    out = ROOT / "reports" / "standard_calibration_20261006"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "per_head_seed.csv", per_head)
    write_csv(out / "per_seed.csv", per_seed)
    write_csv(out / "summary.csv", summary)
    (out / "parameters.json").write_text(json.dumps(dict(
        bins=BINS, classes_per_head=2, probability_threshold=None,
        equal_width_boundary="left-closed, right-open; last bin includes 1",
        adaptive_sort="probability, then sample ID; ten equal-count bins",
        adaptive_boundary_ties_split=int(tie_count), seeds=list(SEEDS),
        aggregation="mean six heads within seed; mean/sample SD across seeds",
        definition="Nixon et al. 2019, arXiv:1904.01685v1, SCE and ACE",
    ), indent=2), encoding="utf-8")
    for row in summary:
        print(f'{row["model"]} {row["metric"]}: {row["mean"]:.10f} +/- {row["sample_sd"]:.10f}')
    print(f"Split adaptive boundary ties: {tie_count}")


if __name__ == "__main__":
    main()
