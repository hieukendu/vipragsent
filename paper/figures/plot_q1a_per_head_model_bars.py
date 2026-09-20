"""Plot provisional Q1a per-head and macro model bars.

The input CSV is intentionally schema-driven. The script preserves the
artifact-provided ``system`` and ``backbone`` identities, highlights the one
row marked ``scope=primary``, and performs only the recorded conversion from
[0, 1] to percentage points. It does not search for or add comparison rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = [
    ("implicit_sentiment", "Implicit sentiment"),
    ("sarcasm", "Sarcasm"),
    ("irony", "Irony"),
    ("idiom_figurative", "Idiom / figurative"),
    ("code_switching", "Code-switching"),
    ("mocking", "Mocking"),
    ("macro_pragmatic_f1", "Macro-pragmatic F1"),
]
REQUIRED = {
    "system",
    "backbone",
    "n",
    "seeds",
    "scope",
    *(field for metric, _ in METRICS for field in (f"{metric}_mean", f"{metric}_sd")),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Missing CSV header: {path}")
        missing = sorted(REQUIRED - set(reader.fieldnames))
        if missing:
            raise ValueError(f"Missing Q1a columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError("Q1a input has no rows")
    primary = [row for row in rows if row["scope"].strip().lower() == "primary"]
    if len(primary) != 1:
        raise ValueError(f"Expected exactly one scope=primary row, found {len(primary)}")
    for row in rows:
        int(row["n"])
        for metric, _ in METRICS:
            mean = row[f"{metric}_mean"].strip()
            sd = row[f"{metric}_sd"].strip()
            if mean and sd:
                float(mean)
                float(sd)
            elif mean or sd:
                raise ValueError(f"Both mean and SD must be present or blank for {row['system']} / {metric}")
    primary_row = primary[0]
    return [primary_row] + [row for row in rows if row is not primary_row]


def plot(rows: list[dict[str, str]], output_stem: Path, input_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 7.5,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(4, 2, figsize=(12.5, 11.5), sharex=False)
    axes = axes.ravel()
    names = [
        "ViPragSent (ours, XLM-R-large)" if row["scope"].strip().lower() == "primary" else row["system"]
        for row in rows
    ]
    colors = ["#0072B2" if row["scope"].strip().lower() == "primary" else "#9CA3AF" for row in rows]
    y = np.arange(len(rows))
    for axis, (metric, title) in zip(axes, METRICS):
        present = np.array([bool(row[f"{metric}_mean"].strip() and row[f"{metric}_sd"].strip()) for row in rows])
        means = np.array([100.0 * float(row[f"{metric}_mean"]) if present[index] else 0.0 for index, row in enumerate(rows)])
        sds = np.array([100.0 * float(row[f"{metric}_sd"]) if present[index] else 0.0 for index, row in enumerate(rows)])
        if present.any():
            axis.barh(
                y[present],
                means[present],
                xerr=sds[present],
                color=np.array(colors)[present].tolist(),
                edgecolor="#374151",
                linewidth=0.35,
                capsize=2,
                height=0.68,
            )
        for index in y[~present]:
            axis.text(2.0, index, "UNVERIFIED", va="center", fontsize=7.5, color="#6B7280")
        axis.set_title(title)
        axis.set_xlim(0, 100)
        axis.set_xticks(np.arange(0, 101, 20))
        axis.set_xticklabels([str(value) for value in range(0, 101, 20)])
        axis.set_yticks(y, names)
        axis.axvline(0, color="#111827", linewidth=0.8)
        axis.grid(axis="x", color="#D1D5DB", linewidth=0.6, alpha=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(axis="y", length=0)
        axis.invert_yaxis()
        axis.set_xlabel("Macro-F1 (%)")
        axis.tick_params(axis="x", labelbottom=True)
    axes[-1].axis("off")
    n_values = sorted({int(row["n"]) for row in rows})
    n_caption = ", ".join(str(value) for value in n_values)
    n_caption = f"n={n_caption} per system" if len(n_values) == 1 else f"n values={n_caption}"
    fig.suptitle("Q1a per-head and macro model comparison - WORKING FULL INVENTORY", fontsize=13, y=0.995)
    fig.text(
        0.5,
        0.01,
        f"Bars show saved-run means; whiskers show sample SD ({n_caption}). Model identities and scope are preserved from the input CSV. "
        "Blank cells are unverified working-layout placeholders; this preview is not coverage-complete and is not inserted into the manuscript. "
        "The reports/q1a_best_f1_table values are withheld pending Nash source/hash/protocol audit and are not used here.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.965))
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    metadata = {
        "preview": True,
        "question": "Q1a",
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "row_count": len(rows),
        "primary_system": "ViPragSent (ours, XLM-R-large)",
        "metrics": [metric for metric, _ in METRICS],
        "unit_conversion": "mean and sample SD multiplied by 100 for percentage-point display",
        "coverage_status": "working full-inventory layout; blank cells are unverified placeholders; regenerate from canonical evidence",
    }
    output_stem.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path(__file__).parent / "source_data" / "table2_q1a_baselines.csv")
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path(__file__).parent / "figure_q1a_per_head_model_bars_preview",
    )
    args = parser.parse_args()
    rows = load_rows(args.input)
    plot(rows, args.output_stem, args.input)
    print(f"Wrote provisional Q1a preview for {len(rows)} systems from {args.input}")


if __name__ == "__main__":
    main()
