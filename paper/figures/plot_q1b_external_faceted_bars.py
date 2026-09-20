"""Plot provisional Q1b external-retention facets.

The plot uses horizontal bars in one panel per external dataset/aggregate so
the layout remains readable when the evidence package restores more model
rows. Missing cells remain missing; no imputation or identity relabeling is
performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATASETS = [
    ("AIVIVN", "AIVIVN"),
    ("UIT_VSFC", "UIT-VSFC"),
    ("UIT_VSMEC", "UIT-VSMEC"),
    ("ordinary_f1", "Ordinary F1 (aggregate; ranking withheld)"),
]
REQUIRED = {"model", "backbone", "scope", "record_status", "n", "seeds"}
REQUIRED.update(field for key, _ in DATASETS for field in (f"{key}_mean", f"{key}_sd"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Missing CSV header: {path}")
        missing = sorted(REQUIRED - set(reader.fieldnames))
        if missing:
            raise ValueError(f"Missing Q1b columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError("Q1b input has no rows")
    primary = [row for row in rows if row["scope"].strip().lower() == "primary"]
    if len(primary) != 1:
        raise ValueError(f"Expected exactly one scope=primary row, found {len(primary)}")
    for row in rows:
        int(row["n"])
        for key, _ in DATASETS:
            mean = row[f"{key}_mean"].strip()
            sd = row[f"{key}_sd"].strip()
            if mean:
                float(mean)
                if sd:
                    float(sd)
            elif sd:
                raise ValueError(f"SD cannot be present without a mean for {row['model']} / {key}")
    primary_row = primary[0]
    return [primary_row] + [row for row in rows if row is not primary_row]


def plot(rows: list[dict[str, str]], output_stem: Path, input_path: Path, paper_size: bool = False) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figsize = (6.9, 8.6) if paper_size else (11.5, 7.5)
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=False)
    axes = axes.ravel()
    names = [
        "ViPragSent (ours, XLM-R-large)" if row["scope"].strip().lower() == "primary" else row["model"]
        for row in rows
    ]
    colors = ["#0072B2" if row["scope"].strip().lower() == "primary" else "#9CA3AF" for row in rows]
    y = np.arange(len(rows))
    for axis, (key, title) in zip(axes, DATASETS):
        means: list[float] = []
        sds: list[float] = []
        mean_present: list[bool] = []
        sd_present: list[bool] = []
        for row in rows:
            mean = row[f"{key}_mean"].strip()
            sd = row[f"{key}_sd"].strip()
            if mean:
                means.append(float(mean))
                sds.append(float(sd) if sd else 0.0)
                mean_present.append(True)
                sd_present.append(bool(sd))
            else:
                means.append(0.0)
                sds.append(0.0)
                mean_present.append(False)
                sd_present.append(False)
        means_array = np.array(means)
        sds_array = np.array(sds)
        mean_present_array = np.array(mean_present)
        sd_present_array = np.array(sd_present)
        if mean_present_array.any():
            axis.barh(
                y[mean_present_array],
                means_array[mean_present_array],
                color=np.array(colors)[mean_present_array].tolist(),
                edgecolor="#374151",
                linewidth=0.35,
                height=0.68,
            )
        for index, (mean, sd, has_mean, has_sd) in enumerate(zip(means_array, sds_array, mean_present_array, sd_present_array)):
            if has_sd:
                axis.errorbar(mean, index, xerr=sd, fmt="none", ecolor="#111827", elinewidth=0.9, capsize=2, zorder=3)
            elif has_mean:
                axis.text(min(mean + 1.0, 82.0), index, "n=1; SD unavailable", va="center", fontsize=6.5, color="#6B7280")
            else:
                axis.text(1.5, index, "missing", va="center", fontsize=7.5, color="#6B7280")
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
    title = (
        "Q1b external retention - WORKING DRAFT / PENDING CROSS-COHORT VERIFICATION"
        if paper_size
        else "Q1b external retention - PROVISIONAL CURRENT SUBSET"
    )
    fig.suptitle(title, fontsize=13, y=0.995)
    footer = (
        "Bars show saved-run means; whiskers show sample SD where available. Rows with n=1 show no SD whisker. "
        "Ordinary F1 is the unweighted aggregate of the three datasets, shown for layout only; direct ranking is "
        "withheld pending per-dataset matched ID/gold proof. Primary AIVIVN normalized-test SHA-256 prefix: "
        "27DA59A4...; baseline prefix: F6F6A78E.... Hash comparisons are per-dataset cross-group checks, "
        "not one hash shared across datasets. Model identities, missingness, "
        "and primary scope are preserved; this is a working draft."
    )
    fig.text(
        0.5,
        0.01,
        "\n".join(textwrap.wrap(footer, width=125)),
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.965))
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    metadata = {
        "preview": True,
        "question": "Q1b",
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "row_count": len(rows),
        "primary_system": "ViPragSent (ours, XLM-R-large)",
        "datasets": [key for key, _ in DATASETS],
        "unit": "percentage-point macro-F1",
        "aggregate_definition": "ordinary F1 is the unweighted mean of UIT-VSFC, UIT-VSMEC, and AIVIVN",
        "coverage_status": "working full-inventory layout; blank cells are unverified placeholders; regenerate from canonical evidence",
        "ordinary_ranking_status": "WITHHELD_PENDING_MATCHED_ID_GOLD_PROOF",
        "cohort_provenance": {
            "primary_aivivn_normalized_test_sha256_prefix": "27DA59A4...",
            "baseline_normalized_test_sha256_prefix": "F6F6A78E...",
            "hash_comparison_semantics": "per-dataset cross-group matching; not cross-dataset hash equality",
            "baseline_hash_scope_reported": "dataset-specific prefixes; semantic cohort equality under investigation",
        },
    }
    output_stem.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent / "source_data" / "q1b_external_provisional_subset.csv",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path(__file__).parent / "figure_q1b_external_faceted_bars_preview",
    )
    parser.add_argument("--paper-size", action="store_true", help="Use a 6.9-inch-wide paper figure canvas")
    args = parser.parse_args()
    rows = load_rows(args.input)
    plot(rows, args.output_stem, args.input, paper_size=args.paper_size)
    print(f"Wrote provisional Q1b preview for {len(rows)} systems from {args.input}")


if __name__ == "__main__":
    main()
