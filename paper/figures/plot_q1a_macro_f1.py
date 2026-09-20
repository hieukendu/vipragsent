"""Create the compact Q1a macro-pragmatic F1 comparison figure.

The input CSV is an exact paper-local copy of the validated artifact-package
table. The script performs only a unit conversion from [0, 1] to percent and
plots the recorded mean plus sample SD values; it does not re-estimate metrics.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "source_data" / "table2_q1a_baselines.csv"
OUTPUT_PNG = ROOT / "figure_q1a_macro_f1_baselines.png"
OUTPUT_PDF = ROOT / "figure_q1a_macro_f1_baselines.pdf"

TARGET = "ViPragSent (XLM-R-large)"
LABELS = {
    "PhoBERT (single-task)": "PhoBERT single-task",
    "PhoBERT (fine-tune)": "PhoBERT fine-tune",
    "XLM-R-large (baseline)": "XLM-R-large baseline",
    "Sailor-7B SFT": "Sailor-7B SFT",
    "Vistral-7B SFT": "Vistral-7B SFT",
    TARGET: "ViPragSent XLM-R-large",
}


def load_rows() -> list[dict[str, str]]:
    with INPUT.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = set(LABELS)
    actual = {row["system"] for row in rows}
    if actual != expected:
        raise ValueError(f"Unexpected Q1a systems: {sorted(actual ^ expected)}")
    if any(int(row["n"]) != 3 for row in rows):
        raise ValueError("Q1a figure requires n=3 for every plotted system")
    return rows


def main() -> None:
    rows = sorted(load_rows(), key=lambda row: float(row["macro_pragmatic_f1_mean"]))
    names = [LABELS[row["system"]] for row in rows]
    means = np.array([100 * float(row["macro_pragmatic_f1_mean"]) for row in rows])
    sds = np.array([100 * float(row["macro_pragmatic_f1_sd"]) for row in rows])
    is_target = np.array([row["system"] == TARGET for row in rows])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots(figsize=(6.7, 3.05))
    y = np.arange(len(rows))
    baseline_color = "#6B7280"
    target_color = "#0072B2"
    colors = np.where(is_target, target_color, baseline_color)
    markers = np.where(is_target, "D", "o")

    for yi, mean, sd, color, marker, target in zip(y, means, sds, colors, markers, is_target):
        ax.errorbar(
            mean,
            yi,
            xerr=sd,
            fmt=marker,
            color=color,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=7 if target else 5.5,
            ecolor="#111827",
            elinewidth=1.0,
            capsize=2.5,
            capthick=1.0,
            zorder=3,
        )
        label = f"{mean:.2f} $\\pm$ {sd:.2f}"
        ax.annotate(
            label,
            (mean + sd, yi),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            fontweight="bold" if target else "normal",
            color="#111827",
        )

    ax.set_yticks(y, names)
    ax.set_xlabel("Macro-pragmatic F1 (%)")
    ax.set_xlim(84, 95.5)
    ax.set_xticks(np.arange(84, 96, 2))
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.grid(axis="x", color="#D1D5DB", linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.text(
        0,
        -0.20,
        "Points show mean; whiskers show sample SD. All systems: n=3 logical runs "
        "(labels 21, 22, 23; remote seeds 20260521, 20260522, 20260523).",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUTPUT_PNG, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
