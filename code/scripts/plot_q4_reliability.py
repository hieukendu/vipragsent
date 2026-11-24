"""Regenerate the matched Q4 comparison from stored reliability bins."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "paper/figures"
INPUT = ROOT / "source_data" / "matched_reliability_bins.csv"
OUTPUT = ROOT / "figure_q4_reliability.png"
TARGET = "ViPragSent"
LABELS = ["implicit_sentiment", "sarcasm", "irony", "idiom_figurative", "code_switching", "mocking"]
SYSTEMS = {
    "Pragmatic-only": ("Pragmatic-only XLM-R", "#D55E00", "s"),
    TARGET: ("ViPragSent XLM-R", "#0072B2", "o"),
}


def load_records() -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    with INPUT.open(encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            grouped[record["model"]].append(record)
    if set(grouped) != set(SYSTEMS):
        raise ValueError("Expected exactly the two matched systems")
    for records in grouped.values():
        pairs = {(r["label"], int(r["seed"])) for r in records}
        keys = {(r["label"], int(r["seed"]), int(r["bin"])) for r in records}
        if len(pairs) != 18 or len(records) != len(keys) or len(records) > 180:
            raise ValueError("Expected six heads and three seeds without duplicate bins")
    return grouped


def aggregate(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    confidence: list[float] = []
    empirical: list[float] = []
    for bin_index in range(10):
        bins = [
            record
            for record in records
            if int(record["bin"]) == bin_index and int(record["count"]) > 0
        ]
        if not bins:
            confidence.append(np.nan)
            empirical.append(np.nan)
        else:
            confidence.append(float(np.mean([float(item["mean_probability"]) for item in bins])))
            empirical.append(float(np.mean([float(item["empirical_rate"]) for item in bins])))
    return np.asarray(confidence), np.asarray(empirical)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    grouped = load_records()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 6.8,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(3.3, 2.25))
    for system, (label, color, marker) in SYSTEMS.items():
        confidence, empirical = aggregate(grouped[system])
        ax.plot(
            confidence,
            empirical,
            label=label,
            color=color,
            marker=marker,
            linewidth=1.35,
            markersize=3.8,
            markeredgecolor="white",
            markeredgewidth=0.35,
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#6B7280", linewidth=1.0, label="Perfect calibration")
    ax.set_xlabel("Mean positive probability")
    ax.set_ylabel("Empirical positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(color="#D1D5DB", linewidth=0.55, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.01), ncol=1, frameon=False, borderaxespad=0)
    fig.tight_layout(pad=0.5)
    fig.savefig(OUTPUT, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    main()
