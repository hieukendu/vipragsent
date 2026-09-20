"""Regenerate the compact Q4 reliability figure from local calibration records."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "source_data" / "calibration_records.jsonl"
OUTPUT = ROOT / "figure_q4_reliability.png"
TARGET = "ViPragSent (XLM-R-large)"
LABELS = ["implicit_sentiment", "sarcasm", "irony", "idiom_figurative", "code_switching", "mocking"]
SYSTEMS = {
    "Vistral-7B SFT": ("Vistral-7B SFT", "#D55E00", "s"),
    TARGET: ("ViPragSent XLM-R", "#0072B2", "o"),
}


def load_records() -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    with INPUT.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("question") == "Q4":
                grouped[record["system"]].append(record)
    if set(grouped) != set(SYSTEMS) or any(len(records) != 3 for records in grouped.values()):
        raise ValueError("Q4 records must contain exactly three runs for both systems")
    if any(sorted(record["seed"] for record in records) != [21, 22, 23] for records in grouped.values()):
        raise ValueError("Q4 records must use logical run labels 21,22,23")
    return grouped


def aggregate(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    confidence: list[float] = []
    empirical: list[float] = []
    for bin_index in range(10):
        bins = [
            record["reliability_bins"][label][bin_index]
            for record in records
            for label in LABELS
            if record["reliability_bins"][label][bin_index]["count"] > 0
        ]
        if not bins:
            confidence.append(np.nan)
            empirical.append(np.nan)
        else:
            confidence.append(float(np.mean([item["mean_confidence"] for item in bins])))
            empirical.append(float(np.mean([item["empirical_positive_rate"] for item in bins])))
    return np.asarray(confidence), np.asarray(empirical)


def main() -> None:
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
    ax.set_xlabel("Mean confidence")
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
