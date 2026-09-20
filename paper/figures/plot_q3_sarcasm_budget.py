"""Regenerate the compact Q3 sarcasm-budget figure from the local artifact table."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "source_data" / "table_q3_low_resource.csv"
OUTPUT = ROOT / "figure_q3_sarcasm_budget.png"

TARGET = "ViPragSent (XLM-R-large)"
SYSTEMS = {
    "PhoBERT (fine-tune)": ("PhoBERT FT", "#6B7280", "o"),
    "Vistral-7B SFT": ("Vistral-7B SFT", "#D55E00", "s"),
    TARGET: ("ViPragSent XLM-R", "#0072B2", "o"),
}
BUDGET_ORDER = ["32", "64", "128", "256", "512", "full"]


def load_rows() -> list[dict[str, str]]:
    with INPUT.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if {row["system"] for row in rows} != set(SYSTEMS):
        raise ValueError("Unexpected Q3 system set")
    for row in rows:
        if row["n"] == "0":
            if row["system"] != "PhoBERT (fine-tune)" or row["seeds"] or any(
                row[field]
                for field in (
                    "macro_pragmatic_f1_mean",
                    "macro_pragmatic_f1_sd",
                    "sarcasm_f1_mean",
                    "sarcasm_f1_sd",
                )
            ):
                raise ValueError("Only explicitly missing PhoBERT rows may use n=0")
            continue
        if int(row["n"]) != 3 or row["seeds"] != "21,22,23":
            raise ValueError("Complete Q3 rows must report n=3 and logical runs 21,22,23")
    return rows


def main() -> None:
    rows = load_rows()
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
    positions = np.arange(len(BUDGET_ORDER))
    for system, (label, color, marker) in SYSTEMS.items():
        by_budget = {row["budget"]: row for row in rows if row["system"] == system}
        x = positions
        mean = np.array(
            [
                100 * float(by_budget[budget]["sarcasm_f1_mean"])
                if budget in by_budget and by_budget[budget]["n"] == "3"
                else np.nan
                for budget in BUDGET_ORDER
            ],
            dtype=float,
        )
        sd = np.array(
            [
                100 * float(by_budget[budget]["sarcasm_f1_sd"])
                if budget in by_budget and by_budget[budget]["n"] == "3"
                else np.nan
                for budget in BUDGET_ORDER
            ],
            dtype=float,
        )
        ax.errorbar(
            x,
            mean,
            yerr=sd,
            label=label,
            color=color,
            marker=marker,
            linewidth=1.35,
            markersize=3.8,
            capsize=1.8,
            capthick=0.8,
            elinewidth=0.8,
            markeredgecolor="white",
            markeredgewidth=0.35,
        )
    ax.set_xlabel("Labelled sarcasm budget")
    ax.set_ylabel("Sarcasm F1 (%)")
    ax.set_xticks(positions, BUDGET_ORDER)
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.2, len(BUDGET_ORDER) - 0.8)
    ax.grid(axis="y", color="#D1D5DB", linewidth=0.55, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.01), ncol=1, frameon=False, borderaxespad=0)
    fig.tight_layout(pad=0.5)
    fig.savefig(OUTPUT, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    main()
