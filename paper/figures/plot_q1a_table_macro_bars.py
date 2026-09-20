"""Plot the expanded Q1a Table 1 as a macro-pragmatic F1 bar chart.

The input is the canonical Q1a row inventory. This script performs only the
recorded conversion from [0, 1] to percentage points and preserves the row
status/seed coverage in the visual annotations; it does not re-estimate any
metric or fill missing uncertainty.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parent
INPUT = ROOT.parent / "revision_q1a_extra" / "q1a_extra_canonical_rows.csv"
OUTPUT_PNG = ROOT / "figure_q1a_table_macro_bars.png"
OUTPUT_PDF = ROOT / "figure_q1a_table_macro_bars.pdf"
OUTPUT_METADATA = ROOT / "figure_q1a_table_macro_bars.metadata.json"

REQUIRED_COLUMNS = {
    "system_id",
    "display_name",
    "status",
    "seed_count",
    "expected_seed_count",
    "macro_pragmatic_f1_mean",
    "macro_pragmatic_f1_sample_sd",
}
PRIMARY_ID = "vipragsent_xlmr_large"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_rows() -> list[dict[str, str]]:
    with INPUT.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Q1a canonical CSV has no header")
        missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames))
        if missing:
            raise ValueError(f"Missing Q1a canonical columns: {missing}")
        rows = list(reader)
    if len(rows) != 12:
        raise ValueError(f"Expected the expanded 12-row Q1a inventory, found {len(rows)}")
    if sum(row["system_id"] == PRIMARY_ID for row in rows) != 1:
        raise ValueError("Expected exactly one XLM-R-large primary row")
    for row in rows:
        int(row["seed_count"])
        int(row["expected_seed_count"])
        float(row["macro_pragmatic_f1_mean"])
        sd = row["macro_pragmatic_f1_sample_sd"].strip()
        if sd:
            float(sd)
    primary = next(row for row in rows if row["system_id"] == PRIMARY_ID)
    return [primary] + [row for row in rows if row["system_id"] != PRIMARY_ID]


def row_style(row: dict[str, str]) -> tuple[str, str]:
    if row["system_id"] == PRIMARY_ID:
        return "#0072B2", "primary"
    if row["status"].startswith("PROVISIONAL"):
        return "#D55E00", "provisional"
    if int(row["seed_count"]) == 1:
        return "#E69F00", "one-shot"
    return "#8A8F98", "multi-seed"


def main() -> None:
    rows = load_rows()
    values = np.array([100.0 * float(row["macro_pragmatic_f1_mean"]) for row in rows])
    sds = np.array(
        [
            100.0 * float(row["macro_pragmatic_f1_sample_sd"])
            if row["macro_pragmatic_f1_sample_sd"].strip()
            else np.nan
            for row in rows
        ]
    )
    y = np.arange(len(rows))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.3,
            "axes.labelsize": 9.3,
            "xtick.labelsize": 8.1,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.6,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots(figsize=(8.1, 4.75))
    for row, yi, mean, sd in zip(rows, y, values, sds):
        color, _ = row_style(row)
        ax.barh(
            yi,
            mean,
            height=0.66,
            color=color,
            edgecolor="#374151",
            linewidth=0.35,
            hatch="//" if row["status"].startswith("PROVISIONAL") else None,
            zorder=2,
        )
        if not np.isnan(sd):
            ax.errorbar(
                mean,
                yi,
                xerr=sd,
                fmt="none",
                ecolor="#111827",
                elinewidth=0.9,
                capsize=2.2,
                capthick=0.9,
                zorder=3,
            )
        seed_count = int(row["seed_count"])
        if np.isnan(sd):
            annotation = f"{mean:.2f} (n={seed_count})"
        elif row["status"].startswith("PROVISIONAL"):
            annotation = f"{mean:.2f} ± {sd:.2f} (n={seed_count}; provisional)"
        else:
            annotation = f"{mean:.2f} ± {sd:.2f}"
        ax.annotate(
            annotation,
            (mean + (0 if np.isnan(sd) else sd), yi),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=7.4,
            fontweight="bold" if row["system_id"] == PRIMARY_ID else "normal",
            color="#111827",
            clip_on=False,
        )

    labels = [
        "ViPragSent (ours, XLM-R-large)" if row["system_id"] == PRIMARY_ID else row["display_name"]
        for row in rows
    ]
    ax.set_yticks(y, labels)
    ax.set_xlabel("Macro-pragmatic F1 (percentage points)")
    ax.set_xlim(0, 109)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_ylim(-0.85, len(rows) - 0.15)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#D1D5DB", linewidth=0.65, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.text(
        0.0,
        -0.16,
        "Bars show saved means; whiskers show sample SD when available. The hatched CoT row is provisional at n=2; one-shot rows have no SD.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color="#374151",
    )
    legend_handles = [
        Patch(facecolor="#0072B2", edgecolor="#374151", label="Primary"),
        Patch(facecolor="#8A8F98", edgecolor="#374151", label="Multi-seed row"),
        Patch(facecolor="#E69F00", edgecolor="#374151", label="n=1 record"),
        Patch(facecolor="#D55E00", edgecolor="#374151", hatch="//", label="Provisional n=2"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0, 1.015),
        ncol=4,
        frameon=False,
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0, 0.065, 1, 0.94))
    fig.savefig(OUTPUT_PNG, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "figure": "figure_q1a_table_macro_bars",
        "question": "Q1a",
        "input": str(INPUT),
        "input_sha256": sha256(INPUT),
        "row_count": len(rows),
        "unit_conversion": "mean and sample SD multiplied by 100 for percentage-point display",
        "rows": [
            {
                "system_id": row["system_id"],
                "display_name": row["display_name"],
                "status": row["status"],
                "seed_count": int(row["seed_count"]),
                "expected_seed_count": int(row["expected_seed_count"]),
                "macro_pragmatic_f1_mean": float(row["macro_pragmatic_f1_mean"]),
                "macro_pragmatic_f1_sample_sd": row["macro_pragmatic_f1_sample_sd"].strip() or None,
            }
            for row in rows
        ],
        "claim_boundary": "The chart represents the expanded Table 1 inventory; primary baseline dominance is limited to complete comparable three-seed baseline families.",
    }
    OUTPUT_METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote Q1a bar chart for {len(rows)} canonical rows from {INPUT}")


if __name__ == "__main__":
    main()
