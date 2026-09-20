"""Draw the compact ViPragSent architecture from paper-local method evidence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT.parent / "evidence" / "method_evidence.json"
OUTPUT_PNG = ROOT / "figure_architecture_schematic.png"
OUTPUT_PDF = ROOT / "figure_architecture_schematic.pdf"
BLUE = "#0072B2"
BLUE_FILL = "#E6F2F8"
INK = "#1F2937"
MUTED = "#6B7280"
GRAY_FILL = "#F3F4F6"
GOLD_FILL = "#FFF7E6"


def read_and_validate() -> dict:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    if evidence["primary_model"]["system"] != "ViPragSent (XLM-R-large)":
        raise ValueError("Unexpected primary model")
    if evidence["primary_model"]["pooling"]["rule"] != "first non-padding encoder state":
        raise ValueError("Unexpected pooling rule")
    if len(evidence["label_schema"]["pragmatic_binary_labels"]) != 6:
        raise ValueError("Expected six pragmatic labels")
    if evidence["label_schema"]["polarity_head"]["classes"] != ["negative", "neutral", "positive"]:
        raise ValueError("Unexpected polarity head")
    if len(evidence["label_schema"]["emotion_head"]["classes"]) != 7:
        raise ValueError("Expected seven emotion classes")
    decoder = evidence["rationale_component"]
    if decoder["implementation_details"]["decoder_layers"] != 2 or not decoder["inference"]["rationale_training"] or decoder["inference"]["rationale_inference"]:
        raise ValueError("Unexpected rationale decoder protocol")
    weighting = evidence["objective"]["uncertainty_weighting"]
    if len(weighting["task_components"]) != 8 or evidence["objective"]["rationale_term"] != "beta * rationale_loss, beta=0.3":
        raise ValueError("Unexpected objective specification")
    return evidence


def box(ax, xy, width, height, text, face, edge=INK, fontsize=10.4, weight="normal"):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.06",
        linewidth=0.9,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        color=INK,
        fontsize=fontsize,
        weight=weight,
        linespacing=1.08,
        clip_on=True,
    )
    return patch


def arrow(ax, start, end, label=None, label_xy=None, color=INK):
    ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": color, "linewidth": 0.9, "shrinkA": 2, "shrinkB": 2})
    if label:
        x, y = label_xy or ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.text(x, y, label, ha="center", va="bottom", color=MUTED, fontsize=7.4)


def main() -> None:
    read_and_validate()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "savefig.dpi": 300,
            # TrueType embedding avoids stippled Type 3 glyphs in the
            # XeLaTeX/Tectonic-embedded PDF while retaining editable text.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(6.9, 3.05))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.05)
    ax.axis("off")

    box(ax, (0.15, 3.0), 1.15, 0.62, "Input\ntokens", GRAY_FILL, fontsize=9.5)
    box(ax, (1.55, 3.0), 1.55, 0.62, "XLM-R-large\nencoder", BLUE_FILL, edge=BLUE, fontsize=9.5, weight="bold")
    box(ax, (3.45, 3.0), 1.65, 0.62, "Hidden\nsequence H", BLUE_FILL, edge=BLUE, fontsize=8.8, weight="bold")
    arrow(ax, (1.3, 3.31), (1.65, 3.31))
    arrow(ax, (3.1, 3.31), (3.45, 3.31))

    box(ax, (3.55, 1.98), 1.55, 0.62, "First non-pad\npooling h*", BLUE_FILL, edge=BLUE, fontsize=9.0)
    box(ax, (5.45, 1.85), 1.70, 0.94, "6 pragmatic\nbinary heads\n+ polarity\n+ emotion", BLUE_FILL, edge=BLUE, fontsize=8.0)
    arrow(ax, (4.27, 3.0), (4.27, 2.60))
    arrow(ax, (5.10, 2.29), (5.45, 2.29))

    box(ax, (3.20, 0.55), 1.90, 0.70, "2-layer Transformer\nrationale decoder", GRAY_FILL, fontsize=8.0)
    box(ax, (5.55, 0.55), 1.55, 0.70, "Teacher-forced\nrationale tokens\n(shifted input)", GRAY_FILL, fontsize=7.4)
    arrow(ax, (4.92, 3.0), (5.18, 2.73))
    arrow(ax, (5.18, 2.73), (5.18, 1.32))
    arrow(ax, (5.18, 1.32), (5.05, 0.90))
    arrow(ax, (5.55, 0.90), (5.10, 0.90))

    box(ax, (7.45, 1.25), 2.25, 1.82, "Training objective\n\n8 uncertainty-weighted\nclassification losses\n+ beta * rationale CE\nbeta = 0.3", GOLD_FILL, edge="#B7791F", fontsize=8.2, weight="bold")
    arrow(ax, (7.15, 2.31), (7.45, 2.31))
    # Route decoder logits below the teacher-forced input box so the two
    # training paths remain visually distinct.
    arrow(ax, (5.00, 0.55), (5.00, 0.28))
    arrow(ax, (5.00, 0.28), (7.25, 0.28))
    arrow(ax, (7.25, 0.28), (7.45, 1.55))

    fig.tight_layout(pad=0.15)
    fig.savefig(OUTPUT_PNG, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    main()
