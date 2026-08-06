from __future__ import annotations

import csv
import json
import os
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .atomic import atomic_write_json
from .constants import PRAGMATIC_LABELS
from .data.annotation import cohen_kappa, krippendorff_alpha_nominal, raw_agreement

SAMPLING_SEED = 20260525
MANUAL_CATEGORIES = (
    "missing broader discourse/context",
    "sarcasm or irony cue failure",
    "idiom/figurative interpretation failure",
    "code-switching or borrowed-token failure",
    "mocking target or stance failure",
    "ordinary sentiment/emotion confusion",
    "ambiguous or insufficient context",
    "probable annotation issue",
    "other",
)
ERROR_ANALYSIS_COLUMNS = [
    "sample_id", "label", "text", "gold_label", "phobert_prediction", "azure_prediction",
    "full_vistral_prediction", "phobert_confidence", "azure_confidence", "full_vistral_confidence",
    "stratum", "selection_reason", "reviewer_1_category", "reviewer_2_category", "adjudicated_category",
]


def _read_prediction_rows(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    systems: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    paths = sorted({path for name in ("test_predictions.jsonl", "predictions.jsonl") for path in root.rglob(name)})
    if not paths:
        raise ValueError("Manual candidates require frozen test prediction files")
    for path in paths:
        system = path.parent.parent.name if path.parent.name.isdigit() else path.parent.name
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            if not sample_id or sample_id in systems[system]:
                raise ValueError(f"Duplicate or missing sample ID in {path}: {sample_id!r}")
            systems[system][sample_id] = row
    return systems


def _prediction(row: Mapping[str, Any], label: str) -> tuple[int, float]:
    predictions = row.get("predictions", row.get("pragmatic", {}))
    probabilities = row.get("probabilities", {})
    value = int(predictions[label])
    probability = float(probabilities.get(label, value))
    return value, probability


def _atomic_csv(path: Path, columns: list[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _candidate_rows(systems: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> list[dict[str, Any]]:
    required = ("phobert_pragmatic_finetune", "azure_pragmatic_8shot", "vipragsent_full_vistral")
    if any(name not in systems for name in required):
        raise ValueError(f"Manual candidates require prediction systems: {required}")
    common_ids = set.intersection(*(set(systems[name]) for name in required))
    rows: list[dict[str, Any]] = []
    for sample_id in sorted(common_ids):
        for label in PRAGMATIC_LABELS:
            full = systems[required[2]][sample_id]
            gold = int(full.get("gold", full.get("targets", {}).get(label, -1)))
            if gold not in (0, 1):
                continue
            predictions = {name: _prediction(systems[name][sample_id], label) for name in required}
            error_systems = [name for name, (prediction, _) in predictions.items() if prediction != gold]
            if not error_systems:
                continue
            disagreement = len(set(prediction for prediction, _ in predictions.values())) > 1
            rows.append({
                "sample_id": sample_id,
                "label": label,
                "text": str(full.get("text", "")),
                "gold_label": gold,
                "phobert_prediction": predictions[required[0]][0],
                "azure_prediction": predictions[required[1]][0],
                "full_vistral_prediction": predictions[required[2]][0],
                "phobert_confidence": predictions[required[0]][1],
                "azure_confidence": predictions[required[1]][1],
                "full_vistral_confidence": predictions[required[2]][1],
                "stratum": "+".join(sorted(error_systems)),
                "selection_reason": "disagreement" if disagreement else "model_error",
            })
    return rows


def _stratified_sample(rows: list[dict[str, Any]], target: int = 400) -> tuple[list[dict[str, Any]], int]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for system in ("phobert_pragmatic_finetune", "azure_pragmatic_8shot", "vipragsent_full_vistral"):
            if system in row["stratum"]:
                grouped[(row["label"], system)].append(row)
    rng = random.Random(SAMPLING_SEED)
    for values in grouped.values():
        values.sort(key=lambda row: (row["selection_reason"] != "disagreement", row["sample_id"]))
        rng.shuffle(values)
        values.sort(key=lambda row: row["selection_reason"] != "disagreement")
    selected: list[dict[str, Any]] = []
    used: set[tuple[str, str]] = set()
    keys = sorted(grouped)
    while len(selected) < target:
        progress = False
        for key in keys:
            while grouped[key] and (grouped[key][0]["sample_id"], key[0]) in used:
                grouped[key].pop(0)
            if not grouped[key]:
                continue
            row = grouped[key].pop(0)
            pair = (row["sample_id"], row["label"])
            if pair in used:
                continue
            used.add(pair)
            selected.append(row)
            progress = True
            if len(selected) == target:
                break
        if not progress:
            break
    return selected, max(0, target - len(selected))


def export_manual_candidates(run_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    systems = _read_prediction_rows(Path(run_root))
    rows, shortfall = _stratified_sample(_candidate_rows(systems))
    columns = ERROR_ANALYSIS_COLUMNS
    output = Path(output_root)
    candidate_path = output / "error_analysis_candidates.csv"
    _atomic_csv(candidate_path, columns, rows)
    template_path = output / "error_analysis_annotation_template.csv"
    _atomic_csv(template_path, ["sample_id", "label", "reviewer", "category", "notes"], [])
    final_path = output / "error_analysis_final.csv"
    _atomic_csv(final_path, columns, [])
    qualitative: list[dict[str, Any]] = []
    for row in rows:
        full_correct = row["full_vistral_prediction"] == row["gold_label"]
        baseline_error = row["phobert_prediction"] != row["gold_label"] or row["azure_prediction"] != row["gold_label"]
        if full_correct and baseline_error and row["label"] in {"sarcasm", "code_switching"}:
            margin = abs(row["full_vistral_confidence"] - 0.5)
            qualitative.append({"sample_id": row["sample_id"], "label": row["label"], "text": row["text"], "full_confidence_margin": margin, "approval": "pending"})
    qualitative.sort(key=lambda row: (-row["full_confidence_margin"], row["sample_id"], row["label"]))
    qualitative_path = output / "qualitative_candidates.jsonl"
    qualitative_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = qualitative_path.with_name(f".{qualitative_path.name}.tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in qualitative), encoding="utf-8", newline="\n")
    os.replace(temporary, qualitative_path)
    qualitative_template = output / "qualitative_approval_template.csv"
    _atomic_csv(qualitative_template, ["sample_id", "reviewer", "approved", "notes"], [])
    (output / "qualitative_final.jsonl").write_text("", encoding="utf-8")
    summary = {"candidate_count": len(rows), "required_count": 400, "shortfall": shortfall, "sampling_seed": SAMPLING_SEED, "qualitative_candidate_count": len(qualitative), "qualitative_required_labels": ["sarcasm", "code_switching"]}
    atomic_write_json(output / "manual_candidate_status.json", summary)
    return {"files": [candidate_path, template_path, final_path, qualitative_path, qualitative_template, output / "qualitative_final.jsonl", output / "manual_candidate_status.json"], **summary}


def compute_reviewer_agreement(rows: Iterable[Mapping[str, Any]], *, field: str = "adjudicated_category") -> dict[str, Any]:
    rows = list(rows)
    first = [row.get("reviewer_1_category", "") for row in rows]
    second = [row.get("reviewer_2_category", "") for row in rows]
    if not rows or not all(first) or not all(second):
        return {"status": "PENDING", "reason": "two independent reviewer columns are incomplete"}
    return {"status": "PASS", "field": field, "n": len(rows), "raw_agreement": raw_agreement(first, second), "cohen_kappa": cohen_kappa(first, second), "krippendorff_alpha_nominal": krippendorff_alpha_nominal(zip(first, second)), "adjudication_complete": all(row.get(field) for row in rows)}
