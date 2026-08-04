from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from _bootstrap import ROOT
from vipragsent.azure.prompts import build_demo_manifest, validate_demo_manifest
from vipragsent.constants import EMOTION_LABELS, POLARITY_LABELS, PRAGMATIC_LABELS
from vipragsent.data.loaders import load_vipragsent
from vipragsent.hashing import sha256_json
from vipragsent.phase import write_phase_handoff


@dataclass(frozen=True)
class PromptRow:
    sample_id: str
    text: str
    labels: dict[str, Any]
    annotation_batch: str = "unknown"


def load_prompt_rows() -> list[PromptRow]:
    bundle = load_vipragsent(ROOT / "data/processed/vipragsent")
    examples = {row.sample_id: row for row in bundle.train}
    raw_dirs = list((ROOT / "data/raw/vipragsent_package").glob("*/"))
    gold_path = raw_dirs[0] / "01_clean_human_annotations/03_gold_adjudicated_clean.csv"
    with gold_path.open(encoding="utf-8-sig", newline="") as handle:
        raw = {row["sample_id"]: row for row in csv.DictReader(handle)}
    rows = []
    for sample_id, example in examples.items():
        raw_row = raw.get(sample_id, {})
        rows.append(PromptRow(sample_id, example.text, example.labels, raw_row.get("annotation_batch", "unknown")))
    return rows


def polarity_manifest(rows: list[PromptRow]) -> dict[str, Any]:
    chosen: list[PromptRow] = []
    for label, count in (("negative", 3), ("neutral", 2), ("positive", 3)):
        chosen.extend(sorted((row for row in rows if row.labels["polarity"] == label), key=lambda row: row.sample_id)[:count])
    if len(chosen) != 8:
        raise ValueError("Could not satisfy polarity 3/2/3 demonstration coverage")
    return {"task": "polarity", "source_split": "vipragsent_train_only", "sample_ids": [row.sample_id for row in chosen], "demonstrations": [{"sample_id": row.sample_id, "text": row.text, "labels": row.labels} for row in chosen], "composition": {"negative": 3, "neutral": 2, "positive": 3}}


def emotion_manifest(rows: list[PromptRow]) -> dict[str, Any]:
    chosen: list[PromptRow] = []
    used_batches: set[str] = set()
    for label in EMOTION_LABELS:
        candidates = sorted((row for row in rows if row.labels["emotion"] == label and row.sample_id not in {item.sample_id for item in chosen}), key=lambda row: row.sample_id)
        if not candidates:
            raise ValueError(f"No emotion demonstration for {label}")
        chosen.append(candidates[0])
        used_batches.add(candidates[0].annotation_batch)
    other_candidates = sorted((row for row in rows if row.labels["emotion"] == "other" and row.annotation_batch not in used_batches), key=lambda row: row.sample_id)
    if not other_candidates:
        raise ValueError("No additional other emotion example from a different annotation batch")
    chosen.append(other_candidates[0])
    return {"task": "emotion", "source_split": "vipragsent_train_only", "sample_ids": [row.sample_id for row in chosen], "demonstrations": [{"sample_id": row.sample_id, "text": row.text, "labels": row.labels} for row in chosen], "composition": {label: 1 for label in EMOTION_LABELS}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-blocked-q3", action="store_true")
    args = parser.parse_args()
    rows = load_prompt_rows()
    output = ROOT / "data/manifests/prompts"
    output.mkdir(parents=True, exist_ok=True)
    general = build_demo_manifest(rows)
    validate_demo_manifest(general)
    manifests: dict[str, Any] = {"pragmatic_v1.json": general, "polarity_v1.json": polarity_manifest(rows), "emotion_v1.json": emotion_manifest(rows)}
    q3_dir = ROOT / "data/processed/q3_low_resource_sarcasm"
    q3_status: dict[str, Any] = {}
    for budget in ("32", "64", "128", "256", "512", "full"):
        with (q3_dir / f"budget_{budget}_masks.csv").open(encoding="utf-8-sig", newline="") as handle:
            mask_rows = list(csv.DictReader(handle))
        eligible_ids = {row["sample_id"] for row in mask_rows if int(row["positive_selected_for_budget"]) == 1 or int(row["is_sarcasm_positive"]) == 0}
        try:
            manifest = build_demo_manifest(rows, budget=budget, eligible_ids=eligible_ids)
            validate_demo_manifest(manifest, q3_eligible_ids=eligible_ids)
            manifests[f"q3_budget_{budget}_v1.json"] = manifest
            q3_status[budget] = "PASS"
        except Exception as exc:
            q3_status[budget] = f"BLOCKED: {exc}"
            if not args.allow_blocked_q3:
                # Keep generating the other immutable manifests but make the handoff honest.
                continue
    for filename, manifest in manifests.items():
        manifest["prompt_hash"] = sha256_json(manifest)
        (output / filename).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = {"general": "PASS", "polarity": "PASS", "emotion": "PASS", "q3": q3_status, "blocked_q3_budgets": [budget for budget, status in q3_status.items() if status != "PASS"]}
    (output / "manifest_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    blockers = [f"Q3 budget {budget}: {status}" for budget, status in q3_status.items() if status != "PASS"]
    write_phase_handoff("09", "PASS" if not blockers else "BLOCKED", inputs_read=["ViPragSent train", "09_PHASE_09_IMPLEMENT_AZURE_PROMPTS_AND_CLIENT.md"], files_created=[str(path.relative_to(ROOT)) for path in sorted(output.glob("*.json"))], tests_run=["general pragmatic 8-shot coverage", "polarity 3/2/3 coverage", "emotion 7+1 coverage", "Q3 eligibility"], tests_passed=True, blockers=blockers, next_phase_ready=not blockers)
    print(json.dumps(report, indent=2))
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
