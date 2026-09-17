#!/usr/bin/env python3
"""Build an HF-only, paper-shaped ViPragSent comparison package.

The script intentionally uses the authenticated Hugging Face HTTP API and the
already audited HF tree.  It does not import or execute the repository's
experiment code and it never downloads model-weight payloads.  The primary
scope is ViPragSent with XLM-R-large; ordinary baseline systems are retained
for comparison, while ViPragSent variants with another backbone are recorded
as discovered-but-excluded evidence.

Outputs are written below reports/hf_vipragsent_naacl_comparison_2026-09-15.
The result is ANALYZED rather than VERIFIED because this script does not rerun
training or inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import statistics
import sys
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from hf_vipragsent_audit import (  # noqa: E402
    content_url,
    curl_get,
    raw_path_for,
    read_token,
)

DEFAULT_OUT = Path("reports") / "hf_vipragsent_naacl_comparison_2026-09-15"
RECHECK_DIR = Path("reports") / "hf_vipragsent_remote_recheck_2026-09-15"
OLD_AUDIT_DIR = Path("reports") / "hf_vipragsent_remote_audit_2026-09-15"
TREE_PATH = RECHECK_DIR / "tree_manifest.jsonl"
SEEDS = (21, 22, 23)
DATES = {seed: f"202605{seed:02d}" for seed in SEEDS}
HEADS = (
    "implicit_sentiment",
    "sarcasm",
    "irony",
    "idiom_figurative",
    "code_switching",
    "mocking",
)
HEAD_LABELS = {
    "implicit_sentiment": "Implicit",
    "sarcasm": "Sarcasm",
    "irony": "Irony",
    "idiom_figurative": "Idiom/figurative",
    "code_switching": "Code-switching",
    "mocking": "Mocking",
}
Q1A_ORDER = (
    "PhoBERT (single-task)",
    "PhoBERT (fine-tune)",
    "XLM-R-large (baseline)",
    "Sailor-7B SFT",
    "Vistral-7B SFT",
    "ViPragSent (XLM-R-large)",
)
Q1A_PAPER_SCHEMA_ORDER = (
    "PhoBERT (single-task)",
    "PhoBERT (fine-tune)",
    "XLM-R-large (baseline)",
    "Sailor-7B SFT",
    "Vistral-7B SFT",
    "GPT-4.1-mini zero-shot",
    "GPT-4.1-mini 8-shot",
    "ViPragSent-no-aux (Vistral)",
    "ViPragSent-CoT-only (Vistral)",
    "ViPragSent-explanation-only (Vistral)",
    "ViPragSent full (Vistral)",
    "ViPragSent (XLM-R-large)",
)
Q1A_PAPER_SCHEMA_BACKBONES = {
    "PhoBERT (single-task)": "PhoBERT-base",
    "PhoBERT (fine-tune)": "PhoBERT-base",
    "XLM-R-large (baseline)": "XLM-R-large",
    "Sailor-7B SFT": "Sailor-7B",
    "Vistral-7B SFT": "Vistral-7B",
    "GPT-4.1-mini zero-shot": "GPT-4.1-mini",
    "GPT-4.1-mini 8-shot": "GPT-4.1-mini",
    "ViPragSent-no-aux (Vistral)": "Vistral-7B",
    "ViPragSent-CoT-only (Vistral)": "Vistral-7B",
    "ViPragSent-explanation-only (Vistral)": "Vistral-7B",
    "ViPragSent full (Vistral)": "Vistral-7B",
    "ViPragSent (XLM-R-large)": "XLM-R-large",
}
Q3_SYSTEM_ORDER = (
    "PhoBERT (fine-tune)",
    "Vistral-7B SFT",
    "ViPragSent (XLM-R-large)",
)
Q3_BUDGETS = ("32", "64", "128", "256", "512", "full")
FLOAT_KEYS = {
    "macro_pragmatic_f1",
    "macro_pragmatic_ece",
    "polarity_dev_ece",
    "ece",
    "ece_macro",
    "vsfc_macro_f1",
    "vsmec_macro_f1",
    "aivivn_macro_f1",
    "ord_f1",
    "successful_gpu_hours",
    "azure_cost_usd",
}
RUN_SEGMENT_PREFIXES = ("q1a_", "q1b_", "q2_", "q3_", "q4_", "xlmr_followup_")
_RUN_ENTRY_INDEX_CACHE: dict[int, dict[str, list[dict[str, Any]]]] = {}


def utc_now() -> str:
    from datetime import datetime

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def jsonl_dump(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")


def csv_dump(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fields: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fields.append(key)
        fieldnames = fields
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_key(entry: dict[str, Any]) -> str:
    return f"{entry.get('repo_type', '')}:{entry.get('repo_id', '')}:{entry.get('path', '')}"


def seed_from_run(run_id: str) -> int | None:
    match = re.search(r"202605(21|22|23)(?:$|[^0-9])", run_id)
    return int(match.group(1)) if match else None


def numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, str):
        try:
            result = float(value)
        except ValueError:
            return None
        return result if math.isfinite(result) else None
    return None


def path_parts(path: str) -> list[str]:
    return [part for part in path.split("/") if part]


def has_run_segment(path: str, run_id: str) -> bool:
    return run_id in path_parts(path)


def load_tree(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if not rows:
        raise RuntimeError(f"fresh HF tree manifest is missing or empty: {path}")
    return rows


def run_entry_index(entries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    cache_key = id(entries)
    cached = _RUN_ENTRY_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        if entry.get("type") != "file":
            continue
        parts = path_parts(str(entry.get("path", "")))
        for part in parts:
            if part.startswith(RUN_SEGMENT_PREFIXES):
                index[part].append(entry)
    result = dict(index)
    _RUN_ENTRY_INDEX_CACHE[cache_key] = result
    return result


def pick_entry(
    entries: list[dict[str, Any]],
    run_id: str,
    suffix_parts: tuple[str, ...],
    preferred: tuple[str, ...] = (),
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for entry in run_entry_index(entries).get(run_id, []):
        path = str(entry.get("path", ""))
        parts = path_parts(path)
        try:
            run_index = parts.index(run_id)
        except ValueError:
            continue
        if tuple(parts[run_index + 1 :]) != suffix_parts:
            continue
        if "components" in parts[run_index + 1 :]:
            continue
        candidates.append(entry)
    if not candidates:
        return None

    def score(entry: dict[str, Any]) -> tuple[int, int, int, str]:
        path = str(entry.get("path", ""))
        haystack = f"{path} {entry.get('repo_id', '')}"
        preference = sum((len(preferred) - index) * 100 for index, term in enumerate(preferred) if term in haystack)
        # Prefer a complete sizeable metric file over a 123-byte NOT_STARTED
        # placeholder when both are present.
        size = int(entry.get("size") or 0)
        placeholder_penalty = -1000 if size < 500 else 0
        return (preference + placeholder_penalty, size, 0 if "overflow" in str(entry.get("repo_id", "")) else 1, path)

    return sorted(candidates, key=score, reverse=True)[0]


def pick_terminal_metrics_entry(
    entries: list[dict[str, Any]],
    run_id: str,
    preferred: tuple[str, ...] = (),
) -> dict[str, Any] | None:
    """Select a run-level metrics.json receipt, including nested run receipts."""
    candidates: list[dict[str, Any]] = []
    for entry in run_entry_index(entries).get(run_id, []):
        path = str(entry.get("path", ""))
        parts = path_parts(path)
        try:
            run_index = parts.index(run_id)
        except ValueError:
            continue
        suffix = parts[run_index + 1 :]
        if not suffix or suffix[-1] != "metrics.json" or "components" in suffix:
            continue
        candidates.append(entry)
    if not candidates:
        return None

    def score(entry: dict[str, Any]) -> tuple[int, int, str]:
        path = str(entry.get("path", ""))
        haystack = f"{path} {entry.get('repo_id', '')}"
        preference = sum((len(preferred) - index) * 100 for index, term in enumerate(preferred) if term in haystack)
        return (preference, -int(entry.get("size") or 0), path)

    return sorted(candidates, key=score, reverse=True)[0]


def make_spec(
    *,
    artifact_id: str,
    kind: str,
    question: str,
    system: str,
    run_id: str,
    scope: str,
    backbone: str,
    entry: dict[str, Any] | None,
    note: str = "",
) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "kind": kind,
        "question": question,
        "system": system,
        "run_id": run_id,
        "seed": seed_from_run(run_id),
        "scope": scope,
        "backbone": backbone,
        "entry": entry,
        "note": note,
    }


def build_specs(entries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    specs: list[dict[str, Any]] = []
    coverage_missing: list[dict[str, Any]] = []

    def add_metric(
        question: str,
        system: str,
        run_id: str,
        scope: str,
        backbone: str,
        preferred: tuple[str, ...] = (),
        note: str = "",
    ) -> None:
        entry = pick_entry(entries, run_id, ("metrics", "test_metrics.json"), preferred)
        specs.append(
            make_spec(
                artifact_id=f"metric::{run_id}",
                kind="metric",
                question=question,
                system=system,
                run_id=run_id,
                scope=scope,
                backbone=backbone,
                entry=entry,
                note=note,
            )
        )

    def add_missing(question: str, system: str, run_id: str, scope: str, backbone: str, note: str) -> None:
        coverage_missing.append(
            {
                "question": question,
                "system": system,
                "run_id": run_id,
                "scope": scope,
                "backbone": backbone,
                "requested_seed": seed_from_run(run_id),
                "note": note,
            }
        )

    # Q1a: standard paper baselines.  The server_20260808 copies contain the
    # complete top-level test metrics for the comparable three seeds.
    q1a_standard = (
        ("PhoBERT (single-task)", "q1a_phobert_pragmatic_single_task", "PhoBERT-base"),
        ("PhoBERT (fine-tune)", "q1a_phobert_pragmatic_finetune", "PhoBERT-base"),
        ("XLM-R-large (baseline)", "q1a_xlmr_pragmatic_finetune", "XLM-R-large"),
        ("Sailor-7B SFT", "q1a_sailor_pragmatic_sft", "Sailor-7B"),
        ("Vistral-7B SFT", "q1a_vistral_pragmatic_sft", "Vistral-7B"),
    )
    for system, base, backbone in q1a_standard:
        for seed in SEEDS:
            run_id = f"{base}_{DATES[seed]}"
            add_metric("Q1a", system, run_id, "baseline", backbone, ("server_20260808",))

    # The in-scope ViPragSent XLM-R-large run has a long, explicit run name.
    full_xlmr_base = "q1a_vipragsent_full_XLM_R_large_optimization_pragmatic_warmup020_irony105_implicit101_sarcasm101_code104_v37"
    for seed in SEEDS:
        run_id = f"{full_xlmr_base}_{DATES[seed]}"
        add_metric("Q1a", "ViPragSent (XLM-R-large)", run_id, "primary", "XLM-R-large", ("vipragsent-v8-local",))

    # Discovered but excluded ViPragSent variants: the original scope says to
    # keep ViPragSent/XLM-R-large and exclude ViPragSent with another backbone.
    for system, base, preferred in (
        ("ViPragSent-no-aux (Vistral)", "q1a_vipragsent_no_auxiliary_vistral", ("server_20260808",)),
        ("ViPragSent full (Vistral)", "q1a_vipragsent_full_vistral", ("overflow-017", "overflow-016")),
    ):
        for seed in SEEDS:
            run_id = f"{base}_{DATES[seed]}"
            add_metric(
                "Q1a",
                system,
                run_id,
                "excluded_non_xlmr_vipragsent",
                "Vistral-7B",
                preferred,
                "Discovered in HF but excluded from primary scope because this ViPragSent variant is not XLM-R-large.",
            )

    # The paper-schema CoT/explanation rows are not ordinary XLM-R baselines.
    # They are nevertheless retained as discovered evidence, with their
    # direct reasoning-metric files, and remain excluded from the primary
    # comparison because their backbone is Vistral.  The CoT clean rerun has
    # only seeds 21 and 23 in the current HF tree; explanation-only has all
    # three requested seeds.
    for seed in SEEDS:
        run_id = f"q1a_cot_only_vistral_clean_rerun_003__seed_{DATES[seed]}"
        entry = pick_entry(
            entries,
            run_id,
            ("metrics", "test_reasoning_metrics.json"),
            ("overflow-003", "overflow-024", "overflow-006"),
        )
        specs.append(
            make_spec(
                artifact_id=f"metric::{run_id}",
                kind="metric",
                question="Q1a",
                system="ViPragSent-CoT-only (Vistral)",
                run_id=run_id,
                scope="excluded_non_xlmr_vipragsent",
                backbone="Vistral-7B",
                entry=entry,
                note="Direct reasoning metric uses the HF all-zero-fallback metric; excluded because the ViPragSent backbone is Vistral, not XLM-R-large.",
            )
        )
    for seed in SEEDS:
        run_id = f"q1a_explanation_only_vistral_{DATES[seed]}"
        entry = pick_entry(
            entries,
            run_id,
            ("metrics", "test_reasoning_metrics.json"),
            ("overflow-003", "overflow-004", "overflow-010", "overflow-014"),
        )
        specs.append(
            make_spec(
                artifact_id=f"metric::{run_id}",
                kind="metric",
                question="Q1a",
                system="ViPragSent-explanation-only (Vistral)",
                run_id=run_id,
                scope="excluded_non_xlmr_vipragsent",
                backbone="Vistral-7B",
                entry=entry,
                note="Direct reasoning metric uses the HF all-zero-fallback metric; excluded because the ViPragSent backbone is Vistral, not XLM-R-large.",
            )
        )

    # GPT baseline receipts are present, but both are explicitly NOT_STARTED
    # and contain no comparable test metric.  Keep this missingness visible.
    for system, base, note in (
        ("GPT-4.1-mini zero-shot", "q1a_azure_gpt41_mini_zeroshot", "HF status is NOT_STARTED rather than a completed comparable test metric pack."),
        ("GPT-4.1-mini 8-shot", "q1a_azure_gpt41_mini_8shot", "HF status is NOT_STARTED rather than a completed comparable test metric pack."),
    ):
        add_missing("Q1a", system, base, "missing_or_incomplete", "GPT-4.1-mini", note)

    # Q2: XLM-R-large ViPragSent ablations.
    q2_variants = (
        ("Full", "xlmr_followup_q2_full"),
        ("− emotion auxiliary", "xlmr_followup_q2_no_emotion_auxiliary"),
        ("− polarity auxiliary", "xlmr_followup_q2_no_polarity_auxiliary"),
        ("− explanation/CoT auxiliary", "xlmr_followup_q2_no_rationale"),
        ("− task-uncertainty weighting", "xlmr_followup_q2_no_uncertainty_weighting"),
        ("− multi-task bundle", "xlmr_followup_q2_no_multitask"),
    )
    for label, base in q2_variants:
        for seed in SEEDS:
            add_metric("Q2", f"ViPragSent XLM-R: {label}", f"{base}_{DATES[seed]}", "primary", "XLM-R-large", ("xlmr-q1b-q4-followup",))

    # Q3: primary XLM-R curve plus two directly comparable baseline curves.
    q3_specs = (
        ("ViPragSent (XLM-R-large)", "xlmr_followup_q3_full_{budget}", "primary", "XLM-R-large", ("xlmr-q1b-q4-followup",)),
        ("PhoBERT (fine-tune)", "q3_phobert_pragmatic_finetune_{budget}", "baseline", "PhoBERT-base", ("vipragsent-v7-6693",)),
        ("Vistral-7B SFT", "q3_vistral_pragmatic_sft_{budget}", "baseline", "Vistral-7B", ("vipragsent-v7-6693", "vipragsent-v8-local")),
    )
    for system, template, scope, backbone, preferred in q3_specs:
        for budget in Q3_BUDGETS:
            for seed in SEEDS:
                base = template.format(budget=budget)
                add_metric("Q3", f"{system} [{budget}]", f"{base}_{DATES[seed]}", scope, backbone, preferred)

    # Q4 calibration and its source training histories.
    q4_systems = (
        ("ViPragSent (XLM-R-large)", "xlmr_followup_q4_full", "primary", "XLM-R-large", ("xlmr-q1b-q4-followup",), "paper_artifacts/q4_pragmatic_calibration_per_seed.json", "source/training_history.json"),
        ("Vistral-7B SFT", "q4_vistral_pragmatic_sft", "baseline", "Vistral-7B", ("vipragsent-v8-local",), "paper_artifacts/q4_pragmatic_calibration_per_seed.json", "source/history.json"),
    )
    for system, base, scope, backbone, preferred, cal_suffix, history_suffix in q4_systems:
        for seed in SEEDS:
            run_id = f"{base}_{DATES[seed]}"
            cal_entry = pick_entry(entries, run_id, tuple(cal_suffix.split("/")), preferred)
            specs.append(make_spec(artifact_id=f"calibration::{run_id}", kind="calibration", question="Q4", system=system, run_id=run_id, scope=scope, backbone=backbone, entry=cal_entry))
            history_entry = pick_entry(entries, run_id, tuple(history_suffix.split("/")), preferred)
            specs.append(make_spec(artifact_id=f"history::{run_id}", kind="history", question="Q4", system=system, run_id=run_id, scope=scope, backbone=backbone, entry=history_entry))

    # Q1b retention metrics are available as compact structured files for
    # Sailor, Vistral and the in-scope XLM-R ViPragSent checkpoint.
    q1b_systems = (
        ("Sailor-7B SFT", "q1b_sailor_multitask_8head", "baseline", "Sailor-7B", ("vipragsent-v7-6693",)),
        ("Vistral-7B SFT", "q1b_vistral_multitask_8head", "baseline", "Vistral-7B", ("vipragsent-v7-6693",)),
        ("ViPragSent (XLM-R-large)", "xlmr_followup_q1b_vipragsent_full", "primary", "XLM-R-large", ("xlmr-q1b-q4-followup",)),
    )
    for system, base, scope, backbone, preferred in q1b_systems:
        for seed in SEEDS:
            run_id = f"{base}_{DATES[seed]}"
            entry = pick_entry(entries, run_id, ("metrics", "external_retention_metrics.json"), preferred)
            specs.append(make_spec(artifact_id=f"external::{run_id}", kind="external", question="Q1b", system=system, run_id=run_id, scope=scope, backbone=backbone, entry=entry))

    # Prediction files for an HF-native confusion matrix over all three
    # primary Q1a seeds.  They are not weights and are bounded at ~2.5 MB/run.
    for seed in SEEDS:
        run_id = f"{full_xlmr_base}_{DATES[seed]}"
        entry = pick_entry(entries, run_id, ("predictions", "test_predictions.jsonl"), ("vipragsent-v8-local",))
        specs.append(make_spec(artifact_id=f"prediction::{run_id}", kind="prediction", question="Q1a", system="ViPragSent (XLM-R-large)", run_id=run_id, scope="primary", backbone="XLM-R-large", entry=entry))

    # Preserve the exact GPT baseline receipt files even though they are not
    # completed comparable metrics.  This makes the missingness claim
    # auditable instead of relying only on the coverage note.
    for system, run_id, preferred in (
        ("GPT-4.1-mini zero-shot", "q1a_azure_gpt41_mini_zeroshot", ("overflow-022",)),
        ("GPT-4.1-mini 8-shot", "q1a_azure_gpt41_mini_8shot", ("compact-20260820",)),
    ):
        status_entry = pick_terminal_metrics_entry(entries, run_id, preferred)
        specs.append(
            make_spec(
                artifact_id=f"status::{run_id}",
                kind="status",
                question="Q1a",
                system=system,
                run_id=run_id,
                scope="missing_or_incomplete",
                backbone="GPT-4.1-mini",
                entry=status_entry,
                note="Receipt is preserved for audit; no completed comparable test metric is available.",
            )
        )

    # Resource provenance is kept separate from test metrics.  Q1a is used
    # for the paper-style cost table; Q2 resource files also support the
    # normalized-cost column in the ablation table.
    resource_specs: list[dict[str, Any]] = []
    review_specs: list[dict[str, Any]] = []
    for base_spec in [spec for spec in specs if spec["kind"] == "metric" and spec["question"] in ("Q1a", "Q2")]:
        preferred = (
            "server_20260808",
            "vipragsent-v8-local",
            "vipragsent-v7-6693",
            "xlmr-q1b-q4-followup",
        )
        resource_entry = pick_entry(entries, base_spec["run_id"], ("training", "resource_usage.json"), preferred)
        resource_specs.append(
            make_spec(
                artifact_id=f"resource::{base_spec['run_id']}",
                kind="resource",
                question=base_spec["question"],
                system=base_spec["system"],
                run_id=base_spec["run_id"],
                scope=base_spec["scope"],
                backbone=base_spec["backbone"],
                entry=resource_entry,
            )
        )
        if base_spec["question"] == "Q1a":
            review_entry = pick_entry(entries, base_spec["run_id"], ("review_summary.json",), preferred)
            review_specs.append(
                make_spec(
                    artifact_id=f"review::{base_spec['run_id']}",
                    kind="review",
                    question=base_spec["question"],
                    system=base_spec["system"],
                    run_id=base_spec["run_id"],
                    scope=base_spec["scope"],
                    backbone=base_spec["backbone"],
                    entry=review_entry,
                )
            )
    specs.extend(resource_specs)
    specs.extend(review_specs)

    return specs, coverage_missing


def fetch_sources(specs: list[dict[str, Any]], out: Path, env_path: Path, workers: int = 4) -> dict[str, dict[str, Any]]:
    token = read_token(env_path)
    unique: dict[str, dict[str, Any]] = {}
    for spec in specs:
        entry = spec.get("entry")
        if entry:
            unique[source_key(entry)] = entry

    def fetch(entry: dict[str, Any]) -> dict[str, Any]:
        key = source_key(entry)
        destination = raw_path_for(out, entry)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            if destination.exists():
                body = destination.read_bytes()
                status = "cached"
            else:
                headers, body = curl_get(content_url(entry), token, timeout=600, retries=5)
                destination.write_bytes(body)
                status = "fetched"
            digest = sha256_bytes(body)
            return {
                "source_key": key,
                "repo_id": entry.get("repo_id"),
                "repo_type": entry.get("repo_type"),
                "source_path": entry.get("path"),
                "tree_oid": entry.get("oid"),
                "tree_size": entry.get("size"),
                "fetch_status": status,
                "fetch_bytes": len(body),
                "fetch_sha256": digest,
                "raw_path": str(destination.relative_to(out)),
                "url": content_url(entry),
            }
        except Exception as exc:
            return {
                "source_key": key,
                "repo_id": entry.get("repo_id"),
                "repo_type": entry.get("repo_type"),
                "source_path": entry.get("path"),
                "tree_oid": entry.get("oid"),
                "tree_size": entry.get("size"),
                "fetch_status": "error",
                "fetch_error": str(exc),
                "raw_path": str(destination.relative_to(out)),
                "url": content_url(entry),
            }

    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, entry): key for key, entry in unique.items()}
        for future in as_completed(futures):
            row = future.result()
            results[row["source_key"]] = row
            print(f"[fetch] {row.get('fetch_status')} {row.get('source_path')}", flush=True)
    return results


def bytes_for(spec: dict[str, Any], source_rows: dict[str, dict[str, Any]], out: Path) -> bytes | None:
    entry = spec.get("entry")
    if not entry:
        return None
    row = source_rows.get(source_key(entry))
    if not row or row.get("fetch_status") == "error":
        return None
    path = out / row["raw_path"]
    return path.read_bytes() if path.exists() else None


def json_for(spec: dict[str, Any], source_rows: dict[str, dict[str, Any]], out: Path) -> Any | None:
    raw = bytes_for(spec, source_rows, out)
    if raw is None:
        return None
    try:
        return json.loads(raw.decode("utf-8-sig", errors="replace"))
    except Exception:
        return None


def get_test_object(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    if isinstance(value.get("test"), dict):
        return value["test"]
    return value


def extract_metric(spec: dict[str, Any], data: Any) -> dict[str, Any]:
    obj = get_test_object(data)
    row: dict[str, Any] = {
        "question": spec["question"],
        "system": spec["system"],
        "scope": spec["scope"],
        "backbone": spec["backbone"],
        "run_id": spec["run_id"],
        "seed": spec.get("seed"),
        "artifact_id": spec["artifact_id"],
        "status": "METRIC_FOUND",
        "note": spec.get("note", ""),
        "metric_name": obj.get("primary_metric_name") if isinstance(obj, dict) else None,
        "inference_output_source": obj.get("inference_output_source") if isinstance(obj, dict) else None,
        "truncation_rate": numeric(obj.get("truncation_rate")) if isinstance(obj, dict) else None,
    }
    per_label = obj.get("per_label_f1") if isinstance(obj, dict) else None
    if not isinstance(per_label, dict) and isinstance(obj, dict):
        per_label = obj.get("primary_per_label_f1")
    if not isinstance(per_label, dict):
        per_label = {}
    for head in HEADS:
        row[head] = numeric(per_label.get(head))
    row["macro_pragmatic_f1"] = numeric(obj.get("macro_pragmatic_f1"))
    if row["macro_pragmatic_f1"] is None:
        row["macro_pragmatic_f1"] = numeric(obj.get("primary_macro_f1"))
    ece = None
    for key in ("macro_pragmatic_ece", "polarity_dev_ece", "ece", "ece_macro"):
        candidate = numeric(obj.get(key))
        if candidate is not None:
            ece = candidate
            break
    row["ece"] = ece
    if row["macro_pragmatic_f1"] is None and not any(row[head] is not None for head in HEADS):
        row["status"] = "METRIC_MISSING_OR_INCOMPLETE"
    return row


def extract_external(spec: dict[str, Any], data: Any) -> dict[str, Any]:
    obj = data if isinstance(data, dict) else {}
    return {
        "question": spec["question"],
        "system": spec["system"],
        "scope": spec["scope"],
        "backbone": spec["backbone"],
        "run_id": spec["run_id"],
        "seed": spec.get("seed"),
        "artifact_id": spec["artifact_id"],
        "vsfc_macro_f1": numeric(obj.get("vsfc_macro_f1")),
        "vsmec_macro_f1": numeric(obj.get("vsmec_macro_f1")),
        "aivivn_macro_f1": numeric(obj.get("aivivn_macro_f1")),
        "ord_f1": numeric(obj.get("ord_f1")),
        "status": "EXTERNAL_RETENTION_FOUND" if numeric(obj.get("vsfc_macro_f1")) is not None else "EXTERNAL_RETENTION_MISSING",
    }


def extract_calibration(spec: dict[str, Any], data: Any) -> dict[str, Any]:
    obj = data if isinstance(data, dict) else {}
    return {
        "question": spec["question"],
        "system": spec["system"],
        "scope": spec["scope"],
        "backbone": spec["backbone"],
        "run_id": spec["run_id"],
        "seed": spec.get("seed"),
        "artifact_id": spec["artifact_id"],
        "macro_pragmatic_ece": numeric(obj.get("macro_pragmatic_ece")),
        "bin_count": obj.get("bin_count"),
        "reliability_bins": obj.get("reliability_bins") if isinstance(obj.get("reliability_bins"), dict) else {},
        "status": obj.get("status", "UNKNOWN"),
    }


def extract_status(spec: dict[str, Any], data: Any) -> dict[str, Any]:
    obj = data if isinstance(data, dict) else {}
    return {
        "question": spec["question"],
        "system": spec["system"],
        "scope": spec["scope"],
        "backbone": spec["backbone"],
        "run_id": spec["run_id"],
        "seed": spec.get("seed"),
        "artifact_id": spec["artifact_id"],
        "mode": obj.get("mode"),
        "remote_run_id": obj.get("run_id"),
        "status": obj.get("status"),
        "synthetic_results": obj.get("synthetic_results"),
        "note": spec.get("note", ""),
    }


def stat(values: list[float]) -> dict[str, Any]:
    clean = [float(value) for value in values if numeric(value) is not None]
    if not clean:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None}
    return {
        "n": len(clean),
        "mean": statistics.mean(clean),
        "sd": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "min": min(clean),
        "max": max(clean),
    }


def grouped_metric_stats(rows: list[dict[str, Any]], group_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "METRIC_FOUND":
            continue
        groups[tuple(row.get(field) for field in group_fields)].append(row)
    output: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        summary = {field: value for field, value in zip(group_fields, key)}
        summary["seeds"] = ",".join(str(row.get("seed")) for row in sorted(group, key=lambda x: x.get("seed") or 0))
        summary["n"] = len(group)
        for metric in (*HEADS, "macro_pragmatic_f1", "ece"):
            values = [row.get(metric) for row in group if numeric(row.get(metric)) is not None]
            stats = stat(values)
            for suffix, value in stats.items():
                summary[f"{metric}_{suffix}"] = value
        output.append(summary)
    return output


def grouped_external_stats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "EXTERNAL_RETENTION_FOUND":
            groups[row["system"]].append(row)
    output: list[dict[str, Any]] = []
    for system, group in sorted(groups.items()):
        result = {"system": system, "n": len(group), "seeds": ",".join(str(x["seed"]) for x in sorted(group, key=lambda x: x["seed"]))}
        for metric in ("vsfc_macro_f1", "vsmec_macro_f1", "aivivn_macro_f1", "ord_f1"):
            for suffix, value in stat([row.get(metric) for row in group]).items():
                result[f"{metric}_{suffix}"] = value
        output.append(result)
    return output


def fmt(value: Any, digits: int = 3) -> str:
    return "N/A" if value is None or not isinstance(value, (int, float)) else f"{value:.{digits}f}"


def fmt_pct(summary: dict[str, Any], metric: str) -> str:
    mean = summary.get(f"{metric}_mean")
    sd = summary.get(f"{metric}_sd")
    if mean is None:
        return "N/A"
    return f"{mean * 100:.1f} ± {sd * 100:.1f}"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def write_q1a_table(out: Path, summaries: list[dict[str, Any]]) -> str:
    order = {name: index for index, name in enumerate(Q1A_ORDER)}
    rows = sorted([row for row in summaries if row.get("question") == "Q1a" and row.get("scope") in ("baseline", "primary")], key=lambda x: order.get(x.get("system"), 99))
    csv_rows: list[dict[str, Any]] = []
    md_rows: list[list[Any]] = []
    for row in rows:
        record = {"system": row.get("system"), "backbone": row.get("backbone"), "n": row.get("n"), "seeds": row.get("seeds"), "scope": row.get("scope")}
        for metric in (*HEADS, "macro_pragmatic_f1"):
            record[f"{metric}_mean"] = row.get(f"{metric}_mean")
            record[f"{metric}_sd"] = row.get(f"{metric}_sd")
        csv_rows.append(record)
        md_rows.append([row.get("system"), row.get("backbone"), row.get("n"), *(fmt_pct(row, head) for head in HEADS), fmt_pct(row, "macro_pragmatic_f1")])
    csv_dump(out / "tables" / "table2_q1a_baselines.csv", csv_rows)
    return markdown_table(["System", "Backbone", "n", *(HEAD_LABELS[h] for h in HEADS), "Macro-prag"], md_rows)


def write_q1a_paper_schema_ledger(out: Path, summaries: list[dict[str, Any]], metric_rows: list[dict[str, Any]]) -> str:
    """Write every Q1a row from the PDF schema without imputing missing data."""
    by_system = {row.get("system"): row for row in summaries if row.get("question") == "Q1a"}
    notes = {
        "GPT-4.1-mini zero-shot": "NOT_STARTED receipt; no comparable test metric",
        "GPT-4.1-mini 8-shot": "NOT_STARTED receipt; no comparable test metric",
        "ViPragSent-CoT-only (Vistral)": "Partial direct reasoning metrics; excluded non-XLM-R ViPragSent",
        "ViPragSent-explanation-only (Vistral)": "Direct reasoning metrics; excluded non-XLM-R ViPragSent",
        "ViPragSent-no-aux (Vistral)": "Complete metric pack but excluded non-XLM-R ViPragSent",
        "ViPragSent full (Vistral)": "Complete metric pack but excluded non-XLM-R ViPragSent",
    }
    records: list[dict[str, Any]] = []
    md_rows: list[list[Any]] = []
    for system in Q1A_PAPER_SCHEMA_ORDER:
        row = by_system.get(system)
        if row is None:
            n = 0
            seeds = ""
            scope = "missing_or_incomplete"
            status = "MISSING"
            metric_name = None
        else:
            n = int(row.get("n") or 0)
            seeds = row.get("seeds") or ""
            scope = row.get("scope") or "unknown"
            status = "COMPLETE_SEEDS" if n == len(SEEDS) else ("PARTIAL" if n else "MISSING")
            names = sorted({str(item.get("metric_name")) for item in metric_rows if item.get("system") == system and item.get("metric_name")})
            metric_name = ";".join(names) if names else None
        record: dict[str, Any] = {
            "system": system,
            "backbone": Q1A_PAPER_SCHEMA_BACKBONES[system],
            "scope": scope,
            "status": status,
            "n": n,
            "seeds": seeds,
            "metric_name": metric_name,
            "note": notes.get(system, "Comparable HF test metric files" if n else "No selected comparable HF test metric"),
        }
        for metric in (*HEADS, "macro_pragmatic_f1"):
            record[f"{metric}_mean"] = row.get(f"{metric}_mean") if row else None
            record[f"{metric}_sd"] = row.get(f"{metric}_sd") if row else None
        records.append(record)
        md_rows.append([
            system,
            Q1A_PAPER_SCHEMA_BACKBONES[system],
            scope,
            status,
            n,
            seeds or "—",
            fmt_pct(row, "macro_pragmatic_f1") if row else "N/A",
        ])
    csv_dump(out / "tables" / "table2_q1a_paper_schema_coverage.csv", records)
    return markdown_table(["System", "Backbone", "Scope", "Status", "n", "Seeds", "Macro-prag"], md_rows)


def write_q2_table(out: Path, summaries: list[dict[str, Any]], resource_rows: list[dict[str, Any]]) -> str:
    rows = [row for row in summaries if row.get("question") == "Q2" and row.get("scope") == "primary"]
    rows.sort(key=lambda x: str(x.get("system")))
    resource_by_system: dict[str, list[float]] = defaultdict(list)
    for item in resource_rows:
        if item.get("question") == "Q2" and numeric(item.get("successful_gpu_hours")) is not None:
            resource_by_system[str(item.get("system"))].append(float(item["successful_gpu_hours"]))
    full_name = "ViPragSent XLM-R: Full"
    full_values = resource_by_system.get(full_name, [])
    full_mean = statistics.mean(full_values) if full_values else None
    csv_rows: list[dict[str, Any]] = []
    md_rows: list[list[Any]] = []
    for row in rows:
        cost_values = resource_by_system.get(str(row.get("system")), [])
        cost_mean = statistics.mean(cost_values) if cost_values else None
        normalized_cost = cost_mean / full_mean if cost_mean is not None and full_mean else None
        record = {"variant": row.get("system"), "n": row.get("n"), "seeds": row.get("seeds"), "pragmatic_f1_mean": row.get("macro_pragmatic_f1_mean"), "pragmatic_f1_sd": row.get("macro_pragmatic_f1_sd"), "ece_mean": row.get("ece_mean"), "ece_sd": row.get("ece_sd"), "ordinary_f1": None, "normalized_cost": normalized_cost, "gpu_hours_mean": cost_mean, "gpu_hours_n": len(cost_values)}
        csv_rows.append(record)
        ece_mean = row.get("ece_mean")
        ece_sd = row.get("ece_sd")
        ece_display = "N/A" if ece_mean is None else f"{ece_mean * 1000:.1f} ± {ece_sd * 1000:.1f}"
        cost_display = "N/A" if normalized_cost is None else f"{normalized_cost:.2f}"
        md_rows.append([row.get("system"), row.get("n"), row.get("seeds"), fmt_pct(row, "macro_pragmatic_f1"), ece_display, "N/A", cost_display])
    csv_dump(out / "tables" / "table4_q2_xlmr_ablation.csv", csv_rows)
    return markdown_table(["Variant", "n", "Seeds", "Prag.F1", "ECE ×10³ ↓", "Ord.F1", "Cost"], md_rows)


def write_q3_table(out: Path, summaries: list[dict[str, Any]]) -> str:
    rows = [row for row in summaries if row.get("question") == "Q3"]
    rows.sort(key=lambda x: (Q3_SYSTEM_ORDER.index(x.get("system", "").split(" [", 1)[0]) if x.get("system", "").split(" [", 1)[0] in Q3_SYSTEM_ORDER else 99, Q3_BUDGETS.index(str(x.get("system", "").rsplit("[", 1)[-1].rstrip("]"))) if str(x.get("system", "").rsplit("[", 1)[-1].rstrip("]")) in Q3_BUDGETS else 99))
    csv_rows: list[dict[str, Any]] = []
    md_rows: list[list[Any]] = []
    for row in rows:
        system, budget = row.get("system", "").rsplit(" [", 1)
        budget = budget.rstrip("]")
        csv_rows.append({"system": system, "budget": budget, "n": row.get("n"), "seeds": row.get("seeds"), "macro_pragmatic_f1_mean": row.get("macro_pragmatic_f1_mean"), "macro_pragmatic_f1_sd": row.get("macro_pragmatic_f1_sd"), "sarcasm_f1_mean": row.get("sarcasm_mean"), "sarcasm_f1_sd": row.get("sarcasm_sd")})
        md_rows.append([system, budget, row.get("n"), row.get("seeds"), fmt_pct(row, "sarcasm"), fmt_pct(row, "macro_pragmatic_f1")])
    csv_dump(out / "tables" / "table_q3_low_resource.csv", csv_rows)
    return markdown_table(["System", "Budget", "n", "Seeds", "Sarcasm F1", "Macro-prag F1"], md_rows)


def write_q4_table(out: Path, calibration_rows: list[dict[str, Any]]) -> str:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in calibration_rows:
        if row.get("macro_pragmatic_ece") is not None:
            groups[row["system"]].append(row)
    rows: list[dict[str, Any]] = []
    md_rows: list[list[Any]] = []
    for system, group in sorted(groups.items()):
        values = [row["macro_pragmatic_ece"] for row in group]
        stats = stat(values)
        result = {"system": system, "n": len(group), "seeds": ",".join(str(row["seed"]) for row in sorted(group, key=lambda x: x["seed"])), **{f"ece_{k}": v for k, v in stats.items()}}
        rows.append(result)
        md_rows.append([system, result["n"], result["seeds"], f"{result['ece_mean']:.4f} ± {result['ece_sd']:.4f}"])
    csv_dump(out / "tables" / "table_q4_calibration.csv", rows)
    return markdown_table(["System", "n", "Seeds", "Macro pragmatic ECE ↓"], md_rows)


def write_q1b_table(out: Path, external_rows: list[dict[str, Any]]) -> str:
    summaries = grouped_external_stats(external_rows)
    md_rows: list[list[Any]] = []
    for row in summaries:
        md_rows.append([row["system"], row["n"], row["seeds"], fmt_pct(row, "vsfc_macro_f1"), fmt_pct(row, "vsmec_macro_f1"), fmt_pct(row, "aivivn_macro_f1"), fmt_pct(row, "ord_f1")])
    csv_dump(out / "tables" / "table3_ordinary_retention.csv", summaries)
    return markdown_table(["System", "n", "Seeds", "UIT-VSFC", "UIT-VSMEC", "AIVIVN-2019", "Ord.F1"], md_rows)


def write_cost_table(out: Path, cost_rows: list[dict[str, Any]]) -> str:
    """Aggregate Q1a measured resource provenance in a Table-5-shaped form."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cost_rows:
        if row.get("question") == "Q1a" and row.get("system") in Q1A_ORDER and numeric(row.get("successful_gpu_hours")) is not None:
            groups[str(row.get("system"))].append(row)
    rows: list[dict[str, Any]] = []
    md_rows: list[list[Any]] = []
    order = {name: index for index, name in enumerate(Q1A_PAPER_SCHEMA_ORDER)}
    for system, group in sorted(groups.items(), key=lambda item: order.get(item[0], 99)):
        gpu = stat([row.get("successful_gpu_hours") for row in group])
        failed = stat([row.get("failed_or_retried_gpu_hours") for row in group])
        azure_values = [numeric(row.get("azure_cost_usd")) for row in group]
        azure_stats = stat(azure_values)
        raw_statuses = sorted({str(row.get("azure_cost_status")) for row in group if row.get("azure_cost_status")})
        record = {
            "system": system,
            "n": len(group),
            "seeds": ",".join(str(row.get("seed")) for row in sorted(group, key=lambda item: item.get("seed") or 0)),
            "gpu_hours_mean": gpu["mean"],
            "gpu_hours_sd": gpu["sd"],
            "failed_or_retried_gpu_hours_mean": failed["mean"],
            "azure_cost_usd_mean": azure_stats["mean"],
            "azure_cost_usd_sd": azure_stats["sd"],
            "azure_cost_status": ";".join(raw_statuses) or "N/A",
        }
        rows.append(record)
        md_rows.append([
            system,
            record["n"],
            record["seeds"],
            fmt(record["gpu_hours_mean"], 3),
            fmt(record["gpu_hours_sd"], 3),
            "N/A" if record["azure_cost_usd_mean"] is None else fmt(record["azure_cost_usd_mean"], 2),
            record["azure_cost_status"],
        ])
    csv_dump(out / "tables" / "table5_cost_inventory.csv", rows)
    return markdown_table(["System", "n", "Seeds", "GPU-h mean", "GPU-h SD", "Azure $ mean", "Azure cost status"], md_rows)


def plot_q1a(out: Path, summaries: list[dict[str, Any]]) -> None:
    data = [row for row in summaries if row.get("question") == "Q1a" and row.get("scope") in ("baseline", "primary") and row.get("system") in Q1A_ORDER]
    data.sort(key=lambda x: Q1A_ORDER.index(x["system"]))
    if not data:
        return
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(HEADS))
    width = 0.78 / max(1, len(data))
    fig, ax = plt.subplots(figsize=(13, 5.8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(data)))
    for index, row in enumerate(data):
        means = [row.get(f"{head}_mean", 0) * 100 for head in HEADS]
        errors = [row.get(f"{head}_sd", 0) * 100 for head in HEADS]
        ax.bar(x - 0.39 + width / 2 + index * width, means, width, yerr=errors, capsize=2, label=row["system"], color=colors[index])
    ax.set_ylabel("Binary macro-F1 (%)")
    ax.set_xticks(x, [HEAD_LABELS[head] for head in HEADS], rotation=18, ha="right")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Q1a baseline comparison on the HF test artefacts")
    ax.legend(ncol=2, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(figures / "figure2_q1a_baseline_grouped_bars.png", dpi=220)
    plt.close(fig)

    base = next((row for row in data if row["system"] == "PhoBERT (single-task)"), None)
    if base is None:
        return
    gain_rows = [row for row in data if row["system"] != base["system"]]
    fig, ax = plt.subplots(figsize=(13, 5.8))
    width = 0.78 / max(1, len(gain_rows))
    for index, row in enumerate(gain_rows):
        means = [(row.get(f"{head}_mean", 0) - base.get(f"{head}_mean", 0)) * 100 for head in HEADS]
        ax.bar(x - 0.39 + width / 2 + index * width, means, width, label=row["system"], color=colors[(index + 1) % len(colors)])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Gain over PhoBERT single-task (percentage points)")
    ax.set_xticks(x, [HEAD_LABELS[head] for head in HEADS], rotation=18, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Per-phenomenon gain over the single-task baseline")
    ax.legend(ncol=2, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(figures / "figure3_gain_over_phobert_single_task.png", dpi=220)
    plt.close(fig)


def plot_q3(out: Path, summaries: list[dict[str, Any]]) -> None:
    rows = [row for row in summaries if row.get("question") == "Q3"]
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    positions = np.arange(len(Q3_BUDGETS))
    colors = {"PhoBERT (fine-tune)": "#4c78a8", "Vistral-7B SFT": "#f58518", "ViPragSent (XLM-R-large)": "#54a24b"}
    for system in Q3_SYSTEM_ORDER:
        means: list[float] = []
        errors: list[float] = []
        available: list[float] = []
        for budget in Q3_BUDGETS:
            row = next((r for r in rows if r.get("system") == f"{system} [{budget}]"), None)
            if row and row.get("sarcasm_mean") is not None:
                means.append(row["sarcasm_mean"] * 100)
                errors.append((row.get("sarcasm_sd") or 0) * 100)
                available.append(float(len(available)))
            else:
                means.append(np.nan)
                errors.append(0)
                available.append(float(len(available)))
        ax.errorbar(positions, means, yerr=errors, marker="o", linewidth=2, capsize=3, label=system, color=colors[system])
    ax.set_xticks(positions, Q3_BUDGETS)
    ax.set_xlabel("Labelled sarcasm budget")
    ax.set_ylabel("Sarcasm binary macro-F1 (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.set_title("Q3 low-resource comparison; error bars are sample SD over available seeds")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "figure4_q3_sarcasm_budget_lines.png", dpi=220)
    plt.close(fig)


def plot_q4_reliability(out: Path, calibration_rows: list[dict[str, Any]]) -> None:
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    colors = {"Vistral-7B SFT": "#f58518", "ViPragSent (XLM-R-large)": "#54a24b"}
    for system in ("Vistral-7B SFT", "ViPragSent (XLM-R-large)"):
        rows = [row for row in calibration_rows if row.get("system") == system]
        pooled: dict[int, dict[str, float]] = defaultdict(lambda: {"count": 0.0, "confidence": 0.0, "positive": 0.0})
        for row in rows:
            bins = row.get("reliability_bins", {})
            if not isinstance(bins, dict):
                continue
            for head_bins in bins.values():
                if not isinstance(head_bins, list):
                    continue
                for item in head_bins:
                    if not isinstance(item, dict):
                        continue
                    index = int(item.get("bin_index", 0))
                    count = numeric(item.get("count")) or 0.0
                    pooled[index]["count"] += count
                    pooled[index]["confidence"] += count * (numeric(item.get("mean_confidence")) or 0.0)
                    pooled[index]["positive"] += count * (numeric(item.get("empirical_positive_rate")) or 0.0)
        points = [(v["confidence"] / v["count"], v["positive"] / v["count"]) for _, v in sorted(pooled.items()) if v["count"] > 0]
        if points:
            xs, ys = zip(*points)
            ax.plot(xs, ys, marker="o", linewidth=2, label=system, color=colors[system])
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.6, label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Empirical positive rate")
    ax.grid(alpha=0.25)
    ax.set_title("Q4 reliability diagrams from HF calibration artefacts")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "figure7_q4_reliability_diagrams.png", dpi=220)
    plt.close(fig)


def plot_history(out: Path, history_rows: list[dict[str, Any]]) -> None:
    if not history_rows:
        return
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in history_rows:
        value = numeric(row.get("dev_macro_pragmatic_f1"))
        epoch = numeric(row.get("epoch"))
        if value is not None and epoch is not None:
            grouped[(row["system"], epoch)].append(value)
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    colors = {"Vistral-7B SFT": "#f58518", "ViPragSent (XLM-R-large)": "#54a24b"}
    for system in ("Vistral-7B SFT", "ViPragSent (XLM-R-large)"):
        epochs = sorted(epoch for (name, epoch) in grouped if name == system)
        if not epochs:
            continue
        means = [statistics.mean(grouped[(system, epoch)]) * 100 for epoch in epochs]
        ax.plot(epochs, means, marker="o", linewidth=2, label=system, color=colors[system])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dev pragmatic macro-F1 (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.set_title("Available Q4 training histories")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "figure6_q4_training_curves.png", dpi=220)
    plt.close(fig)


def build_confusion(out: Path, prediction_specs: list[dict[str, Any]], source_rows: dict[str, dict[str, Any]]) -> None:
    matrices = {head: [[0, 0], [0, 0]] for head in HEADS}
    source_count = 0
    for spec in prediction_specs:
        raw = bytes_for(spec, source_rows, out)
        if raw is None:
            continue
        source_count += 1
        for line in raw.splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            gold = item.get("gold", {})
            pred = item.get("predictions", {})
            for head in HEADS:
                g = int(gold.get(head, 0))
                p = int(pred.get(head, 0))
                if g in (0, 1) and p in (0, 1):
                    matrices[head][g][p] += 1
    if source_count == 0:
        return
    rows: list[dict[str, Any]] = []
    for head, matrix in matrices.items():
        rows.extend([{"head": head, "gold": gold, "predicted_0": matrix[gold][0], "predicted_1": matrix[gold][1]} for gold in (0, 1)])
    csv_dump(out / "tables" / "figure5_confusion_matrix_counts.csv", rows)
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.2), constrained_layout=True)
    for ax, head in zip(axes.flat, HEADS):
        matrix = np.asarray(matrices[head], dtype=float)
        row_sums = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0) * 100
        image = ax.imshow(normalized, vmin=0, vmax=100, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{normalized[i, j]:.1f}", ha="center", va="center", color="black")
        ax.set_title(HEAD_LABELS[head])
        ax.set_xticks((0, 1), ("pred 0", "pred 1"))
        ax.set_yticks((0, 1), ("gold 0", "gold 1"))
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82, pad=0.03, label="Row-normalized (%)")
    fig.suptitle("Q1a ViPragSent XLM-R-large confusion matrices pooled over seeds 21–23")
    fig.savefig(out / "figures" / "figure5_pragmatic_confusion_matrices.png", dpi=220)
    plt.close(fig)


def copy_audit_pointers(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for source, target in (
        (RECHECK_DIR / "inventory_status.json", out / "hf_inventory_status.json"),
        (RECHECK_DIR / "remote_recheck_summary.json", out / "hf_remote_recheck_summary.json"),
        (RECHECK_DIR / "remote_recheck_hashes.sha256", out / "hf_remote_recheck_hashes.sha256"),
        (OLD_AUDIT_DIR / "xlmr_weight_inventory.csv", out / "xlmr_weight_inventory.csv"),
    ):
        if source.exists():
            shutil.copy2(source, target)


def candidate_inventory(out: Path, entries: list[dict[str, Any]]) -> None:
    patterns = (
        "q1a_phobert",
        "q1a_xlmr",
        "q1a_sailor",
        "q1a_vistral",
        "q1a_vipragsent",
        "q1a_azure_gpt41",
        "q1a_cot_only",
        "q1a_explanation_only",
        "q2_",
        "xlmr_followup_q2",
        "q3_",
        "xlmr_followup_q3",
        "q4_",
        "xlmr_followup_q4",
        "q1b_",
        "xlmr_followup_q1b",
    )
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for entry in entries:
        path = str(entry.get("path", ""))
        lower = path.lower()
        match = next((pattern for pattern in patterns if pattern.lower() in lower), None)
        if not match:
            continue
        counts[match]["tree_entries"] += 1
        if entry.get("type") == "file":
            counts[match]["files"] += 1
        if path.endswith("/metrics/test_metrics.json"):
            counts[match]["test_metric_files"] += 1
        if path.endswith("/FINAL_SCIENCE_RESULT.json"):
            counts[match]["final_result_files"] += 1
        if path.endswith("/paper_artifacts/q4_pragmatic_calibration_per_seed.json"):
            counts[match]["calibration_files"] += 1
        if path.endswith("/metrics/external_retention_metrics.json"):
            counts[match]["retention_files"] += 1
    rows = [{"pattern": key, **value} for key, value in sorted(counts.items())]
    csv_dump(out / "candidate_baseline_inventory.csv", rows)


def write_coverage(out: Path, metric_rows: list[dict[str, Any]], calibration_rows: list[dict[str, Any]], external_rows: list[dict[str, Any]], missing: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    def add(question: str, system: str, scope: str, backbone: str, requested: int, found_seeds: list[int], note: str) -> None:
        found = sorted(set(found_seeds))
        status = "COMPLETE_SEEDS" if len(found) == requested else ("PARTIAL" if found else "MISSING")
        rows.append({"question": question, "system": system, "scope": scope, "backbone": backbone, "requested_seed_count": requested, "found_seed_count": len(found), "found_seeds": ",".join(str(x) for x in found), "status": status, "note": note})
    for system in Q1A_ORDER:
        found = [int(row["seed"]) for row in metric_rows if row.get("question") == "Q1a" and row.get("system") == system and row.get("status") == "METRIC_FOUND"]
        scope = "primary" if system == "ViPragSent (XLM-R-large)" else "baseline"
        backbone = "XLM-R-large" if system == "ViPragSent (XLM-R-large)" else next((row.get("backbone") for row in metric_rows if row.get("system") == system), "unknown")
        add("Q1a", system, scope, backbone, 3, found, "HF test metric files")
    for system in sorted({row["system"] for row in metric_rows if row.get("question") == "Q1a" and row.get("scope") == "excluded_non_xlmr_vipragsent"}):
        found = [int(row["seed"]) for row in metric_rows if row.get("system") == system and row.get("status") == "METRIC_FOUND"]
        add("Q1a", system, "excluded_non_xlmr_vipragsent", "Vistral-7B", 3, found, "Excluded by the XLM-R-large ViPragSent scope")
    for row in missing:
        add(row["question"], row["system"], row["scope"], row["backbone"], 1, [], row["note"])
    for system in sorted({row["system"] for row in metric_rows if row.get("question") == "Q2"}):
        found = [int(row["seed"]) for row in metric_rows if row.get("question") == "Q2" and row.get("system") == system and row.get("status") == "METRIC_FOUND"]
        add("Q2", system, "primary", "XLM-R-large", 3, found, "HF test metric files; six ablation variants")
    for system in Q3_SYSTEM_ORDER:
        for budget in Q3_BUDGETS:
            name = f"{system} [{budget}]"
            found = [int(row["seed"]) for row in metric_rows if row.get("question") == "Q3" and row.get("system") == name and row.get("status") == "METRIC_FOUND"]
            scope = "primary" if system == "ViPragSent (XLM-R-large)" else "baseline"
            backbone = "XLM-R-large" if scope == "primary" else system.split("(")[-1].rstrip(")")
            add("Q3", name, scope, backbone, 3, found, "HF test metric files; incomplete budgets remain visible")
    for system in sorted({row["system"] for row in calibration_rows}):
        found = [int(row["seed"]) for row in calibration_rows if row.get("system") == system and row.get("macro_pragmatic_ece") is not None]
        scope = "primary" if system == "ViPragSent (XLM-R-large)" else "baseline"
        add("Q4", system, scope, "XLM-R-large" if scope == "primary" else "Vistral-7B", 3, found, "Dedicated calibration JSON")
    for system in sorted({row["system"] for row in external_rows}):
        found = [int(row["seed"]) for row in external_rows if row.get("system") == system and row.get("status") == "EXTERNAL_RETENTION_FOUND"]
        scope = "primary" if system == "ViPragSent (XLM-R-large)" else "baseline"
        add("Q1b", system, scope, "XLM-R-large" if scope == "primary" else system.split("(")[-1].rstrip(")"), 3, found, "External retention metrics")
    csv_dump(out / "coverage_matrix.csv", rows)
    return rows


def write_report(out: Path, table2: str, table2_full: str, table3: str, table4: str, table_q3: str, table_q4: str, table5: str, coverage: list[dict[str, Any]], source_rows: dict[str, dict[str, Any]], metric_rows: list[dict[str, Any]], resource_rows: list[dict[str, Any]]) -> None:
    complete_sources = sum(row.get("fetch_status") in ("fetched", "cached") for row in source_rows.values())
    primary_q1a = next((row for row in grouped_metric_stats(metric_rows, ("question", "system", "scope", "backbone")) if row.get("system") == "ViPragSent (XLM-R-large)" and row.get("question") == "Q1a"), None)
    primary_display = "N/A" if primary_q1a is None else fmt_pct(primary_q1a, "macro_pragmatic_f1")
    complete_count = sum(row["status"] == "COMPLETE_SEEDS" for row in coverage)
    report = f"""# ViPragSent NAACL comparison artefact package

Generated: `{utc_now()}`  
Analysis state: **ANALYZED** (HF artefact analysis; no training or inference rerun).

## Scope and evidence

This package follows the experiment schema in `main.pdf` (Q1a/Q1b/Q2/Q3/Q4,
baseline comparison, low-resource curve, retention, calibration and training
history). The PDF is used as a schema reference only; its reported numbers are
not copied into these tables.

- Source: authenticated Hugging Face API using the token from `.env`; no GitHub code was used.
- Fresh HF recheck: see [`hf_remote_recheck_summary.json`](hf_remote_recheck_summary.json).
- Account inventory: 30 repositories, 480,738 tree entries, and complete pagination for all 30 repositories.
- Selected structured sources fetched or cached in this package: {complete_sources}.
- Primary target: **ViPragSent with XLM-R-large**, seeds 21/22/23.
- Ordinary baselines remain in the comparison. ViPragSent variants using Vistral are recorded as discovered but excluded from the primary scope, following the original filtering instruction.
- The PDF describes five seeds; this HF snapshot provides three requested seeds (21/22/23) for the primary Q1a/Q2/Q3/Q4 groups. No five-seed claim is made here.

## Q1a baseline table

Primary ViPragSent XLM-R-large macro-pragmatic F1: **{primary_display}** (mean ± sample SD over available seeds).

{table2}

The machine-readable version is [`tables/table2_q1a_baselines.csv`](tables/table2_q1a_baselines.csv), and the grouped-bar and gain plots are [`figures/figure2_q1a_baseline_grouped_bars.png`](figures/figure2_q1a_baseline_grouped_bars.png) and [`figures/figure3_gain_over_phobert_single_task.png`](figures/figure3_gain_over_phobert_single_task.png).

### Q1a paper-schema coverage ledger

The following ledger keeps every Q1a row represented in the reference PDF. It
does not turn a missing or excluded HF run into a number. The rank-A primary
comparison above therefore contains only the standard baselines plus the
XLM-R-large ViPragSent target; the ledger documents the remaining rows and
their evidence state.

{table2_full}

The full ledger is [`tables/table2_q1a_paper_schema_coverage.csv`](tables/table2_q1a_paper_schema_coverage.csv). GPT-4.1-mini zero-shot and 8-shot have only `NOT_STARTED` receipts in the current HF tree; the exact receipt records are preserved in [`baseline_status_records.jsonl`](baseline_status_records.jsonl). The CoT-only and explanation-only rows have direct reasoning metrics, but are excluded from the primary table because they are Vistral-backbone ViPragSent variants.

## Q1b ordinary-sentiment retention

{table3}

This table is intentionally limited to systems for which the HF snapshot
contains `external_retention_metrics.json`. Missing systems are listed in
[`coverage_matrix.csv`](coverage_matrix.csv), not replaced with paper values.

## Q2 XLM-R-large ablations

{table4}

Ordinary-sentiment F1 is **N/A** because the selected
XLM-R Q2 test metrics do not expose ordinary-sentiment F1. Normalized cost is
computed from the HF `training/resource_usage.json` GPU-hour means relative
to the full XLM-R run; it is not copied from the PDF. ECE is shown as ×10³ to
mirror the paper's table convention. Full data are in
[`tables/table4_q2_xlmr_ablation.csv`](tables/table4_q2_xlmr_ablation.csv).

## Q3 low-resource curve

{table_q3}

The line graph is [`figures/figure4_q3_sarcasm_budget_lines.png`](figures/figure4_q3_sarcasm_budget_lines.png). Missing seed/budget combinations remain visible as partial coverage rather than being imputed.

## Q4 calibration and training history

{table_q4}

The reliability diagram is [`figures/figure7_q4_reliability_diagrams.png`](figures/figure7_q4_reliability_diagrams.png), and the available dev training histories are [`figures/figure6_q4_training_curves.png`](figures/figure6_q4_training_curves.png). A pooled six-head confusion-matrix figure based on the three primary Q1a prediction JSONL files is [`figures/figure5_pragmatic_confusion_matrices.png`](figures/figure5_pragmatic_confusion_matrices.png).

## Q1a cost and resource provenance

{table5}

This primary-scope table reports only GPU-hour/resource values actually present
in the HF run artefacts for the standard baselines and XLM-R-large target.
Azure/API cost is shown as a status when HF records
`NOT_APPLICABLE` or does not record a numeric value; the PDF's annotation and
API cost values are not copied into this package. Machine-readable files are
[`tables/table5_cost_inventory.csv`](tables/table5_cost_inventory.csv),
[`cost_inventory_q1a.csv`](cost_inventory_q1a.csv), and
[`resource_usage_records.csv`](resource_usage_records.csv).

## Coverage, exclusions and reproducibility boundary

- Complete-seed coverage rows in the matrix: {complete_count}/{len(coverage)}.
- `coverage_matrix.csv` records incomplete GPT/CoT/explanation runs, excluded non-XLM-R ViPragSent variants, partial Q3 baselines, and unavailable fields.
- `candidate_baseline_inventory.csv` records the wider HF candidate tree scan.
- [`xlmr_weight_inventory.csv`](xlmr_weight_inventory.csv) is the weight metadata inventory from the audited XLM-R scope. Weight payloads were not downloaded or deserialized; this package is an analysis artefact, not a checkpoint-resume operation.
- [`selected_run_metrics.csv`](selected_run_metrics.csv), [`calibration_records.jsonl`](calibration_records.jsonl), [`external_retention_records.csv`](external_retention_records.csv), and [`source_manifest.jsonl`](source_manifest.jsonl) preserve the per-seed values and SHA-256 provenance used here.
- [`baseline_status_records.jsonl`](baseline_status_records.jsonl) preserves the GPT baseline receipt status rather than treating an unfinished run as a metric.
- [`resource_usage_records.csv`](resource_usage_records.csv) preserves the Q1a/Q2 resource records used for Table 4 cost normalization and Table 5.
- [`review_checks.json`](review_checks.json) and [`artifact_hashes.sha256`](artifact_hashes.sha256) are the final local integrity checks.

The package is suitable as a paper-preparation basis, but claims should remain
bounded by the three-seed HF coverage and the explicit missingness table until
an independent rerun or equivalent reproducibility check promotes the result
from **ANALYZED** to **VERIFIED**.
"""
    (out / "naacl_comparison_report.md").write_text(report, encoding="utf-8")


def final_hashes(out: Path) -> None:
    hash_path = out / "artifact_hashes.sha256"
    lines: list[str] = []
    for path in sorted(out.rglob("*")):
        if not path.is_file() or path == hash_path:
            continue
        lines.append(f"{sha256_file(path)}  {path.relative_to(out).as_posix()}")
    hash_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    copy_audit_pointers(out)
    entries = load_tree(args.tree)
    candidate_inventory(out, entries)
    specs, missing = build_specs(entries)
    source_rows = fetch_sources(specs, out, args.env, args.workers)

    metric_rows: list[dict[str, Any]] = []
    external_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    prediction_specs: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    review_by_run: dict[str, dict[str, Any]] = {}
    for spec in specs:
        if spec.get("entry") is None:
            continue
        if spec["kind"] == "metric":
            metric_rows.append(extract_metric(spec, json_for(spec, source_rows, out)))
        elif spec["kind"] == "external":
            external_rows.append(extract_external(spec, json_for(spec, source_rows, out)))
        elif spec["kind"] == "calibration":
            calibration_rows.append(extract_calibration(spec, json_for(spec, source_rows, out)))
        elif spec["kind"] == "history":
            history = json_for(spec, source_rows, out)
            if isinstance(history, list):
                for point in history:
                    if isinstance(point, dict):
                        history_rows.append({"system": spec["system"], "run_id": spec["run_id"], "seed": spec.get("seed"), **point})
        elif spec["kind"] == "prediction":
            prediction_specs.append(spec)
        elif spec["kind"] == "status":
            data = json_for(spec, source_rows, out)
            status_rows.append(extract_status(spec, data))
        elif spec["kind"] == "resource":
            data = json_for(spec, source_rows, out)
            if isinstance(data, dict):
                resource_rows.append({
                    "question": spec["question"],
                    "system": spec["system"],
                    "scope": spec["scope"],
                    "backbone": spec["backbone"],
                    "run_id": spec["run_id"],
                    "seed": spec.get("seed"),
                    "artifact_id": spec["artifact_id"],
                    "successful_gpu_hours": numeric(data.get("successful_gpu_hours")),
                    "failed_or_retried_gpu_hours": numeric(data.get("failed_or_retried_gpu_hours")),
                    "status": "RESOURCE_FOUND",
                })
        elif spec["kind"] == "review":
            data = json_for(spec, source_rows, out)
            if isinstance(data, dict):
                review_by_run[spec["run_id"]] = data

    run_cost_rows: list[dict[str, Any]] = []
    for row in resource_rows:
        review = review_by_run.get(row["run_id"], {})
        azure_raw = review.get("azure_cost_usd") if isinstance(review, dict) else None
        run_cost_rows.append({
            **row,
            "azure_cost_usd": numeric(azure_raw),
            "azure_cost_status": str(azure_raw) if azure_raw is not None else "NOT_RECORDED",
        })
    csv_dump(out / "cost_inventory_q1a.csv", [row for row in run_cost_rows if row.get("question") == "Q1a"])
    csv_dump(out / "resource_usage_records.csv", run_cost_rows)

    source_manifest_rows: list[dict[str, Any]] = []
    for spec in specs:
        entry = spec.get("entry")
        row = {key: value for key, value in spec.items() if key != "entry"}
        row["source_key"] = source_key(entry) if entry else ""
        if entry:
            row.update(source_rows.get(source_key(entry), {}))
        source_manifest_rows.append(row)
    jsonl_dump(out / "source_manifest.jsonl", source_manifest_rows)
    jsonl_dump(out / "calibration_records.jsonl", calibration_rows)
    jsonl_dump(out / "baseline_status_records.jsonl", status_rows)
    csv_dump(out / "selected_run_metrics.csv", metric_rows)
    csv_dump(out / "external_retention_records.csv", external_rows)
    csv_dump(out / "training_history_records.csv", history_rows)

    summaries = grouped_metric_stats(metric_rows, ("question", "system", "scope", "backbone"))
    table2 = write_q1a_table(out, summaries)
    table2_full = write_q1a_paper_schema_ledger(out, summaries, metric_rows)
    table3 = write_q1b_table(out, external_rows)
    table4 = write_q2_table(out, summaries, resource_rows)
    table_q3 = write_q3_table(out, summaries)
    table_q4 = write_q4_table(out, calibration_rows)
    table5 = write_cost_table(out, run_cost_rows)
    coverage = write_coverage(out, metric_rows, calibration_rows, external_rows, missing)
    plot_q1a(out, summaries)
    plot_q3(out, summaries)
    plot_q4_reliability(out, calibration_rows)
    plot_history(out, history_rows)
    build_confusion(out, prediction_specs, source_rows)

    write_report(out, table2, table2_full, table3, table4, table_q3, table_q4, table5, coverage, source_rows, metric_rows, resource_rows)

    primary_groups = [row for row in summaries if row.get("scope") == "primary"]
    range_failures: list[dict[str, Any]] = []
    for row in metric_rows:
        for field in (*HEADS, "macro_pragmatic_f1", "ece"):
            value = numeric(row.get(field))
            if value is not None and not 0 <= value <= 1:
                range_failures.append({"run_id": row.get("run_id"), "field": field, "value": value})
    primary_q1a = [row for row in metric_rows if row.get("question") == "Q1a" and row.get("system") == "ViPragSent (XLM-R-large)" and row.get("status") == "METRIC_FOUND"]
    primary_q2 = [row for row in metric_rows if row.get("question") == "Q2" and row.get("status") == "METRIC_FOUND"]
    primary_q3 = [row for row in metric_rows if row.get("question") == "Q3" and row.get("scope") == "primary" and row.get("status") == "METRIC_FOUND"]
    primary_q4 = [row for row in calibration_rows if row.get("scope") == "primary" and row.get("macro_pragmatic_ece") is not None]
    q2_resource_groups = {
        row.get("system")
        for row in resource_rows
        if row.get("question") == "Q2" and numeric(row.get("successful_gpu_hours")) is not None
    }
    q2_resource_three_seed = all(
        len({row.get("seed") for row in resource_rows if row.get("question") == "Q2" and row.get("system") == system}) == 3
        for system in q2_resource_groups
    )
    checks = {
        "analysis_state": "ANALYZED",
        "fresh_tree_manifest_exists": args.tree.exists(),
        "source_fetch_error_count": sum(row.get("fetch_status") == "error" for row in source_rows.values()),
        "source_sha256_local_mismatch_count": sum(row.get("fetch_status") in ("fetched", "cached") and (not (out / row["raw_path"]).exists() or sha256_file(out / row["raw_path"]) != row.get("fetch_sha256")) for row in source_rows.values()),
        "numeric_range_failures": range_failures,
        "primary_q1a_seed_count": len({row.get("seed") for row in primary_q1a}),
        "primary_q2_group_count": len({row.get("system") for row in primary_q2}),
        "primary_q2_all_groups_three_seeds": all(len({row.get("seed") for row in primary_q2 if row.get("system") == system}) == 3 for system in {row.get("system") for row in primary_q2}),
        "primary_q3_budget_count": len({row.get("system") for row in primary_q3}),
        "primary_q3_all_budgets_three_seeds": all(len({row.get("seed") for row in primary_q3 if row.get("system") == system}) == 3 for system in {row.get("system") for row in primary_q3}),
        "primary_q4_seed_count": len({row.get("seed") for row in primary_q4}),
        "primary_scope_backbone_filter": all(row.get("backbone") == "XLM-R-large" for row in primary_groups),
        "q2_resource_group_count": len(q2_resource_groups),
        "q2_resource_all_groups_three_seeds": q2_resource_three_seed,
        "gpt_status_record_count": len(status_rows),
        "required_report_exists": (out / "naacl_comparison_report.md").exists(),
        "required_tables_exist": all((out / "tables" / name).exists() for name in ("table2_q1a_baselines.csv", "table2_q1a_paper_schema_coverage.csv", "table3_ordinary_retention.csv", "table4_q2_xlmr_ablation.csv", "table_q3_low_resource.csv", "table_q4_calibration.csv", "table5_cost_inventory.csv")),
    }
    checks["status"] = "PASS" if checks["source_fetch_error_count"] == 0 and checks["source_sha256_local_mismatch_count"] == 0 and not range_failures and checks["primary_scope_backbone_filter"] and checks["required_report_exists"] and checks["required_tables_exist"] and checks["q2_resource_group_count"] == 6 and checks["q2_resource_all_groups_three_seeds"] and checks["gpt_status_record_count"] == 2 else "FAIL"
    json_dump(out / "review_checks.json", checks)
    json_dump(out / "analysis_status.json", {"status": "ANALYZED", "generated_at": utc_now(), "verification_note": "No independent rerun was performed."})
    final_hashes(out)
    print(f"[done] {out}", flush=True)
    print(f"[done] checks={checks['status']} sources={len(source_rows)} metric_rows={len(metric_rows)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", type=Path, default=TREE_PATH)
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
