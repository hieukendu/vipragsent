from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..atomic import atomic_write_json
from ..constants import PRAGMATIC_LABELS
from ..hashing import sha256_file, sha256_json
from ..statistics.bootstrap import hierarchical_bootstrap
from .metrics import binary_macro_f1, macro_pragmatic_f1

METHOD_ID = "paired_hierarchical_bootstrap_sign_plus_one_v1"
EFFECTIVE_FALLBACK_FIELD = "effective_full_split_all_zero_fallback"


def _code_commit(root: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def prediction_row_alignment_signature(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, tuple[tuple[str, int], ...]], ...]:
    if not rows:
        raise ValueError("Table 2 confidence intervals require non-empty predictions")
    seen: set[str] = set()
    signature: list[tuple[str, tuple[tuple[str, int], ...]]] = []
    for index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", f"__position_{index}"))
        if sample_id in seen:
            raise ValueError(f"Table 2 confidence intervals require unique sample_id values: {sample_id}")
        seen.add(sample_id)
        gold = row.get("gold", {})
        if not isinstance(gold, Mapping) or any(label not in gold for label in PRAGMATIC_LABELS):
            raise ValueError("Table 2 confidence intervals require all six pragmatic gold labels per prediction row")
        signature.append((sample_id, tuple((label, int(gold[label])) for label in PRAGMATIC_LABELS)))
    return tuple(signature)


def validate_prediction_row_alignment(seed_rows: Sequence[Sequence[Mapping[str, Any]]]) -> None:
    if not seed_rows:
        raise ValueError("Table 2 confidence intervals require at least one seed prediction set")
    reference = prediction_row_alignment_signature(seed_rows[0])
    for rows in seed_rows[1:]:
        if prediction_row_alignment_signature(rows) != reference:
            raise ValueError("Table 2 confidence interval inputs are not aligned by sample order/gold labels")


def _prediction_values(row: Mapping[str, Any]) -> Mapping[str, Any]:
    if EFFECTIVE_FALLBACK_FIELD in row:
        values = row.get(EFFECTIVE_FALLBACK_FIELD)
    else:
        values = row.get("predictions")
    if not isinstance(values, Mapping):
        raise ValueError("Table 2 confidence intervals require prediction labels per prediction row")
    return values


def prediction_rows_to_pair(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    prediction_row_alignment_signature(rows)
    true = {label: [] for label in PRAGMATIC_LABELS}
    predicted = {label: [] for label in PRAGMATIC_LABELS}
    for row in rows:
        gold = row.get("gold", {})
        values = _prediction_values(row)
        if any(label not in gold or label not in values for label in PRAGMATIC_LABELS):
            raise ValueError("Table 2 confidence intervals require all six pragmatic labels per prediction row")
        for label in PRAGMATIC_LABELS:
            true[label].append(int(gold[label]))
            predicted[label].append(int(values[label]))
    return true, predicted


def evaluate_q1a_confidence_intervals(
    seed_rows: Sequence[Sequence[dict[str, Any]]],
    *,
    prediction_hash: str,
    config_hash: str,
    code_commit: str,
    resamples: int = 1000,
    bootstrap_seed: int = 20260525,
) -> dict[str, Any]:
    if not seed_rows:
        raise ValueError("Table 2 confidence intervals require at least one seed prediction set")
    validate_prediction_row_alignment(seed_rows)
    pairs = [prediction_rows_to_pair(rows) for rows in seed_rows]
    labels: dict[str, dict[str, float]] = {}
    for label in PRAGMATIC_LABELS:
        result = hierarchical_bootstrap(
            [(true[label], predicted[label]) for true, predicted in pairs],
            lambda gold, pred: binary_macro_f1(gold, pred),
            resamples=resamples,
            seed=bootstrap_seed,
        )
        labels[label] = {"estimate": result.observed, "low": result.ci_low, "high": result.ci_high}
    macro = hierarchical_bootstrap(
        [(true, predicted) for true, predicted in pairs],
        lambda gold, pred: macro_pragmatic_f1(gold, pred),
        resamples=resamples,
        seed=bootstrap_seed,
    )
    return {
        "status": "PASS",
        "method": {
            "method_id": METHOD_ID,
            "resampling_unit": "seed_then_test_example",
            "resamples": int(resamples),
            "bootstrap_seed": int(bootstrap_seed),
            "confidence_level": 0.95,
            "cross_seed_behavior": "hierarchical resampling; estimate and interval computed jointly, bounds are not averaged",
            "azure_fixed_prediction_behavior": "test-example resampling only",
            "prediction_source": f"{EFFECTIVE_FALLBACK_FIELD}_when_present_else_predictions",
            "valid_only_usage": "diagnostic_only",
        },
        "labels": labels,
        "macro": {"estimate": macro.observed, "low": macro.ci_low, "high": macro.ci_high},
        "prediction_hash": prediction_hash,
        "config_hash": config_hash,
        "code_commit": code_commit,
        "seed_count": len(seed_rows),
        "interval_count": len(labels) + 1,
        "result_hash": sha256_json({"labels": labels, "macro": {"estimate": macro.observed, "low": macro.ci_low, "high": macro.ci_high}, "prediction_hash": prediction_hash}),
    }


def write_q1a_confidence_intervals(run_root: str | Path, *, root: str | Path = ".") -> dict[str, Any]:
    run_root = Path(run_root)
    root = Path(root)
    path = run_root / "predictions/test_predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    config_path = run_root / "config_snapshot.yaml"
    report = evaluate_q1a_confidence_intervals(
        [rows],
        prediction_hash=sha256_file(path),
        config_hash=sha256_file(config_path) if config_path.exists() else "NOT_APPLICABLE",
        code_commit=_code_commit(root),
    )
    atomic_write_json(run_root / "metrics/test_confidence_intervals.json", report)
    return report
