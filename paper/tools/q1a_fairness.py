"""Row-level Q1a fairness and recorded-metric verification.

This script only reads supplied/local artifacts and writes reports below
the paper/evidence directory. It never trains or invokes a model.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, pstdev, stdev
from typing import Any


PAPER = Path(r"D:\vipragsent\paper")
RAW = PAPER / "raw_fairness"
EVIDENCE = PAPER / "evidence"
PACKAGE = Path(
    r"D:\vipragsent-pr\reports\hf_vipragsent_naacl_comparison_2026-09-15"
)
REFERENCE_TEST_CSVS = (
    Path(r"D:\vipragsent\data\processed\vipragsent\test.csv"),
    Path(r"D:\vipragsent-pr\data\processed\vipragsent\test.csv"),
)

PRAGMATIC_LABELS = (
    "implicit_sentiment",
    "sarcasm",
    "irony",
    "idiom_figurative",
    "code_switching",
    "mocking",
)

TARGET_RUNS = {
    21: {
        "run_id": (
            "q1a_vipragsent_full_XLM_R_large_optimization_pragmatic_"
            "warmup020_irony105_implicit101_sarcasm101_code104_v37_20260521"
        ),
        "prediction_file": RAW / "target_21.jsonl",
        "metric_file": RAW / "target_21_metrics.json",
    },
    22: {
        "run_id": (
            "q1a_vipragsent_full_XLM_R_large_optimization_pragmatic_"
            "warmup020_irony105_implicit101_sarcasm101_code104_v37_20260522"
        ),
        "prediction_file": RAW / "target_22.jsonl",
        "metric_file": RAW / "target_22_metrics.json",
    },
    23: {
        "run_id": (
            "q1a_vipragsent_full_XLM_R_large_optimization_pragmatic_"
            "warmup020_irony105_implicit101_sarcasm101_code104_v37_20260523"
        ),
        "prediction_file": RAW / "target_23.jsonl",
        "metric_file": RAW / "target_23_metrics.json",
    },
}

BASELINE_RUNS = {
    "phobert_single": {
        seed: {
            "run_id": f"q1a_phobert_pragmatic_single_task_202605{seed:02d}",
            "prediction_file": RAW / f"phobert_single_{seed}.jsonl",
            "metric_file": RAW / f"phobert_single_{seed}_metrics.json",
        }
        for seed in (21, 22, 23)
    },
    "phobert_finetune": {
        seed: {
            "run_id": f"q1a_phobert_pragmatic_finetune_202605{seed:02d}",
            "prediction_file": RAW / f"phobert_finetune_{seed}.jsonl",
            "metric_file": RAW / f"phobert_finetune_{seed}_metrics.json",
        }
        for seed in (21, 22, 23)
    },
    "xlmr_baseline": {
        seed: {
            "run_id": f"q1a_xlmr_pragmatic_finetune_202605{seed:02d}",
            "prediction_file": RAW / f"xlmr_baseline_{seed}.jsonl",
            "metric_file": RAW / f"xlmr_baseline_{seed}_metrics.json",
        }
        for seed in (21, 22, 23)
    },
    "sailor": {
        seed: {
            "run_id": f"q1a_sailor_pragmatic_sft_202605{seed:02d}",
            "prediction_file": RAW / f"sailor_{seed}.jsonl",
            "metric_file": RAW / f"sailor_{seed}_metrics.json",
        }
        for seed in (21, 22, 23)
    },
    "vistral": {
        seed: {
            "run_id": f"q1a_vistral_pragmatic_sft_202605{seed:02d}",
            "prediction_file": RAW / f"vistral_{seed}.jsonl",
            "metric_file": RAW / f"vistral_{seed}_metrics.json",
        }
        for seed in (21, 22, 23)
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_lines(values: list[Any]) -> str:
    payload = "\n".join(
        json.dumps(value, sort_keys=True, separators=(",", ":"))
        for value in values
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            rows.append(value)
    return rows


def f1_for_class(true: list[int], pred: list[int], positive: int) -> float:
    tp = sum(a == positive and b == positive for a, b in zip(true, pred))
    fp = sum(a != positive and b == positive for a, b in zip(true, pred))
    fn = sum(a == positive and b != positive for a, b in zip(true, pred))
    denominator = 2 * tp + fp + fn
    return (2 * tp / denominator) if denominator else 0.0


def binary_macro_f1(true: list[int], pred: list[int]) -> float:
    if len(true) != len(pred):
        raise ValueError("true and pred lengths differ")
    return (f1_for_class(true, pred, 0) + f1_for_class(true, pred, 1)) / 2.0


def score_rows(rows: list[dict[str, Any]]) -> tuple[dict[str, float], float]:
    per_label: dict[str, float] = {}
    for label in PRAGMATIC_LABELS:
        true = [int(row["gold"][label]) for row in rows]
        pred = [int(row["predictions"][label]) for row in rows]
        per_label[label] = binary_macro_f1(true, pred)
    return per_label, mean(per_label.values())


def row_signature(row: dict[str, Any], field: str) -> tuple[int, ...]:
    return tuple(int(row[field][label]) for label in PRAGMATIC_LABELS)


def validate_rows(rows: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    ids = [str(row.get("sample_id", "")) for row in rows]
    duplicate_ids = sorted({sample_id for sample_id in ids if ids.count(sample_id) > 1})
    missing_id_rows = [index for index, sample_id in enumerate(ids) if not sample_id]
    invalid_gold: list[dict[str, Any]] = []
    invalid_predictions: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        for field, invalid in (
            ("gold", invalid_gold),
            ("predictions", invalid_predictions),
        ):
            values = row.get(field, {})
            for label in PRAGMATIC_LABELS:
                value = values.get(label) if isinstance(values, dict) else None
                if value not in (0, 1):
                    invalid.append(
                        {"row_index": index, "label": label, "value": value}
                    )
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "row_count": len(rows),
        "unique_id_count": len(set(ids)),
        "duplicate_id_examples": duplicate_ids[:10],
        "missing_id_row_indices": missing_id_rows[:10],
        "invalid_gold_examples": invalid_gold[:10],
        "invalid_prediction_examples": invalid_predictions[:10],
        "all_rows_structurally_valid": (
            len(rows) == 2000
            and len(set(ids)) == len(ids)
            and not missing_id_rows
            and not duplicate_ids
            and not invalid_gold
            and not invalid_predictions
        ),
        "sample_id_sequence_sha256": sha256_lines(ids),
        "pragmatic_gold_sequence_sha256": sha256_lines(
            [row_signature(row, "gold") for row in rows]
        ),
        "pragmatic_prediction_sequence_sha256": sha256_lines(
            [row_signature(row, "predictions") for row in rows]
        ),
        "gold_keys_observed": sorted(
            {
                key
                for row in rows
                for key in row.get("gold", {})
            }
        ),
        "prediction_keys_observed": sorted(
            {
                key
                for row in rows
                for key in row.get("predictions", {})
            }
        ),
        "text_field_present_in_any_prediction_row": any(
            "text" in row for row in rows
        ),
        "token_ids_present_in_any_prediction_row": any(
            "input_ids" in row or "token_ids" in row for row in rows
        ),
    }


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                run_id = record.get("run_id")
                if run_id:
                    records[run_id] = record
    return records


def load_source_manifest(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    output: dict[str, dict[str, dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            run_id = record.get("run_id")
            kind = record.get("kind")
            if run_id and kind in {"metric", "prediction"}:
                output.setdefault(run_id, {})[kind] = record
    return output


def compare_rows(
    left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    left_ids = [str(row["sample_id"]) for row in left_rows]
    right_ids = [str(row["sample_id"]) for row in right_rows]
    left_by_id = {str(row["sample_id"]): row for row in left_rows}
    right_by_id = {str(row["sample_id"]): row for row in right_rows}
    left_set = set(left_by_id)
    right_set = set(right_by_id)
    common_ids = sorted(left_set & right_set)
    left_only = sorted(left_set - right_set)
    right_only = sorted(right_set - left_set)
    gold_mismatches = [
        sample_id
        for sample_id in common_ids
        if row_signature(left_by_id[sample_id], "gold")
        != row_signature(right_by_id[sample_id], "gold")
    ]
    prediction_differences = [
        sample_id
        for sample_id in common_ids
        if row_signature(left_by_id[sample_id], "predictions")
        != row_signature(right_by_id[sample_id], "predictions")
    ]
    return {
        "left_row_count": len(left_rows),
        "right_row_count": len(right_rows),
        "id_sequence_equal": left_ids == right_ids,
        "id_set_equal": left_set == right_set,
        "left_only_id_count": len(left_only),
        "right_only_id_count": len(right_only),
        "left_only_id_examples": left_only[:10],
        "right_only_id_examples": right_only[:10],
        "common_id_count": len(common_ids),
        "gold_by_id_equal_on_common_ids": not gold_mismatches,
        "gold_mismatch_count_on_common_ids": len(gold_mismatches),
        "gold_mismatch_id_examples": gold_mismatches[:10],
        "pragmatic_id_gold_cohort_equivalent": (
            left_set == right_set and not gold_mismatches
        ),
        "prediction_difference_count_on_common_ids": len(prediction_differences),
        "prediction_difference_id_examples": prediction_differences[:10],
    }


def inspect_reference_test_csv(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    ids = [str(row.get("sample_id", "")) for row in rows]
    texts = [str(row.get("text", "")) for row in rows]
    return {
        "path": str(path),
        "exists": True,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "row_count": len(rows),
        "text_column_present": "text" in (reader.fieldnames or []),
        "sample_id_sequence_sha256": sha256_lines(ids),
        "text_sequence_sha256": sha256_lines(texts),
        "text_available_for_local_reference_only": True,
    }


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    all_run_specs: list[tuple[str, str, int, dict[str, Any]]] = []
    for seed, spec in TARGET_RUNS.items():
        all_run_specs.append(("primary", "ViPragSent (XLM-R-large)", seed, spec))
    for family, seed_specs in BASELINE_RUNS.items():
        for seed, spec in seed_specs.items():
            all_run_specs.append(("baseline", family, seed, spec))

    verification_records = load_records(
        PACKAGE / "artifact_verification_records.jsonl"
    )
    source_records = load_source_manifest(PACKAGE / "source_manifest.jsonl")

    loaded: dict[str, dict[str, Any]] = {}
    run_table: list[dict[str, Any]] = []
    for scope, system, seed, spec in all_run_specs:
        prediction_file = spec["prediction_file"]
        metric_file = spec["metric_file"]
        rows = load_jsonl(prediction_file)
        metric = json.loads(metric_file.read_text(encoding="utf-8"))
        per_label, recomputed_macro = score_rows(rows)
        recorded_macro = float(metric["macro_pragmatic_f1"])
        recorded_per_label = {
            label: float(metric["per_label_f1"][label])
            for label in PRAGMATIC_LABELS
        }
        run_id = spec["run_id"]
        verification = verification_records.get(run_id, {})
        source = source_records.get(run_id, {})
        rows_report = validate_rows(rows, prediction_file)
        loaded[run_id] = {
            "rows": rows,
            "scope": scope,
            "system": system,
            "seed": seed,
            "run_id": run_id,
        }
        run_table.append(
            {
                "scope": scope,
                "system": system,
                "seed": seed,
                "remote_seed": 20260500 + seed,
                "run_id": run_id,
                "prediction_file": str(prediction_file),
                "metric_file": str(metric_file),
                "prediction_file_sha256": rows_report["sha256"],
                "prediction_file_bytes": rows_report["bytes"],
                "recorded_verification_state": verification.get(
                    "verification_state", "NOT_IN_PACKAGE_RECORD"
                ),
                "recorded_artifact_status": verification.get("artifact_status"),
                "dataset_fingerprint": verification.get("dataset_fingerprint"),
                "independent_rerun": verification.get(
                    "independent_rerun", "NOT_RECORDED"
                ),
                "artifact_repo": verification.get("repo_id"),
                "artifact_source_metric": source.get("metric", {}).get("source_path"),
                "artifact_source_prediction": source.get("prediction", {}).get(
                    "source_path"
                ),
                "rows_structurally_valid": rows_report[
                    "all_rows_structurally_valid"
                ],
                "row_count": rows_report["row_count"],
                "unique_id_count": rows_report["unique_id_count"],
                "sample_id_sequence_sha256": rows_report[
                    "sample_id_sequence_sha256"
                ],
                "pragmatic_gold_sequence_sha256": rows_report[
                    "pragmatic_gold_sequence_sha256"
                ],
                "pragmatic_prediction_sequence_sha256": rows_report[
                    "pragmatic_prediction_sequence_sha256"
                ],
                "recorded_macro_pragmatic_f1": recorded_macro,
                "recomputed_macro_pragmatic_f1": recomputed_macro,
                "macro_absolute_delta": recomputed_macro - recorded_macro,
                "recorded_per_label_f1": recorded_per_label,
                "recomputed_per_label_f1": per_label,
                "per_label_absolute_delta": {
                    label: per_label[label] - recorded_per_label[label]
                    for label in PRAGMATIC_LABELS
                },
                "metric_recompute_matches_recorded": (
                    abs(recomputed_macro - recorded_macro) <= 1e-12
                    and all(
                        abs(per_label[label] - recorded_per_label[label]) <= 1e-12
                        for label in PRAGMATIC_LABELS
                    )
                ),
                "recompute_kind": "DETERMINISTIC_RECOMPUTE_FROM_RECORDED_PREDICTION_ROWS",
                "independent_inference_or_training": False,
                "metric_prediction_count": metric.get("prediction_count"),
                "metric_thresholds": metric.get("thresholds"),
                "metric_thresholds_source": metric.get("thresholds_source"),
                "metric_test_threshold_tuning": metric.get("test_threshold_tuning"),
            }
        )

    target_ids = list(TARGET_RUNS)
    baseline_ids = [
        (family, seed)
        for family, seed_specs in BASELINE_RUNS.items()
        for seed in seed_specs
    ]
    pairwise: list[dict[str, Any]] = []
    for target_seed in target_ids:
        target_run_id = TARGET_RUNS[target_seed]["run_id"]
        target_rows = loaded[target_run_id]["rows"]
        for family, baseline_seed in baseline_ids:
            baseline_run_id = BASELINE_RUNS[family][baseline_seed]["run_id"]
            baseline_rows = loaded[baseline_run_id]["rows"]
            comparison = compare_rows(target_rows, baseline_rows)
            pairwise.append(
                {
                    "target_seed": target_seed,
                    "target_run_id": target_run_id,
                    "baseline_family": family,
                    "baseline_seed": baseline_seed,
                    "baseline_run_id": baseline_run_id,
                    **comparison,
                }
            )

    equivalence_values = [
        item["pragmatic_id_gold_cohort_equivalent"] for item in pairwise
    ]
    all_equivalent = bool(equivalence_values) and all(equivalence_values)
    by_family: dict[str, dict[str, Any]] = {}
    for family in BASELINE_RUNS:
        family_pairs = [
            item for item in pairwise if item["baseline_family"] == family
        ]
        by_family[family] = {
            "all_three_seeds_equivalent_to_each_target": all(
                item["pragmatic_id_gold_cohort_equivalent"]
                for item in family_pairs
            ),
            "pairs": family_pairs,
        }

    primary_values = [
        row["recorded_macro_pragmatic_f1"]
        for row in run_table
        if row["scope"] == "primary"
    ]
    xlmr_baseline_values = [
        row["recorded_macro_pragmatic_f1"]
        for row in run_table
        if row["system"] == "xlmr_baseline"
    ]
    system_summaries: dict[str, dict[str, Any]] = {}
    for system in sorted({row["system"] for row in run_table}):
        values = [
            row["recorded_macro_pragmatic_f1"]
            for row in run_table
            if row["system"] == system
        ]
        system_summaries[system] = {
            "n_seeds": len(values),
            "seed_values": values,
            "mean_macro_pragmatic_f1": mean(values),
            "sample_sd_macro_pragmatic_f1": stdev(values),
            "population_sd_macro_pragmatic_f1": pstdev(values),
            "sd_convention_for_table": "sample SD (n-1)",
        }
    headline = {
        "primary_system": "ViPragSent (XLM-R-large)",
        "baseline_system": "XLM-R-large (baseline)",
        "primary_seed_values": primary_values,
        "baseline_seed_values": xlmr_baseline_values,
        "primary_mean_macro_pragmatic_f1": mean(primary_values),
        "baseline_mean_macro_pragmatic_f1": mean(xlmr_baseline_values),
        "primary_sd_sample_macro_pragmatic_f1": stdev(primary_values),
        "baseline_sd_sample_macro_pragmatic_f1": stdev(xlmr_baseline_values),
        "primary_sd_population_macro_pragmatic_f1": pstdev(primary_values),
        "baseline_sd_population_macro_pragmatic_f1": pstdev(xlmr_baseline_values),
        "table_sd_convention": "sample SD (n-1), statistics.stdev",
        "difference_points": 100
        * (mean(primary_values) - mean(xlmr_baseline_values)),
        "headline_status": (
            "RECORDED_ARTIFACT_METRICS_AND_DETERMINISTIC_RECOMPUTE"
        ),
        "not_an_independent_rerun": True,
    }

    report = {
        "report_type": "Q1a row-level test fairness and metric check",
        "created_by": "evidence_protocol_specialist",
        "created_utc": "2026-09-17",
        "scope": {
            "question": "Q1a",
            "evaluated_labels": list(PRAGMATIC_LABELS),
            "rows_required_per_run": 2000,
            "comparison_rule": (
                "The pragmatic ID-gold cohort is equivalent when sample-ID "
                "sets match exactly and all six pragmatic gold labels match "
                "by ID. This is not proof of identical raw text or tokenization."
            ),
            "score_rule": (
                "Per-label binary macro-F1 averages class-0 and class-1 F1; "
                "macro_pragmatic_f1 averages the six labels, matching "
                "src/vipragsent/evaluation/metrics.py."
            ),
        },
        "dataset_fingerprint_interpretation": {
            "target_fingerprint": (
                "B906C090400BAE115C9C5E3C35E32FA410AC519AE09209EBA741F198087C24F9"
            ),
            "standard_baseline_fingerprint": (
                "A13573E38550ABD55D7F63E983C602BEA13C2765A1FFECA285F718478532AF0D"
            ),
            "fingerprints_equal": False,
            "interpretation": (
                "The recorded preprocessing/data fingerprints differ. "
                "Row-level pragmatic test equivalence is therefore checked "
                "explicitly rather than inferred from fingerprints."
            ),
        },
        "test_input_text_evidence": {
            "prediction_rows_contain_text": False,
            "prediction_rows_contain_token_ids": False,
            "reference_test_csvs": [
                inspect_reference_test_csv(path) for path in REFERENCE_TEST_CSVS
            ],
            "reference_csvs_byte_identical": (
                len(REFERENCE_TEST_CSVS) == 2
                and all(path.exists() for path in REFERENCE_TEST_CSVS)
                and sha256_file(REFERENCE_TEST_CSVS[0])
                == sha256_file(REFERENCE_TEST_CSVS[1])
            ),
            "artifact_text_equivalence_verified": False,
            "tokenized_input_equivalence_verified": False,
            "interpretation": (
                "The stored prediction JSONL exposes sample_id, gold, "
                "predictions, probabilities, and logits but no text or "
                "token IDs. Equal IDs and pragmatic gold labels establish "
                "the evaluated ID-gold cohort only. Local processed test.csv "
                "files are reference data, not an attestation that every "
                "remote artifact used identical text/tokenization."
            ),
        },
        "test_cohort_vs_training_protocol": {
            "cohort_gate_does_not_prove_training_protocol_identity": True,
            "training_protocol_comparison_status": (
                "REPORTED_SEPARATELY_IN_METHOD_EVIDENCE"
            ),
            "interpretation": (
                "A common ID-gold evaluation cohort does not make the "
                "primary and baseline training variants the same protocol."
            ),
        },
        "headline": headline,
        "system_summaries": system_summaries,
        "runs": run_table,
        "pairwise_target_baseline": pairwise,
        "by_baseline_family": by_family,
        "decision": {
            "target_baseline_pair_count": len(pairwise),
            "all_target_baseline_pairs_pragmatic_id_gold_cohort_equivalent": (
                all_equivalent
            ),
            "direct_q1a_score_comparison_justified_for_pragmatic_id_gold_cohort": (
                all_equivalent
            ),
            "same_input_text_or_tokenization_verified": False,
            "training_protocol_identity_established_by_this_gate": False,
            "reason": (
                "All 45 target/baseline pairs have exact pragmatic test-ID "
                "sets and identical six-label gold tuples by sample ID; "
                "the claim is intentionally limited to the ID-gold cohort."
                if all_equivalent
                else (
                    "At least one target/baseline pair differs in IDs or "
                    "pragmatic gold labels; report raw scores separately and "
                    "do not claim a direct same-test comparison."
                )
            ),
            "prediction_rows_are_not_independent_reruns": True,
        },
    }

    (EVIDENCE / "q1a_fairness_comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with (EVIDENCE / "q1a_fairness_run_table.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = [
            "scope",
            "system",
            "seed",
            "remote_seed",
            "run_id",
            "dataset_fingerprint",
            "recorded_verification_state",
            "independent_rerun",
            "row_count",
            "unique_id_count",
            "prediction_file_bytes",
            "prediction_file_sha256",
            "sample_id_sequence_sha256",
            "pragmatic_gold_sequence_sha256",
            "pragmatic_prediction_sequence_sha256",
            "recorded_macro_pragmatic_f1",
            "recomputed_macro_pragmatic_f1",
            "macro_absolute_delta",
            "metric_recompute_matches_recorded",
            "recompute_kind",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {
                key: row.get(key)
                for key in fieldnames
            }
            for row in run_table
        )

    verified_table_inputs = {
        "table_id": "Q1a",
        "metric_name": "macro_pragmatic_f1",
        "unit": "proportion",
        "display_percent_rule": "100 * proportion; round only for display",
        "summary_statistics": {
            "sd_convention": "sample SD (n-1), statistics.stdev",
            "systems": system_summaries,
        },
        "primary": [
            {
                "seed": row["seed"],
                "remote_seed": row["remote_seed"],
                "run_id": row["run_id"],
                "value": row["recorded_macro_pragmatic_f1"],
                "value_recomputed_from_rows": row[
                    "recomputed_macro_pragmatic_f1"
                ],
                "source_file": row["metric_file"],
                "prediction_file": row["prediction_file"],
                "verification_state": row["recorded_verification_state"],
                "independent_rerun": row["independent_rerun"],
            }
            for row in run_table
            if row["scope"] == "primary"
        ],
        "baselines": [
            {
                "system": row["system"],
                "seed": row["seed"],
                "remote_seed": row["remote_seed"],
                "run_id": row["run_id"],
                "value": row["recorded_macro_pragmatic_f1"],
                "value_recomputed_from_rows": row[
                    "recomputed_macro_pragmatic_f1"
                ],
                "source_file": row["metric_file"],
                "prediction_file": row["prediction_file"],
                "verification_state": row["recorded_verification_state"],
                "independent_rerun": row["independent_rerun"],
            }
            for row in run_table
            if row["scope"] == "baseline"
        ],
        "fairness_gate": {
            "target_baseline_pair_count": len(pairwise),
            "all_pairs_pragmatic_id_gold_cohort_equivalent": all_equivalent,
            "direct_comparison_allowed_for_pragmatic_metric": all_equivalent,
            "same_input_text_or_tokenization_verified": False,
        },
        "provenance_note": (
            "Values are recorded artifact metrics and deterministic "
            "recomputations from the supplied prediction rows. No model was "
            "reloaded, trained, or independently inferred. The fairness "
            "gate verifies an ID-gold cohort, not raw text/tokenization or "
            "training-protocol identity."
        ),
    }
    (EVIDENCE / "q1a_verified_table_inputs.json").write_text(
        json.dumps(verified_table_inputs, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    numeric_ledger = {
        "ledger_type": "numeric_claim_ledger",
        "question": "Q1a",
        "claims": [
            {
                "claim_id": "Q1A-HEADLINE-PRIMARY",
                "claim": (
                    "The recorded three-seed ViPragSent XLM-R-large "
                    "macro-pragmatic-F1 mean is 0.9366583803341149 "
                    "(93.66583803341149%)."
                ),
                "value": headline["primary_mean_macro_pragmatic_f1"],
                "display_percent": 100
                * headline["primary_mean_macro_pragmatic_f1"],
                "authority": "recorded metric JSON files plus row recomputation",
                "source_run_ids": [
                    row["run_id"]
                    for row in run_table
                    if row["scope"] == "primary"
                ],
                "status": "RECORDED_AND_DETERMINISTICALLY_RECOMPUTED",
                "independent_rerun": "NOT_PERFORMED",
            },
            {
                "claim_id": "Q1A-HEADLINE-XLMR-BASELINE",
                "claim": (
                    "The recorded three-seed XLM-R pragmatic baseline "
                    "macro-pragmatic-F1 mean is 0.929342505569368 "
                    "(92.9342505569368%)."
                ),
                "value": headline["baseline_mean_macro_pragmatic_f1"],
                "display_percent": 100
                * headline["baseline_mean_macro_pragmatic_f1"],
                "authority": "recorded metric JSON files plus row recomputation",
                "source_run_ids": [
                    row["run_id"]
                    for row in run_table
                    if row["system"] == "xlmr_baseline"
                ],
                "status": "RECORDED_AND_DETERMINISTICALLY_RECOMPUTED",
                "independent_rerun": "NOT_PERFORMED",
            },
            {
                "claim_id": "Q1A-HEADLINE-GAP",
                "claim": (
                    "The recorded primary-minus-XLM-R-baseline mean gap is "
                    "0.73158747647469 percentage points."
                ),
                "value_percentage_points": headline["difference_points"],
                "authority": "derived from the two recorded three-seed means",
                "status": "DERIVED_FROM_RECORDED_VALUES",
                "independent_rerun": "NOT_APPLICABLE",
            },
            {
                "claim_id": "Q1A-PRIMARY-SAMPLE-SD",
                "claim": (
                    "The three-seed primary macro-pragmatic-F1 sample SD "
                    "is derived from the recorded per-seed metrics."
                ),
                "value": headline["primary_sd_sample_macro_pragmatic_f1"],
                "display_percent": 100
                * headline["primary_sd_sample_macro_pragmatic_f1"],
                "authority": "statistics.stdev over three recorded seed values",
                "status": "DERIVED_FROM_RECORDED_VALUES",
                "independent_rerun": "NOT_APPLICABLE",
            },
            {
                "claim_id": "Q1A-XLMR-BASELINE-SAMPLE-SD",
                "claim": (
                    "The three-seed XLM-R baseline macro-pragmatic-F1 "
                    "sample SD is derived from the recorded per-seed metrics."
                ),
                "value": headline["baseline_sd_sample_macro_pragmatic_f1"],
                "display_percent": 100
                * headline["baseline_sd_sample_macro_pragmatic_f1"],
                "authority": "statistics.stdev over three recorded seed values",
                "status": "DERIVED_FROM_RECORDED_VALUES",
                "independent_rerun": "NOT_APPLICABLE",
            },
        ],
    }
    (EVIDENCE / "q1a_numeric_claim_ledger.json").write_text(
        json.dumps(numeric_ledger, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "target_baseline_pair_count": len(pairwise),
                "all_pairs_pragmatic_id_gold_cohort_equivalent": all_equivalent,
                "primary_mean": headline["primary_mean_macro_pragmatic_f1"],
                "xlmr_baseline_mean": headline["baseline_mean_macro_pragmatic_f1"],
                "primary_sample_sd": headline[
                    "primary_sd_sample_macro_pragmatic_f1"
                ],
                "xlmr_baseline_sample_sd": headline[
                    "baseline_sd_sample_macro_pragmatic_f1"
                ],
                "difference_points": headline["difference_points"],
                "output_files": [
                    str(EVIDENCE / "q1a_fairness_comparison.json"),
                    str(EVIDENCE / "q1a_fairness_run_table.csv"),
                    str(EVIDENCE / "q1a_verified_table_inputs.json"),
                    str(EVIDENCE / "q1a_numeric_claim_ledger.json"),
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
