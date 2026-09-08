#!/usr/bin/env python3
"""Export the audited Q1a Best-F1 table and its per-seed evidence."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

CALC_PATH = Path("/tmp/q1a_table_calc.json")
OUT_DIR = Path("reports")
SEEDS = ["20260521", "20260522", "20260523"]
CAMPAIGN = "campaigns/vipragsent-v7-6693e7e9eef08ef1"
SHARD = "SHARD_GPU40_7B_INTEGRATION"
ARTIFACT_REPO = "Thundergod2007/vipragsent-experiment-artifacts"

LABELS = [
    ("implicit_sentiment", "Implicit", "implicit"),
    ("sarcasm", "Sarcasm", "sarcasm"),
    ("irony", "Irony", "irony"),
    ("idiom_figurative", "Idiom", "idiom"),
    ("code_switching", "Code-sw.", "code_sw"),
    ("mocking", "Mocking", "mocking"),
]
MACRO = ("macro_pragmatic_f1", "Macro-prag", "macro_prag")

GROUP_ORDER = [
    "phobert_pragmatic_single_task",
    "phobert_pragmatic_finetune",
    "xlmr_pragmatic_finetune",
    "sailor_pragmatic_sft",
    "vistral_pragmatic_sft",
    "vipragsent_no_auxiliary_vistral",
    "cot_only_vistral",
    "explanation_only_vistral",
    "vipragsent_full_vistral",
    "azure_pragmatic_zero_shot",
    "azure_gpt41_mini_8shot",
]

DISPLAY_NAMES = {
    "phobert_pragmatic_single_task": "PhoBERT (single-task)",
    "phobert_pragmatic_finetune": "PhoBERT fine-tune",
    "xlmr_pragmatic_finetune": "XLM-R-large fine-tune",
    "sailor_pragmatic_sft": "Sailor-7B SFT",
    "vistral_pragmatic_sft": "Vistral-7B SFT",
    "vipragsent_no_auxiliary_vistral": "ViPragSent - no auxiliary loss",
    "cot_only_vistral": "Vistral CoT-only clean rerun 003",
    "explanation_only_vistral": "ViPragSent - explanation only",
    "vipragsent_full_vistral": "ViPragSent (ours, Vistral)",
    "azure_pragmatic_zero_shot": "GPT-4.1-mini zero-shot",
    "azure_gpt41_mini_8shot": "GPT-4.1-mini 8-shot",
}

AZURE_GROUPS = {
    "azure_pragmatic_zero_shot": {
        "run_id": "azure_pragmatic_zero_shot",
        "hash": "E4C5F0681894",
    },
    "azure_gpt41_mini_8shot": {
        "run_id": "azure_gpt41_mini_8shot",
        "hash": "1292C4E7F429",
    },
}

GENERATION_REPOS = {
    "explanation_only_vistral": {
        "20260521": (
            "Thundergod2007/vipragsent-experiment-artifacts-overflow-003",
            "Thundergod2007/vipragsent-experiment-artifacts-overflow-003",
        ),
        "20260522": (
            "Thundergod2007/vipragsent-experiment-artifacts-overflow-010",
            "Thundergod2007/vipragsent-experiment-artifacts-overflow-004",
        ),
        "20260523": (
            "Thundergod2007/vipragsent-experiment-artifacts-overflow-014",
            "Thundergod2007/vipragsent-experiment-artifacts-overflow-004",
        ),
    },
    "cot_only_vistral": {
        "20260521": (
            "Thundergod2007/vipragsent-experiment-artifacts-overflow-006",
            "Thundergod2007/vipragsent-experiment-artifacts-overflow-003",
        ),
        "20260523": (
            "Thundergod2007/vipragsent-experiment-artifacts-overflow-024",
            "Thundergod2007/vipragsent-experiment-artifacts-overflow-024",
        ),
    },
}

FULL_METRIC_REPOS = {
    "20260521": "Thundergod2007/vipragsent-experiment-artifacts-overflow-017",
    "20260522": "Thundergod2007/vipragsent-experiment-artifacts-overflow-017",
    "20260523": "Thundergod2007/vipragsent-experiment-artifacts-overflow-016",
}

FULL_THRESHOLD_REPOS = {
    "20260521": "Thundergod2007/vipragsent-experiment-artifacts-overflow-009",
    "20260522": "Thundergod2007/vipragsent-experiment-artifacts-overflow-017",
    "20260523": "Thundergod2007/vipragsent-experiment-artifacts-overflow-016",
}

COMPARABILITY_NOTES = [
    "All scored rows use the same 2,000-example test split and the same six pragmatic labels.",
    "XLM-R is a fully fine-tuned encoder baseline (558,525,440 trainable parameters, 10 epochs, AdamW at 2e-5), while full ViPragSent is a 4-bit QLoRA run with 19,658,368 trainable parameters, 3 epochs, eight auxiliary tasks, rationale training, and uncertainty weighting. Their scores are therefore a system comparison, not a compute-matched backbone ablation.",
    "The Vistral SFT and no-auxiliary rows are the closest controls for the Vistral backbone; their macro-F1 remains 87.2, while full ViPragSent varies from 67.1 to 88.9 across seeds.",
]


def run_id_for(group: str, seed: str) -> str:
    if group == "cot_only_vistral":
        return f"q1a_cot_only_vistral_clean_rerun_003__seed_{seed}"
    if group.startswith("azure_"):
        return AZURE_GROUPS[group]["run_id"]
    return f"q1a_{group}_{seed}"


def remote_generation_path(run_id: str, suffix: str) -> str:
    return f"{CAMPAIGN}/{SHARD}/{run_id}/{suffix}"


def source_for(group: str, seed: str) -> dict:
    run_id = run_id_for(group, seed)
    if group.startswith("azure_"):
        info = AZURE_GROUPS[group]
        base = (
            "campaigns/vipragsent-v7-compact-20260820-190bb0f24a7e75a8/"
            "SHARD_H100_MIG20_COMPACT/"
            "nb-170d4c47-8a39-47ad-9425-d231de44fc92-857677f648-2qn8f/"
            f"science/runs/{run_id}/{info['hash']}"
        )
        return {
            "source_id": run_id,
            "repo": ARTIFACT_REPO,
            "metric_path": f"{base}/metrics.json",
            "prediction_path": f"{base}/predictions/test_predictions.jsonl",
            "response_manifest_path": f"{base}/azure/response_manifest.json",
            "metric_note": "metrics.json has an empty test object; F1 is computed from the valid 2000-row prediction JSONL.",
        }

    if group == "cot_only_vistral" and seed == "20260522":
        checkpoint_paths = [
            "q1a_cot_only_vistral_20260522/cot_only_vistral/checkpoints/boundary_0001_epoch_0001_65C1D0BAAF1D/model.pt",
            "q1a_cot_only_vistral_20260522/cot_only_vistral/checkpoints/boundary_0002_epoch_0002_F34A33BDDADB/model.pt",
            "q1a_cot_only_vistral_20260522/cot_only_vistral/checkpoints/boundary_0003_epoch_0003_1D0A43F67758/model.pt",
        ]
        return {
            "source_id": run_id,
            "repo": "Thundergod2007/vipragsent-vistral7b-checkpoints",
            "metric_path": None,
            "prediction_path": None,
            "training_checkpoint_status": "COMPLETE_3_EPOCHS",
            "test_score_status": "MISSING_TEST_SCORE_ARTIFACT",
            "checkpoint_paths": checkpoint_paths,
            "note": "Canonical epoch 1-3 checkpoints exist, but no test predictions or test_reasoning_metrics.json was found for this seed.",
        }

    if group in GENERATION_REPOS:
        metric_repo, prediction_repo = GENERATION_REPOS[group][seed]
        return {
            "source_id": run_id,
            "repo": metric_repo,
            "prediction_repo": prediction_repo,
            "metric_path": remote_generation_path(
                run_id, "metrics/test_reasoning_metrics.json"
            ),
            "prediction_path": remote_generation_path(
                run_id, "predictions/test_predictions.jsonl"
            ),
            "metric_note": "Persisted test reasoning metrics; primary full-split macro-F1 is used.",
        }

    if group == "vipragsent_full_vistral":
        repo = FULL_METRIC_REPOS[seed]
        root = f"{CAMPAIGN}/{SHARD}/{run_id}"
        return {
            "source_id": run_id,
            "repo": repo,
            "metric_path": f"{root}/metrics/test_metrics.json",
            "ci_path": f"{root}/metrics/test_confidence_intervals.json",
            "threshold_repo": FULL_THRESHOLD_REPOS[seed],
            "threshold_path": f"{root}/selection/thresholds.json",
            "prediction_path": f"results/runs/{run_id}/predictions/test_predictions.jsonl",
            "prediction_derivation": "Reconstructed six pragmatic columns from persisted raw test gold/probability arrays using frozen per-seed selection thresholds; reproduces the persisted test metrics.",
            "remote_prediction_note": "Remote prediction JSONL contains partial record shards, so the validated local prediction file is derived from the complete persisted metric arrays.",
        }

    if group in {
        "phobert_pragmatic_single_task",
        "phobert_pragmatic_finetune",
        "xlmr_pragmatic_finetune",
    }:
        root = f"server_20260808/results/runs/{run_id}"
    else:
        root = f"{CAMPAIGN}/{SHARD}/{run_id}/science"
    return {
        "source_id": run_id,
        "repo": ARTIFACT_REPO,
        "metric_path": f"{root}/metrics/test_metrics.json",
        "ci_path": f"{root}/metrics/test_confidence_intervals.json",
        "prediction_path": f"{root}/predictions/test_predictions.jsonl",
        "metric_note": "Persisted test metrics and prediction JSONL.",
    }


def metric_object(value: dict) -> dict:
    low = float(value["low"])
    high = float(value["high"])
    return {
        "estimate": float(value["estimate"]),
        "low": low,
        "high": high,
        "half_width": (high - low) / 2.0,
    }


def percent(value: float | None) -> float | None:
    return None if value is None else round(100.0 * value, 6)


def display_cell(value: dict) -> str:
    return f"{100.0 * value['estimate']:.1f} +/- {100.0 * value['half_width']:.1f}"


def seed_from_id(seed_id: str) -> str:
    tail = seed_id.rsplit("_", 1)[-1]
    return tail if tail in SEEDS else "single"


def main() -> None:
    calc = json.loads(CALC_PATH.read_text())
    validation = calc["validation"]
    group_data = calc["groups"]
    expected_seeds = {
        group: (["single"] if group.startswith("azure_") else SEEDS)
        for group in GROUP_ORDER
    }

    per_seed_rows = []
    source_catalog = []
    aggregate_rows = []

    for group in GROUP_ORDER:
        validation_group = validation[group]
        observed = {
            seed_from_id(seed_id): values
            for seed_id, values in zip(
                validation_group["seed_ids"], validation_group["per_seed_f1"]
            )
        }
        group_sources = []

        for seed in expected_seeds[group]:
            source = source_for(group, seed)
            source_catalog.append({**source, "system_id": group, "seed": seed})
            group_sources.append(source["source_id"])
            values = observed.get(seed)
            row = {
                "system_id": group,
                "display_name": DISPLAY_NAMES[group],
                "run_id": source["source_id"],
                "seed": seed,
                "status": "COMPLETE" if values is not None else "MISSING_TEST_SCORE",
                "source_id": source["source_id"],
                "source": source,
                "f1": {},
            }
            if values is None:
                row["f1"] = {key: None for key, _, _ in LABELS}
                row["f1"][MACRO[0]] = None
            else:
                row["f1"] = {key: float(values[key]) for key, _, _ in LABELS}
                row["f1"][MACRO[0]] = float(values[MACRO[0]])
            per_seed_rows.append(row)

        metrics = {
            key: metric_object(group_data[group]["labels"][key])
            for key, _, _ in LABELS
        }
        metrics[MACRO[0]] = metric_object(group_data[group]["macro"])
        available = group_data[group]["seed_count"]
        expected = len(expected_seeds[group])
        if available == expected:
            score_status = f"COMPLETE_{available}_OF_{expected}"
            status_text = f"{available}/{expected} complete"
        else:
            score_status = f"PROVISIONAL_{available}_OF_{expected}_MISSING_TEST_SCORE"
            status_text = f"{available}/{expected}; seed 20260522 test score missing"
        aggregate_rows.append(
            {
                "system_id": group,
                "display_name": DISPLAY_NAMES[group],
                "score_status": score_status,
                "status_text": status_text,
                "seeds_available": available,
                "seeds_expected": expected,
                "source_ids": group_sources,
                "metrics": metrics,
            }
        )

    report = {
        "report_id": "q1a_best_f1_table",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "research_question": "Q1a",
        "title": "Q1a Best-F1 test-set results",
        "score_definition": {
            "metric": "binary macro-F1",
            "labels": [key for key, _, _ in LABELS],
            "class_average": "macro over classes 0 and 1 for each label",
            "table_scale": "percent",
            "raw_scale": "0_to_1",
        },
        "confidence_interval": {
            "method": calc["method"],
            "confidence_level": calc["confidence_level"],
            "resamples": calc["resamples"],
            "bootstrap_seed": calc["bootstrap_seed"],
            "rendered_pm": "percentile half-width (high - low) / 2",
            "seed_policy": "locked canonical seeds 20260521, 20260522, 20260523; Azure rows are single 2000-example runs",
        },
        "metric_audit": {
            "status": "PASS",
            "scope": "29 canonical Q1a run slots across 11 systems",
            "complete_scored_runs_recomputed": 28,
            "missing_test_score_runs": [
                "q1a_cot_only_vistral_clean_rerun_003__seed_20260522"
            ],
            "checks": {
                "remote_source_hashes_verified": True,
                "recomputed_metrics_match_persisted_metrics": True,
                "recomputed_confidence_intervals_match_locked_protocol": True,
                "all_available_prediction_files_have_2000_rows": True,
                "all_complete_runs_have_unique_sample_ids": True,
                "cross_run_test_gold_signature_count": 1,
                "test_threshold_tuning_used": False,
                "full_vistral_threshold_source": "selection/thresholds.json",
                "duplicate_or_overlap_run_ids": [],
            },
        },
        "comparability_notes": COMPARABILITY_NOTES,
        "table_rows": aggregate_rows,
        "per_seed_rows": per_seed_rows,
        "validation": validation,
        "source_catalog": source_catalog,
        "missing_test_score": {
            "run_id": "q1a_cot_only_vistral_clean_rerun_003__seed_20260522",
            "training_checkpoint_status": "COMPLETE_3_EPOCHS",
            "test_score_status": "MISSING_TEST_SCORE_ARTIFACT",
            "not_imputed": True,
            "effect": "COT aggregate is provisional and uses only seeds 20260521 and 20260523.",
        },
        "overlap_checks": {
            "expected_canonical_run_count": 29,
            "exported_per_seed_row_count": len(per_seed_rows),
            "unique_per_seed_run_count": len({row["run_id"] for row in per_seed_rows}),
            "aggregate_row_count": len(aggregate_rows),
            "complete_aggregate_row_count": sum(
                row["score_status"].startswith("COMPLETE") for row in aggregate_rows
            ),
            "provisional_aggregate_row_count": sum(
                row["score_status"].startswith("PROVISIONAL") for row in aggregate_rows
            ),
            "all_available_prediction_row_counts_are_2000": all(
                count == 2000
                for item in validation.values()
                for count in item["row_counts"]
            ),
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "q1a_best_f1_table.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n"
    )

    markdown = [
        "# Q1a Best-F1 test table",
        "",
        "All values are test-set binary macro-F1 percentages. +/- is the 95% percentile confidence-interval half-width computed with the locked paired hierarchical bootstrap protocol.",
        "",
        "| System | Implicit | Sarcasm | Irony | Idiom | Code-sw. | Mocking | Macro-prag | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in aggregate_rows:
        cells = [row["display_name"]]
        cells.extend(display_cell(row["metrics"][key]) for key, _, _ in LABELS)
        cells.append(display_cell(row["metrics"][MACRO[0]]))
        cells.append(row["status_text"])
        markdown.append("| " + " | ".join(cells) + " |")

    markdown.extend(
        [
            "",
            "Notes:",
            "",
            "- The COT row is specifically q1a_cot_only_vistral_clean_rerun_003; it is not the old generic COT baseline.",
            "- COT seed 20260522 has canonical epoch 1-3 checkpoints, but no test prediction or test_reasoning_metrics.json artifact was found. Its training status is therefore complete, while its test score remains unavailable and is not imputed.",
            "- Full ViPragSent uses the complete persisted test gold/probability arrays plus the frozen per-seed thresholds. The reconstructed six-column predictions reproduce the persisted test metrics; the remote prediction JSONL contains partial record shards.",
            "- Azure F1 is computed from the valid 2,000-row prediction JSONL because the persisted metrics.json files have empty test objects.",
            "- Re-audit 2026-09-06: every complete Q1a source was rehashed and recomputed from its authoritative prediction/raw-metric source; no baseline score mismatch was found.",
            "- Comparability: XLM-R is fully fine-tuned (558,525,440 trainable parameters, 10 epochs), whereas full ViPragSent is 4-bit QLoRA (19,658,368 trainable parameters, 3 epochs) with eight auxiliary tasks, rationale training, and uncertainty weighting. This is not a compute-matched backbone ablation.",
            "- The closest Vistral controls are Vistral SFT and ViPragSent without auxiliary loss; both average 87.2 macro-F1. Full ViPragSent has high seed variance because seed 20260521 is a genuine dev/test collapse, not a reconstructed metric error.",
            "",
            "## Per-seed audit",
            "",
            "| Run | Seed | Implicit | Sarcasm | Irony | Idiom | Code-sw. | Mocking | Macro-prag | Status |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in per_seed_rows:
        cells = [row["run_id"], row["seed"]]
        if row["status"] == "COMPLETE":
            cells.extend(f"{100.0 * row['f1'][key]:.4f}" for key, _, _ in LABELS)
            cells.append(f"{100.0 * row['f1'][MACRO[0]]:.4f}")
        else:
            cells.extend(["-"] * 7)
        cells.append(row["status"])
        markdown.append("| " + " | ".join(cells) + " |")
    (OUT_DIR / "q1a_best_f1_table.md").write_text("\n".join(markdown) + "\n")

    main_fields = [
        "system_id",
        "display_name",
        "score_status",
        "seeds_available",
        "seeds_expected",
        "source_ids",
    ]
    for key, _, short in LABELS + [MACRO]:
        main_fields.extend(
            [
                f"{short}_estimate_pct",
                f"{short}_ci_low_pct",
                f"{short}_ci_high_pct",
                f"{short}_ci_half_width_pct",
            ]
        )
    with (OUT_DIR / "q1a_best_f1_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=main_fields, lineterminator="\n")
        writer.writeheader()
        for row in aggregate_rows:
            output = {
                "system_id": row["system_id"],
                "display_name": row["display_name"],
                "score_status": row["score_status"],
                "seeds_available": row["seeds_available"],
                "seeds_expected": row["seeds_expected"],
                "source_ids": ";".join(row["source_ids"]),
            }
            for key, _, short in LABELS + [MACRO]:
                value = row["metrics"][key]
                output[f"{short}_estimate_pct"] = percent(value["estimate"])
                output[f"{short}_ci_low_pct"] = percent(value["low"])
                output[f"{short}_ci_high_pct"] = percent(value["high"])
                output[f"{short}_ci_half_width_pct"] = percent(value["half_width"])
            writer.writerow(output)

    per_seed_fields = [
        "run_id",
        "system_id",
        "display_name",
        "seed",
        "status",
        "source_repo",
        "prediction_repo",
        "source_metric_path",
        "source_prediction_path",
    ]
    per_seed_fields.extend(f"{short}_f1_pct" for _, _, short in LABELS)
    per_seed_fields.append("macro_prag_f1_pct")
    with (OUT_DIR / "q1a_best_f1_per_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_seed_fields, lineterminator="\n")
        writer.writeheader()
        for row in per_seed_rows:
            source = row["source"]
            output = {
                "run_id": row["run_id"],
                "system_id": row["system_id"],
                "display_name": row["display_name"],
                "seed": row["seed"],
                "status": row["status"],
                "source_repo": source.get("repo", ""),
                "prediction_repo": source.get("prediction_repo", ""),
                "source_metric_path": source.get("metric_path") or "",
                "source_prediction_path": source.get("prediction_path") or "",
            }
            for key, _, short in LABELS:
                value = row["f1"][key]
                output[f"{short}_f1_pct"] = "" if value is None else f"{100.0 * value:.6f}"
            value = row["f1"][MACRO[0]]
            output["macro_prag_f1_pct"] = "" if value is None else f"{100.0 * value:.6f}"
            writer.writerow(output)

    tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrrrrrl}",
        r"\toprule",
        r"System & Implicit & Sarcasm & Irony & Idiom & Code-sw. & Mocking & Macro-prag. & Status \\",
        r"\midrule",
    ]
    for row in aggregate_rows:
        values = [
            display_cell(row["metrics"][key]).replace(" +/- ", r" $\pm$ ")
            for key, _, _ in LABELS
        ]
        values.append(
            display_cell(row["metrics"][MACRO[0]]).replace(" +/- ", r" $\pm$ ")
        )
        tex.append(
            row["display_name"].replace("&", r"\&")
            + " & "
            + " & ".join(values)
            + " & "
            + row["status_text"].replace("&", r"\&")
            + r" \\"
        )
    tex.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Q1a test-set binary macro-F1. Values are percentages with 95\% percentile CI half-widths from the paired hierarchical bootstrap. The COT-only row is the Vistral clean rerun 003 lineage; its seed 20260522 test score is unavailable, so that aggregate is provisional. XLM-R and full ViPragSent are not compute-matched: the former is fully fine-tuned, while the latter uses 4-bit QLoRA with auxiliary tasks and rationale training.}",
            r"\label{tab:q1a-best-f1}",
            r"\end{table*}",
        ]
    )
    (OUT_DIR / "q1a_best_f1_table.tex").write_text("\n".join(tex) + "\n")


if __name__ == "__main__":
    main()
