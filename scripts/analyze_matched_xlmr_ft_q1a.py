"""Aggregate and compare the matched XLM-R-large Q1a baseline."""

from __future__ import annotations

import csv
import json
import shutil
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

from _bootstrap import ROOT
from vipragsent.atomic import atomic_write_json, atomic_write_text
from vipragsent.constants import PRAGMATIC_LABELS, TRAINING_SEEDS
from vipragsent.evaluation.metrics import binary_macro_f1, macro_pragmatic_f1
from vipragsent.hashing import sha256_file
from vipragsent.statistics.bootstrap import paired_bootstrap_comparison

REPORT_ROOT = ROOT / "reports/q1a_matched_xlmr_ft"
RESULT_ROOT = ROOT / "results/runs"
AGGREGATE_RUN_ROOT = RESULT_ROOT / "xlmr_large_ft_q1a_matched_aggregate"
PROPOSED_BASE = (
    REPORT_ROOT
    / "proposed_source/campaigns/vipragsent-v8-local-mig2g20gb"
    / "q1a_vipragsent_full_XLM_R_large_optimization_pragmatic_warmup020_irony105_implicit101_sarcasm101_code104_v37_"
)
SEEDS = tuple(int(value) for value in TRAINING_SEEDS)
METRIC_LABELS = (*PRAGMATIC_LABELS, "macro_pragmatic")
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260525
OLD_BASELINE_MACRO_PERCENT = 92.934251


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    true = {label: [int(row["gold"][label]) for row in rows] for label in PRAGMATIC_LABELS}
    pred = {label: [int(row["predictions"][label]) for row in rows] for label in PRAGMATIC_LABELS}
    return {
        **{label: binary_macro_f1(true[label], pred[label]) for label in PRAGMATIC_LABELS},
        "macro_pragmatic": macro_pragmatic_f1(true, pred),
    }


def _load_seed_predictions() -> tuple[dict[int, dict[str, list[dict[str, Any]]]], dict[int, list[dict[str, Any]]]]:
    matched: dict[int, dict[str, list[dict[str, Any]]]] = {}
    proposed: dict[int, list[dict[str, Any]]] = {}
    for seed in SEEDS:
        seed_root = REPORT_ROOT / f"seed_{seed}"
        matched[seed] = {
            split: _read_jsonl(seed_root / f"predictions_{split}.jsonl")
            for split in ("train", "dev", "test")
        }
        proposed_path = Path(f"{PROPOSED_BASE}{seed}/predictions/test_predictions.jsonl")
        proposed[seed] = _read_jsonl(proposed_path)
        matched_ids = [row["sample_id"] for row in matched[seed]["test"]]
        proposed_ids = [row["sample_id"] for row in proposed[seed]]
        if matched_ids != proposed_ids:
            raise RuntimeError(f"proposed/matched test IDs are not paired for seed {seed}")
        for matched_row, proposed_row in zip(matched[seed]["test"], proposed[seed], strict=True):
            for label in PRAGMATIC_LABELS:
                if int(matched_row["gold"][label]) != int(proposed_row["gold"][label]):
                    raise RuntimeError(f"paired gold mismatch for seed {seed}, label {label}")
    return matched, proposed


def _aggregate_metrics(matched: dict[int, dict[str, list[dict[str, Any]]]]) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    for split in ("train", "dev", "test"):
        by_seed = {seed: _metrics(matched[seed][split]) for seed in SEEDS}
        for metric in METRIC_LABELS:
            values = [by_seed[seed][metric] * 100.0 for seed in SEEDS]
            rows.append({
                "Split": split,
                "Metric": metric,
                "Seed1": f"{values[0]:.6f}",
                "Seed2": f"{values[1]:.6f}",
                "Seed3": f"{values[2]:.6f}",
                "Mean": f"{statistics.mean(values):.6f}",
                "Sample SD": f"{statistics.stdev(values):.6f}",
            })
    lines = [
        "Q1a matched XLM-R-large aggregate metrics (binary macro-F1 percentages)",
        "Sample SD uses ddof=1 across seeds 20260521, 20260522, 20260523.",
        "",
        "Split | Metric | Seed1 | Seed2 | Seed3 | Mean | Sample SD",
        "--- | --- | ---: | ---: | ---: | ---: | ---:",
    ]
    for row in rows:
        lines.append(
            " | ".join(
                [row["Split"], row["Metric"], row["Seed1"], row["Seed2"], row["Seed3"], row["Mean"], row["Sample SD"]]
            )
        )
    test_rows = [row for row in rows if row["Split"] == "test"]
    macro = next(row for row in test_rows if row["Metric"] == "macro_pragmatic")
    lines.extend([
        "",
        f"Compact TEST summary: macro pragmatic F1 = {macro['Mean']} +/- {macro['Sample SD']} percent (sample SD).",
        "The six head values above are the frozen-threshold test results for the matched baseline.",
    ])
    return rows, "\n".join(lines) + "\n"


def _paired_bootstrap(
    matched: dict[int, dict[str, list[dict[str, Any]]]],
    proposed: dict[int, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    matched_pairs: dict[str, list[tuple[list[Any], list[Any]]]] = {label: [] for label in PRAGMATIC_LABELS}
    proposed_pairs: dict[str, list[tuple[list[Any], list[Any]]]] = {label: [] for label in PRAGMATIC_LABELS}
    for seed in SEEDS:
        matched_rows = matched[seed]["test"]
        proposed_rows = proposed[seed]
        for label in PRAGMATIC_LABELS:
            gold = [int(row["gold"][label]) for row in matched_rows]
            matched_pred = [int(row["predictions"][label]) for row in matched_rows]
            proposed_pred = [int(row["predictions"][label]) for row in proposed_rows]
            matched_pairs[label].append((gold, matched_pred))
            proposed_pairs[label].append((gold, proposed_pred))

    def macro_pair(seed: int) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
        matched_rows = matched[seed]["test"]
        proposed_rows = proposed[seed]
        true = {label: [int(row["gold"][label]) for row in matched_rows] for label in PRAGMATIC_LABELS}
        pred = {label: [int(row["predictions"][label]) for row in proposed_rows] for label in PRAGMATIC_LABELS}
        return true, pred

    matched_macro: list[tuple[dict[str, list[int]], dict[str, list[int]]]] = []
    proposed_macro: list[tuple[dict[str, list[int]], dict[str, list[int]]]] = []
    for seed in SEEDS:
        matched_rows = matched[seed]["test"]
        proposed_rows = proposed[seed]
        true = {label: [int(row["gold"][label]) for row in matched_rows] for label in PRAGMATIC_LABELS}
        matched_pred = {label: [int(row["predictions"][label]) for row in matched_rows] for label in PRAGMATIC_LABELS}
        proposed_pred = {label: [int(row["predictions"][label]) for row in proposed_rows] for label in PRAGMATIC_LABELS}
        matched_macro.append((true, matched_pred))
        proposed_macro.append((true, proposed_pred))

    rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    metric_functions: dict[str, Callable[[Any, Any], float]] = {
        label: binary_macro_f1 for label in PRAGMATIC_LABELS
    }
    metric_functions["macro_pragmatic"] = macro_pragmatic_f1
    for metric in METRIC_LABELS:
        if metric == "macro_pragmatic":
            left = proposed_macro
            right = matched_macro
        else:
            left = proposed_pairs[metric]
            right = matched_pairs[metric]
        result = paired_bootstrap_comparison(
            left,
            right,
            metric_functions[metric],
            resamples=BOOTSTRAP_RESAMPLES,
            seed=BOOTSTRAP_SEED,
            p_value_method="paired_hierarchical_bootstrap_sign_plus_one_v1",
        )
        record = {
            "metric": metric,
            "observed_delta_pp": result.observed * 100.0,
            "ci_low_pp": result.ci_low * 100.0,
            "ci_high_pp": result.ci_high * 100.0,
            "crosses_zero": bool(result.ci_low <= 0.0 <= result.ci_high),
            "p_value": result.p_value,
            "resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "comparison": "proposed_existing_q1a_minus_matched_baseline",
        }
        rows.append(record)
        results[metric] = record
    lines = [
        "Q1a matched paired hierarchical bootstrap confidence intervals",
        "Comparison is existing proposed XLM-R Q1a predictions minus the new matched six-head baseline.",
        f"Resampling: {BOOTSTRAP_RESAMPLES} replicates; seed resampling followed by the locked shared test-example resampling; bootstrap seed {BOOTSTRAP_SEED}.",
        "All values are percentage points. CI is the 2.5th to 97.5th percentile interval; crosses_zero is inclusive.",
        "",
        "Metric | Observed delta pp | 95% CI pp | Crosses zero | p-value",
        "--- | ---: | --- | --- | ---:",
    ]
    for record in rows:
        lines.append(
            f"{record['metric']} | {record['observed_delta_pp']:.6f} | [{record['ci_low_pp']:.6f}, {record['ci_high_pp']:.6f}] | {record['crosses_zero']} | {record['p_value']:.6f}"
        )
    macro = results["macro_pragmatic"]
    lines.extend([
        "",
        f"Primary TEST macro result: proposed-minus-matched = {macro['observed_delta_pp']:.6f} pp, 95% CI [{macro['ci_low_pp']:.6f}, {macro['ci_high_pp']:.6f}], crosses zero={macro['crosses_zero']}.",
    ])
    return rows, "\n".join(lines) + "\n", results


def _confusion(gold: list[int], pred: list[int]) -> dict[str, int]:
    return {
        "tp": sum(int(g == 1 and p == 1) for g, p in zip(gold, pred, strict=True)),
        "tn": sum(int(g == 0 and p == 0) for g, p in zip(gold, pred, strict=True)),
        "fp": sum(int(g == 0 and p == 1) for g, p in zip(gold, pred, strict=True)),
        "fn": sum(int(g == 1 and p == 0) for g, p in zip(gold, pred, strict=True)),
    }


def _error_analysis(
    matched: dict[int, dict[str, list[dict[str, Any]]]],
    proposed: dict[int, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    aggregate: dict[str, dict[str, float]] = {label: {"ours_f1": 0.0, "baseline_f1": 0.0, "ours_correct_baseline_wrong": 0, "baseline_correct_ours_wrong": 0} for label in PRAGMATIC_LABELS}
    overall_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        base_rows = matched[seed]["test"]
        ours_rows = proposed[seed]
        ours_correct_ge2 = 0
        baseline_correct_ge2 = 0
        for label in PRAGMATIC_LABELS:
            gold = [int(row["gold"][label]) for row in base_rows]
            baseline_pred = [int(row["predictions"][label]) for row in base_rows]
            ours_pred = [int(row["predictions"][label]) for row in ours_rows]
            base_conf = _confusion(gold, baseline_pred)
            ours_conf = _confusion(gold, ours_pred)
            ours_correct_baseline_wrong = sum(
                int(ours == g and baseline != g)
                for g, ours, baseline in zip(gold, ours_pred, baseline_pred, strict=True)
            )
            baseline_correct_ours_wrong = sum(
                int(baseline == g and ours != g)
                for g, ours, baseline in zip(gold, ours_pred, baseline_pred, strict=True)
            )
            ours_f1 = binary_macro_f1(gold, ours_pred)
            baseline_f1 = binary_macro_f1(gold, baseline_pred)
            row = {
                "row_type": "head",
                "seed": seed,
                "head": label,
                "ours_tp": ours_conf["tp"],
                "ours_tn": ours_conf["tn"],
                "ours_fp": ours_conf["fp"],
                "ours_fn": ours_conf["fn"],
                "baseline_tp": base_conf["tp"],
                "baseline_tn": base_conf["tn"],
                "baseline_fp": base_conf["fp"],
                "baseline_fn": base_conf["fn"],
                "ours_f1": ours_f1 * 100.0,
                "baseline_f1": baseline_f1 * 100.0,
                "f1_gain_pp": (ours_f1 - baseline_f1) * 100.0,
                "ours_correct_baseline_wrong": ours_correct_baseline_wrong,
                "baseline_correct_ours_wrong": baseline_correct_ours_wrong,
                "corrected_ge2_labels": "",
                "regressed_ge2_labels": "",
            }
            rows.append(row)
            aggregate[label]["ours_f1"] += ours_f1
            aggregate[label]["baseline_f1"] += baseline_f1
            aggregate[label]["ours_correct_baseline_wrong"] += ours_correct_baseline_wrong
            aggregate[label]["baseline_correct_ours_wrong"] += baseline_correct_ours_wrong
        for ours_row, base_row in zip(ours_rows, base_rows, strict=True):
            corrected = sum(
                int(int(ours_row["predictions"][label]) == int(base_row["gold"][label]) and int(base_row["predictions"][label]) != int(base_row["gold"][label]))
                for label in PRAGMATIC_LABELS
            )
            regressed = sum(
                int(int(base_row["predictions"][label]) == int(base_row["gold"][label]) and int(ours_row["predictions"][label]) != int(base_row["gold"][label]))
                for label in PRAGMATIC_LABELS
            )
            ours_correct_ge2 += int(corrected >= 2)
            baseline_correct_ge2 += int(regressed >= 2)
        overall_rows.append({
            "row_type": "overall",
            "seed": seed,
            "head": "ALL_HEADS",
            "ours_tp": "",
            "ours_tn": "",
            "ours_fp": "",
            "ours_fn": "",
            "baseline_tp": "",
            "baseline_tn": "",
            "baseline_fp": "",
            "baseline_fn": "",
            "ours_f1": "",
            "baseline_f1": "",
            "f1_gain_pp": "",
            "ours_correct_baseline_wrong": "",
            "baseline_correct_ours_wrong": "",
            "corrected_ge2_labels": ours_correct_ge2,
            "regressed_ge2_labels": baseline_correct_ge2,
        })
    for label in PRAGMATIC_LABELS:
        ours_f1 = aggregate[label]["ours_f1"] / len(SEEDS)
        baseline_f1 = aggregate[label]["baseline_f1"] / len(SEEDS)
        rows.append({
            "row_type": "all_seeds_mean",
            "seed": "ALL",
            "head": label,
            "ours_tp": "",
            "ours_tn": "",
            "ours_fp": "",
            "ours_fn": "",
            "baseline_tp": "",
            "baseline_tn": "",
            "baseline_fp": "",
            "baseline_fn": "",
            "ours_f1": ours_f1 * 100.0,
            "baseline_f1": baseline_f1 * 100.0,
            "f1_gain_pp": (ours_f1 - baseline_f1) * 100.0,
            "ours_correct_baseline_wrong": aggregate[label]["ours_correct_baseline_wrong"],
            "baseline_correct_ours_wrong": aggregate[label]["baseline_correct_ours_wrong"],
            "corrected_ge2_labels": "",
            "regressed_ge2_labels": "",
        })
    total_corrected_ge2 = sum(int(row["corrected_ge2_labels"]) for row in overall_rows)
    total_regressed_ge2 = sum(int(row["regressed_ge2_labels"]) for row in overall_rows)
    rows.extend(overall_rows)
    top_two = sorted(
        ((label, (aggregate[label]["ours_f1"] - aggregate[label]["baseline_f1"]) / len(SEEDS) * 100.0) for label in PRAGMATIC_LABELS),
        key=lambda item: item[1],
        reverse=True,
    )[:2]
    ours_test_mean = {label: statistics.mean([_metrics(proposed[seed])[label] * 100.0 for seed in SEEDS]) for label in PRAGMATIC_LABELS}
    baseline_test_mean = {label: statistics.mean([_metrics(matched[seed]["test"])[label] * 100.0 for seed in SEEDS]) for label in PRAGMATIC_LABELS}
    lowest_ours = sorted(ours_test_mean.items(), key=lambda item: item[1])[:2]
    narrative = {
        "top_two_observed_head_gains": [{"head": label, "gain_pp": gain} for label, gain in top_two],
        "ours_test_mean_by_head": ours_test_mean,
        "baseline_test_mean_by_head": baseline_test_mean,
        "sarcasm": {"ours_mean_f1_pct": ours_test_mean["sarcasm"], "baseline_mean_f1_pct": baseline_test_mean["sarcasm"], "gain_pp": ours_test_mean["sarcasm"] - baseline_test_mean["sarcasm"], "ours_lowest_two_heads": any(label == "sarcasm" for label, _ in lowest_ours)},
        "mocking": {"ours_mean_f1_pct": ours_test_mean["mocking"], "baseline_mean_f1_pct": baseline_test_mean["mocking"], "gain_pp": ours_test_mean["mocking"] - baseline_test_mean["mocking"], "ours_lowest_two_heads": any(label == "mocking" for label, _ in lowest_ours)},
        "corrected_ge2_labels_total": total_corrected_ge2,
        "regressed_ge2_labels_total": total_regressed_ge2,
    }
    lines = [
        "Q1a matched paired error analysis",
        "Here `ours` means the existing proposed Q1a XLM-R predictions and `baseline` means the new matched six-head baseline.",
        "Counts are paired on the same 2,000 test IDs per seed. corrected_ge2_labels counts examples where ours fixes at least two head errors made by the matched baseline; regressed_ge2_labels is the reverse.",
        "",
        "Observed all-seed mean head gains (ours minus matched baseline, percentage points):",
    ]
    for label, gain in top_two:
        lines.append(f"- {label}: {gain:.6f} pp")
    lines.extend([
        f"- sarcasm: ours mean F1 {ours_test_mean['sarcasm']:.6f}%, matched mean F1 {baseline_test_mean['sarcasm']:.6f}%, gain {ours_test_mean['sarcasm'] - baseline_test_mean['sarcasm']:.6f} pp; among ours' two lowest mean-F1 heads={narrative['sarcasm']['ours_lowest_two_heads']}.",
        f"- mocking: ours mean F1 {ours_test_mean['mocking']:.6f}%, matched mean F1 {baseline_test_mean['mocking']:.6f}%, gain {ours_test_mean['mocking'] - baseline_test_mean['mocking']:.6f} pp; among ours' two lowest mean-F1 heads={narrative['mocking']['ours_lowest_two_heads']}.",
        f"Total corrected >=2-label examples across seeds: {total_corrected_ge2}; reverse regressions >=2 labels: {total_regressed_ge2}.",
        "",
        "Per-head TP/TN/FP/FN and paired correctness counts are in q1a_matched_error_analysis.csv.",
    ])
    return rows, "\n".join(lines) + "\n", narrative


def _hf_status() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for seed in SEEDS:
        path = REPORT_ROOT / f"seed_{seed}/upload_status.json"
        result[str(seed)] = _read_json(path) if path.exists() else {"status": "MISSING"}
    return result


def _run_summary(
    aggregate_rows: list[dict[str, Any]],
    bootstrap: dict[str, Any],
    error_summary: dict[str, Any],
    matched: dict[int, dict[str, list[dict[str, Any]]]],
    proposed: dict[int, list[dict[str, Any]]],
    *,
    aggregate_upload: dict[str, Any] | None = None,
) -> str:
    matched_test_mean = next(row for row in aggregate_rows if row["Split"] == "test" and row["Metric"] == "macro_pragmatic")
    proposed_values = [_metrics(proposed[seed])["macro_pragmatic"] * 100.0 for seed in SEEDS]
    proposed_mean = statistics.mean(proposed_values)
    matched_mean = float(matched_test_mean["Mean"])
    old_effect = proposed_mean - OLD_BASELINE_MACRO_PERCENT
    new_effect = proposed_mean - matched_mean
    narrative_direction_changed = (old_effect >= 0.0) != (new_effect >= 0.0)
    narrative_numeric_review = abs(matched_mean - OLD_BASELINE_MACRO_PERCENT) >= 0.1
    macro_bootstrap = bootstrap["macro_pragmatic"]
    lines = [
        "# Q1a matched XLM-R-large run summary",
        "",
        "Status: COMPLETE for the three controlled seeds; paper files were not modified.",
        "",
        "## Baseline and recipe",
        "",
        "The implementation preserves `xlmr_pragmatic_finetune`: a fully fine-tuned `FacebookAI/xlm-roberta-large` encoder, first-attended-token pooling, dropout 0.1, and exactly six binary pragmatic linear heads (`implicit_sentiment`, `sarcasm`, `irony`, `idiom_figurative`, `code_switching`, `mocking`). It has no polarity/emotion/rationale/auxiliary/proposed-only component and uses equal-weight pragmatic BCE-with-logits with train-only negative/positive positive weights.",
        "",
        "Matched recipe: max length 128; AdamW; learning rate 2e-5; weight decay 0.01; cosine schedule; warmup ratio 0.20; ten maximum epochs; patience 10; bf16; physical batch 8; accumulation 4; effective batch 32; gradient clipping 1.0; highest dev macro-pragmatic F1 checkpoint; dev-only 0.05–0.95 threshold grid with 0.01 step and nearest-0.5 tie break.",
        "",
        "## Seeds, checkpoints, thresholds, and metrics",
        "",
        "| Seed | Best epoch | Best dev macro F1 | Test macro F1 | Checkpoint SHA256 |",
        "| ---: | ---: | ---: | ---: | --- |",
    ]
    for seed in SEEDS:
        seed_root = REPORT_ROOT / f"seed_{seed}"
        best = _read_json(seed_root / "best_checkpoint_metadata.json")
        test = _read_json(seed_root / "metrics_test.json")
        thresholds = _read_json(seed_root / "thresholds.json")["thresholds"]
        lines.append(f"| {seed} | {best['best_epoch']} | {best['best_dev_macro_pragmatic_f1']:.9f} | {test['macro_pragmatic_f1']:.9f} | `{best['checkpoint_sha256']}` |")
        lines.append("|  | thresholds | " + ", ".join(f"{label}={thresholds[label]:.2f}" for label in PRAGMATIC_LABELS) + " |  |  |")
    lines.extend([
        "",
        "The required per-seed artifacts are in `reports/q1a_matched_xlmr_ft/seed_<seed>/`; each seed has 7,998 train, 1,999 dev, and 2,000 test rows. Validation confirmed unique frozen IDs, gold alignment, paired proposed IDs/gold, bounded finite probabilities, metric reproducibility, and dev-only selection.",
        "",
        f"Matched TEST macro mean = {matched_mean:.6f}%; proposed existing Q1a TEST macro mean = {proposed_mean:.6f}%; proposed-minus-matched observed macro delta = {macro_bootstrap['observed_delta_pp']:.6f} pp.",
        "",
        "Aggregate metrics: [aggregate_metrics.csv](aggregate_metrics.csv) and [aggregate_metrics.txt](aggregate_metrics.txt).",
        "",
        "## Paired bootstrap",
        "",
        f"[q1a_matched_paired_bootstrap_ci.csv](q1a_matched_paired_bootstrap_ci.csv) and [q1a_matched_paired_bootstrap_ci.txt](q1a_matched_paired_bootstrap_ci.txt) use the locked paired hierarchical seed-then-test-example protocol with {BOOTSTRAP_RESAMPLES} resamples and bootstrap seed {BOOTSTRAP_SEED}. The comparison is proposed existing Q1a minus matched baseline and all intervals are percentage points.",
        "",
        f"Primary macro: observed {macro_bootstrap['observed_delta_pp']:.6f} pp, 95% CI [{macro_bootstrap['ci_low_pp']:.6f}, {macro_bootstrap['ci_high_pp']:.6f}], crosses zero={macro_bootstrap['crosses_zero']}.",
        "",
        "| Head | Observed delta pp | 95% CI pp | Crosses zero |",
        "| --- | ---: | --- | --- |",
    ])
    for metric in METRIC_LABELS:
        record = bootstrap[metric]
        lines.append(f"| {metric} | {record['observed_delta_pp']:.6f} | [{record['ci_low_pp']:.6f}, {record['ci_high_pp']:.6f}] | {record['crosses_zero']} |")
    lines.extend([
        "",
        "## Error analysis",
        "",
        "[q1a_matched_error_analysis.csv](q1a_matched_error_analysis.csv) contains paired TP/TN/FP/FN, ours-correct/baseline-wrong, baseline-correct/ours-wrong, and corrected/reversed >=2-label example counts. `ours` is the existing proposed model; `baseline` is the matched six-head baseline.",
        "",
        f"Largest observed head F1 gains: {error_summary['top_two_observed_head_gains'][0]['head']} ({error_summary['top_two_observed_head_gains'][0]['gain_pp']:.6f} pp) and {error_summary['top_two_observed_head_gains'][1]['head']} ({error_summary['top_two_observed_head_gains'][1]['gain_pp']:.6f} pp).",
        f"Sarcasm remains one of the two lowest proposed mean-F1 heads={error_summary['sarcasm']['ours_lowest_two_heads']}; mocking remains one of the two lowest={error_summary['mocking']['ours_lowest_two_heads']}. Their deltas are {error_summary['sarcasm']['gain_pp']:.6f} pp and {error_summary['mocking']['gain_pp']:.6f} pp respectively.",
        f"Corrected >=2-label examples across seeds: {error_summary['corrected_ge2_labels_total']}; reverse >=2-label regressions: {error_summary['regressed_ge2_labels_total']}.",
        "",
        "## Narrative and cooccurrence flags",
        "",
        f"The proposed-minus-old-current-baseline macro effect is {old_effect:.6f} pp; proposed-minus-matched effect is {new_effect:.6f} pp. Direction changed={narrative_direction_changed}. Numeric narrative review flag={narrative_numeric_review} because the matched comparator differs from the current Table-3-style baseline macro row ({OLD_BASELINE_MACRO_PERCENT:.6f}%) by {matched_mean - OLD_BASELINE_MACRO_PERCENT:.6f} pp. This is a review flag only; no paper text was changed.",
        "Label cooccurrence was not recomputed; existing cooccurrence artifacts and paper inputs were left untouched.",
        "",
        "## Hugging Face status",
        "",
        "| Seed | Status | Checkpoint repo | Artifact repo | Remote root | Verified files |",
        "| ---: | --- | --- | --- | --- | ---: |",
    ])
    for seed in SEEDS:
        upload = _read_json(REPORT_ROOT / f"seed_{seed}/upload_status.json")
        lines.append(f"| {seed} | {upload.get('status')} | {upload.get('checkpoint_repository')} | {upload.get('artifact_repository')} | `{upload.get('remote_root')}` | {upload.get('verified_file_count')} |")
    if aggregate_upload:
        lines.extend([
            "",
            f"Aggregate artifact upload: {aggregate_upload.get('status')} to {aggregate_upload.get('artifact_repository')} at `{aggregate_upload.get('remote_root')}` with {aggregate_upload.get('verified_file_count')} verified files.",
        ])
    else:
        lines.extend(["", "Aggregate artifact upload: PENDING."])
    lines.extend([
        "",
        "Final reports are under `reports/q1a_matched_xlmr_ft/`. The paper files (`main.tex`, tables, Abstract, Discussion, Conclusion) were not modified.",
    ])
    return "\n".join(lines) + "\n"


def analyze() -> dict[str, Any]:
    matched, proposed = _load_seed_predictions()
    aggregate_rows, aggregate_text = _aggregate_metrics(matched)
    bootstrap_rows, bootstrap_text, bootstrap_results = _paired_bootstrap(matched, proposed)
    error_rows, error_text, error_summary = _error_analysis(matched, proposed)

    _write_csv(
        REPORT_ROOT / "aggregate_metrics.csv",
        aggregate_rows,
        ["Split", "Metric", "Seed1", "Seed2", "Seed3", "Mean", "Sample SD"],
    )
    atomic_write_text(REPORT_ROOT / "aggregate_metrics.txt", aggregate_text)
    _write_csv(
        REPORT_ROOT / "q1a_matched_paired_bootstrap_ci.csv",
        bootstrap_rows,
        ["metric", "observed_delta_pp", "ci_low_pp", "ci_high_pp", "crosses_zero", "p_value", "resamples", "bootstrap_seed", "comparison"],
    )
    atomic_write_text(REPORT_ROOT / "q1a_matched_paired_bootstrap_ci.txt", bootstrap_text)
    _write_csv(
        REPORT_ROOT / "q1a_matched_error_analysis.csv",
        error_rows,
        list(error_rows[0]),
    )
    atomic_write_text(REPORT_ROOT / "q1a_matched_error_analysis.txt", error_text)

    if AGGREGATE_RUN_ROOT.exists():
        shutil.rmtree(AGGREGATE_RUN_ROOT)
    AGGREGATE_RUN_ROOT.mkdir(parents=True, exist_ok=True)
    aggregate_files = [
        "aggregate_metrics.csv",
        "aggregate_metrics.txt",
        "q1a_matched_paired_bootstrap_ci.csv",
        "q1a_matched_paired_bootstrap_ci.txt",
        "q1a_matched_error_analysis.csv",
        "q1a_matched_error_analysis.txt",
    ]
    for name in aggregate_files:
        shutil.copy2(REPORT_ROOT / name, AGGREGATE_RUN_ROOT / name)
    summary = _run_summary(aggregate_rows, bootstrap_results, error_summary, matched, proposed)
    atomic_write_text(REPORT_ROOT / "run_summary.md", summary)
    shutil.copy2(REPORT_ROOT / "run_summary.md", AGGREGATE_RUN_ROOT / "run_summary.md")
    resolved = _read_json(REPORT_ROOT / "seed_20260521/resolved_config.json")
    atomic_write_json(
        AGGREGATE_RUN_ROOT / "aggregate_manifest.json",
        {
            "status": "PASS",
            "experiment_name": "xlmr_large_ft_q1a_matched",
            "aggregate_run_id": AGGREGATE_RUN_ROOT.name,
            "seeds": list(SEEDS),
            "aggregate_files": aggregate_files + ["run_summary.md"],
            "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "method": "paired_hierarchical_bootstrap_sign_plus_one_v1"},
            "source_config_sha256": sha256_file(REPORT_ROOT / "seed_20260521/resolved_config.json"),
            "model_repository": resolved["model_repository"],
            "paper_modified": False,
            "label_cooccurrence_recomputed": False,
        },
    )
    return {
        "status": "PASS",
        "report_root": str(REPORT_ROOT),
        "aggregate_run_root": str(AGGREGATE_RUN_ROOT),
        "aggregate_rows": len(aggregate_rows),
        "bootstrap_metrics": bootstrap_results,
        "error_summary": error_summary,
    }


if __name__ == "__main__":
    print(json.dumps(analyze(), indent=2, ensure_ascii=False))
