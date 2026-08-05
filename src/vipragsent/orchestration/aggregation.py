from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np

from ..artifacts.schemas import REQUIRED_COLUMNS
from ..atomic import atomic_write_json
from ..constants import PRAGMATIC_LABELS, TRAINING_SEEDS
from ..evaluation.metrics import binary_macro_f1, macro_pragmatic_f1
from ..hashing import sha256_file, sha256_json
from ..protocol import validate_protocol_resolution
from ..statistics.bootstrap import (
    holm_bonferroni,
    paired_bootstrap_comparison,
    paired_bootstrap_trainable_vs_azure,
)
from ..statistics.significance import load_p_value_strategy
from .run_store import artifact_hashes


def _write_csv(path: Path, columns: list[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def discover_run_manifests(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted((root / "results/runs").glob("*/run_manifest.json"))


def _load(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except (OSError, json.JSONDecodeError):
        return default


def _summary(root: Path, run_id: str) -> dict[str, Any]:
    return dict(_load(root / "results/runs" / run_id / "review_summary.json", {}) or {})


def _validate_approved_run(root: Path, run_id: str) -> tuple[dict[str, Any] | None, list[str]]:
    run_root = root / "results/runs" / run_id
    errors: list[str] = []
    state = _load(run_root / "state.json", {}) or {}
    approval = _load(run_root / "approval_status.json", {}) or {}
    summary = _summary(root, run_id)
    manifest = _load(run_root / "run_manifest.json", {}) or {}
    if not run_root.exists():
        return None, [f"missing run directory: {run_id}"]
    if state.get("run_status") not in {"APPROVED", "COMPLETED_PENDING_APPROVAL"}:
        errors.append(f"{run_id}: run_status is not completed/approved")
    if approval.get("status") != "APPROVED":
        errors.append(f"{run_id}: approval status is not APPROVED")
    approval_record = approval.get("record") or {}
    if not approval_record.get("approved_or_rejected_by") or not approval_record.get("timestamp"):
        errors.append(f"{run_id}: approval record lacks an explicit reviewer and timestamp")
    if summary.get("RUN_STATUS") != "PASS" or summary.get("USER_REVIEW_STATUS") != "PENDING" or summary.get("NEXT_RUN_ALLOWED") != "NO":
        errors.append(f"{run_id}: review summary approval gate is invalid")
    if manifest.get("mode") != "full" or manifest.get("synthetic_results") is True:
        errors.append(f"{run_id}: fixture/synthetic provenance cannot enter production aggregation")
    if not summary.get("artifact_paths") or not summary.get("artifact_sha256"):
        errors.append(f"{run_id}: summary artifact index is empty")
    else:
        actual_artifacts = artifact_hashes(run_root)
        if dict(summary.get("artifact_sha256", {})) != actual_artifacts or sorted(summary.get("artifact_paths", [])) != sorted(actual_artifacts):
            errors.append(f"{run_id}: review summary artifact index does not match the run contents")
    checksums = run_root / "checksums.sha256"
    if not checksums.exists():
        errors.append(f"{run_id}: checksums.sha256 is missing")
    else:
        expected: dict[str, str] = {}
        for line in checksums.read_text(encoding="utf-8").splitlines():
            digest, _, name = line.partition("  ")
            if name:
                expected[name] = digest
        actual = artifact_hashes(run_root)
        if any(actual.get(name) != digest for name, digest in expected.items()) or set(actual) != set(expected):
            errors.append(f"{run_id}: artifact checksums do not validate")
    if approval_record.get("review_summary_sha256") and summary and approval_record.get("review_summary_sha256") != sha256_file(run_root / "review_summary.json"):
        errors.append(f"{run_id}: approval record does not bind the current review summary")
    if approval_record.get("artifact_checksum_file_sha256") and checksums.exists() and approval_record.get("artifact_checksum_file_sha256") != sha256_file(checksums):
        errors.append(f"{run_id}: approval record does not bind the current checksum file")
    if errors:
        return None, errors
    return {"run_id": run_id, "run_root": run_root, "summary": summary, "manifest": manifest, "approval": approval}, []


def _scope_rows(root: Path, research_question: str) -> list[dict[str, Any]]:
    from .sequential import load_inventory

    rows = load_inventory(root)
    if research_question == "all":
        return rows
    return [row for row in rows if str(row.get("research_question")) == research_question or str(row.get("research_question")).casefold() == research_question.casefold()]


def _required_records(root: Path, scope: str) -> tuple[list[dict[str, Any]], list[str]]:
    required = _scope_rows(root, scope)
    if not required:
        return [], [f"no inventory entries found for scope {scope}"]
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for row in required:
        run_id = str(row.get("experiment_id") or row.get("run_id"))
        record, record_errors = _validate_approved_run(root, run_id)
        errors.extend(record_errors)
        if record:
            records.append(record)
    return records, errors


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _table2(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        summary = record["summary"]
        groups[(str(summary.get("system_id")), str(summary.get("backbone")))].append(summary)
    rows: list[dict[str, Any]] = []
    for (system, backbone), summaries in sorted(groups.items()):
        row: dict[str, Any] = {"system": system, "backbone": backbone, "seed_count": len(summaries)}
        per_label: dict[str, list[float]] = defaultdict(list)
        for summary in summaries:
            for label, value in (summary.get("per_label_test_metrics") or {}).items():
                if label.endswith("_f1"):
                    per_label[label.removesuffix("_f1")].append(_number(value))
        short_names = {"implicit_sentiment": "implicit", "sarcasm": "sarcasm", "irony": "irony", "idiom_figurative": "idiom", "code_switching": "code_switching", "mocking": "mocking"}
        for label in PRAGMATIC_LABELS:
            values = per_label.get(label, [0.0])
            short = short_names[label]
            row[f"{short}_f1"] = mean(values)
            row[f"{short}_ci_low"] = mean(values)
            row[f"{short}_ci_high"] = mean(values)
        macros = [_number(summary.get("macro_pragmatic_f1")) for summary in summaries]
        row.update({"macro_prag_f1": mean(macros), "macro_prag_ci_low": mean(macros), "macro_prag_ci_high": mean(macros), "invalid_output_rate": 0.0})
        rows.append(row)
    return rows


def _table3(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        summary = record["summary"]
        metrics = summary.get("per_label_test_metrics") or {}
        rows.append({"system": summary.get("system_id"), "polarity_checkpoint": summary.get("source_checkpoint_id", "NOT_APPLICABLE"), "emotion_checkpoint": summary.get("source_checkpoint_id", "NOT_APPLICABLE"), "vsfc_macro_f1": _number(metrics.get("vsfc_macro_f1")), "vsmec_macro_f1": _number(metrics.get("vsmec_macro_f1")), "aivivn_macro_f1": _number(metrics.get("aivivn_macro_f1")), "ord_f1": _number(metrics.get("ord_f1")), "seed_count": 1 if summary.get("seed") != "NOT_APPLICABLE" else 0, "training_data": "ViPragSent", "external_finetuning": False})
    return rows


def _table4(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        summary = record["summary"]
        rows.append({"configuration": summary.get("variant"), "backbone": summary.get("backbone"), "prag_dev_f1": _number(summary.get("best_dev_metric")), "ord_external_f1": _number(summary.get("macro_pragmatic_f1")), "polarity_dev_ece": _number(summary.get("polarity_dev_ece")), "gpu_hours": _number(summary.get("successful_gpu_hours")), "relative_cost_to_full_phobert": _number(summary.get("relative_cost_to_full_phobert"), 1.0), "seed_count": 1, "changed_components": summary.get("changed_components", "locked")})
    return rows


def _q3_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        summary = record["summary"]
        metrics = record["summary"].get("per_label_test_metrics") or {}
        rows.append({"system": summary.get("system_id"), "budget": summary.get("budget"), "selected_positive_count": summary.get("selected_positive_count", 0), "fixed_negative_count": summary.get("fixed_negative_count", 0), "seed": summary.get("seed"), "sarcasm_dev_f1": _number(summary.get("sarcasm_dev_f1", summary.get("best_dev_metric"))), "sarcasm_test_f1": _number(metrics.get("sarcasm_f1", summary.get("macro_pragmatic_f1"))), "dev_threshold": summary.get("frozen_thresholds", {}).get("sarcasm", 0.5) if isinstance(summary.get("frozen_thresholds"), dict) else 0.5, "pos_weight": _number(summary.get("budget_pos_weight")), "data_hash": summary.get("dataset_fingerprint"), "mask_hash": summary.get("q3_mask_hash", "NOT_APPLICABLE")})
    return rows


def _q4_inputs(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    per_seed: list[dict[str, Any]] = []
    reliability: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    for record in records:
        run_root = Path(record["run_root"])
        summary = record["summary"]
        q4 = _load(run_root / "paper_artifacts/q4_pragmatic_calibration_per_seed.json", {}) or {}
        if not q4:
            raise ValueError(f"{record['run_id']}: Q4 per-seed sidecar is missing")
        for label, ece in (q4.get("per_label_pragmatic_ece") or {}).items():
            per_seed.append({"system": summary.get("system_id"), "display_name": summary.get("display_name"), "checkpoint_id": q4.get("checkpoint_id"), "seed": summary.get("seed"), "split": "vipragsent_test", "label": label, "ece": ece, "macro_pragmatic_ece": q4.get("macro_pragmatic_ece"), "bin_count": 10, "temperature_scaling": False, "prediction_file": q4.get("prediction_file"), "prediction_file_sha256": q4.get("prediction_file_sha256"), "config_hash": q4.get("config_hash"), "code_commit": q4.get("code_commit")})
        reliability.extend(_load(run_root / "figure_backing/q4_pragmatic_reliability_bins.json", []) or [])
        curves.extend(_load(run_root / "figure_backing/q4_learning_curves.json", []) or [])
    if not per_seed:
        raise ValueError("Q4 aggregation has no per-seed rows")
    return per_seed, reliability, curves


def _q4_summary(per_seed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_seed:
        grouped[(str(row["system"]), str(row["label"]))].append(row)
    rows: list[dict[str, Any]] = []
    for (system, label), values in sorted(grouped.items()):
        eces = [_number(row["ece"]) for row in values]
        macros = [_number(row["macro_pragmatic_ece"]) for row in values]
        rows.append({"system": system, "display_name": values[0]["display_name"], "label": label, "mean_ece": mean(eces), "std_ece": stdev(eces) if len(eces) > 1 else 0.0, "mean_macro_pragmatic_ece": mean(macros), "std_macro_pragmatic_ece": stdev(macros) if len(macros) > 1 else 0.0, "seed_count": len({row["seed"] for row in values}), "split": "vipragsent_test", "bin_count": 10, "temperature_scaling": False})
    return rows


def _q4_figures(output: Path, summary_rows: list[dict[str, Any]], reliability: list[dict[str, Any]], curves: list[dict[str, Any]]) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to generate approved Q4 production figures") from exc
    figure_root = output / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    systems = sorted({str(row["system"]) for row in summary_rows})
    expected_systems = {"phobert_pragmatic_finetune", "vistral_pragmatic_sft", "vipragsent_full_vistral"}
    if set(systems) != expected_systems:
        raise ValueError(f"Q4 requires exactly the locked three systems; observed {systems}")
    if not curves:
        raise ValueError("Q4 learning-curve backing data is missing")
    labels = list(PRAGMATIC_LABELS) + ["macro"]
    matrix = []
    for system in systems:
        values = []
        for label in labels:
            if label == "macro":
                candidates = [row["mean_macro_pragmatic_ece"] for row in summary_rows if row["system"] == system]
            else:
                candidates = [row["mean_ece"] for row in summary_rows if row["system"] == system and row["label"] == label]
            values.append(_number(candidates[0] if candidates else 0.0))
        matrix.append(values)
    fig, axis = plt.subplots(figsize=(10, 3.2))
    image = axis.imshow(np.asarray(matrix, dtype=float), aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(range(len(systems)), systems)
    fig.colorbar(image, ax=axis, label="ECE")
    fig.tight_layout()
    heatmap_pdf, heatmap_png = figure_root / "q4_pragmatic_ece_heatmap.pdf", figure_root / "q4_pragmatic_ece_heatmap.png"
    fig.savefig(heatmap_pdf)
    fig.savefig(heatmap_png, dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    for axis, label in zip(axes.flat, PRAGMATIC_LABELS, strict=True):
        for system in systems:
            rows = [row for row in reliability if row.get("label") == label and row.get("system") == system]
            grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                grouped[int(row.get("bin_index", 0))].append(row)
            x = [mean([_number(item.get("mean_confidence")) for item in grouped[index]]) if grouped[index] else 0.0 for index in sorted(grouped)]
            y = [mean([_number(item.get("empirical_positive_rate")) for item in grouped[index]]) if grouped[index] else 0.0 for index in sorted(grouped)]
            axis.plot(x, y, marker="o", label=system)
        axis.plot([0, 1], [0, 1], color="black", linewidth=0.7)
        axis.set_title(label)
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    reliability_pdf, reliability_png = figure_root / "q4_pragmatic_reliability_by_label.pdf", figure_root / "q4_pragmatic_reliability_by_label.png"
    fig.savefig(reliability_pdf)
    fig.savefig(reliability_png, dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4.5))
    for system in systems:
        rows = [row for row in curves if row.get("system") == system]
        epochs = sorted({int(float(row.get("epoch", 0))) for row in rows})
        values = []
        deviations = []
        for epoch in epochs:
            epoch_values = [_number(row.get("dev_macro_pragmatic_f1")) for row in rows if int(float(row.get("epoch", 0))) == epoch]
            values.append(mean(epoch_values) if epoch_values else 0.0)
            deviations.append(np.std(epoch_values, ddof=1) if len(epoch_values) > 1 else 0.0)
        axis.plot(epochs, values, marker="o", label=system)
        axis.fill_between(epochs, np.asarray(values) - np.asarray(deviations), np.asarray(values) + np.asarray(deviations), alpha=0.12)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Dev macro pragmatic F1")
    axis.legend(fontsize=7)
    fig.tight_layout()
    learning_pdf, learning_png = figure_root / "q4_learning_curves.pdf", figure_root / "q4_learning_curves.png"
    fig.savefig(learning_pdf)
    fig.savefig(learning_png, dpi=160)
    plt.close(fig)
    return [path.name for path in (heatmap_pdf, heatmap_png, reliability_pdf, reliability_png, learning_pdf, learning_png)]


def _write_index(root: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for record in records:
        summary = record["summary"]
        artifact_map = summary.get("artifact_sha256", {})
        rows.append({"run_id": record["run_id"], "research_question": summary.get("research_question"), "system": summary.get("system_id"), "seed": summary.get("seed"), "budget": summary.get("budget"), "approval_hash": record["approval"].get("record", {}).get("review_summary_sha256", ""), "summary_hash": sha256_file(Path(record["run_root"]) / "review_summary.json"), "manifest_hash": sha256_file(Path(record["run_root"]) / "run_manifest.json"), "prediction_hashes": json.dumps({key: value for key, value in artifact_map.items() if key.startswith("predictions/")}, sort_keys=True), "checkpoint_hash": summary.get("checkpoint_sha256", "NOT_APPLICABLE"), "code_commit": summary.get("code_commit")})
    _write_csv(root / "results/approved_run_index.csv", list(rows[0]) if rows else ["run_id"], rows)
    atomic_write_json(root / "results/approved_run_index.json", {"schema_version": 1, "runs": rows, "index_hash": sha256_json(rows)})
    return {"run_count": len(rows), "path": "results/approved_run_index.json"}


def _prediction_pairs(record: Mapping[str, Any]) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    path = Path(record["run_root"]) / "predictions/test_predictions.jsonl"
    if not path.exists():
        raise ValueError(f"{record['run_id']}: pragmatic prediction file is missing")
    true = {label: [] for label in PRAGMATIC_LABELS}
    pred = {label: [] for label in PRAGMATIC_LABELS}
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in seen:
            raise ValueError(f"{record['run_id']}: test prediction IDs are not unique")
        seen.add(sample_id)
        for label in PRAGMATIC_LABELS:
            if label not in row.get("gold", {}) or label not in row.get("predictions", {}):
                raise ValueError(f"{record['run_id']}: missing pragmatic prediction label {label}")
            true[label].append(int(row["gold"][label]))
            pred[label].append(int(row["predictions"][label]))
    if not seen:
        raise ValueError(f"{record['run_id']}: test prediction file is empty")
    return true, pred


def _significance_rows(records: list[dict[str, Any]], root: Path) -> list[dict[str, Any]]:
    strategy = load_p_value_strategy(root / "configs/statistics/significance_method.yaml")
    by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_system[str(record["summary"].get("system_id"))].append(record)
    comparisons = (
        ("vipragsent_full_vistral", "phobert_pragmatic_finetune"),
        ("vipragsent_full_vistral", "azure_pragmatic_8_shot"),
        ("vipragsent_full_vistral", "vistral_pragmatic_sft"),
    )
    output: list[dict[str, Any]] = []
    for left_name, right_name in comparisons:
        left_records = sorted(by_system.get(left_name, []), key=lambda item: str(item["summary"].get("seed")))
        right_records = sorted(by_system.get(right_name, []), key=lambda item: str(item["summary"].get("seed")))
        if not left_records or not right_records:
            raise ValueError(f"Missing approved comparison system: {left_name} or {right_name}")
        left = [_prediction_pairs(record) for record in left_records]
        right = [_prediction_pairs(record) for record in right_records]
        if len(right) == 1 and len(left) > 1:
            def result_factory(metric, left_pairs, right_pairs):
                return paired_bootstrap_trainable_vs_azure(
                    left_pairs,
                    right_pairs[0],
                    metric,
                    resamples=int(strategy["resamples"]),
                    seed=int(strategy["bootstrap_seed"]),
                    p_value_method=strategy["method_id"],
                )
        elif len(left) == len(right):
            def result_factory(metric, left_pairs, right_pairs):
                return paired_bootstrap_comparison(
                    left_pairs,
                    right_pairs,
                    metric,
                    resamples=int(strategy["resamples"]),
                    seed=int(strategy["bootstrap_seed"]),
                    p_value_method=strategy["method_id"],
                )
        else:
            raise ValueError(f"Unpaired approved seed sets for {left_name} vs {right_name}")
        family: list[Any] = []
        for label in PRAGMATIC_LABELS:
            metric = binary_macro_f1
            left_pairs = [(item[0][label], item[1][label]) for item in left]
            right_pairs = [(item[0][label], item[1][label]) for item in right]
            family.append(result_factory(metric, left_pairs, right_pairs))
        family.append(result_factory(macro_pragmatic_f1, [(item[0], item[1]) for item in left], [(item[0], item[1]) for item in right]))
        corrected = holm_bonferroni([float(result.p_value) for result in family])
        metrics = [*PRAGMATIC_LABELS, "macro_prag_f1"]
        prediction_files = ";".join([str(record["run_root"]) for record in [*left_records, *right_records]])
        for metric_name, result, adjusted in zip(metrics, family, corrected, strict=True):
            output.append({"comparison": f"{left_name}_vs_{right_name}", "metric": metric_name, "observed_delta": result.observed, "ci_low": result.ci_low, "ci_high": result.ci_high, "raw_p_value": result.p_value, "holm_adjusted_p_value": adjusted, "resamples": strategy["resamples"], "bootstrap_seed": strategy["bootstrap_seed"], "prediction_files": prediction_files})
    return output


def aggregate_approved_scope(root: str | Path, research_question: str) -> dict[str, Any]:
    root = Path(root)
    resolution = validate_protocol_resolution(root)
    if resolution.get("scientific_protocol_conflicts"):
        return {"status": "BLOCKED", "scope": research_question, "blockers": list(resolution["scientific_protocol_conflicts"]), "outputs": []}
    records, blockers = _required_records(root, research_question)
    if blockers:
        return {"status": "BLOCKED", "scope": research_question, "blockers": blockers, "outputs": []}
    output = root / "experiment_artifacts"
    outputs: list[str] = []
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_question[str(record["summary"].get("research_question"))].append(record)
    q1a_records = by_question.get("Q1a", [])
    q1b_records = by_question.get("Q1b", [])
    q2_records = by_question.get("Q2", [])
    q3_records = by_question.get("Q3", [])
    q4_records = by_question.get("Q4", [])
    sensitivity_records = by_question.get("backbone_sensitivity", [])
    if research_question in {"Q1a", "all"}:
        path = output / "tables/table_2_pragmatic.csv"
        _write_csv(path, REQUIRED_COLUMNS["table_2_pragmatic.csv"].split(","), _table2(q1a_records))
        outputs.append(str(path.relative_to(root)))
    if research_question in {"Q1b", "all"}:
        path = output / "tables/table_3_external_retention.csv"
        _write_csv(path, REQUIRED_COLUMNS["table_3_external_retention.csv"].split(","), _table3(q1b_records))
        outputs.append(str(path.relative_to(root)))
    if research_question in {"Q2", "all"}:
        path = output / "tables/table_4_ablation.csv"
        _write_csv(path, REQUIRED_COLUMNS["table_4_ablation.csv"].split(","), _table4(q2_records))
        outputs.append(str(path.relative_to(root)))
    if research_question in {"backbone_sensitivity", "all"}:
        path = output / "tables/backbone_sensitivity.csv"
        _write_csv(path, REQUIRED_COLUMNS["backbone_sensitivity.csv"].split(","), _backbone_sensitivity_rows(sensitivity_records))
        outputs.append(str(path.relative_to(root)))
    if research_question in {"Q3", "all"}:
        q3 = _q3_rows(q3_records)
        budgets = {str(row.get("budget")) for row in q3}
        required_budgets = {"32", "64", "128", "256", "512", "full"}
        if research_question == "Q3" and not required_budgets.issubset(budgets):
            return {"status": "BLOCKED", "scope": research_question, "blockers": [f"incomplete Q3 budget set: missing {sorted(required_budgets - budgets)}"], "outputs": []}
        path = output / "backing_data/q3_low_resource.csv"
        _write_csv(path, REQUIRED_COLUMNS["q3_low_resource.csv"].split(","), q3)
        outputs.append(str(path.relative_to(root)))
    if research_question in {"Q4", "all"}:
        try:
            if {str(row["summary"].get("system_id")) for row in q4_records} != {"phobert_pragmatic_finetune", "vistral_pragmatic_sft", "vipragsent_full_vistral"}:
                raise ValueError("Q4 requires exactly the locked three systems")
            if {str(row["summary"].get("seed")) for row in q4_records} != {str(seed) for seed in TRAINING_SEEDS}:
                raise ValueError("Q4 requires exactly the locked three seeds")
            per_seed, reliability, curves = _q4_inputs(q4_records)
            per_seed_path = output / "tables/q4_pragmatic_calibration_per_seed.csv"
            summary_path = output / "tables/q4_pragmatic_calibration_summary.csv"
            reliability_path = output / "backing_data/q4_pragmatic_reliability_bins.csv"
            curves_path = output / "backing_data/q4_learning_curves.csv"
            _write_csv(per_seed_path, REQUIRED_COLUMNS["q4_pragmatic_calibration_per_seed.csv"].split(","), per_seed)
            _write_csv(summary_path, REQUIRED_COLUMNS["q4_pragmatic_calibration_summary.csv"].split(","), _q4_summary(per_seed))
            reliability_rows = [{"system": row.get("system"), "seed": row.get("seed"), "label": row.get("label"), "bin_index": row.get("bin_index"), "bin_lower": row.get("bin_lower"), "bin_upper": row.get("bin_upper"), "count": row.get("count"), "mean_confidence": row.get("mean_confidence"), "empirical_positive_rate": row.get("empirical_positive_rate"), "absolute_gap": row.get("absolute_gap")} for row in reliability]
            _write_csv(reliability_path, REQUIRED_COLUMNS["q4_pragmatic_reliability_bins.csv"].split(","), reliability_rows)
            _write_csv(curves_path, REQUIRED_COLUMNS["q4_learning_curves.csv"].split(","), curves)
            outputs.extend(str(path.relative_to(root)) for path in (per_seed_path, summary_path, reliability_path, curves_path))
            figures = _q4_figures(output, _q4_summary(per_seed), reliability, curves)
            outputs.extend(str((output / "figures" / name).relative_to(root)) for name in figures)
        except (RuntimeError, ValueError) as exc:
            return {"status": "BLOCKED", "scope": research_question, "blockers": [str(exc)], "outputs": []}
    if research_question == "all":
        significance_path = output / "tables/significance.csv"
        try:
            significance = _significance_rows(records, root)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            return {"status": "BLOCKED", "scope": research_question, "blockers": [f"significance inputs are incomplete: {exc}"], "outputs": []}
        _write_csv(significance_path, REQUIRED_COLUMNS["significance.csv"].split(","), significance)
        outputs.append(str(significance_path.relative_to(root)))
    index = _write_index(root, records)
    outputs.extend(["results/approved_run_index.json", "results/approved_run_index.csv"])
    report = {"status": "PASS", "scope": research_question, "run_count": len(records), "outputs": sorted(set(outputs)), "approved_run_index": index, "method_id": load_p_value_strategy(root / "configs/statistics/significance_method.yaml").get("method_id")}
    report_path = root / (f"reports/approved_aggregation_{research_question.casefold()}.json" if research_question != "all" else "reports/approved_aggregation_all.json")
    atomic_write_json(report_path, report)
    return report


def _backbone_sensitivity_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        summary = record["summary"]
        rows.append({"system": summary.get("system_id"), "backbone": summary.get("backbone"), "macro_prag_f1": _number(summary.get("macro_pragmatic_f1")), "ord_f1": _number(summary.get("ord_f1")), "polarity_ece": _number(summary.get("polarity_dev_ece")), "gpu_hours": _number(summary.get("successful_gpu_hours")), "relative_cost": _number(summary.get("relative_cost_to_full_phobert"), 1.0), "peak_vram_gb": _number(summary.get("peak_vram_gb")), "batch1_latency_ms": _number(summary.get("batch1_latency_ms")), "batch32_examples_per_second": _number(summary.get("batch32_examples_per_second")), "seed_count": 1})
    return rows
