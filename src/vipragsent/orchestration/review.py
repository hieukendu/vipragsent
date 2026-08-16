from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from ..hashing import sha256_file, sha256_json
from .contracts import ExecutionKind, RunContext, RunEntry
from .provenance import expected_inference_provenance, validate_inference_provenance
from .q1b_dependencies import (
    build_q1b_dependency_graph,
    load_q1b_producer_registry,
    q1b_source_sha256,
)
from .run_store import artifact_hashes, git_commit, utc_now
from .variant_diff import changed_components_against_full_phobert

COMMON_FIELDS = (
    "run_id", "research_question", "system_id", "display_name", "variant", "backbone", "seed", "budget", "execution_kind",
    "execution_mode", "run_status", "user_review_status", "next_run_allowed", "dataset_fingerprint", "split_hashes",
    "model_repository", "model_revision", "tokenizer_revision", "preprocessing_name", "preprocessing_version", "configuration_hash",
    "code_commit", "start_time", "end_time", "wall_clock_seconds", "warnings", "blockers", "validation_status", "artifact_paths", "artifact_sha256",
    "additional_training", "source_system_id", "same_seed_source", "direct_classification_outputs_used",
    "rationale_decoder_enabled_at_inference", "native_causal_lm_generation_used", "inference_output_source",
)
TRAINABLE_FIELDS = (
    "optimizer", "learning_rate", "weight_decay", "scheduler", "warmup_ratio", "precision", "physical_batch_size",
    "gradient_accumulation_steps", "effective_batch_size", "maximum_epochs", "actual_epochs", "best_epoch", "best_dev_metric",
    "best_dev_loss", "checkpoint_path", "checkpoint_sha256", "frozen_thresholds", "per_label_dev_metrics", "per_label_test_metrics",
)
GENERATION_FIELDS = (
    "generation_protocol_id", "generation_prompt_hash", "judge_protocol_id", "judge_prompt_hash", "judge_schema_hash",
    "judge_model", "judge_model_version", "judge_temperature", "decoding", "rationale_source_hash",
    "primary_metric_name", "primary_macro_f1", "primary_per_label_f1", "valid_only_macro_f1", "valid_only_per_label_f1",
    "coverage_rate", "invalid_generation_rate", "invalid_judge_output_rate", "missing_prediction_rate", "truncation_rate",
    "judge_usage", "judge_cache_statistics",
)


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except (OSError, json.JSONDecodeError):
        return default


def _load_yaml(path: Path, default: Any) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else default
    except (OSError, yaml.YAMLError):
        return default


def _not_applicable(reason: str) -> str:
    return "NOT_APPLICABLE"


def build_review_summary(context: RunContext, entry: RunEntry, state: Mapping[str, Any]) -> dict[str, Any]:
    run_root = Path(context.run_root)
    manifest = _load_json(run_root / "run_manifest.json", {})
    generation = entry.system_id in {"cot_only_vistral", "explanation_only_vistral"}
    dev = _load_json(run_root / ("metrics/dev_reasoning_metrics.json" if generation else "metrics/dev_metrics.json"), {})
    test = _load_json(run_root / ("metrics/test_reasoning_metrics.json" if generation else "metrics/test_metrics.json"), {})
    selection = _load_json(run_root / "selection/selection_metric.json", {})
    thresholds = _load_json(run_root / "selection/thresholds.json", {})
    checkpoint = _load_json(run_root / "selection/best_checkpoint.json", {})
    checkpoint_manifest = _load_json(run_root / "checkpoints/checkpoint_manifest.json", {})
    artifacts = artifact_hashes(run_root)
    prediction_hashes = {name: digest for name, digest in artifacts.items() if name.startswith("predictions/")}
    data_manifest = context.root / "data/manifests/dataset_manifest.json"
    trainable = entry.execution_kind == ExecutionKind.TRAINABLE.value
    q4 = _load_json(run_root / "paper_artifacts/q4_pragmatic_calibration_per_seed.json", {})
    usage = _load_json(run_root / "azure/usage.json", {})
    external_manifest = _load_json(run_root / "external/external_evaluation_manifest.json", {})
    external_metrics = _load_json(run_root / "metrics/external_retention_metrics.json", {}) or _load_json(run_root / "metrics/partial_external_retention_metrics.json", {})
    start_time = str(state.get("created_at") or utc_now())
    end_time = utc_now()
    applicability: dict[str, str] = {}
    protocol_provenance = expected_inference_provenance(entry.system_id, execution_kind=entry.execution_kind)
    observed_provenance = {key: manifest.get(key, value) for key, value in protocol_provenance.items()}
    if manifest:
        provenance_errors = validate_inference_provenance(manifest, source="run_manifest", allow_fixture_parser=context.fixture)
        if provenance_errors:
            raise ValueError("; ".join(provenance_errors))
    fields: dict[str, Any] = {
        "run_id": entry.run_id,
        "experiment_id": None if entry.is_azure else entry.run_id,
        "azure_job_id": entry.run_id if entry.is_azure else None,
        "research_question": entry.research_question,
        "system_id": entry.system_id,
        "display_name": entry.display_name,
        "variant": entry.variant,
        "backbone": entry.backbone,
        "seed": entry.seed if entry.seed not in (None, "") else (None if entry.research_question == "Q3" and entry.is_azure else "NOT_APPLICABLE"),
        "budget": entry.budget if entry.budget not in (None, "") else "NOT_APPLICABLE",
        "execution_kind": entry.execution_kind,
        "execution_mode": "fixture_synthetic" if context.fixture else "production_sequential_review_gated",
        "run_status": "PASS",
        "user_review_status": "PENDING",
        "next_run_allowed": "NO",
        "dataset_fingerprint": manifest.get("data_fingerprint") or (sha256_file(data_manifest) if data_manifest.exists() else "fixture"),
        "split_hashes": {split: prediction_hashes.get(f"predictions/{split}_predictions.jsonl", "NOT_APPLICABLE") for split in ("dev", "test")},
        "model_repository": entry.model_repository or manifest.get("model_repository") or ("fixture" if context.fixture else "NOT_APPLICABLE"),
        "model_revision": entry.model_revision or manifest.get("model_revision") or ("fixture" if context.fixture else "NOT_APPLICABLE"),
        "tokenizer_revision": entry.tokenizer_revision or manifest.get("tokenizer_revision") or ("fixture" if context.fixture else "NOT_APPLICABLE"),
        "preprocessing_name": entry.preprocessing_name or manifest.get("preprocessing_name") or ("fixture_unicode_nfc" if context.fixture else "NOT_APPLICABLE"),
        "preprocessing_version": entry.preprocessing_version or manifest.get("preprocessing_version") or ("fixture-v1" if context.fixture else "NOT_APPLICABLE"),
        "configuration_hash": manifest.get("config_hash") or sha256_file(run_root / "config_snapshot.yaml"),
        "code_commit": manifest.get("code_commit") or git_commit(context.root),
        "start_time": start_time,
        "end_time": end_time,
        "wall_clock_seconds": 0.0,
        "warnings": list(state.get("warnings", [])),
        "blockers": [],
        "validation_status": "PASS",
        "artifact_paths": sorted(artifacts),
        "artifact_sha256": artifacts,
        "primary_dev_selection_metric": selection.get("name", "NOT_APPLICABLE"),
        "frozen_thresholds": thresholds if thresholds else "NOT_APPLICABLE",
        "per_label_dev_metrics": dev.get("per_label_f1", "NOT_APPLICABLE"),
        "per_label_test_metrics": test.get("per_label_f1", "NOT_APPLICABLE"),
        "macro_pragmatic_f1": test.get("macro_pragmatic_f1", dev.get("macro_pragmatic_f1", "NOT_APPLICABLE")),
        "sarcasm_dev_f1": dev.get("sarcasm_dev_f1", "NOT_APPLICABLE"),
        "sarcasm_test_f1": test.get("sarcasm_test_f1", "NOT_APPLICABLE"),
        "RUN_STATUS": "PASS",
        "USER_REVIEW_STATUS": "PENDING",
        "NEXT_RUN_ALLOWED": "NO",
        "applicability": applicability,
        "prediction_hashes": prediction_hashes,
        **observed_provenance,
        "provenance_contract_version": int(manifest.get("provenance_contract_version", 1)),
    }
    if generation:
        protocol = _load_yaml(context.root / "configs/experiments/generation_reasoning_protocol.yaml", {})
        judge_usage = _load_json(run_root / "judge/usage.json", {})
        fields.update({
            "generation_protocol_id": protocol.get("protocol_version", "NOT_APPLICABLE"),
            "generation_prompt_hash": protocol.get("generation_prompt_hash", "NOT_APPLICABLE"),
            "judge_protocol_id": protocol.get("judge_protocol_id", "NOT_APPLICABLE"),
            "judge_prompt_hash": protocol.get("judge_prompt_hash", "NOT_APPLICABLE"),
            "judge_schema_hash": protocol.get("judge_schema_hash", "NOT_APPLICABLE"),
            "judge_model": protocol.get("judge_model", "NOT_APPLICABLE"),
            "judge_model_version": protocol.get("judge_model_version", "NOT_APPLICABLE"),
            "judge_temperature": protocol.get("judge_temperature", "NOT_APPLICABLE"),
            "decoding": protocol.get("decoding", "NOT_APPLICABLE"),
            "rationale_source_hash": checkpoint_manifest.get("rationale_source_hash", protocol.get("systems", {}).get("cot_only_vistral", {}).get("rationale_source", "NOT_APPLICABLE")),
            "primary_metric_name": test.get("primary_metric_name", "full_split_macro_pragmatic_f1_all_zero_fallback"),
            "primary_macro_f1": test.get("primary_macro_f1", "NOT_APPLICABLE"),
            "primary_per_label_f1": test.get("primary_per_label_f1", "NOT_APPLICABLE"),
            "valid_only_macro_f1": test.get("valid_only_macro_f1", "NOT_APPLICABLE"),
            "valid_only_per_label_f1": test.get("valid_only_per_label_f1", "NOT_APPLICABLE"),
            "coverage_rate": test.get("coverage_rate", "NOT_APPLICABLE"),
            "invalid_generation_rate": test.get("invalid_generation_rate", "NOT_APPLICABLE"),
            "invalid_judge_output_rate": test.get("invalid_judge_output_rate", "NOT_APPLICABLE"),
            "missing_prediction_rate": test.get("missing_prediction_rate", "NOT_APPLICABLE"),
            "truncation_rate": test.get("truncation_rate", "NOT_APPLICABLE"),
            "judge_usage": judge_usage,
            "judge_cache_statistics": {key: judge_usage.get(key, 0) for key in ("judge_cache_hits", "judge_cache_misses", "judge_retry_count", "judge_request_count")},
            "additional_training": entry.system_id == "cot_only_vistral",
            "direct_classification_outputs_used": False,
            "inference_output_source": "judge_of_generated_reasoning" if entry.system_id == "cot_only_vistral" else "judge_of_rationale_decoder_output",
        })
        if entry.system_id == "explanation_only_vistral":
            source = _load_json(run_root / "source/source_provenance.json", {})
            source_data = source.get("source", source)
            fields.update({
                "source_run_id": source_data.get("run_id", source.get("source_run_id", "NOT_APPLICABLE")),
                "source_checkpoint_path": source_data.get("checkpoint_path", "NOT_APPLICABLE"),
                "source_checkpoint_sha256": source_data.get("checkpoint_sha256", source.get("source_checkpoint_sha256", "NOT_APPLICABLE")),
                "source_approval_sha256": source_data.get("approval_sha256", source.get("source_approval_sha256", "NOT_APPLICABLE")),
                "additional_training": False,
                "direct_classification_outputs_used": False,
            })
            fields["source_system_id"] = source_data.get("source_system_id", fields["source_system_id"])
            fields["same_seed_source"] = source_data.get("same_seed_source", fields["same_seed_source"])
            fields["rationale_decoder_enabled_at_inference"] = source_data.get("rationale_decoder_enabled_at_inference", fields["rationale_decoder_enabled_at_inference"])
            fields["native_causal_lm_generation_used"] = source_data.get("native_causal_lm_generation_used", fields["native_causal_lm_generation_used"])
            fields["rationale_source_hash"] = source_data.get("checkpoint_sha256", source.get("source_checkpoint_sha256", "NOT_APPLICABLE"))
    if trainable:
        training_config = _load_json(run_root / "training/optimizer_summary.json", {})
        scheduler_config = _load_json(run_root / "training/scheduler_summary.json", {})
        resolved = _load_json(run_root / "training/resolved_training_config.json", {})
        history = _load_json(run_root / "training/history.json", [])
        best_epoch = checkpoint.get("best_epoch") or selection.get("best_epoch")
        best_dev_loss = None
        for row in history:
            try:
                if best_epoch is not None and float(row.get("epoch")) == float(best_epoch):
                    best_dev_loss = row.get("dev_loss")
                    break
            except (TypeError, ValueError):
                continue
        fields.update({
            "optimizer": training_config.get("optimizer"),
            "learning_rate": training_config.get("learning_rate"),
            "weight_decay": training_config.get("weight_decay"),
            "scheduler": scheduler_config.get("scheduler"),
            "warmup_ratio": scheduler_config.get("warmup_ratio"),
            "precision": resolved.get("precision") or manifest.get("precision"),
            "physical_batch_size": resolved.get("physical_batch_size") or manifest.get("physical_batch_size"),
            "gradient_accumulation_steps": resolved.get("gradient_accumulation_steps") or manifest.get("gradient_accumulation_steps"),
            "effective_batch_size": resolved.get("effective_batch_size") or manifest.get("effective_batch_size"),
            "maximum_epochs": resolved.get("maximum_epochs") or manifest.get("maximum_epochs"),
            "actual_epochs": len(history),
            "best_epoch": best_epoch or "NOT_APPLICABLE",
            "early_stopping_reason": "patience_exhausted" if history and len(history) < int(resolved.get("maximum_epochs", len(history))) else "maximum_epochs_reached",
            "best_dev_metric": selection.get("value"),
            "best_dev_loss": best_dev_loss,
            "checkpoint_path": checkpoint.get("path"),
            "checkpoint_sha256": checkpoint.get("sha256") or checkpoint_manifest.get("checkpoint_sha256"),
        })
        fields["class_weights"] = _load_json(run_root / "training/class_weights.json", "NOT_APPLICABLE")
        fields["rationale_applicability"] = {"training": bool(resolved.get("rationale_training")), "inference": bool(resolved.get("rationale_inference", False))}
        fields["q3_budget_data"] = {"budget": entry.budget, "selected_positive_count": checkpoint_manifest.get("selected_positive_count"), "fixed_negative_count": checkpoint_manifest.get("fixed_negative_count"), "pos_weight": checkpoint_manifest.get("pos_weight"), "mask_hash": checkpoint_manifest.get("q3_mask_hash", "NOT_APPLICABLE")} if entry.research_question == "Q3" else "NOT_APPLICABLE"
        if entry.research_question == "Q3":
            fields.update({"selected_positive_count": checkpoint_manifest.get("selected_positive_count"), "fixed_negative_count": checkpoint_manifest.get("fixed_negative_count"), "budget_pos_weight": checkpoint_manifest.get("pos_weight"), "q3_mask_hash": checkpoint_manifest.get("q3_mask_hash")})
        fields["changed_components"] = changed_components_against_full_phobert(context.root, entry.system_id)
    else:
        for field in TRAINABLE_FIELDS:
            applicability[field] = "NOT_APPLICABLE"
        applicability["training"] = entry.execution_kind
        fields.update({field: "NOT_APPLICABLE" for field in TRAINABLE_FIELDS})
        fields["not_applicable_reason"] = "This entry does not create a new trainable checkpoint under its locked execution_kind."
        fields["changed_components"] = changed_components_against_full_phobert(context.root, entry.system_id) if (context.root / "configs/experiments/system_execution_registry.yaml").exists() else "NOT_APPLICABLE"
    if entry.research_question == "Q4":
        fields.update({
            "per_label_pragmatic_ece": q4.get("per_label_pragmatic_ece", "NOT_APPLICABLE"),
            "macro_pragmatic_ece": q4.get("macro_pragmatic_ece", "NOT_APPLICABLE"),
            "temperature_scaling": False,
            "bin_count": 10,
            "probability_aggregation": "none",
            "source_checkpoint_id": q4.get("checkpoint_id", entry.source_checkpoint_id or "NOT_APPLICABLE"),
            "source_prediction_hash": q4.get("prediction_file_sha256", "NOT_APPLICABLE"),
        })
    else:
        fields.update({"per_label_pragmatic_ece": "NOT_APPLICABLE", "macro_pragmatic_ece": "NOT_APPLICABLE", "temperature_scaling": False, "bin_count": "NOT_APPLICABLE", "probability_aggregation": "NOT_APPLICABLE", "source_checkpoint_id": "NOT_APPLICABLE", "source_prediction_hash": "NOT_APPLICABLE"})
    if entry.research_question == "Q1b":
        # Carry the executor's canonical producer binding into the review
        # summary.  Aggregation also checks the raw metrics/manifest files so
        # contradictory duplicate payloads cannot be hidden here.
        for field in (
            "producer_id",
            "producer_run_id",
            "producer_kind",
            "checkpoint_key",
            "source_seed",
            "dependency_graph_sha256",
            "dependency_source_sha256",
        ):
            if field in external_manifest:
                fields[field] = external_manifest[field]
            elif field in external_metrics:
                fields[field] = external_metrics[field]
        for field in ("external_finetuning", "train_loader_created", "optimizer_steps", "backward_calls", "training_applicability"):
            if field in external_manifest:
                fields[field] = external_manifest[field]
            elif field in external_metrics:
                fields[field] = external_metrics[field]
        if entry.is_azure:
            azure_definition = next(
                (item for item in load_q1b_producer_registry(context.root).values() if item.producer_kind == "approved_azure_output"),
                None,
            )
            if azure_definition is not None:
                graph = build_q1b_dependency_graph(context.root)
                fields.update(
                    {
                        "producer_id": azure_definition.producer_id,
                        "producer_run_id": azure_definition.producer_id,
                        "producer_kind": azure_definition.producer_kind,
                        "checkpoint_key": azure_definition.checkpoint_key(None),
                        "source_seed": None,
                        "dependency_graph_sha256": sha256_json(graph),
                        "dependency_source_sha256": q1b_source_sha256(context.root),
                        "external_finetuning": False,
                        "train_loader_created": False,
                        "optimizer_steps": 0,
                        "backward_calls": 0,
                        "training_applicability": "NOT_APPLICABLE",
                    }
                )
    if entry.is_azure:
        fields.update({
            "azure_request_count": usage.get("request_count", "NOT_APPLICABLE"),
            "azure_input_tokens": usage.get("input_tokens", "NOT_APPLICABLE"),
            "azure_cached_input_tokens": usage.get("cached_input_tokens", "NOT_APPLICABLE"),
            "azure_non_cached_input_tokens": usage.get("non_cached_input_tokens", "NOT_APPLICABLE"),
            "azure_output_tokens": usage.get("output_tokens", "NOT_APPLICABLE"),
            "azure_cost_usd": usage.get("total_azure_cost_usd", "NOT_APPLICABLE"),
            "azure_non_cached_input_cost_usd": usage.get("non_cached_input_cost_usd", "NOT_APPLICABLE"),
            "azure_cached_input_cost_usd": usage.get("cached_input_cost_usd", "NOT_APPLICABLE"),
            "azure_output_cost_usd": usage.get("output_cost_usd", "NOT_APPLICABLE"),
            "azure_cost_accounting_method": usage.get("cost_accounting_method", "NOT_APPLICABLE"),
            "azure_cost_verification_status": usage.get("cost_verification_status", "NOT_APPLICABLE"),
            "azure_usage_records_path": usage.get("usage_records_path", "NOT_APPLICABLE"),
            "azure_cost_ledger_path": usage.get("cost_ledger_path", "NOT_APPLICABLE"),
            "azure_invalid_output_rate": usage.get("invalid_output_rate", 0.0),
            "azure_cache_hits": usage.get("cache_hits", 0),
            "azure_cache_misses": usage.get("cache_misses", 0),
            "azure_failed_requests": usage.get("failed_requests", 0),
            "azure_retried_requests": usage.get("retried_requests", 0),
        })
    else:
        fields.update({key: "NOT_APPLICABLE" for key in ("azure_request_count", "azure_input_tokens", "azure_cached_input_tokens", "azure_non_cached_input_tokens", "azure_output_tokens", "azure_cost_usd", "azure_non_cached_input_cost_usd", "azure_cached_input_cost_usd", "azure_output_cost_usd", "azure_cost_accounting_method", "azure_cost_verification_status", "azure_usage_records_path", "azure_cost_ledger_path", "azure_invalid_output_rate", "azure_cache_hits", "azure_cache_misses", "azure_failed_requests", "azure_retried_requests")})
    fields.update({
        "peak_vram_gb": _load_json(run_root / "training/resource_usage.json", {}).get("peak_vram_gb", "NOT_APPLICABLE"),
        "successful_gpu_hours": _load_json(run_root / "training/resource_usage.json", {}).get("successful_gpu_hours", "NOT_APPLICABLE"),
        "failed_or_retried_gpu_hours": _load_json(run_root / "training/resource_usage.json", {}).get("failed_or_retried_gpu_hours", "NOT_APPLICABLE"),
    })
    fields["summary_hash_input"] = sha256_json({key: value for key, value in fields.items() if key not in {"artifact_sha256", "artifact_paths"}})
    return fields


def validate_review_summary(summary: Mapping[str, Any], *, completed: bool = False) -> list[str]:
    errors: list[str] = []
    for field in COMMON_FIELDS:
        if field not in summary or summary[field] in (None, ""):
            errors.append(f"missing review-summary field: {field}")
    if summary.get("RUN_STATUS") not in {"PASS", "BLOCKED", "FAIL"}:
        errors.append("RUN_STATUS is invalid")
    if summary.get("USER_REVIEW_STATUS") != "PENDING":
        errors.append("USER_REVIEW_STATUS must remain PENDING")
    if summary.get("NEXT_RUN_ALLOWED") != "NO":
        errors.append("NEXT_RUN_ALLOWED must remain NO")
    errors.extend(validate_inference_provenance(summary, source="review_summary", allow_fixture_parser=False))
    if completed:
        if summary.get("RUN_STATUS") != "PASS" or summary.get("validation_status") != "PASS":
            errors.append("completed summary must be public PASS with validation_status=PASS")
        if not summary.get("artifact_paths") or not summary.get("artifact_sha256"):
            errors.append("completed summary must contain non-empty artifact paths and hashes")
        if summary.get("run_status") != "PASS":
            errors.append("completed summary run_status must be public PASS")
        if summary.get("system_id") in {"cot_only_vistral", "explanation_only_vistral"}:
            for field in GENERATION_FIELDS:
                if summary.get(field) in (None, "", "NOT_APPLICABLE"):
                    errors.append(f"completed generation summary field is unresolved: {field}")
            if summary.get("inference_output_source") == "classification_heads" or summary.get("direct_classification_outputs_used") is not False:
                errors.append("generation summary exposes direct classification outputs")
            if summary.get("system_id") == "explanation_only_vistral" and summary.get("additional_training") is not False:
                errors.append("explanation-only summary declares additional training")
        elif summary.get("execution_kind") == ExecutionKind.TRAINABLE.value:
            for field in TRAINABLE_FIELDS:
                if summary.get(field) in (None, "", "NOT_APPLICABLE"):
                    errors.append(f"completed trainable summary field is unresolved: {field}")
            trainable_payload = {key: summary.get(key) for key in TRAINABLE_FIELDS}
            trainable_payload.update({
                "resolved_training_config": summary.get("resolved_training_config"),
                "class_weights": summary.get("class_weights"),
                "rationale_applicability": summary.get("rationale_applicability"),
                "q3_budget_data": summary.get("q3_budget_data"),
            })
            encoded = json.dumps(trainable_payload, ensure_ascii=False, sort_keys=True)
            for placeholder in ("locked_config", "measured", "TODO"):
                if placeholder in encoded:
                    errors.append(f"completed trainable summary contains unresolved placeholder: {placeholder}")
        else:
            if summary.get("not_applicable_reason") in (None, ""):
                errors.append("non-trainable summary requires not_applicable_reason")
    return errors
