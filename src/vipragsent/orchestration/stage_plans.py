from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..hashing import sha256_file, sha256_json


@dataclass(frozen=True)
class ExecutionStagePlan:
    plan_id: str
    stages: tuple[str, ...]
    execution_kind: str
    executor_kind: str
    evaluation_strategy: str | None = None
    research_question: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "stages": list(self.stages),
            "execution_kind": self.execution_kind,
            "executor_kind": self.executor_kind,
            "evaluation_strategy": self.evaluation_strategy,
            "research_question": self.research_question,
        }


def load_stage_plan_registry(root: str | Path = ".") -> dict[str, ExecutionStagePlan]:
    root = Path(root)
    path = root / "configs/experiments/execution_stage_plans.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    plans: dict[str, ExecutionStagePlan] = {}
    for plan_id, raw in (payload.get("plans") or {}).items():
        stages = tuple(str(stage) for stage in raw.get("stages", ()))
        if not stages or stages[0] != "preflight" or stages[-1] != "generate_review_summary":
            raise ValueError(f"stage plan {plan_id} must start with preflight and end with review")
        plans[str(plan_id)] = ExecutionStagePlan(
            plan_id=str(plan_id),
            stages=stages,
            execution_kind=str(raw.get("execution_kind", "")),
            executor_kind=str(raw.get("executor_kind", "")),
            evaluation_strategy=raw.get("evaluation_strategy"),
            research_question=raw.get("research_question"),
        )
    return plans


def resolve_stage_plan(root: str | Path, entry: Mapping[str, Any], execution_spec: Any | None = None) -> ExecutionStagePlan:
    root = Path(root)
    plans = load_stage_plan_registry(root)
    execution_kind = str(entry.get("execution_kind", ""))
    research_question = str(entry.get("research_question", ""))
    evaluation_strategy = str(entry.get("evaluation_strategy", ""))
    if execution_kind == "azure" or str(entry.get("backbone", "")) == "azure":
        return plans["azure"]
    registry_executor = str(getattr(execution_spec, "executor_kind", ""))
    if not registry_executor and entry.get("system_id"):
        from .system_registry import resolve_execution_spec

        try:
            resolved_spec = resolve_execution_spec(root, str(entry["system_id"]))
            registry_executor = resolved_spec.executor_kind
            if not evaluation_strategy:
                evaluation_strategy = resolved_spec.evaluation_strategy
        except ValueError:
            registry_executor = ""
    system_id = str(entry.get("system_id", ""))
    if system_id == "cot_only_vistral":
        return plans["cot_only_vistral_generation"]
    if system_id == "explanation_only_vistral":
        return plans["explanation_only_vistral_reuse"]
    if research_question == "Q1b" and execution_kind == "evaluation_only" and evaluation_strategy in {"q1b_external_retention", "q1b_external_retention_v1"}:
        return plans["q1b_evaluation_only"]
    if research_question == "Q4" and execution_kind == "artifact_extraction":
        return plans["q4_source_extraction"]
    if execution_kind == "checkpoint_reuse":
        return plans["checkpoint_reuse"]
    if execution_kind == "component_bundle" or registry_executor in {"single_task_bundle", "independent_checkpoint_bundle"}:
        return plans["component_bundle"]
    if execution_kind == "generation" or registry_executor in {"generation_baseline", "generation_trainable", "rationale_checkpoint_reuse"}:
        raise ValueError(f"generation baseline {system_id!r} has no exact registered system plan")
    if execution_kind == "trainable" and registry_executor == "single_model_trainable":
        return plans["trainable_classifier"]
    raise ValueError(
        f"No exact execution stage plan for execution_kind={execution_kind!r}, "
        f"evaluation_strategy={evaluation_strategy!r}, research_question={research_question!r}, "
        f"executor_kind={registry_executor!r}"
    )


def validate_stage_plan_registry(root: str | Path = ".") -> dict[str, Any]:
    root = Path(root)
    plans = load_stage_plan_registry(root)
    expected = {
        "trainable_classifier": ("preflight", "train", "evaluate_dev", "freeze_selection", "evaluate_test", "export_artifacts", "validate_artifacts", "generate_review_summary"),
        "component_bundle": ("preflight", "execute_components", "combine_component_predictions", "evaluate_dev", "freeze_component_selection", "evaluate_test", "export_artifacts", "validate_artifacts", "generate_review_summary"),
        "cot_only_vistral_generation": ("preflight", "train_generation", "generate_dev_reasoning", "judge_dev_reasoning", "compute_dev_reasoning_metrics", "freeze_selection", "generate_test_reasoning", "judge_test_reasoning", "compute_test_reasoning_metrics", "export_artifacts", "validate_artifacts", "generate_review_summary"),
        "explanation_only_vistral_reuse": ("preflight", "resolve_approved_full_vistral_source", "validate_source_checkpoint", "generate_dev_reasoning_from_rationale_decoder", "judge_dev_reasoning", "compute_dev_reasoning_metrics", "generate_test_reasoning_from_rationale_decoder", "judge_test_reasoning", "compute_test_reasoning_metrics", "export_artifacts", "validate_artifacts", "generate_review_summary"),
        "q1b_evaluation_only": ("preflight", "resolve_approved_source", "evaluate_external_tests", "export_artifacts", "validate_artifacts", "generate_review_summary"),
        "q4_source_extraction": ("preflight", "resolve_approved_source", "validate_source_predictions", "extract_pragmatic_calibration", "extract_learning_history", "export_artifacts", "validate_artifacts", "generate_review_summary"),
        "checkpoint_reuse": ("preflight", "resolve_approved_source", "evaluate_reused_test", "export_artifacts", "validate_artifacts", "generate_review_summary"),
        "azure": ("preflight", "execute_api_job", "validate_responses", "export_artifacts", "validate_artifacts", "generate_review_summary"),
    }
    errors: list[str] = []
    for plan_id, stages in expected.items():
        if plan_id not in plans:
            errors.append(f"missing plan: {plan_id}")
        elif plans[plan_id].stages != stages:
            errors.append(f"plan {plan_id} has unexpected stages")
    errors.extend(
        [
            "Q1b plan contains training/selection stage"
            for plan in plans.values()
            if plan.plan_id == "q1b_evaluation_only" and any(stage in plan.stages for stage in ("train", "train_generation", "freeze_selection"))
        ]
    )
    errors.extend(
        [
            "Q4 plan contains training stage"
            for plan in plans.values()
            if plan.plan_id == "q4_source_extraction" and any(stage in plan.stages for stage in ("train", "train_generation", "execute_components"))
        ]
    )
    path = root / "configs/experiments/execution_stage_plans.yaml"
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "plan_count": len(plans),
        "registry_sha256": sha256_file(path),
        "plans": {key: value.as_dict() for key, value in sorted(plans.items())},
        "registry_hash": sha256_json({key: value.as_dict() for key, value in sorted(plans.items())}),
    }
