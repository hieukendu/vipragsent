from __future__ import annotations

from collections.abc import Mapping
from typing import Any

GENERATION_SYSTEMS = frozenset({"cot_only_vistral", "explanation_only_vistral"})
INFERENCE_OUTPUT_SOURCES = frozenset(
    {
        "classification_heads",
        "judge_of_generated_reasoning",
        "judge_of_rationale_decoder_output",
        "parsed_generated_labels",
    }
)


def expected_inference_provenance(
    system_id: str,
    *,
    execution_kind: str | None = None,
) -> dict[str, Any]:
    """Return the immutable inference facts for one registered system.

    The values are protocol facts, not observations inferred from a model
    class.  Callers must validate persisted artifacts against this contract so
    a stale model property cannot silently change paper-facing provenance.
    """

    if system_id == "explanation_only_vistral":
        return {
            "additional_training": False,
            "source_system_id": "vipragsent_full_vistral",
            "same_seed_source": True,
            "direct_classification_outputs_used": False,
            "rationale_decoder_enabled_at_inference": True,
            "native_causal_lm_generation_used": False,
            "inference_output_source": "judge_of_rationale_decoder_output",
        }
    if system_id == "cot_only_vistral":
        return {
            "additional_training": True,
            "source_system_id": "NOT_APPLICABLE",
            "same_seed_source": False,
            "direct_classification_outputs_used": False,
            "rationale_decoder_enabled_at_inference": False,
            "native_causal_lm_generation_used": True,
            "inference_output_source": "judge_of_generated_reasoning",
        }
    return {
        "additional_training": execution_kind in {"trainable", "component_bundle", "generation"},
        "source_system_id": "NOT_APPLICABLE",
        "same_seed_source": False,
        "direct_classification_outputs_used": True,
        "rationale_decoder_enabled_at_inference": False,
        "native_causal_lm_generation_used": False,
        "inference_output_source": "classification_heads",
    }


def validate_inference_provenance(
    payload: Mapping[str, Any],
    *,
    source: str = "artifact",
    allow_fixture_parser: bool = True,
) -> list[str]:
    """Validate persisted system-specific inference provenance.

    ``parsed_generated_labels`` remains available only to synthetic fixture
    compatibility paths.  A production artifact must identify the actual
    reasoning source and its decoder/generation behavior explicitly.
    """

    system_id = str(payload.get("system_id") or payload.get("system") or "")
    expected = expected_inference_provenance(system_id, execution_kind=payload.get("execution_kind"))
    errors: list[str] = []
    mode = str(payload.get("mode", ""))
    for field, expected_value in expected.items():
        if field not in payload:
            errors.append(f"{source}: missing provenance field {field}")
            continue
        observed = payload.get(field)
        if field == "inference_output_source" and allow_fixture_parser and mode == "fixture" and observed == "parsed_generated_labels":
            continue
        if observed != expected_value:
            errors.append(f"{source}: {field}={observed!r}, expected {expected_value!r} for {system_id}")
    if mode == "full" and payload.get("inference_output_source") == "parsed_generated_labels":
        errors.append(f"{source}: parsed_generated_labels is fixture-only and cannot enter production")
    if payload.get("inference_output_source") not in INFERENCE_OUTPUT_SOURCES:
        errors.append(f"{source}: inference_output_source is not an approved value")
    return errors
