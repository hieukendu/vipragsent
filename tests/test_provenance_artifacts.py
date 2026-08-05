from __future__ import annotations

import json
from pathlib import Path

from vipragsent.orchestration.contracts import RunContext, RunEntry
from vipragsent.orchestration.provenance import (
    expected_inference_provenance,
    validate_inference_provenance,
)
from vipragsent.orchestration.run_store import RunStore

ROOT = Path(__file__).resolve().parents[1]


def _entry(system_id: str, execution_kind: str) -> RunEntry:
    return RunEntry.from_mapping(
        {
            "experiment_id": f"provenance-{system_id}",
            "research_question": "Q1a",
            "system_id": system_id,
            "display_name": system_id,
            "variant": system_id,
            "backbone": "vistral_7b",
            "seed": 20260521,
            "execution_kind": execution_kind,
            "_repository_root": str(ROOT),
        }
    )


def test_explanation_manifest_truthful_rationale_inference(tmp_path: Path) -> None:
    context = RunContext(
        root=ROOT,
        entry=_entry("explanation_only_vistral", "checkpoint_reuse"),
        fixture=False,
        run_root=tmp_path / "explanation",
    )
    RunStore(context).initialize()
    manifest = json.loads((context.run_root / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["additional_training"] is False
    assert manifest["source_system_id"] == "vipragsent_full_vistral"
    assert manifest["same_seed_source"] is True
    assert manifest["direct_classification_outputs_used"] is False
    assert manifest["rationale_decoder_enabled_at_inference"] is True
    assert manifest["native_causal_lm_generation_used"] is False
    assert manifest["inference_output_source"] == "judge_of_rationale_decoder_output"


def test_explanation_validator_accepts_truthful_provenance() -> None:
    payload = {
        "system_id": "explanation_only_vistral",
        "mode": "full",
        **expected_inference_provenance("explanation_only_vistral", execution_kind="checkpoint_reuse"),
    }
    assert validate_inference_provenance(payload, source="golden explanation") == []
    invalid = payload | {"rationale_decoder_enabled_at_inference": False}
    assert validate_inference_provenance(invalid, source="mutated explanation")


def test_cot_manifest_marks_native_causal_generation(tmp_path: Path) -> None:
    context = RunContext(
        root=ROOT,
        entry=_entry("cot_only_vistral", "generation"),
        fixture=False,
        run_root=tmp_path / "cot",
    )
    RunStore(context).initialize()
    manifest = json.loads((context.run_root / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["rationale_decoder_enabled_at_inference"] is False
    assert manifest["native_causal_lm_generation_used"] is True
    assert manifest["direct_classification_outputs_used"] is False
    assert manifest["inference_output_source"] == "judge_of_generated_reasoning"


def test_generation_provenance_system_specific() -> None:
    cot = {"system_id": "cot_only_vistral", "mode": "full", **expected_inference_provenance("cot_only_vistral", execution_kind="generation")}
    explanation = {"system_id": "explanation_only_vistral", "mode": "full", **expected_inference_provenance("explanation_only_vistral", execution_kind="checkpoint_reuse")}
    assert validate_inference_provenance(cot, source="cot") == []
    assert validate_inference_provenance(explanation, source="explanation") == []
    assert validate_inference_provenance(cot | {"inference_output_source": explanation["inference_output_source"]}, source="cross-wired")
