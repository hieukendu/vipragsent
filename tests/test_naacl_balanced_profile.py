from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/naacl_balanced_runtime_profile.yaml"
REPORT = ROOT / "reports/runtime_optimization/naacl_balanced_profile.json"
MARKDOWN = ROOT / "reports/runtime_optimization/naacl_balanced_profile.md"


def _artifacts() -> tuple[dict, dict]:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    return config, report


def test_profile_fixture_schema_and_artifact_parity() -> None:
    config, report = _artifacts()
    assert config["schema_version"] == report["schema_version"] == 1
    assert config["profile_id"] == report["profile_id"] == "LUNA_NAACL_PROFILE"
    assert config["activation"]["default_enabled"] is False
    assert config["activation"]["execution_enabled"] is False
    assert config["scope"]["real_execution"] == "prohibited"
    assert report["status"] == "POLICY_ONLY_READY_AFTER_WAVE0_ACCEPTED"
    assert MARKDOWN.exists()


def test_q3_retains_exact_balanced_cartesian_slice() -> None:
    config, report = _artifacts()
    expected_systems = {"phobert_pragmatic_finetune", "vistral_pragmatic_sft", "vipragsent_full_vistral"}
    expected_budgets = {32, 128, 512, "full"}
    expected_seeds = {20260521, 20260522, 20260523}
    assert set(config["q3"]["systems"]) == expected_systems
    assert set(config["q3"]["budgets"]) == expected_budgets
    assert set(config["q3"]["seeds"]) == expected_seeds
    assert config["q3"]["expected_cell_count"] == 3 * 4 * 3
    assert report["q3"]["systems"] == config["q3"]["systems"]
    assert report["q3"]["budgets"] == config["q3"]["budgets"]
    assert report["q3"]["seeds"] == config["q3"]["seeds"]


def test_exclusions_are_explicit_and_original_source_is_immutable() -> None:
    config, report = _artifacts()
    exclusions = config["exclusions"]
    assert {item["q3_system"] for item in exclusions if "q3_system" in item} == {"xlmr_pragmatic_finetune", "azure_gpt41_mini_8shot"}
    assert {item["q3_budget"] for item in exclusions if "q3_budget" in item} == {64, 256}
    assert config["source"]["source_is_read_only"] is True
    assert report["source"]["read_only"] is True


def test_q1b_is_evaluation_only_and_dependency_aware() -> None:
    config, report = _artifacts()
    q1b = config["q1b"]
    policy = q1b["dependency_policy"]
    assert q1b["execution_kind"] == "evaluation_only"
    assert q1b["training_applicability"] == "NOT_APPLICABLE"
    assert q1b["optimizer_steps"] == 0
    assert q1b["official_external_tests_only"] is True
    assert policy["same_seed_required"] is True
    assert policy["checkpoint_key_must_match_consumer"] is True
    assert policy["missing_source_action"] == "block_profile_aggregation"
    assert report["q1b"]["dependency_policy"] == policy


def test_aggregation_is_profile_aware_and_fail_closed() -> None:
    config, report = _artifacts()
    aggregation = config["aggregation"]
    assert aggregation["profile_aware"] is True
    assert aggregation["include_only_profile_cells"] is True
    assert aggregation["missing_cell_action"] == "fail_closed"
    assert aggregation["q1b"]["require_resolved_dependencies"] is True
    assert aggregation["q1b"]["exclude_training_metrics"] is True
    assert aggregation["synthetic_or_fixture_rows"] == "forbidden"
    assert report["aggregation"]["profile_aware"] is True
    assert report["aggregation"]["synthetic_or_fixture_rows"] == "forbidden"
