from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from vipragsent.runtime import naacl_profile
from vipragsent.runtime.naacl_profile import (
    ProfileValidationError,
    build_naacl_profile_snapshot,
    validate_naacl_profile,
)


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
    assert config["activation"]["real_execution"] == "PROHIBITED"
    assert config["scope"]["real_execution"] == "PROHIBITED"
    assert report["status"] == "POLICY_ONLY_READY_AFTER_WAVE0_ACCEPTED"
    assert report["activation"]["real_execution"] == "PROHIBITED"
    assert report["exclusions"]["real_execution"] == config["scope"]["real_execution"]
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


def test_q3_q2_protocol_binding_matches_sources() -> None:
    snapshot = build_naacl_profile_snapshot(ROOT)
    assert snapshot["status"] == "PASS", snapshot
    q3 = snapshot["protocol_binding"]["q3"]
    q2 = snapshot["protocol_binding"]["q2"]
    assert q3["retained_systems"] == [
        "phobert_pragmatic_finetune",
        "vistral_pragmatic_sft",
        "vipragsent_full_vistral",
    ]
    assert q3["excluded_systems"] == ["xlmr_pragmatic_finetune", "azure_gpt41_mini_8shot"]
    assert q3["source_budgets"] == ["32", "64", "128", "256", "512", "full"]
    assert q3["retained_budgets"] == ["32", "128", "512", "full"]
    assert q3["excluded_budgets"] == ["64", "256"]
    assert q3["seeds"] == [20260521, 20260522, 20260523]
    assert q3["expected_cell_count"] == 36
    assert q2["retained_variants"] == [
        "full",
        "no_emotion_auxiliary",
        "no_polarity_auxiliary",
        "no_rationale",
        "no_multitask",
        "no_uncertainty_weighting",
    ]
    assert q2["expected_variant_count"] == 6
    assert {item["path"] for item in snapshot["protocol_sources"]["files"]} == {
        "configs/experiments/q3/system_aliases.yaml",
        "configs/experiments/q3/protocol.yaml",
        "configs/experiments/q2/protocol.yaml",
        "src/vipragsent/constants.py",
    }


def _copy_profile_tree(tmp_path: Path) -> None:
    import shutil

    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    shutil.copytree(ROOT / "reports", tmp_path / "reports")
    shutil.copytree(ROOT / "src", tmp_path / "src")


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


def test_q1b_binding_matches_audited_graph_and_includes_azure_null_seed() -> None:
    snapshot = build_naacl_profile_snapshot(ROOT)
    assert snapshot["status"] == "PASS", snapshot
    binding = snapshot["q1b"]
    assert binding["consumer_count"] == 22
    assert binding["graph_edge_count"] == 21
    assert binding["profile_edge_count"] == 22
    assert binding["seeded_consumer_count"] == 21
    assert binding["seedless_consumer_count"] == 1
    assert len(binding["consumer_edges"]) == 22
    azure = next(edge for edge in binding["consumer_edges"] if edge["consumer_id"] == "q1b_azure_gpt41_mini")
    assert azure["producer_id"] == "azure_gpt41_mini_dedicated_prompts"
    assert azure["producer_kind"] == "approved_azure_output"
    assert azure["checkpoint_key"] == "azure_gpt41_mini:dedicated_prompts"
    assert azure["seed"] is None
    assert azure["graph_edge"] is False
    assert all(edge["producer_kind"] == "trainable_checkpoint" for edge in binding["consumer_edges"] if edge["consumer_id"] != "q1b_azure_gpt41_mini")
    assert snapshot["graph"]["sha256"]
    assert snapshot["source"]["sha256"]


def test_checked_in_report_validates_and_real_execution_parity_is_unambiguous() -> None:
    snapshot = validate_naacl_profile(ROOT)
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["protocol_sources"] == snapshot["protocol_sources"]
    assert report["protocol_binding"] == snapshot["protocol_binding"]
    assert report["q1b"]["dependency_binding"] == snapshot["q1b"]
    assert report["q1b"]["digests"] == {
        "graph_sha256": snapshot["graph"]["sha256"],
        "source_sha256": snapshot["source"]["sha256"],
    }
    assert report["activation"]["real_execution"] == "PROHIBITED"
    assert report["exclusions"]["real_execution"] == "PROHIBITED"


def test_validator_fails_closed_on_binding_drift(tmp_path: Path) -> None:
    _copy_profile_tree(tmp_path)
    report_path = tmp_path / "reports/runtime_optimization/naacl_balanced_profile.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["q1b"]["dependency_binding"]["consumer_edges"][0]["seed"] = 999
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ProfileValidationError, match="dependency binding"):
        validate_naacl_profile(tmp_path)


def test_validator_fails_closed_on_live_graph_key_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    original = naacl_profile.build_q1b_dependency_graph

    def drifted_graph(*args, **kwargs):
        graph = original(*args, **kwargs)
        edge = next(edge for edge in graph["edges"] if str(edge.get("consumer_id", "")).startswith("q1b_"))
        edge["produced_checkpoint_key"] = "drifted-key"
        return graph

    monkeypatch.setattr(naacl_profile, "build_q1b_dependency_graph", drifted_graph)
    with pytest.raises(ProfileValidationError, match="produced_checkpoint_key"):
        validate_naacl_profile(ROOT)


def test_validator_fails_closed_on_q3_alias_drift(tmp_path: Path) -> None:
    _copy_profile_tree(tmp_path)
    path = tmp_path / "configs/experiments/q3/system_aliases.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["q3_system_aliases"][0]["resolved_system_id"] = "drifted_q3_system"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProfileValidationError, match="retained Q3 system alias drift"):
        validate_naacl_profile(tmp_path)


def test_validator_fails_closed_on_q3_budget_seed_and_cell_drift(tmp_path: Path) -> None:
    _copy_profile_tree(tmp_path)
    protocol_path = tmp_path / "configs/experiments/q3/protocol.yaml"
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    protocol["q3"]["budgets"] = [32, 64, 128, 512, "full", 1024]
    protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    config_path = tmp_path / "configs/experiments/naacl_balanced_runtime_profile.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["q3"]["seeds"] = [20260521, 20260522, 20260524]
    config["q3"]["expected_cell_count"] = 35
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProfileValidationError, match="Q3 protocol source is missing budget 64 or 256|Q3 profile seed drift|Q3 expected cell count drift"):
        validate_naacl_profile(tmp_path)


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
