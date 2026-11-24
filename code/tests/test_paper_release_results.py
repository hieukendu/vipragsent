from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RELEASE = __import__("release_paths").default_release(ROOT)


def validator():
    spec = importlib.util.spec_from_file_location("aggregate_validator", RELEASE / "validate_saved_results.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_external_rows_have_three_seed_confusion_counts():
    module = validator()
    data = json.loads((RELEASE / "replay/AGGREGATE_RESULTS.json").read_text())
    assert len(data["runs"]) == 257
    assert data["individual_predictions_included"] is False
    external = [r for r in data["runs"] if r["target"].startswith("Q1b/")]
    systems = {r["target"].split("/")[1] for r in external}
    assert len(systems) == 6
    assert len(external) == 54
    for system in systems:
        cohort = [r for r in external if r["target"].split("/")[1] == system]
        assert len(cohort) == 9
        assert {int(r["target"].split("/")[2]) for r in cohort} == set(module.SEEDS)
        assert {r["dataset"] for r in cohort} == {"uit_vsfc", "uit_vsmec", "aivivn_human_derived_3way"}
        for run in cohort:
            assert module.same(module.f1(run["confusion_matrix"]), run["macro_f1"])


def test_validation_rejects_gold_misalignment_even_when_f1_is_unchanged(monkeypatch):
    module = validator()
    original_read = module.read
    payload = original_read("replay/AGGREGATE_RESULTS.json")
    chosen = next(r for r in payload["runs"] if "dataset" not in r
                  and r["confusion_matrices"]["sarcasm"][0][1] != r["confusion_matrices"]["sarcasm"][1][0])
    matrix = chosen["confusion_matrices"]["sarcasm"]
    chosen["confusion_matrices"]["sarcasm"] = [list(row) for row in zip(*matrix)]
    assert module.same(module.f1(chosen["confusion_matrices"]["sarcasm"]), chosen["head_f1"]["sarcasm"])
    monkeypatch.setattr(module, "read", lambda name: payload if name == "replay/AGGREGATE_RESULTS.json" else original_read(name))
    with pytest.raises(ValueError, match="gold marginals"):
        module.main()


def test_original_threshold_grid_reselects_all_frozen_thresholds():
    module = validator()
    data = json.loads((RELEASE / "replay/Q3_reported/VISTRAL32_THRESHOLD_SEARCH.json").read_text())
    assert len(data["versions"]) == 1
    version = data["versions"][0]
    assert version["version"] == data["verified_selection_version"] == "original_selection"
    assert version["gold_and_IDs_match_original_dev"]
    assert version["dev_file_matches_freeze_hash"]
    for head in version["heads"].values():
        assert len(head["grid"]) == 91
        best = min(head["grid"], key=lambda r: (-module.f1(r["confusion_matrix"]), abs(r["threshold"] - .5), r["threshold"]))
        assert best["threshold"] == head["frozen_threshold"] == head["reselected_threshold"]
        assert head["saved_dev_prediction_mismatch_at_frozen_threshold"] == 0


@pytest.mark.parametrize("counts", [[[1.5, 0], [0, 1]], [[-1, 0], [0, 1]], [[1, 2, 3]]])
def test_validation_rejects_invalid_confusion_counts(counts):
    with pytest.raises(ValueError, match="Invalid confusion counts"):
        validator().f1(counts)


def test_primary_runner_does_not_queue_an_unreported_q3_protocol():
    tree = ast.parse((ROOT / "scripts/run_priority12_xlmr.py").read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {"_tasks", "_run_id"}]
    namespace = {"argparse": __import__("argparse"), "Any": object,
                 "Q1_RUN_ALL_VARIANTS": ("full", "no_emotion_auxiliary", "no_polarity_auxiliary", "no_rationale", "no_uncertainty_weighting", "all_multipliers_1", "beta_01", "beta_05"), "SEEDS": (20260521, 20260522, 20260523)}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "runner_helpers", "exec"), namespace)
    tasks = namespace["_tasks"](namespace["argparse"].Namespace(run_all=True))
    assert len(tasks) == 24 and all(t["experiment"] == "priority1" for t in tasks)
    with pytest.raises(ValueError, match="paper-reported Q3"):
        namespace["_run_id"]("priority2", None, "32", 20260521)


def test_generic_inventory_and_producers_match_reported_baselines():
    from vipragsent.orchestration.inventory import build_expected_runs
    from vipragsent.orchestration.q1b_dependencies import build_q1b_dependency_graph
    from vipragsent.orchestration.system_registry import validate_execution_registry, load_execution_registry
    from vipragsent.models.variants import VariantConfig

    inventory = build_expected_runs(ROOT)
    assert not inventory["scientific_protocol_conflicts"]
    assert validate_execution_registry(ROOT)["status"] == "PASS"
    graph = build_q1b_dependency_graph(ROOT)
    assert graph["status"] == "PASS" and not graph["errors"]
    assert len(graph["edges"]) == 21
    assert inventory["counts_by_question"] == {"Q1a": 14, "Q1b": 18, "Q2": 0, "Q3": 36, "Q4": 3, "backbone_sensitivity": 0}
    specs = load_execution_registry(ROOT)
    producers = [s for s in specs.values() if s.system_id.startswith("q1b_train_")]
    assert len(producers) == 6
    for spec in producers:
        model_config = VariantConfig(name=spec.variant_id)
        assert not model_config.has_rationale_decoder
        assert not model_config.has_uncertainty_weighting
        assert not spec.rationale_training and not spec.uncertainty_tasks
        consumer = specs[spec.system_id.removeprefix("q1b_train_")]
        assert consumer.variant_id == spec.variant_id
        assert not VariantConfig(name=consumer.variant_id).has_rationale_decoder


def test_q4_accepts_validated_same_seed_primary_runner_output(tmp_path):
    from types import SimpleNamespace
    from vipragsent.hashing import sha256_file
    from vipragsent.orchestration.xlmr_followup import _resolve_v37_source
    from vipragsent.orchestration.status import RuntimeBlocked

    seed = 20260521
    run_id = f"priority12_v37_q1_full_{seed}"
    source = tmp_path / "results/runs" / run_id
    files = {"checkpoints/best/model.pt": "test-only checkpoint bytes",
             "predictions/test_predictions.jsonl": "[]", "training/history.json": "[]",
             "config_snapshot.yaml": "test_only: true"}
    for name, data in files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data)
    checksum = sha256_file(source / "checkpoints/best/model.pt")
    (source / "manifest.json").write_text(json.dumps({"status": "PASS", "model_family": "xlmr_large", "variant_id": "vipragsent_full_xlmr_large", "seed": seed, "checkpoint_sha256": checksum}))
    (source / "validation.json").write_text(json.dumps({"status": "PASS"}))
    (source / "checkpoints/checkpoint_manifest.json").write_text(json.dumps({"status": "PASS", "checkpoint_sha256": checksum}))
    entry = SimpleNamespace(raw={"source_run_id": run_id}, seed=seed)
    resolved, _, _, _ = _resolve_v37_source(tmp_path, entry)
    assert resolved == source.resolve()
    entry.seed += 1
    with pytest.raises(RuntimeBlocked, match="same-seed"):
        _resolve_v37_source(tmp_path, entry)
