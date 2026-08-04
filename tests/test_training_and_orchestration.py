from __future__ import annotations

import json
from pathlib import Path

import torch

from vipragsent.models.variants import VariantConfig, build_dummy_model
from vipragsent.orchestration.dag import DAGNode, ExperimentDAG, load_master_dag
from vipragsent.orchestration.status import ProtocolConflict, RunExitCode
from vipragsent.training.engine import EvaluationAccessGate


def test_test_access_gate_requires_frozen_checkpoint() -> None:
    gate = EvaluationAccessGate()
    try:
        gate.assert_test_allowed()
    except RuntimeError:
        pass
    else:
        raise AssertionError("test access should be blocked")
    gate.freeze_checkpoint()
    gate.assert_test_allowed()


def test_master_dag_contains_required_nodes() -> None:
    root = Path(__file__).resolve().parents[1]
    dag = load_master_dag(root / "configs/experiments/master_matrix.yaml")
    ids = set(dag.nodes)
    assert {"table3_checkpoint_training", "backbone_sensitivity", "error_analysis_candidate_export", "qualitative_candidate_export", "paper_artifact_schema_validation"}.issubset(ids)
    assert len(dag.topological_order()) == len(ids)


def test_fixture_state_and_manifest_are_complete() -> None:
    root = Path(__file__).resolve().parents[1]
    state = json.loads((root / "runs/fixture/dag_state.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "runs/fixture/FIXTURE_VALIDATION_MANIFEST.json").read_text(encoding="utf-8"))
    assert state["status"] == "PASS"
    assert manifest["mode"] == "fixture"
    assert manifest["fixture_validation_passed"] is True
    assert manifest["synthetic_results"] is True
    assert manifest["core_experiments_ready"] is False
    assert manifest["manual_paper_analysis_pending"] is True
    assert not (root / "FINAL_EXPERIMENT_MANIFEST.json").exists()


def test_dag_marks_protocol_conflict_with_protocol_exit_code(tmp_path: Path) -> None:
    dag = ExperimentDAG([DAGNode("conflict", "validation", ())])

    def blocked_handler(_: DAGNode):
        raise ProtocolConflict("locked protocol is unresolved")

    state = dag.run(tmp_path / "dag_state.json", {"validation": blocked_handler})
    assert state["status"] == "BLOCKED"
    assert state["protocol_conflict"] is True
    assert state["exit_code"] == RunExitCode.PROTOCOL_FAILURE
