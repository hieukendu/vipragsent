from __future__ import annotations

import json
from pathlib import Path

from scripts._bootstrap import ROOT
from scripts.audit_final_production_correctness import _is_local_only_path
from scripts.audit_production_implementation import _runtime_command
from scripts.readiness_utils import merge_snapshot_into_report
from scripts.validate_schemas import has_material_artifacts

from vipragsent.constants import RUNTIME_PREFLIGHT_CHECKLIST
from vipragsent.models import factory
from vipragsent.models.backbones import DummyBackbone
from vipragsent.models.variants import VARIANT_IDS
from vipragsent.orchestration import single_run
from vipragsent.orchestration.contracts import RunContext, RunEntry, StageOutcome
from vipragsent.orchestration.inventory import build_expected_runs
from vipragsent.orchestration.run_store import RunStore
from vipragsent.phase import PHASE15_SMOKE_TESTS, write_phase_handoff
from vipragsent.runtime.model_assets import (
    cache_record_from_snapshot,
    read_family_status,
    resolve_local_snapshot,
    write_family_status,
)
from vipragsent.runtime.phase15_state import reconcile_phase15_state


def _write_phase15_fixture(root: Path, *, smoke_status: str = "PASS") -> None:
    snapshot = root / "data/model_cache/phobert_base"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
    (snapshot / "bpe.codes").write_text("fixture\n", encoding="utf-8")
    base = {
        "model_family": "phobert_base",
        "repo_id": "fixture/phobert-base",
        "revision": "model-revision",
        "tokenizer_revision": "tokenizer-revision",
    }
    cache_path = root / "data/model_cache_status/phobert_base.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(base | {"category": "cache", "status": "PASS", "local_path": "data/model_cache/phobert_base", "snapshot_files": ["config.json", "bpe.codes"], "manifest_hash": "cache"}), encoding="utf-8")
    smoke_path = root / "data/model_smoke_status/phobert_base.json"
    smoke_path.parent.mkdir(parents=True, exist_ok=True)
    smoke_path.write_text(json.dumps(base | {"category": "smoke", "status": smoke_status, "actual_local_loads": True, "checks": {name.replace(" ", "_"): smoke_status == "PASS" for name in PHASE15_SMOKE_TESTS}, "blockers": [] if smoke_status == "PASS" else ["smoke failed"], "verification_hash": "smoke"}), encoding="utf-8")
    batch_path = root / "data/batch_probe_status/phobert_base.json"
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    batch_path.write_text(json.dumps({"model_family": "phobert_base", "category": "batch", "status": "PASS", "frozen": True, "fixture_probe": False, "successful_batch": 32, "effective_batch_size": 32, "gradient_accumulation_steps": 1, "hardware_identity": "fixture", "probe_hash": "batch", "blockers": []}), encoding="utf-8")
    write_phase_handoff("15", "PASS", report_root=root / "reports/phases", model_family="phobert_base")
    (root / "PROJECT_STATE.json").write_text(json.dumps({"phase15_runtime_ready": False, "runtime_environment_ready": False, "weights_downloaded": False, "real_experiment_ready": False, "full_run_started": False, "approved_run_count": 0, "real_run_count": 0, "runtime_blockers": ["stale"], "implementation_blockers": [], "scientific_protocol_conflicts": []}), encoding="utf-8")
    (root / "SETUP_READY.md").write_text("# Setup readiness\n\nPHASE15_RUNTIME_READY=false\nRUNTIME_ENVIRONMENT_READY=false\nWEIGHTS_DOWNLOADED=false\nREAL_EXPERIMENT_READY=false\nFINAL_AGGREGATION_READY=false\nREAL_RUN_COUNT=0\nAPPROVED_RUN_COUNT=0\n\n## Runtime blockers\n- stale\n\n## Exact next action\nold\n", encoding="utf-8")


def test_gitkeep_only_artifact_tree_is_not_treated_as_material(tmp_path: Path) -> None:
    assert (ROOT / RUNTIME_PREFLIGHT_CHECKLIST).is_file()
    (tmp_path / "tables").mkdir()
    (tmp_path / "tables/.gitkeep").write_text("# placeholder\n", encoding="utf-8")
    assert has_material_artifacts(tmp_path) is False
    (tmp_path / "tables/example.csv").write_text("not a locked table\n", encoding="utf-8")
    assert has_material_artifacts(tmp_path) is True


def test_audits_use_the_active_python_interpreter() -> None:
    assert _runtime_command(["python", "-m", "pytest"])[0].endswith("/python")


def test_final_audit_ignores_only_known_local_environment_roots() -> None:
    assert _is_local_only_path(".venv/lib/python3.11/site-packages/pkg.py")
    assert _is_local_only_path('".venv/lib/python3.11/site-packages/pkg.py"')
    assert _is_local_only_path("data/model_cache/phobert_base/config.json")
    assert _is_local_only_path("src/pkg/__pycache__/module.pyc")
    assert not _is_local_only_path(".env")
    assert not _is_local_only_path("src/vipragsent/runtime/model_assets.py")


def test_cache_status_paths_are_portable_and_resolvable(tmp_path: Path) -> None:
    snapshot = tmp_path / "data/model_cache/phobert_base"
    snapshot.mkdir(parents=True)
    record = cache_record_from_snapshot(
        "phobert_base",
        {"repo_id": "fixture/repo", "revision": "model", "tokenizer_revision": "tokenizer"},
        snapshot,
        root=tmp_path,
    )
    assert record["local_path"] == "data/model_cache/phobert_base"
    write_family_status(tmp_path, "phobert_base", "cache", record)
    stored = read_family_status(tmp_path, "phobert_base", "cache")
    assert stored["local_path"] == "data/model_cache/phobert_base"
    assert resolve_local_snapshot(tmp_path, stored["local_path"]) == snapshot
    assert resolve_local_snapshot(tmp_path, "/old/checkout/data/model_cache/phobert_base") == snapshot


def test_stale_readiness_snapshot_cannot_overwrite_current_report() -> None:
    report = {"status": "FAIL", "ci_status": "FAIL", "code_commit_at_audit": "current"}
    stale = {
        "audited_code_commit": "0" * 40,
        "report_only_commit_expected": True,
        "ci": {"conclusion": "success"},
    }
    merged = merge_snapshot_into_report(report, stale, root=ROOT)
    assert merged["status"] == "FAIL"
    assert merged["ci_status"] == "FAIL"
    assert merged["snapshot_merge_status"] == "SKIPPED_STALE"


def test_component_factory_uses_locked_variant_ids(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(factory, "load_pretrained_backbone", lambda *_args, **_kwargs: DummyBackbone(hidden_size=8, vocab_size=32))
    monkeypatch.setattr(factory, "place_non_quantized_model", lambda model, *_args, **_kwargs: model)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for component in ("implicit_sentiment", "sarcasm", "irony", "idiom_figurative", "code_switching", "mocking", "polarity", "emotion"):
        model, _ = factory.build_production_component_model(
            "phobert_base",
            component,
            local_snapshot=snapshot,
            selected_device="cpu",
        )
        assert model.config.name in VARIANT_IDS
        assert model.output_key == component


def test_preflight_then_all_resumes_the_authoritative_state_file(monkeypatch, tmp_path: Path) -> None:
    row = next(item for item in build_expected_runs(ROOT)["rows"] if item["run_id"].startswith("q1a_phobert_pragmatic_finetune_"))
    entry = RunEntry.from_mapping(row, run_id=row["run_id"])
    monkeypatch.setattr(single_run, "build_single_experiment_stage_registry", lambda *_args: {})
    monkeypatch.setattr(single_run, "build_single_azure_stage_registry", lambda *_args: {})
    import vipragsent.orchestration.stage_registry as stage_registry

    monkeypatch.setattr(stage_registry, "_review_summary", lambda *_args: StageOutcome.passed())
    def passed() -> StageOutcome:
        return StageOutcome.passed()
    _, first_code = single_run.execute_single_run(
        tmp_path,
        entry,
        kind="experiment",
        stage="preflight",
        run_id=entry.run_id,
        fixture=True,
        injected_handlers={"preflight": passed},
    )
    state_path = tmp_path / "runs/fixture/results/runs" / entry.run_id / "state.json"
    assert first_code == 0
    assert state_path.exists()
    _, second_code = single_run.execute_single_run(
        tmp_path,
        entry,
        kind="experiment",
        stage="all",
        run_id=entry.run_id,
        fixture=True,
        injected_handlers={stage: passed for stage in entry.stages},
    )
    assert second_code == 0


def test_resume_invalidates_preflight_after_code_revision_change(monkeypatch, tmp_path: Path) -> None:
    entry = RunEntry.from_mapping(
        {
            "experiment_id": "fixture_resume_revision",
            "research_question": "Q1a",
            "system_id": "phobert_pragmatic_finetune",
            "display_name": "fixture",
            "variant": "fixture",
            "backbone": "phobert_base",
            "execution_kind": "trainable",
            "stages": ["preflight", "train"],
        },
        run_id="fixture_resume_revision",
    )
    monkeypatch.setattr("vipragsent.orchestration.run_store.git_commit", lambda _root: "current-commit")
    monkeypatch.setattr("vipragsent.orchestration.run_store.git_tree", lambda _root: "current-tree")
    store = RunStore(RunContext(tmp_path, entry))
    state = store.initialize()
    state["code_commit"] = "old-commit"
    state["code_tree"] = "old-tree"
    state["run_status"] = "FAIL"
    state["stages"]["preflight"] = {"status": "PASS"}
    store.save(state)

    resumed = store.initialize(resume=True)

    assert resumed["stages"]["preflight"]["status"] == "NOT_STARTED"
    assert resumed["stages"]["preflight"]["invalidation_reason"]


def test_phase15_state_refresh_preserves_verified_runtime_evidence(tmp_path: Path) -> None:
    _write_phase15_fixture(tmp_path)

    report = reconcile_phase15_state(tmp_path, require_local_snapshot=True)
    state = json.loads((tmp_path / "PROJECT_STATE.json").read_text(encoding="utf-8"))

    assert report["status"] == "PASS"
    assert report["local_snapshot"]["available"] is True
    assert state["phase15_runtime_ready"] is True
    assert state["runtime_environment_ready"] is True
    assert state["real_experiment_ready"] is False
    assert state["full_run_started"] is False
    assert state["approved_run_count"] == 0
    assert state["runtime_blockers"] == []


def test_phase15_state_refresh_does_not_promote_failed_smoke(tmp_path: Path) -> None:
    _write_phase15_fixture(tmp_path, smoke_status="BLOCKED")

    report = reconcile_phase15_state(tmp_path, require_local_snapshot=True)
    state = json.loads((tmp_path / "PROJECT_STATE.json").read_text(encoding="utf-8"))

    assert report["status"] == "BLOCKED"
    assert state["phase15_runtime_ready"] is False
    assert state["runtime_environment_ready"] is False
    assert state["runtime_blockers"]
