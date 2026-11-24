from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from build_submission import audit_bytes, source_paths  # noqa: E402
from prepare_inputs import copy  # noqa: E402


def test_active_source_and_license_are_anonymous() -> None:
    for path in source_paths(ROOT):
        audit_bytes(path.relative_to(ROOT).as_posix(), path.read_bytes())
    assert "Anonymous contributors" in (ROOT / "LICENSE").read_text(encoding="utf-8")


def test_anonymity_audit_rejects_private_paths_and_tokens() -> None:
    with pytest.raises(ValueError):
        audit_bytes(".restore_private/secret.txt", b"backup")
    with pytest.raises(ValueError):
        audit_bytes("sample.pyc", b"bytecode")
    with pytest.raises(ValueError):
        audit_bytes("config.yaml", b"checkpoint_repository: private-account/private-checkpoints")


def test_reasoning_prompt_and_schema_hashes_are_locked() -> None:
    config = yaml.safe_load((ROOT / "configs/experiments/generation_reasoning_protocol.yaml").read_text())
    for key in ("generation_prompt", "judge_prompt", "judge_schema"):
        path = ROOT / config[key + "_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest().upper() == config[key + "_hash"]


def test_runtime_hashes_lock_actual_release_bytes() -> None:
    metadata = json.loads((ROOT / "data/manifests/dataset_manifest.json").read_text())
    rationale = ROOT / "data/processed/rationales/azure_rationale_input_train.jsonl"
    assert hashlib.sha256(rationale.read_bytes()).hexdigest().upper() == metadata["active_rationale_sha256"]
    generated = ROOT / "data/processed/rationales/approved_generated_rationales_train.jsonl"
    assert hashlib.sha256(generated.read_bytes()).hexdigest().upper() == metadata["approved_generated_rationale_sha256"]
    mask_root = ROOT / "data/processed/q3_low_resource_sarcasm"
    masks = json.loads((mask_root / "manifest.json").read_text())
    for budget, expected in masks["mask_hashes"].items():
        assert hashlib.sha256((mask_root / f"budget_{budget}_masks.csv").read_bytes()).hexdigest().upper() == expected
    report = json.loads((ROOT / "data/manifests/release_contract.json").read_text())
    assert report["current_input_hash_locks_verified"] is True
    assert report["historical_provenance_verified"] is False
    assert report["release_content_verification"]["status"] == "PASS"


def test_missing_bundle_is_not_reported_as_metric_success() -> None:
    for script in ("verify_predictions.py", "verify_current_q2.py"):
        result = subprocess.run([sys.executable, "-B", str(ROOT / script)], capture_output=True, text=True)
        assert result.returncode == 2
        assert "BLOCKED" in result.stderr
        assert '"status": "PASS"' not in result.stdout


def test_upload_code_and_destinations_are_absent() -> None:
    for name in ("xlmr_followup.yaml", "q1a_matched_xlmr_ft.yaml"):
        config = yaml.safe_load((ROOT / "configs/experiments" / name).read_text())
        assert "upload" not in config
    assert not (ROOT / "scripts/hf_artifact_uploader.py").exists()
    removed = {"_append_queue", "_append_hf_queue", "_append_upload_queue", "_start_uploader"}
    for name in ("run_xlmr_followup.py", "run_visobert_baseline.py",
                 "run_priority12_xlmr.py"):
        content = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        tree = ast.parse(content)
        assert not any(isinstance(n, ast.FunctionDef) and n.name in removed for n in ast.walk(tree))
        assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                       and n.func.id in removed for n in ast.walk(tree))
        assert "hf_artifact_uploader" not in content


def test_analysis_does_not_emit_personal_upload_metadata() -> None:
    for name in ("analyze_matched_xlmr_ft_q1a.py",):
        tree = ast.parse((ROOT / "scripts" / name).read_text(encoding="utf-8"))
        strings = [node.value for node in ast.walk(tree)
                   if isinstance(node, ast.Constant) and isinstance(node.value, str)]
        assert not any(any(key in value for key in
                           ("upload_status.json", "artifact_repository", "checkpoint_repository", "hf_upload"))
                       for value in strings)


def test_preparation_never_overwrites_a_conflicting_file(tmp_path: Path) -> None:
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_text("correct")
    target.write_text("user work")
    with pytest.raises(RuntimeError, match="not overwritten"):
        copy(source, target)
    assert target.read_text() == "user work"
    target.write_text("correct")
    assert copy(source, target) == 0


def test_local_uncertainty_reader_requires_eight_scalars(tmp_path: Path) -> None:
    import torch

    spec = importlib.util.spec_from_file_location("local_variance_reader", ROOT / "scripts/read_learned_log_variances.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "parameters.pt"
    state = {f"log_variances.task{i}": torch.tensor(float(i)) for i in range(8)}
    torch.save({"loss_aggregator_state_dict": state}, path)
    assert len(module.read_parameters(path)) == 8
    torch.save({"loss_aggregator_state_dict": {}}, path)
    with pytest.raises(ValueError, match="eight"):
        module.read_parameters(path)
