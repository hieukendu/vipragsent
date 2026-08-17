from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

import vipragsent.training.generation_checkpoint as checkpoint_module
from vipragsent.hashing import sha256_file
from vipragsent.orchestration.approval import validate_approval_record
from vipragsent.orchestration.executors.generation import ReasoningGenerationExecutor
from vipragsent.training.generation_checkpoint import (
    GENERATION_CHECKPOINT_POINTER_SCHEMA_VERSION,
    GENERATION_SELECTION_METRIC_NAME,
    GenerationCheckpointError,
    canonical_generation_epoch_path,
    read_generation_checkpoint_pointer,
    save_generation_checkpoint,
    write_generation_checkpoint_pointer,
)


def _provenance() -> dict[str, object]:
    dataset_hash = "A" * 64
    return {
        "model": {"name": "tiny", "revision": "r1"},
        "model_artifact": {"identity": "tiny-model@r1"},
        "tokenizer_artifact": {"identity": "tiny-tokenizer@r1"},
        "dataset": {"identity": "tiny-dataset", "hash": dataset_hash},
        "data_hash": dataset_hash,
        "optimizer": {"name": "AdamW", "revision": "torch"},
        "scheduler": {"name": "linear", "total_steps": 4},
        "rng": {"seed": 17, "algorithm": "torch+numpy+python"},
        "data_order": {"epoch": 1, "indices": [2, 0, 1]},
        "config": {"batch_size": 1, "gradient_accumulation": 1},
        "model_environment": {"device": "cpu", "dtype": "float32"},
    }


def _save_epoch(
    run_root: Path,
    epoch: int,
    *,
    variant_fingerprint: str = "variant-v1",
    selection_metric: float | None = None,
) -> str:
    path = run_root / canonical_generation_epoch_path(epoch)
    manifest = save_generation_checkpoint(
        path,
        nn.Linear(2, 1),
        None,
        None,
        {"epoch": epoch, "selection_metric": selection_metric, "data_order": []},
        _provenance(),
        fixture_mode=True,
        epoch=epoch,
        variant_fingerprint=variant_fingerprint,
        selection_metric_name=GENERATION_SELECTION_METRIC_NAME,
        selection_metric_value=selection_metric,
    )
    return str(manifest.checkpoint_sha256)


class _TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2


def _protocol_root(tmp_path: Path) -> Path:
    source = Path(__file__).resolve().parents[1]
    for relative in (
        "configs/experiments/generation_reasoning_protocol.yaml",
        "prompts/protocols/cot_only_reasoning_vi_v1.txt",
        "prompts/protocols/reasoning_judge_gpt41mini_zeroshot_v1.txt",
        "schemas/reasoning_judge_output.schema.json",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
    protocol_path = tmp_path / "configs/experiments/generation_reasoning_protocol.yaml"
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    protocol["generation_prompt_hash"] = sha256_file(tmp_path / str(protocol["generation_prompt_path"]))
    protocol["judge_prompt_hash"] = sha256_file(tmp_path / str(protocol["judge_prompt_path"]))
    protocol["judge_schema_hash"] = sha256_file(tmp_path / str(protocol["judge_schema_path"]))
    protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    return tmp_path


def _executor(tmp_path: Path) -> ReasoningGenerationExecutor:
    root = _protocol_root(tmp_path)
    return ReasoningGenerationExecutor(
        root,
        model=nn.Linear(2, 1),
        tokenizer=_TinyTokenizer(),
        judge=object(),
        run_root=root / "run",
        seed=17,
        data_hash="A" * 64,
        dataset_identity="tiny-dataset",
        model_artifact_identity="tiny-model@r1",
        tokenizer_artifact_identity="tiny-tokenizer@r1",
        fixture_mode=True,
    )


def test_epoch_payload_is_written_once_and_pointers_share_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    writes: list[Path] = []
    original_save = checkpoint_module.save_checkpoint

    def counting_save(path: str | Path, payload: object) -> None:
        writes.append(Path(path))
        original_save(path, payload)

    monkeypatch.setattr(checkpoint_module, "save_checkpoint", counting_save)
    run_root = tmp_path / "run"
    _save_epoch(run_root, 1)
    latest = write_generation_checkpoint_pointer(run_root, "latest", canonical_generation_epoch_path(1), variant_fingerprint="variant-v1")
    best = write_generation_checkpoint_pointer(
        run_root,
        "best",
        canonical_generation_epoch_path(1),
        selection_metric_value=0.75,
        variant_fingerprint="variant-v1",
    )

    assert writes == [run_root / canonical_generation_epoch_path(1)]
    assert sorted(path.relative_to(run_root).as_posix() for path in run_root.rglob("model.pt")) == ["checkpoints/epoch_0001/model.pt"]
    assert set(latest) == {
        "schema_version",
        "path",
        "epoch",
        "checkpoint_sha256",
        "provenance_sha256",
        "variant_fingerprint",
        "selection_metric_name",
        "selection_metric_value",
    }
    assert latest["schema_version"] == GENERATION_CHECKPOINT_POINTER_SCHEMA_VERSION
    assert latest["path"] == best["path"] == "checkpoints/epoch_0001/model.pt"
    assert latest["checkpoint_sha256"] == best["checkpoint_sha256"]
    assert read_generation_checkpoint_pointer(run_root, "latest")["epoch"] == 1
    assert read_generation_checkpoint_pointer(run_root, "best")["selection_metric_value"] == pytest.approx(0.75)


def test_pointer_updates_move_only_tiny_metadata(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _save_epoch(run_root, 1, selection_metric=0.7)
    _save_epoch(run_root, 2, selection_metric=0.8)
    write_generation_checkpoint_pointer(run_root, "latest", canonical_generation_epoch_path(1), selection_metric_value=0.7, variant_fingerprint="variant-v1")
    write_generation_checkpoint_pointer(run_root, "best", canonical_generation_epoch_path(1), selection_metric_value=0.7, variant_fingerprint="variant-v1")
    write_generation_checkpoint_pointer(run_root, "latest", canonical_generation_epoch_path(2), selection_metric_value=0.8, variant_fingerprint="variant-v1")

    assert read_generation_checkpoint_pointer(run_root, "latest")["epoch"] == 2
    assert read_generation_checkpoint_pointer(run_root, "best")["epoch"] == 1
    assert not (run_root / "checkpoints/latest/model.pt").exists()
    assert not (run_root / "checkpoints/best/model.pt").exists()


def test_executor_resolves_latest_best_and_rolls_back_without_copying(tmp_path: Path) -> None:
    executor = _executor(tmp_path)
    for epoch, value, metric in ((1, 1.0, 0.7), (2, 2.0, 0.8)):
        for parameter in executor.model.parameters():
            parameter.data.fill_(value)
        path = canonical_generation_epoch_path(epoch)
        executor.write_epoch_checkpoint(epoch, selection_metric=metric)
        executor.write_checkpoint_pointer("latest", path, selection_metric_value=metric)
        executor.write_checkpoint_pointer("best", path, selection_metric_value=metric)

    latest = executor.load_latest_checkpoint(restore_training_state=False)
    assert latest["run_state"]["epoch"] == 2
    assert float(next(executor.model.parameters()).detach().flatten()[0]) == pytest.approx(2.0)
    absolute_latest = executor.load_checkpoint(executor.run_root / "checkpoints/latest_checkpoint.json", restore_training_state=False)
    assert absolute_latest["run_state"]["epoch"] == 2
    best = executor.load_best_checkpoint(restore_training_state=False)
    assert best["run_state"]["epoch"] == 2
    rollback = executor.rollback_to_epoch(1, update_latest=True)
    assert rollback["run_state"]["epoch"] == 1
    assert read_generation_checkpoint_pointer(executor.run_root, "latest")["epoch"] == 1
    canonical_payloads = sorted(
        path.relative_to(executor.run_root).as_posix()
        for path in (executor.run_root / "checkpoints").glob("epoch_[0-9][0-9][0-9][0-9]/model.pt")
    )
    assert canonical_payloads == ["checkpoints/epoch_0001/model.pt", "checkpoints/epoch_0002/model.pt"]


def test_checkpoint_manifest_carries_pointer_provenance_and_export_identity(tmp_path: Path) -> None:
    executor = _executor(tmp_path)
    executor.write_epoch_checkpoint(1, selection_metric=0.75)
    path = canonical_generation_epoch_path(1)
    executor.write_checkpoint_pointer("latest", path, selection_metric_value=0.75)
    executor.write_checkpoint_pointer("best", path, selection_metric_value=0.75)
    manifest = executor.write_checkpoint_manifest(best_epoch=1, selection_metric=0.75, latest_epoch=1, latest_selection_metric=0.75)
    best = read_generation_checkpoint_pointer(executor.run_root, "best")

    assert manifest["best"] == manifest["latest"] == path
    assert manifest["best_pointer"] == "checkpoints/best_checkpoint.json"
    assert manifest["latest_pointer"] == "checkpoints/latest_checkpoint.json"
    assert manifest["checkpoint_sha256"] == best["checkpoint_sha256"]
    assert manifest["provenance_sha256"] == best["provenance_sha256"]
    assert manifest["variant_fingerprint"] == best["variant_fingerprint"]


def test_crash_before_pointer_preserves_previous_latest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_root = tmp_path / "run"
    _save_epoch(run_root, 1)
    write_generation_checkpoint_pointer(run_root, "latest", canonical_generation_epoch_path(1), variant_fingerprint="variant-v1")
    _save_epoch(run_root, 2)
    pointer_path = run_root / "checkpoints/latest_checkpoint.json"
    before = pointer_path.read_bytes()
    original_write = checkpoint_module.atomic_write_json

    def crash(path: str | Path, payload: object) -> None:
        if Path(path) == pointer_path:
            raise OSError("simulated crash before pointer replace")
        original_write(path, payload)

    monkeypatch.setattr(checkpoint_module, "atomic_write_json", crash)
    with pytest.raises(OSError, match="simulated crash"):
        write_generation_checkpoint_pointer(run_root, "latest", canonical_generation_epoch_path(2), variant_fingerprint="variant-v1")
    assert pointer_path.read_bytes() == before
    assert (run_root / canonical_generation_epoch_path(2)).is_file()


def test_missing_target_and_out_of_root_are_rejected(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    with pytest.raises(GenerationCheckpointError, match="target is missing"):
        write_generation_checkpoint_pointer(run_root, "latest", "checkpoints/epoch_0001/model.pt", variant_fingerprint="variant-v1")
    outside = tmp_path / "outside/model.pt"
    with pytest.raises(GenerationCheckpointError, match="outside the run root"):
        write_generation_checkpoint_pointer(run_root, "latest", outside, variant_fingerprint="variant-v1")


def test_pointer_sha_and_epoch_path_mismatches_are_rejected(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _save_epoch(run_root, 1)
    write_generation_checkpoint_pointer(run_root, "latest", canonical_generation_epoch_path(1), variant_fingerprint="variant-v1")
    pointer_path = run_root / "checkpoints/latest_checkpoint.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["checkpoint_sha256"] = "0" * 64
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
    with pytest.raises(GenerationCheckpointError, match="pointer SHA"):
        read_generation_checkpoint_pointer(run_root, "latest")

    pointer["checkpoint_sha256"] = sha256_file(run_root / canonical_generation_epoch_path(1))
    pointer["epoch"] = 2
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
    with pytest.raises(GenerationCheckpointError, match="epoch disagrees"):
        read_generation_checkpoint_pointer(run_root, "latest")


def test_legacy_physical_best_remains_readable_without_rewrite(tmp_path: Path) -> None:
    executor = _executor(tmp_path)
    path = executor.run_root / "checkpoints/best/model.pt"
    executor.write_checkpoint("checkpoints/best/model.pt", epoch=1, selection_metric=0.5)
    before = path.read_bytes()
    report = executor.load_best_checkpoint(restore_training_state=False)
    assert report["run_state"]["epoch"] == 1
    assert report["checkpoint_pointer"]["legacy"] is True
    assert path.read_bytes() == before
    assert not (executor.run_root / "checkpoints/best_checkpoint.json").exists()


def test_approval_validation_binds_modern_generation_pointers(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _save_epoch(run_root, 1)
    write_generation_checkpoint_pointer(run_root, "latest", canonical_generation_epoch_path(1), variant_fingerprint="variant-v1")
    write_generation_checkpoint_pointer(run_root, "best", canonical_generation_epoch_path(1), selection_metric_value=0.5, variant_fingerprint="variant-v1")
    (run_root / "state.json").parent.mkdir(parents=True, exist_ok=True)
    (run_root / "state.json").write_text(json.dumps({"run_id": "run", "run_status": "APPROVED", "approval_status": "APPROVED"}), encoding="utf-8")
    summary_path = run_root / "review_summary.json"
    checksums_path = run_root / "checksums.sha256"
    summary_path.write_text("{}", encoding="utf-8")
    checksums_path.write_text("{}", encoding="utf-8")
    timestamp = "2026-08-17T00:00:00Z"
    record = {
        "run_id": "run",
        "decision": "approve",
        "review_note": "ok",
        "approved_or_rejected_by": "reviewer",
        "timestamp": timestamp,
        "review_summary_sha256": sha256_file(summary_path),
        "artifact_checksum_file_sha256": sha256_file(checksums_path),
    }
    (run_root / "approval_status.json").write_text(
        json.dumps({"run_id": "run", "status": "APPROVED", "approved_by": "reviewer", "approved_at": timestamp, "record": record}),
        encoding="utf-8",
    )
    assert validate_approval_record(run_root) == []

    pointer_path = run_root / "checkpoints/best_checkpoint.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["checkpoint_sha256"] = "0" * 64
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
    errors = validate_approval_record(run_root)
    assert any("best generation checkpoint pointer is invalid" in error for error in errors)
