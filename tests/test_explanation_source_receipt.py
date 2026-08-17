from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vipragsent.orchestration import explanation_runtime, stage_registry
from vipragsent.orchestration.contracts import RunContext, RunEntry
from vipragsent.orchestration.executors import explanation_reuse
from vipragsent.orchestration.executors.explanation_reuse import (
    SourceReceiptError,
    resolve_approved_full_vistral_source,
    validate_source_checkpoint,
    validate_validated_source_identity,
)
from vipragsent.orchestration.explanation_runtime import (
    ExplanationOnlyConfig,
    ExplanationOnlyRequest,
    ExplanationOnlyRuntime,
    ExplanationRuntimeError,
    SharedInferenceIdentity,
    SourceCheckpointIdentity,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _approved_source_fixture(root: Path) -> tuple[Path, dict[str, str]]:
    run_root = root / "results/runs/source-run"
    checkpoint = run_root / "checkpoints/best/model.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"approved source checkpoint")
    config = run_root / "config_snapshot.yaml"
    config.write_text("resolved: source\n", encoding="utf-8")
    checksums = run_root / "checksums.sha256"
    checksums.write_text("checkpoint-entry\n", encoding="utf-8")
    summary = run_root / "review_summary.json"
    _json(
        summary,
        {
            "system_id": "vipragsent_full_vistral",
            "seed": 20260521,
            "reusable_checkpoint_key": "vipragsent_full_vistral:20260521",
            "config_hash": _digest(config),
            "data_hash": "dataset-hash",
            "dataset_identity": "dataset-identity",
            "model_revision": "model-revision",
            "tokenizer_revision": "tokenizer-revision",
            "variant_fingerprint": "variant-fingerprint",
        },
    )
    approval = run_root / "approval_status.json"
    _json(
        approval,
        {
            "status": "APPROVED",
            "review_summary_sha256": _digest(summary),
            "artifact_checksum_file_sha256": _digest(checksums),
        },
    )
    _json(run_root / "state.json", {"run_status": "APPROVED"})
    _json(
        run_root / "checkpoints/checkpoint_manifest.json",
        {
            "best": "checkpoints/best/model.pt",
            "checkpoint_sha256": _digest(checkpoint),
            "variant_fingerprint": "variant-fingerprint",
            "model_revision": "model-revision",
            "tokenizer_revision": "tokenizer-revision",
            "data_hash": "dataset-hash",
            "dataset_identity": "dataset-identity",
        },
    )
    return checkpoint, {"seed": "20260521", "source_checkpoint_id": "vipragsent_full_vistral:20260521"}


class _Backbone(nn.Module):
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        del attention_mask
        return SimpleNamespace(last_hidden_state=input_ids.to(dtype=torch.float32).unsqueeze(-1).expand(-1, -1, 4))


class _Decoder:
    def greedy_decode(self, hidden: torch.Tensor, attention: torch.Tensor, bos: int, eos: int, maximum: int) -> torch.Tensor:
        del maximum
        values = hidden[:, :, 0].masked_fill(~attention.bool(), -1).max(dim=1).values.to(dtype=torch.long)
        return torch.stack((torch.full_like(values, bos), values, torch.full_like(values, eos)), dim=1)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(1))
        self.backbone = _Backbone()
        self.rationale_decoder = _Decoder()


class _Tokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def decode(self, ids: object, **_: object) -> str:
        values = ids.tolist() if isinstance(ids, torch.Tensor) else list(ids)  # type: ignore[arg-type]
        return " ".join(str(value) for value in values if value not in {0, 1, 2})


def _identity() -> SharedInferenceIdentity:
    return SharedInferenceIdentity(
        protocol_hash="protocol-hash",
        environment_identity="cpu-test",
        environment_version="torch-test",
    )


def test_source_is_hashed_once_and_receipt_reuse_performs_zero_more_hashes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint, entry = _approved_source_fixture(tmp_path)
    calls: list[Path] = []
    original_hash = explanation_reuse.sha256_file

    def counted_hash(path: str | Path) -> str:
        candidate = Path(path).resolve()
        if candidate == checkpoint.resolve():
            calls.append(candidate)
        return original_hash(path)

    monkeypatch.setattr(explanation_reuse, "sha256_file", counted_hash)
    monkeypatch.setattr(explanation_runtime, "sha256_file", counted_hash)

    receipt_root = tmp_path / "explanation-run"
    source = resolve_approved_full_vistral_source(tmp_path, entry, receipt_root=receipt_root, device="cpu")
    assert calls == [checkpoint.resolve()]
    receipt = validate_validated_source_identity(receipt_root)
    assert receipt is not None
    assert Path(receipt["checkpoint_path"]).is_absolute()
    assert receipt["checkpoint_sha256"] == source.checkpoint_sha256
    assert {"inode", "size", "mtime_ns", "device"}.issubset(receipt)
    assert validate_source_checkpoint(
        tmp_path,
        source,
        receipt_root=receipt_root,
        device="cpu",
    )["status"] == "PASS"

    request = ExplanationOnlyRequest(
        seed=20260521,
        source_checkpoint=SourceCheckpointIdentity.from_approved_source(source),
        config=ExplanationOnlyConfig(identity=_identity()),
        data_hash="dataset-hash",
        dataset_identity="dataset-identity",
        artifact_root=receipt_root,
        fixture_mode=False,
    )
    runtime = ExplanationOnlyRuntime(_Model(), _Tokenizer(), request, run_root=receipt_root)
    runtime.generate_reasoning_split(
        "dev",
        [{"sample_id": "d1", "input_ids": torch.tensor([[3, 4]]), "attention_mask": torch.ones((1, 2), dtype=torch.long)}],
    )
    assert calls == [checkpoint.resolve()], "receipt validation must not hash the physical checkpoint again"


def test_production_request_fails_closed_without_a_source_receipt(tmp_path: Path) -> None:
    checkpoint = tmp_path / "unverified-model.pt"
    checkpoint.write_bytes(b"unverified")
    request = ExplanationOnlyRequest(
        seed=20260521,
        source_checkpoint=SourceCheckpointIdentity(20260521, checkpoint, _digest(checkpoint), variant_fingerprint="variant"),
        config=ExplanationOnlyConfig(identity=_identity()),
        data_hash="dataset-hash",
        dataset_identity="dataset-identity",
        artifact_root=tmp_path / "explanation-run",
        fixture_mode=False,
    )
    with pytest.raises(ExplanationRuntimeError, match="validated source receipt"):
        request.validate()
    with pytest.raises(ExplanationRuntimeError, match="validated source receipt"):
        _ = request.fingerprint


def test_production_source_and_validate_stages_reuse_the_receipt_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint, source_entry = _approved_source_fixture(tmp_path)
    entry = RunEntry.from_mapping(
        {
            "run_id": "explanation-run",
            "system_id": "explanation_only_vistral",
            "execution_kind": "checkpoint_reuse",
            "backbone": "vistral_7b",
            "research_question": "Q1a",
            "seed": 20260521,
            "source_checkpoint_id": source_entry["source_checkpoint_id"],
        }
    )
    context = RunContext(tmp_path, entry, run_root=tmp_path / "explanation-run")
    calls: list[Path] = []
    original_hash = explanation_reuse.sha256_file

    def counted_hash(path: str | Path) -> str:
        candidate = Path(path).resolve()
        if candidate == checkpoint.resolve():
            calls.append(candidate)
        return original_hash(path)

    monkeypatch.setattr(explanation_reuse, "sha256_file", counted_hash)
    monkeypatch.setattr(stage_registry, "_resolve_production_device", lambda _root: (0, None))
    resolved = stage_registry._production_explanation_stage(context, entry, "resolve_approved_full_vistral_source")
    assert resolved.status == "PASS"
    assert (context.run_root / "source/validated_source_identity.json").exists()
    validated = stage_registry._production_explanation_stage(context, entry, "validate_source_checkpoint")
    assert validated.status == "PASS"
    assert calls == [checkpoint.resolve()]


@pytest.mark.parametrize("replacement", [False, True])
def test_mutated_or_replaced_checkpoint_is_rejected_without_rehashing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    replacement: bool,
) -> None:
    checkpoint, entry = _approved_source_fixture(tmp_path)
    receipt_root = tmp_path / "explanation-run"
    resolve_approved_full_vistral_source(tmp_path, entry, receipt_root=receipt_root, device="cpu")
    calls = 0
    original_hash = explanation_reuse.sha256_file

    def counted_hash(path: str | Path) -> str:
        nonlocal calls
        if Path(path).resolve() == checkpoint.resolve():
            calls += 1
        return original_hash(path)

    monkeypatch.setattr(explanation_reuse, "sha256_file", counted_hash)
    if replacement:
        replacement_path = checkpoint.with_name("replacement-model.pt")
        replacement_path.write_bytes(b"replacement source checkpoint")
        os.replace(replacement_path, checkpoint)
    else:
        checkpoint.write_bytes(b"mutated source checkpoint")

    with pytest.raises(SourceReceiptError, match="changed or replaced"):
        validate_validated_source_identity(receipt_root)
    assert calls == 0
