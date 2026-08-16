from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vipragsent.hashing import sha256_file
from vipragsent.orchestration.explanation_runtime import (
    EXPLANATION_ENGINE_ID,
    ExplanationOnlyConfig,
    ExplanationOnlyRequest,
    ExplanationOnlyRuntime,
    ExplanationRuntimeError,
    SharedInferenceIdentity,
    SourceCheckpointIdentity,
    resolve_explanation_source,
    validate_three_seed_binding,
)


class _TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(1))
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        self.calls.append((input_ids.detach().cpu().clone(), attention_mask.detach().cpu().clone()))
        hidden = input_ids.to(dtype=torch.float32).unsqueeze(-1).expand(-1, -1, 4)
        return SimpleNamespace(last_hidden_state=hidden + self.anchor)


class _TinyDecoder:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def greedy_decode(self, hidden: torch.Tensor, attention: torch.Tensor, bos: int, eos: int, maximum: int) -> torch.Tensor:
        self.calls.append((int(hidden.size(0)), int(hidden.size(1))))
        values = hidden[:, :, 0].masked_fill(~attention.bool(), -1).max(dim=1).values.to(dtype=torch.long)
        return torch.stack((torch.full_like(values, bos), values + 20, torch.full_like(values, eos)), dim=1)


class _TinyFullModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _TinyBackbone()
        self.rationale_decoder = _TinyDecoder()
        self.config = SimpleNamespace(use_cache=False)
        self.forward_called = False

    def forward(self, **_: object) -> dict[str, object]:
        self.forward_called = True
        raise AssertionError("classification/full forward path must not be used")


class _Tokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def decode(self, ids: object, **_: object) -> str:
        values = ids.tolist() if isinstance(ids, torch.Tensor) else list(ids)  # type: ignore[arg-type]
        return " ".join(f"tok{value}" for value in values if value not in {0, 1, 2})


def _identity(*, environment: str = "cpu-fixture") -> SharedInferenceIdentity:
    return SharedInferenceIdentity(
        protocol_hash="protocol-sha",
        environment_identity=environment,
        environment_version="torch-fixture",
    )


def _request(tmp_path: Path, source_path: Path, *, config: ExplanationOnlyConfig | None = None, batch_size: int | None = 1, legacy_root: Path | None = None) -> ExplanationOnlyRequest:
    return ExplanationOnlyRequest(
        seed=20260521,
        source_checkpoint=SourceCheckpointIdentity(20260521, source_path, sha256_file(source_path), variant_fingerprint="full-v1"),
        config=config or ExplanationOnlyConfig(identity=_identity()),
        data_hash="fixture-explanation-data",
        dataset_identity="fixture-explanation",
        batch_size=batch_size,
        artifact_root=tmp_path,
        legacy_artifact_root=legacy_root or tmp_path / "legacy",
        fixture_mode=True,
    )


def _records() -> list[dict[str, object]]:
    return [
        {"sample_id": "short", "input_ids": torch.tensor([[3, 4]]), "attention_mask": torch.tensor([[1, 1]])},
        {"sample_id": "long", "input_ids": torch.tensor([[5, 6, 7, 8]]), "attention_mask": torch.tensor([[1, 1, 1, 1]])},
        {"sample_id": "masked", "input_ids": torch.tensor([[9, 10, 11]]), "attention_mask": torch.tensor([[0, 1, 1]])},
    ]


def _approved_source_root(tmp_path: Path, checkpoint: Path, *, dataset_hash: str) -> Path:
    run_root = tmp_path / "results" / "runs" / "approved-full-vistral"
    (run_root / "checkpoints").mkdir(parents=True)
    checkpoint_target = run_root / "checkpoints" / "best.pt"
    checkpoint_target.write_bytes(checkpoint.read_bytes())
    config_path = run_root / "config_snapshot.yaml"
    config_path.write_text("system_id: vipragsent_full_vistral\n", encoding="utf-8")
    summary = {
        "system_id": "vipragsent_full_vistral",
        "seed": 20260521,
        "reusable_checkpoint_key": "vipragsent_full_vistral:20260521",
        "variant_fingerprint": "full-v1",
        "model_revision": "model-r1",
        "tokenizer_revision": "tokenizer-r1",
        "config_hash": sha256_file(config_path),
        "dataset_identity": "dataset-v1",
        "dataset_fingerprint": dataset_hash,
    }
    summary_path = run_root / "review_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    approval_path = run_root / "approval_status.json"
    approval_path.write_text(
        json.dumps(
            {
                "status": "APPROVED",
                "review_summary_sha256": sha256_file(summary_path),
            }
        ),
        encoding="utf-8",
    )
    checksums_path = run_root / "checksums.sha256"
    checksums_path.write_text("approved artifact list\n", encoding="utf-8")
    manifest_path = run_root / "checkpoints" / "checkpoint_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "best": "checkpoints/best.pt",
                "checkpoint_sha256": sha256_file(checkpoint_target),
                "variant_fingerprint": "full-v1",
                "model_revision": "model-r1",
                "tokenizer_revision": "tokenizer-r1",
            }
        ),
        encoding="utf-8",
    )
    state_path = run_root / "state.json"
    state_path.write_text(json.dumps({"run_status": "APPROVED"}), encoding="utf-8")
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    approval["artifact_checksum_file_sha256"] = sha256_file(checksums_path)
    approval["approval_sha256"] = sha256_file(approval_path)
    approval_path.write_text(json.dumps(approval), encoding="utf-8")
    return tmp_path


def test_runtime_is_inference_only_and_does_not_call_full_forward(tmp_path: Path) -> None:
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"approved-source")
    runtime = ExplanationOnlyRuntime(_TinyFullModel(), _Tokenizer(), _request(tmp_path / "run", checkpoint), run_root=tmp_path / "run")
    rows = runtime.generate_reasoning_split("dev", _records())
    assert [row["sample_id"] for row in rows] == ["short", "long", "masked"]
    assert runtime.model.forward_called is False
    assert not (tmp_path / "run/checkpoints").exists()
    with pytest.raises(ExplanationRuntimeError, match="no training path"):
        runtime.train()
    with pytest.raises(ExplanationRuntimeError, match="cannot create an optimizer"):
        runtime.create_optimizer()


def test_source_checkpoint_mismatch_and_unauthorized_fallback_are_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"approved-source")
    source = SourceCheckpointIdentity(20260521, checkpoint, "wrong-hash", variant_fingerprint="full-v1")
    with pytest.raises(ExplanationRuntimeError, match="hash mismatch"):
        ExplanationOnlyRuntime(
            _TinyFullModel(),
            _Tokenizer(),
            ExplanationOnlyRequest(
                seed=20260521,
                source_checkpoint=source,
                config=ExplanationOnlyConfig(identity=_identity()),
                data_hash="fixture-data",
                dataset_identity="fixture",
                artifact_root=tmp_path / "mismatch",
                fixture_mode=True,
            ),
            run_root=tmp_path / "mismatch",
        )
    with pytest.raises(ExplanationRuntimeError, match="unauthorized"):
        ExplanationOnlyRuntime(
            _TinyFullModel(),
            _Tokenizer(),
            ExplanationOnlyRequest(
                seed=20260521,
                source_checkpoint=SourceCheckpointIdentity(
                    20260521,
                    checkpoint,
                    sha256_file(checkpoint),
                    source_checkpoint_key="cot_only_vistral:20260521",
                ),
                config=ExplanationOnlyConfig(identity=_identity()),
                data_hash="fixture-data",
                dataset_identity="fixture",
                artifact_root=tmp_path / "unauthorized",
                fixture_mode=True,
            ),
            run_root=tmp_path / "unauthorized",
        )


def test_production_rejects_probe_source_but_accepts_resolved_approved_source(tmp_path: Path) -> None:
    checkpoint = tmp_path / "probe.pt"
    checkpoint.write_bytes(b"same-file-hash-is-not-approval")
    data_hash = "B" * 64
    unapproved = SourceCheckpointIdentity(
        20260521,
        checkpoint,
        sha256_file(checkpoint),
        variant_fingerprint="full-v1",
        model_revision="model-r1",
        tokenizer_revision="tokenizer-r1",
        review_summary_sha256="C" * 64,
        approval_sha256="D" * 64,
        checksum_file_sha256="E" * 64,
        config_sha256="F" * 64,
        dataset_identity="dataset-v1",
        dataset_hash=data_hash,
    )
    with pytest.raises(ExplanationRuntimeError, match="resolved through"):
        ExplanationOnlyRuntime(
            _TinyFullModel(),
            _Tokenizer(),
            ExplanationOnlyRequest(
                seed=20260521,
                source_checkpoint=unapproved,
                config=ExplanationOnlyConfig(identity=_identity()),
                data_hash=data_hash,
                dataset_identity="dataset-v1",
                artifact_root=tmp_path / "unapproved",
            ),
            run_root=tmp_path / "unapproved",
        )

    root = _approved_source_root(tmp_path / "approved", checkpoint, dataset_hash=data_hash)
    approved = resolve_explanation_source(root, seed=20260521)
    runtime = ExplanationOnlyRuntime(
        _TinyFullModel(),
        _Tokenizer(),
        ExplanationOnlyRequest(
            seed=20260521,
            source_checkpoint=approved,
            config=ExplanationOnlyConfig(identity=_identity()),
            data_hash=data_hash,
            dataset_identity="dataset-v1",
            artifact_root=root / "explanation",
        ),
        run_root=root / "explanation",
    )
    assert runtime.source.source_checkpoint_key == "vipragsent_full_vistral:20260521"


def test_engine_identity_is_bound_and_changed_engine_cannot_resume(tmp_path: Path) -> None:
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"approved-source")
    run_root = tmp_path / "run"
    first = ExplanationOnlyRuntime(_TinyFullModel(), _Tokenizer(), _request(run_root, checkpoint), run_root=run_root)
    first.generate_reasoning_split("dev", _records())
    changed = SharedInferenceIdentity(engine_id=EXPLANATION_ENGINE_ID, engine_version="task-h-v2", protocol_hash="protocol-sha", environment_identity="cpu-fixture", environment_version="torch-fixture")
    with pytest.raises(ExplanationRuntimeError, match="engine identity mismatch"):
        ExplanationOnlyRuntime(_TinyFullModel(), _Tokenizer(), _request(run_root, checkpoint, config=ExplanationOnlyConfig(identity=changed)), run_root=run_root)


def test_order_resume_finalization_and_batch_parity(tmp_path: Path) -> None:
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"approved-source")
    records = _records()
    interrupted_root = tmp_path / "interrupted"
    first_model = _TinyFullModel()
    first = ExplanationOnlyRuntime(first_model, _Tokenizer(), _request(interrupted_root, checkpoint), run_root=interrupted_root)

    def interrupt(_: object, __: object) -> None:
        raise RuntimeError("stop after committed chunk")

    with pytest.raises(RuntimeError, match="stop"):
        first.generate_reasoning_split("dev", records, on_committed_chunk=interrupt)
    resumed_model = _TinyFullModel()
    resumed = ExplanationOnlyRuntime(resumed_model, _Tokenizer(), _request(interrupted_root, checkpoint), run_root=interrupted_root)
    resumed_rows = resumed.generate_reasoning_split("dev", records)
    assert [row["sample_id"] for row in resumed_rows] == ["short", "long", "masked"]
    assert len(resumed_model.backbone.calls) == 2
    assert resumed.generate_reasoning_split("dev", records) == resumed_rows

    batch_root = tmp_path / "batch"
    batch_identity = _identity()
    batch_config = ExplanationOnlyConfig(identity=batch_identity, generation_profile={"status": "PASS", "selected_batch_size": 2, "profiled": True})
    batched_request = _request(batch_root, checkpoint, config=batch_config, batch_size=2)
    batched = ExplanationOnlyRuntime(_TinyFullModel(), _Tokenizer(), batched_request, run_root=batch_root)
    batched_rows = batched.generate_reasoning_split("dev", records)
    baseline_root = tmp_path / "baseline"
    baseline_config = ExplanationOnlyConfig(identity=batch_identity, generation_profile={"status": "PASS", "selected_batch_size": 1, "profiled": True})
    baseline = ExplanationOnlyRuntime(_TinyFullModel(), _Tokenizer(), _request(baseline_root, checkpoint, config=baseline_config, batch_size=1), run_root=baseline_root)
    baseline_rows = baseline.generate_reasoning_split("dev", records)
    assert [(row["sample_id"], row["generated_reasoning"]) for row in batched_rows] == [(row["sample_id"], row["generated_reasoning"]) for row in baseline_rows]


def test_three_seeds_share_one_frozen_identity(tmp_path: Path) -> None:
    requests: list[ExplanationOnlyRequest] = []
    for seed in (20260521, 20260522, 20260523):
        checkpoint = tmp_path / f"source-{seed}.pt"
        checkpoint.write_bytes(f"source-{seed}".encode())
        requests.append(
            ExplanationOnlyRequest(
                seed=seed,
                source_checkpoint=SourceCheckpointIdentity(seed, checkpoint, sha256_file(checkpoint), variant_fingerprint="full-v1"),
                config=ExplanationOnlyConfig(identity=_identity()),
                data_hash="fixture-data",
                dataset_identity="fixture",
                fixture_mode=True,
            )
        )
    assert validate_three_seed_binding(requests) == requests[0].config.identity.fingerprint


def test_legacy_artifacts_are_separate_and_duplicate_input_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"approved-source")
    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    legacy_file = legacy_root / "reasoning.jsonl"
    legacy_file.write_text("legacy-artifact\n", encoding="utf-8")
    run_root = tmp_path / "explanation"
    runtime = ExplanationOnlyRuntime(
        _TinyFullModel(),
        _Tokenizer(),
        _request(run_root, checkpoint, legacy_root=legacy_root),
        run_root=run_root,
    )
    with pytest.raises(ExplanationRuntimeError, match="duplicate sample IDs"):
        runtime.generate_reasoning_split("dev", [_records()[0], _records()[0]])
    assert legacy_file.read_text(encoding="utf-8") == "legacy-artifact\n"
