from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

import vipragsent.data.tokenizers as tokenizers
import vipragsent.models.factory as model_factory
import vipragsent.orchestration.stage_registry as stage_registry
from vipragsent.data.loaders import DatasetExample
from vipragsent.hashing import sha256_file, sha256_json
from vipragsent.orchestration.contracts import RunContext, RunEntry, StageOutcome


def _entry() -> RunEntry:
    return RunEntry.from_mapping(
        {
            "run_id": "q1a_explanation_only_vistral_20260521",
            "research_question": "Q1a",
            "system_id": "explanation_only_vistral",
            "display_name": "Vistral explanation-only",
            "variant": "explanation_only",
            "backbone": "vistral_7b",
            "seed": 20260521,
            "execution_kind": "checkpoint_reuse",
            "model_revision": "model-rev",
            "tokenizer_revision": "tokenizer-rev",
            "source_checkpoint_id": "explanation_only_vistral:20260521",
        }
    )


def test_production_explanation_dispatch_uses_resumable_runtime_not_legacy_executor(
    monkeypatch, tmp_path: Path
) -> None:
    entry = _entry()
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    (run_root / "source").mkdir()
    (run_root / "source/source_provenance.json").write_text(json.dumps({"status": "PASS"}), encoding="utf-8")

    source_root = tmp_path / "approved-source"
    source_root.mkdir()
    checkpoint = source_root / "checkpoints/best.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"approved-source-checkpoint")
    data_hash = "A" * 64
    (source_root / "review_summary.json").write_text(
        json.dumps({"dataset_identity": "dataset-v1", "dataset_fingerprint": data_hash}),
        encoding="utf-8",
    )
    source = SimpleNamespace(
        run_id="full-vistral-20260521",
        run_root=source_root,
        checkpoint_path=checkpoint,
        checkpoint_sha256=sha256_file(checkpoint),
        review_summary_sha256="summary-sha",
        approval_sha256="approval-sha",
        checksum_file_sha256="checksums-sha",
        config_sha256="config-sha",
        variant_fingerprint="full-variant",
        seed=20260521,
        model_revision="model-rev",
        tokenizer_revision="tokenizer-rev",
        as_dict=lambda _root: {
            "run_id": "full-vistral-20260521",
            "checkpoint_sha256": sha256_file(checkpoint),
            "seed": 20260521,
        },
    )
    protocol = {"protocol_version": "reasoning_generation_shared_judge_v1", "decoding": {"max_new_tokens": 160}}
    runtime_calls: list[tuple[object, object]] = []

    class _Runtime:
        def __init__(self, model, tokenizer, request, *, run_root) -> None:
            del model, tokenizer
            runtime_calls.append((request, Path(run_root)))

        def generate_reasoning_split(self, split, records):
            path = run_root / "reasoning" / f"{split}_reasoning.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "".join(json.dumps({"sample_id": row["sample_id"]}) + "\n" for row in records),
                encoding="utf-8",
            )
            return list(records)

    class _Judge:
        def __init__(self, *_args, **_kwargs) -> None:
            self.protocol = protocol

    monkeypatch.setattr(
        stage_registry,
        "_resolve_production_explanation_source",
        lambda context, observed: source,
    )
    monkeypatch.setattr(stage_registry, "_execution_spec", lambda *_args: SimpleNamespace(model_family="vistral_7b"))
    monkeypatch.setattr(stage_registry, "_resolve_production_device", lambda *_args: (0, None))
    monkeypatch.setattr(stage_registry, "read_family_status", lambda *_args: {"local_path": str(tmp_path / "cache")})
    monkeypatch.setattr(stage_registry, "resolve_local_snapshot", lambda *_args: tmp_path / "cache")
    monkeypatch.setattr(stage_registry, "infer_required_head_prefixes", lambda *_args: ())
    monkeypatch.setattr(stage_registry, "load_checkpoint", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(stage_registry, "resolve_model_input_device", lambda *_args: torch.device("cpu"))
    monkeypatch.setattr(stage_registry, "assert_runtime_device_contract", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(stage_registry, "write_device_report", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(model_factory, "build_production_model", lambda *_args, **_kwargs: (object(), SimpleNamespace(tokenizer_revision="tokenizer-rev")))
    monkeypatch.setattr(tokenizers, "create_tokenizer", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(stage_registry, "ReasoningJudge", _Judge)
    monkeypatch.setattr(stage_registry, "load_vipragsent", lambda *_args: SimpleNamespace(dev=[DatasetExample("dev-1", "text", {}, "dev")], test=[]))
    monkeypatch.setattr(stage_registry, "_encode_text", lambda *_args: (torch.tensor([[1, 2]]), torch.tensor([[1, 1]])))
    monkeypatch.setattr(stage_registry, "ExplanationOnlyRuntime", _Runtime)

    def _legacy_executor_must_not_run(*_args, **_kwargs):
        raise AssertionError("production explanation dispatch used the legacy executor")

    monkeypatch.setattr(stage_registry, "ExplanationReuseExecutor", _legacy_executor_must_not_run, raising=False)

    context = RunContext(
        tmp_path,
        entry,
        run_root=run_root,
        metadata={"data_hash": data_hash, "dataset_identity": "dataset-v1", "environment_version": "test-v1"},
    )
    outcome = stage_registry._production_explanation_stage(
        context,
        entry,
        "generate_dev_reasoning_from_rationale_decoder",
    )

    assert outcome.status == "PASS", outcome
    assert len(runtime_calls) == 1
    request, observed_root = runtime_calls[0]
    assert observed_root == run_root
    assert request.seed == entry.seed
    assert request.source_checkpoint.source_checkpoint_key == "vipragsent_full_vistral:20260521"
    assert request.source_checkpoint.checkpoint_sha256 == sha256_file(checkpoint)
    assert request.data_hash == data_hash
    assert request.dataset_identity == "dataset-v1"
    assert request.config.identity.protocol_hash == sha256_json(protocol)
    assert request.config.identity.environment_version == "test-v1"
    assert (run_root / "reasoning/dev_reasoning.jsonl").exists()


def test_explanation_registry_routes_reasoning_stage_to_explanation_handler(monkeypatch, tmp_path: Path) -> None:
    entry = _entry()
    context = RunContext(tmp_path, entry, run_root=tmp_path / "run")
    calls: list[str] = []

    def explanation_handler(_context, _entry, stage):
        calls.append(stage)
        return StageOutcome.passed()

    def generation_handler(*_args, **_kwargs):
        raise AssertionError("explanation stage was routed to the causal generation handler")

    monkeypatch.setattr(stage_registry, "_explanation_stage", explanation_handler)
    monkeypatch.setattr(stage_registry, "_generation_stage", generation_handler)
    handlers = stage_registry.build_single_experiment_stage_registry(tmp_path, entry, context)

    outcome = handlers["generate_dev_reasoning"]()

    assert outcome.status == "PASS"
    assert calls == ["generate_dev_reasoning_from_rationale_decoder"]
