from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn

from vipragsent.evaluation.reasoning_judge import ReasoningJudge
from vipragsent.hashing import sha256_file
from vipragsent.orchestration.executors.generation import (
    GenerationPersistenceError,
    GenerationRecordError,
    ReasoningGenerationExecutor,
    reversible_inference_context,
    select_generation_batch_size,
)
from vipragsent.orchestration.generation_persistence import GenerationChunkStore


class _BatchFixtureModel(nn.Module):
    def __init__(self, *, fail_token: int | None = None) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(use_cache=False)
        self.fail_token = fail_token
        self.generate_calls: list[tuple[int, torch.Tensor]] = []

    def generate(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor, **_: object) -> torch.Tensor:
        self.generate_calls.append((int(input_ids.size(0)), attention_mask.detach().cpu().clone()))
        if self.fail_token is not None and bool((input_ids == self.fail_token).any()):
            raise GenerationRecordError("fixture sample failure")
        continuation = torch.tensor([[7, 8, 2]] * input_ids.size(0), dtype=torch.long)
        return torch.cat((input_ids.cpu(), continuation), dim=1)


class _ModelWideRuntimeErrorModel(_BatchFixtureModel):
    def generate(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor, **_: object) -> torch.Tensor:
        raise RuntimeError("model-wide runtime failure")


class _BatchFixtureTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def decode(self, ids: object, **_: object) -> str:
        values = ids.tolist() if isinstance(ids, torch.Tensor) else list(ids)  # type: ignore[arg-type]
        return "".join(f"t{value}" for value in values if value not in {0, 2})


class _LastNonPadCausalFixtureModel(nn.Module):
    """Fixture whose continuation depends on each row's actual final token."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(use_cache=False)

    def generate(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor, **_: object) -> torch.Tensor:
        positions = torch.arange(input_ids.size(1), device=input_ids.device).expand_as(input_ids)
        last_positions = positions.masked_fill(attention_mask == 0, -1).max(dim=1).values
        last_tokens = input_ids.gather(1, last_positions.unsqueeze(1)).squeeze(1)
        continuation = torch.stack((last_tokens + 10, torch.full_like(last_tokens, 2)), dim=1)
        return torch.cat((input_ids, continuation), dim=1)


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


def _executor(
    tmp_path: Path,
    model: nn.Module,
    *,
    profile: dict[str, object] | None = None,
    data_hash: str = "NOT_PROVIDED",
    config_hash: str = "NOT_PROVIDED",
    code_identity: str | None = None,
) -> ReasoningGenerationExecutor:
    root = _protocol_root(tmp_path)
    judge = ReasoningJudge(
        root,
        transport=lambda **_: {"labels": {label: 0 for label in ("sarcasm", "irony", "idiom_figurative", "code_switching", "mocking", "implicit_sentiment")}},
        cache_root=root / "judge-cache",
        sleep_fn=lambda _: None,
    )
    return ReasoningGenerationExecutor(
        root,
        model=model,
        tokenizer=_BatchFixtureTokenizer(),
        judge=judge,
        run_root=root / "run",
        fixture_mode=True,
        generation_profile=profile,
        data_hash=data_hash,
        config_hash=config_hash,
        code_identity=code_identity,
    )


def _records() -> list[dict[str, object]]:
    gold = {label: 0 for label in ("sarcasm", "irony", "idiom_figurative", "code_switching", "mocking", "implicit_sentiment")}
    return [
        {"sample_id": "a", "input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]]), "gold": gold},
        {"sample_id": "b", "input_ids": torch.tensor([[3, 4, 5, 6]]), "attention_mask": torch.tensor([[1, 1, 1, 1]]), "gold": gold},
        {"sample_id": "bad", "input_ids": torch.tensor([[9, 10, 11]]), "attention_mask": torch.tensor([[1, 1, 1]]), "gold": gold},
        {"sample_id": "d", "input_ids": torch.tensor([[12, 13]]), "attention_mask": torch.tensor([[1, 1]]), "gold": gold},
    ]


def test_generation_batch_requires_passing_profile_and_defaults_safe() -> None:
    assert select_generation_batch_size() == 1
    with pytest.raises(GenerationPersistenceError, match="explicit passing"):
        select_generation_batch_size(requested=2)
    assert select_generation_batch_size({"status": "PASS", "selected_batch_size": 2, "profiled": True}) == 2
    with pytest.raises(GenerationPersistenceError, match="profiled/approved evidence"):
        select_generation_batch_size({"status": "PASS", "selected_batch_size": 2})
    with pytest.raises(GenerationPersistenceError):
        select_generation_batch_size({"status": "BLOCKED", "selected_batch_size": 2})


def test_reversible_inference_context_restores_training_and_cache(tmp_path: Path) -> None:
    model = _BatchFixtureModel()
    model.train()
    with reversible_inference_context(model):
        assert not model.training
        assert model.config.use_cache is True
    assert model.training is True
    assert model.config.use_cache is False


def test_batched_generation_pads_stops_preserves_failures_and_resumes(tmp_path: Path) -> None:
    model = _BatchFixtureModel(fail_token=9)
    executor = _executor(tmp_path, model, profile={"status": "PASS", "selected_batch_size": 2, "profiled": True})
    records = _records()
    first = executor.generate_reasoning_split("dev", records)
    assert [row["sample_id"] for row in first] == ["a", "b", "bad", "d"]
    assert first[2]["generation_status"] == "INVALID"
    assert first[2]["failure_reason"]
    assert first[0]["truncated"] is False
    assert model.generate_calls[0][0] == 2
    assert model.generate_calls[0][1].tolist() == [[0, 0, 1, 1], [1, 1, 1, 1]]
    assert (executor.run_root / "reasoning/dev_chunks_manifest.json").exists()

    resumed_model = _BatchFixtureModel(fail_token=9)
    resumed = _executor(tmp_path, resumed_model, profile={"status": "PASS", "selected_batch_size": 2, "profiled": True})
    second = resumed.generate_reasoning_split("dev", records)
    assert second == first
    assert resumed_model.generate_calls == []


def test_model_wide_runtime_error_propagates_and_leaves_manifest_incomplete(tmp_path: Path) -> None:
    executor = _executor(
        tmp_path,
        _ModelWideRuntimeErrorModel(),
        profile={"status": "PASS", "selected_batch_size": 2, "profiled": True},
    )

    with pytest.raises(RuntimeError, match="model-wide runtime failure"):
        executor.generate_reasoning_split("dev", _records()[:2])

    manifest = json.loads((executor.run_root / "reasoning/dev_chunks_manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is False
    assert manifest["chunks"] == []
    assert not (executor.run_root / "reasoning/dev_reasoning.jsonl").exists()


def test_fixture_equivalence_harness_covers_candidate_batches(tmp_path: Path) -> None:
    executor = _executor(tmp_path, _BatchFixtureModel())
    report = executor.fixture_generation_equivalence("dev", _records()[:2])
    assert report == {"fixture_only": True, "baseline_batch_size": 1, "candidates": [1, 2, 4], "equivalent": True}


def test_left_padded_causal_generation_is_equivalent_for_batch_1_2_4(tmp_path: Path) -> None:
    executor = _executor(tmp_path, _LastNonPadCausalFixtureModel())
    records = [
        {"sample_id": "short", "input_ids": torch.tensor([[11, 12]]), "attention_mask": torch.tensor([[1, 1]])},
        {"sample_id": "long", "input_ids": torch.tensor([[21, 22, 23, 24]]), "attention_mask": torch.tensor([[1, 1, 1, 1]])},
        {"sample_id": "mid", "input_ids": torch.tensor([[31, 32, 33]]), "attention_mask": torch.tensor([[1, 1, 1]])},
        {"sample_id": "tiny", "input_ids": torch.tensor([[41]]), "attention_mask": torch.tensor([[1]])},
    ]
    baseline = executor._generate_reasoning_rows("dev", records, batch_size=1)
    baseline_projection = [(row["sample_id"], row["generated_reasoning"]) for row in baseline]
    for batch_size in (2, 4):
        current = executor._generate_reasoning_rows("dev", records, batch_size=batch_size)
        assert [(row["sample_id"], row["generated_reasoning"]) for row in current] == baseline_projection
    batch = executor._inference_batch(records[:2])
    assert batch["input_ids"].tolist() == [[0, 0, 11, 12], [21, 22, 23, 24]]
    assert batch["attention_mask"].tolist() == [[0, 0, 1, 1], [1, 1, 1, 1]]


def test_committed_chunk_rejects_changed_rows(tmp_path: Path) -> None:
    executor = _executor(tmp_path, _BatchFixtureModel(), profile={"status": "PASS", "selected_batch_size": 2, "profiled": True})
    executor.generate_reasoning_split("dev", _records()[:2])
    manifest = json.loads((executor.run_root / "reasoning/dev_chunks_manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert len(manifest["chunks"]) == 1
    assert manifest["generation_contract"]["input_record_digest"]
    assert manifest["generation_contract"]["record_order_digest"]


@pytest.mark.parametrize(
    ("first_kwargs", "second_kwargs", "match"),
    [
        ({"data_hash": "A" * 64}, {"data_hash": "B" * 64}, "contract identity mismatch"),
        ({"config_hash": "config-a"}, {"config_hash": "config-b"}, "contract identity mismatch"),
        ({"code_identity": "code-a"}, {"code_identity": "code-b"}, "contract identity mismatch"),
    ],
)
def test_generation_resume_rejects_changed_data_config_or_code(
    tmp_path: Path,
    first_kwargs: dict[str, object],
    second_kwargs: dict[str, object],
    match: str,
) -> None:
    first = _executor(tmp_path, _BatchFixtureModel(), **first_kwargs)
    first.generate_reasoning_split("dev", _records()[:2])
    second = _executor(tmp_path, _BatchFixtureModel(), **second_kwargs)
    with pytest.raises(GenerationPersistenceError, match=match):
        second.generate_reasoning_split("dev", _records()[:2])


def test_generation_resume_rejects_changed_model_state(tmp_path: Path) -> None:
    first = _executor(tmp_path, _BatchFixtureModel())
    first.generate_reasoning_split("dev", _records()[:2])
    changed_model = _BatchFixtureModel()
    with torch.no_grad():
        changed_model.anchor.fill_(2)
    second = _executor(tmp_path, changed_model)
    with pytest.raises(GenerationPersistenceError, match="contract identity mismatch"):
        second.generate_reasoning_split("dev", _records()[:2])


def test_generation_resume_rejects_changed_record_order(tmp_path: Path) -> None:
    first = _executor(tmp_path, _BatchFixtureModel())
    records = _records()[:2]
    first.generate_reasoning_split("dev", records)
    second = _executor(tmp_path, _BatchFixtureModel())
    with pytest.raises(GenerationPersistenceError, match="input identity mismatch"):
        second.generate_reasoning_split("dev", list(reversed(records)))


def test_model_state_identity_is_cached_across_generation_splits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vipragsent.orchestration.executors.generation as generation_module

    calls = 0
    original = generation_module._model_state_identity

    def counted(model: nn.Module) -> dict[str, str]:
        nonlocal calls
        calls += 1
        return original(model)

    monkeypatch.setattr(generation_module, "_model_state_identity", counted)
    executor = _executor(tmp_path, _BatchFixtureModel())
    executor.generate_reasoning_split("dev", _records()[:2])
    executor.generate_reasoning_split("test", _records()[:2])
    assert calls == 1


def test_training_epoch_generation_binds_to_persisted_checkpoint_without_live_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vipragsent.orchestration.executors.generation as generation_module

    executor = _executor(tmp_path, _BatchFixtureModel())
    observed_contracts: dict[str, dict[str, object]] = {}

    def forbidden_live_hash(_: nn.Module) -> dict[str, str]:
        raise AssertionError("DEV generation must use the canonical checkpoint identity")

    monkeypatch.setattr(generation_module, "_model_state_identity", forbidden_live_hash)
    model = executor.model
    train = [{"sample_id": "train", "input_ids": torch.tensor([[1, 2]]), "target_ids": torch.tensor([[3, 4]])}]
    dev = [{"sample_id": "dev", "input_ids": torch.tensor([[1, 2]]), "gold": {}}]
    test = [{"sample_id": "test", "input_ids": torch.tensor([[1, 2]]), "gold": {}}]

    monkeypatch.setattr(
        executor,
        "train_generation",
        lambda *_, epoch_start=1, **__: [{"epoch": float(epoch_start), "train_loss": 0.0}],
    )

    def generate(split: str, records: list[dict[str, object]], **__: object) -> list[dict[str, object]]:
        observed_contracts[split] = executor._generation_contract(split, records)
        return [{"sample_id": str(records[0]["sample_id"]), "generation_status": "PASS", "generated_reasoning": split}]

    monkeypatch.setattr(executor, "generate_reasoning_split", generate)
    monkeypatch.setattr(executor, "judge_reasoning_split", lambda _split, rows, _gold, **__: (list(rows), []))
    monkeypatch.setattr(executor, "compute_split_metrics", lambda split, _rows, **__: {"primary_macro_f1": 0.5 if split == "dev" else 0.0})

    executor.run_cot(
        train_records=train,
        dev_records=dev,
        test_records=test,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        epochs=1,
    )

    epoch_checkpoint = executor.run_root / "checkpoints/epoch_1/model.pt"
    assert observed_contracts["dev"]["checkpoint_identity"] == {"checkpoint_sha256": sha256_file(epoch_checkpoint)}


def test_generation_store_requires_contract_outside_explicit_fixture_or_legacy_mode(tmp_path: Path) -> None:
    with pytest.raises(GenerationPersistenceError, match="canonical generation contract"):
        GenerationChunkStore(tmp_path / "production", "dev", ["a"])
    GenerationChunkStore(tmp_path / "fixture", "dev", ["a"], fixture_mode=True)
    GenerationChunkStore(tmp_path / "legacy", "dev", ["a"], legacy_mode=True)


def test_production_contract_recurses_identity_values_and_accepts_canonical_config(tmp_path: Path) -> None:
    contract = {
        "contract_version": 1,
        "source_identity": {"run_id": "run-1", "source": "checkpoint-1"},
        "code_identity": {"commit": "commit-1", "source_fingerprint": "source-1"},
        "model_identity": {"identity": "model@revision-1"},
        "tokenizer_identity": {"identity": "tokenizer@revision-1"},
        "checkpoint_identity": {"checkpoint_sha256": "A" * 64},
        "config_identity": {
            "config_hash": "config-1",
            "protocol": {"protocol_id": "protocol-1", "decoding": {"do_sample": False}},
            "generation_profile": None,
        },
        "dataset_identity": {"identity": "dataset-1", "hash": "B" * 64},
        "split": "dev",
        "data_hash": "B" * 64,
        "input_record_digest": "C" * 64,
        "record_order_digest": "D" * 64,
        "seed": 20260521,
        "system_identity": {"system_id": "cot_only_vistral"},
        "budget": "full",
    }
    GenerationChunkStore(tmp_path / "valid", "dev", ["a"], generation_contract=contract)

    invalid = {
        **contract,
        "source_identity": {"nested": {"identity": "NOT_PROVIDED"}},
    }
    with pytest.raises(GenerationPersistenceError, match="source_identity"):
        GenerationChunkStore(tmp_path / "invalid", "dev", ["a"], generation_contract=invalid)


def test_chunk_commit_is_idempotent_and_rejects_rewrites(tmp_path: Path) -> None:
    store = GenerationChunkStore(tmp_path, "dev", ["a"], fixture_mode=True)
    row = {"sample_id": "a", "generated_reasoning": "stable"}
    first = store.commit([row])
    assert store.commit([row]) == first
    with pytest.raises(GenerationPersistenceError, match="rewrite"):
        store.commit([{**row, "generated_reasoning": "changed"}])


def test_generation_chunk_store_keeps_committed_state_in_memory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    reads: list[Path] = []
    signatures: list[Path] = []
    original_read_rows = GenerationChunkStore._read_rows
    original_file_signature = GenerationChunkStore._file_signature

    def counted_read_rows(path: Path) -> list[dict[str, object]]:
        reads.append(path)
        return original_read_rows(path)

    monkeypatch.setattr(GenerationChunkStore, "_read_rows", staticmethod(counted_read_rows))

    def counted_file_signature(path: Path) -> tuple[int, int, int, int]:
        signatures.append(path)
        return original_file_signature(path)

    monkeypatch.setattr(GenerationChunkStore, "_file_signature", staticmethod(counted_file_signature))
    sample_ids = [f"sample-{index}" for index in range(12)]
    store = GenerationChunkStore(tmp_path, "dev", sample_ids, fixture_mode=True)
    for sample_id in sample_ids:
        store.commit([{"sample_id": sample_id, "generated_reasoning": sample_id}])
        assert store.committed_sample_ids() >= {sample_id}

    # A normal writer validates no historical JSONL chunk again during a
    # commit or state query.  Completion performs the one permitted final
    # full validation of all persisted chunks.
    assert reads == []
    assert [row["sample_id"] for row in store.committed_rows()] == sample_ids
    store.mark_complete()
    assert len(reads) == len(sample_ids)
    assert len(signatures) <= 2 * len(sample_ids) + 3


def test_generation_chunk_store_reconciles_only_external_append_and_detects_mutation(tmp_path: Path) -> None:
    first = GenerationChunkStore(tmp_path, "dev", ["a", "b"], fixture_mode=True)
    second = GenerationChunkStore(tmp_path, "dev", ["a", "b"], fixture_mode=True)
    first.commit([{"sample_id": "a", "generated_reasoning": "a"}])
    second.commit([{"sample_id": "b", "generated_reasoning": "b"}])
    assert second.committed_sample_ids() == {"a", "b"}

    chunk_path = tmp_path / "reasoning/dev_chunks/chunk_000000.jsonl"
    chunk_path.write_text('{"generated_reasoning":"tampered","sample_id":"a"}\n', encoding="utf-8")
    with pytest.raises(GenerationPersistenceError, match="missing or corrupt"):
        second.mark_complete()


def test_generation_chunk_store_reconciliation_is_transactional_on_corrupt_append(tmp_path: Path) -> None:
    store = GenerationChunkStore(tmp_path, "dev", ["a", "b"], fixture_mode=True)
    store.commit([{"sample_id": "a", "generated_reasoning": "a"}])

    external = GenerationChunkStore(tmp_path, "dev", ["a", "b"], fixture_mode=True)
    external.commit([{"sample_id": "b", "generated_reasoning": "b"}])
    chunk_path = tmp_path / "reasoning/dev_chunks/chunk_000001.jsonl"
    chunk_path.write_text('{"generated_reasoning":"corrupt","sample_id":"b"}\n', encoding="utf-8")

    before_failure = deepcopy(store.__dict__)
    with pytest.raises(GenerationPersistenceError, match="missing or corrupt"):
        store.mark_complete()
    assert store.__dict__ == before_failure

    with pytest.raises(GenerationPersistenceError, match="missing or corrupt"):
        store.mark_complete()
    assert store.__dict__ == before_failure

    chunk_path.write_text('{"generated_reasoning":"b","sample_id":"b"}\n', encoding="utf-8")
    store.mark_complete()

    assert store.committed_sample_ids() == {"a", "b"}
    assert store.next_index() == 2
    assert json.loads(store.manifest_path.read_text(encoding="utf-8"))["complete"] is True


def test_generation_chunk_store_rejects_out_of_order_and_noncontiguous_appends(tmp_path: Path) -> None:
    store = GenerationChunkStore(tmp_path, "dev", ["a", "b"], fixture_mode=True)
    with pytest.raises(GenerationPersistenceError, match="exact sample record ordering"):
        store.commit([{"sample_id": "b", "generated_reasoning": "b"}])

    manifest_path = tmp_path / "reasoning/dev_chunks_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["chunks"] = [{
        "index": 2,
        "path": "reasoning/dev_chunks/chunk_000002.jsonl",
        "sample_ids": [],
        "sha256": "0" * 64,
        "row_count": 0,
    }]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(GenerationPersistenceError, match="contiguous"):
        GenerationChunkStore(tmp_path, "dev", ["a", "b"], fixture_mode=True)
