from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from vipragsent.constants import PRAGMATIC_LABELS
from vipragsent.evaluation.reasoning_judge import ReasoningJudge
from vipragsent.hashing import sha256_file
from vipragsent.models.factory import build_production_model
from vipragsent.orchestration import stage_registry
from vipragsent.orchestration.contracts import RunContext, RunEntry
from vipragsent.orchestration.executors.generation import (
    GenerationCheckpointError,
    ReasoningGenerationExecutor,
)


class _FakeCausalModel(nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_size: int = 8) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, vocab_size=vocab_size, use_cache=True)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.generate_calls: list[dict[str, object]] = []
        self.forward_batch_sizes: list[int] = []
        self.forward_attention_masks: list[torch.Tensor | None] = []

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None, attention_mask: torch.Tensor | None = None, **_: object) -> dict[str, torch.Tensor]:
        self.forward_batch_sizes.append(int(input_ids.size(0)))
        self.forward_attention_masks.append(attention_mask.detach().clone() if attention_mask is not None else None)
        logits = self.lm_head(self.embedding(input_ids))
        result: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            result["loss"] = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100
            )
        return result

    def generate(self, **kwargs: object) -> torch.Tensor:
        self.generate_calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        assert isinstance(input_ids, torch.Tensor)
        return torch.cat((input_ids, torch.ones((input_ids.size(0), 2), dtype=torch.long)), dim=1)


class _FakeFeatureModel(_FakeCausalModel):
    pass


class _FakeAutoModel:
    calls: list[dict[str, object]] = []
    model = _FakeFeatureModel()

    @staticmethod
    def from_pretrained(*_: object, **kwargs: object) -> _FakeFeatureModel:
        _FakeAutoModel.calls.append(kwargs)
        return _FakeAutoModel.model


class _FakeAutoModelForCausalLM:
    calls: list[dict[str, object]] = []
    model = _FakeCausalModel()

    @staticmethod
    def from_pretrained(*_: object, **kwargs: object) -> _FakeCausalModel:
        _FakeAutoModelForCausalLM.calls.append(kwargs)
        return _FakeAutoModelForCausalLM.model


class _FakeBitsAndBytesConfig:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class _FakeLoraConfig:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        _FakeLoraConfig.last = kwargs


def _fake_prepare(model: nn.Module) -> nn.Module:
    return model


def _fake_get_peft_model(model: nn.Module, _: object) -> nn.Module:
    if not hasattr(model, "lora_adapter"):
        model.register_parameter("lora_adapter", nn.Parameter(torch.ones(2, 2)))
    return model


def _modules() -> tuple[SimpleNamespace, SimpleNamespace]:
    transformers = SimpleNamespace(
        BitsAndBytesConfig=_FakeBitsAndBytesConfig,
        AutoModel=_FakeAutoModel,
        AutoModelForCausalLM=_FakeAutoModelForCausalLM,
    )
    peft = SimpleNamespace(
        LoraConfig=_FakeLoraConfig,
        get_peft_model=_fake_get_peft_model,
        prepare_model_for_kbit_training=_fake_prepare,
    )
    return transformers, peft


def _registry(tmp_path: Path) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(
        "models:\n"
        "  vistral_7b:\n"
        "    repo_id: fixture/vistral\n"
        "    revision: locked\n"
        "    tokenizer_revision: locked\n"
        "    architecture: decoder\n"
        "    quantization: nf4\n"
        "    trust_remote_code: false\n",
        encoding="utf-8",
    )
    return path


def test_cot_factory_uses_causal_lm_loader(tmp_path: Path) -> None:
    transformers, peft = _modules()
    _FakeAutoModel.calls.clear()
    _FakeAutoModelForCausalLM.calls.clear()
    model, _ = build_production_model(
        "vistral_7b",
        "cot_only_vistral",
        registry_path=_registry(tmp_path),
        local_snapshot=tmp_path,
        selected_device="cpu",
        transformers_module=transformers,
        peft_module=peft,
    )
    assert _FakeAutoModelForCausalLM.calls
    assert not _FakeAutoModel.calls
    assert callable(getattr(model, "generate", None))


def test_cot_peft_task_type_is_causal_lm(tmp_path: Path) -> None:
    transformers, peft = _modules()
    build_production_model(
        "vistral_7b",
        "cot_only_vistral",
        registry_path=_registry(tmp_path),
        local_snapshot=tmp_path,
        selected_device="cpu",
        transformers_module=transformers,
        peft_module=peft,
    )
    assert _FakeLoraConfig.last["task_type"] == "CAUSAL_LM"


def test_cot_model_has_generate(tmp_path: Path) -> None:
    transformers, peft = _modules()
    model, _ = build_production_model(
        "vistral_7b",
        "cot_only_vistral",
        registry_path=_registry(tmp_path),
        local_snapshot=tmp_path,
        selected_device="cpu",
        transformers_module=transformers,
        peft_module=peft,
    )
    inputs = torch.tensor([[1, 2, 3]])
    generated = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs), do_sample=False, num_beams=1, max_new_tokens=160, repetition_penalty=1.0)
    assert generated.shape[1] == 5


def test_cot_forward_returns_token_logits(tmp_path: Path) -> None:
    transformers, peft = _modules()
    model, _ = build_production_model(
        "vistral_7b",
        "cot_only_vistral",
        registry_path=_registry(tmp_path),
        local_snapshot=tmp_path,
        selected_device="cpu",
        transformers_module=transformers,
        peft_module=peft,
    )
    inputs = torch.tensor([[1, 2, 3]])
    output = model(input_ids=inputs)
    assert output["logits"].shape == (1, 3, 32)


def test_cot_forward_with_labels_returns_finite_loss(tmp_path: Path) -> None:
    transformers, peft = _modules()
    model, _ = build_production_model(
        "vistral_7b",
        "cot_only_vistral",
        registry_path=_registry(tmp_path),
        local_snapshot=tmp_path,
        selected_device="cpu",
        transformers_module=transformers,
        peft_module=peft,
    )
    inputs = torch.tensor([[1, 2, 3]])
    output = model(input_ids=inputs, labels=inputs)
    assert torch.isfinite(output["loss"])


def test_cot_lora_trainable_base_frozen(tmp_path: Path) -> None:
    transformers, peft = _modules()
    model, _ = build_production_model(
        "vistral_7b",
        "cot_only_vistral",
        registry_path=_registry(tmp_path),
        local_snapshot=tmp_path,
        selected_device="cpu",
        transformers_module=transformers,
        peft_module=peft,
    )
    assert model.backbone.lora_adapter.requires_grad
    assert not model.backbone.embedding.weight.requires_grad


def test_classifier_7b_factory_path_unchanged(tmp_path: Path) -> None:
    transformers, peft = _modules()
    _FakeAutoModel.calls.clear()
    _FakeAutoModelForCausalLM.calls.clear()
    model, _ = build_production_model(
        "vistral_7b",
        "vipragsent_full_vistral",
        registry_path=_registry(tmp_path),
        local_snapshot=tmp_path,
        selected_device="cpu",
        transformers_module=transformers,
        peft_module=peft,
    )
    assert _FakeAutoModel.calls
    assert not _FakeAutoModelForCausalLM.calls
    assert _FakeLoraConfig.last["task_type"] == "FEATURE_EXTRACTION"
    assert hasattr(model, "heads")


class _TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def decode(self, ids: object, **_: object) -> str:
        return "reasoning"


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
    *,
    data_hash: str = "NOT_PROVIDED",
    dataset_identity: str | None = None,
    model_artifact_identity: str | None = None,
    tokenizer_artifact_identity: str | None = None,
    production_provenance_required: bool = False,
    fixture_mode: bool = True,
) -> ReasoningGenerationExecutor:
    root = _protocol_root(tmp_path)
    judge = ReasoningJudge(root, transport=lambda **_: {"labels": {label: 0 for label in ("sarcasm", "irony", "idiom_figurative", "code_switching", "mocking", "implicit_sentiment")}}, cache_root=root / "judge-cache", sleep_fn=lambda _: None)
    return ReasoningGenerationExecutor(
        root,
        model=_FakeCausalModel(),
        tokenizer=_TinyTokenizer(),
        judge=judge,
        run_root=root / "run",
        seed=20260521,
        data_hash=data_hash,
        dataset_identity=dataset_identity,
        model_artifact_identity=model_artifact_identity,
        tokenizer_artifact_identity=tokenizer_artifact_identity,
        production_provenance_required=production_provenance_required,
        fixture_mode=fixture_mode,
    )


def _rows() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    train = [{"input_ids": torch.tensor([[1, 2]]), "target_ids": torch.tensor([[3, 4]])}]
    gold = {label: 0 for label in ("sarcasm", "irony", "idiom_figurative", "code_switching", "mocking", "implicit_sentiment")}
    dev = [{"sample_id": "dev-1", "input_ids": torch.tensor([[1, 2]]), "gold": gold}]
    test = [{"sample_id": "test-1", "input_ids": torch.tensor([[1, 2]]), "gold": gold}]
    return train, dev, test


def test_generation_training_obeys_physical_batch_and_accumulation_contract(tmp_path: Path) -> None:
    root = _protocol_root(tmp_path)
    judge = ReasoningJudge(
        root,
        transport=lambda **_: {"labels": {label: 0 for label in ("sarcasm", "irony", "idiom_figurative", "code_switching", "mocking", "implicit_sentiment")}},
        cache_root=root / "judge-cache",
        sleep_fn=lambda _: None,
    )
    model = _FakeCausalModel()
    executor = ReasoningGenerationExecutor(
        root,
        model=model,
        tokenizer=_TinyTokenizer(),
        judge=judge,
        run_root=root / "run",
        fixture_mode=True,
        physical_batch_size=2,
        gradient_accumulation_steps=2,
    )
    records = []
    for index in range(5):
        input_length = 2 + index % 2
        target_length = 2 + (index + 1) % 2
        records.append(
            {
                "input_ids": torch.arange(1, input_length + 1).unsqueeze(0),
                "attention_mask": torch.ones((1, input_length), dtype=torch.long),
                "target_ids": torch.arange(8, 8 + target_length).unsqueeze(0),
                "target_mask": torch.ones((1, target_length), dtype=torch.bool),
            }
        )

    class _Scheduler:
        def __init__(self) -> None:
            self.steps = 0

        def step(self) -> None:
            self.steps += 1

    scheduler = _Scheduler()
    history = executor.train_generation(
        records,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        scheduler=scheduler,
        epochs=1,
    )

    assert model.forward_batch_sizes == [2, 2, 1]
    assert scheduler.steps == 2
    assert history[0]["micro_batches"] == 3
    assert history[0]["optimizer_steps"] == 2
    assert history[0]["physical_batch_size"] == 2
    assert history[0]["gradient_accumulation_steps"] == 2
    assert all(mask is not None for mask in model.forward_attention_masks)


def test_generation_checkpoint_load_changes_weights(tmp_path: Path) -> None:
    executor = _executor(tmp_path)
    path = executor.run_root / "checkpoints/distinct/model.pt"
    original = {key: value.detach().clone() for key, value in executor.model.state_dict().items()}
    for parameter in executor.model.parameters():
        parameter.data.add_(1.0)
    digest = executor.write_checkpoint("checkpoints/distinct/model.pt")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["schema_version"] == 2
    assert "model_state_dict" in payload
    assert "model" not in payload
    assert (path.with_suffix(path.suffix + ".manifest.json")).exists()
    for parameter in executor.model.parameters():
        parameter.data.zero_()
    report = executor.load_checkpoint(path, expected_sha256=digest)
    assert report["status"] == "PASS"
    assert any(not torch.equal(original[key], value) for key, value in executor.model.state_dict().items())


def test_generation_checkpoint_hash_mismatch_blocks(tmp_path: Path) -> None:
    executor = _executor(tmp_path)
    digest = executor.write_checkpoint("checkpoints/best/model.pt")
    with pytest.raises(GenerationCheckpointError, match="hash mismatch"):
        executor.load_checkpoint(executor.run_root / "checkpoints/best/model.pt", expected_sha256="0" * len(digest))


def test_generation_checkpoint_dataset_identity_mismatch_blocks(tmp_path: Path) -> None:
    source = _executor(tmp_path, data_hash="DATA_A", dataset_identity="DATASET")
    path = source.run_root / "checkpoints/latest/model.pt"
    source.write_checkpoint("checkpoints/latest/model.pt")
    sidecar = json.loads(path.with_suffix(path.suffix + ".manifest.json").read_text(encoding="utf-8"))
    assert sidecar["provenance"]["data_hash"] == "DATA_A"
    assert sidecar["provenance"]["dataset"] == {"identity": "DATASET", "hash": "DATA_A"}
    changed_dataset = _executor(tmp_path, data_hash="DATA_B", dataset_identity="DATASET")
    with pytest.raises(GenerationCheckpointError, match="provenance identity mismatch"):
        changed_dataset.load_checkpoint(path)


@pytest.mark.parametrize("data_hash", [None, "", "NOT_PROVIDED", "fixture-data"])
def test_production_generation_blocks_missing_or_placeholder_context_data_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    data_hash: str | None,
) -> None:
    entry = RunEntry.from_mapping(
        {
            "run_id": "cot-provenance",
            "system_id": "cot_only_vistral",
            "execution_kind": "generation",
            "backbone": "vistral_7b",
            "research_question": "Q1",
        }
    )
    context = RunContext(
        tmp_path,
        entry,
        metadata={} if data_hash is None else {"data_hash": data_hash},
    )
    reached_model_resolution = False

    def unexpected_model_resolution(*_: object, **__: object) -> object:
        nonlocal reached_model_resolution
        reached_model_resolution = True
        raise AssertionError("model resolution must not run before provenance validation")

    monkeypatch.setattr(stage_registry, "_execution_spec", unexpected_model_resolution)
    outcome = stage_registry._production_generation_stage(context, entry, "train_generation")
    assert outcome.status == "BLOCKED"
    assert "data_hash" in outcome.error
    assert reached_model_resolution is False
    assert not (context.run_root / "checkpoints").exists()


@pytest.mark.parametrize("data_hash", ["", "NOT_PROVIDED", "fixture-data"])
def test_production_generation_requires_real_dataset_hash_before_checkpoint(
    tmp_path: Path,
    data_hash: str,
) -> None:
    with pytest.raises(GenerationCheckpointError, match="real dataset hash"):
        _executor(
            tmp_path,
            data_hash=data_hash,
            production_provenance_required=True,
            fixture_mode=False,
        )
    assert not (tmp_path / "run/checkpoints").exists()


def test_generation_checkpoint_artifact_identity_mismatch_blocks(tmp_path: Path) -> None:
    source = _executor(
        tmp_path,
        data_hash="DATA_A",
        model_artifact_identity="model@A",
        tokenizer_artifact_identity="tokenizer@A",
    )
    path = source.run_root / "checkpoints/latest/model.pt"
    source.write_checkpoint("checkpoints/latest/model.pt")
    changed_artifacts = _executor(
        tmp_path,
        data_hash="DATA_A",
        model_artifact_identity="model@B",
        tokenizer_artifact_identity="tokenizer@A",
    )
    with pytest.raises(GenerationCheckpointError, match="provenance identity mismatch"):
        changed_artifacts.load_checkpoint(path)

    tokenizer_source = _executor(
        tmp_path / "tokenizer",
        data_hash="DATA_A",
        model_artifact_identity="model@A",
        tokenizer_artifact_identity="tokenizer@A",
    )
    tokenizer_path = tokenizer_source.run_root / "checkpoints/latest/model.pt"
    tokenizer_source.write_checkpoint("checkpoints/latest/model.pt")
    changed_tokenizer = _executor(
        tmp_path / "tokenizer",
        data_hash="DATA_A",
        model_artifact_identity="model@A",
        tokenizer_artifact_identity="tokenizer@B",
    )
    with pytest.raises(GenerationCheckpointError, match="provenance identity mismatch"):
        changed_tokenizer.load_checkpoint(tokenizer_path)


def test_generation_checkpoint_sidecar_failure_is_fail_closed(tmp_path: Path) -> None:
    executor = _executor(tmp_path)
    path = executor.run_root / "checkpoints/latest/model.pt"
    executor.write_checkpoint("checkpoints/latest/model.pt")
    path.with_suffix(path.suffix + ".manifest.json").unlink()
    with pytest.raises(GenerationCheckpointError, match="sidecar manifest"):
        executor.load_checkpoint(path)


def test_generation_checkpoint_restores_all_resume_state_and_rng_streams(tmp_path: Path) -> None:
    random.seed(91)
    np.random.seed(91)
    torch.manual_seed(91)
    executor = _executor(tmp_path)
    optimizer = torch.optim.AdamW(executor.model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    records = [
        {"input_ids": torch.tensor([[1, 2]]), "target_ids": torch.tensor([[3, 4]])},
        {"input_ids": torch.tensor([[2, 3]]), "target_ids": torch.tensor([[4, 5]])},
    ]
    executor.train_generation(records, optimizer=optimizer, scheduler=scheduler)
    order = ["train-0", "train-1"]
    path = executor.run_root / "checkpoints/latest/model.pt"
    executor.write_checkpoint(
        "checkpoints/latest/model.pt",
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        selection_metric=0.75,
        data_order=order,
    )
    expected_python = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = torch.rand(3)
    for parameter in executor.model.parameters():
        parameter.data.zero_()
    random.random()
    np.random.random()
    torch.rand(3)
    restored_optimizer = torch.optim.AdamW(executor.model.parameters(), lr=0.01)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda _: 1.0)
    report = executor.load_checkpoint(
        path,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        expected_data_order=order,
    )
    assert report["run_state"]["epoch"] == 1
    assert report["run_state"]["data_order"] == order
    assert report["optimizer_state_present"] is True
    assert report["optimizer_restored"] is True
    assert report["scheduler_state_present"] is True
    assert report["scheduler_restored"] is True
    assert report["rng_state_present"] is True
    assert report["rng_restore"]["restored"] is True
    assert restored_scheduler.last_epoch == scheduler.last_epoch
    assert restored_optimizer.state
    assert random.random() == expected_python
    assert float(np.random.random()) == expected_numpy
    assert torch.equal(torch.rand(3), expected_torch)


def test_generation_test_stage_requires_freeze(tmp_path: Path) -> None:
    executor = _executor(tmp_path)
    executor.write_checkpoint("checkpoints/best/model.pt")
    with pytest.raises(GenerationCheckpointError, match="freeze"):
        executor.load_frozen_checkpoint()


def test_generation_test_stage_loads_frozen_best_checkpoint(tmp_path: Path) -> None:
    executor = _executor(tmp_path)
    digest = executor.write_checkpoint("checkpoints/best/model.pt")
    selection = executor.run_root / "selection"
    selection.mkdir(parents=True, exist_ok=True)
    (selection / "freeze_manifest.json").write_text(json.dumps({"frozen": True, "checkpoint_path": "checkpoints/best/model.pt", "checkpoint_sha256": digest}), encoding="utf-8")
    report = executor.load_frozen_checkpoint()
    assert report["status"] == "PASS"


def test_production_generation_profile_defaults_to_safe_batch_one(tmp_path: Path) -> None:
    entry = RunEntry.from_mapping({"run_id": "profile", "system_id": "cot_only_vistral", "execution_kind": "generation", "backbone": "vistral_7b", "research_question": "Q1"})
    context = RunContext(tmp_path, entry)
    profile = stage_registry._production_generation_profile(context)
    assert profile["status"] == "PASS"
    assert profile["selected_batch_size"] == 1
    assert profile["source"] == "default-safe-generation-batch-one"


def test_selected_dev_artifact_marker_is_validated_for_reuse(tmp_path: Path) -> None:
    reasoning = tmp_path / "reasoning/dev_reasoning.jsonl"
    predictions = tmp_path / "predictions/dev_predictions.jsonl"
    judge = tmp_path / "judge/dev_judge_responses.jsonl"
    metrics = tmp_path / "metrics/dev_reasoning_metrics.json"
    chunks_manifest = tmp_path / "reasoning/dev_chunks_manifest.json"
    checkpoint = tmp_path / "checkpoints/best/model.pt"
    for path in (reasoning, predictions, judge, metrics, chunks_manifest, checkpoint):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    marker = {
        "status": "PASS",
        "epoch": 2,
        "checkpoint_sha256": sha256_file(checkpoint),
        "reasoning_sha256": sha256_file(reasoning),
        "predictions_sha256": sha256_file(predictions),
        "judge_sha256": sha256_file(judge),
        "metrics_sha256": sha256_file(metrics),
        "chunks_manifest_sha256": sha256_file(chunks_manifest),
    }
    (tmp_path / "selection").mkdir(parents=True)
    (tmp_path / "selection/dev_artifacts.json").write_text(json.dumps(marker), encoding="utf-8")
    (tmp_path / "selection/best_checkpoint.json").write_text(json.dumps({"best_epoch": 2, "checkpoint_sha256": marker["checkpoint_sha256"], "dev_artifacts": {"epoch": 2}}), encoding="utf-8")
    assert stage_registry._selected_dev_artifacts_reusable(tmp_path)
    predictions.write_text("changed\n", encoding="utf-8")
    assert not stage_registry._selected_dev_artifacts_reusable(tmp_path)


def test_generation_stage_loads_requested_epoch_checkpoint(tmp_path: Path) -> None:
    executor = _executor(tmp_path)
    for value in (1.0, 2.0):
        for parameter in executor.model.parameters():
            parameter.data.fill_(value)
        executor.write_checkpoint(f"checkpoints/epoch_{int(value)}/model.pt")
    report = executor.load_checkpoint(executor.run_root / "checkpoints/epoch_1/model.pt")
    assert report["status"] == "PASS"
    assert float(next(executor.model.parameters()).detach().flatten()[0]) == pytest.approx(1.0)


def _write_resume_selection_fixture(tmp_path: Path) -> str:
    executor = _executor(tmp_path)
    optimizer = torch.optim.SGD(executor.model.parameters(), lr=0.01)
    for parameter in executor.model.parameters():
        parameter.data.fill_(1.0)
    best_hash = executor.write_checkpoint(
        "checkpoints/best/model.pt",
        optimizer=optimizer,
        epoch=1,
        selection_metric=0.9,
        data_order=["train-0"],
    )
    for parameter in executor.model.parameters():
        parameter.data.fill_(2.0)
    resume_hash = executor.write_checkpoint(
        "checkpoints/epoch_2/model.pt",
        optimizer=optimizer,
        epoch=2,
        selection_metric=0.1,
        data_order=["train-0"],
    )
    selection_path = executor.run_root / "selection/best_checkpoint.json"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "best_epoch": 1,
                "selection_metric": "dev_metric",
                "value": 0.9,
                "checkpoint_path": "checkpoints/best/model.pt",
                "checkpoint_sha256": best_hash,
            }
        ),
        encoding="utf-8",
    )
    return resume_hash


def test_generation_resume_keeps_persisted_best_separate_from_arbitrary_resume_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    resume_hash = _write_resume_selection_fixture(tmp_path)
    executor = _executor(tmp_path)
    model = executor.model
    train = [{"sample_id": "train-0", "input_ids": torch.tensor([[1, 2]]), "target_ids": torch.tensor([[3, 4]])}]
    gold = {label: 0 for label in PRAGMATIC_LABELS}
    dev = [{"sample_id": "dev-1", "input_ids": torch.tensor([[1, 2]]), "gold": gold}]
    test = [{"sample_id": "test-1", "input_ids": torch.tensor([[1, 2]]), "gold": gold}]

    def train_generation(*_: object, epoch_start: int = 1, **__: object) -> list[dict[str, float]]:
        for parameter in model.parameters():
            parameter.data.fill_(3.0)
        return [{"epoch": float(epoch_start), "train_loss": 0.0}]

    monkeypatch.setattr(executor, "train_generation", train_generation)
    monkeypatch.setattr(
        executor,
        "generate_reasoning_split",
        lambda split, records, **__: [{"sample_id": str(records[0]["sample_id"]), "generation_status": "PASS", "generated_reasoning": split}],
    )
    monkeypatch.setattr(executor, "judge_reasoning_split", lambda _split, rows, _gold, **__: (list(rows), []))
    monkeypatch.setattr(executor, "compute_split_metrics", lambda _split, _rows, **__: {"primary_macro_f1": 0.5})

    result = executor.run_cot(
        train_records=train,
        dev_records=dev,
        test_records=test,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        epochs=3,
        resume_from="checkpoints/epoch_2/model.pt",
    )

    selection = json.loads((executor.run_root / "selection/best_checkpoint.json").read_text(encoding="utf-8"))
    assert result["best_epoch"] == 1
    assert result["best_dev_metric"] == pytest.approx(0.9)
    assert selection["best_epoch"] == 1
    assert selection["checkpoint_sha256"] != resume_hash
    assert float(next(model.parameters()).detach().flatten()[0]) == pytest.approx(1.0)


def test_generation_resume_rejects_a_non_advancing_target_epoch(tmp_path: Path) -> None:
    _write_resume_selection_fixture(tmp_path)
    executor = _executor(tmp_path)
    rows = [{"sample_id": "train-0", "input_ids": torch.tensor([[1, 2]]), "target_ids": torch.tensor([[3, 4]])}]
    with pytest.raises(GenerationCheckpointError, match="must advance beyond checkpoint epoch"):
        executor.run_cot(
            train_records=rows,
            dev_records=[{"sample_id": "dev-1", "input_ids": torch.tensor([[1, 2]]), "gold": {}}],
            test_records=[{"sample_id": "test-1", "input_ids": torch.tensor([[1, 2]]), "gold": {}}],
            optimizer=torch.optim.SGD(executor.model.parameters(), lr=0.01),
            epochs=2,
            resume_from="checkpoints/epoch_2/model.pt",
        )


def test_generation_resume_appends_existing_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_resume_selection_fixture(tmp_path)
    executor = _executor(tmp_path)
    history_path = executor.run_root / "training/history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps([{"epoch": 1, "train_loss": 0.9}, {"epoch": 2, "train_loss": 0.8}]), encoding="utf-8")
    model = executor.model
    rows = [{"sample_id": "train-0", "input_ids": torch.tensor([[1, 2]]), "target_ids": torch.tensor([[3, 4]])}]
    gold = {label: 0 for label in PRAGMATIC_LABELS}
    dev = [{"sample_id": "dev-1", "input_ids": torch.tensor([[1, 2]]), "gold": gold}]
    test = [{"sample_id": "test-1", "input_ids": torch.tensor([[1, 2]]), "gold": gold}]

    def train_generation(*_: object, epoch_start: int = 1, **__: object) -> list[dict[str, float]]:
        for parameter in model.parameters():
            parameter.data.fill_(3.0)
        return [{"epoch": float(epoch_start), "train_loss": 0.0}]

    monkeypatch.setattr(executor, "train_generation", train_generation)
    monkeypatch.setattr(executor, "generate_reasoning_split", lambda split, records, **__: [{"sample_id": str(records[0]["sample_id"]), "generation_status": "PASS", "generated_reasoning": split}])
    monkeypatch.setattr(executor, "judge_reasoning_split", lambda _split, rows, _gold, **__: (list(rows), []))
    monkeypatch.setattr(executor, "compute_split_metrics", lambda _split, _rows, **__: {"primary_macro_f1": 0.5})

    executor.run_cot(
        train_records=rows,
        dev_records=dev,
        test_records=test,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        epochs=3,
        resume_from="checkpoints/epoch_2/model.pt",
    )
    persisted = json.loads(history_path.read_text(encoding="utf-8"))
    assert [int(row["epoch"]) for row in persisted] == [1, 2, 3]


def test_generation_latest_checkpoint_tracks_last_completed_epoch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    executor = _executor(tmp_path)
    model = executor.model
    train = [{"sample_id": "train-0", "input_ids": torch.tensor([[1, 2]]), "target_ids": torch.tensor([[3, 4]])}]
    gold = {label: 0 for label in PRAGMATIC_LABELS}
    dev = [{"sample_id": "dev-1", "input_ids": torch.tensor([[1, 2]]), "gold": gold}]
    test = [{"sample_id": "test-1", "input_ids": torch.tensor([[1, 2]]), "gold": gold}]
    metrics = iter((0.9, 0.5, 0.4))

    def train_generation(*_: object, epoch_start: int = 1, **__: object) -> list[dict[str, float]]:
        for parameter in model.parameters():
            parameter.data.fill_(float(epoch_start))
        return [{"epoch": float(epoch_start), "train_loss": 0.0}]

    monkeypatch.setattr(executor, "train_generation", train_generation)
    monkeypatch.setattr(
        executor,
        "generate_reasoning_split",
        lambda split, records, **__: [{"sample_id": str(records[0]["sample_id"]), "generation_status": "PASS", "generated_reasoning": split}],
    )
    monkeypatch.setattr(executor, "judge_reasoning_split", lambda _split, rows, _gold, **__: (list(rows), []))
    monkeypatch.setattr(executor, "compute_split_metrics", lambda _split, _rows, **__: {"primary_macro_f1": next(metrics) if _split == "dev" else 0.0})

    result = executor.run_cot(
        train_records=train,
        dev_records=dev,
        test_records=test,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        epochs=3,
    )

    assert result["best_epoch"] == 1
    latest = executor.load_checkpoint(
        "checkpoints/latest/model.pt",
        expected_data_order=["train-0"],
        restore_training_state=False,
    )
    assert latest["run_state"]["epoch"] == 3
    assert result["latest_checkpoint_sha256"] != result["checkpoint_sha256"]


def test_generation_resume_requires_intact_persisted_selection_metadata(tmp_path: Path) -> None:
    resume_hash = _write_resume_selection_fixture(tmp_path)
    executor = _executor(tmp_path)
    selection_path = executor.run_root / "selection/best_checkpoint.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["value"] = 0.1
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    train = [{"sample_id": "train-0", "input_ids": torch.tensor([[1, 2]]), "target_ids": torch.tensor([[3, 4]])}]
    rows = [{"sample_id": "dev-1", "gold": {}}]
    with pytest.raises(GenerationCheckpointError, match="metric disagrees"):
        executor.run_cot(
            train_records=train,
            dev_records=rows,
            test_records=rows,
            optimizer=torch.optim.SGD(executor.model.parameters(), lr=0.01),
            epochs=3,
            resume_from="checkpoints/epoch_2/model.pt",
        )

    selection_path.unlink()
    with pytest.raises(GenerationCheckpointError, match="manifest is missing"):
        executor.run_cot(
            train_records=train,
            dev_records=rows,
            test_records=rows,
            optimizer=torch.optim.SGD(executor.model.parameters(), lr=0.01),
            epochs=3,
            resume_from="checkpoints/epoch_2/model.pt",
        )
    assert resume_hash
