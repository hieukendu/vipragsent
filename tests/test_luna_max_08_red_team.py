from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import yaml
from torch import nn

from vipragsent.azure.client import (
    AzureCache,
    AzureResponsesClient,
    AzureSafetyCeilings,
    AzureSettings,
)
from vipragsent.azure.schemas import strict_label_schema
from vipragsent.constants import EMOTION_LABELS, POLARITY_LABELS, PRAGMATIC_LABELS, TRAINING_SEEDS
from vipragsent.data.loaders import DatasetExample
from vipragsent.evaluation.confidence_intervals import evaluate_q1a_confidence_intervals
from vipragsent.evaluation.reasoning_judge import ReasoningJudge
from vipragsent.models.factory import build_production_model
from vipragsent.models.qlora import build_qlora_backbone
from vipragsent.orchestration.aggregation import _table2
from vipragsent.orchestration.executors.component_production import ProductionComponentRunner
from vipragsent.orchestration.q1b_predictor import DiskBackedQ1BPredictor
from vipragsent.orchestration.sequential import load_inventory
from vipragsent.training.checkpoints import build_checkpoint_payload, save_checkpoint
from vipragsent.training.engine import CheckpointManager, RunState

ROOT = Path(__file__).resolve().parents[1]
ALL_LABELS = {
    **{label: 0 for label in PRAGMATIC_LABELS},
    "polarity": "neutral",
    "emotion": "other",
}


def _example(sample_id: str, *, split: str = "train", index: int = 0) -> DatasetExample:
    labels = dict(ALL_LABELS)
    labels["sarcasm"] = index % 2
    labels["polarity"] = ("negative", "neutral", "positive")[index % 3]
    labels["emotion"] = ("anger", "enjoyment", "other")[index % 3]
    return DatasetExample(sample_id, f"text {sample_id}", labels, split)


def _locked_component_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, component: str = "sarcasm") -> tuple[ProductionComponentRunner, nn.Module, list[DatasetExample]]:
    from vipragsent.orchestration.executors import component_production as module

    class CountingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.0))
            self.seen = 0
            self.train_seen = 0

        def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, Any]:
            del attention_mask
            self.seen += int(input_ids.shape[0])
            if self.training:
                self.train_seen += int(input_ids.shape[0])
            width = 1 if component in PRAGMATIC_LABELS else 3 if component == "polarity" else 7
            logits = self.weight.expand(input_ids.shape[0], width)
            return {"logits": {component: logits}}

    model = CountingModel()
    train = []
    for i in range(14):
        row = _example(f"train-{i}", index=i)
        for label in PRAGMATIC_LABELS:
            row.labels[label] = i % 2
        row.labels["polarity"] = POLARITY_LABELS[i % len(POLARITY_LABELS)]
        row.labels["emotion"] = EMOTION_LABELS[i % len(EMOTION_LABELS)]
        train.append(row)
    dev = [_example("dev-0", split="dev", index=0), _example("dev-1", split="dev", index=1)]
    test = [_example("test-0", split="test", index=0), _example("test-1", split="test", index=1)]
    bundle = SimpleNamespace(train=train, dev=dev, test=test, fingerprint="fixture-bundle")
    entry = SimpleNamespace(system_id="phobert_pragmatic_single_task", backbone="phobert_base", seed=20260521, raw={"selection_metric": "macro_prag_f1_dev"})
    resolved = SimpleNamespace(
        optimizer="adamw",
        learning_rate=0.1,
        weight_decay=0.01,
        scheduler="linear",
        warmup_ratio=0.1,
        physical_batch_size=2,
        gradient_accumulation_steps=2,
        effective_batch_size=4,
        maximum_epochs=2,
        precision="fp32",
        gradient_clipping=1.0,
        patience=2,
        minimum_delta=0.0001,
        config_hash="fixture-config",
    )
    monkeypatch.setattr(module, "resolve_execution_spec", lambda *_args: SimpleNamespace())
    monkeypatch.setattr(module, "resolve_training_config", lambda *_args, **_kwargs: resolved)
    monkeypatch.setattr(module, "read_family_status", lambda *_args: {"successful_batch": 2})
    monkeypatch.setattr(module, "_encode_text", lambda *_args, **_kwargs: (torch.tensor([[1, 2]]), torch.ones(1, 2, dtype=torch.long)))
    monkeypatch.setattr(module, "build_optimizer", lambda model, **_kwargs: (torch.optim.SGD(model.parameters(), lr=0.1), {"optimizer": "sgd"}))
    monkeypatch.setattr(module, "build_scheduler", lambda optimizer, **_kwargs: (torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0), {"scheduler": "constant"}))
    runner = ProductionComponentRunner(tmp_path, entry=entry, bundle=bundle)
    runner.tokenizer = object()
    return runner, model, train


def test_red_team_component_runner_consumes_all_examples_and_locked_epochs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner, model, train = _locked_component_runner(monkeypatch, tmp_path)
    result = runner("sarcasm", model, tmp_path / "component")
    assert model.train_seen == len(train) * 2, "production runner silently stopped after one training example"
    assert result["optimizer_steps"] > 1, "locked batch/accumulation contract was bypassed"
    assert len(result["history"]) == 2, "locked epoch budget was not represented in history"
    assert all(float(row["train_loss"]) > 0 for row in result["history"])


def test_red_team_component_runner_passes_pragmatic_pos_weight(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from vipragsent.orchestration.executors import component_production as module

    binary_calls: list[Any] = []
    multiclass_calls: list[Any] = []
    original_binary = module.F.binary_cross_entropy_with_logits
    original_multiclass = module.F.cross_entropy

    def binary(*args: Any, **kwargs: Any) -> torch.Tensor:
        binary_calls.append(kwargs.get("pos_weight"))
        return original_binary(*args, **kwargs)

    def multiclass(*args: Any, **kwargs: Any) -> torch.Tensor:
        multiclass_calls.append(kwargs.get("weight"))
        return original_multiclass(*args, **kwargs)

    monkeypatch.setattr(module.F, "binary_cross_entropy_with_logits", binary)
    monkeypatch.setattr(module.F, "cross_entropy", multiclass)
    runner, model, _ = _locked_component_runner(monkeypatch, tmp_path / "binary")
    runner("sarcasm", model, tmp_path / "binary" / "component")
    assert binary_calls and binary_calls[0] is not None, "pragmatic pos_weight was silently omitted"


def test_red_team_component_runner_passes_multiclass_weights(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from vipragsent.orchestration.executors import component_production as module

    multiclass_calls: list[Any] = []
    original_multiclass = module.F.cross_entropy

    def multiclass(*args: Any, **kwargs: Any) -> torch.Tensor:
        multiclass_calls.append(kwargs.get("weight"))
        return original_multiclass(*args, **kwargs)

    monkeypatch.setattr(module.F, "cross_entropy", multiclass)
    runner2, model2, _ = _locked_component_runner(monkeypatch, tmp_path / "multiclass", component="polarity")
    runner2("polarity", model2, tmp_path / "multiclass" / "component")
    assert multiclass_calls and multiclass_calls[0] is not None, "multiclass class weights were silently omitted"


def test_red_team_cot_factory_uses_native_causal_loader_and_keeps_classifier_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []

    class FakeCausal(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=8, vocab_size=32)
            self.weight = nn.Parameter(torch.ones(1))

        def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None, **_: Any) -> Any:
            logits = torch.zeros(*input_ids.shape, 32)
            return SimpleNamespace(logits=logits, loss=torch.tensor(0.5) if labels is not None else None)

        def generate(self, **_: Any) -> torch.Tensor:
            return torch.tensor([[1, 2]])

    class FakeEncoder(FakeCausal):
        pass

    class AutoModel:
        @staticmethod
        def from_pretrained(*_args: Any, **_kwargs: Any) -> nn.Module:
            calls.append("AutoModel")
            return FakeEncoder()

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*_args: Any, **_kwargs: Any) -> nn.Module:
            calls.append("AutoModelForCausalLM")
            return FakeCausal()

    fake_transformers = SimpleNamespace(AutoModel=AutoModel, AutoModelForCausalLM=AutoModelForCausalLM)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "vistral_7b": {"repo_id": "fixture/vistral", "revision": "locked", "tokenizer_revision": "locked", "architecture": "causal", "quantization": "none"},
                }
            }
        ),
        encoding="utf-8",
    )
    cot, _ = build_production_model("vistral_7b", "cot_only_vistral", registry_path=registry, local_snapshot=snapshot, hidden_size=8, vocab_size=32, selected_device="cpu")
    assert "AutoModelForCausalLM" in calls, "CoT factory did not request a causal LM"
    assert hasattr(cot, "generate")
    assert calls.count("AutoModel") == 0, "causal CoT construction fell back to encoder AutoModel"


def test_red_team_cot_peft_uses_causal_task_type_and_classifier_7b_stays_feature_extraction() -> None:
    calls: dict[str, Any] = {}

    class FakeBits:
        def __init__(self, **kwargs: Any) -> None:
            calls["bits"] = kwargs

    class FakeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace()
            self.weight = nn.Parameter(torch.ones(1))

        def gradient_checkpointing_enable(self) -> None:
            pass

        def generate(self, **_: Any) -> torch.Tensor:
            return torch.tensor([[1, 2]])

    class AutoModel:
        @staticmethod
        def from_pretrained(*_args: Any, **_kwargs: Any) -> FakeModel:
            calls["loader"] = "AutoModel"
            return FakeModel()

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*_args: Any, **_kwargs: Any) -> FakeModel:
            calls["loader"] = "AutoModelForCausalLM"
            return FakeModel()

    class LoraConfig:
        def __init__(self, **kwargs: Any) -> None:
            calls["lora"] = kwargs

    def get_peft(model: nn.Module, _: Any) -> nn.Module:
        model.register_parameter("lora_adapter", nn.Parameter(torch.ones(1)))
        return model

    modules = SimpleNamespace(
        BitsAndBytesConfig=FakeBits,
        AutoModel=AutoModel,
        AutoModelForCausalLM=AutoModelForCausalLM,
    )
    peft = SimpleNamespace(LoraConfig=LoraConfig, get_peft_model=get_peft, prepare_model_for_kbit_training=lambda model: model)
    model = build_qlora_backbone("fixture/repo", revision="locked", selected_device="cpu", transformers_module=modules, peft_module=peft, task_type="CAUSAL_LM")
    assert calls["loader"] == "AutoModelForCausalLM", "the QLoRA generation branch did not use a causal loader"
    assert calls["lora"]["task_type"] == "CAUSAL_LM", "CoT PEFT task type silently remained FEATURE_EXTRACTION"
    assert any(parameter.requires_grad for name, parameter in model.named_parameters() if "lora" in name)
    assert all(not parameter.requires_grad for name, parameter in model.named_parameters() if "lora" not in name)


def test_red_team_generation_final_test_requires_frozen_checkpoint_and_reloads_best(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from vipragsent.orchestration.executors import generation as module

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.0))

        def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None, **_: Any) -> dict[str, torch.Tensor]:
            del labels
            return {"logits": torch.zeros(input_ids.size(0), input_ids.size(1), 8) + self.weight}

        def generate(self, **_: Any) -> torch.Tensor:
            return torch.tensor([[1, 2]])

    class FakeJudge:
        diagnostics: dict[str, int] = {}
        judge_protocol_id = "judge"
        prompt_hash = "prompt"
        schema_hash = "schema"

    monkeypatch.setattr(module, "validate_reasoning_protocol_files", lambda _root: {"status": "PASS", "errors": []})
    model = TinyModel()
    executor = module.ReasoningGenerationExecutor(ROOT, model=model, tokenizer=SimpleNamespace(decode=lambda *_args, **_kwargs: "reason"), judge=FakeJudge(), run_root=tmp_path, fixture_mode=True)
    epochs_seen: list[float] = []

    def train_generation(*_args: Any, epoch_start: int = 1, **_kwargs: Any) -> list[dict[str, float]]:
        model.weight.data.fill_(float(epoch_start))
        return [{"epoch": float(epoch_start), "train_loss": 1.0, "optimizer_steps": 1.0}]

    monkeypatch.setattr(executor, "train_generation", train_generation)
    monkeypatch.setattr(executor, "generate_reasoning_split", lambda split, _records: epochs_seen.append(float(model.weight)) or [{"sample_id": split, "generation_status": "PASS", "generated_reasoning": "reason", "truncated": False}])
    monkeypatch.setattr(executor, "judge_reasoning_split", lambda _split, rows, _gold: (list(rows), []))
    monkeypatch.setattr(executor, "compute_split_metrics", lambda split, _rows: {"primary_macro_f1": 1.0 if split == "dev" and float(model.weight) == 1.0 else 0.0})
    executor.write_checkpoint = executor.write_checkpoint  # keep the real state-writing path
    with pytest.raises((RuntimeError, AttributeError)):
        executor.generate_test_reasoning("test", [])
    result = executor.run_cot(train_records=[], dev_records=[{"sample_id": "dev", "gold": {}}], test_records=[{"sample_id": "test", "gold": {}}], optimizer=torch.optim.SGD(model.parameters(), lr=0.1), epochs=2)
    assert result["best_epoch"] == 1
    assert float(model.weight) == 1.0, "test generation used the last in-memory weights instead of frozen best weights"
    assert epochs_seen[-1] == 1.0


def test_red_team_checkpoint_schema_rejects_silent_nonloading(tmp_path: Path) -> None:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    loss = nn.Linear(1, 1)
    state = RunState(epoch=1, best_epoch=1, best_metric=0.5, no_improvement_epochs=0, status="PASS")
    manager = CheckpointManager(tmp_path / "checkpoints", "run")
    saved = manager.save("epoch_1", model, optimizer, scheduler, loss, state)
    payload = torch.load(saved, map_location="cpu", weights_only=False)
    assert payload.get("schema_version") == 2, "production checkpoint writer still emits the legacy schema"
    assert "model_state_dict" in payload and "model" not in payload

    bad = tmp_path / "bad.pt"
    torch.save({"schema_version": 2, "model_state_dict": {"unrelated.weight": torch.ones(1)}}, bad)
    manager.path.mkdir(parents=True, exist_ok=True)
    bad.replace(manager.path / "best.pt")
    with pytest.raises((RuntimeError, ValueError, KeyError)):
        manager.load_best(nn.Linear(2, 2), nn.Linear(1, 1))


def test_red_team_custom_q1b_predictor_moves_inputs_and_rejects_zero_matching_checkpoint(tmp_path: Path) -> None:
    class DeviceCheckingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1, device="meta"))
            self.observed_device: torch.device | None = None

        def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, Any]:
            self.observed_device = input_ids.device
            if input_ids.device != self.weight.device:
                raise AssertionError(f"custom predictor left inputs on {input_ids.device}, model is on {self.weight.device}")
            return {"logits": {"polarity": torch.tensor([[0.0, 1.0, 0.0]])}}

    source = SimpleNamespace(
        checkpoint_path=tmp_path / "checkpoint.pt",
        checkpoint_key="phobert_pol_single:20260521",
        seed=20260521,
        as_dict=lambda _root: {},
    )
    model = DeviceCheckingModel()
    save_checkpoint(
        source.checkpoint_path,
        build_checkpoint_payload(model, None, None, None, {"status": "fixture"}),
    )
    predictor = DiskBackedQ1BPredictor(
        ROOT,
        {"system_id": "phobert_pol_single", "seed": 20260521},
        source=source,
        model=model,
        tokenizer=SimpleNamespace(batch_encode=lambda _texts, **_kwargs: {"input_ids": [[1, 2]], "attention_mask": [[1, 1]]}),
    )
    assert predictor.predict("vsfc", _example("external", split="test")) == "neutral"
    assert model.observed_device == next(model.parameters()).device


def test_red_team_q1b_loader_rejects_zero_matching_keys(tmp_path: Path) -> None:
    from vipragsent.orchestration.q1b_predictor import DiskBackedQ1BPredictor

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

        def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, Any]:
            del input_ids, attention_mask
            return {"logits": {"polarity": torch.tensor([[0.0, 1.0, 0.0]])}}

    checkpoint = tmp_path / "unrelated_checkpoint.pt"
    torch.save({"model_state_dict": {"totally.unrelated": torch.ones(1)}}, checkpoint)
    source = SimpleNamespace(
        checkpoint_path=checkpoint,
        checkpoint_key="phobert_pol_single:20260521",
        seed=20260521,
        as_dict=lambda _root: {},
    )
    predictor = DiskBackedQ1BPredictor(
        ROOT,
        {"system_id": "phobert_pol_single", "seed": 20260521},
        source=source,
        model=Model(),
        tokenizer=SimpleNamespace(batch_encode=lambda _texts, **_kwargs: {"input_ids": [[1, 2]], "attention_mask": [[1, 1]]}),
    )
    with pytest.raises((RuntimeError, ValueError, KeyError)):
        predictor.validate_checkpoint()
    checkpoint.unlink(missing_ok=True)


def test_red_team_q1b_inventory_has_real_trainable_producers_for_every_consumer() -> None:
    from vipragsent.orchestration.q1b_dependencies import build_q1b_dependency_graph

    inventory = load_inventory(ROOT)
    q1b_ids = {str(row.get("experiment_id") or row.get("run_id")) for row in inventory if row["research_question"] == "Q1b" and row["backbone"] != "azure"}
    graph = build_q1b_dependency_graph(ROOT, inventory_rows=inventory)
    assert graph["status"] == "PASS", graph["errors"]
    assert graph["paper_inventory_count_before"] == graph["paper_inventory_count_after"] == 162
    edges = [edge for edge in graph["edges"] if edge.get("consumer_id") in q1b_ids]
    assert len(edges) == len(q1b_ids), "every Q1B consumer must have exactly one graph producer"
    assert all(edge["expected_checkpoint_key"] == edge["produced_checkpoint_key"] for edge in edges)


def _generation_table_records(tmp_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for seed in TRAINING_SEEDS:
        run_root = tmp_path / str(seed)
        run_root.mkdir(parents=True)
        rows = []
        for index in range(4):
            rows.append({"sample_id": f"s{index}", "gold": {label: index % 2 for label in PRAGMATIC_LABELS}, "predictions": {label: (index + (seed % 2)) % 2 for label in PRAGMATIC_LABELS}, "effective_full_split_all_zero_fallback": {label: index % 2 for label in PRAGMATIC_LABELS}})
        prediction_path = run_root / "predictions/test_predictions.jsonl"
        prediction_path.parent.mkdir()
        prediction_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        records.append({
            "run_id": f"run-{seed}",
            "run_root": run_root,
            "summary": {
                "system_id": "cot_only_vistral",
                "backbone": "vistral_7b",
                "seed": seed,
                "primary_per_label_f1": {label: 0.5 for label in PRAGMATIC_LABELS},
                "primary_macro_f1": 0.5,
                "invalid_generation_rate": 0.0,
                "invalid_judge_output_rate": 0.0,
            },
        })
    return records


def test_red_team_table2_generation_uses_joint_ci_and_all_zero_fallback(tmp_path: Path) -> None:
    rows = _table2(_generation_table_records(tmp_path))
    row = rows[0]
    assert row["macro_prag_ci_low"] != "NOT_APPLICABLE"
    assert row["macro_prag_ci_high"] != "NOT_APPLICABLE"
    assert row["macro_prag_f1"] == pytest.approx(1.0), "generation CI path ignored effective all-zero-fallback predictions"

    seed_rows = []
    for seed in TRAINING_SEEDS:
        seed_rows.append([{"sample_id": f"s{index}", "gold": {label: index % 2 for label in PRAGMATIC_LABELS}, "predictions": {label: index % 2 for label in PRAGMATIC_LABELS}} for index in range(4)])
    direct = evaluate_q1a_confidence_intervals(seed_rows, prediction_hash="x", config_hash="x", code_commit="x", resamples=30)
    assert row["macro_prag_ci_low"] == pytest.approx(direct["macro"]["low"])


def test_red_team_reasoning_judge_uses_public_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    valid = {label: 0 for label in PRAGMATIC_LABELS}
    valid["polarity"] = "neutral"
    valid["emotion"] = "other"
    calls: list[str] = []

    class PublicClient:
        def __init__(self, _settings: Any, **_kwargs: Any) -> None:
            self.cache = None

        def create_structured(self, **_kwargs: Any) -> dict[str, Any]:
            calls.append("create_structured")
            return {"valid": True, "labels": valid, "observed_model": "GPT-4.1-mini", "observed_model_version": "2025-04-14"}

        def _default_transport(self, **_kwargs: Any) -> Any:
            raise AssertionError("ReasoningJudge bypassed the public structured client")

    import vipragsent.azure.client as azure_module
    import vipragsent.evaluation.reasoning_judge as judge_module

    monkeypatch.setattr(azure_module, "AzureResponsesClient", PublicClient)
    monkeypatch.setattr(judge_module, "AzureResponsesClient", PublicClient, raising=False)
    monkeypatch.setattr(azure_module.AzureSettings, "from_env", classmethod(lambda cls: AzureSettings("https://azure.example/", "https://azure.example/openai/v1/", "judge", None, "api_key", "not-a-secret")))
    monkeypatch.setattr(judge_module, "validate_reasoning_protocol_files", lambda _root: {"status": "PASS", "errors": []})
    judge = ReasoningJudge(ROOT, cache_root=tmp_path / "judge-cache", sleep_fn=lambda _seconds: None)
    result = judge.judge("generated reasoning only")
    assert result["valid"] is True
    assert calls == ["create_structured"]


def test_red_team_reasoning_judge_parses_nested_responses_payload(tmp_path: Path) -> None:
    valid = {label: 0 for label in PRAGMATIC_LABELS}
    valid["polarity"] = "neutral"
    valid["emotion"] = "other"
    settings = AzureSettings("https://azure.example/", "https://azure.example/openai/v1/", "judge", None, "api_key", "not-a-secret")
    payload = {"output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(valid)}]}], "model": "GPT-4.1-mini", "version": "2025-04-14"}
    client = AzureResponsesClient(settings, transport=lambda **_kwargs: payload, cache=AzureCache(tmp_path / "azure-cache"), safety=AzureSafetyCeilings(allow_unknown_spend=True))
    record = client.create_structured(prompt="reason", task="all", schema={"strict": True, "schema": strict_label_schema("all")}, max_output_tokens=20)
    assert record["labels"] == valid, "nested Responses output_text payload was not parsed by the shared client"


def test_red_team_reasoning_judge_caches_terminal_invalid_and_validates_model_version(tmp_path: Path) -> None:
    settings = AzureSettings("https://azure.example/", "https://azure.example/openai/v1/", "judge", None, "api_key", "not-a-secret")
    calls = 0

    def invalid_transport(**_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"parsed": {"unexpected": 1}, "model": "GPT-4.1-mini", "version": "2025-04-14"}

    client = AzureResponsesClient(settings, transport=invalid_transport, cache=AzureCache(tmp_path / "cache"), safety=AzureSafetyCeilings(allow_unknown_spend=True))
    schema = {"strict": True, "schema": strict_label_schema("all")}
    first = client.create_structured(prompt="reason", task="all", schema=schema, max_output_tokens=20, return_invalid=True)
    second = client.create_structured(prompt="reason", task="all", schema=schema, max_output_tokens=20, return_invalid=True)
    assert first["valid"] is False and second["valid"] is False
    assert calls == 1, "terminal-invalid structured output was not cached"


def test_red_team_reasoning_judge_rejects_model_version_mismatch() -> None:
    settings = AzureSettings("https://azure.example/", "https://azure.example/openai/v1/", "judge", None, "api_key", "not-a-secret")
    mismatch = AzureResponsesClient(settings, transport=lambda **_kwargs: {"parsed": {label: 0 for label in PRAGMATIC_LABELS} | {"polarity": "neutral", "emotion": "other"}, "model": "GPT-4.1-mini", "version": "wrong"}, safety=AzureSafetyCeilings(allow_unknown_spend=True))
    schema = {"strict": True, "schema": strict_label_schema("all")}
    with pytest.raises(Exception, match="version"):
        mismatch.create_structured(prompt="reason", task="all", schema=schema, max_output_tokens=20)


def test_red_team_explanation_and_cot_provenance_are_system_specific() -> None:
    from vipragsent.orchestration.provenance import (
        expected_inference_provenance,
        validate_inference_provenance,
    )

    explanation = expected_inference_provenance("explanation_only_vistral")
    cot = expected_inference_provenance("cot_only_vistral")
    assert explanation["rationale_decoder_enabled_at_inference"] is True
    assert explanation["native_causal_lm_generation_used"] is False
    assert explanation["inference_output_source"] == "judge_of_rationale_decoder_output"
    assert cot["native_causal_lm_generation_used"] is True
    assert cot["rationale_decoder_enabled_at_inference"] is False
    assert not validate_inference_provenance({"system_id": "explanation_only_vistral", **explanation})
    assert not validate_inference_provenance({"system_id": "cot_only_vistral", **cot})


def test_red_team_audits_bind_executable_test_and_observed_hashes() -> None:
    report_path = ROOT / "reports/local_production_correctness_closure.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report.get("status") == "PASS"
    assert report.get("production_proof") is False, "synthetic closure evidence was presented as production proof"
    evidence = report.get("evidence") or []
    assert evidence, "audit PASS has no executable evidence rows"
    assert all(item.get("test_name") for item in evidence), "audit PASS is not bound to an executable test"
    assert all(item.get("fixture_input_sha256") or item.get("golden_input_sha256") for item in evidence), "audit lacks input evidence"
    assert all(item.get("observed_output_sha256") for item in evidence), "audit lacks observed output evidence"
    assert all(item.get("synthetic") is True for item in evidence)
