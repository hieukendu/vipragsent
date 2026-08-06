from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vipragsent.orchestration.executors.component_production as component_production
from vipragsent.constants import EMOTION_LABELS, POLARITY_LABELS, PRAGMATIC_LABELS
from vipragsent.data.loaders import DatasetBundle, DatasetExample
from vipragsent.orchestration.executors.component_bundle import run_component_bundle
from vipragsent.training.class_weights import synthetic_class_weights


class _TinyTokenizer:
    def batch_encode(self, texts: list[str], *, max_length: int) -> dict[str, list[list[int]]]:
        encoded: list[list[int]] = []
        for text in texts:
            values = [ord(char) % 11 + 1 for char in text[:max_length]] or [1]
            encoded.append(values)
        return {"input_ids": encoded, "attention_mask": [[1] * len(values) for values in encoded]}


class _TinyComponentModel(nn.Module):
    def __init__(self, component: str) -> None:
        super().__init__()
        self.component = component
        output_size = 1 if component in PRAGMATIC_LABELS else len(POLARITY_LABELS if component == "polarity" else EMOTION_LABELS)
        self.projection = nn.Linear(1, output_size)
        self.training_examples = 0

    def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        if self.training:
            self.training_examples += int(input_ids.size(0))
        pooled = (input_ids.float() * attention_mask.float()).sum(dim=1, keepdim=True) / attention_mask.sum(dim=1, keepdim=True).clamp_min(1)
        return {"logits": {self.component: self.projection(pooled)}}


def _labels(index: int) -> dict[str, int | str]:
    return {
        **{label: index % 2 for label in PRAGMATIC_LABELS},
        "polarity": POLARITY_LABELS[index % len(POLARITY_LABELS)],
        "emotion": EMOTION_LABELS[index % len(EMOTION_LABELS)],
    }


def _bundle(train_count: int = 5) -> DatasetBundle:
    return DatasetBundle(
        splits={
            "train": [DatasetExample(f"train-{index}", f"train text {index}", _labels(index), "train") for index in range(train_count)],
            "dev": [DatasetExample(f"dev-{index}", f"dev text {index}", _labels(index + 1), "dev") for index in range(3)],
            "test": [DatasetExample(f"test-{index}", f"test text {index}", _labels(index + 2), "test") for index in range(4)],
        },
        fingerprint="data-fingerprint",
        manifest={"synthetic": True},
    )


def _runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    component: str = "sarcasm",
    train_count: int = 5,
    physical_batch_size: int = 2,
    gradient_accumulation_steps: int = 2,
    maximum_epochs: int = 2,
    patience: int = 10,
) -> tuple[component_production.ProductionComponentRunner, _TinyComponentModel, Path]:
    config = SimpleNamespace(
        optimizer="AdamW",
        learning_rate=0.01,
        weight_decay=0.0,
        scheduler="linear",
        warmup_ratio=0.0,
        physical_batch_size=physical_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        maximum_epochs=maximum_epochs,
        patience=patience,
        minimum_delta=0.0,
        gradient_clipping=1.0,
        config_hash="resolved-config",
    )
    monkeypatch.setattr(component_production, "resolve_execution_spec", lambda *_args: object())
    monkeypatch.setattr(component_production, "resolve_training_config", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(component_production, "read_family_status", lambda *_args: {"successful_batch": physical_batch_size})
    entry = SimpleNamespace(system_id="synthetic_component", backbone="phobert_base", seed=20260521)
    runner = component_production.ProductionComponentRunner(
        tmp_path,
        entry=entry,
        bundle=_bundle(train_count),
        class_weights=synthetic_class_weights(dataset_hash="data-fingerprint", code_commit="test"),
    )
    runner.tokenizer = _TinyTokenizer()
    model = _TinyComponentModel(component)
    return runner, model, tmp_path / "component"


def test_component_runner_consumes_all_training_examples(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner, model, component_root = _runner(monkeypatch, tmp_path, train_count=5, maximum_epochs=2)
    result = runner("sarcasm", model, component_root)
    assert model.training_examples == 10
    assert result["examples_seen"] == 10


def test_component_runner_runs_locked_epochs_or_early_stops(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner, model, component_root = _runner(monkeypatch, tmp_path, maximum_epochs=3, patience=10)
    result = runner("sarcasm", model, component_root)
    assert len(result["history"]) == 3
    assert result["actual_epochs"] <= 3


def test_component_runner_uses_gradient_accumulation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner, model, component_root = _runner(monkeypatch, tmp_path, train_count=5, physical_batch_size=2, gradient_accumulation_steps=2, maximum_epochs=2)
    result = runner("sarcasm", model, component_root)
    assert result["optimizer_steps"] == 4
    assert result["scheduler_summary"]["total_optimizer_steps"] == 4
    assert [row["optimizer_steps"] for row in result["history"]] == [2, 2]


def test_component_runner_uses_pragmatic_pos_weight(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner, model, _ = _runner(monkeypatch, tmp_path)
    observed: dict[str, torch.Tensor] = {}
    original = component_production.F.binary_cross_entropy_with_logits

    def wrapped(input_tensor: torch.Tensor, target: torch.Tensor, **kwargs: object) -> torch.Tensor:
        observed["pos_weight"] = kwargs["pos_weight"]  # type: ignore[assignment]
        return original(input_tensor, target, **kwargs)

    monkeypatch.setattr(component_production.F, "binary_cross_entropy_with_logits", wrapped)
    runner._loss(model, "sarcasm", runner.bundle.train[:2], torch.device("cpu"), runner._resolved_class_weights())
    assert observed["pos_weight"].item() == pytest.approx(3.0)


def test_component_runner_uses_multiclass_weights(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner, model, _ = _runner(monkeypatch, tmp_path, component="polarity")
    observed: dict[str, torch.Tensor] = {}
    original = component_production.F.cross_entropy

    def wrapped(input_tensor: torch.Tensor, target: torch.Tensor, **kwargs: object) -> torch.Tensor:
        observed["weight"] = kwargs["weight"]  # type: ignore[assignment]
        return original(input_tensor, target, **kwargs)

    monkeypatch.setattr(component_production.F, "cross_entropy", wrapped)
    runner._loss(model, "polarity", runner.bundle.train[:2], torch.device("cpu"), runner._resolved_class_weights())
    assert observed["weight"].tolist() == pytest.approx([1.0, 1.1, 1.2])


def test_component_runner_multiclass_threshold_not_applicable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner, _, _ = _runner(monkeypatch, tmp_path, component="polarity")
    rows = [
        {"gold": {"polarity": POLARITY_LABELS[0]}, "predictions": {"polarity": POLARITY_LABELS[0]}, "probabilities": {"polarity": [1.0, 0.0, 0.0]}},
        {"gold": {"polarity": POLARITY_LABELS[1]}, "predictions": {"polarity": POLARITY_LABELS[1]}, "probabilities": {"polarity": [0.0, 1.0, 0.0]}},
        {"gold": {"polarity": POLARITY_LABELS[2]}, "predictions": {"polarity": POLARITY_LABELS[2]}, "probabilities": {"polarity": [0.0, 0.0, 1.0]}},
    ]
    threshold, metric, name = runner._selection("polarity", rows)
    assert threshold == "NOT_APPLICABLE"
    assert metric == pytest.approx(1.0)
    assert name == "dev_polarity_macro_f1"


def test_component_runner_writes_real_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner, model, component_root = _runner(monkeypatch, tmp_path)
    result = runner("sarcasm", model, component_root)
    checkpoint = Path(result["best_checkpoint_path"])
    assert checkpoint.stat().st_size > 0
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["schema_version"] == 2
    assert payload["model_state_dict"]
    assert payload["metadata"]["component"] == "sarcasm"


def test_component_runner_distinct_dev_test_ids(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner, model, component_root = _runner(monkeypatch, tmp_path)
    result = runner("sarcasm", model, component_root)
    assert [row["sample_id"] for row in result["dev_rows"]] == ["dev-0", "dev-1", "dev-2"]
    assert [row["sample_id"] for row in result["test_rows"]] == ["test-0", "test-1", "test-2", "test-3"]


def test_component_runner_resume_preserves_hashes(tmp_path: Path) -> None:
    kwargs = {"dev_sample_ids": ("dev-0", "dev-1"), "test_sample_ids": ("test-0", "test-1"), "seed": 20260521, "config_hash": "config", "data_hash": "data", "model_hash": "model"}
    first = run_component_bundle(tmp_path, executor_kind="single_task_bundle", **kwargs)
    state_before = json.loads((tmp_path / "components/state.json").read_text(encoding="utf-8"))
    second = run_component_bundle(tmp_path, executor_kind="single_task_bundle", resume=True, **kwargs)
    state_after = json.loads((tmp_path / "components/state.json").read_text(encoding="utf-8"))
    assert second["component_checkpoint_sha256"] == first["component_checkpoint_sha256"]
    assert state_after["config_hash"] == state_before["config_hash"] == "config"
    assert state_after["data_hash"] == state_before["data_hash"] == "data"


def test_component_runner_releases_previous_component(tmp_path: Path) -> None:
    class _Owner:
        def __init__(self) -> None:
            self.released = 0

        def load(self, _component: str) -> object:
            return object()

        def run(self, _component: str, _model: object, _root: Path) -> None:
            return None

        def release_runtime(self) -> None:
            self.released += 1

    owner = _Owner()
    run_component_bundle(
        tmp_path,
        executor_kind="single_task_bundle",
        dev_sample_ids=("dev-0", "dev-1"),
        test_sample_ids=("test-0", "test-1"),
        seed=20260521,
        config_hash="config",
        data_hash="data",
        model_hash="model",
        model_loader=owner.load,
        component_runner=owner.run,
        allow_synthetic=True,
    )
    assert owner.released == len(PRAGMATIC_LABELS)
