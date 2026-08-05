from __future__ import annotations

import json
import os
import random
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from ..atomic import atomic_write_json
from ..constants import EMOTION_LABELS, POLARITY_LABELS, PRAGMATIC_LABELS
from ..evaluation.metrics import binary_macro_f1, macro_pragmatic_f1, multiclass_macro_f1
from ..evaluation.thresholds import tune_binary_threshold
from ..models.losses import (
    UncertaintyWeightedMultiTaskLoss,
    classification_losses,
    equal_weight_loss,
    token_cross_entropy,
)
from .seeding import seed_everything


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_epochs: int = 1
    effective_batch_size: int = 32
    physical_batch_size: int = 2
    max_grad_norm: float = 1.0
    patience: int = 2
    min_delta: float = 0.0001
    precision: str = "bf16"
    gradient_accumulation_steps: int = 1
    primary_metric: str = "dev_macro_pragmatic_f1"
    scheduler: str = "linear"
    warmup_ratio: float = 0.1
    use_uncertainty_weighting: bool = True
    rationale_beta: float = 0.3

    def __post_init__(self) -> None:
        if self.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError(f"Unsupported precision: {self.precision}")
        if self.gradient_accumulation_steps < 1 or self.max_epochs < 1:
            raise ValueError("max_epochs and gradient_accumulation_steps must be positive")
        if self.primary_metric not in {
            "dev_macro_pragmatic_f1",
            "dev_sarcasm_macro_f1",
            "dev_sarcasm_binary_macro_f1",
            "dev_polarity_macro_f1",
            "dev_emotion_macro_f1",
        }:
            raise ValueError(f"Unsupported primary metric: {self.primary_metric}")


@dataclass
class RunState:
    epoch: int = 0
    best_metric: float = float("-inf")
    best_loss: float = float("inf")
    best_epoch: int | None = None
    no_improvement_epochs: int = 0
    optimizer_updates: int = 0
    micro_batches: int = 0
    status: str = "PENDING"
    thresholds: dict[str, float] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SelectionResult:
    metric: float
    total_loss: float
    thresholds: dict[str, float] = field(default_factory=dict)
    true: dict[str, list[Any]] = field(default_factory=dict)
    probabilities: dict[str, Any] = field(default_factory=dict)
    predictions: dict[str, list[Any]] = field(default_factory=dict)
    logits: dict[str, list[Any]] = field(default_factory=dict)


SelectionCallback = Callable[["TrainingEngine", list[dict[str, Any]]], SelectionResult]


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().tolist(),
    }
    if torch.cuda.is_available():
        state["cuda"] = [item.tolist() for item in torch.cuda.get_rng_state_all()]
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    if not state:
        return

    def as_tuple(value: Any) -> Any:
        return tuple(as_tuple(item) for item in value) if isinstance(value, list) else value

    random.setstate(as_tuple(state["python"]))
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), numpy_state[2], numpy_state[3], numpy_state[4]))
    torch.set_rng_state(torch.tensor(state["torch"], dtype=torch.uint8))
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all([torch.tensor(item, dtype=torch.uint8) for item in state["cuda"]])


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


class CheckpointManager:
    def __init__(self, root: str | Path, run_id: str) -> None:
        self.path = Path(root) / run_id
        self.path.mkdir(parents=True, exist_ok=True)

    def _payload(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        loss_aggregator: nn.Module,
        state: RunState,
    ) -> dict[str, Any]:
        return {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "loss_aggregator": loss_aggregator.state_dict(),
            "state": asdict(state),
            "rng_state": _rng_state(),
        }

    def save(
        self,
        name: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        loss_aggregator: nn.Module,
        state: RunState,
    ) -> Path:
        path = self.path / f"{name}.pt"
        _atomic_torch_save(path, self._payload(model, optimizer, scheduler, loss_aggregator, state))
        return path

    def _load_payload(
        self,
        path: Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scheduler: Any,
        loss_aggregator: nn.Module,
        *,
        restore_training_state: bool,
    ) -> RunState:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        if restore_training_state and optimizer is not None:
            optimizer.load_state_dict(payload["optimizer"])
            if scheduler is not None and payload.get("scheduler") is not None:
                scheduler.load_state_dict(payload["scheduler"])
            _restore_rng_state(payload.get("rng_state", {}))
        loss_aggregator.load_state_dict(payload.get("loss_aggregator", {}))
        return RunState(**payload["state"])

    def load_latest(self, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any, loss_aggregator: nn.Module) -> RunState | None:
        checkpoints = sorted(self.path.glob("epoch_*.pt"))
        if not checkpoints:
            return None
        return self._load_payload(checkpoints[-1], model, optimizer, scheduler, loss_aggregator, restore_training_state=True)

    def load_best(self, model: nn.Module, loss_aggregator: nn.Module) -> RunState | None:
        path = self.path / "best.pt"
        if not path.exists():
            return None
        return self._load_payload(path, model, None, None, loss_aggregator, restore_training_state=False)


class EvaluationAccessGate:
    def __init__(self) -> None:
        self.checkpoint_frozen = False

    def freeze_checkpoint(self) -> None:
        self.checkpoint_frozen = True

    def assert_test_allowed(self) -> None:
        if not self.checkpoint_frozen:
            raise RuntimeError("Test evaluation is prohibited before checkpoint selection is frozen")


def _active_pragmatic_keys(logits: Mapping[str, Tensor]) -> tuple[str, ...]:
    return tuple(key for key in PRAGMATIC_LABELS if key in logits)


class TrainingEngine:
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        *,
        run_id: str,
        checkpoint_root: str | Path = "checkpoints",
        runtime_hooks: Mapping[str, Callable[..., Any]] | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.run_id = run_id
        self.runtime_hooks = dict(runtime_hooks or {})
        variant_config = getattr(model, "config", None)
        variant_uncertainty = bool(getattr(variant_config, "has_uncertainty_weighting", True))
        self.uses_uncertainty_weighting = bool(config.use_uncertainty_weighting and variant_uncertainty)
        uncertainty_tasks = getattr(variant_config, "uncertainty_task_keys", None) or (*PRAGMATIC_LABELS, "polarity", "emotion")
        self.loss_aggregator = UncertaintyWeightedMultiTaskLoss(config.rationale_beta, tasks=uncertainty_tasks)
        try:
            device = next(model.parameters()).device
            self.loss_aggregator.to(device)
        except StopIteration:
            pass

        decay: list[nn.Parameter] = []
        no_decay: list[nn.Parameter] = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.endswith(".bias") or "norm" in name.lower() or "layernorm" in name.lower():
                no_decay.append(parameter)
            else:
                decay.append(parameter)
        optimizer_groups: list[dict[str, Any]] = []
        if decay:
            optimizer_groups.append({"params": decay, "weight_decay": config.weight_decay, "name": "model_decay"})
        if no_decay:
            optimizer_groups.append({"params": no_decay, "weight_decay": 0.0, "name": "model_no_decay"})
        if self.uses_uncertainty_weighting:
            optimizer_groups.append({"params": list(self.loss_aggregator.parameters()), "weight_decay": 0.0, "name": "uncertainty_no_decay"})
        self.optimizer = torch.optim.AdamW(optimizer_groups, lr=config.learning_rate)
        self.checkpoints = CheckpointManager(checkpoint_root, run_id)
        self.gate = EvaluationAccessGate()
        self.scheduler: Any = None
        self._scheduler_total_steps = 0

    def _ensure_scheduler(self, total_steps: int) -> None:
        if self.scheduler is not None and self._scheduler_total_steps == total_steps:
            return
        warmup = max(0, int(total_steps * self.config.warmup_ratio))

        def schedule(step: int) -> float:
            if step < warmup and warmup:
                return max(step, 1) / warmup
            progress = (step - warmup) / max(total_steps - warmup, 1)
            if self.config.scheduler == "cosine":
                return 0.5 * (1.0 + np.cos(np.pi * min(max(progress, 0.0), 1.0)))
            return max(0.0, 1.0 - min(max(progress, 0.0), 1.0))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, schedule)
        self._scheduler_total_steps = total_steps

    def _autocast(self) -> Any:
        if self.config.precision not in {"bf16", "fp16"} or not torch.cuda.is_available():
            return torch.autocast(device_type="cpu", enabled=False)
        dtype = torch.bfloat16 if self.config.precision == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor]]:
        with self._autocast():
            output = self.model(
                batch["input_ids"],
                batch["attention_mask"],
                rationale_input_ids=batch.get("rationale_input_ids"),
                rationale_attention_mask=batch.get("rationale_attention_mask"),
            )
            losses = classification_losses(
                output["logits"],
                batch["targets"],
                pragmatic_pos_weight=batch.get("pragmatic_pos_weight"),
                polarity_weight=batch.get("polarity_weight"),
                emotion_weight=batch.get("emotion_weight"),
                active_tasks=self.model.config.active_tasks,
                target_masks=batch.get("target_masks"),
                sarcasm_target_mask=batch.get("sarcasm_target_mask"),
            )
            rationale_loss = None
            if "rationale_logits" in output:
                rationale_targets = output.get("rationale_labels", batch.get("rationale_targets"))
                if rationale_targets is not None:
                    rationale_loss = token_cross_entropy(
                        output["rationale_logits"],
                        rationale_targets,
                        sample_mask=batch.get("rationale_loss_mask"),
                    )
            if self.uses_uncertainty_weighting:
                return self.loss_aggregator(losses, rationale_loss)
            reference = next(self.model.parameters(), None)
            return equal_weight_loss(losses, rationale_loss, rationale_beta=self.config.rationale_beta, reference=reference)

    def _default_selection(self, batches: list[dict[str, Any]]) -> SelectionResult:
        self.model.eval()
        true_prag = {key: [] for key in PRAGMATIC_LABELS}
        prob_prag = {key: [] for key in PRAGMATIC_LABELS}
        true_polarity: list[int] = []
        prob_polarity: list[list[float]] = []
        true_emotion: list[int] = []
        prob_emotion: list[list[float]] = []
        logits_export: dict[str, list[Any]] = {}
        losses: list[float] = []
        with torch.no_grad():
            for batch in batches:
                total, _ = self._loss(batch)
                losses.append(float(total.detach().cpu()))
                output = self.model(batch["input_ids"], batch["attention_mask"])
                logits = output.get("logits", {})
                for key in _active_pragmatic_keys(logits):
                    values = logits[key].detach().cpu()
                    logits_export.setdefault(key, []).extend(values.tolist())
                    prob_prag[key].extend(torch.sigmoid(values).tolist())
                    true_prag[key].extend(batch["targets"][key].detach().cpu().int().tolist())
                if "polarity" in logits:
                    values = logits["polarity"].detach().cpu()
                    logits_export.setdefault("polarity", []).extend(values.tolist())
                    prob_polarity.extend(torch.softmax(values, dim=-1).tolist())
                    true_polarity.extend(batch["targets"]["polarity"].detach().cpu().int().tolist())
                if "emotion" in logits:
                    values = logits["emotion"].detach().cpu()
                    logits_export.setdefault("emotion", []).extend(values.tolist())
                    prob_emotion.extend(torch.softmax(values, dim=-1).tolist())
                    true_emotion.extend(batch["targets"]["emotion"].detach().cpu().int().tolist())

        thresholds: dict[str, float] = {}
        predictions: dict[str, list[Any]] = {}
        primary = self.config.primary_metric
        active_prag = _active_pragmatic_keys({key: torch.empty(0) for key in prob_prag if prob_prag[key]})
        if active_prag:
            pragmatic_true = {key: true_prag[key] for key in active_prag}
            pragmatic_prob = {key: prob_prag[key] for key in active_prag}
            thresholds = {key: tune_binary_threshold(pragmatic_true[key], pragmatic_prob[key]) for key in active_prag}
            predictions.update({key: [int(value >= thresholds[key]) for value in pragmatic_prob[key]] for key in active_prag})
        if primary in {"dev_sarcasm_macro_f1", "dev_sarcasm_binary_macro_f1"}:
            if "sarcasm" not in predictions:
                raise ValueError("Q3 selection requires sarcasm logits on the dev set")
            metric = binary_macro_f1(true_prag["sarcasm"], predictions["sarcasm"])
        elif primary == "dev_polarity_macro_f1":
            if not prob_polarity:
                raise ValueError("Polarity selection requires polarity logits on the dev set")
            predictions["polarity"] = np.asarray(prob_polarity).argmax(axis=1).tolist()
            metric = multiclass_macro_f1(true_polarity, predictions["polarity"], range(len(POLARITY_LABELS)))
        elif primary == "dev_emotion_macro_f1":
            if not prob_emotion:
                raise ValueError("Emotion selection requires emotion logits on the dev set")
            predictions["emotion"] = np.asarray(prob_emotion).argmax(axis=1).tolist()
            metric = multiclass_macro_f1(true_emotion, predictions["emotion"], range(len(EMOTION_LABELS)))
        elif active_prag == PRAGMATIC_LABELS:
            metric = macro_pragmatic_f1(true_prag, {key: predictions[key] for key in PRAGMATIC_LABELS})
        elif all(key in true_prag and true_prag[key] for key in active_prag):
            metric = float(np.mean([binary_macro_f1(true_prag[key], predictions[key]) for key in active_prag]))
        else:
            raise ValueError("The default dev selector has no supported task outputs; provide a selection callback")
        true: dict[str, list[Any]] = {key: values for key, values in true_prag.items() if values}
        probabilities: dict[str, Any] = {key: values for key, values in prob_prag.items() if values}
        if true_polarity:
            true["polarity"] = true_polarity
            probabilities["polarity"] = prob_polarity
        if true_emotion:
            true["emotion"] = true_emotion
            probabilities["emotion"] = prob_emotion
        return SelectionResult(float(metric), float(np.mean(losses) if losses else 0.0), thresholds, true, probabilities, predictions, logits_export)

    def _evaluate_dev(self, batches: list[dict[str, Any]], selection_callback: SelectionCallback | None = None) -> SelectionResult:
        return selection_callback(self, batches) if selection_callback else self._default_selection(batches)

    @staticmethod
    def _prediction_rows(selection: SelectionResult, *, sample_ids: list[str] | None = None) -> list[dict[str, Any]]:
        count = max((len(values) for values in selection.predictions.values()), default=0)
        rows: list[dict[str, Any]] = []
        for index in range(count):
            row: dict[str, Any] = {"sample_id": sample_ids[index] if sample_ids and index < len(sample_ids) else f"row_{index:06d}"}
            row["predictions"] = {key: values[index] for key, values in selection.predictions.items() if index < len(values)}
            row["gold"] = {key: values[index] for key, values in selection.true.items() if index < len(values)}
            row["probabilities"] = {
                key: values[index] for key, values in selection.probabilities.items() if index < len(values)
            }
            row["logits"] = {key: values[index] for key, values in selection.logits.items() if index < len(values)}
            rows.append(row)
        return rows

    def _write_prediction_jsonl(self, path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def _export_run_outputs(
        self,
        state: RunState,
        dev_selection: SelectionResult,
        *,
        dev_batches: list[dict[str, Any]],
        test_batches: list[dict[str, Any]] | None,
        output_root: Path,
        metadata: Mapping[str, Any] | None,
        selection_callback: SelectionCallback | None,
    ) -> None:
        output_root.mkdir(parents=True, exist_ok=True)
        dev_ids = [sample_id for batch in dev_batches for sample_id in batch.get("sample_ids", [])]
        self._write_prediction_jsonl(output_root / "dev_predictions.jsonl", self._prediction_rows(dev_selection, sample_ids=dev_ids))
        test_selection = None
        if test_batches is not None:
            self.gate.assert_test_allowed()
            test_selection = self._evaluate_dev(test_batches, selection_callback)
            test_ids = [sample_id for batch in test_batches for sample_id in batch.get("sample_ids", [])]
            self._write_prediction_jsonl(output_root / "test_predictions.jsonl", self._prediction_rows(test_selection, sample_ids=test_ids))
        atomic_write_json(output_root / "thresholds.json", state.thresholds)
        atomic_write_json(output_root / "training_history.json", state.history)
        atomic_write_json(
            output_root / "run_manifest.json",
            {
                "run_id": self.run_id,
                "status": state.status,
                "best_metric": state.best_metric,
                "best_epoch": state.best_epoch,
                "primary_metric": self.config.primary_metric,
                "thresholds_frozen": bool(self.gate.checkpoint_frozen),
                "dev_metric": dev_selection.metric,
                "test_metric": test_selection.metric if test_selection else None,
                "prediction_files": [name for name, enabled in (("dev_predictions.jsonl", True), ("test_predictions.jsonl", test_selection is not None)) if enabled],
                "precision_requested": self.config.precision,
                "precision_runtime": self.config.precision if torch.cuda.is_available() else "fp32_cpu_fallback",
                **dict(metadata or {}),
            },
        )

    def train(
        self,
        batches: Iterable[dict[str, Any]],
        *,
        seed: int,
        dev_batches: Iterable[dict[str, Any]] | None = None,
        test_batches: Iterable[dict[str, Any]] | None = None,
        resume: bool = False,
        selection_callback: SelectionCallback | None = None,
        output_root: str | Path | None = None,
        run_metadata: Mapping[str, Any] | None = None,
    ) -> RunState:
        seed_everything(seed)
        batches_list = list(batches)
        if not batches_list:
            raise ValueError("Training requires at least one batch")
        dev_list = list(dev_batches or batches_list)
        test_list = list(test_batches) if test_batches is not None else None
        updates_per_epoch = (len(batches_list) + self.config.gradient_accumulation_steps - 1) // self.config.gradient_accumulation_steps
        self._ensure_scheduler(updates_per_epoch * self.config.max_epochs)
        state = self.checkpoints.load_latest(self.model, self.optimizer, self.scheduler, self.loss_aggregator) if resume else None
        state = state or RunState(status="RUNNING")
        for epoch in range(state.epoch, self.config.max_epochs):
            self.model.train()
            started = time.perf_counter()
            if hook := self.runtime_hooks.get("on_epoch_start"):
                hook(epoch + 1, state)
            train_losses: list[float] = []
            self.optimizer.zero_grad(set_to_none=True)
            for start in range(0, len(batches_list), self.config.gradient_accumulation_steps):
                window = batches_list[start:start + self.config.gradient_accumulation_steps]
                window_loss = 0.0
                for batch in window:
                    total, _ = self._loss(batch)
                    (total / len(window)).backward()
                    window_loss += float(total.detach().cpu())
                    state.micro_batches += 1
                nn.utils.clip_grad_norm_(list(self.model.parameters()) + list(self.loss_aggregator.parameters()), self.config.max_grad_norm)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                state.optimizer_updates += 1
                train_losses.append(window_loss / len(window))
                if hook := self.runtime_hooks.get("on_optimizer_step"):
                    hook(state.optimizer_updates, state)
            dev_selection = self._evaluate_dev(dev_list, selection_callback)
            pragmatic_metric = (
                macro_pragmatic_f1(
                    {key: dev_selection.true[key] for key in PRAGMATIC_LABELS},
                    {key: dev_selection.predictions[key] for key in PRAGMATIC_LABELS},
                )
                if all(key in dev_selection.true and key in dev_selection.predictions for key in PRAGMATIC_LABELS)
                else None
            )
            record = {
                "epoch": float(epoch + 1),
                "train_loss": float(np.mean(train_losses)),
                "dev_loss": dev_selection.total_loss,
                "dev_metric": dev_selection.metric,
                "dev_macro_pragmatic_f1": pragmatic_metric,
                "seconds": time.perf_counter() - started,
            }
            if torch.cuda.is_available():
                record["peak_memory_gb"] = float(torch.cuda.max_memory_allocated() / (1024**3))
                torch.cuda.reset_peak_memory_stats()
            state.history.append(record)
            state.epoch = epoch + 1
            state.thresholds = dev_selection.thresholds
            improved = dev_selection.metric > state.best_metric + self.config.min_delta or (
                abs(dev_selection.metric - state.best_metric) <= self.config.min_delta and dev_selection.total_loss < state.best_loss
            )
            if improved:
                state.best_metric = dev_selection.metric
                state.best_loss = dev_selection.total_loss
                state.best_epoch = epoch + 1
                state.no_improvement_epochs = 0
                self.checkpoints.save("best", self.model, self.optimizer, self.scheduler, self.loss_aggregator, state)
            else:
                state.no_improvement_epochs += 1
            self.checkpoints.save(f"epoch_{epoch + 1:03d}", self.model, self.optimizer, self.scheduler, self.loss_aggregator, state)
            if hook := self.runtime_hooks.get("on_epoch_end"):
                hook(epoch + 1, state, dev_selection)
            if state.no_improvement_epochs >= self.config.patience:
                break
        state.status = "PASS"
        best_state = self.checkpoints.load_best(self.model, self.loss_aggregator)
        if best_state is not None:
            state.best_metric = best_state.best_metric
            state.best_loss = best_state.best_loss
            state.best_epoch = best_state.best_epoch
            state.thresholds = best_state.thresholds
        atomic_write_json(self.checkpoints.path / "run_state.json", asdict(state))
        atomic_write_json(self.checkpoints.path / "thresholds.json", state.thresholds)
        self.gate.freeze_checkpoint()
        final_selection = self._evaluate_dev(dev_list, selection_callback)
        self._export_run_outputs(
            state,
            final_selection,
            dev_batches=dev_list,
            test_batches=test_list,
            output_root=Path(output_root) if output_root else self.checkpoints.path,
            metadata={"seed": seed, **dict(run_metadata or {})},
            selection_callback=selection_callback,
        )
        return state

    def assert_test_access(self) -> None:
        self.gate.assert_test_allowed()
