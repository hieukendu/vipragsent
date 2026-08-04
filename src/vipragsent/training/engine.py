from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor, nn

from ..evaluation.metrics import macro_pragmatic_f1
from ..evaluation.thresholds import tune_pragmatic_thresholds
from ..models.losses import UncertaintyWeightedMultiTaskLoss, classification_losses, token_cross_entropy
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


@dataclass
class RunState:
    epoch: int = 0
    best_metric: float = float("-inf")
    best_loss: float = float("inf")
    best_epoch: int | None = None
    no_improvement_epochs: int = 0
    status: str = "pending"
    history: list[dict[str, float]] = field(default_factory=list)


class CheckpointManager:
    def __init__(self, root: str | Path, run_id: str) -> None:
        self.path = Path(root) / run_id
        self.path.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, model: nn.Module, optimizer: torch.optim.Optimizer, state: RunState) -> Path:
        path = self.path / f"{name}.pt"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "state": asdict(state)}, path)
        return path

    def load_latest(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> RunState | None:
        checkpoints = sorted(self.path.glob("epoch_*.pt"))
        if not checkpoints:
            return None
        payload = torch.load(checkpoints[-1], map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        state = RunState(**payload["state"])
        return state


class EvaluationAccessGate:
    def __init__(self) -> None:
        self.checkpoint_frozen = False

    def freeze_checkpoint(self) -> None:
        self.checkpoint_frozen = True

    def assert_test_allowed(self) -> None:
        if not self.checkpoint_frozen:
            raise RuntimeError("Test evaluation is prohibited before checkpoint selection is frozen")


class TrainingEngine:
    def __init__(self, model: nn.Module, config: TrainingConfig, *, run_id: str, checkpoint_root: str | Path = "checkpoints") -> None:
        self.model = model
        self.config = config
        self.run_id = run_id
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.loss_aggregator = UncertaintyWeightedMultiTaskLoss()
        self.optimizer.add_param_group({"params": self.loss_aggregator.parameters()})
        self.checkpoints = CheckpointManager(checkpoint_root, run_id)
        self.gate = EvaluationAccessGate()

    def _loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor]]:
        output = self.model(batch["input_ids"], batch["attention_mask"], rationale_input_ids=batch.get("rationale_input_ids"))
        losses = classification_losses(
            output["logits"],
            batch["targets"],
            pragmatic_pos_weight=batch.get("pragmatic_pos_weight"),
            polarity_weight=batch.get("polarity_weight"),
            emotion_weight=batch.get("emotion_weight"),
            active_tasks=self.model.config.active_tasks,
        )
        rationale_loss = None
        if "rationale_logits" in output and "rationale_targets" in batch:
            rationale_loss = token_cross_entropy(output["rationale_logits"], batch["rationale_targets"])
        return self.loss_aggregator(losses, rationale_loss)

    def train(self, batches: Iterable[dict[str, Any]], *, seed: int, resume: bool = False) -> RunState:
        seed_everything(seed)
        state = self.checkpoints.load_latest(self.model, self.optimizer) if resume else None
        state = state or RunState(status="running")
        batches = list(batches)
        if not batches:
            raise ValueError("Training requires at least one batch")
        for epoch in range(state.epoch, self.config.max_epochs):
            self.model.train()
            started = time.perf_counter()
            self.optimizer.zero_grad(set_to_none=True)
            train_losses: list[float] = []
            for index, batch in enumerate(batches):
                total, _ = self._loss(batch)
                (total / self.config.gradient_accumulation_steps).backward()
                if (index + 1) % self.config.gradient_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(list(self.model.parameters()) + list(self.loss_aggregator.parameters()), self.config.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                train_losses.append(float(total.detach().cpu()))
            train_loss = sum(train_losses) / len(train_losses)
            metric = 0.0
            epoch_record = {"epoch": float(epoch + 1), "train_loss": train_loss, "dev_macro_pragmatic_f1": metric, "seconds": time.perf_counter() - started}
            state.history.append(epoch_record)
            state.epoch = epoch + 1
            improved = metric > state.best_metric + self.config.min_delta or (
                abs(metric - state.best_metric) <= self.config.min_delta and train_loss < state.best_loss
            )
            if improved:
                state.best_metric = metric
                state.best_loss = train_loss
                state.best_epoch = epoch + 1
                state.no_improvement_epochs = 0
                self.checkpoints.save("best", self.model, self.optimizer, state)
            else:
                state.no_improvement_epochs += 1
            self.checkpoints.save(f"epoch_{epoch + 1:03d}", self.model, self.optimizer, state)
            if state.no_improvement_epochs > self.config.patience:
                break
        state.status = "completed"
        (self.checkpoints.path / "run_state.json").write_text(json.dumps(asdict(state), indent=2) + "\n", encoding="utf-8")
        self.gate.freeze_checkpoint()
        return state

    def assert_test_access(self) -> None:
        self.gate.assert_test_allowed()
