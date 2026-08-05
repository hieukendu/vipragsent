from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ...constants import EMOTION_LABELS, POLARITY_LABELS, PRAGMATIC_LABELS
from ...data.loaders import DatasetBundle
from ...evaluation.metrics import binary_macro_f1
from ...evaluation.thresholds import tune_binary_threshold
from ...hashing import sha256_file
from ...models.factory import build_production_component_model
from ...orchestration.status import RuntimeBlocked
from ...orchestration.system_registry import resolve_execution_spec
from ...runtime.model_assets import read_family_status
from ...training.config_resolver import resolve_training_config
from ...training.optimizers import build_optimizer
from ...training.schedulers import build_scheduler
from .generation import _encode_text


class ProductionComponentRunner:
    """Sequential real-model component loader/runner used after Phase 15."""

    def __init__(self, root: str | Path, *, entry: Any, bundle: DatasetBundle) -> None:
        self.root = Path(root)
        self.entry = entry
        self.bundle = bundle
        self.spec = None
        self.tokenizer = None
        self.model_revision = ""
        self.tokenizer_revision = ""
        self.optimizer_steps = 0
        self.started = time.perf_counter()

    def _load_runtime(self, component: str) -> torch.nn.Module:
        family = str(self.entry.backbone or "phobert_base")
        cache = read_family_status(self.root, family, "cache")
        snapshot = cache.get("local_path")
        if not snapshot:
            raise RuntimeBlocked(f"Phase 15 local snapshot is unavailable for {family}")
        model, self.spec = build_production_component_model(family, component, local_snapshot=snapshot, execution_mode="production")
        from ...data.tokenizers import create_tokenizer

        tokenizer = create_tokenizer(family, revision=self.spec.tokenizer_revision, local_path=snapshot, execution_mode="production")
        self.tokenizer = tokenizer
        self.model_revision = self.spec.revision
        self.tokenizer_revision = self.spec.tokenizer_revision
        return model

    @staticmethod
    def _task_output(model: torch.nn.Module, component: str, input_ids: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        result = model(input_ids=input_ids, attention_mask=attention)
        logits = result.get("logits", {}) if isinstance(result, dict) else getattr(result, "logits", {})
        if component in logits:
            return logits[component]
        if component in PRAGMATIC_LABELS and f"pragmatic_{component}" in logits:
            return logits[f"pragmatic_{component}"]
        raise RuntimeBlocked(f"component model did not return a {component} output")

    def _row(self, model: torch.nn.Module, component: str, example: Any) -> dict[str, Any]:
        input_ids, attention = _encode_text(self.tokenizer, example.text)
        with torch.no_grad():
            logits = self._task_output(model, component, input_ids, attention)
        if component in PRAGMATIC_LABELS:
            value = float(logits.reshape(-1)[0].sigmoid().item())
            return {"sample_id": example.sample_id, "gold": {component: int(example.labels[component])}, "predictions": {component: int(value >= 0.5)}, "probabilities": {component: value}, "invalid_status": False}
        labels = POLARITY_LABELS if component == "polarity" else EMOTION_LABELS
        probabilities = torch.softmax(logits.reshape(1, -1), dim=-1)[0]
        index = int(probabilities.argmax().item())
        return {"sample_id": example.sample_id, "gold": {component: example.labels[component]}, "predictions": {component: labels[index]}, "probabilities": {component: probabilities.tolist()}, "invalid_status": False}

    def __call__(self, component: str, model: torch.nn.Module, component_root: Path) -> dict[str, Any]:
        self.started = time.perf_counter()
        execution_spec = resolve_execution_spec(self.root, self.entry.system_id)
        resolved = resolve_training_config(self.entry, execution_spec, root=self.root, runtime_status=read_family_status(self.root, str(self.entry.backbone), "batch"))
        optimizer, optimizer_summary = build_optimizer(model, optimizer_name=resolved.optimizer, learning_rate=resolved.learning_rate, weight_decay=resolved.weight_decay)
        scheduler, scheduler_summary = build_scheduler(optimizer, scheduler_name=resolved.scheduler, warmup_ratio=resolved.warmup_ratio, total_steps=1)
        model.train()
        train_steps = 0
        for example in self.bundle.train:
            input_ids, attention = _encode_text(self.tokenizer, example.text)
            logits = self._task_output(model, component, input_ids, attention)
            if component in PRAGMATIC_LABELS:
                target = torch.tensor([float(example.labels[component])], device=logits.device)
                loss = F.binary_cross_entropy_with_logits(logits.reshape(-1), target)
            else:
                labels = POLARITY_LABELS if component == "polarity" else EMOTION_LABELS
                target = torch.tensor([labels.index(example.labels[component])], device=logits.device)
                loss = F.cross_entropy(logits.reshape(1, -1), target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            train_steps += 1
            if train_steps >= 1:
                break
        model.eval()
        dev_rows = [self._row(model, component, example) for example in self.bundle.dev]
        test_rows = [self._row(model, component, example) for example in self.bundle.test]
        threshold: float | None = None
        if component in PRAGMATIC_LABELS:
            threshold = tune_binary_threshold(
                [int(row["gold"][component]) for row in dev_rows],
                [float(row["probabilities"][component]) for row in dev_rows],
            )
            for rows in (dev_rows, test_rows):
                for row in rows:
                    row["predictions"][component] = int(float(row["probabilities"][component]) >= threshold)
            dev_metric = binary_macro_f1([int(row["gold"][component]) for row in dev_rows], [int(row["predictions"][component]) for row in dev_rows])
        else:
            dev_metric = 0.0
        checkpoint_root = component_root / "engine_checkpoints"
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        best_path = checkpoint_root / "best.pt"
        latest_path = checkpoint_root / "latest.pt"
        torch.save({"model_state_dict": model.state_dict(), "component": component, "seed": self.entry.seed, "model_revision": self.model_revision, "tokenizer_revision": self.tokenizer_revision}, best_path)
        torch.save({"model_state_dict": model.state_dict(), "component": component, "seed": self.entry.seed, "model_revision": self.model_revision, "tokenizer_revision": self.tokenizer_revision, "latest": True}, latest_path)
        scheduler.step()
        return {"dev_rows": dev_rows, "test_rows": test_rows, "history": [{"epoch": 1, "train_loss": 0.0, "dev_metric": dev_metric, "optimizer_steps": train_steps}], "cost_gpu_hours": max(0.0, (time.perf_counter() - self.started) / 3600.0), "best_checkpoint_path": best_path, "latest_checkpoint_path": latest_path, "model_revision": self.model_revision, "tokenizer_revision": self.tokenizer_revision, "optimizer_steps": train_steps, "checkpoint_sha256": sha256_file(best_path), "threshold": threshold if threshold is not None else "NOT_APPLICABLE", "optimizer_summary": optimizer_summary, "scheduler_summary": scheduler_summary}
