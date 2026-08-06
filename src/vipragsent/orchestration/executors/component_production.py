from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from ...atomic import atomic_write_json
from ...constants import EMOTION_LABELS, POLARITY_LABELS, PRAGMATIC_LABELS
from ...data.loaders import DatasetBundle
from ...evaluation.metrics import binary_macro_f1, multiclass_macro_f1
from ...evaluation.thresholds import tune_binary_threshold
from ...hashing import sha256_file, sha256_json
from ...models.factory import build_production_component_model
from ...orchestration.run_store import git_commit
from ...orchestration.status import RuntimeBlocked
from ...orchestration.system_registry import resolve_execution_spec
from ...runtime.device import (
    assert_runtime_device_contract,
    move_batch_to_device,
    resolve_model_input_device,
    write_device_report,
)
from ...runtime.hardware import validate_hardware
from ...runtime.model_assets import read_family_status, resolve_local_snapshot
from ...training.class_weights import ClassWeightBundle, compute_train_only_class_weights
from ...training.config_resolver import resolve_training_config
from ...training.optimizers import build_optimizer
from ...training.schedulers import build_scheduler
from .generation import _encode_text


class ProductionComponentRunner:
    """Sequential real-model component loader/runner used after Phase 15."""

    def __init__(
        self,
        root: str | Path,
        *,
        entry: Any,
        bundle: DatasetBundle,
        class_weights: ClassWeightBundle | Mapping[str, Any] | None = None,
    ) -> None:
        self.root = Path(root)
        self.entry = entry
        self.bundle = bundle
        self.class_weights = class_weights
        self.spec = None
        self.tokenizer = None
        self.model_revision = ""
        self.tokenizer_revision = ""
        self.optimizer_steps = 0
        self.started = time.perf_counter()
        self._device_report_written = False

    def release_runtime(self) -> None:
        """Drop tokenizer/runtime references before the next independent component."""
        self.tokenizer = None
        self.spec = None
        self.model_revision = ""
        self.tokenizer_revision = ""
        self._device_report_written = False

    def _load_runtime(self, component: str) -> torch.nn.Module:
        family = str(self.entry.backbone or "phobert_base")
        cache = read_family_status(self.root, family, "cache")
        snapshot = resolve_local_snapshot(self.root, cache.get("local_path"))
        if not snapshot:
            raise RuntimeBlocked(f"Phase 15 local snapshot is unavailable for {family}")
        hardware = validate_hardware(self.root)
        if hardware.get("status") != "PASS":
            raise RuntimeBlocked("validated CUDA runtime is unavailable for the production component job")
        selected_device = hardware.get("selected_device_index")
        model, self.spec = build_production_component_model(
            family,
            component,
            local_snapshot=snapshot,
            execution_mode="production",
            selected_device=selected_device,
        )
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
        if isinstance(logits, torch.Tensor):
            return logits
        if component in logits:
            return logits[component]
        if component in PRAGMATIC_LABELS and f"pragmatic_{component}" in logits:
            return logits[f"pragmatic_{component}"]
        raise RuntimeBlocked(f"component model did not return a {component} output")

    @staticmethod
    def _pad_encoded(encoded: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor]:
        if not encoded:
            raise ValueError("component batch cannot be empty")
        max_length = max(int(item[0].numel()) for item in encoded)
        input_ids = torch.zeros((len(encoded), max_length), dtype=torch.long)
        attention = torch.zeros_like(input_ids)
        for index, (ids, mask) in enumerate(encoded):
            ids = ids.reshape(-1).to(dtype=torch.long)
            mask = mask.reshape(-1).to(dtype=torch.long)
            length = min(max_length, int(ids.numel()))
            input_ids[index, :length] = ids[:length]
            attention[index, :length] = mask[:length]
        return {"input_ids": input_ids, "attention_mask": attention}

    def _encode_batch(self, examples: Sequence[Any]) -> dict[str, torch.Tensor]:
        if self.tokenizer is None:
            raise RuntimeBlocked("component tokenizer is not initialized")
        return self._pad_encoded([_encode_text(self.tokenizer, example.text) for example in examples])

    def _resolved_class_weights(self) -> dict[str, Any]:
        weights = self.class_weights
        if weights is None:
            weights = compute_train_only_class_weights(
                self.bundle.train,
                dataset_hash=str(self.bundle.fingerprint),
                code_commit=git_commit(self.root),
            )
        payload = weights.as_dict() if isinstance(weights, ClassWeightBundle) else dict(weights)
        if payload.get("source_split") != "train":
            raise ValueError("component class weights must be computed from the frozen train split")
        required = {"pragmatic_pos_weight", "polarity_weight", "emotion_weight"}
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"component class weights are missing required fields: {missing}")
        return payload

    def _device_report(self, model: torch.nn.Module, device: torch.device, batch: Mapping[str, Any], loss: torch.Tensor) -> None:
        if self._device_report_written:
            return
        backbone = getattr(model, "backbone", model)
        contract = getattr(backbone, "_vipragsent_qlora_contract", {})
        report = assert_runtime_device_contract(
            model,
            device,
            model_family=str(getattr(self.entry, "backbone", "unknown")),
            quantized=bool(getattr(model, "_vipragsent_quantized", False) or getattr(backbone, "_vipragsent_quantized", False)),
            device_map=contract.get("device_map") if isinstance(contract, Mapping) else None,
            batch=batch,
            loss=loss,
            require_lora=bool(getattr(model, "_vipragsent_quantized", False) or getattr(backbone, "_vipragsent_quantized", False)),
        )
        component_report = Path(self._active_component_root) / "training/device_report.json"
        write_device_report(component_report, report)
        self._device_report_written = True

    def _loss(
        self,
        model: torch.nn.Module,
        component: str,
        examples: Sequence[Any],
        device: torch.device,
        weights: Mapping[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        batch = move_batch_to_device(self._encode_batch(examples), device)
        logits = self._task_output(model, component, batch["input_ids"], batch["attention_mask"])
        if logits.device != device:
            raise RuntimeBlocked(f"component logits are on {logits.device}, expected {device}")
        if component in PRAGMATIC_LABELS:
            target = torch.tensor([float(example.labels[component]) for example in examples], dtype=torch.float32, device=device)
            pos_weight = torch.tensor(float(weights["pragmatic_pos_weight"][component]), dtype=logits.dtype, device=device)
            loss = F.binary_cross_entropy_with_logits(logits.reshape(-1), target, pos_weight=pos_weight)
        else:
            labels = POLARITY_LABELS if component == "polarity" else EMOTION_LABELS
            target = torch.tensor([labels.index(str(example.labels[component])) for example in examples], dtype=torch.long, device=device)
            class_weight = weights["polarity_weight" if component == "polarity" else "emotion_weight"]
            weight = torch.tensor([float(class_weight[label]) for label in labels], dtype=logits.dtype, device=device)
            loss = F.cross_entropy(logits.reshape(len(examples), -1), target, weight=weight)
        return loss, batch

    def _predict_split(
        self,
        model: torch.nn.Module,
        component: str,
        examples: Sequence[Any],
        device: torch.device,
        *,
        threshold: float | None,
        physical_batch_size: int,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        labels = POLARITY_LABELS if component == "polarity" else EMOTION_LABELS
        for start in range(0, len(examples), physical_batch_size):
            batch_examples = examples[start : start + physical_batch_size]
            batch = move_batch_to_device(self._encode_batch(batch_examples), device)
            with torch.no_grad():
                logits = self._task_output(model, component, batch["input_ids"], batch["attention_mask"])
            if logits.device != device:
                raise RuntimeBlocked(f"component logits are on {logits.device}, expected {device}")
            if component in PRAGMATIC_LABELS:
                probabilities = logits.reshape(-1).sigmoid().detach().cpu().tolist()
                for example, probability in zip(batch_examples, probabilities, strict=True):
                    value = float(probability)
                    rows.append({"sample_id": example.sample_id, "gold": {component: int(example.labels[component])}, "predictions": {component: int(value >= (threshold if threshold is not None else 0.5))}, "probabilities": {component: value}, "invalid_status": False})
            else:
                probabilities = torch.softmax(logits.reshape(len(batch_examples), -1), dim=-1).detach().cpu().tolist()
                for example, values in zip(batch_examples, probabilities, strict=True):
                    index = int(torch.tensor(values).argmax().item())
                    rows.append({"sample_id": example.sample_id, "gold": {component: example.labels[component]}, "predictions": {component: labels[index]}, "probabilities": {component: [float(value) for value in values]}, "invalid_status": False})
        expected = [str(example.sample_id) for example in examples]
        observed = [str(row["sample_id"]) for row in rows]
        if observed != expected:
            raise RuntimeBlocked(f"component predictions are not aligned to the {len(expected)}-example split")
        return rows

    def _selection(self, component: str, rows: Sequence[Mapping[str, Any]]) -> tuple[float | str, float, str]:
        if not rows:
            raise ValueError("component dev split cannot be empty")
        if component in PRAGMATIC_LABELS:
            threshold = tune_binary_threshold(
                [int(row["gold"][component]) for row in rows],
                [float(row["probabilities"][component]) for row in rows],
            )
            for row in rows:
                row["predictions"][component] = int(float(row["probabilities"][component]) >= threshold)  # type: ignore[index]
            metric = binary_macro_f1([int(row["gold"][component]) for row in rows], [int(row["predictions"][component]) for row in rows])
            return threshold, float(metric), "dev_binary_macro_f1"
        labels = POLARITY_LABELS if component == "polarity" else EMOTION_LABELS
        metric = multiclass_macro_f1(
            [str(row["gold"][component]) for row in rows],
            [str(row["predictions"][component]) for row in rows],
            labels,
        )
        return "NOT_APPLICABLE", float(metric), f"dev_{component}_macro_f1"

    @staticmethod
    def _checkpoint_payload(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        *,
        epoch: int,
        best_metric: float,
        best_epoch: int,
        config_hash: str,
        data_hash: str,
        sample_hashes: Mapping[str, str],
        component: str = "unknown",
        seed: int | None = None,
        model_revision: str = "",
        tokenizer_revision: str = "",
        class_weights_hash: str = "",
    ) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss_aggregator_state_dict": {},
            "run_state": {"epoch": epoch, "best_metric": best_metric, "best_epoch": best_epoch},
            "rng_state": {"torch": torch.get_rng_state()},
            "metadata": {
                "checkpoint_schema": "vipragsent.component.v2",
                "component": component,
                "seed": seed,
                "config_hash": config_hash,
                "data_hash": data_hash,
                "sample_hashes": dict(sample_hashes),
                "model_revision": model_revision,
                "tokenizer_revision": tokenizer_revision,
                "class_weights_hash": class_weights_hash,
            },
        }

    @staticmethod
    def _load_checkpoint(model: torch.nn.Module, path: Path) -> dict[str, int]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("model_state_dict") if isinstance(payload, Mapping) else None
        if not isinstance(state, Mapping) or not state:
            raise RuntimeError(f"component checkpoint has no model_state_dict: {path}")
        expected = set(model.state_dict())
        actual = set(state)
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if missing or unexpected:
            raise RuntimeError(f"component checkpoint key mismatch; missing={missing}, unexpected={unexpected}")
        model.load_state_dict(state, strict=True)
        return {"expected_keys": len(expected), "matched_keys": len(expected), "missing_keys": 0, "unexpected_keys": 0}

    def __call__(self, component: str, model: torch.nn.Module, component_root: Path) -> dict[str, Any]:
        self.started = time.perf_counter()
        self._active_component_root = Path(component_root)
        self._device_report_written = False
        execution_spec = resolve_execution_spec(self.root, self.entry.system_id)
        resolved = resolve_training_config(self.entry, execution_spec, root=self.root, runtime_status=read_family_status(self.root, str(self.entry.backbone), "batch"))
        weights = self._resolved_class_weights()
        class_weights_hash = str(weights.get("content_hash") or sha256_json(weights))
        atomic_write_json(component_root / "training/class_weights.json", weights)
        sample_hashes = {
            "train_order_sha256": sha256_json([str(example.sample_id) for example in self.bundle.train]),
            "dev_order_sha256": sha256_json([str(example.sample_id) for example in self.bundle.dev]),
            "test_order_sha256": sha256_json([str(example.sample_id) for example in self.bundle.test]),
        }
        device = resolve_model_input_device(model)
        optimizer, optimizer_summary = build_optimizer(model, optimizer_name=resolved.optimizer, learning_rate=resolved.learning_rate, weight_decay=resolved.weight_decay)
        train_batches = [self.bundle.train[index : index + resolved.physical_batch_size] for index in range(0, len(self.bundle.train), resolved.physical_batch_size)]
        updates_per_epoch = math.ceil(len(train_batches) / resolved.gradient_accumulation_steps)
        total_steps = updates_per_epoch * resolved.maximum_epochs
        scheduler, scheduler_summary = build_scheduler(optimizer, scheduler_name=resolved.scheduler, warmup_ratio=resolved.warmup_ratio, total_steps=total_steps)
        scheduler_summary.update({"steps_per_epoch": updates_per_epoch, "gradient_accumulation_steps": resolved.gradient_accumulation_steps, "epochs": resolved.maximum_epochs})
        checkpoint_root = component_root / "engine_checkpoints"
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        best_path = checkpoint_root / "best.pt"
        latest_path = checkpoint_root / "latest.pt"
        history: list[dict[str, Any]] = []
        best_metric = float("-inf")
        best_loss = float("inf")
        best_epoch = 0
        no_improvement = 0
        total_optimizer_steps = 0
        total_examples = 0
        best_threshold: float | str = "NOT_APPLICABLE"
        selection_metric_name = ""
        for epoch in range(1, resolved.maximum_epochs + 1):
            model.train()
            epoch_started = time.perf_counter()
            epoch_losses: list[float] = []
            epoch_optimizer_steps = 0
            for start in range(0, len(train_batches), resolved.gradient_accumulation_steps):
                window = train_batches[start : start + resolved.gradient_accumulation_steps]
                optimizer.zero_grad(set_to_none=True)
                window_loss = 0.0
                for examples in window:
                    loss, batch = self._loss(model, component, examples, device, weights)
                    if not self._device_report_written:
                        self._device_report(model, device, batch, loss)
                    (loss / len(window)).backward()
                    value = float(loss.detach().cpu())
                    epoch_losses.append(value)
                    window_loss += value
                    total_examples += len(examples)
                clip_grad_norm_(list(model.parameters()), resolved.gradient_clipping)
                optimizer.step()
                scheduler.step()
                total_optimizer_steps += 1
                epoch_optimizer_steps += 1
            model.eval()
            dev_rows = self._predict_split(model, component, self.bundle.dev, device, threshold=None, physical_batch_size=resolved.physical_batch_size)
            threshold, dev_metric, selection_metric_name = self._selection(component, dev_rows)
            for row in dev_rows:
                if component in PRAGMATIC_LABELS:
                    row["predictions"][component] = int(float(row["probabilities"][component]) >= float(threshold))
            with torch.no_grad():
                dev_loss_values = [self._loss(model, component, self.bundle.dev[index : index + resolved.physical_batch_size], device, weights)[0].detach().cpu().item() for index in range(0, len(self.bundle.dev), resolved.physical_batch_size)]
            dev_loss = float(sum(dev_loss_values) / len(dev_loss_values))
            record = {
                "epoch": epoch,
                "train_loss": float(sum(epoch_losses) / len(epoch_losses)),
                "dev_loss": dev_loss,
                "dev_metric": dev_metric,
                "selection_metric": selection_metric_name,
                "threshold": threshold,
                "optimizer_steps": epoch_optimizer_steps,
                "examples_seen": len(self.bundle.train),
                "seconds": time.perf_counter() - epoch_started,
            }
            history.append(record)
            latest_payload = self._checkpoint_payload(
                model,
                optimizer,
                scheduler,
                epoch=epoch,
                best_metric=best_metric,
                best_epoch=best_epoch,
                config_hash=str(resolved.config_hash),
                data_hash=str(self.bundle.fingerprint),
                sample_hashes=sample_hashes,
                component=component,
                seed=int(self.entry.seed),
                model_revision=self.model_revision,
                tokenizer_revision=self.tokenizer_revision,
                class_weights_hash=class_weights_hash,
            )
            torch.save(latest_payload, latest_path)
            improved = dev_metric > best_metric + resolved.minimum_delta or (abs(dev_metric - best_metric) <= resolved.minimum_delta and dev_loss < best_loss)
            if improved:
                best_metric = dev_metric
                best_loss = dev_loss
                best_epoch = epoch
                best_threshold = threshold
                no_improvement = 0
                best_payload = self._checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    epoch=epoch,
                    best_metric=best_metric,
                    best_epoch=best_epoch,
                    config_hash=str(resolved.config_hash),
                    data_hash=str(self.bundle.fingerprint),
                    sample_hashes=sample_hashes,
                    component=component,
                    seed=int(self.entry.seed),
                    model_revision=self.model_revision,
                    tokenizer_revision=self.tokenizer_revision,
                    class_weights_hash=class_weights_hash,
                )
                torch.save(best_payload, best_path)
            else:
                no_improvement += 1
            if no_improvement >= resolved.patience:
                break
        if not best_path.exists():
            raise RuntimeError("component training completed without a selected checkpoint")
        load_report = self._load_checkpoint(model, best_path)
        atomic_write_json(component_root / "training/checkpoint_load_report.json", load_report)
        atomic_write_json(component_root / "training/history.json", history)
        atomic_write_json(component_root / "training/optimizer_summary.json", optimizer_summary)
        atomic_write_json(component_root / "training/scheduler_summary.json", scheduler_summary)
        atomic_write_json(component_root / "selection/freeze_manifest.json", {"frozen": True, "best_epoch": best_epoch, "selection_metric": best_metric, "threshold": best_threshold, "checkpoint_sha256": sha256_file(best_path), "checkpoint_schema_version": 2, "config_hash": resolved.config_hash, "data_hash": self.bundle.fingerprint, "sample_hashes": sample_hashes, "class_weights_hash": class_weights_hash, "model_revision": self.model_revision, "tokenizer_revision": self.tokenizer_revision})
        model.eval()
        dev_rows = self._predict_split(model, component, self.bundle.dev, device, threshold=float(best_threshold) if component in PRAGMATIC_LABELS else None, physical_batch_size=resolved.physical_batch_size)
        test_rows = self._predict_split(model, component, self.bundle.test, device, threshold=float(best_threshold) if component in PRAGMATIC_LABELS else None, physical_batch_size=resolved.physical_batch_size)
        atomic_write_json(component_root / "training/run_summary.json", {"component": component, "train_examples": len(self.bundle.train), "dev_examples": len(self.bundle.dev), "test_examples": len(self.bundle.test), "actual_epochs": len(history), "maximum_epochs": resolved.maximum_epochs, "best_epoch": best_epoch, "optimizer_steps": total_optimizer_steps, "examples_seen": total_examples, "early_stopping": len(history) < resolved.maximum_epochs})
        return {"dev_rows": dev_rows, "test_rows": test_rows, "history": history, "best_epoch": best_epoch, "actual_epochs": len(history), "best_dev_metric": best_metric, "dev_metric": best_metric, "selection_metric_name": selection_metric_name, "cost_gpu_hours": max(0.0, (time.perf_counter() - self.started) / 3600.0), "best_checkpoint_path": best_path, "latest_checkpoint_path": latest_path, "model_revision": self.model_revision, "tokenizer_revision": self.tokenizer_revision, "optimizer_steps": total_optimizer_steps, "examples_seen": total_examples, "checkpoint_sha256": sha256_file(best_path), "checkpoint_schema_version": 2, "checkpoint_load_report": load_report, "checkpoint_metadata": {"component": component, "seed": int(self.entry.seed), "config_hash": str(resolved.config_hash), "data_hash": str(self.bundle.fingerprint), "sample_hashes": sample_hashes, "class_weights_hash": class_weights_hash}, "threshold": best_threshold, "optimizer_summary": optimizer_summary, "scheduler_summary": scheduler_summary, "class_weights": weights, "class_weights_hash": class_weights_hash, "device_report_path": str(component_root / "training/device_report.json"), "sample_hashes": sample_hashes}
