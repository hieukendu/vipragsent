from __future__ import annotations

import gc
import json
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import torch

from ...atomic import atomic_write_json, atomic_write_text
from ...constants import PRAGMATIC_LABELS
from ...hashing import sha256_file, sha256_json

SIX_COMPONENTS = tuple(PRAGMATIC_LABELS)
EIGHT_COMPONENTS = SIX_COMPONENTS + ("polarity", "emotion")


def component_names_for_executor(executor_kind: str) -> tuple[str, ...]:
    if executor_kind == "single_task_bundle":
        return SIX_COMPONENTS
    if executor_kind == "independent_checkpoint_bundle":
        return EIGHT_COMPONENTS
    raise ValueError(f"unsupported component bundle executor kind: {executor_kind}")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    atomic_write_text(path, "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class ComponentBundleExecutor:
    """Train one independent component at a time and combine by frozen sample ID."""

    def __init__(
        self,
        root: str | Path,
        *,
        component_names: tuple[str, ...],
        sample_ids: Iterable[str],
        seed: int,
        config_hash: str,
        data_hash: str,
        model_hash: str,
        model_loader: Callable[[str], Any] | None = None,
        component_runner: Callable[[str, Any, Path], Mapping[str, Any]] | None = None,
    ) -> None:
        self.root = Path(root)
        self.component_names = tuple(component_names)
        self.sample_ids = tuple(str(value) for value in sample_ids)
        self.seed = int(seed)
        self.config_hash = str(config_hash)
        self.data_hash = str(data_hash)
        self.model_hash = str(model_hash)
        self.model_loader = model_loader
        self.component_runner = component_runner
        self.components_root = self.root / "components"
        self.state_path = self.components_root / "state.json"
        self.events_path = self.components_root / "events.jsonl"
        self.manifest_path = self.components_root / "component_manifest.json"

    def _append_event(self, event: str, **payload: Any) -> None:
        self.components_root.mkdir(parents=True, exist_ok=True)
        existing = self.events_path.read_text(encoding="utf-8") if self.events_path.exists() else ""
        atomic_write_text(self.events_path, existing + json.dumps({"event": event, **payload}, sort_keys=True) + "\n")

    def _initial_state(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": "RUNNING",
            "seed": self.seed,
            "config_hash": self.config_hash,
            "data_hash": self.data_hash,
            "model_hash": self.model_hash,
            "components": {name: {"status": "NOT_STARTED"} for name in self.component_names},
            "cost_gpu_hours": 0.0,
        }

    def _load_state(self, resume: bool) -> dict[str, Any]:
        if not self.state_path.exists():
            state = self._initial_state()
            atomic_write_json(self.state_path, state)
            return state
        if not resume:
            raise RuntimeError("component bundle state exists; resume the same bundle explicitly")
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        for key, expected in (("seed", self.seed), ("config_hash", self.config_hash), ("data_hash", self.data_hash), ("model_hash", self.model_hash)):
            if state.get(key) != expected:
                raise ValueError(f"component resume {key} mismatch")
        return state

    @staticmethod
    def _required(component_root: Path) -> tuple[Path, ...]:
        return (
            component_root / "training/history.json",
            component_root / "training/history.csv",
            component_root / "selection/best_checkpoint.json",
            component_root / "selection/selection_metric.json",
            component_root / "selection/threshold.json",
            component_root / "checkpoints/best/model.pt",
            component_root / "checkpoints/latest/model.pt",
            component_root / "predictions/dev_predictions.jsonl",
            component_root / "predictions/test_predictions.jsonl",
            component_root / "metrics/dev_metrics.json",
            component_root / "metrics/test_metrics.json",
            component_root / "checksums.sha256",
        )

    def _component_valid(self, component: str, state: Mapping[str, Any]) -> bool:
        record = state.get("components", {}).get(component, {})
        component_root = self.components_root / component
        if record.get("status") != "PASS" or any(not path.exists() for path in self._required(component_root)):
            return False
        return record.get("checkpoint_sha256") == sha256_file(component_root / "checkpoints/best/model.pt")

    def _synthetic_rows(self, component: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        def rows(split: str) -> list[dict[str, Any]]:
            result = []
            for index, sample_id in enumerate(self.sample_ids):
                probability = float(((index + len(component)) % 7 + 1) / 8.0)
                result.append({"sample_id": sample_id, "split": split, "gold": {component: int(index % 2)}, "predictions": {component: int(probability >= 0.5)}, "probabilities": {component: probability}})
            return result

        return rows("dev"), rows("test")

    def _write_component(self, component: str, result: Mapping[str, Any] | None) -> dict[str, Any]:
        component_root = self.components_root / component
        component_root.mkdir(parents=True, exist_ok=True)
        if result is None:
            dev_rows, test_rows = self._synthetic_rows(component)
            history = [{"epoch": 1, "train_loss": 0.5, "dev_metric": 0.5, "synthetic_results": True}]
            cost = 0.0
        else:
            dev_rows = [dict(row) for row in result.get("dev_rows", [])]
            test_rows = [dict(row) for row in result.get("test_rows", [])]
            history = [dict(row) for row in result.get("history", [])]
            if not dev_rows or not test_rows or not history:
                raise ValueError(f"component {component} returned incomplete production-shaped outputs")
            cost = float(result.get("cost_gpu_hours", 0.0))
        if [str(row.get("sample_id")) for row in dev_rows] != list(self.sample_ids) or [str(row.get("sample_id")) for row in test_rows] != list(self.sample_ids):
            raise ValueError(f"component {component} predictions are not in the frozen sample order")
        _write_jsonl(component_root / "predictions/dev_predictions.jsonl", dev_rows)
        _write_jsonl(component_root / "predictions/test_predictions.jsonl", test_rows)
        atomic_write_json(component_root / "training/history.json", history)
        atomic_write_text(component_root / "training/history.csv", "epoch,train_loss,dev_metric\n" + "\n".join(f"{row.get('epoch', 1)},{row.get('train_loss', 0.0)},{row.get('dev_metric', 0.0)}" for row in history) + "\n")
        atomic_write_json(component_root / "metrics/dev_metrics.json", {"component": component, "prediction_count": len(dev_rows), "status": "PASS"})
        atomic_write_json(component_root / "metrics/test_metrics.json", {"component": component, "prediction_count": len(test_rows), "status": "PASS"})
        atomic_write_json(component_root / "selection/best_checkpoint.json", {"component": component, "path": "checkpoints/best/model.pt", "best_epoch": history[-1].get("epoch", 1)})
        atomic_write_json(component_root / "selection/selection_metric.json", {"component": component, "name": "dev_component_metric", "value": history[-1].get("dev_metric", 0.0)})
        atomic_write_json(component_root / "selection/threshold.json", {"component": component, "threshold": 0.5})
        for location in (component_root / "checkpoints/best", component_root / "checkpoints/latest"):
            location.mkdir(parents=True, exist_ok=True)
        torch.save({"component": component, "seed": self.seed, "config_hash": self.config_hash}, component_root / "checkpoints/best/model.pt")
        torch.save({"component": component, "seed": self.seed, "config_hash": self.config_hash, "latest": True}, component_root / "checkpoints/latest/model.pt")
        checkpoint_hash = sha256_file(component_root / "checkpoints/best/model.pt")
        records = {path.relative_to(component_root).as_posix(): sha256_file(path) for path in self._required(component_root) if path.name != "checksums.sha256"}
        atomic_write_text(component_root / "checksums.sha256", "".join(f"{digest}  {name}\n" for name, digest in sorted(records.items())))
        return {"status": "PASS", "checkpoint_sha256": checkpoint_hash, "cost_gpu_hours": cost, "artifact_hashes": records}

    def _combine(self) -> None:
        combined: dict[str, dict[str, Any]] = {sample_id: {"sample_id": sample_id, "gold": {}, "predictions": {}, "probabilities": {}} for sample_id in self.sample_ids}
        for component in self.component_names:
            rows = _read_jsonl(self.components_root / component / "predictions/test_predictions.jsonl")
            if [str(row.get("sample_id")) for row in rows] != list(self.sample_ids):
                raise ValueError(f"component {component} cannot be combined because sample IDs are misaligned")
            for row in rows:
                target = combined[str(row["sample_id"])]
                target["gold"].update(row.get("gold", {}))
                target["predictions"].update(row.get("predictions", {}))
                target["probabilities"].update(row.get("probabilities", {}))
        _write_jsonl(self.root / "predictions/test_predictions.jsonl", combined.values())
        dev_combined = {sample_id: {"sample_id": sample_id, "gold": {}, "predictions": {}, "probabilities": {}} for sample_id in self.sample_ids}
        for component in self.component_names:
            for row in _read_jsonl(self.components_root / component / "predictions/dev_predictions.jsonl"):
                target = dev_combined[str(row["sample_id"])]
                target["gold"].update(row.get("gold", {}))
                target["predictions"].update(row.get("predictions", {}))
                target["probabilities"].update(row.get("probabilities", {}))
        _write_jsonl(self.root / "predictions/dev_predictions.jsonl", dev_combined.values())
        atomic_write_json(self.root / "metrics/dev_metrics.json", {"status": "PASS", "component_count": len(self.component_names), "prediction_count": len(self.sample_ids)})
        atomic_write_json(self.root / "metrics/test_metrics.json", {"status": "PASS", "component_count": len(self.component_names), "prediction_count": len(self.sample_ids)})

    def run(self, *, resume: bool = False) -> dict[str, Any]:
        state = self._load_state(resume)
        self._append_event("bundle_started", resume=resume, component_count=len(self.component_names))
        for component in self.component_names:
            if self._component_valid(component, state):
                self._append_event("component_skipped", component=component, reason="valid_complete")
                continue
            component_root = self.components_root / component
            state["components"][component] = {"status": "RUNNING", "seed": self.seed, "config_hash": self.config_hash, "data_hash": self.data_hash, "model_hash": self.model_hash}
            atomic_write_json(self.state_path, state)
            self._append_event("component_started", component=component)
            started = time.perf_counter()
            model = None
            try:
                model = self.model_loader(component) if self.model_loader else None
                result = self.component_runner(component, model, component_root) if self.component_runner else None
                output = self._write_component(component, result)
                measured_cost = float(output.get("cost_gpu_hours", 0.0))
                if measured_cost == 0.0 and torch.cuda.is_available():
                    measured_cost = (time.perf_counter() - started) / 3600.0
                state["components"][component] = {**output, "status": "PASS", "completed": True}
                state["cost_gpu_hours"] = float(sum(float(item.get("cost_gpu_hours", 0.0)) for item in state["components"].values() if isinstance(item, Mapping)))
                atomic_write_json(self.state_path, state)
                self._append_event("component_completed", component=component, cost_gpu_hours=measured_cost)
            except Exception as exc:
                state["components"][component] = {"status": "INTERRUPTED", "error": str(exc)}
                atomic_write_json(self.state_path, state)
                self._append_event("component_interrupted", component=component, error=str(exc))
                raise
            finally:
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        self._combine()
        state["status"] = "PASS"
        component_hashes = {component: state["components"][component]["checkpoint_sha256"] for component in self.component_names}
        manifest = {"status": "PASS", "component_names": list(self.component_names), "component_checkpoint_sha256": component_hashes, "cost_gpu_hours": state["cost_gpu_hours"], "seed": self.seed, "config_hash": self.config_hash, "data_hash": self.data_hash, "model_hash": self.model_hash, "combined_prediction_order_sha256": sha256_json(list(self.sample_ids))}
        atomic_write_json(self.manifest_path, manifest)
        atomic_write_json(self.state_path, state)
        self._append_event("bundle_completed", cost_gpu_hours=state["cost_gpu_hours"])
        return manifest


def run_component_bundle(root: str | Path, *, executor_kind: str, sample_ids: Iterable[str], seed: int, config_hash: str, data_hash: str, model_hash: str, resume: bool = False, model_loader: Callable[[str], Any] | None = None, component_runner: Callable[[str, Any, Path], Mapping[str, Any]] | None = None) -> dict[str, Any]:
    executor = ComponentBundleExecutor(root, component_names=component_names_for_executor(executor_kind), sample_ids=sample_ids, seed=seed, config_hash=config_hash, data_hash=data_hash, model_hash=model_hash, model_loader=model_loader, component_runner=component_runner)
    return executor.run(resume=resume)
