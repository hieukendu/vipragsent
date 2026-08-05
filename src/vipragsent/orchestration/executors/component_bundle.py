from __future__ import annotations

import gc
import json
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import torch

from ...atomic import atomic_write_json, atomic_write_text
from ...constants import EMOTION_LABELS, POLARITY_LABELS, PRAGMATIC_LABELS
from ...evaluation.metrics import binary_macro_f1, macro_pragmatic_f1, multiclass_macro_f1
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
        dev_sample_ids: Iterable[str] | None = None,
        test_sample_ids: Iterable[str] | None = None,
        sample_ids: Iterable[str] | None = None,
        seed: int,
        config_hash: str,
        data_hash: str,
        model_hash: str,
        model_loader: Callable[[str], Any] | None = None,
        component_runner: Callable[[str, Any, Path], Mapping[str, Any]] | None = None,
        allow_synthetic: bool = True,
    ) -> None:
        self.root = Path(root)
        self.component_names = tuple(component_names)
        if sample_ids is not None and dev_sample_ids is None and test_sample_ids is None:
            dev_sample_ids = sample_ids
            test_sample_ids = sample_ids
        if dev_sample_ids is None or test_sample_ids is None:
            raise ValueError("component bundles require separate dev_sample_ids and test_sample_ids")
        self.dev_sample_ids = tuple(str(value) for value in dev_sample_ids)
        self.test_sample_ids = tuple(str(value) for value in test_sample_ids)
        for name, values in (("dev", self.dev_sample_ids), ("test", self.test_sample_ids)):
            if not values or len(values) != len(set(values)):
                raise ValueError(f"{name} component sample IDs must be non-empty and unique")
        self.sample_ids = self.test_sample_ids  # compatibility for old fixture callers
        self.seed = int(seed)
        self.config_hash = str(config_hash)
        self.data_hash = str(data_hash)
        self.model_hash = str(model_hash)
        self.model_loader = model_loader
        self.component_runner = component_runner
        self.allow_synthetic = bool(allow_synthetic)
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
            "dev_sample_count": len(self.dev_sample_ids),
            "test_sample_count": len(self.test_sample_ids),
            "dev_order_sha256": sha256_json(list(self.dev_sample_ids)),
            "test_order_sha256": sha256_json(list(self.test_sample_ids)),
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
        for key, expected in (("seed", self.seed), ("config_hash", self.config_hash), ("data_hash", self.data_hash), ("model_hash", self.model_hash), ("dev_order_sha256", sha256_json(list(self.dev_sample_ids))), ("test_order_sha256", sha256_json(list(self.test_sample_ids)))):
            if state.get(key) != expected:
                raise ValueError(f"component resume {key} mismatch")
        return state

    @staticmethod
    def _required(component_root: Path) -> tuple[Path, ...]:
        return (
            component_root / "training/history.json",
            component_root / "training/history.csv",
            component_root / "selection/freeze_manifest.json",
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
        if record.get("checkpoint_sha256") != sha256_file(component_root / "checkpoints/best/model.pt"):
            return False
        artifact_hashes = record.get("artifact_hashes", {})
        if not artifact_hashes:
            return False
        for name, expected_hash in artifact_hashes.items():
            if name == "checksums.sha256":
                continue
            path = component_root / name
            if not path.exists() or sha256_file(path) != expected_hash:
                return False
        return all(
            (component_root / name).exists()
            for name in ("predictions/dev_predictions.jsonl", "predictions/test_predictions.jsonl")
        )

    def _synthetic_rows(self, component: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        def rows(split: str) -> list[dict[str, Any]]:
            result = []
            sample_ids = self.dev_sample_ids if split == "dev" else self.test_sample_ids
            for index, sample_id in enumerate(sample_ids):
                probability = float(((index + len(component)) % 7 + 1) / 8.0)
                result.append({"sample_id": sample_id, "split": split, "gold": {component: int(index % 2)}, "predictions": {component: int(probability >= 0.5)}, "probabilities": {component: probability}})
            return result

        return rows("dev"), rows("test")

    def _write_component(self, component: str, result: Mapping[str, Any] | None) -> dict[str, Any]:
        component_root = self.components_root / component
        component_root.mkdir(parents=True, exist_ok=True)
        if result is None and not self.allow_synthetic:
            raise RuntimeError(f"production component {component} did not return a real engine result")
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
        selected_threshold = (result or {}).get("threshold", 0.5 if component in SIX_COMPONENTS else "NOT_APPLICABLE")
        best_epoch = int((result or {}).get("best_epoch", history[-1].get("epoch", 1)))
        selection_metric = float((result or {}).get("best_dev_metric", (result or {}).get("dev_metric", history[-1].get("dev_metric", 0.0))))
        selection_metric_name = str((result or {}).get("selection_metric_name", "dev_component_metric"))
        checkpoint_schema_version = int((result or {}).get("checkpoint_schema_version", 1))
        if [str(row.get("sample_id")) for row in dev_rows] != list(self.dev_sample_ids) or [str(row.get("sample_id")) for row in test_rows] != list(self.test_sample_ids):
            raise ValueError(f"component {component} predictions are not in the frozen sample order")
        _write_jsonl(component_root / "predictions/dev_predictions.jsonl", dev_rows)
        _write_jsonl(component_root / "predictions/test_predictions.jsonl", test_rows)
        atomic_write_json(component_root / "training/history.json", history)
        atomic_write_text(component_root / "training/history.csv", "epoch,train_loss,dev_metric\n" + "\n".join(f"{row.get('epoch', 1)},{row.get('train_loss', 0.0)},{row.get('dev_metric', 0.0)}" for row in history) + "\n")
        if result is not None:
            for name in ("optimizer_summary", "scheduler_summary", "class_weights", "checkpoint_load_report", "checkpoint_metadata"):
                if name in result:
                    atomic_write_json(component_root / f"training/{name}.json", result[name])
        def metric_payload(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
            is_binary = component in SIX_COMPONENTS
            gold = [int(row.get("gold", {}).get(component, 0)) for row in rows] if is_binary else []
            predictions = [int(row.get("predictions", {}).get(component, 0)) for row in rows] if is_binary else []
            if is_binary:
                metric = float(binary_macro_f1(gold, predictions))
            else:
                labels = POLARITY_LABELS if component == "polarity" else EMOTION_LABELS
                metric = multiclass_macro_f1(
                    [str(row.get("gold", {}).get(component, "")) for row in rows],
                    [str(row.get("predictions", {}).get(component, "")) for row in rows],
                    labels,
                )
            return {
                "status": "PASS",
                "component": component,
                "split": split,
                "metric_name": "binary_macro_f1" if is_binary else "task_macro_f1",
                "metric": metric,
                "prediction_count": len(rows),
                "invalid_count": sum(bool(row.get("invalid_status", False)) for row in rows),
                "threshold": selected_threshold if is_binary else "NOT_APPLICABLE",
                "threshold_applicability": "APPLICABLE" if is_binary else "NOT_APPLICABLE",
                "threshold_reason": "dev threshold is defined for pragmatic binary components" if is_binary else "polarity/emotion uses multiclass macro-F1; binary threshold is prohibited",
            }
        atomic_write_json(component_root / "metrics/dev_metrics.json", metric_payload(dev_rows, "dev"))
        atomic_write_json(component_root / "metrics/test_metrics.json", metric_payload(test_rows, "test"))
        dev_metric = selection_metric
        atomic_write_json(component_root / "selection/selection_metric.json", {"component": component, "name": selection_metric_name, "value": dev_metric, "split": "dev", "threshold": selected_threshold})
        atomic_write_json(component_root / "selection/threshold.json", {"component": component, "threshold": selected_threshold if component in SIX_COMPONENTS else "NOT_APPLICABLE", "applicability": "APPLICABLE" if component in SIX_COMPONENTS else "NOT_APPLICABLE", "reason": "pragmatic binary threshold tuned on dev" if component in SIX_COMPONENTS else "multiclass component has no binary threshold"})
        for location in (component_root / "checkpoints/best", component_root / "checkpoints/latest"):
            location.mkdir(parents=True, exist_ok=True)
        if result is None:
            torch.save({"component": component, "seed": self.seed, "config_hash": self.config_hash, "synthetic_results": True}, component_root / "checkpoints/best/model.pt")
            torch.save({"component": component, "seed": self.seed, "config_hash": self.config_hash, "latest": True, "synthetic_results": True}, component_root / "checkpoints/latest/model.pt")
        else:
            best_source = result.get("best_checkpoint_path") or result.get("checkpoint_path")
            latest_source = result.get("latest_checkpoint_path") or best_source
            if not best_source or not latest_source:
                raise ValueError(f"component {component} production result did not return actual checkpoint paths")
            best_path = Path(str(best_source))
            latest_path = Path(str(latest_source))
            if not best_path.exists() or not latest_path.exists():
                raise FileNotFoundError(f"component {component} returned missing checkpoint path")
            import shutil

            shutil.copy2(best_path, component_root / "checkpoints/best/model.pt")
            shutil.copy2(latest_path, component_root / "checkpoints/latest/model.pt")
            if checkpoint_schema_version >= 2:
                for checkpoint_path in (component_root / "checkpoints/best/model.pt", component_root / "checkpoints/latest/model.pt"):
                    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                    if not isinstance(payload, Mapping) or payload.get("schema_version") != checkpoint_schema_version or not isinstance(payload.get("model_state_dict"), Mapping) or not payload["model_state_dict"]:
                        raise ValueError(f"component {component} checkpoint is not a populated schema-v{checkpoint_schema_version} checkpoint: {checkpoint_path}")
        checkpoint_hash = sha256_file(component_root / "checkpoints/best/model.pt")
        model_revision = str((result or {}).get("model_revision", "fixture"))
        tokenizer_revision = str((result or {}).get("tokenizer_revision", "fixture"))
        selection_metric = dev_metric
        atomic_write_json(component_root / "selection/best_checkpoint.json", {"component": component, "path": "checkpoints/best/model.pt", "best_epoch": best_epoch, "checkpoint_sha256": checkpoint_hash})
        atomic_write_json(component_root / "selection/freeze_manifest.json", {"frozen": True, "component": component, "best_epoch": best_epoch, "selection_metric": selection_metric, "selection_metric_name": selection_metric_name, "threshold": selected_threshold, "checkpoint_schema_version": checkpoint_schema_version, "checkpoint_sha256": checkpoint_hash, "config_hash": self.config_hash, "dataset_hash": self.data_hash})
        atomic_write_json(component_root / "checkpoints/checkpoint_manifest.json", {"status": "PASS", "schema_version": checkpoint_schema_version, "component": component, "seed": self.seed, "model_revision": model_revision, "tokenizer_revision": tokenizer_revision, "config_hash": self.config_hash, "dataset_hash": self.data_hash, "best_epoch": best_epoch, "selection_metric": selection_metric, "selection_metric_name": selection_metric_name, "checkpoint_path": "checkpoints/best/model.pt", "checkpoint_sha256": checkpoint_hash, "checkpoint_metadata": (result or {}).get("checkpoint_metadata", {}), "sample_hashes": (result or {}).get("sample_hashes", {}), "class_weights_hash": (result or {}).get("class_weights_hash", "NOT_APPLICABLE"), "synthetic_results": result is None})
        required = self._required(component_root) + (component_root / "checkpoints/checkpoint_manifest.json",)
        optional = []
        if result is not None and "device_report_path" in result:
            device_report = Path(str(result["device_report_path"]))
            if not device_report.exists():
                raise FileNotFoundError(f"component {component} returned missing device report: {device_report}")
            optional.append(device_report)
        records = {path.relative_to(component_root).as_posix(): sha256_file(path) for path in (*required, *optional) if path.name != "checksums.sha256"}
        atomic_write_text(component_root / "checksums.sha256", "".join(f"{digest}  {name}\n" for name, digest in sorted(records.items())))
        return {"status": "PASS", "checkpoint_sha256": checkpoint_hash, "checkpoint_schema_version": checkpoint_schema_version, "cost_gpu_hours": cost, "artifact_hashes": records, "prediction_sha256": {name: records[name] for name in ("predictions/dev_predictions.jsonl", "predictions/test_predictions.jsonl")}, "best_epoch": best_epoch, "selection_metric": selection_metric, "selection_metric_name": selection_metric_name, "model_revision": model_revision, "tokenizer_revision": tokenizer_revision, "sample_hashes": (result or {}).get("sample_hashes", {}), "class_weights_hash": (result or {}).get("class_weights_hash", "NOT_APPLICABLE"), "synthetic_results": result is None}

    def _release_runtime_resources(self) -> None:
        owners: list[Any] = []
        for callback in (self.model_loader, self.component_runner):
            owner = getattr(callback, "__self__", callback)
            if owner is not None and all(owner is not existing for existing in owners):
                owners.append(owner)
        for owner in owners:
            release = getattr(owner, "release_runtime", None)
            if callable(release):
                release()

    def _combine(self) -> None:
        combined: dict[str, dict[str, Any]] = {sample_id: {"sample_id": sample_id, "gold": {}, "predictions": {}, "probabilities": {}} for sample_id in self.test_sample_ids}
        for component in self.component_names:
            rows = _read_jsonl(self.components_root / component / "predictions/test_predictions.jsonl")
            if [str(row.get("sample_id")) for row in rows] != list(self.test_sample_ids):
                raise ValueError(f"component {component} cannot be combined because sample IDs are misaligned")
            for row in rows:
                target = combined[str(row["sample_id"])]
                target["gold"].update(row.get("gold", {}))
                target["predictions"].update(row.get("predictions", {}))
                target["probabilities"].update(row.get("probabilities", {}))
        _write_jsonl(self.root / "predictions/test_predictions.jsonl", combined.values())
        dev_combined = {sample_id: {"sample_id": sample_id, "gold": {}, "predictions": {}, "probabilities": {}} for sample_id in self.dev_sample_ids}
        for component in self.component_names:
            for row in _read_jsonl(self.components_root / component / "predictions/dev_predictions.jsonl"):
                target = dev_combined[str(row["sample_id"])]
                target["gold"].update(row.get("gold", {}))
                target["predictions"].update(row.get("predictions", {}))
                target["probabilities"].update(row.get("probabilities", {}))
        _write_jsonl(self.root / "predictions/dev_predictions.jsonl", dev_combined.values())
        def combined_metrics(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
            true = {label: [int(row.get("gold", {}).get(label, 0)) for row in rows] for label in PRAGMATIC_LABELS}
            pred = {label: [int(row.get("predictions", {}).get(label, 0)) for row in rows] for label in PRAGMATIC_LABELS}
            payload = {"status": "PASS", "split": split, "component_count": len(self.component_names), "prediction_count": len(rows), "per_label_f1": {label: binary_macro_f1(true[label], pred[label]) for label in PRAGMATIC_LABELS}, "macro_pragmatic_f1": macro_pragmatic_f1(true, pred), "invalid_count": sum(bool(row.get("invalid_status", False)) for row in rows), "component_names": list(self.component_names)}
            if "polarity" in self.component_names:
                payload["polarity_macro_f1"] = multiclass_macro_f1(
                    [str(row.get("gold", {}).get("polarity", "")) for row in rows],
                    [str(row.get("predictions", {}).get("polarity", "")) for row in rows],
                    POLARITY_LABELS,
                )
            if "emotion" in self.component_names:
                payload["emotion_macro_f1"] = multiclass_macro_f1(
                    [str(row.get("gold", {}).get("emotion", "")) for row in rows],
                    [str(row.get("predictions", {}).get("emotion", "")) for row in rows],
                    EMOTION_LABELS,
                )
            return payload
        atomic_write_json(self.root / "metrics/dev_metrics.json", combined_metrics(list(dev_combined.values()), "dev"))
        atomic_write_json(self.root / "metrics/test_metrics.json", combined_metrics(list(combined.values()), "test"))

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
                output["cost_gpu_hours"] = measured_cost
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
                self._release_runtime_resources()
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        self._combine()
        state["status"] = "PASS"
        component_hashes = {component: state["components"][component]["checkpoint_sha256"] for component in self.component_names}
        manifest = {"status": "PASS", "schema_version": 2, "component_names": list(self.component_names), "component_count": len(self.component_names), "component_checkpoint_sha256": component_hashes, "component_checkpoint_schema_versions": {component: state["components"][component].get("checkpoint_schema_version", 1) for component in self.component_names}, "component_best_epochs": {component: state["components"][component].get("best_epoch") for component in self.component_names}, "component_selection_metrics": {component: state["components"][component].get("selection_metric") for component in self.component_names}, "cost_gpu_hours": state["cost_gpu_hours"], "total_measured_gpu_hours": state["cost_gpu_hours"], "seed": self.seed, "config_hash": self.config_hash, "data_hash": self.data_hash, "model_hash": self.model_hash, "dev_sample_count": len(self.dev_sample_ids), "test_sample_count": len(self.test_sample_ids), "dev_order_sha256": sha256_json(list(self.dev_sample_ids)), "test_order_sha256": sha256_json(list(self.test_sample_ids)), "combined_prediction_order_sha256": sha256_json({"dev": list(self.dev_sample_ids), "test": list(self.test_sample_ids)}), "resume_status": "RESUMABLE" if resume else "NEW"}
        atomic_write_json(self.manifest_path, manifest)
        atomic_write_json(self.state_path, state)
        self._append_event("bundle_completed", cost_gpu_hours=state["cost_gpu_hours"])
        return manifest


def run_component_bundle(root: str | Path, *, executor_kind: str, dev_sample_ids: Iterable[str] | None = None, test_sample_ids: Iterable[str] | None = None, sample_ids: Iterable[str] | None = None, seed: int, config_hash: str, data_hash: str, model_hash: str, resume: bool = False, model_loader: Callable[[str], Any] | None = None, component_runner: Callable[[str, Any, Path], Mapping[str, Any]] | None = None, allow_synthetic: bool = True) -> dict[str, Any]:
    executor = ComponentBundleExecutor(root, component_names=component_names_for_executor(executor_kind), dev_sample_ids=dev_sample_ids, test_sample_ids=test_sample_ids, sample_ids=sample_ids, seed=seed, config_hash=config_hash, data_hash=data_hash, model_hash=model_hash, model_loader=model_loader, component_runner=component_runner, allow_synthetic=allow_synthetic)
    return executor.run(resume=resume)
