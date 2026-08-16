from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ...atomic import atomic_write_json, atomic_write_text
from ...evaluation.reasoning_judge import (
    ReasoningJudge,
    build_reasoning_prediction_row,
    compute_reasoning_metrics,
    load_reasoning_protocol,
    validate_reasoning_protocol_files,
)
from ...hashing import sha256_file
from ...models.generation import parse_cot_generation_record
from ...runtime.device import (
    assert_runtime_device_contract,
    move_batch_to_model_device,
    resolve_model_input_device,
    write_device_report,
)
from ...training.generation_checkpoint import (
    GenerationCheckpointError,
    is_real_dataset_hash,
    load_generation_checkpoint,
    save_generation_checkpoint,
)


class GenerationProtocolConflict(RuntimeError):
    pass


GENERATION_CHECKPOINT_SCHEMA_VERSION = 2


def generation_optimizer_steps_per_epoch(record_count: int, physical_batch_size: int, gradient_accumulation_steps: int) -> int:
    if record_count < 1 or physical_batch_size < 1 or gradient_accumulation_steps < 1:
        raise ValueError("generation record count, physical batch, and accumulation must be positive")
    micro_batches = (record_count + physical_batch_size - 1) // physical_batch_size
    return (micro_batches + gradient_accumulation_steps - 1) // gradient_accumulation_steps


def teacher_forced_generation_loss(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    target_mask: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute causal token CE from model logits without a classifier fallback."""
    if input_ids.ndim != 2 or target_ids.ndim != 2:
        raise ValueError("generation inputs and targets must be rank-two token tensors")
    if input_ids.size(0) != target_ids.size(0):
        raise ValueError("generation inputs and targets must have the same batch dimension")
    combined = torch.cat((input_ids, target_ids), dim=1)
    labels = torch.full_like(combined, -100)
    labels[:, input_ids.size(1):] = target_ids
    if target_mask is not None:
        if target_mask.shape != target_ids.shape:
            raise ValueError("generation target mask must match target token shape")
        labels[:, input_ids.size(1):] = target_ids.masked_fill(~target_mask.to(dtype=torch.bool), -100)
    model_kwargs: dict[str, Any] = {"input_ids": combined, "labels": labels}
    if attention_mask is not None:
        if attention_mask.shape != input_ids.shape:
            raise ValueError("generation input attention mask must match input token shape")
        target_attention = target_mask.to(dtype=torch.bool) if target_mask is not None else torch.ones_like(target_ids, dtype=torch.bool)
        model_kwargs["attention_mask"] = torch.cat((attention_mask.to(dtype=torch.bool), target_attention), dim=1)
    output = model(**model_kwargs)
    logits = output["logits"] if isinstance(output, Mapping) else getattr(output, "logits", None)
    if logits is None:
        raise ValueError("causal generation model did not return token logits")
    if logits.size(1) != labels.size(1):
        raise ValueError("causal model logits do not align with target tokens")
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100)


def _stable_artifact_identity(
    artifact: Any,
    explicit: Mapping[str, Any] | str | None,
    *,
    config: Any | None = None,
) -> dict[str, Any]:
    """Build a JSON-stable model/tokenizer artifact identity."""
    if isinstance(explicit, Mapping):
        identity = dict(explicit)
    elif explicit is not None:
        identity = {"identity": str(explicit)}
    else:
        source = config if config is not None else artifact
        repository = getattr(source, "_name_or_path", None) or getattr(source, "name_or_path", None)
        init_kwargs = getattr(artifact, "init_kwargs", {})
        if not repository and isinstance(init_kwargs, Mapping):
            repository = init_kwargs.get("name_or_path")
        repository = str(repository or f"{type(artifact).__module__}.{type(artifact).__qualname__}")
        revision = getattr(source, "_commit_hash", None) or getattr(source, "revision", None)
        if not revision and isinstance(init_kwargs, Mapping):
            revision = init_kwargs.get("revision")
        revision = str(revision or "local")
        identity = {
            "identity": f"{repository}@{revision}",
            "repository": repository,
            "revision": revision,
        }
    if not str(identity.get("identity", "")):
        repository = str(identity.get("repository", "local"))
        revision = str(identity.get("revision", "local"))
        identity["identity"] = f"{repository}@{revision}"
    return identity


class GenerationExecutor:
    """Fixture compatibility executor for the pre-resolution parser contract.

    The sequential production registry never dispatches this compatibility class.
    It remains available for old CPU fixtures that explicitly provide the legacy
    parser and delimiter-shaped fake output.
    """

    def __init__(self, root: str | Path, *, model: torch.nn.Module, tokenizer: Any | None = None, parser: Callable[[str], Mapping[str, Any]] = parse_cot_generation_record) -> None:
        self.root = Path(root)
        self.model = model
        self.tokenizer = tokenizer
        self.parser = parser

    def _decode(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        if self.tokenizer is None or not hasattr(self.tokenizer, "decode"):
            raise ValueError("a tokenizer with decode() is required for generation")
        return str(self.tokenizer.decode(value, skip_special_tokens=False))

    def generate_split(self, split: str, records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        generation_rows: list[dict[str, Any]] = []
        prediction_rows: list[dict[str, Any]] = []
        valid = 0
        invalid = 0
        self.model.eval()
        with torch.no_grad():
            for record in records:
                input_ids = record["input_ids"]
                attention_mask = record.get("attention_mask", torch.ones_like(input_ids))
                generated = self.model.generate(input_ids=input_ids, attention_mask=attention_mask)
                text = self._decode(generated[0] if getattr(generated, "ndim", 0) == 2 else generated)
                parsed: dict[str, Any]
                failure_reason: str | None = None
                try:
                    parsed = dict(self.parser(text))
                    if parsed.get("parser_status", parsed.get("status", "PASS")) != "PASS":
                        raise ValueError(str(parsed.get("invalid_reason", "strict parser rejected generation")))
                    valid += 1
                except Exception as exc:
                    parsed = {"labels": {}, "rationale": None}
                    failure_reason = f"{type(exc).__name__}: {exc}"
                    invalid += 1
                generation_rows.append({"sample_id": str(record["sample_id"]), "raw_generation": text, "parse_status": "PASS" if failure_reason is None else "INVALID", "failure_reason": failure_reason})
                prediction_rows.append({"sample_id": str(record["sample_id"]), "gold": dict(record.get("gold", {})), "predictions": dict(parsed.get("labels", {})), "probabilities": {}, "raw_generation": text, "parse_status": "PASS" if failure_reason is None else "INVALID", "failure_reason": failure_reason})
        atomic_write_text(self.root / f"generations/{split}_generations.jsonl", "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in generation_rows))
        atomic_write_text(self.root / f"predictions/{split}_predictions.jsonl", "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in prediction_rows))
        return {"requested": valid + invalid, "valid": valid, "invalid": invalid, "prediction_path": f"predictions/{split}_predictions.jsonl"}

    def train_generation(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        optimizer: torch.optim.Optimizer,
        epochs: int = 1,
        scheduler: Any | None = None,
        gradient_clipping: float | None = None,
        epoch_start: int = 1,
    ) -> list[dict[str, float]]:
        rows = list(records)
        if not rows:
            raise ValueError("generation training requires non-empty records")
        history: list[dict[str, float]] = []
        for epoch in range(epoch_start, epoch_start + epochs):
            self.model.train()
            losses: list[float] = []
            for record in rows:
                loss = teacher_forced_generation_loss(self.model, record["input_ids"], record["target_ids"], target_mask=record.get("target_mask"))
                loss.backward()
                if gradient_clipping is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(gradient_clipping))
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss.detach().cpu()))
            history.append({"epoch": float(epoch), "train_loss": float(sum(losses) / len(losses))})
        atomic_write_json(self.root / "training/history.json", history)
        atomic_write_text(self.root / "training/history.csv", "epoch,train_loss\n" + "\n".join(f"{row['epoch']},{row['train_loss']}" for row in history) + "\n")
        return history

    def run(self, *, dev_records: Iterable[Mapping[str, Any]], test_records: Iterable[Mapping[str, Any]], optimizer: torch.optim.Optimizer | None = None, train_records: Iterable[Mapping[str, Any]] | None = None, epochs: int = 1) -> dict[str, Any]:
        if optimizer is not None and train_records is not None:
            history = self.train_generation(train_records, optimizer=optimizer, epochs=epochs)
        else:
            history = [{"epoch": 0.0, "train_loss": 0.0}]
            atomic_write_json(self.root / "training/history.json", history)
        dev = self.generate_split("dev", dev_records)
        test = self.generate_split("test", test_records)
        parser_report = {"dev": dev, "test": test, "strict_parser": True, "semantic_repair": False}
        atomic_write_json(self.root / "generation/parser_report.json", parser_report)
        atomic_write_json(self.root / "metrics/dev_metrics.json", {"status": "PASS", "valid_generation_count": dev["valid"], "invalid_generation_count": dev["invalid"]})
        atomic_write_json(self.root / "metrics/test_metrics.json", {"status": "PASS", "valid_generation_count": test["valid"], "invalid_generation_count": test["invalid"]})
        atomic_write_json(self.root / "selection/best_checkpoint.json", {"path": "checkpoints/best/model.pt", "best_epoch": history[-1]["epoch"]})
        atomic_write_json(self.root / "selection/selection_metric.json", {"name": "dev_macro_pragmatic_f1", "value": "computed_from_parsed_labels"})
        atomic_write_json(self.root / "selection/thresholds.json", {"source": "strict_parser", "status": "NOT_APPLICABLE"})
        compatibility_provenance = {
            "model": {"class": f"{type(self.model).__module__}.{type(self.model).__qualname__}"},
            "model_artifact": {"identity": f"{type(self.model).__module__}.{type(self.model).__qualname__}@local"},
            "tokenizer_artifact": {"identity": f"{type(self.tokenizer).__module__}.{type(self.tokenizer).__qualname__}@local"},
            "dataset": {"identity": "compatibility", "hash": "NOT_PROVIDED"},
            "data_hash": "NOT_PROVIDED",
            "optimizer": {
                "class": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}" if optimizer is not None else None,
                "param_group_count": len(optimizer.param_groups) if optimizer is not None else 0,
            },
            "scheduler": {"class": None},
            "rng": {"seed": None, "streams": ["python", "numpy", "torch"]},
            "data_order": {"sample_ids": []},
            "config": {"executor": "causal_generation"},
            "model_environment": {"device": str(resolve_model_input_device(self.model))},
        }
        for checkpoint_name in ("best", "latest"):
            save_generation_checkpoint(
                self.root / f"checkpoints/{checkpoint_name}/model.pt",
                self.model,
                optimizer,
                None,
                {"epoch": history[-1]["epoch"], "data_order": []},
                compatibility_provenance,
                metadata={"executor": "causal_generation", "model_class": type(self.model).__name__},
                fixture_mode=True,
            )
        return {"status": "PASS", "dev": dev, "test": test, "checkpoint_sha256": sha256_file(self.root / "checkpoints/best/model.pt")}


def generation_targets_available(root: str | Path = ".") -> bool:
    """Return true when the resolved reasoning protocol is locally complete."""
    root = Path(root)
    try:
        return validate_reasoning_protocol_files(root)["status"] == "PASS"
    except (OSError, ValueError, KeyError):
        return False


def _encode_text(tokenizer: Any, text: str, *, max_length: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(tokenizer, "batch_encode"):
        encoded = tokenizer.batch_encode([text], max_length=max_length)
        return torch.tensor(encoded["input_ids"], dtype=torch.long), torch.tensor(encoded["attention_mask"], dtype=torch.long)
    if callable(tokenizer):
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
        attention_mask = encoded.get("attention_mask") if isinstance(encoded, Mapping) else getattr(encoded, "attention_mask", None)
        input_ids = input_ids if isinstance(input_ids, torch.Tensor) else torch.tensor(input_ids, dtype=torch.long)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        elif not isinstance(attention_mask, torch.Tensor):
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
        return input_ids, attention_mask
    if hasattr(tokenizer, "encode"):
        values = tokenizer.encode(text, max_length=max_length)
        input_ids = torch.tensor([values], dtype=torch.long)
        return input_ids, torch.ones_like(input_ids)
    raise ValueError("a tokenizer with batch_encode(), __call__(), or encode() is required")


def build_cot_training_records(
    root: str | Path,
    examples: Iterable[Mapping[str, Any]],
    *,
    tokenizer: Any,
    max_input_length: int = 128,
    max_target_length: int = 160,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build causal records from the approved rationale artifact only."""
    root = Path(root)
    examples = list(examples)
    protocol = load_reasoning_protocol(root)
    source_path = root / str(protocol["systems"]["cot_only_vistral"]["rationale_source"])
    if not source_path.exists():
        raise RuntimeError(f"approved rationale source is missing: {source_path}")
    source: dict[str, str] = {}
    for line in source_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        sample_id = str(item.get("sample_id", ""))
        rationale = str(item.get("rationale", "")).strip()
        if not sample_id or not rationale:
            raise ValueError("approved rationale source contains an incomplete row")
        if sample_id in source:
            raise ValueError(f"approved rationale source contains duplicate sample ID: {sample_id}")
        source[sample_id] = rationale
    prompt = (root / str(protocol["generation_prompt_path"])).read_text(encoding="utf-8")
    records: list[dict[str, Any]] = []
    skipped: list[str] = []
    for example in examples:
        sample_id = str(example.get("sample_id", ""))
        rationale = source.get(sample_id)
        if not rationale:
            skipped.append(sample_id)
            continue
        input_ids, attention_mask = _encode_text(tokenizer, prompt.replace("{TEXT}", str(example.get("text", ""))), max_length=max_input_length)
        target_ids, target_attention = _encode_text(tokenizer, rationale, max_length=max_target_length)
        records.append({
            "sample_id": sample_id,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_ids": target_ids,
            "target_mask": target_attention.bool(),
            "rationale_source": str(source_path.relative_to(root)).replace("\\", "/"),
        })
    return records, {"requested_count": len(examples), "usable_count": len(records), "skipped_count": len(skipped), "skipped_sample_ids": skipped, "source_path": str(source_path.relative_to(root)).replace("\\", "/"), "source_sha256": sha256_file(source_path)}


class ReasoningGenerationExecutor:
    """Dedicated causal reasoning executor for the resolved CoT baseline."""

    def __init__(
        self,
        root: str | Path,
        *,
        model: torch.nn.Module,
        tokenizer: Any,
        judge: ReasoningJudge,
        run_root: str | Path | None = None,
        seed: int | str | None = None,
        config_hash: str = "NOT_PROVIDED",
        data_hash: str = "NOT_PROVIDED",
        dataset_identity: str | None = None,
        model_artifact_identity: Mapping[str, Any] | str | None = None,
        tokenizer_artifact_identity: Mapping[str, Any] | str | None = None,
        production_provenance_required: bool = False,
        fixture_mode: bool = False,
        physical_batch_size: int = 1,
        gradient_accumulation_steps: int = 1,
        pad_token_id: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.run_root = Path(run_root) if run_root is not None else self.root
        self.model = model
        self.tokenizer = tokenizer
        self.judge = judge
        self.seed = seed
        self.config_hash = config_hash
        self.data_hash = str(data_hash)
        self.dataset_identity = str(dataset_identity or self.data_hash)
        self.production_provenance_required = bool(production_provenance_required)
        self.fixture_mode = bool(fixture_mode)
        model_config = getattr(model, "config", None)
        self.model_artifact_identity = _stable_artifact_identity(
            model,
            model_artifact_identity,
            config=model_config,
        )
        self.tokenizer_artifact_identity = _stable_artifact_identity(tokenizer, tokenizer_artifact_identity)
        if physical_batch_size < 1 or gradient_accumulation_steps < 1:
            raise ValueError("generation physical batch and gradient accumulation must be positive")
        self.physical_batch_size = int(physical_batch_size)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        tokenizer_pad_token_id = getattr(tokenizer, "pad_token_id", None)
        tokenizer_eos_token_id = getattr(tokenizer, "eos_token_id", None)
        self.pad_token_id = int(pad_token_id if pad_token_id is not None else tokenizer_pad_token_id if tokenizer_pad_token_id is not None else tokenizer_eos_token_id if tokenizer_eos_token_id is not None else 0)
        self.device = resolve_model_input_device(model)
        if self.fixture_mode and self.device.type != "cpu":
            raise GenerationCheckpointError("fixture/legacy generation mode requires a CPU model")
        if not is_real_dataset_hash(self.data_hash) and (self.production_provenance_required or not self.fixture_mode):
            raise GenerationCheckpointError(
                "production generation requires a real dataset hash or explicit CPU fixture mode"
            )
        self._device_report_written = False
        self.protocol = load_reasoning_protocol(self.root)
        validation = validate_reasoning_protocol_files(self.root)
        if validation["status"] != "PASS":
            raise ValueError("reasoning protocol is not validated")
        self.run_root.mkdir(parents=True, exist_ok=True)

    def _prepare_device_batch(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        moved = move_batch_to_model_device(batch, self.model, device=self.device)
        if not self._device_report_written:
            report = assert_runtime_device_contract(
                self.model,
                self.device,
                model_family="vistral_7b",
                batch=moved,
            )
            write_device_report(self.run_root / "training/device_report.json", report)
            self._device_report_written = True
        return moved

    def _decode(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        if not hasattr(self.tokenizer, "decode"):
            raise ValueError("a tokenizer with decode() is required")
        return str(self.tokenizer.decode(value, skip_special_tokens=True))

    def _record_inputs(self, record: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        if "input_ids" in record:
            input_ids = record["input_ids"]
            attention_mask = record.get("attention_mask")
            input_ids = input_ids if isinstance(input_ids, torch.Tensor) else torch.tensor(input_ids, dtype=torch.long)
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            attention_mask = attention_mask if isinstance(attention_mask, torch.Tensor) else torch.tensor(attention_mask, dtype=torch.long)
            if attention_mask.ndim == 1:
                attention_mask = attention_mask.unsqueeze(0)
            return input_ids, attention_mask
        return _encode_text(self.tokenizer, str(record.get("prompt_text", record.get("text", ""))))

    def generate_reasoning_split(self, split: str, records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        generation_rows: list[dict[str, Any]] = []
        decoding = self.protocol["decoding"]
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        self.model.eval()
        with torch.no_grad():
            for record in records:
                sample_id = str(record["sample_id"])
                try:
                    input_ids, attention_mask = self._record_inputs(record)
                    batch = self._prepare_device_batch({"input_ids": input_ids, "attention_mask": attention_mask})
                    generated = self.model.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        do_sample=bool(decoding["do_sample"]),
                        num_beams=int(decoding["num_beams"]),
                        max_new_tokens=int(decoding["max_new_tokens"]),
                        repetition_penalty=float(decoding["repetition_penalty"]),
                        num_return_sequences=int(decoding["num_return_sequences"]),
                    )
                    sequence = generated[0] if getattr(generated, "ndim", 0) == 2 else generated
                    input_length = int(input_ids.shape[-1])
                    continuation = sequence[input_length:] if getattr(sequence, "numel", lambda: 0)() > input_length else sequence
                    text = self._decode(continuation)
                    truncated = bool(eos_id is not None and len(continuation) >= int(decoding["max_new_tokens"]) and eos_id not in continuation.tolist())
                    status = "PASS" if text.strip() else "INVALID"
                    failure_reason = None if status == "PASS" else "empty_reasoning"
                except Exception as exc:
                    text = ""
                    truncated = False
                    status = "INVALID"
                    failure_reason = f"{type(exc).__name__}: {exc}"
                generation_rows.append({
                    "sample_id": sample_id,
                    "split": split,
                    "generated_reasoning": text,
                    "raw_generation": text,
                    "generation_status": status,
                    "failure_reason": failure_reason,
                    "truncated": truncated,
                })
        atomic_write_text(self.run_root / f"reasoning/{split}_reasoning.jsonl", "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in generation_rows))
        return generation_rows

    def judge_reasoning_split(self, split: str, generation_rows: Iterable[Mapping[str, Any]], gold_by_id: Mapping[str, Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        predictions: list[dict[str, Any]] = []
        decisions: list[dict[str, Any]] = []
        for generation in generation_rows:
            sample_id = str(generation["sample_id"])
            if generation.get("generation_status") != "PASS":
                decision = {"valid": False, "labels": None, "raw_response": None, "invalid_stage": "generation", "invalid_reason": generation.get("failure_reason") or "generation_failed", "retry_count": 0, "cache_hit": False}
            else:
                decision = self.judge.judge(str(generation.get("generated_reasoning", "")))
            decisions.append({"sample_id": sample_id, **dict(decision)})
            predictions.append(build_reasoning_prediction_row(sample_id, gold_by_id[sample_id], str(generation.get("generated_reasoning", "")), decision, truncated=bool(generation.get("truncated"))))
        self.judge.write_artifacts(self.run_root, split, predictions, decisions)
        atomic_write_text(self.run_root / f"predictions/{split}_predictions.jsonl", "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in predictions))
        return predictions, decisions

    def compute_split_metrics(self, split: str, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        metrics = compute_reasoning_metrics(rows, diagnostics=self.judge.diagnostics)
        metrics.update({"status": "PASS", "split": split, "judge_protocol_id": self.judge.judge_protocol_id, "judge_prompt_hash": self.judge.prompt_hash, "judge_schema_hash": self.judge.schema_hash, "generation_protocol_id": self.protocol["protocol_version"]})
        atomic_write_json(self.run_root / f"metrics/{split}_reasoning_metrics.json", metrics)
        return metrics

    @staticmethod
    def _sequence(value: Any, *, dtype: torch.dtype) -> torch.Tensor:
        tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=dtype)
        tensor = tensor.to(dtype=dtype)
        if tensor.ndim == 2 and tensor.size(0) == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim != 1:
            raise ValueError("generation training records must contain one-dimensional token rows")
        return tensor

    def _collate_training_records(self, records: list[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        if not records:
            raise ValueError("cannot collate an empty generation training batch")
        input_rows = [self._sequence(record["input_ids"], dtype=torch.long) for record in records]
        input_masks = [
            self._sequence(record["attention_mask"], dtype=torch.long) if "attention_mask" in record else torch.ones_like(input_row, dtype=torch.long)
            for record, input_row in zip(records, input_rows, strict=True)
        ]
        target_rows = [self._sequence(record["target_ids"], dtype=torch.long) for record in records]
        target_masks = [
            self._sequence(record["target_mask"], dtype=torch.bool) if "target_mask" in record else torch.ones_like(target_row, dtype=torch.bool)
            for record, target_row in zip(records, target_rows, strict=True)
        ]
        max_input = max(row.numel() for row in input_rows)
        max_target = max(row.numel() for row in target_rows)
        input_ids = torch.full((len(records), max_input), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(records), max_input), dtype=torch.long)
        target_ids = torch.full((len(records), max_target), self.pad_token_id, dtype=torch.long)
        target_mask = torch.zeros((len(records), max_target), dtype=torch.bool)
        for index, (input_row, input_mask, target_row, target_row_mask) in enumerate(zip(input_rows, input_masks, target_rows, target_masks, strict=True)):
            input_length = input_row.numel()
            target_length = target_row.numel()
            if input_mask.numel() != input_length or target_row_mask.numel() != target_length:
                raise ValueError("generation token and mask lengths must match")
            input_ids[index, :input_length] = input_row
            attention_mask[index, :input_length] = input_mask
            target_ids[index, :target_length] = target_row
            target_mask[index, :target_length] = target_row_mask
        return {"input_ids": input_ids, "attention_mask": attention_mask, "target_ids": target_ids, "target_mask": target_mask}

    def train_generation(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        optimizer: torch.optim.Optimizer,
        epochs: int = 1,
        scheduler: Any | None = None,
        gradient_clipping: float | None = None,
        epoch_start: int = 1,
    ) -> list[dict[str, float]]:
        usable = [dict(record) for record in records if record.get("input_ids") is not None and record.get("target_ids") is not None]
        if not usable:
            raise ValueError("generation training requires approved non-empty rationale records")
        self._last_data_order = self._record_order(usable)
        batches = [
            self._collate_training_records(usable[start:start + self.physical_batch_size])
            for start in range(0, len(usable), self.physical_batch_size)
        ]
        history: list[dict[str, float]] = []
        for epoch in range(epoch_start, epoch_start + epochs):
            self.model.train()
            losses: list[float] = []
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps = 0
            for start in range(0, len(batches), self.gradient_accumulation_steps):
                window = batches[start:start + self.gradient_accumulation_steps]
                window_loss = 0.0
                for raw_batch in window:
                    batch = self._prepare_device_batch(raw_batch)
                    loss = teacher_forced_generation_loss(
                        self.model,
                        batch["input_ids"],
                        batch["target_ids"],
                        target_mask=batch["target_mask"],
                        attention_mask=batch["attention_mask"],
                    )
                    (loss / len(window)).backward()
                    window_loss += float(loss.detach().cpu())
                if gradient_clipping is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(gradient_clipping))
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                losses.append(window_loss / len(window))
            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": float(sum(losses) / len(losses)),
                    "optimizer_steps": float(optimizer_steps),
                    "micro_batches": float(len(batches)),
                    "physical_batch_size": float(self.physical_batch_size),
                    "gradient_accumulation_steps": float(self.gradient_accumulation_steps),
                }
            )
        atomic_write_json(self.run_root / "training/history.json", history)
        atomic_write_text(self.run_root / "training/history.csv", "epoch,train_loss,optimizer_steps\n" + "\n".join(f"{row['epoch']},{row['train_loss']},{row['optimizer_steps']}" for row in history) + "\n")
        return history

    @staticmethod
    def _record_order(records: Sequence[Mapping[str, Any]]) -> list[str]:
        return [str(record.get("sample_id", f"record-{index}")) for index, record in enumerate(records)]

    def _checkpoint_provenance(
        self,
        *,
        optimizer: torch.optim.Optimizer | None,
        scheduler: Any | None,
        data_order: Sequence[str],
    ) -> dict[str, Any]:
        config = getattr(self.model, "config", None)
        model_identity = {
            "class": f"{type(self.model).__module__}.{type(self.model).__qualname__}",
            "model_type": str(getattr(config, "model_type", type(self.model).__name__)),
            "revision": str(getattr(config, "_commit_hash", "local")),
        }
        optimizer_identity = {
            "class": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}" if optimizer is not None else None,
            "param_group_count": len(optimizer.param_groups) if optimizer is not None else 0,
        }
        scheduler_identity = {
            "class": f"{type(scheduler).__module__}.{type(scheduler).__qualname__}" if scheduler is not None else None,
        }
        try:
            dtype = str(next(self.model.parameters()).dtype).replace("torch.", "")
        except StopIteration:
            dtype = "unknown"
        return {
            "model": model_identity,
            "model_artifact": dict(self.model_artifact_identity),
            "tokenizer_artifact": dict(self.tokenizer_artifact_identity),
            "dataset": {"identity": self.dataset_identity, "hash": self.data_hash},
            "data_hash": self.data_hash,
            "optimizer": optimizer_identity,
            "scheduler": scheduler_identity,
            "rng": {
                "seed": self.seed,
                "streams": ["python", "numpy", "torch"],
            },
            "data_order": {"sample_ids": list(data_order)},
            "config": {
                "config_hash": self.config_hash,
                "generation_prompt_hash": self.protocol["generation_prompt_hash"],
                "physical_batch_size": self.physical_batch_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
            },
            "model_environment": {
                "device": str(self.device),
                "dtype": dtype,
            },
        }

    def write_checkpoint(
        self,
        relative_path: str = "checkpoints/latest/model.pt",
        *,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        epoch: int | None = None,
        selection_metric: float | None = None,
        metadata: Mapping[str, Any] | None = None,
        data_order: Sequence[str] | None = None,
    ) -> str:
        """Write one canonical model/state file and its fail-closed sidecar."""
        path = self.run_root / relative_path
        order = [str(item) for item in (getattr(self, "_last_data_order", ()) if data_order is None else data_order)]
        provenance = self._checkpoint_provenance(optimizer=optimizer, scheduler=scheduler, data_order=order)
        run_state = {
            "epoch": epoch,
            "selection_metric": selection_metric,
            "data_order": order,
        }
        manifest = save_generation_checkpoint(
            path,
            self.model,
            optimizer,
            scheduler,
            run_state,
            provenance,
            metadata={
                "executor_kind": "generation_trainable",
                "seed": self.seed,
                "generation_protocol_id": self.protocol["protocol_version"],
                "generation_prompt_hash": self.protocol["generation_prompt_hash"],
                "config_hash": self.config_hash,
                "data_hash": self.data_hash,
                **dict(metadata or {}),
            },
            production_provenance_required=self.production_provenance_required,
            fixture_mode=self.fixture_mode,
        )
        return manifest.checkpoint_sha256

    def _write_checkpoint_load_report(self, report: Mapping[str, Any]) -> None:
        atomic_write_json(self.run_root / "checkpoints/load_report.json", dict(report))

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        expected_sha256: str | None = None,
        allow_legacy_fixture: bool = False,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        expected_data_order: Sequence[str] | None = None,
        restore_training_state: bool = True,
    ) -> dict[str, Any]:
        """Load a canonical generation checkpoint with exact resume identity checks."""
        checkpoint_path = Path(path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = self.run_root / checkpoint_path
        if not checkpoint_path.exists():
            raise GenerationCheckpointError(f"generation checkpoint is missing: {checkpoint_path}")
        observed_hash = sha256_file(checkpoint_path)
        if expected_sha256 and observed_hash != str(expected_sha256):
            raise GenerationCheckpointError(
                f"generation checkpoint hash mismatch: expected {expected_sha256}, observed {observed_hash}"
            )
        expected_provenance = None
        expected_provenance = self._checkpoint_provenance(
            optimizer=optimizer,
            scheduler=scheduler,
            data_order=[str(item) for item in (expected_data_order or ())],
        )
        compared_fields = [
            "model",
            "model_artifact",
            "tokenizer_artifact",
            "dataset",
            "data_hash",
            "rng",
            "config",
            "model_environment",
        ]
        if expected_data_order is not None:
            compared_fields.append("data_order")
        if optimizer is not None:
            compared_fields.append("optimizer")
        if scheduler is not None:
            compared_fields.append("scheduler")
        loaded = load_generation_checkpoint(
            checkpoint_path,
            self.model,
            expected_provenance=expected_provenance,
            compare_provenance_fields=compared_fields,
            optimizer=optimizer,
            scheduler=scheduler,
            allow_legacy_fixture=allow_legacy_fixture,
            restore_training_state=restore_training_state,
            report_path=self.run_root / "checkpoints/load_report.json",
            production_provenance_required=self.production_provenance_required,
            fixture_mode=self.fixture_mode,
        )
        report = loaded.checkpoint.report.as_dict()
        report.update({
            "checkpoint_sha256": observed_hash,
            "legacy_fixture_migration": loaded.checkpoint.report.legacy_compatibility,
            "run_state": dict(loaded.run_state),
            "provenance": loaded.manifest.provenance if loaded.manifest is not None else None,
        })
        return report

    def load_epoch_checkpoint(
        self,
        epoch: int,
        *,
        expected_sha256: str | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        expected_data_order: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        return self.load_checkpoint(
            f"checkpoints/epoch_{int(epoch)}/model.pt",
            expected_sha256=expected_sha256,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_data_order=expected_data_order,
        )

    def load_selected_checkpoint(
        self,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        expected_data_order: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        selection_path = self.run_root / "selection/best_checkpoint.json"
        if not selection_path.exists():
            raise GenerationCheckpointError("selected generation checkpoint manifest is missing")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        checkpoint_name = selection.get("path") or selection.get("checkpoint_path")
        expected_hash = selection.get("sha256") or selection.get("checkpoint_sha256")
        if not checkpoint_name:
            raise GenerationCheckpointError("selected generation checkpoint path is missing")
        return self.load_checkpoint(
            checkpoint_name,
            expected_sha256=expected_hash,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_data_order=expected_data_order,
        )

    def load_frozen_checkpoint(
        self,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        expected_data_order: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        freeze_path = self.run_root / "selection/freeze_manifest.json"
        if not freeze_path.exists():
            raise GenerationCheckpointError("generation test execution requires a selection freeze manifest")
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if freeze.get("frozen") is not True:
            raise GenerationCheckpointError("generation selection freeze manifest is not frozen")
        checkpoint = freeze.get("checkpoint") if isinstance(freeze.get("checkpoint"), Mapping) else freeze
        checkpoint_name = checkpoint.get("path") or checkpoint.get("checkpoint_path")
        expected_hash = checkpoint.get("sha256") or checkpoint.get("checkpoint_sha256")
        if not checkpoint_name or not expected_hash:
            raise GenerationCheckpointError("frozen generation checkpoint path and hash are required")
        return self.load_checkpoint(
            checkpoint_name,
            expected_sha256=expected_hash,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_data_order=expected_data_order,
        )

    def write_checkpoint_manifest(
        self,
        *,
        best_path: str = "checkpoints/best/model.pt",
        latest_path: str = "checkpoints/latest/model.pt",
        best_epoch: int,
        selection_metric: float,
        rationale_source_hash: str = "NOT_PROVIDED",
    ) -> dict[str, Any]:
        best = self.run_root / best_path
        latest = self.run_root / latest_path
        if not best.exists() or not latest.exists():
            raise GenerationCheckpointError("generation checkpoint manifest requires best and latest checkpoints")
        manifest = {
            "status": "PASS",
            "schema_version": GENERATION_CHECKPOINT_SCHEMA_VERSION,
            "executor_kind": "generation_trainable",
            "best": best_path.replace("\\", "/"),
            "latest": latest_path.replace("\\", "/"),
            "best_checkpoint_sha256": sha256_file(best),
            "latest_checkpoint_sha256": sha256_file(latest),
            "checkpoint_sha256": sha256_file(best),
            "best_epoch": int(best_epoch),
            "selection_metric": float(selection_metric),
            "generation_prompt_hash": self.protocol["generation_prompt_hash"],
            "config_hash": self.config_hash,
            "data_hash": self.data_hash,
            "rationale_source_hash": rationale_source_hash,
        }
        atomic_write_json(self.run_root / "checkpoints/checkpoint_manifest.json", manifest)
        return manifest

    def run_cot(
        self,
        *,
        train_records: Iterable[Mapping[str, Any]],
        dev_records: Iterable[Mapping[str, Any]],
        test_records: Iterable[Mapping[str, Any]],
        optimizer: torch.optim.Optimizer,
        epochs: int = 1,
        scheduler: Any | None = None,
        resume_from: str | Path | None = None,
    ) -> dict[str, Any]:
        train_rows = list(train_records)
        dev_rows = list(dev_records)
        test_rows = list(test_records)
        train_order = self._record_order(train_rows)
        gold_dev = {str(row["sample_id"]): row["gold"] for row in dev_rows}
        gold_test = {str(row["sample_id"]): row["gold"] for row in test_rows}
        best_metric = float("-inf")
        best_epoch = 0
        best_hash = ""
        all_history: list[dict[str, float]] = []
        start_epoch = 1
        if resume_from is not None:
            resumed = self.load_checkpoint(
                resume_from,
                optimizer=optimizer,
                scheduler=scheduler,
                expected_data_order=train_order,
            )
            state = resumed["run_state"]
            start_epoch = int(state.get("epoch") or 0) + 1
            best_epoch = int(state.get("epoch") or 0)
            best_metric = float(state.get("selection_metric") if state.get("selection_metric") is not None else best_metric)
            best_path = self.run_root / "checkpoints/best/model.pt"
            best_hash = sha256_file(best_path) if best_path.exists() else sha256_file(Path(resume_from) if Path(resume_from).is_absolute() else self.run_root / Path(resume_from))
        for epoch in range(start_epoch, epochs + 1):
            history = self.train_generation(train_rows, optimizer=optimizer, scheduler=scheduler, epochs=1, epoch_start=epoch)
            all_history.extend(history)
            generated_dev = self.generate_reasoning_split("dev", dev_rows)
            dev_predictions, _ = self.judge_reasoning_split("dev", generated_dev, gold_dev)
            dev_metrics = self.compute_split_metrics("dev", dev_predictions)
            checkpoint_hash = self.write_checkpoint(
                f"checkpoints/epoch_{epoch}/model.pt",
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                selection_metric=float(dev_metrics["primary_macro_f1"]),
                data_order=train_order,
            )
            if float(dev_metrics["primary_macro_f1"]) > best_metric:
                best_metric = float(dev_metrics["primary_macro_f1"])
                best_epoch = epoch
                best_hash = checkpoint_hash
                self.write_checkpoint(
                    "checkpoints/best/model.pt",
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    selection_metric=best_metric,
                    data_order=train_order,
                )
            atomic_write_json(self.run_root / f"metrics/dev_reasoning_metrics_epoch_{epoch}.json", dev_metrics)
        if not best_hash:
            best_hash = self.write_checkpoint(
                "checkpoints/best/model.pt",
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=best_epoch,
                selection_metric=best_metric if best_metric != float("-inf") else None,
                data_order=train_order,
            )
        atomic_write_json(self.run_root / "training/history.json", all_history)
        atomic_write_json(self.run_root / "selection/best_checkpoint.json", {"status": "PASS", "best_epoch": best_epoch, "selection_metric": "full_split_macro_pragmatic_f1_all_zero_fallback_dev", "value": best_metric, "checkpoint_path": "checkpoints/best/model.pt", "checkpoint_sha256": best_hash})
        self.load_checkpoint(
            "checkpoints/best/model.pt",
            expected_sha256=best_hash,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_data_order=train_order,
        )
        latest_hash = self.write_checkpoint(
            "checkpoints/latest/model.pt",
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=best_epoch,
            selection_metric=best_metric,
            data_order=train_order,
        )
        manifest = self.write_checkpoint_manifest(best_epoch=best_epoch, selection_metric=best_metric)
        atomic_write_json(self.run_root / "selection/freeze_manifest.json", {"frozen": True, "checkpoint": {"path": "checkpoints/best/model.pt", "sha256": best_hash}, "best_epoch": best_epoch, "selection_metric": best_metric, "test_access": False, "checkpoint_manifest_sha256": sha256_file(self.run_root / "checkpoints/checkpoint_manifest.json")})
        generated_test = self.generate_reasoning_split("test", test_rows)
        test_predictions, _ = self.judge_reasoning_split("test", generated_test, gold_test)
        test_metrics = self.compute_split_metrics("test", test_predictions)
        return {"status": "PASS", "best_epoch": best_epoch, "best_dev_metric": best_metric, "checkpoint_sha256": best_hash, "latest_checkpoint_sha256": latest_hash, "checkpoint_manifest": manifest, "dev_metrics": dev_metrics, "test_metrics": test_metrics}


ProductionGenerationExecutor = ReasoningGenerationExecutor
