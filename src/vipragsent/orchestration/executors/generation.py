from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
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


class GenerationProtocolConflict(RuntimeError):
    pass


class GenerationCheckpointError(RuntimeError):
    """Raised when a generation checkpoint cannot be trusted or loaded."""


GENERATION_CHECKPOINT_SCHEMA_VERSION = 2


def teacher_forced_generation_loss(model: torch.nn.Module, input_ids: torch.Tensor, target_ids: torch.Tensor, *, target_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Compute causal token CE from model logits without a classifier fallback."""
    if input_ids.ndim != 2 or target_ids.ndim != 2:
        raise ValueError("generation inputs and targets must be rank-two token tensors")
    combined = torch.cat((input_ids, target_ids), dim=1)
    labels = torch.full_like(combined, -100)
    labels[:, input_ids.size(1):] = target_ids
    if target_mask is not None:
        labels[:, input_ids.size(1):] = target_ids.masked_fill(~target_mask.to(dtype=torch.bool), -100)
    output = model(input_ids=combined, labels=labels)
    logits = output["logits"] if isinstance(output, Mapping) else getattr(output, "logits", None)
    if logits is None:
        raise ValueError("causal generation model did not return token logits")
    if logits.size(1) != labels.size(1):
        raise ValueError("causal model logits do not align with target tokens")
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100)


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
        (self.root / "checkpoints/best").mkdir(parents=True, exist_ok=True)
        (self.root / "checkpoints/latest").mkdir(parents=True, exist_ok=True)
        torch.save({"executor": "causal_generation", "model_class": type(self.model).__name__}, self.root / "checkpoints/best/model.pt")
        torch.save({"executor": "causal_generation", "model_class": type(self.model).__name__}, self.root / "checkpoints/latest/model.pt")
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
    ) -> None:
        self.root = Path(root)
        self.run_root = Path(run_root) if run_root is not None else self.root
        self.model = model
        self.tokenizer = tokenizer
        self.judge = judge
        self.seed = seed
        self.config_hash = config_hash
        self.data_hash = data_hash
        self.protocol = load_reasoning_protocol(self.root)
        validation = validate_reasoning_protocol_files(self.root)
        if validation["status"] != "PASS":
            raise ValueError("reasoning protocol is not validated")
        self.run_root.mkdir(parents=True, exist_ok=True)

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
                    generated = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
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
        history: list[dict[str, float]] = []
        for epoch in range(epoch_start, epoch_start + epochs):
            self.model.train()
            losses: list[float] = []
            for record in usable:
                loss = teacher_forced_generation_loss(self.model, record["input_ids"], record["target_ids"], target_mask=record.get("target_mask"))
                loss.backward()
                if gradient_clipping is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(gradient_clipping))
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss.detach().cpu()))
            history.append({"epoch": float(epoch), "train_loss": float(sum(losses) / len(losses)), "optimizer_steps": float(len(losses))})
        atomic_write_json(self.run_root / "training/history.json", history)
        atomic_write_text(self.run_root / "training/history.csv", "epoch,train_loss,optimizer_steps\n" + "\n".join(f"{row['epoch']},{row['train_loss']},{row['optimizer_steps']}" for row in history) + "\n")
        return history

    def write_checkpoint(
        self,
        relative_path: str = "checkpoints/latest/model.pt",
        *,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        epoch: int | None = None,
        selection_metric: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Write the canonical v2 generation checkpoint payload."""
        path = self.run_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": GENERATION_CHECKPOINT_SCHEMA_VERSION,
            "model_state_dict": dict(self.model.state_dict()),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else {},
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None and hasattr(scheduler, "state_dict") else {},
            "loss_aggregator_state_dict": {},
            "run_state": {"epoch": epoch, "selection_metric": selection_metric},
            "rng_state": {"torch": torch.get_rng_state()},
            "metadata": {
                "executor_kind": "generation_trainable",
                "seed": self.seed,
                "generation_protocol_id": self.protocol["protocol_version"],
                "generation_prompt_hash": self.protocol["generation_prompt_hash"],
                "config_hash": self.config_hash,
                "data_hash": self.data_hash,
                **dict(metadata or {}),
            },
        }
        torch.save(payload, path)
        return sha256_file(path)

    def _write_checkpoint_load_report(self, report: Mapping[str, Any]) -> None:
        atomic_write_json(self.run_root / "checkpoints/load_report.json", dict(report))

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        expected_sha256: str | None = None,
        allow_legacy_fixture: bool = False,
    ) -> dict[str, Any]:
        """Load a v2 checkpoint and reject silent or total key mismatches."""
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
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(payload, Mapping):
            raise GenerationCheckpointError("generation checkpoint payload is not a mapping")
        schema_version = payload.get("schema_version")
        legacy_migration = False
        if schema_version == GENERATION_CHECKPOINT_SCHEMA_VERSION:
            state = payload.get("model_state_dict")
        elif allow_legacy_fixture and schema_version in (None, 1):
            state = payload.get("model_state_dict") or payload.get("model")
            legacy_migration = True
        else:
            raise GenerationCheckpointError(
                f"unsupported generation checkpoint schema: {schema_version!r}; production requires schema 2"
            )
        if not isinstance(state, Mapping) or not state:
            raise GenerationCheckpointError("generation checkpoint has no model state")
        state = dict(state)
        model_keys = set(self.model.state_dict())
        state_keys = set(state)
        matched = model_keys & state_keys
        missing = sorted(model_keys - state_keys)
        unexpected = sorted(state_keys - model_keys)
        match_ratio = len(matched) / len(model_keys) if model_keys else 0.0
        report = {
            "status": "PASS" if not missing and not unexpected and matched else "FAIL",
            "schema_version": schema_version if schema_version is not None else 1,
            "legacy_fixture_migration": legacy_migration,
            "path": str(checkpoint_path),
            "checkpoint_sha256": observed_hash,
            "matched_count": len(matched),
            "missing_count": len(missing),
            "unexpected_count": len(unexpected),
            "match_ratio": match_ratio,
            "missing_keys": missing,
            "unexpected_keys": unexpected,
        }
        self._write_checkpoint_load_report(report)
        if not matched:
            raise GenerationCheckpointError("generation checkpoint has no matching model parameters")
        if missing or unexpected:
            raise GenerationCheckpointError(
                f"generation checkpoint key mismatch: missing={missing}, unexpected={unexpected}"
            )
        try:
            self.model.load_state_dict(state, strict=True)
        except (RuntimeError, ValueError) as exc:
            raise GenerationCheckpointError(f"generation checkpoint failed strict load: {exc}") from exc
        return report

    def load_epoch_checkpoint(self, epoch: int, *, expected_sha256: str | None = None) -> dict[str, Any]:
        return self.load_checkpoint(f"checkpoints/epoch_{int(epoch)}/model.pt", expected_sha256=expected_sha256)

    def load_selected_checkpoint(self) -> dict[str, Any]:
        selection_path = self.run_root / "selection/best_checkpoint.json"
        if not selection_path.exists():
            raise GenerationCheckpointError("selected generation checkpoint manifest is missing")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        checkpoint_name = selection.get("path") or selection.get("checkpoint_path")
        expected_hash = selection.get("sha256") or selection.get("checkpoint_sha256")
        if not checkpoint_name:
            raise GenerationCheckpointError("selected generation checkpoint path is missing")
        return self.load_checkpoint(checkpoint_name, expected_sha256=expected_hash)

    def load_frozen_checkpoint(self) -> dict[str, Any]:
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
        return self.load_checkpoint(checkpoint_name, expected_sha256=expected_hash)

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

    def run_cot(self, *, train_records: Iterable[Mapping[str, Any]], dev_records: Iterable[Mapping[str, Any]], test_records: Iterable[Mapping[str, Any]], optimizer: torch.optim.Optimizer, epochs: int = 1) -> dict[str, Any]:
        train_rows = list(train_records)
        dev_rows = list(dev_records)
        test_rows = list(test_records)
        gold_dev = {str(row["sample_id"]): row["gold"] for row in dev_rows}
        gold_test = {str(row["sample_id"]): row["gold"] for row in test_rows}
        best_metric = float("-inf")
        best_epoch = 0
        best_hash = ""
        all_history: list[dict[str, float]] = []
        for epoch in range(1, epochs + 1):
            history = self.train_generation(train_rows, optimizer=optimizer, epochs=1, epoch_start=epoch)
            all_history.extend(history)
            generated_dev = self.generate_reasoning_split("dev", dev_rows)
            dev_predictions, _ = self.judge_reasoning_split("dev", generated_dev, gold_dev)
            dev_metrics = self.compute_split_metrics("dev", dev_predictions)
            checkpoint_hash = self.write_checkpoint(f"checkpoints/epoch_{epoch}/model.pt", optimizer=optimizer, epoch=epoch, selection_metric=float(dev_metrics["primary_macro_f1"]))
            if float(dev_metrics["primary_macro_f1"]) > best_metric:
                best_metric = float(dev_metrics["primary_macro_f1"])
                best_epoch = epoch
                best_hash = checkpoint_hash
                self.write_checkpoint("checkpoints/best/model.pt", optimizer=optimizer, epoch=epoch, selection_metric=best_metric)
            atomic_write_json(self.run_root / f"metrics/dev_reasoning_metrics_epoch_{epoch}.json", dev_metrics)
        atomic_write_json(self.run_root / "training/history.json", all_history)
        atomic_write_json(self.run_root / "selection/best_checkpoint.json", {"status": "PASS", "best_epoch": best_epoch, "selection_metric": "full_split_macro_pragmatic_f1_all_zero_fallback_dev", "value": best_metric, "checkpoint_path": "checkpoints/best/model.pt", "checkpoint_sha256": best_hash})
        self.load_checkpoint("checkpoints/best/model.pt", expected_sha256=best_hash)
        latest_hash = self.write_checkpoint("checkpoints/latest/model.pt", optimizer=optimizer, epoch=best_epoch, selection_metric=best_metric)
        manifest = self.write_checkpoint_manifest(best_epoch=best_epoch, selection_metric=best_metric)
        atomic_write_json(self.run_root / "selection/freeze_manifest.json", {"frozen": True, "checkpoint": {"path": "checkpoints/best/model.pt", "sha256": best_hash}, "best_epoch": best_epoch, "selection_metric": best_metric, "test_access": False, "checkpoint_manifest_sha256": sha256_file(self.run_root / "checkpoints/checkpoint_manifest.json")})
        generated_test = self.generate_reasoning_split("test", test_rows)
        test_predictions, _ = self.judge_reasoning_split("test", generated_test, gold_test)
        test_metrics = self.compute_split_metrics("test", test_predictions)
        return {"status": "PASS", "best_epoch": best_epoch, "best_dev_metric": best_metric, "checkpoint_sha256": best_hash, "latest_checkpoint_sha256": latest_hash, "checkpoint_manifest": manifest, "dev_metrics": dev_metrics, "test_metrics": test_metrics}


ProductionGenerationExecutor = ReasoningGenerationExecutor
