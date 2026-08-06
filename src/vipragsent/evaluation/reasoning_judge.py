from __future__ import annotations

import hashlib
import json
import time
import unicodedata
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

from ..atomic import atomic_write_json, atomic_write_text
from ..azure.client import AzureCache, AzureResponsesClient, AzureSettings
from ..constants import PRAGMATIC_LABELS
from ..hashing import sha256_file
from .metrics import binary_macro_f1

JUDGE_LABELS = tuple(PRAGMATIC_LABELS)
RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})


class JudgeTransportError(RuntimeError):
    """Transport failure carrying the status and Retry-After evidence."""

    def __init__(self, message: str, *, status_code: int | None = None, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


class JudgeSemanticError(ValueError):
    """Terminal response/schema error; semantic errors are never retried."""


def normalize_reasoning(value: str) -> str:
    """Apply the locked cache normalization without changing words."""
    return unicodedata.normalize("NFC", str(value).replace("\r\n", "\n").replace("\r", "\n")).strip()


def validate_judge_labels(value: Mapping[str, Any]) -> dict[str, int]:
    if set(value) != set(JUDGE_LABELS):
        missing = sorted(set(JUDGE_LABELS) - set(value))
        extra = sorted(set(value) - set(JUDGE_LABELS))
        raise JudgeSemanticError(f"strict judge keys mismatch; missing={missing}; extra={extra}")
    output: dict[str, int] = {}
    for label in JUDGE_LABELS:
        item = value[label]
        if isinstance(item, bool) or not isinstance(item, int) or item not in (0, 1):
            raise JudgeSemanticError(f"judge value for {label} must be integer 0 or 1")
        output[label] = int(item)
    return output


def load_reasoning_protocol(root: str | Path = ".") -> dict[str, Any]:
    root = Path(root)
    path = root / "configs/experiments/generation_reasoning_protocol.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("generation reasoning protocol must be a mapping")
    required = (
        "protocol_version",
        "generation_prompt_path",
        "generation_prompt_hash",
        "judge_protocol_id",
        "judge_prompt_path",
        "judge_prompt_hash",
        "judge_schema_path",
        "judge_schema_hash",
        "judge_model_version",
        "decoding",
        "cache",
        "retry",
        "invalid_policy",
        "metrics",
        "systems",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"reasoning protocol is missing fields: {missing}")
    return payload


def validate_reasoning_protocol_files(root: str | Path = ".") -> dict[str, Any]:
    root = Path(root)
    protocol = load_reasoning_protocol(root)
    checks: dict[str, bool] = {}
    errors: list[str] = []
    for field, path_field, hash_field in (
        ("generation", "generation_prompt_path", "generation_prompt_hash"),
        ("judge", "judge_prompt_path", "judge_prompt_hash"),
        ("schema", "judge_schema_path", "judge_schema_hash"),
    ):
        path = root / str(protocol[path_field])
        exists = path.exists()
        checks[f"{field}_file_exists"] = exists
        if not exists:
            errors.append(f"missing reasoning protocol file: {path}")
            continue
        observed = sha256_file(path)
        expected = str(protocol[hash_field])
        checks[f"{field}_hash_matches"] = expected not in {"", "TO_BE_FILLED_AFTER_PROTOCOL_FILES_ARE_WRITTEN"} and observed == expected
        if not checks[f"{field}_hash_matches"]:
            errors.append(f"reasoning protocol hash mismatch: {path}")
        if field == "schema":
            try:
                schema = json.loads(path.read_text(encoding="utf-8"))
                checks["strict_schema"] = schema.get("type") == "object" and schema.get("additionalProperties") is False and set(schema.get("required", [])) == set(JUDGE_LABELS)
                if not checks["strict_schema"]:
                    errors.append("reasoning judge schema is not the locked strict six-key schema")
            except (OSError, json.JSONDecodeError) as exc:
                checks["strict_schema"] = False
                errors.append(f"reasoning judge schema is invalid: {exc}")
    checks["judge_receives_reasoning_only"] = protocol.get("judge_input") == "generated_reasoning_only" and protocol.get("original_sentence_visible") is False
    checks["semantic_repair_disabled"] = protocol.get("semantic_repair") is False
    checks["single_judge_protocol"] = protocol.get("judge_protocol_id") == "reasoning_judge_gpt41mini_zeroshot_v1"
    checks["exact_metrics"] = protocol.get("metrics", {}).get("primary") == "full_split_macro_pragmatic_f1_all_zero_fallback" and protocol.get("metrics", {}).get("secondary") == "valid_only_macro_pragmatic_f1"
    checks["retry_contract"] = protocol.get("retry", {}).get("maximum_total_attempts") == 5 and protocol.get("retry", {}).get("semantic_retry") is False
    checks["system_semantics"] = (
        protocol.get("systems", {}).get("cot_only_vistral", {}).get("training") == "generation_only_causal_cross_entropy"
        and protocol.get("systems", {}).get("explanation_only_vistral", {}).get("training") == "no_additional_training"
    )
    errors.extend(key for key, passed in checks.items() if not passed and key not in {"generation_file_exists", "judge_file_exists", "schema_file_exists", "generation_hash_matches", "judge_hash_matches", "schema_hash_matches", "strict_schema"})
    return {"status": "PASS" if not errors else "FAIL", "checks": checks, "errors": sorted(set(errors)), "protocol": protocol}


def validate_azure_judge_manifest(root: str | Path = ".") -> dict[str, Any]:
    root = Path(root)
    path = root / "data/manifests/azure_deployment.json"
    if not path.exists():
        raise JudgeTransportError("Azure deployment manifest is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    observed_model = str(metadata.get("model") or payload.get("model_family") or "")
    observed_version = str(metadata.get("version") or payload.get("expected_model_version") or "")
    if payload.get("verified") is not True or str(payload.get("deployment")) != "gpt-4.1-mini":
        raise JudgeTransportError("Azure reasoning-judge deployment manifest is not verified")
    if "gpt-4.1-mini" not in observed_model.casefold() or observed_version != "2025-04-14":
        raise JudgeTransportError("Azure reasoning-judge model/version does not match the locked protocol")
    return {"status": "PASS", "path": path.relative_to(root).as_posix(), "sha256": sha256_file(path), "deployment": payload["deployment"], "model": observed_model, "version": observed_version}


def _transport_error(response: Any) -> JudgeTransportError | None:
    if not isinstance(response, Mapping):
        return None
    status = response.get("status_code", response.get("status"))
    try:
        status_code = int(status) if status is not None else None
    except (TypeError, ValueError):
        status_code = None
    if status_code is not None and status_code >= 400:
        retry_after = response.get("retry_after")
        try:
            retry_after = float(retry_after) if retry_after is not None else None
        except (TypeError, ValueError):
            retry_after = None
        return JudgeTransportError(str(response.get("error") or "judge transport failure"), status_code=status_code, retry_after=retry_after)
    return None


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    if isinstance(exc, JudgeTransportError):
        return exc.status_code in RETRYABLE_STATUS_CODES
    return False


def _retry_after(exc: Exception) -> float | None:
    value = getattr(exc, "retry_after", None)
    if value is not None:
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return None
    headers = getattr(exc, "headers", None)
    if headers:
        try:
            value = headers.get("Retry-After") or headers.get("retry-after")
            return max(0.0, float(value)) if value is not None else None
        except (TypeError, ValueError):
            return None
    return None


def _parse_response(response: Any) -> dict[str, int]:
    if isinstance(response, Mapping):
        for key in ("parsed", "output", "response", "labels"):
            if key in response:
                candidate = response[key]
                if isinstance(candidate, Mapping):
                    return validate_judge_labels(candidate)
                if isinstance(candidate, str):
                    return _parse_response(candidate)
        if all(label in response for label in JUDGE_LABELS):
            return validate_judge_labels(response)
        text = response.get("output_text") or response.get("text")
        if text is not None:
            return _parse_response(text)
    if isinstance(response, bytes):
        response = response.decode("utf-8")
    if isinstance(response, str):
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError as exc:
            raise JudgeSemanticError("judge response is not valid JSON") from exc
        if not isinstance(parsed, Mapping):
            raise JudgeSemanticError("judge response JSON is not an object")
        return validate_judge_labels(parsed)
    raise JudgeSemanticError("judge response does not contain the strict six-key object")


def _empty_diagnostics() -> dict[str, int]:
    return {"judge_cache_hits": 0, "judge_cache_misses": 0, "judge_retry_count": 0, "judge_input_tokens": 0, "judge_output_tokens": 0, "judge_request_count": 0}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class ReasoningJudge:
    """Shared, reasoning-only judge with terminal-invalid caching."""

    def __init__(
        self,
        root: str | Path,
        *,
        transport: Callable[..., Any] | None = None,
        client: AzureResponsesClient | None = None,
        cache_root: str | Path | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        max_attempts: int = 5,
        require_deployment_manifest: bool = False,
    ) -> None:
        self.root = Path(root)
        self.protocol = load_reasoning_protocol(self.root)
        self.protocol_validation = validate_reasoning_protocol_files(self.root)
        if self.protocol_validation["status"] != "PASS":
            raise ValueError("reasoning protocol files are not validated: " + "; ".join(self.protocol_validation["errors"]))
        if require_deployment_manifest:
            validate_azure_judge_manifest(self.root)
        self.prompt_template = (self.root / str(self.protocol["judge_prompt_path"])).read_text(encoding="utf-8")
        self.schema = json.loads((self.root / str(self.protocol["judge_schema_path"])).read_text(encoding="utf-8"))
        self.sleep_fn = sleep_fn
        self.max_attempts = int(max_attempts)
        if self.max_attempts != 5:
            raise ValueError("the locked reasoning judge allows exactly five total attempts")
        self.cache_root = Path(cache_root) if cache_root is not None else self.root / "results/reasoning_judge_cache"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        if client is not None and transport is not None:
            raise ValueError("pass either client or transport, not both")
        if client is None:
            if transport is not None:
                settings = AzureSettings(
                    endpoint="https://fixture.azure.com/",
                    base_url="https://fixture.azure.com/openai/v1/",
                    deployment="gpt-4.1-mini",
                    batch_deployment=None,
                    auth_mode="api_key",
                    model_family="GPT-4.1-mini",
                    expected_model_version=str(self.protocol["judge_model_version"]),
                )

                def fixture_transport(**kwargs: Any) -> Any:
                    return transport(
                        prompt=kwargs["input"],
                        schema=self.schema,
                        temperature=kwargs["temperature"],
                        max_output_tokens=kwargs["max_output_tokens"],
                        metadata=kwargs.get("metadata", {}),
                    )

                client = AzureResponsesClient(settings, transport=fixture_transport, cache=AzureCache(self.cache_root))
            else:
                settings = AzureSettings.from_env()
                client = AzureResponsesClient(settings, cache=AzureCache(self.cache_root))
        elif client.cache is None:
            client.cache = AzureCache(self.cache_root)
        self.client = client
        self.transport = transport
        self.diagnostics = _empty_diagnostics()

    @property
    def prompt_hash(self) -> str:
        return str(self.protocol["judge_prompt_hash"])

    @property
    def schema_hash(self) -> str:
        return str(self.protocol["judge_schema_hash"])

    @property
    def judge_protocol_id(self) -> str:
        return str(self.protocol["judge_protocol_id"])

    def cache_key(self, reasoning: str) -> str:
        normalized = normalize_reasoning(reasoning)
        identity = "".join((self.judge_protocol_id, str(self.protocol["judge_model_version"]), str(self.protocol["judge_temperature"]), self.prompt_hash, self.schema_hash, normalized))
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.client.cache.path_for(key) if self.client.cache is not None else self.cache_root / f"{key}.json"

    def _read_cache(self, key: str) -> dict[str, Any] | None:
        if self.client.cache is None:
            return None
        return self.client.cache.get(key, expected_model_family=self.client.settings.model_family, expected_model_version=str(self.protocol["judge_model_version"]))

    def _write_cache(self, key: str, value: Mapping[str, Any]) -> None:
        if self.client.cache is not None:
            self.client.cache.put(key, dict(value))

    def _invalid_generation_record(self, key: str, reason: str) -> dict[str, Any]:
        return {
            "valid": False,
            "labels": None,
            "raw_response": None,
            "invalid_stage": "generation",
            "invalid_reason": reason,
            "cache_key": key,
            "cache_hit": False,
            "retry_count": 0,
            "normalized_reasoning_sha256": hashlib.sha256(b"").hexdigest(),
            "usage": {},
            "observed_model": self.client.settings.model_family,
            "observed_model_version": str(self.protocol["judge_model_version"]),
            "expected_model_family": self.client.settings.model_family,
            "expected_model_version": str(self.protocol["judge_model_version"]),
        }

    def _render(self, reasoning: str) -> str:
        return self.prompt_template.replace("{GENERATED_REASONING}", normalize_reasoning(reasoning))

    def _judge_response(self, response: Any, *, request_id: str, retry_count: int, key: str, normalized: str) -> dict[str, Any]:
        labels = _parse_response(response)
        usage = response.get("usage", {}) if isinstance(response, Mapping) else {}
        record = {
            "valid": True,
            "labels": labels,
            "raw_response": response,
            "invalid_stage": None,
            "invalid_reason": None,
            "cache_key": key,
            "cache_hit": False,
            "request_id": response.get("id", request_id) if isinstance(response, Mapping) else request_id,
            "retry_count": retry_count,
            "normalized_reasoning_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
            "usage": dict(usage) if isinstance(usage, Mapping) else {},
        }
        self._write_cache(key, record)
        return record

    def judge(self, reasoning: str) -> dict[str, Any]:
        normalized = normalize_reasoning(reasoning)
        key = self.cache_key(normalized)
        cached = self._read_cache(key)
        if cached is not None and "valid" in cached:
            self.diagnostics["judge_cache_hits"] += 1
            return dict(cached) | {"cache_hit": True}
        self.diagnostics["judge_cache_misses"] += 1
        if not normalized:
            record = self._invalid_generation_record(key, "empty_reasoning")
            self._write_cache(key, record)
            return record
        prompt = self._render(normalized)
        request_id = hashlib.sha256((key + prompt).encode("utf-8")).hexdigest()[:24]
        record = self.client.create_structured(
            prompt=prompt,
            task="reasoning_judge",
            schema={"strict": True, "schema": self.schema},
            max_output_tokens=int(self.protocol["judge_max_output_tokens"]),
            input_payload=normalized,
            expected_model_version=str(self.protocol["judge_model_version"]),
            cache_identity={
                "judge_protocol_id": self.judge_protocol_id,
                "request_id": request_id,
            },
            cache_key=key,
            output_validator=validate_judge_labels,
            return_invalid=True,
            terminal_invalid_stage="judge_response",
            retries=self.max_attempts - 1,
            sleep=self.sleep_fn,
        )
        record = dict(record)
        retry_count = int(record.get("retry_count", 0) or 0)
        record["cache_key"] = key
        record["normalized_reasoning_sha256"] = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        record.setdefault("cache_hit", False)
        self._write_cache(key, record)
        self.diagnostics["judge_request_count"] += 1 + retry_count
        self.diagnostics["judge_retry_count"] += retry_count
        usage = record.get("usage", {})
        self.diagnostics["judge_input_tokens"] += int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0) if isinstance(usage, Mapping) else 0
        self.diagnostics["judge_output_tokens"] += int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0) if isinstance(usage, Mapping) else 0
        return record

    def write_artifacts(self, run_root: str | Path, split: str, rows: Iterable[Mapping[str, Any]], decisions: Iterable[Mapping[str, Any]]) -> None:
        run_root = Path(run_root)
        decisions_list = [dict(item) for item in decisions]
        del rows
        atomic_write_text(run_root / f"judge/{split}_judge_responses.jsonl", "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in decisions_list))
        invalid: list[dict[str, Any]] = []
        for response_path in sorted((run_root / "judge").glob("*_judge_responses.jsonl")):
            invalid.extend(item for item in (_read_jsonl(response_path)) if item.get("valid") is not True)
        atomic_write_text(run_root / "judge/invalid_outputs.jsonl", "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in invalid))
        cache_entries = sorted(path.name for path in self.cache_root.glob("*.json"))
        atomic_write_json(run_root / "judge/cache_manifest.json", {"judge_protocol_id": self.judge_protocol_id, "prompt_hash": self.prompt_hash, "schema_hash": self.schema_hash, "cache_entries": cache_entries, "cache_entry_count": len(cache_entries)})
        atomic_write_json(run_root / "judge/usage.json", dict(self.diagnostics))


def _zero_prediction() -> dict[str, int]:
    return {label: 0 for label in JUDGE_LABELS}


def build_reasoning_prediction_row(sample_id: str, gold: Mapping[str, Any], generated_reasoning: str, decision: Mapping[str, Any], *, truncated: bool = False) -> dict[str, Any]:
    valid = decision.get("valid") is True
    labels = dict(decision.get("labels") or {}) if valid else _zero_prediction()
    return {
        "sample_id": str(sample_id),
        "gold": {label: int(gold[label]) for label in JUDGE_LABELS},
        "generated_reasoning": generated_reasoning,
        "judge_raw_response": decision.get("raw_response"),
        "valid_prediction": valid,
        "invalid_stage": decision.get("invalid_stage"),
        "invalid_reason": decision.get("invalid_reason"),
        "effective_prediction_all_zero_fallback": labels,
        "effective_full_split_all_zero_fallback": labels,
        "predictions": labels,
        "valid_prediction_labels": labels if valid else None,
        "truncated": bool(truncated),
        "judge_cache_key": decision.get("cache_key"),
        "judge_cache_hit": bool(decision.get("cache_hit", False)),
        "judge_retry_count": int(decision.get("retry_count", 0) or 0),
    }


def compute_reasoning_metrics(rows: Iterable[Mapping[str, Any]], *, diagnostics: Mapping[str, Any] | None = None) -> dict[str, Any]:
    records = [dict(row) for row in rows]
    requested = len(records)
    def effective_prediction(row: Mapping[str, Any]) -> tuple[dict[str, int], bool]:
        candidate = row.get("effective_full_split_all_zero_fallback") or row.get("predictions")
        if not isinstance(candidate, Mapping) or any(label not in candidate for label in JUDGE_LABELS):
            return _zero_prediction(), True
        return {label: int(candidate[label]) for label in JUDGE_LABELS}, False

    normalized_records: list[dict[str, Any]] = []
    missing_prediction_count = 0
    for row in records:
        prediction, missing = effective_prediction(row)
        missing_prediction_count += int(missing)
        normalized = dict(row)
        normalized["predictions"] = prediction
        normalized_records.append(normalized)
    valid_records = [row for row in normalized_records if row.get("valid_prediction") is True and not effective_prediction(row)[1]]
    primary_per_label: dict[str, float] = {}
    valid_per_label: dict[str, float] = {}
    for label in JUDGE_LABELS:
        gold = [int(row["gold"][label]) for row in normalized_records]
        primary = [int(row["predictions"][label]) for row in normalized_records]
        primary_per_label[label] = binary_macro_f1(gold, primary) if normalized_records else 0.0
        if valid_records:
            valid_per_label[label] = binary_macro_f1([int(row["gold"][label]) for row in valid_records], [int(row["predictions"][label]) for row in valid_records])
    invalid = requested - len(valid_records)
    invalid_generation = sum(row.get("invalid_stage") == "generation" for row in normalized_records)
    invalid_judge = sum(row.get("invalid_stage") in {"judge_request", "judge_response"} for row in normalized_records)
    truncated = sum(bool(row.get("truncated")) for row in normalized_records)
    output = {
        "primary_metric_name": "full_split_macro_pragmatic_f1_all_zero_fallback",
        "primary_macro_f1": sum(primary_per_label.values()) / len(JUDGE_LABELS) if records else 0.0,
        "primary_per_label_f1": primary_per_label,
        "valid_only_metric_name": "valid_only_macro_pragmatic_f1",
        "valid_only_macro_f1": sum(valid_per_label.values()) / len(JUDGE_LABELS) if valid_records else "NOT_APPLICABLE",
        "valid_only_per_label_f1": valid_per_label if valid_records else "NOT_APPLICABLE",
        "requested_count": requested,
        "valid_count": len(valid_records),
        "invalid_count": invalid,
        "coverage_rate": len(valid_records) / requested if requested else 0.0,
        "invalid_generation_rate": invalid_generation / requested if requested else 0.0,
        "invalid_judge_output_rate": invalid_judge / requested if requested else 0.0,
        "missing_prediction_rate": missing_prediction_count / requested if requested else 0.0,
        "truncation_rate": truncated / requested if requested else 0.0,
        "significance_prediction_source": "effective_full_split_all_zero_fallback",
    }
    output.update({key: int(value) for key, value in (diagnostics or {}).items() if key in _empty_diagnostics()})
    return output
