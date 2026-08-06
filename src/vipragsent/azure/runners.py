from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from ..atomic import atomic_write_json, exclusive_lock
from ..data.rationales import rationale_only_target, rationale_plus_labels_target
from ..hashing import sha256_json
from .client import AzureResponsesClient
from .prompts import PromptSpec, validate_task_demo_manifest
from .schemas import strict_rationale_schema


class RationaleRunner:
    def __init__(self, client: AzureResponsesClient, *, output_path: str | Path, failure_path: str | Path, prompt_version: str = "rationale_v1") -> None:
        self.client = client
        self.output_path = Path(output_path)
        self.failure_path = Path(failure_path)
        self.prompt_version = prompt_version

    def run(self, inputs: Iterable[Mapping[str, Any]], prompt_builder: Callable[[Mapping[str, Any]], str], *, dry_run: bool = False) -> dict[str, Any]:
        rows = sorted((dict(item) for item in inputs), key=lambda item: str(item["sample_id"]))
        completed: dict[str, dict[str, Any]] = {}
        if self.output_path.exists():
            for line in self.output_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    item = json.loads(line)
                    completed[str(item["sample_id"])] = item
        failures: list[dict[str, Any]] = []
        requests = 0
        schema = {"strict": True, "schema": strict_rationale_schema()}
        for item in rows:
            sample_id = str(item["sample_id"])
            if sample_id in completed:
                continue
            requests += 1
            if dry_run:
                continue
            prompt = prompt_builder(item)
            try:
                response = self.client.create_structured(prompt=prompt, task="rationale", schema=schema, max_output_tokens=256, sample_id=sample_id, input_payload=item)
                target = rationale_only_target(response["labels"]["rationale"])
                completed[sample_id] = {
                    "sample_id": sample_id,
                    "rationale_target": target,
                    "prompt_hash": response["prompt_hash"],
                    "schema_hash": response["schema_hash"],
                    "deployment": response["deployment"],
                    "response_id": response["response_id"],
                    "observed_model": response["observed_model"],
                    "observed_model_version": response["observed_model_version"],
                    "usage": response.get("usage", {}),
                }
            except Exception as exc:
                failures.append({"sample_id": sample_id, "status": "FAILED", "error": str(exc)})
        if not dry_run:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with exclusive_lock(self.output_path.with_suffix(".lock")):
                self.output_path.write_text("".join(json.dumps(completed[key], ensure_ascii=False, sort_keys=True) + "\n" for key in sorted(completed)), encoding="utf-8", newline="\n")
            atomic_write_json(self.failure_path, failures)
        return {"input_count": len(rows), "requests_needed": requests, "completed": len(completed), "failures": len(failures), "dry_run": dry_run, "scope": "vipragsent_train_only"}


class PromptedBaselineRunner:
    def __init__(self, client: AzureResponsesClient, *, output_path: str | Path, failure_path: str | Path) -> None:
        self.client = client
        self.output_path = Path(output_path)
        self.failure_path = Path(failure_path)

    def run(self, rows: Iterable[Mapping[str, Any]], *, task: str, prompt_builder: Callable[[Mapping[str, Any]], PromptSpec], manifest: Mapping[str, Any], dry_run: bool = False) -> dict[str, Any]:
        validate_task_demo_manifest(manifest, task)
        inputs = list(rows)
        completed: dict[str, dict[str, Any]] = {}
        if self.output_path.exists():
            for line in self.output_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    item = json.loads(line)
                    completed[str(item["sample_id"])] = item
        max_tokens = {"pragmatic": 128, "polarity": 32, "emotion": 32}[task]
        failures: list[dict[str, Any]] = []
        requests = 0
        schema = {"strict": True, "schema": __import__("vipragsent.azure.schemas", fromlist=["strict_label_schema"]).strict_label_schema(task)}
        demo_hash = sha256_json({"sample_ids": manifest["sample_ids"], "prompt_hash": manifest.get("prompt_hash")})
        for row in inputs:
            sample_id = str(row["sample_id"])
            if sample_id in completed:
                continue
            requests += 1
            if dry_run:
                continue
            try:
                prompt = prompt_builder(row)
                response = self.client.create_structured(prompt=prompt.text, task=task, schema=schema, max_output_tokens=max_tokens, sample_id=sample_id, input_payload=row, demonstration_manifest_hash=demo_hash)
                completed[sample_id] = {"sample_id": sample_id, "status": "PASS", "labels": response["labels"], "prompt_hash": response["prompt_hash"], "schema_hash": response["schema_hash"], "demonstration_manifest_hash": demo_hash, "response_id": response["response_id"], "deployment": response["deployment"], "observed_model": response["observed_model"], "observed_model_version": response["observed_model_version"], "usage": response.get("usage", {})}
            except Exception as exc:
                completed[sample_id] = {"sample_id": sample_id, "status": "INVALID", "labels": None, "error": str(exc)}
                failures.append({"sample_id": sample_id, "status": "INVALID", "error": str(exc)})
        if not dry_run:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with exclusive_lock(self.output_path.with_suffix(".lock")):
                self.output_path.write_text("".join(json.dumps(completed[key], ensure_ascii=False, sort_keys=True) + "\n" for key in sorted(completed)), encoding="utf-8", newline="\n")
            atomic_write_json(self.failure_path, failures)
        return {"input_count": len(inputs), "requests_needed": requests, "completed": len(completed), "invalid_output_count": len(failures), "invalid_output_rate": len(failures) / len(inputs) if inputs else 0.0, "dry_run": dry_run, "task": task, "demonstration_manifest_hash": demo_hash}


def build_cot_target(rationale_target: str, labels: Mapping[str, Any]) -> str:
    if "<LABELS>" in rationale_target:
        raise ValueError("Full rationale target must not already contain labels")
    inner = rationale_target.split("<RATIONALE>", 1)[-1].split("</RATIONALE>", 1)[0].strip()
    return rationale_plus_labels_target(inner, dict(labels))
