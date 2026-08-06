from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ..data.rationales import rationale_only_target
from ..hashing import sha256_json


def validate_rationale_text(text: str) -> str:
    if "<LABELS>" in text or "</LABELS>" in text:
        raise ValueError("Rationale-only responses must not contain labels")
    if not text.strip():
        raise ValueError("Rationale must not be empty")
    if "<RATIONALE>" in text and "</RATIONALE>" in text:
        inner = text.split("<RATIONALE>", 1)[1].split("</RATIONALE>", 1)[0].strip()
    else:
        inner = text.strip()
    return rationale_only_target(inner)


class RationaleCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        item = json.loads(line)
                        self.entries[item["sample_id"]] = item

    def get(self, sample_id: str) -> dict[str, Any] | None:
        return self.entries.get(sample_id)

    def put(self, item: Mapping[str, Any]) -> None:
        self.entries[str(item["sample_id"])] = dict(item)
        self.path.write_text("".join(json.dumps(self.entries[key], ensure_ascii=False, sort_keys=True) + "\n" for key in sorted(self.entries)), encoding="utf-8")


def generate_rationales(
    inputs: list[Mapping[str, Any]],
    request: Callable[[Mapping[str, Any]], str],
    *,
    cache: RationaleCache,
    failure_manifest: str | Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    requests = 0
    for item in inputs:
        if cache.get(str(item["sample_id"])):
            continue
        requests += 1
        if dry_run:
            continue
        try:
            raw = request(item)
            target = validate_rationale_text(raw)
            cache.put({
                "sample_id": item["sample_id"],
                "comment": item["comment"],
                "rationale_target": target,
                "source_input_hash": sha256_json(item),
            })
        except Exception as exc:
            failures.append({"sample_id": item["sample_id"], "error": str(exc)})
    Path(failure_manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(failure_manifest).write_text(json.dumps(failures, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"input_count": len(inputs), "requests_needed": requests, "completed": len(cache.entries), "failures": len(failures), "dry_run": dry_run}
