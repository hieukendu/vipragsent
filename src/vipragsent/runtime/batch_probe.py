from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from ..hashing import sha256_json
from .model_assets import write_family_status

DEFAULT_CANDIDATE_ORDER = (32, 16, 8, 4, 2, 1)


def probe_physical_batch(
    root: str | Path,
    model_family: str,
    *,
    probe: Callable[[int], Any] | None = None,
    candidate_order: Iterable[int] = DEFAULT_CANDIDATE_ORDER,
    effective_batch_size: int = 32,
    hardware_identity: str = "unknown",
    fake: bool = False,
) -> dict[str, Any]:
    """Select the largest successful batch and retain every failure/OOM observation."""
    successes: list[int] = []
    failures: list[dict[str, Any]] = []
    for candidate in candidate_order:
        try:
            if fake and probe is None:
                ok = candidate <= 4
            else:
                ok = bool(probe(candidate) if probe else False)
            if ok:
                successes.append(candidate)
            else:
                failures.append({"batch": candidate, "reason": "probe_returned_false", "oom": False})
        except RuntimeError as exc:
            message = str(exc)
            failures.append({"batch": candidate, "reason": message, "oom": "out of memory" in message.casefold() or "oom" in message.casefold()})
        except Exception as exc:
            failures.append({"batch": candidate, "reason": f"{type(exc).__name__}: {exc}", "oom": False})
    successful_batch = max(successes) if successes else None
    gradient_accumulation = None if successful_batch is None else max(1, (effective_batch_size + successful_batch - 1) // successful_batch)
    result = {
        "model_family": model_family,
        "status": "PASS" if successful_batch is not None else "BLOCKED",
        "frozen": successful_batch is not None,
        "candidate_order": list(candidate_order),
        "successful_batch": successful_batch,
        "successful_candidates": successes,
        "failed_candidates": failures,
        "oom_evidence": [item for item in failures if item["oom"]],
        "effective_batch_size": effective_batch_size,
        "gradient_accumulation_steps": gradient_accumulation,
        "hardware_identity": hardware_identity,
        "fixture_probe": fake,
        "probe_hash": sha256_json({"family": model_family, "successful_batch": successful_batch, "failures": failures}),
    }
    write_family_status(root, model_family, "batch", result)
    return result
