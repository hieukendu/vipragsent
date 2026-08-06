from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..atomic import atomic_write_json, atomic_write_text
from .contracts import RunContext, RunEntry, RunStatus, StageStatus
from .run_store import RunStore, artifact_hashes, utc_now
from .stage_registry import (
    build_single_azure_stage_registry,
    build_single_experiment_stage_registry,
)


def _blocked_summary(context: RunContext, *, run_status: str, reason: str, blockers: list[str], preflight_only: bool = False) -> dict[str, Any]:
    run_root = Path(context.run_root)
    summary = {
        "run_id": context.entry.run_id,
        "experiment_id": None if context.entry.is_azure else context.entry.run_id,
        "azure_job_id": context.entry.run_id if context.entry.is_azure else None,
        "research_question": context.entry.research_question,
        "system_id": context.entry.system_id,
        "display_name": context.entry.display_name,
        "variant": context.entry.variant,
        "backbone": context.entry.backbone,
        "seed": context.entry.seed if context.entry.seed not in (None, "") else "NOT_APPLICABLE",
        "budget": context.entry.budget if context.entry.budget not in (None, "") else "NOT_APPLICABLE",
        "execution_kind": context.entry.execution_kind,
        "execution_mode": "fixture_synthetic" if context.fixture else "production_sequential_review_gated",
        "run_status": run_status,
        "user_review_status": "PENDING",
        "next_run_allowed": "NO",
        "completion_reason": reason,
        "preflight_only": preflight_only,
        "warnings": ["execution was not completed"],
        "blockers": blockers,
        "validation_status": "BLOCKED",
        "artifact_paths": sorted(artifact_hashes(run_root)),
        "artifact_sha256": artifact_hashes(run_root),
        "RUN_STATUS": "BLOCKED",
        "USER_REVIEW_STATUS": "PENDING",
        "NEXT_RUN_ALLOWED": "NO",
    }
    atomic_write_json(run_root / "review_summary.json", summary)
    atomic_write_text(run_root / "review_summary.md", "# Sequential Run Review Summary\n\n" + f"RUN_STATUS: BLOCKED\nUSER_REVIEW_STATUS: PENDING\nNEXT_RUN_ALLOWED: NO\n\ncompletion_reason: {reason}\n\nblockers:\n" + "\n".join(f"- {item}" for item in blockers) + "\n")
    return summary


def _outcome_error(status: str, message: str) -> dict[str, Any]:
    return {"status": status, "error": message, "blockers": [message]}


def execute_single_run(
    root: str | Path,
    entry_mapping: Mapping[str, Any] | RunEntry,
    *,
    kind: str,
    stage: str,
    run_id: str,
    resume: bool = False,
    dry_run: bool = False,
    fixture: bool = False,
    injected_handlers: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], int]:
    root = Path(root)
    entry = entry_mapping if isinstance(entry_mapping, RunEntry) else RunEntry.from_mapping(entry_mapping, run_id=run_id)
    if entry.run_id != run_id:
        raise ValueError(f"run_id={run_id!r} does not match inventory entry {entry.run_id!r}")
    valid_stages = set(entry.stages) | {"all", "train_or_reuse", "train_or_run"}
    if stage in {"train_or_run", "train_or_reuse"}:
        if "train" in entry.stages:
            stage = "train"
        elif "execute_components" in entry.stages:
            stage = "execute_components"
        elif "train_generation" in entry.stages:
            stage = "train_generation"
        elif "resolve_approved_full_vistral_source" in entry.stages:
            stage = "resolve_approved_full_vistral_source"
    if stage not in valid_stages:
        raise ValueError(f"Unknown sequential stage: {stage}")
    store_root = root / "runs" / "fixture" / "results" / "runs" if fixture else root / "results" / "runs"
    state_path = store_root / entry.run_id / "run_state.json"
    # The documented two-command workflow is preflight followed by all. That second command continues the same run.
    continue_existing = stage == "all" and state_path.exists()
    context = RunContext(root, entry, fixture=fixture, dry_run=dry_run, metadata={"resume": resume or continue_existing})
    store = RunStore(context)
    state = store.initialize(resume=resume or continue_existing)
    if dry_run:
        report = {"run_id": run_id, "kind": kind, "stages": list(entry.stages if stage == "all" else (stage,)), "dry_run": True, "passed": True, "message": "No execution was performed; stop and await explicit user approval before a real run."}
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return report, 0

    if entry.is_azure:
        registry = build_single_azure_stage_registry(root, entry, context)
    else:
        registry = build_single_experiment_stage_registry(root, entry, context)
    registry.update(dict(injected_handlers or {}))
    target = list(entry.stages if stage == "all" else (stage,))
    if stage != "preflight" and "preflight" not in target:
        target.insert(0, "preflight")
    if stage == "preflight":
        target = ["preflight"]

    for current in target:
        previous = state.get("stages", {}).get(current, {})
        if previous.get("status") in {StageStatus.PASS.value, StageStatus.SKIPPED_BY_DESIGN.value}:
            continue
        store.start_stage(state, current)
        try:
            if current == "generate_review_summary":
                from .stage_registry import _review_summary

                raw_outcome = _review_summary(context, entry, state)
                outcome = raw_outcome.as_dict() if hasattr(raw_outcome, "as_dict") else dict(raw_outcome)
            else:
                handler = registry.get(current)
                if handler is None:
                    outcome = {"status": StageStatus.FAIL.value, "error": f"No stage handler registered for {current}", "blockers": [f"No stage handler registered for {current}"]}
                else:
                    raw = handler()
                    outcome = raw.as_dict() if hasattr(raw, "as_dict") else dict(raw)
            expected_files = [str(item) for item in outcome.get("expected_files", [])]
            if outcome.get("status") == StageStatus.PASS.value:
                missing = [name for name in expected_files if not (Path(context.run_root) / name).exists()]
                if missing:
                    outcome = _outcome_error(StageStatus.FAIL.value, "stage completed without required files: " + "; ".join(missing))
                else:
                    store.write_checksums()
            store.complete_stage(state, current, outcome)
        except KeyboardInterrupt:
            state["stages"][current] = {"status": StageStatus.INTERRUPTED.value, "ended_at": utc_now()}
            store.save(state)
            store.append_event("stage_interrupted", {"stage": current})
            return state, 2
        except Exception as exc:
            outcome = _outcome_error(StageStatus.FAIL.value, f"{type(exc).__name__}: {exc}")
            store.complete_stage(state, current, outcome)
        if outcome.get("status") in {StageStatus.BLOCKED.value, StageStatus.FAIL.value}:
            state = store.load()
            _blocked_summary(context, run_status=state.get("run_status", RunStatus.BLOCKED.value), reason="STAGE_BLOCKED" if outcome.get("status") == StageStatus.BLOCKED.value else "STAGE_FAILED", blockers=list(outcome.get("blockers", [])) or [str(outcome.get("error", "stage did not pass"))])
            store.write_checksums()
            return state, 2 if outcome.get("status") == StageStatus.BLOCKED.value else 4
        state = store.load()

    if stage == "preflight":
        preflight_passed = state.get("stages", {}).get("preflight", {}).get("status") == StageStatus.PASS.value
        if preflight_passed:
            store.mark_preflight_ready(state)
            state = store.load()
            summary = _blocked_summary(context, run_status=RunStatus.PREFLIGHT_READY.value, reason="PREFLIGHT_ONLY_COMPLETE", blockers=["Execution stages have not been run; explicit later execution is required"], preflight_only=True)
            store.write_checksums()
            print(json.dumps({"state": state, "review_summary": summary}, indent=2, ensure_ascii=False))
            return state, 0

    if stage != "all":
        state["run_status"] = RunStatus.BLOCKED.value
        store.save(state)
        summary = _blocked_summary(context, run_status=RunStatus.BLOCKED.value, reason="BLOCKED_PENDING_EXECUTION", blockers=[f"Only stage {stage!r} was requested; the run is not complete"], preflight_only=False)
        store.write_checksums()
        print(json.dumps({"state": state, "review_summary": summary}, indent=2, ensure_ascii=False))
        return state, 2

    store.mark_completed_pending_approval(state)
    state = store.load()
    store.write_checksums()
    summary_path = Path(context.run_root) / "review_summary.md"
    if summary_path.exists():
        print(summary_path.read_text(encoding="utf-8"))
    else:
        print(json.dumps(state, indent=2, ensure_ascii=False))
    return state, 0
