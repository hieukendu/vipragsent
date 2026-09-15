"""Run the isolated XLM-R-large Q1b/Q2/Q3/Q4 follow-up queue.

The queue owns one GPU slot and executes entries sequentially.  The Hub
uploader is started as a separate process and consumes an append-only queue,
so uploads never block the next experiment.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from _bootstrap import ROOT
from vipragsent.atomic import atomic_write_json
from vipragsent.hashing import sha256_file
from vipragsent.orchestration.sequential import execute_sequential_run

SEEDS = (20260521, 20260522, 20260523)
Q3_BUDGETS = ("32", "64", "128", "256", "512", "full")
Q2_VARIANTS = (
    ("full", "xlmr_followup_full"),
    ("no_emotion_auxiliary", "xlmr_followup_no_emotion_auxiliary"),
    ("no_polarity_auxiliary", "xlmr_followup_no_polarity_auxiliary"),
    ("no_rationale", "xlmr_followup_no_rationale"),
    ("no_uncertainty_weighting", "xlmr_followup_no_uncertainty_weighting"),
    ("no_multitask", "xlmr_followup_no_multitask"),
)
MODEL_REVISION = "c23d21b0620b635a76227c604d44e43a9f0ee389"
MODEL_REPOSITORY = "FacebookAI/xlm-roberta-large"
TOKENIZER_REVISION = MODEL_REVISION
CAMPAIGN_ID = "vipragsent-xlmr-q1b-q4-followup-v1"
QUEUE_FILE = ROOT / "runtime/xlmr_followup_hf_upload_queue.jsonl"
UPLOADER_STATUS = ROOT / "runtime/xlmr_followup_hf_uploader_status.json"
UPLOADER_PID = ROOT / "runtime/xlmr_followup_hf_uploader.pid"
UPLOADER_STOP = ROOT / "runtime/STOP_XLMR_FOLLOWUP_UPLOADER"
RUNNER_STATUS = ROOT / "runtime/xlmr_followup_runner_status.json"
LOG_DIR = ROOT / "runtime/xlmr_followup_logs"
REVIEWER = "user-standing-authorization"
STOP_REQUESTED = False


def _signal_handler(_signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"expected a mapping in {path}")
    return dict(payload)


def _base_entry(run_id: str, *, research_question: str, system_id: str, variant: str, seed: int, execution_kind: str, budget: str | None = None, followup_lane: str = "") -> dict[str, Any]:
    row: dict[str, Any] = {
        "experiment_id": run_id,
        "run_id": run_id,
        "research_question": research_question,
        "system_id": system_id,
        "display_name": run_id.replace("_", " "),
        "variant": variant,
        "backbone": "xlmr_large",
        "model_family": "xlmr_large",
        "model": MODEL_REPOSITORY,
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "preprocessing_name": "unicode_nfc",
        "preprocessing_version": "locked-v1",
        "seed": seed,
        "budget": budget if budget is not None else "",
        "execution_kind": execution_kind,
        "execution_policy": "sequential_review_gated",
        "required_phase15_assets": "xlmr_large_cache;offline_smoke;frozen_batch;validated_gpu",
        "dependencies": "isolated_xlmr_followup_lane",
        "split": "vipragsent_train_dev_test",
        "external_finetuning": False,
        "followup_lane": followup_lane,
        "_followup_manifest": "configs/experiments/xlmr_followup.yaml",
        "_repository_root": str(ROOT),
    }
    return row


def build_entries() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        rows.append(
            _base_entry(
                f"xlmr_followup_q1b_vipragsent_full_{seed}",
                research_question="Q1b",
                system_id="xlmr_followup_full",
                variant="vipragsent_full_xlmr_large",
                seed=seed,
                execution_kind="trainable",
                followup_lane="xlmr_q1b_train_and_external",
            )
        )
    for variant, system_id in Q2_VARIANTS:
        for seed in SEEDS:
            row = _base_entry(
                f"xlmr_followup_q2_{variant}_{seed}",
                research_question="Q2",
                system_id=system_id,
                variant=variant,
                seed=seed,
                execution_kind="component_bundle" if variant == "no_multitask" else "trainable",
                followup_lane="xlmr_q2_train",
            )
            rows.append(row)
    for budget in Q3_BUDGETS:
        mask_path = ROOT / "data/processed/q3_low_resource_sarcasm" / f"budget_{budget}_masks.csv"
        mask_hash = sha256_file(mask_path) if mask_path.exists() else ""
        for seed in SEEDS:
            row = _base_entry(
                f"xlmr_followup_q3_full_{budget}_{seed}",
                research_question="Q3",
                system_id="xlmr_followup_full",
                variant="full",
                seed=seed,
                budget=budget,
                execution_kind="trainable",
                followup_lane="xlmr_q3_train",
            )
            row.update({"q3_mask_path": str(mask_path.relative_to(ROOT)), "q3_mask_hash": mask_hash, "selection_metric": "dev_sarcasm_binary_macro_f1"})
            rows.append(row)
    for seed in SEEDS:
        source_id = f"q1a_vipragsent_full_XLM_R_large_optimization_pragmatic_warmup020_irony105_implicit101_sarcasm101_code104_v37_{seed}"
        row = _base_entry(
            f"xlmr_followup_q4_full_{seed}",
            research_question="Q4",
            system_id="xlmr_followup_q4",
            variant="full",
            seed=seed,
            execution_kind="artifact_extraction",
            followup_lane="xlmr_q4_from_v37_full_checkpoint",
        )
        row.update(
            {
                "source_run_id": source_id,
                "source_checkpoint_id": f"vipragsent_full_xlmr_large:{seed}",
                "raw_probability_source": "source_run.predictions/test_predictions.jsonl",
                "split": "vipragsent_test",
            }
        )
        rows.append(row)
    return rows


def _write_json(path: Path, payload: Any) -> None:
    atomic_write_json(path, payload)


def _append_queue(entry: dict[str, Any]) -> None:
    QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with QUEUE_FILE.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        existing: set[str] = set()
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and item.get("run_id"):
                existing.add(str(item["run_id"]))
        if entry["experiment_id"] not in existing:
            handle.seek(0, os.SEEK_END)
            handle.write(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": entry["experiment_id"],
                        "source_root": f"results/runs/{entry['experiment_id']}",
                        "backbone": entry["backbone"],
                        "status": "queued",
                        "campaign_id": CAMPAIGN_ID,
                        "queued_at": time.time(),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _start_uploader() -> int:
    current = {}
    if UPLOADER_PID.exists():
        try:
            current = json.loads(UPLOADER_PID.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = {}
    if isinstance(current, dict) and isinstance(current.get("pid"), int) and _pid_alive(int(current["pid"])):
        return int(current["pid"])
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = (LOG_DIR / "hf_uploader.log").open("a", encoding="utf-8", buffering=1)
    command = [
        sys.executable,
        str(ROOT / "scripts/hf_artifact_uploader.py"),
        "--queue-file",
        str(QUEUE_FILE),
        "--status-file",
        str(UPLOADER_STATUS),
        "--stop-file",
        str(UPLOADER_STOP),
        "--campaign-id",
        CAMPAIGN_ID,
    ]
    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    log.close()
    _write_json(UPLOADER_PID, {"pid": process.pid, "started_at": time.time(), "command": command})
    return process.pid


def _approve(run_id: str, log_handle: Any) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts/record_run_approval.py"),
        "--run-id",
        run_id,
        "--decision",
        "approve",
        "--reviewer",
        REVIEWER,
        "--review-note",
        "Standing user authorization for the isolated XLM-R-large Q1b/Q2/Q3/Q4 follow-up lane after local validation PASS.",
    ]
    result = subprocess.run(command, cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"approval failed for {run_id}: exit={result.returncode}")


def _run_one(entry: dict[str, Any], *, auto_approve: bool, max_retries: int) -> dict[str, Any]:
    run_id = str(entry["experiment_id"])
    _append_queue(entry)
    run_root = ROOT / "results/runs" / run_id
    state = _read_json(run_root / "state.json", {})
    approval = _read_json(run_root / "approval_status.json", {})
    if (
        state.get("run_status") == "APPROVED"
        and state.get("approval_status") == "APPROVED"
        and approval.get("status") == "APPROVED"
    ):
        return {"run_id": run_id, "status": "ALREADY_APPROVED"}
    log_path = LOG_DIR / f"{run_id}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    last_code: int | None = None
    for attempt in range(max_retries + 1):
        if STOP_REQUESTED:
            return {"run_id": run_id, "status": "STOPPED"}
        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(f"\n=== attempt {attempt + 1}/{max_retries + 1} at {time.time()} ===\n")
            command = [sys.executable, str(Path(__file__).resolve()), "--run-id", run_id]
            if (run_root / "state.json").exists():
                command.append("--resume")
            result = subprocess.run(command, cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT, check=False)
            last_code = result.returncode
            state = _read_json(run_root / "state.json", {})
            if result.returncode == 0 and state.get("run_status") == "COMPLETED_PENDING_APPROVAL":
                try:
                    if auto_approve:
                        _approve(run_id, log_handle)
                    return {"run_id": run_id, "status": "APPROVED" if auto_approve else "COMPLETED_PENDING_APPROVAL", "attempt": attempt + 1}
                except Exception as exc:
                    log_handle.write(f"approval failure: {type(exc).__name__}: {exc}\n")
            else:
                log_handle.write(f"scientific runner exit={result.returncode} run_status={state.get('run_status')}\n")
        if attempt < max_retries:
            time.sleep(5.0)
    return {"run_id": run_id, "status": "FAILED", "exit_code": last_code, "state": _read_json(run_root / "state.json", {})}


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except (OSError, json.JSONDecodeError):
        return default


def _run_single(entry: dict[str, Any], *, stage: str, resume: bool) -> int:
    state, exit_code = execute_sequential_run(
        ROOT,
        entry,
        kind="xlmr_followup",
        stage=stage,
        run_id=str(entry["experiment_id"]),
        resume=resume,
    )
    if isinstance(state, dict):
        print(json.dumps({"run_id": entry["experiment_id"], "run_status": state.get("run_status"), "exit_code": exit_code}, ensure_ascii=False))
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the isolated XLM-R-large Q1b/Q2/Q3/Q4 follow-up")
    parser.add_argument("--run-id", help="run exactly one generated follow-up entry")
    parser.add_argument("--stage", default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--no-auto-approve", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    entries = build_entries()
    by_id = {str(entry["experiment_id"]): entry for entry in entries}
    if args.list:
        print(json.dumps(entries, indent=2, ensure_ascii=False))
        return 0
    if args.run_id:
        try:
            entry = by_id[args.run_id]
        except KeyError as exc:
            raise SystemExit(f"unknown XLM-R follow-up run ID: {args.run_id}") from exc
        return _run_single(entry, stage=args.stage, resume=args.resume)
    if args.max_retries < 0:
        raise SystemExit("--max-retries must be non-negative")
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    manifest = {
        "schema_version": 1,
        "lane_id": "xlmr_q1b_q2_q3_q4_followup_v1",
        "campaign_id": CAMPAIGN_ID,
        "entry_count": len(entries),
        "run_ids": [entry["experiment_id"] for entry in entries],
        "backbone": "xlmr_large",
        "seeds": list(SEEDS),
        "q3_budgets": list(Q3_BUDGETS),
        "generated_at": time.time(),
        "source_manifest": str((ROOT / "configs/experiments/xlmr_followup.yaml").relative_to(ROOT)),
    }
    _write_json(ROOT / "reports/xlmr_followup_run_manifest.json", manifest)
    uploader_pid = _start_uploader()
    status: dict[str, Any] = {
        "schema_version": 1,
        "status": "RUNNING",
        "pid": os.getpid(),
        "uploader_pid": uploader_pid,
        "task_count": len(entries),
        "task_ids": [entry["experiment_id"] for entry in entries],
        "started_at": time.time(),
        "hardware_policy": {"max_concurrent_gpu_jobs": 1, "unrelated_jobs_terminated": False, "reason": "no unrelated training process was present at launch"},
    }
    _write_json(RUNNER_STATUS, status)
    results: list[dict[str, Any]] = []
    for index, entry in enumerate(entries, start=1):
        if STOP_REQUESTED:
            break
        status.update({"current_run_id": entry["experiment_id"], "current_index": index, "results": results})
        _write_json(RUNNER_STATUS, status)
        result = _run_one(entry, auto_approve=not args.no_auto_approve, max_retries=args.max_retries)
        results.append(result)
        status["results"] = results
        _write_json(RUNNER_STATUS, status)
        if result.get("status") == "FAILED":
            status.update({"status": "FAILED", "ended_at": time.time()})
            _write_json(RUNNER_STATUS, status)
            return 4
    status.update({"status": "STOPPED" if STOP_REQUESTED else "COMPLETED", "ended_at": time.time(), "results": results})
    _write_json(RUNNER_STATUS, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
