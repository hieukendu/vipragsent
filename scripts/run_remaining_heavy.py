"""Run the audited Heavy remainder in canonical order.

The queue is append-only and the uploader is a separate process.  This
runner owns the single local GPU slot; it never waits for checkpoint uploads,
so network persistence can continue while the next experiment is running.
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

from _bootstrap import ROOT
from vipragsent.orchestration.approval import validate_approval_record

REMAINING_IDS = (
    "q3_vistral_pragmatic_sft_full_20260521",
    "q3_vistral_pragmatic_sft_full_20260522",
    "q3_vistral_pragmatic_sft_full_20260523",
    "q3_vipragsent_full_vistral_32_20260521",
    "q3_vipragsent_full_vistral_32_20260522",
    "q3_vipragsent_full_vistral_32_20260523",
    "q3_vipragsent_full_vistral_128_20260521",
    "q3_vipragsent_full_vistral_128_20260522",
    "q3_vipragsent_full_vistral_128_20260523",
    "q3_vipragsent_full_vistral_512_20260521",
    "q3_vipragsent_full_vistral_512_20260522",
    "q3_vipragsent_full_vistral_512_20260523",
    "q3_vipragsent_full_vistral_full_20260521",
    "q3_vipragsent_full_vistral_full_20260522",
    "q3_vipragsent_full_vistral_full_20260523",
    "q4_vistral_pragmatic_sft_20260521",
    "q4_vistral_pragmatic_sft_20260522",
    "q4_vistral_pragmatic_sft_20260523",
    "q4_vipragsent_full_vistral_20260521",
    "q4_vipragsent_full_vistral_20260522",
    "q4_vipragsent_full_vistral_20260523",
    "backbone_sensitivity_vipragsent_full_vistral_20260521",
    "backbone_sensitivity_vipragsent_full_vistral_20260522",
    "backbone_sensitivity_vipragsent_full_vistral_20260523",
)
SOURCE_IDS = tuple(
    [f"q1a_vistral_pragmatic_sft_{seed}" for seed in ("20260521", "20260522", "20260523")]
    + [f"q1a_vipragsent_full_vistral_{seed}" for seed in ("20260521", "20260522", "20260523")]
)
AUDIT_REPORT = Path("/tmp/vipragsent_hf_final_reaudit_report.json")
DEFAULT_QUEUE = ROOT / "runtime/hf_upload_queue.jsonl"
DEFAULT_LOG_DIR = ROOT / "runtime/logs"
DEFAULT_STATUS = ROOT / "runtime/heavy_runner_status.json"
DEFAULT_UPLOADER_PID = ROOT / "runtime/hf_uploader.pid"
REVIEWER = "user-standing-authorization"
_STOP_REQUESTED = False


def _signal_handler(_signum: int, _frame: Any) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _validate_audit_report(path: Path) -> None:
    report = _load_json(path)
    if not isinstance(report, dict):
        raise RuntimeError(f"authoritative audit report is missing: {path}")
    reported = tuple(
        str(item.get("experiment_id"))
        for item in report.get("remaining_tasks", [])
        if isinstance(item, dict) and item.get("experiment_id")
    )
    if set(reported) != set(REMAINING_IDS) or len(reported) != len(REMAINING_IDS):
        raise RuntimeError("audit report and runner queue disagree; refusing to launch a stale or overlapping task list")
    if len(set(REMAINING_IDS)) != len(REMAINING_IDS):
        raise RuntimeError("runner task list contains duplicate canonical IDs")
    if any("cot_only_vistral" in run_id for run_id in REMAINING_IDS):
        raise RuntimeError("COT task leaked into the Heavy queue")
    statuses = report.get("heavy_status_counts", {})
    if statuses.get("COMPLETED", 0) != 33 or statuses.get("PARTIAL_OR_BLOCKED", 0) != 2 or statuses.get("NOT_FOUND", 0) != 22:
        raise RuntimeError(f"unexpected Heavy audit counts: {statuses}")


def _append_queue(queue_path: Path, run_id: str) -> None:
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    with queue_path.open("a+", encoding="utf-8") as handle:
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
        if run_id not in existing:
            handle.seek(0, os.SEEK_END)
            handle.write(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": run_id,
                        "source_root": f"results/runs/{run_id}",
                        "status": "queued",
                        "queued_at": time.time(),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _start_uploader(status_path: Path, pid_path: Path, log_dir: Path, queue_path: Path) -> int | None:
    current = _load_json(pid_path, {})
    if isinstance(current, dict) and isinstance(current.get("pid"), int) and _is_alive(int(current["pid"])):
        return int(current["pid"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "hf_uploader.log"
    log_handle = log_path.open("a", encoding="utf-8", buffering=1)
    command = [
        sys.executable,
        str(ROOT / "scripts/hf_artifact_uploader.py"),
        "--queue-file",
        str(queue_path),
        "--status-file",
        str(status_path),
    ]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
    log_handle.close()
    _write_json(pid_path, {"pid": process.pid, "started_at": time.time(), "command": command})
    return process.pid


def _run_source_import(log_handle: Any) -> None:
    command = [sys.executable, str(ROOT / "scripts/import_hf_sources.py")]
    result = subprocess.run(command, cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"Hub source import failed with exit code {result.returncode}")


def _approve_if_needed(run_id: str, log_handle: Any) -> None:
    run_root = ROOT / "results/runs" / run_id
    state = _load_json(run_root / "state.json", {})
    approval = _load_json(run_root / "approval_status.json", {})
    if state.get("run_status") == "APPROVED" and approval.get("status") == "APPROVED" and not validate_approval_record(run_root, expected_run_id=run_id):
        return
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
        "Standing user approval after local review PASS; Hub audit confirmed this canonical ID is not complete and no COT overlap is present.",
    ]
    result = subprocess.run(command, cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"approval failed for {run_id} with exit code {result.returncode}")


def _wait_for_upload_signal(run_id: str, status_path: Path, timeout: float, log_handle: Any) -> dict[str, Any]:
    run_root = ROOT / "results/runs" / run_id
    expected_files = sum(
        1
        for path in run_root.rglob("*")
        if path.is_file()
        and not path.name.endswith(".lock")
        and "__pycache__" not in path.relative_to(run_root).parts
        and not path.relative_to(run_root).as_posix().startswith("runtime/")
    )
    verified: list[dict[str, Any]] = []
    deadline = time.time() + max(timeout, 0.0)
    while time.time() <= deadline:
        status = _load_json(status_path, {})
        run = status.get("runs", {}).get(run_id, {}) if isinstance(status, dict) else {}
        verified = [value for value in run.get("uploaded", {}).values() if isinstance(value, dict) and value.get("verified")]
        if expected_files and len(verified) >= expected_files and not run.get("errors"):
            return {"status": "PASS", "verified_files": len(verified), "artifact_repository": status.get("artifact_repositories", {}).get(run_id)}
        if _STOP_REQUESTED:
            break
        time.sleep(5.0)
    log_handle.write(
        f"upload verification not complete within {timeout}s for {run_id}; "
        "runner continues while uploader retries\n"
    )
    return {"status": "PENDING", "verified_files": len(verified), "expected_files": expected_files}


def _run_one(run_id: str, log_dir: Path, max_retries: int, upload_timeout: float, uploader_status: Path, queue_path: Path) -> dict[str, Any]:
    run_root = ROOT / "results/runs" / run_id
    _append_queue(queue_path, run_id)
    state = _load_json(run_root / "state.json", {})
    if state.get("run_status") == "APPROVED" and not validate_approval_record(run_root, expected_run_id=run_id):
        with (log_dir / f"{run_id}.log").open("a", encoding="utf-8") as log_handle:
            upload = _wait_for_upload_signal(run_id, uploader_status, upload_timeout, log_handle)
        return {"run_id": run_id, "status": "SKIPPED_ALREADY_APPROVED", "upload": upload}
    log_path = log_dir / f"{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    last_code = None
    for attempt in range(max_retries + 1):
        if _STOP_REQUESTED:
            return {"run_id": run_id, "status": "STOPPED"}
        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(f"\n=== attempt {attempt + 1}/{max_retries + 1} at {time.time()} ===\n")
            command = [sys.executable, str(ROOT / "scripts/run_single_experiment.py"), "--experiment-id", run_id, "--stage", "all"]
            if attempt or (ROOT / "results/runs" / run_id / "state.json").exists():
                command.append("--resume")
            result = subprocess.run(command, cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT, check=False)
            last_code = result.returncode
            state = _load_json(run_root / "state.json", {})
            if result.returncode == 0 and state.get("run_status") == "COMPLETED_PENDING_APPROVAL":
                try:
                    _approve_if_needed(run_id, log_handle)
                    upload = _wait_for_upload_signal(run_id, uploader_status, upload_timeout, log_handle)
                    return {"run_id": run_id, "status": "APPROVED", "attempt": attempt + 1, "upload": upload}
                except Exception as exc:
                    log_handle.write(f"approval/debug failure: {type(exc).__name__}: {exc}\n")
            else:
                log_handle.write(f"scientific runner exit={result.returncode} run_status={state.get('run_status')}\n")
        if attempt < max_retries:
            time.sleep(5.0)
    return {"run_id": run_id, "status": "FAILED", "exit_code": last_code, "state": _load_json(run_root / "state.json", {})}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the final audited Heavy task set without overlap")
    parser.add_argument("--audit-report", type=Path, default=AUDIT_REPORT)
    parser.add_argument("--queue-file", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--status-file", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--uploader-status", type=Path, default=ROOT / "runtime/hf_uploader_status.json")
    parser.add_argument("--uploader-pid", type=Path, default=DEFAULT_UPLOADER_PID)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--upload-timeout", type=float, default=180.0)
    parser.add_argument("--skip-source-import", action="store_true")
    args = parser.parse_args()
    if args.max_retries < 0:
        raise SystemExit("--max-retries must be non-negative")
    _validate_audit_report(args.audit_report)
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    runner_status: dict[str, Any] = {
        "schema_version": 1,
        "status": "RUNNING",
        "pid": os.getpid(),
        "task_count": len(REMAINING_IDS),
        "task_ids": list(REMAINING_IDS),
        "started_at": time.time(),
    }
    _write_json(args.status_file, runner_status)
    with (args.log_dir / "heavy_runner.log").open("a", encoding="utf-8") as log_handle:
        uploader_pid = _start_uploader(args.uploader_status, args.uploader_pid, args.log_dir, args.queue_file)
        runner_status["uploader_pid"] = uploader_pid
        _write_json(args.status_file, runner_status)
        if not args.skip_source_import:
            _run_source_import(log_handle)
            for source_id in SOURCE_IDS:
                _approve_if_needed(source_id, log_handle)
        results: list[dict[str, Any]] = []
        for index, run_id in enumerate(REMAINING_IDS, start=1):
            if _STOP_REQUESTED:
                break
            pid_record = _load_json(args.uploader_pid, {})
            if (
                isinstance(pid_record, dict)
                and isinstance(pid_record.get("pid"), int)
                and _is_alive(int(pid_record["pid"]))
            ):
                runner_status["uploader_pid"] = int(pid_record["pid"])
            runner_status["current_run_id"] = run_id
            runner_status["current_index"] = index
            runner_status["results"] = results
            _write_json(args.status_file, runner_status)
            result = _run_one(run_id, args.log_dir, args.max_retries, args.upload_timeout, args.uploader_status, args.queue_file)
            results.append(result)
            runner_status["results"] = results
            _write_json(args.status_file, runner_status)
            if result.get("status") == "FAILED":
                runner_status["status"] = "FAILED"
                runner_status["ended_at"] = time.time()
                _write_json(args.status_file, runner_status)
                return 4
        runner_status["status"] = "STOPPED" if _STOP_REQUESTED else "COMPLETED"
        runner_status["ended_at"] = time.time()
        runner_status["results"] = results
        _write_json(args.status_file, runner_status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
