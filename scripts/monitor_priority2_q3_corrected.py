"""Periodic read-only monitor for the corrected Priority 2 Q3 campaign."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = ROOT / "runtime/priority2_q3_corrected_monitor.log"
DEFAULT_STOP = ROOT / "runtime/STOP_PRIORITY2_Q3_CORRECTED_MONITOR"
RUNNER_STATUS = ROOT / "runtime/priority2_q3_corrected_runner_status.json"
UPLOADER_STATUS = ROOT / "runtime/priority2_q3_corrected_hf_uploader_status.json"
CAMPAIGN_PREFIX = "priority2_q3_corrected_v2_"
OWN_COMMAND_MARKERS = (
    "run_priority2_q3_corrected.py",
    "hf_artifact_uploader.py",
    "monitor_priority2_q3_corrected.py",
)


def load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except (OSError, json.JSONDecodeError):
        return default


def command_output(command: list[str], timeout: int = 10) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"ERROR: {exc}"
    output = (result.stdout or result.stderr or "").strip()
    return output if output else f"exit={result.returncode}"


def process_snapshot() -> dict[str, Any]:
    raw = command_output(["ps", "-eo", "pid,ppid,stat,pcpu,pmem,etime,args"], timeout=15)
    rows: list[str] = []
    unrelated: list[str] = []
    for line in raw.splitlines()[1:]:
        lowered = line.lower()
        if not any(token in lowered for token in ("python", "torch", "accelerat", "transformer", "vllm", "cuda")):
            continue
        if "defunct" in lowered:
            continue
        rows.append(line)
        if not any(marker in line for marker in OWN_COMMAND_MARKERS):
            unrelated.append(line)
    return {"ai_processes": rows, "unrelated_candidates": unrelated}


def current_run_summary(run_id: str | None) -> dict[str, Any]:
    if not run_id:
        return {}
    run = ROOT / "results" / "runs" / run_id
    state = load_json(run / "state.json", {}) or {}
    manifest = load_json(run / "run_manifest.json", {}) or {}
    config = load_json(run / "training" / "resolved_training_config.json", {}) or {}
    device = load_json(run / "training" / "device_report.json", {}) or {}
    checkpoints = []
    checkpoint_dir = run / "_engine_checkpoints" / "model"
    if checkpoint_dir.exists():
        for path in sorted(checkpoint_dir.glob("*.pt"), key=lambda item: item.stat().st_mtime):
            checkpoints.append({"name": path.name, "size": path.stat().st_size, "mtime": path.stat().st_mtime})
    return {
        "run_id": run_id,
        "run_status": state.get("run_status"),
        "approval_status": state.get("approval_status"),
        "stages": state.get("stages", {}),
        "seed": manifest.get("seed"),
        "budget": manifest.get("budget"),
        "system": manifest.get("system_id") or manifest.get("system"),
        "model_revision": manifest.get("model_revision"),
        "model_family": config.get("model_family"),
        "config_hash": config.get("config_hash"),
        "effective_batch_size": config.get("effective_batch_size"),
        "physical_batch_size": config.get("physical_batch_size"),
        "precision": config.get("precision"),
        "selected_device": device.get("selected_device"),
        "device_status": device.get("status"),
        "checkpoints": checkpoints[-4:],
    }


def uploader_summary() -> dict[str, Any]:
    status = load_json(UPLOADER_STATUS, {}) or {}
    runs = status.get("runs", {}) if isinstance(status, dict) else {}
    last_scan = status.get("last_scan", {}) if isinstance(status, dict) else {}
    scan_results = last_scan.get("results", []) if isinstance(last_scan, dict) else []
    compact: dict[str, Any] = {}
    for run_id, payload in runs.items():
        if not isinstance(payload, dict):
            continue
        compact[run_id] = {
            "status": None,
            "pending": None,
            "verified_file_count": None,
            "error_count": len(payload.get("errors", []) or []),
        }
    for payload in scan_results:
        if not isinstance(payload, dict) or not payload.get("run_id"):
            continue
        run_id = str(payload["run_id"])
        compact[run_id] = {
            "status": payload.get("status"),
            "pending": payload.get("pending"),
            "verified_file_count": payload.get("verified_file_count"),
            "local_file_count": payload.get("local_file_count"),
            "uploaded_now": payload.get("uploaded_now"),
            "receipt": payload.get("receipt"),
            "error_count": len((compact.get(run_id) or {}).get("errors", []) or []),
        }
    return {
        "status": status.get("status"),
        "pid": status.get("pid"),
        "run_count": len(compact),
        "runs": compact,
        "last_scan_at": last_scan.get("at") if isinstance(last_scan, dict) else None,
        "unusable_repositories": status.get("unusable_artifact_repositories", []),
    }


def snapshot() -> dict[str, Any]:
    runner = load_json(RUNNER_STATUS, {}) or {}
    run_id = runner.get("current_run_id")
    return {
        "observed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "campaign_id": runner.get("campaign_id"),
        "runner": {
            "status": runner.get("status"),
            "current_index": runner.get("current_index"),
            "current_run_id": run_id,
            "run_count": runner.get("run_count"),
            "completed_entries": len(runner.get("results", []) or []),
        },
        "current_run": current_run_summary(run_id),
        "resources": {
            "cpu_memory": command_output(["free", "-h"]),
            "gpu_processes": command_output([
                "nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"
            ]),
            "processes": process_snapshot(),
        },
        "uploader": uploader_summary(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--stop-file", type=Path, default=DEFAULT_STOP)
    args = parser.parse_args()
    interval = max(300, min(args.interval_seconds, 600))
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    with args.log_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"monitor_started_at": time.time(), "interval_seconds": interval, "pid": os.getpid()}) + "\n")
        handle.flush()
        while not args.stop_file.exists():
            handle.write(json.dumps(snapshot(), ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            for _ in range(interval):
                if args.stop_file.exists():
                    break
                time.sleep(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
