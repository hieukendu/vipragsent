"""Run the corrected Priority 2 Q3 XLM-R-large budget queue sequentially."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from _bootstrap import ROOT
from vipragsent.atomic import atomic_write_json
from vipragsent.hashing import sha256_file
from vipragsent.orchestration.sequential import execute_sequential_run

SEEDS = (20260521, 20260522, 20260523)
BUDGETS = ("32", "64", "128", "256", "512", "full")
MODEL_REPOSITORY = "FacebookAI/xlm-roberta-large"
MODEL_REVISION = "c23d21b0620b635a76227c604d44e43a9f0ee389"
MASK_ROOT = ROOT / "data/processed/q3_low_resource_sarcasm_corrected"
CAMPAIGN_ID = "priority2-q3-corrected-v2-20261002"
RUN_PREFIX = "priority2_q3_corrected_v2"
RUNNER_STATUS = ROOT / "runtime/priority2_q3_corrected_runner_status.json"
LOG_DIR = ROOT / "runtime/priority2_q3_corrected_logs"
QUEUE_FILE = ROOT / "runtime/priority2_q3_corrected_hf_upload_queue.jsonl"
STOP_REQUESTED = False


def _signal_handler(_signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def _load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except (OSError, json.JSONDecodeError):
        return default


def _base_entry(run_id: str, *, seed: int, budget: str) -> dict[str, Any]:
    mask_path = MASK_ROOT / f"budget_{budget}_masks.csv"
    return {
        "experiment_id": run_id,
        "run_id": run_id,
        "research_question": "Q3",
        "system_id": "xlmr_followup_full",
        "display_name": run_id.replace("_", " "),
        "variant": "full",
        "backbone": "xlmr_large",
        "model_family": "xlmr_large",
        "model": MODEL_REPOSITORY,
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "preprocessing_name": "unicode_nfc",
        "preprocessing_version": "locked-v1",
        "seed": seed,
        "budget": budget,
        "execution_kind": "trainable",
        "execution_policy": "sequential_review_gated",
        "required_phase15_assets": "xlmr_large_cache;offline_smoke;frozen_batch;validated_gpu",
        "dependencies": "isolated_priority2_q3_corrected_lane",
        "split": "vipragsent_train_dev_test",
        "external_finetuning": False,
        "followup_lane": "priority2_q3_corrected",
        "q3_mask_path": mask_path.relative_to(ROOT).as_posix(),
        "q3_mask_hash": sha256_file(mask_path),
        "q3_rationale_policy": "sarcasm_only",
        "q3_protocol_path": "configs/experiments/q3/corrected_protocol.yaml",
        "selection_metric": "dev_sarcasm_binary_macro_f1",
        "_followup_manifest": "configs/experiments/q3/corrected_protocol.yaml",
        "_repository_root": str(ROOT),
    }


def build_entries() -> list[dict[str, Any]]:
    return [
        _base_entry(f"{RUN_PREFIX}_{budget}_{seed}", seed=seed, budget=budget)
        for budget in BUDGETS
        for seed in SEEDS
    ]


def _resource_snapshot() -> dict[str, Any]:
    def run(command: list[str]) -> str:
        try:
            return subprocess.run(command, capture_output=True, text=True, check=False, timeout=15).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            return ""

    ps = run(["ps", "-eo", "pid,stat,comm,args"])
    active_ai = [
        line for line in ps.splitlines()
        if any(token in line.lower() for token in ("python", "torch", "train", "accelerat", "transformer", "vllm", "cuda"))
        and "defunct" not in line.lower()
    ]
    return {
        "recorded_at": time.time(),
        "gpu_inventory": run(["nvidia-smi", "-L"]),
        "active_ai_processes": active_ai,
        "cpu_memory": run(["free", "-h"]),
        "policy": "one_priority2_gpu_process; no unrelated active AI process was terminated",
    }


def _append_upload_queue(entry: dict[str, Any]) -> None:
    QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing: set[str] = set()
    if QUEUE_FILE.exists():
        for line in QUEUE_FILE.read_text(encoding="utf-8").splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and payload.get("run_id"):
                existing.add(str(payload["run_id"]))
    if entry["run_id"] in existing:
        return
    with QUEUE_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "schema_version": 1,
            "run_id": entry["run_id"],
            "source_root": f"results/runs/{entry['run_id']}",
            "backbone": entry["backbone"],
            "status": "queued",
            "campaign_id": CAMPAIGN_ID,
            "queued_at": time.time(),
        }, sort_keys=True) + "\n")


def _approve(run_id: str, log_handle: Any) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts/record_run_approval.py"),
        "--run-id",
        run_id,
        "--decision",
        "approve",
        "--reviewer",
        "user-standing-authorization",
        "--review-note",
        "Standing user authorization for corrected Priority 2 Q3 after local validation PASS; all valid seeds are retained.",
    ]
    result = subprocess.run(command, cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"approval failed for {run_id}: exit={result.returncode}")


def _run_one(entry: dict[str, Any], *, auto_approve: bool) -> dict[str, Any]:
    run_id = str(entry["run_id"])
    run_root = ROOT / "results/runs" / run_id
    state = _load_json(run_root / "state.json", {}) or {}
    approval = _load_json(run_root / "approval_status.json", {}) or {}
    if state.get("run_status") == "APPROVED" and approval.get("status") == "APPROVED":
        _append_upload_queue(entry)
        return {"run_id": run_id, "status": "ALREADY_APPROVED"}
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{run_id}.log"
    command = [sys.executable, str(Path(__file__).resolve()), "--run-id", run_id]
    if (run_root / "state.json").exists():
        command.append("--resume")
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n=== run started {time.time()} ===\n")
        log_handle.write(json.dumps(_resource_snapshot(), ensure_ascii=False) + "\n")
        result = subprocess.run(command, cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT, check=False)
        state = _load_json(run_root / "state.json", {}) or {}
        if result.returncode == 0 and state.get("run_status") == "COMPLETED_PENDING_APPROVAL":
            if auto_approve:
                _approve(run_id, log_handle)
                _append_upload_queue(entry)
                return {"run_id": run_id, "status": "APPROVED"}
            return {"run_id": run_id, "status": "COMPLETED_PENDING_APPROVAL"}
        return {"run_id": run_id, "status": "FAILED", "exit_code": result.returncode, "run_status": state.get("run_status")}


def _run_single(entry: dict[str, Any], *, stage: str, resume: bool) -> int:
    state, exit_code = execute_sequential_run(
        ROOT,
        entry,
        kind="priority2_q3_corrected",
        stage=stage,
        run_id=str(entry["run_id"]),
        resume=resume,
    )
    print(json.dumps({"run_id": entry["run_id"], "run_status": state.get("run_status"), "exit_code": exit_code}, ensure_ascii=False))
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--stage", default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-auto-approve", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    entries = build_entries()
    by_id = {str(entry["run_id"]): entry for entry in entries}
    if args.list:
        print(json.dumps(entries, indent=2, ensure_ascii=False))
        return 0
    if args.run_id:
        if args.run_id not in by_id:
            raise SystemExit(f"unknown corrected Priority 2 Q3 run ID: {args.run_id}")
        return _run_single(by_id[args.run_id], stage=args.stage, resume=args.resume)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    status = {
        "schema_version": 1,
        "status": "RUNNING",
        "campaign_id": CAMPAIGN_ID,
        "run_count": len(entries),
        "run_ids": [entry["run_id"] for entry in entries],
        "seeds": list(SEEDS),
        "budgets": list(BUDGETS),
        "mask_root": str(MASK_ROOT.relative_to(ROOT)),
        "protocol": "configs/experiments/q3/corrected_protocol.yaml",
        "started_at": time.time(),
        "results": [],
    }
    atomic_write_json(RUNNER_STATUS, status)
    for index, entry in enumerate(entries, start=1):
        if STOP_REQUESTED:
            break
        status.update({"current_index": index, "current_run_id": entry["run_id"]})
        atomic_write_json(RUNNER_STATUS, status)
        result = _run_one(entry, auto_approve=not args.no_auto_approve)
        status["results"].append(result)
        atomic_write_json(RUNNER_STATUS, status)
        if result.get("status") == "FAILED":
            status.update({"status": "FAILED", "ended_at": time.time()})
            atomic_write_json(RUNNER_STATUS, status)
            return 4
    status.update({"status": "STOPPED" if STOP_REQUESTED else "COMPLETED", "ended_at": time.time()})
    atomic_write_json(RUNNER_STATUS, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
