from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from _bootstrap import ROOT
from vipragsent.atomic import atomic_write_json
from vipragsent.config import load_yaml
from vipragsent.orchestration.dag import load_master_dag
from vipragsent.orchestration.handlers import (
    HandlerEnvironment,
    build_execution_context,
    build_handler_registry,
)
from vipragsent.orchestration.preflight import run_preflight
from vipragsent.orchestration.status import NodeStatus, RunExitCode


def _write_preflight_report(root: Path, result: object) -> None:
    payload = result.as_dict() if hasattr(result, "as_dict") else result
    payload = {"FULL_RUN_PREFLIGHT_PASS": bool(payload.get("passed")), **payload}
    atomic_write_json(root / "reports/full_runtime_preflight.json", payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the ViPragSent experiment DAG")
    parser.add_argument("--config", default="configs/master_run.yaml")
    parser.add_argument("--mode", choices=("fixture", "full"), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--enable-global-full-dag", action="store_true", help="Explicit future override for the disabled global production DAG")
    parser.add_argument("--preflight-only", action="store_true", help="Validate full runtime without creating DAG state")
    args = parser.parse_args()

    if args.preflight_only and args.mode != "full":
        parser.error("--preflight-only is valid only with --mode full")

    master_config = load_yaml(ROOT / args.config)
    if args.mode == "full" and not args.preflight_only:
        print("BLOCKED: global full-DAG execution is disabled by the sequential_review_gated policy.")
        if args.enable_global_full_dag:
            print("The legacy override flag is intentionally rejected; execute exactly one run through the sequential CLI.")
        print("Run exactly one experiment with scripts/run_single_experiment.py --experiment-id <EXPERIMENT_ID>.")
        print("Run exactly one Azure job with scripts/run_single_azure_job.py --job-id <AZURE_JOB_ID>.")
        print("After review, record explicit approval before using scripts/aggregate_approved_runs.py.")
        return RunExitCode.BLOCKED

    if args.mode == "full":
        preflight = run_preflight(ROOT, mode="full")
        _write_preflight_report(ROOT, preflight)
        if args.preflight_only:
            print(json.dumps(preflight.as_dict(), indent=2))
            return preflight.exit_code
        if not preflight.passed:
            print(json.dumps(preflight.as_dict(), indent=2))
            return preflight.exit_code

    run_id = "fixture" if args.mode == "fixture" else "full"
    fixture_root = ROOT / "runs" / "fixture"
    if args.mode == "fixture" and not args.resume and fixture_root.exists():
        shutil.rmtree(fixture_root)
    artifact_root = fixture_root if args.mode == "fixture" else ROOT / "experiment_artifacts"
    context = build_execution_context(ROOT, mode=args.mode, run_id=run_id, artifact_root=artifact_root)
    environment = HandlerEnvironment(ROOT, context)
    matrix_path = ROOT / master_config.get("matrix", args.config)
    dag = load_master_dag(matrix_path)
    state_path = fixture_root / "dag_state.json" if args.mode == "fixture" else ROOT / "runs" / "full" / "dag_state.json"
    state = dag.run(state_path, build_handler_registry(environment), resume=args.resume, force=args.force)
    print(json.dumps(state, indent=2, ensure_ascii=False))
    return {
        NodeStatus.PASS.value: RunExitCode.SUCCESS,
        NodeStatus.BLOCKED.value: RunExitCode.BLOCKED,
        NodeStatus.FAIL.value: RunExitCode.EXECUTION_FAILURE,
    }.get(state.get("status"), RunExitCode.EXECUTION_FAILURE)


if __name__ == "__main__":
    raise SystemExit(main())
