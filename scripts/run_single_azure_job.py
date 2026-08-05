from __future__ import annotations

import argparse
import json

from _bootstrap import ROOT
from vipragsent.orchestration.sequential import execute_sequential_run, find_azure_job


def main() -> int:
    parser = argparse.ArgumentParser(description="Run exactly one Azure job with review gating")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--stage", choices=("preflight", "train_or_run", "evaluate_dev", "freeze_selection", "evaluate_test", "export_artifacts", "validate_artifacts", "generate_review_summary", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the single-job plan without an API request")
    parser.add_argument("--fixture", action="store_true", help="Use a fake Azure transport and synthetic stage markers")
    args = parser.parse_args()
    entry = find_azure_job(args.job_id)
    state, exit_code = execute_sequential_run(ROOT, entry, kind="azure", stage=args.stage, run_id=str(entry["job_id"]), resume=args.resume, dry_run=args.dry_run, fixture=args.fixture)
    if args.dry_run:
        return exit_code
    review_path = ROOT / "results/runs" / str(entry["job_id"]) / "review_summary.md"
    if review_path.exists():
        print(review_path.read_text(encoding="utf-8"))
    else:
        print(json.dumps(state, indent=2, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
