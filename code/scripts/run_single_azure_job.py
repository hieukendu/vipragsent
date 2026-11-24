from __future__ import annotations

import argparse
import json

from _bootstrap import ROOT
from vipragsent.orchestration.sequential import execute_sequential_run, find_azure_job


def inference_entry(job_id: str, inference_run_index: int = 1) -> dict:
    entry = find_azure_job(job_id)
    if inference_run_index not in (1, 2, 3):
        raise ValueError("Inference run index must be 1, 2 or 3")
    if inference_run_index != 1 and entry["job_type"] not in ("pragmatic_zero_shot", "pragmatic_8_shot"):
        raise ValueError("Three inference runs apply only to the Table 1 GPT prompts")
    if entry["job_type"] in ("pragmatic_zero_shot", "pragmatic_8_shot"):
        entry["inference_run_index"] = inference_run_index
        if inference_run_index != 1:
            entry["job_id"] = f"{job_id}_replicate_{inference_run_index}"
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description="Run exactly one Azure job with review gating")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--inference-run-index", type=int, choices=(1, 2, 3), default=1,
                        help="Table 1 GPT output index; each index has a separate run directory and API cache")
    parser.add_argument("--stage", choices=("preflight", "execute_api_job", "validate_responses", "export_artifacts", "validate_artifacts", "generate_review_summary", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the single-job plan without an API request")
    parser.add_argument("--fixture", action="store_true", help="Use a fake Azure transport and synthetic stage markers")
    args = parser.parse_args()
    entry = inference_entry(args.job_id, args.inference_run_index)
    state, exit_code = execute_sequential_run(ROOT, entry, kind="azure", stage=args.stage, run_id=str(entry["job_id"]), resume=args.resume, dry_run=args.dry_run, fixture=args.fixture)
    if args.dry_run:
        return exit_code
    review_root = ROOT / ("runs/fixture/results/runs" if args.fixture else "results/runs") / str(entry["job_id"])
    review_path = review_root / "review_summary.md"
    if review_path.exists():
        print(review_path.read_text(encoding="utf-8"))
    else:
        print(json.dumps(state, indent=2, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
