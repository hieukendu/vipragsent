from __future__ import annotations

import argparse
import json

from _bootstrap import ROOT
from vipragsent.orchestration.sequential import execute_sequential_run, find_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description="Run exactly one ViPragSent experiment with review gating")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--stage", choices=("preflight", "train", "train_or_reuse", "train_or_run", "execute_components", "combine_component_predictions", "evaluate_dev", "freeze_selection", "freeze_component_selection", "evaluate_test", "train_generation", "generate_dev", "parse_dev", "generate_test", "parse_test", "resolve_approved_source", "evaluate_external_tests", "evaluate_reused_test", "validate_source_predictions", "extract_pragmatic_calibration", "extract_learning_history", "export_artifacts", "validate_artifacts", "generate_review_summary", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the single-run plan without model, GPU, or data execution")
    parser.add_argument("--fixture", action="store_true", help="Use only synthetic stage markers for setup validation")
    args = parser.parse_args()
    entry = find_experiment(ROOT, args.experiment_id)
    state, exit_code = execute_sequential_run(ROOT, entry, kind="experiment", stage=args.stage, run_id=str(entry["experiment_id"]), resume=args.resume, dry_run=args.dry_run, fixture=args.fixture)
    if args.dry_run:
        return exit_code
    review_root = ROOT / ("runs/fixture/results/runs" if args.fixture else "results/runs") / str(entry["experiment_id"])
    review_path = review_root / "review_summary.md"
    if review_path.exists():
        print(review_path.read_text(encoding="utf-8"))
    else:
        print(json.dumps(state, indent=2, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
