from __future__ import annotations

import argparse
import json

from _bootstrap import ROOT
from vipragsent.orchestration.aggregation import aggregate_approved_scope


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate one explicitly approved research-question scope")
    parser.add_argument("--research-question", choices=("Q1a", "Q1b", "Q2", "Q3", "Q4", "backbone_sensitivity", "all"), required=True)
    args = parser.parse_args()
    report = aggregate_approved_scope(ROOT, args.research_question)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0 if report.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
