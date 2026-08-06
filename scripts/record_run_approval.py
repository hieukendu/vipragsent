from __future__ import annotations

import argparse
import json

from _bootstrap import ROOT
from vipragsent.orchestration.approval import record_run_approval


def main() -> int:
    parser = argparse.ArgumentParser(description="Record one explicit later-task approval decision")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--decision", choices=("approve", "reject"), required=True)
    parser.add_argument("--review-note", required=True)
    parser.add_argument("--reviewer", "--approved-by", dest="reviewer", required=True, help="Literal reviewer label supplied by the user")
    args = parser.parse_args()
    try:
        result = record_run_approval(ROOT, args.run_id, decision=args.decision, review_note=args.review_note, reviewer=args.reviewer)
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
