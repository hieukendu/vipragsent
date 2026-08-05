from __future__ import annotations

import argparse
import json

from _bootstrap import ROOT
from vipragsent.orchestration.rationale_promotion import promote_approved_rationales


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote one explicitly approved Azure rationale run")
    parser.add_argument("--source-run-id", default="azure_rationale_generation")
    args = parser.parse_args()
    try:
        report = promote_approved_rationales(ROOT, source_run_id=args.source_run_id)
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"}, indent=2, ensure_ascii=False))
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
