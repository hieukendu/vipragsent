from __future__ import annotations

import argparse
import json

from _bootstrap import ROOT
from vipragsent.runtime.phase15_state import reconcile_phase15_state


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconcile official project state from verified Phase 15 artifacts")
    parser.add_argument("--allow-report-only", action="store_true", help="Do not require the local model snapshot; use only for CI/report regeneration")
    args = parser.parse_args()
    report = reconcile_phase15_state(ROOT, require_local_snapshot=not args.allow_report_only)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
