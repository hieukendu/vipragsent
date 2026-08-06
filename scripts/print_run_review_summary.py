from __future__ import annotations

import argparse
import json

from _bootstrap import ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description="Print one sequential run review summary without approving it")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run_root = ROOT / "results/runs" / args.run_id
    if not run_root.exists():
        fixture_root = ROOT / "runs/fixture/results/runs" / args.run_id
        if fixture_root.exists():
            run_root = fixture_root
    summary_path = run_root / "review_summary.md"
    approval_path = run_root / "approval_status.json"
    if not summary_path.exists() or not approval_path.exists():
        print(json.dumps({"run_id": args.run_id, "status": "BLOCKED", "error": "review artifacts are missing"}, indent=2))
        return 2
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    print(summary_path.read_text(encoding="utf-8"))
    print("\n## Approval status\n")
    print(json.dumps(approval, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
