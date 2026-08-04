from __future__ import annotations

import argparse
import json

from _bootstrap import ROOT
from vipragsent.artifacts.exporter import export_fixture_artifacts, export_production_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="Export ViPragSent artifacts")
    parser.add_argument("--mode", choices=("fixture", "production"), required=True)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    if args.mode == "fixture":
        result = export_fixture_artifacts(repo_root=ROOT, run_id=args.run_id or "fixture")
    else:
        result = export_production_artifacts(repo_root=ROOT, run_id=args.run_id or "full")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
