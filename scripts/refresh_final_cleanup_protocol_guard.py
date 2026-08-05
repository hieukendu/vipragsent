from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    from _bootstrap import ROOT
    from readiness_utils import commit_protected_manifest, git_sha, worktree_protected_manifest
except ModuleNotFoundError:
    from scripts._bootstrap import ROOT
    from scripts.readiness_utils import (
        commit_protected_manifest,
        git_sha,
        worktree_protected_manifest,
    )
from vipragsent.atomic import atomic_write_json, atomic_write_text
from vipragsent.orchestration.inventory import build_expected_runs
from vipragsent.protocol import compare_frozen_hashes


def _baseline_inventory(root: Path, commit: str) -> tuple[int, str]:
    result = subprocess.run(["git", "show", f"{commit}:reports/expected_experiment_runs.json"], cwd=root, capture_output=True, text=True, check=True)
    payload = json.loads(result.stdout)
    return len(payload["rows"]), str(payload["inventory_hash"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Hash protected inputs before and after readiness cleanup")
    parser.add_argument("--before-commit", required=True)
    args = parser.parse_args()
    before = commit_protected_manifest(ROOT, args.before_commit)
    after = worktree_protected_manifest(ROOT)
    baseline_count, baseline_hash = _baseline_inventory(ROOT, args.before_commit)
    current = build_expected_runs(ROOT)
    changed_paths = set(subprocess.run(["git", "diff", "--name-only", args.before_commit], cwd=ROOT, capture_output=True, text=True, check=True).stdout.splitlines())
    protected_changed = before["manifest_sha256"] != after["manifest_sha256"]
    frozen = compare_frozen_hashes(ROOT)
    guard = {
        "schema_version": 1,
        "before_commit": args.before_commit,
        "after_commit_or_worktree_head": git_sha(ROOT),
        "before_manifest": before,
        "after_manifest": after,
        "frozen_data_changed": not frozen["unchanged"],
        "labels_changed": any("label" in path.casefold() for path in changed_paths),
        "seeds_changed": any("seed" in path.casefold() for path in changed_paths),
        "threshold_protocol_changed": any("threshold" in path.casefold() for path in changed_paths),
        "generation_protocol_changed": any("generation" in path.casefold() for path in changed_paths),
        "inventory_changed": baseline_count != len(current["rows"]) or baseline_hash != current["inventory_hash"],
        "model_training_code_changed": any(path.startswith(("src/vipragsent/models/", "src/vipragsent/training/")) for path in changed_paths),
        "evaluation_semantics_changed": any(path.startswith("src/vipragsent/evaluation/") for path in changed_paths),
        "protected_source_manifest_changed": protected_changed,
    }
    guard["status"] = "PASS" if not any(guard[key] for key in guard if key.endswith("changed")) else "FAIL"
    atomic_write_json(ROOT / "reports/final_cleanup_protocol_guard.json", guard)
    atomic_write_text(ROOT / "reports/final_cleanup_protocol_guard.md", "\n".join([
        "# Final cleanup protocol guard",
        "",
        f"- Status: `{guard['status']}`",
        f"- Before commit: `{args.before_commit}`",
        f"- After worktree HEAD: `{guard['after_commit_or_worktree_head']}`",
        f"- Protected source manifest before: `{before['manifest_sha256']}`",
        f"- Protected source manifest after: `{after['manifest_sha256']}`",
        f"- Inventory before/after: `{baseline_count}/{len(current['rows'])}`",
        "",
        "All required scientific and protected-source change guards:",
        "",
        *[f"- {key}: `{str(value).lower()}`" for key, value in guard.items() if key.endswith("changed")],
        "",
    ]))
    print(guard["status"])
    return 0 if guard["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
