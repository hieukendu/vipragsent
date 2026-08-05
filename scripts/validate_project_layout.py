from __future__ import annotations

import json
from pathlib import Path

REQUIRED_DIRS = [
    "data/input",
    "data/raw",
    "data/external",
    "data/processed",
    "data/manifests",
    "data/model_cache",
    "configs/azure",
    "configs/models",
    "configs/runtime",
    "configs/experiments",
    "src/vipragsent",
    "scripts",
    "tests",
    "runs",
    "checkpoints",
    "predictions",
    "results",
    "reports/phases",
    "prompts/sequential/phase15",
    "prompts/sequential/experiments",
    "prompts/sequential/azure",
    "prompts/sequential/aggregation",
    "schemas",
    "experiment_artifacts",
]
REQUIRED_FILES = [
    "pyproject.toml",
    "README.md",
    ".gitignore",
    ".env.example",
    "PROJECT_STATE.json",
    "configs/master_run.yaml",
    "configs/execution_policy.yaml",
    "configs/paper_roles.yaml",
    "configs/experiments/master_matrix.yaml",
    "configs/experiments/master_inventory.yaml",
    "configs/experiments/q4/pragmatic_calibration.yaml",
    "configs/models/model_registry.yaml",
    "scripts/run_single_experiment.py",
    "scripts/run_single_azure_job.py",
    "scripts/print_run_review_summary.py",
    "scripts/aggregate_approved_runs.py",
    "schemas/run_review_summary.schema.json",
    "schemas/run_approval_status.schema.json",
    "schemas/q4_pragmatic_calibration.schema.json",
]


def validate(root: str | Path = ".") -> list[str]:
    root_path = Path(root)
    missing = [item for item in REQUIRED_DIRS + REQUIRED_FILES if not (root_path / item).exists()]
    state_path = root_path / "PROJECT_STATE.json"
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("project") != "ViPragSent":
                missing.append("PROJECT_STATE.json:project")
        except json.JSONDecodeError:
            missing.append("PROJECT_STATE.json:valid-json")
    return missing


def main() -> int:
    missing = validate()
    if missing:
        print("Missing layout entries:")
        print("\n".join(f"- {item}" for item in missing))
        return 3
    print("Project layout validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
