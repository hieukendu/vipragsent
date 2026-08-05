from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def pytest_sessionstart(session) -> None:
    """Create the ignored synthetic DAG fixture required by contract tests."""
    state_path = ROOT / "runs/fixture/dag_state.json"
    manifest_path = ROOT / "runs/fixture/FIXTURE_VALIDATION_MANIFEST.json"
    if state_path.exists() and manifest_path.exists():
        return
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_all_experiments.py"),
            "--config",
            "configs/master_run.yaml",
            "--mode",
            "fixture",
        ],
        cwd=ROOT,
        check=True,
    )
