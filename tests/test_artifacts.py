from __future__ import annotations

from pathlib import Path

from vipragsent.artifacts.schemas import validate_artifact_tree


def test_fixture_artifacts_match_locked_columns() -> None:
    root = Path(__file__).resolve().parents[1]
    assert validate_artifact_tree(root / "experiment_artifacts") == []
    assert not any(path.name.lower().startswith("figure5") for path in (root / "experiment_artifacts").rglob("*"))
    assert (root / "experiment_artifacts/manual/error_analysis_candidates.csv").exists()
    assert (root / "experiment_artifacts/manual/qualitative_candidates.jsonl").exists()
