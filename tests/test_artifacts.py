from __future__ import annotations

from pathlib import Path

from vipragsent.artifacts.exporter import export_fixture_artifacts
from vipragsent.artifacts.schemas import validate_artifact_tree


def test_fixture_artifacts_match_locked_columns(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output_root = tmp_path / "fixture_artifacts"
    manifest = export_fixture_artifacts(repo_root=root, output_root=output_root)
    artifact_root = output_root / "artifacts"
    assert validate_artifact_tree(artifact_root) == []
    assert not any(path.name.lower().startswith("figure5") for path in artifact_root.rglob("*"))
    assert (artifact_root / "manual/error_analysis_candidates.csv").exists()
    assert (artifact_root / "manual/qualitative_candidates.jsonl").exists()
    assert manifest["core_experiments_ready"] is False
    assert not (root / "FINAL_EXPERIMENT_MANIFEST.json").exists()
