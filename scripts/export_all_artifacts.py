from __future__ import annotations

from _bootstrap import ROOT
from vipragsent.artifacts.exporter import export_fixture_artifacts


if __name__ == "__main__":
    export_fixture_artifacts(repo_root=ROOT, run_id="export")
    print("Artifact export passed")
