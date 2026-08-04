from __future__ import annotations

import json
from pathlib import Path

from vipragsent.constants import ALL_LABEL_KEYS
from vipragsent.phase import PhaseHandoff


def test_project_state_and_label_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    state = json.loads((root / "PROJECT_STATE.json").read_text(encoding="utf-8"))
    assert state["project"] == "ViPragSent"
    labels = json.loads((root / "configs/labels.json").read_text(encoding="utf-8"))
    assert tuple(labels["canonical_keys"]) == ALL_LABEL_KEYS


def test_phase_handoff_rejects_unknown_status(tmp_path: Path) -> None:
    try:
        PhaseHandoff("00", "UNKNOWN")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown phase status should fail")
