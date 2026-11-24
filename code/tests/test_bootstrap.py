from __future__ import annotations

import json
from pathlib import Path
import pytest

from vipragsent.constants import ALL_LABEL_KEYS
from vipragsent.phase import PhaseHandoff


def test_project_state_and_label_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    labels = json.loads((root / "configs/labels.json").read_text(encoding="utf-8"))
    assert tuple(labels["canonical_keys"]) == ALL_LABEL_KEYS


def test_private_project_state() -> None:
    path = Path(__file__).resolve().parents[1] / "PROJECT_STATE.json"
    if not path.is_file():
        pytest.skip('Private workstation state is not part of the public release')
    assert json.loads(path.read_text(encoding='utf-8'))['project'] == 'ViPragSent'


def test_phase_handoff_rejects_unknown_status(tmp_path: Path) -> None:
    try:
        PhaseHandoff("00", "UNKNOWN")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown phase status should fail")
