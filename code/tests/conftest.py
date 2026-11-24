from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def pytest_configure(config):
    temporary = Path(config.option.basetemp or ROOT / ".work/pytest_tmp").resolve()
    if not temporary.is_relative_to(ROOT.resolve()):
        raise pytest.UsageError("Test temporary files must stay inside the code checkout")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    config.option.basetemp = str(temporary)
