"""Resolve the canonical dataset in a workspace or extracted software/data pair."""
from pathlib import Path


def default_release(code_root: Path | None = None) -> Path:
    root = (code_root or Path(__file__).resolve().parent).resolve()
    candidates = (root.parent / "paper/ARR_clean", root.parent / "ARR_clean")
    for path in candidates:
        if (path / "FILE_MANIFEST.json").is_file():
            return path
    return candidates[0]
