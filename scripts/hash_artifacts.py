from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ROOT
from vipragsent.atomic import atomic_write_text
from vipragsent.hashing import sha256_file


def _excluded(path: Path, root: Path) -> bool:
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    if relative in {"FINAL_CHECKSUMS.sha256", "SETUP_CHECKSUMS.sha256"}:
        return True
    parts = set(Path(relative).parts)
    return bool(parts & {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", "runs", "results", "experiment_artifacts", "checkpoints", "predictions"}) or relative.startswith(("data/raw/", "data/external/", "data/model_cache/", "data/input/"))


def iter_files(root: Path) -> list[Path]:
    root = root.resolve()
    return sorted((path for path in root.rglob("*") if path.is_file() and not _excluded(path, root)), key=lambda path: path.relative_to(root).as_posix())


def write_checksums(root: Path, output: Path) -> int:
    root = root.resolve()
    files = [path for path in iter_files(root) if path.resolve() != output.resolve()]
    lines = [f"{sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in files]
    atomic_write_text(output, "\n".join(lines) + ("\n" if lines else ""))
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser(description="Write stable SHA-256 checksums for non-runtime artifacts")
    parser.add_argument("root", nargs="?", default="experiment_artifacts")
    parser.add_argument("--output", default="experiment_artifacts/checksums.sha256")
    args = parser.parse_args()
    root = (ROOT / args.root).resolve() if not Path(args.root).is_absolute() else Path(args.root).resolve()
    output = (ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output).resolve()
    count = write_checksums(root, output)
    print(f"hashed={count} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
