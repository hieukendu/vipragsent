from __future__ import annotations

import argparse
from pathlib import Path

from vipragsent.hashing import sha256_file


def iter_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and ".git" not in path.parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Write deterministic SHA-256 checksums for a directory")
    parser.add_argument("root", nargs="?", default="experiment_artifacts")
    parser.add_argument("--output", default="experiment_artifacts/checksums.sha256")
    args = parser.parse_args()
    root = Path(args.root)
    output = Path(args.output)
    files = [path for path in iter_files(root) if path.resolve() != output.resolve()]
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in files]
    output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"hashed={len(files)} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
