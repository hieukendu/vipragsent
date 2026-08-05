from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class PhaseHandoff:
    phase: str
    status: str
    inputs_read: list[str] = field(default_factory=list)
    files_created: list[str] = field(default_factory=list)
    tests_run: list[str] = field(default_factory=list)
    tests_passed: bool = False
    blockers: list[str] = field(default_factory=list)
    next_phase_ready: bool = False

    def __post_init__(self) -> None:
        if self.status not in {"PASS", "BLOCKED", "FAIL"}:
            raise ValueError(f"Invalid phase status: {self.status}")

    def write(self, report_root: str | Path = "reports/phases") -> tuple[Path, Path]:
        root = Path(report_root)
        root.mkdir(parents=True, exist_ok=True)
        stem = root / f"phase_{self.phase}"
        payload = asdict(self)
        payload["generated_at_utc"] = datetime.now(UTC).isoformat()
        handoff_path = stem.with_name(stem.name + "_handoff.json")
        status_path = stem.with_name(stem.name + "_status.md")
        handoff_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        lines = [
            f"# Phase {self.phase} status",
            "",
            f"- Status: `{self.status}`",
            f"- Tests passed: `{self.tests_passed}`",
            f"- Next phase ready: `{self.next_phase_ready}`",
            "",
            "## Inputs read",
            *[f"- `{item}`" for item in self.inputs_read],
            "",
            "## Files created",
            *[f"- `{item}`" for item in self.files_created],
            "",
            "## Tests run",
            *[f"- `{item}`" for item in self.tests_run],
            "",
            "## Blockers",
            *([f"- {item}" for item in self.blockers] or ["- None"]),
            "",
        ]
        status_path.write_text("\n".join(lines), encoding="utf-8")
        return status_path, handoff_path


def write_phase_handoff(
    phase: str,
    status: str,
    inputs_read: Iterable[str] = (),
    files_created: Iterable[str] = (),
    tests_run: Iterable[str] = (),
    tests_passed: bool = False,
    blockers: Iterable[str] = (),
    next_phase_ready: bool = False,
    report_root: str | Path = "reports/phases",
) -> PhaseHandoff:
    handoff = PhaseHandoff(
        phase=f"{int(phase):02d}",
        status=status,
        inputs_read=list(inputs_read),
        files_created=list(files_created),
        tests_run=list(tests_run),
        tests_passed=tests_passed,
        blockers=list(blockers),
        next_phase_ready=next_phase_ready,
    )
    handoff.write(report_root)
    return handoff
