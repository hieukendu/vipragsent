from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class NodeStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PASS = "PASS"
    BLOCKED = "BLOCKED"
    FAIL = "FAIL"


class RunExitCode:
    SUCCESS = 0
    BLOCKED = 2
    PROTOCOL_FAILURE = 3
    EXECUTION_FAILURE = 4
    ARTIFACT_FAILURE = 5


class RuntimeBlocked(RuntimeError):
    """A required external runtime, access permission, or dependency is unavailable."""


class ProtocolConflict(RuntimeError):
    """The locked scientific protocol is internally unresolved."""


class ArtifactContractError(RuntimeError):
    """A handler produced missing, invalid, or unverifiable artifacts."""


@dataclass
class HandlerResult:
    status: NodeStatus
    artifacts: list[dict[str, Any] | str] = field(default_factory=list)
    hashes: dict[str, str] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    retryable: bool = False

    def __post_init__(self) -> None:
        self.status = NodeStatus(self.status)
        if self.status is NodeStatus.PASS and self.error:
            raise ValueError("PASS handler results cannot contain an error")
        if self.status in {NodeStatus.BLOCKED, NodeStatus.FAIL} and not self.error:
            raise ValueError(f"{self.status} handler results require an error")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {"status": self.status.value}

    @classmethod
    def passed(
        cls,
        *,
        artifacts: list[dict[str, Any] | str] | None = None,
        hashes: dict[str, str] | None = None,
        summary: dict[str, Any] | None = None,
    ) -> "HandlerResult":
        return cls(NodeStatus.PASS, artifacts or [], hashes or {}, summary or {})

    @classmethod
    def blocked(cls, error: str, *, summary: dict[str, Any] | None = None) -> "HandlerResult":
        return cls(NodeStatus.BLOCKED, error=error, summary=summary or {})

    @classmethod
    def failed(cls, error: str, *, summary: dict[str, Any] | None = None) -> "HandlerResult":
        return cls(NodeStatus.FAIL, error=error, summary=summary or {})
