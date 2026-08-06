from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from ..hashing import sha256_json


@dataclass(frozen=True)
class ExecutionContext:
    mode: str
    run_id: str
    data_fingerprint: str
    config_hash: str
    code_commit: str
    model_revision: str
    tokenizer_revision: str
    artifact_root: str

    def __post_init__(self) -> None:
        if self.mode not in {"fixture", "production", "full"}:
            raise ValueError("Execution mode must be fixture, production, or full")
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if not self.data_fingerprint or not self.config_hash or not self.code_commit:
            raise ValueError("Execution context requires data, config, and code provenance")
        if self.mode != "fixture" and (self.model_revision == "fixture" or self.tokenizer_revision == "fixture"):
            raise ValueError("Fixture model/tokenizer revisions are prohibited outside fixture mode")

    @property
    def artifact_path(self) -> Path:
        return Path(self.artifact_root)

    def validate_artifact_path(self, path: str | Path) -> None:
        candidate = Path(path).resolve()
        root = self.artifact_path.resolve()
        if root not in candidate.parents and candidate != root:
            raise ValueError(f"Artifact path {candidate} is outside context root {root}")

    def provenance(self) -> dict[str, str]:
        return asdict(self) | {"context_hash": sha256_json(asdict(self))}
