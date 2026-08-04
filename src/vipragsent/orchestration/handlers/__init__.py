from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Callable, Mapping

from ...artifacts.exporter import export_fixture_artifacts
from ...artifacts.schemas import validate_artifact_tree
from ...hashing import sha256_file, sha256_json
from ..context import ExecutionContext
from ..dag import DAGNode
from ..status import ArtifactContractError, HandlerResult
from ..production import (
    handle_artifact_validation,
    handle_artifact_export,
    handle_azure_baseline,
    handle_azure_rationale,
    handle_evaluation,
    handle_final_manifest,
    handle_gpu_training,
    handle_manual_candidates,
    handle_preprocessing,
    handle_profiling,
    handle_statistics,
    handle_validation,
)


@dataclass(frozen=True)
class HandlerEnvironment:
    root: Path
    context: ExecutionContext
    services: Mapping[str, Callable[["HandlerEnvironment", DAGNode], HandlerResult]] = field(default_factory=dict)

    @property
    def run_root(self) -> Path:
        path = self.root / "runs" / self.context.run_id
        path.mkdir(parents=True, exist_ok=True)
        return path


def _artifact(path: Path, root: Path) -> dict[str, str]:
    if not path.exists():
        raise ArtifactContractError(f"Handler output is missing: {path}")
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}


def _context_marker(env: HandlerEnvironment, node: DAGNode, *, synthetic: bool) -> dict[str, str | bool]:
    return {
        "node_id": node.node_id,
        "mode": env.context.mode,
        "synthetic_results": synthetic,
        "context_hash": env.context.provenance()["context_hash"],
    }


def _fixture_handler(env: HandlerEnvironment, node: DAGNode) -> HandlerResult:
    return HandlerResult.passed(summary=_context_marker(env, node, synthetic=True))


def _fixture_export(env: HandlerEnvironment, node: DAGNode) -> HandlerResult:
    manifest = export_fixture_artifacts(repo_root=env.root, run_id=env.context.run_id, output_root=env.context.artifact_path)
    return HandlerResult.passed(
        artifacts=[_artifact(Path(manifest["manifest_path"]), env.root)],
        summary=_context_marker(env, node, synthetic=True) | {"fixture_validation_manifest": manifest["manifest_path"]},
    )


def _fixture_validation(env: HandlerEnvironment, node: DAGNode) -> HandlerResult:
    artifact_root = env.context.artifact_path / "artifacts"
    if artifact_root.exists():
        errors = validate_artifact_tree(artifact_root)
        if errors:
            return HandlerResult.failed("Fixture artifact schema validation failed: " + "; ".join(errors))
    return HandlerResult.passed(summary=_context_marker(env, node, synthetic=True))


def build_handler_registry(env: HandlerEnvironment) -> dict[str, Callable[[DAGNode], HandlerResult]]:
    """Build an explicit handler for every DAG kind used by the master matrix."""
    if env.context.mode == "fixture":
        fixture = {kind: (lambda node, _kind=kind: _fixture_handler(env, node)) for kind in {
            "validation", "preprocessing", "azure_rationale", "azure_baseline", "azure_baselines",
            "gpu_training", "evaluation", "statistics", "profiling", "manual_candidates",
            "artifact_validation", "artifact_export", "export", "manifest", "final_manifest",
        }}
        fixture["artifact_validation"] = lambda node: _fixture_validation(env, node)
        fixture["artifact_export"] = lambda node: _fixture_export(env, node)
        fixture["export"] = fixture["artifact_export"]
        fixture["manifest"] = fixture["artifact_export"]
        fixture["final_manifest"] = fixture["artifact_export"]
        return fixture

    production = {
        "preprocessing": lambda node: handle_preprocessing(env, node),
        "azure_rationale": lambda node: handle_azure_rationale(env, node),
        "azure_baseline": lambda node: handle_azure_baseline(env, node),
        "azure_baselines": lambda node: handle_azure_baseline(env, node),
        "gpu_training": lambda node: handle_gpu_training(env, node),
        "evaluation": lambda node: handle_evaluation(env, node),
        "statistics": lambda node: handle_statistics(env, node),
        "profiling": lambda node: handle_profiling(env, node),
        "manual_candidates": lambda node: handle_manual_candidates(env, node),
        "artifact_validation": lambda node: handle_artifact_validation(env, node),
    }
    production["validation"] = lambda node: handle_validation(env, node)
    production["artifact_export"] = lambda node: handle_artifact_export(env, node)
    production["export"] = production["artifact_export"]
    production["manifest"] = lambda node: handle_final_manifest(env, node)
    production["final_manifest"] = production["manifest"]
    return production


def build_execution_context(root: str | Path, *, mode: str, run_id: str, artifact_root: str | Path) -> ExecutionContext:
    root = Path(root)
    config_path = root / "configs/master_run.yaml"
    registry_path = root / "configs/models/model_registry.yaml"
    dataset_manifest = root / "data/manifests/dataset_manifest.json"
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    data_fingerprint = sha256_file(dataset_manifest) if dataset_manifest.exists() else "missing-data-manifest"
    config_hash = sha256_json({"master_run": config_path.read_text(encoding="utf-8"), "registry": registry_path.read_text(encoding="utf-8")})
    revision = "fixture" if mode == "fixture" else sha256_file(registry_path) if registry_path.exists() else "missing-model-registry"
    return ExecutionContext(
        mode=mode,
        run_id=run_id,
        data_fingerprint=data_fingerprint,
        config_hash=config_hash,
        code_commit=commit,
        model_revision=revision,
        tokenizer_revision=revision,
        artifact_root=str(Path(artifact_root)),
    )
