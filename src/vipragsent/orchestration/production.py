from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..artifacts.schemas import validate_artifact_tree
from ..atomic import atomic_write_json
from ..config_validation import validate_config_tree
from ..data.loaders import load_vipragsent
from ..hashing import sha256_file
from ..profiling import Profiler
from ..protocol import validate_protocol_resolution
from ..statistics.significance import load_p_value_strategy
from .dag import DAGNode
from .status import ArtifactContractError, HandlerResult, ProtocolConflict, RuntimeBlocked

if TYPE_CHECKING:
    from .handlers import HandlerEnvironment


def _require_full(env: HandlerEnvironment) -> None:
    if env.context.mode != "full":
        raise ValueError("Production handlers require an explicit full execution context")
    if env.context.artifact_path.name == "fixture" or "runs\\fixture" in str(env.context.artifact_path):
        raise ValueError("Production handlers cannot write under a fixture artifact root")


def _stage_path(env: HandlerEnvironment, node: DAGNode) -> Path:
    return env.run_root / "nodes" / node.node_id / "handler_result.json"


def _artifact(path: Path, root: Path) -> dict[str, str]:
    if not path.exists() or not path.is_file():
        raise ArtifactContractError(f"Expected handler output is missing: {path}")
    return {"path": path.resolve().relative_to(root.resolve()).as_posix(), "sha256": sha256_file(path)}


def _complete(env: HandlerEnvironment, node: DAGNode, result: HandlerResult) -> HandlerResult:
    if result.status.value != "PASS":
        return result
    path = _stage_path(env, node)
    atomic_write_json(path, {
        "node_id": node.node_id,
        "kind": node.kind,
        "provenance": env.context.provenance(),
        "summary": result.summary,
        "artifacts": result.artifacts,
    })
    artifact = _artifact(path, env.root)
    return HandlerResult.passed(
        artifacts=[*result.artifacts, artifact],
        hashes=result.hashes | {artifact["path"]: artifact["sha256"]},
        summary=result.summary | {"handler_result": artifact["path"], "real_implementation": True},
    )


def _service_or_default(env: HandlerEnvironment, node: DAGNode, default: Callable[[], HandlerResult]) -> HandlerResult:
    injected = env.services.get(node.kind)
    result = injected(env, node) if injected else default()
    if not isinstance(result, HandlerResult):
        raise ArtifactContractError(f"Injected service for {node.kind} did not return HandlerResult")
    return _complete(env, node, result)


def _validation(env: HandlerEnvironment, node: DAGNode) -> HandlerResult:
    _require_full(env)
    config = validate_config_tree(env.root)
    if not config["passed"]:
        return HandlerResult.failed("Configuration validation failed: " + "; ".join(config["errors"]))
    data_dir = env.root / "data/processed/vipragsent"
    if not data_dir.exists():
        raise RuntimeBlocked("Processed ViPragSent data is unavailable")
    bundle = load_vipragsent(data_dir)
    protocol = validate_protocol_resolution(env.root)
    if protocol["scientific_protocol_conflicts"]:
        raise ProtocolConflict(", ".join(protocol["scientific_protocol_conflicts"]))
    return HandlerResult.passed(summary={"train": len(bundle.train), "dev": len(bundle.dev), "test": len(bundle.test), "data_fingerprint": bundle.fingerprint})


def handle_validation(env: HandlerEnvironment, node: DAGNode) -> HandlerResult:
    return _service_or_default(env, node, lambda: _validation(env, node))


def _preprocessing(env: HandlerEnvironment, node: DAGNode) -> HandlerResult:
    _require_full(env)
    cache_report = env.root / "data/manifests/tokenization_cache_manifest.json"
    if not cache_report.exists():
        raise RuntimeBlocked("Frozen production tokenization caches are unavailable")
    payload = json.loads(cache_report.read_text(encoding="utf-8"))
    if payload.get("execution_mode") == "fixture":
        raise ArtifactContractError("Fixture tokenization cache cannot enter a full run")
    return HandlerResult.passed(summary={"cache_manifest": str(cache_report.relative_to(env.root).as_posix()), "cache_manifest_sha256": sha256_file(cache_report)})


def handle_preprocessing(env: HandlerEnvironment, node: DAGNode) -> HandlerResult:
    return _service_or_default(env, node, lambda: _preprocessing(env, node))


def _azure_settings() -> Any:
    from ..azure.client import AzureSettings

    try:
        return AzureSettings.from_env()
    except ValueError as exc:
        raise RuntimeBlocked(str(exc)) from exc


def _azure_rationale(env: HandlerEnvironment, node: DAGNode) -> HandlerResult:
    _require_full(env)
    settings = _azure_settings()
    from ..azure.client import AzureCache, AzureResponsesClient
    from ..azure.runners import RationaleRunner

    input_path = env.root / "data/processed/rationales/azure_rationale_input_train.jsonl"
    if not input_path.exists():
        raise RuntimeBlocked("Active rationale input manifest is unavailable")
    inputs = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    output = env.run_root / "azure" / "rationale.jsonl"
    failure = env.run_root / "azure" / "rationale_failures.json"
    runner = RationaleRunner(AzureResponsesClient(settings, cache=AzureCache(env.run_root / "azure/cache")), output_path=output, failure_path=failure)
    summary = runner.run(inputs, lambda item: f"Generate a rationale for this Vietnamese comment:\n{item['comment']}")
    return HandlerResult.passed(artifacts=[_artifact(output, env.root), _artifact(failure, env.root)], summary=summary | {"deployment": settings.deployment, "category": "rationale"})


def handle_azure_rationale(env: HandlerEnvironment, node: DAGNode) -> HandlerResult:
    return _service_or_default(env, node, lambda: _azure_rationale(env, node))


def _azure_baseline(env: HandlerEnvironment, node: DAGNode) -> HandlerResult:
    _require_full(env)
    settings = _azure_settings()
    from ..azure.client import AzureCache, AzureResponsesClient
    from ..azure.prompts import PromptRegistry
    from ..azure.runners import PromptedBaselineRunner

    manifest_path = env.root / "data/manifests/prompts/pragmatic_v1.json"
    input_path = env.root / "data/processed/vipragsent/test.csv"
    if not manifest_path.exists() or not input_path.exists():
        raise RuntimeBlocked("Frozen pragmatic prompt manifest or ViPragSent test split is unavailable")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registry = PromptRegistry(manifest)
    from ..data.loaders import read_csv

    rows = read_csv(input_path)
    output = env.run_root / "azure" / "baseline.jsonl"
    failure = env.run_root / "azure" / "baseline_failures.json"
    runner = PromptedBaselineRunner(AzureResponsesClient(settings, cache=AzureCache(env.run_root / "azure/cache")), output_path=output, failure_path=failure)
    summary = runner.run(rows, task="pragmatic", prompt_builder=lambda row: registry.pragmatic(row["text"]), manifest=manifest)
    return HandlerResult.passed(artifacts=[_artifact(output, env.root), _artifact(failure, env.root)], summary=summary | {"deployment": settings.deployment, "category": "baseline"})


def handle_azure_baseline(env: HandlerEnvironment, node: DAGNode) -> HandlerResult:
    return _service_or_default(env, node, lambda: _azure_baseline(env, node))


def _gpu_training(env: HandlerEnvironment, node: DAGNode) -> HandlerResult:
    _require_full(env)
    if not os.getenv("CUDA_VISIBLE_DEVICES") and not __import__("torch").cuda.is_available():
        raise RuntimeBlocked("GPU training requires CUDA; no model weights or training state were initialized")
    cache_manifest = env.root / "data/model_cache_manifest.json"
    if not cache_manifest.exists():
        raise RuntimeBlocked("Phase 15 verified model cache manifest is unavailable")
    model_spec = json.loads(cache_manifest.read_text(encoding="utf-8"))
    if model_spec.get("weights_downloaded") is not True:
        raise RuntimeBlocked("Phase 15 model weights have not passed offline verification")
    if "gpu_training" not in env.services:
        raise RuntimeBlocked("GPU training requires an injected production batch/run service")


def handle_gpu_training(env: HandlerEnvironment, node: DAGNode) -> HandlerResult:
    return _service_or_default(env, node, lambda: _gpu_training(env, node))


def _evaluation(env: HandlerEnvironment, node: DAGNode) -> HandlerResult:
    _require_full(env)
    evaluation_root = env.run_root / "evaluations"
    manifests = sorted(evaluation_root.glob("*/evaluation.json")) if evaluation_root.exists() else []
    if not manifests:
        raise RuntimeBlocked("Production prediction manifests are unavailable for evaluation")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in manifests]
    output = evaluation_root / f"{node.node_id}.json"
    atomic_write_json(output, {"node_id": node.node_id, "mode": "full", "records": records})
    return HandlerResult.passed(artifacts=[_artifact(output, env.root)], summary={"record_count": len(records)})


def handle_evaluation(env: HandlerEnvironment, node: DAGNode) -> HandlerResult:
    return _service_or_default(env, node, lambda: _evaluation(env, node))


def _statistics(env: HandlerEnvironment, node: DAGNode) -> HandlerResult:
    _require_full(env)
    strategy = load_p_value_strategy(env.root / "configs/statistics/significance_method.yaml")
    evaluations = sorted((env.run_root / "evaluations").glob("*.json"))
    if not evaluations:
        raise RuntimeBlocked("Evaluation outputs are unavailable for statistics")
    output = env.run_root / "statistics" / "statistics.json"
    atomic_write_json(output, {"mode": "full", "strategy": strategy, "evaluation_files": [path.relative_to(env.root).as_posix() for path in evaluations]})
    return HandlerResult.passed(artifacts=[_artifact(output, env.root)], summary={"strategy": strategy["method_id"]})


def handle_statistics(env: HandlerEnvironment, node: DAGNode) -> HandlerResult:
    return _service_or_default(env, node, lambda: _statistics(env, node))


def _profiling(env: HandlerEnvironment, node: DAGNode) -> HandlerResult:
    _require_full(env)
    profile_path = env.run_root / "profiling" / "profiles.json"
    if not profile_path.exists():
        raise RuntimeBlocked("Measured profiling records are unavailable")
    profiles = json.loads(profile_path.read_text(encoding="utf-8"))
    if not profiles:
        raise RuntimeBlocked("Profiling record set is empty")
    return HandlerResult.passed(artifacts=[_artifact(profile_path, env.root)], summary={"profile_count": len(profiles), "profiler": Profiler.__name__})


def handle_profiling(env: HandlerEnvironment, node: DAGNode) -> HandlerResult:
    return _service_or_default(env, node, lambda: _profiling(env, node))


def _manual_candidates(env: HandlerEnvironment, node: DAGNode) -> HandlerResult:
    _require_full(env)
    from ..manual import export_manual_candidates

    result = export_manual_candidates(env.run_root, env.context.artifact_path / "manual")
    artifacts = [_artifact(Path(path), env.root) for path in result["files"]]
    return HandlerResult.passed(artifacts=artifacts, summary=result)


def handle_manual_candidates(env: HandlerEnvironment, node: DAGNode) -> HandlerResult:
    return _service_or_default(env, node, lambda: _manual_candidates(env, node))


def _artifact_validation(env: HandlerEnvironment, node: DAGNode) -> HandlerResult:
    _require_full(env)
    errors = validate_artifact_tree(env.context.artifact_path)
    if errors:
        return HandlerResult.failed("Production artifact schema validation failed: " + "; ".join(errors))
    manifest_path = env.root / "FINAL_EXPERIMENT_MANIFEST.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("mode") != "full" or payload.get("core_experiments_ready") is not True:
            return HandlerResult.failed("Production final manifest provenance is invalid")
    return HandlerResult.passed(summary={"artifact_root": str(env.context.artifact_path)})


def handle_artifact_validation(env: HandlerEnvironment, node: DAGNode) -> HandlerResult:
    return _service_or_default(env, node, lambda: _artifact_validation(env, node))


def _artifact_export(env: HandlerEnvironment, node: DAGNode) -> HandlerResult:
    _require_full(env)
    from ..artifacts.exporter import export_production_artifacts

    manifest = export_production_artifacts(repo_root=env.root, run_id=env.context.run_id, output_root=env.context.artifact_path)
    manifest_path = Path(manifest["manifest_path"])
    return HandlerResult.passed(artifacts=[_artifact(manifest_path, env.root)], summary=manifest)


def handle_artifact_export(env: HandlerEnvironment, node: DAGNode) -> HandlerResult:
    return _service_or_default(env, node, lambda: _artifact_export(env, node))


def _final_manifest(env: HandlerEnvironment, node: DAGNode) -> HandlerResult:
    _require_full(env)
    path = env.root / "FINAL_EXPERIMENT_MANIFEST.json"
    if not path.exists():
        raise RuntimeBlocked("A complete Phase 16 production export has not generated FINAL_EXPERIMENT_MANIFEST.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("mode") != "full" or payload.get("core_experiments_ready") is not True:
        raise ArtifactContractError("Final manifest is not a complete full-run manifest")
    return HandlerResult.passed(artifacts=[_artifact(path, env.root)], summary={"final_manifest": path.relative_to(env.root).as_posix()})


def handle_final_manifest(env: HandlerEnvironment, node: DAGNode) -> HandlerResult:
    return _service_or_default(env, node, lambda: _final_manifest(env, node))
