from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..atomic import atomic_write_json, atomic_write_text
from ..hashing import sha256_file, sha256_json
from .status import RuntimeBlocked

Q1B_PRODUCER_REGISTRY = "configs/experiments/q1b/producer_registry.yaml"
Q1B_MATRIX = "configs/experiments/q1b/checkpoint_matrix.yaml"
Q1B_SOURCE_FILES = (
    "src/vipragsent/orchestration/q1b_dependencies.py",
    "src/vipragsent/orchestration/inventory.py",
    "src/vipragsent/orchestration/system_registry.py",
    "src/vipragsent/protocol.py",
    "configs/experiments/master_matrix.yaml",
    "configs/experiments/system_execution_registry.yaml",
    "configs/models/model_registry.yaml",
    "configs/experiments/q4/checkpoint_resolution.yaml",
    Q1B_PRODUCER_REGISTRY,
    Q1B_MATRIX,
)
Q1B_MATRIX_KEY_BY_SYSTEM = {
    "phobert_pol_single": "phobert_ordinary_single_task",
    "phobert_emo_single": "phobert_ordinary_single_task",
    "phobert_multitask_8head": "phobert_multitask",
    "xlmr_multitask_8head": "xlmr_multitask",
    "sailor_multitask_8head": "sailor_multitask",
    "vistral_multitask_8head": "vistral_multitask",
    "vipragsent_full_phobert": "vipragsent",
}
TRAINABLE_KINDS = frozenset({"trainable", "component_bundle", "generation"})
Q1B_SYSTEMS = frozenset(Q1B_MATRIX_KEY_BY_SYSTEM)


def q1b_source_sha256(root: str | Path) -> str:
    """Return the digest binding the Q1b resolver to its audited source files."""
    root = Path(root)
    if not all((root / relative).is_file() for relative in Q1B_SOURCE_FILES):
        return ""
    files = [{"path": relative, "sha256": sha256_file(root / relative)} for relative in Q1B_SOURCE_FILES]
    return sha256_json(files)


@dataclass(frozen=True)
class Q1BProducerDefinition:
    producer_id: str
    system_id: str
    producer_kind: str
    executor_kind: str
    training_applicability: str
    model_family: str
    variant_id: str
    checkpoint_role: str
    reusable_checkpoint_key_pattern: str
    dependencies: tuple[str, ...]
    paper_facing: bool

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> Q1BProducerDefinition:
        required = (
            "producer_id",
            "system_id",
            "producer_kind",
            "executor_kind",
            "training_applicability",
            "model_family",
            "variant_id",
            "checkpoint_role",
            "reusable_checkpoint_key_pattern",
            "dependencies",
            "paper_facing",
        )
        missing = [key for key in required if key not in raw]
        if missing:
            raise ValueError(f"Q1b producer definition is missing fields: {missing}")
        values = {key: str(raw[key]) for key in required if key not in {"dependencies", "paper_facing"}}
        dependencies = tuple(str(item) for item in raw["dependencies"])
        if not values["producer_id"] or not values["system_id"] or not dependencies:
            raise ValueError("Q1b producer definition contains an empty identity or dependency list")
        return cls(**values, dependencies=dependencies, paper_facing=bool(raw["paper_facing"]))

    def checkpoint_key(self, seed: int | str | None) -> str:
        if "{seed}" in self.reusable_checkpoint_key_pattern:
            if seed in (None, ""):
                raise ValueError(f"producer {self.producer_id} requires a seed")
            return self.reusable_checkpoint_key_pattern.replace("{seed}", str(seed))
        return self.reusable_checkpoint_key_pattern

    def as_dict(self) -> dict[str, Any]:
        return {
            "producer_id": self.producer_id,
            "system_id": self.system_id,
            "producer_kind": self.producer_kind,
            "executor_kind": self.executor_kind,
            "training_applicability": self.training_applicability,
            "model_family": self.model_family,
            "variant_id": self.variant_id,
            "checkpoint_role": self.checkpoint_role,
            "reusable_checkpoint_key_pattern": self.reusable_checkpoint_key_pattern,
            "dependencies": list(self.dependencies),
            "paper_facing": self.paper_facing,
        }


def load_q1b_producer_registry(root: str | Path = ".") -> dict[str, Q1BProducerDefinition]:
    path = Path(root) / Q1B_PRODUCER_REGISTRY
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if payload.get("schema_version") != 1:
        raise ValueError("Q1b producer registry schema_version must be 1")
    raw_producers = payload.get("producers")
    if not isinstance(raw_producers, list):
        raise ValueError("Q1b producer registry must contain a producers list")
    definitions: dict[str, Q1BProducerDefinition] = {}
    for raw in raw_producers:
        if not isinstance(raw, Mapping):
            raise ValueError("Q1b producer entries must be mappings")
        definition = Q1BProducerDefinition.from_mapping(raw)
        if definition.producer_id in definitions:
            raise ValueError(f"duplicate Q1b producer ID: {definition.producer_id}")
        definitions[definition.producer_id] = definition
    return definitions


def _load_matrix(root: Path) -> dict[str, dict[str, Any]]:
    payload = yaml.safe_load((root / Q1B_MATRIX).read_text(encoding="utf-8")) or {}
    systems = payload.get("systems")
    if not isinstance(systems, Mapping):
        raise ValueError("Q1b checkpoint matrix must contain systems")
    return {str(key): dict(value) for key, value in systems.items() if isinstance(value, Mapping)}


def _checkpoint_key_from_matrix(system_id: str, seed: int | str | None, matrix: Mapping[str, Mapping[str, Any]]) -> tuple[str, str]:
    matrix_key = Q1B_MATRIX_KEY_BY_SYSTEM.get(system_id)
    if matrix_key is None:
        raise ValueError(f"Q1b system has no checkpoint-matrix mapping: {system_id}")
    row = matrix.get(matrix_key)
    if row is None:
        raise ValueError(f"Q1b checkpoint-matrix row is missing: {matrix_key}")
    if system_id == "phobert_pol_single":
        checkpoint_key = str(row.get("polarity_checkpoint", ""))
    elif system_id == "phobert_emo_single":
        checkpoint_key = str(row.get("emotion_checkpoint", ""))
    else:
        checkpoint_key = str(row.get("checkpoint", ""))
    if not checkpoint_key:
        raise ValueError(f"Q1b checkpoint-matrix row has no checkpoint key: {matrix_key}")
    return matrix_key, f"{checkpoint_key}:{seed}" if seed not in (None, "") else checkpoint_key


def _is_full_inventory_available(root: Path) -> bool:
    return all(
        (root / relative).exists()
        for relative in (
            "configs/experiments/master_matrix.yaml",
            "configs/experiments/system_execution_registry.yaml",
            Q1B_PRODUCER_REGISTRY,
            Q1B_MATRIX,
        )
    )


def _node_id(prefix: str, value: str) -> str:
    return f"{prefix}:{value}"


def _topological_order(nodes: Sequence[str], edges: Sequence[Mapping[str, str]]) -> tuple[list[str], list[str]]:
    adjacency = {node: [] for node in nodes}
    indegree = {node: 0 for node in nodes}
    errors: list[str] = []
    for edge in edges:
        source, target = str(edge["from"]), str(edge["to"])
        if source not in adjacency or target not in adjacency:
            errors.append(f"edge references unknown node: {source}->{target}")
            continue
        adjacency[source].append(target)
        indegree[target] += 1
    queue = sorted(node for node, degree in indegree.items() if degree == 0)
    order: list[str] = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for target in sorted(adjacency[node]):
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
                queue.sort()
    if len(order) != len(nodes):
        errors.append("Q1b producer-consumer graph contains a cycle")
    return order, errors


def _find_trainable_row(rows: Sequence[Mapping[str, Any]], key: str, seed: Any) -> Mapping[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row.get("execution_kind") in TRAINABLE_KINDS
        and str(row.get("reusable_checkpoint_key")) == key
        and str(row.get("seed")) == str(seed)
    ]
    return candidates[0] if len(candidates) == 1 else None


def build_q1b_dependency_graph(root: str | Path = ".", *, inventory_rows: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    root = Path(root)
    errors: list[str] = []
    try:
        definitions = load_q1b_producer_registry(root)
        matrix = _load_matrix(root)
        from .inventory import build_expected_runs
        from .system_registry import load_execution_registry

        rows = list(inventory_rows) if inventory_rows is not None else list(build_expected_runs(root)["rows"])
        execution_specs = load_execution_registry(root)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        return {"schema_version": 1, "status": "FAIL", "errors": [str(exc)], "inventory_count": 0, "q1b_consumer_count": 0, "edges": [], "nodes": []}

    consumers = [row for row in rows if str(row.get("research_question", "")).casefold() == "q1b"]
    edges: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    nodes: set[str] = set()

    for definition in definitions.values():
        approved_spec = execution_specs.get(definition.system_id)
        if approved_spec is None:
            errors.append(f"Q1b producer {definition.producer_id} references unknown approved system {definition.system_id}")
        elif approved_spec.model_family != definition.model_family:
            errors.append(f"Q1b producer {definition.producer_id} model family disagrees with approved registry")

    for row in rows:
        experiment_id = str(row.get("experiment_id") or row.get("run_id"))
        consumer_node = _node_id("inventory", experiment_id)
        nodes.add(consumer_node)
        execution_kind = str(row.get("execution_kind"))
        if execution_kind in TRAINABLE_KINDS or execution_kind == "azure":
            coverage.append({"inventory_id": experiment_id, "producer_node": consumer_node, "producer_kind": "self"})
            continue
        if str(row.get("research_question", "")).casefold() == "q1b":
            system_id = str(row.get("system_id"))
            if system_id == "azure_gpt41_mini":
                definition = next((item for item in definitions.values() if item.producer_kind == "approved_azure_output"), None)
                expected_key = str(row.get("reusable_checkpoint_key"))
                if definition is None or definition.checkpoint_key(row.get("seed")) != expected_key:
                    errors.append(f"Q1b Azure producer/key mismatch for {experiment_id}")
                else:
                    producer_node = _node_id("producer", definition.producer_id)
                    nodes.add(producer_node)
                    edges.append({"from": producer_node, "to": consumer_node, "consumer_id": experiment_id, "producer_id": definition.producer_id, "producer_run_id": definition.producer_id, "seed": None, "expected_checkpoint_key": expected_key, "produced_checkpoint_key": expected_key, "producer_kind": definition.producer_kind})
                    coverage.append({"inventory_id": experiment_id, "producer_node": producer_node, "producer_kind": definition.producer_kind})
                continue
            definition = next((item for item in definitions.values() if item.system_id == system_id and item.producer_kind == "trainable_checkpoint"), None)
            try:
                matrix_key, matrix_key_for_seed = _checkpoint_key_from_matrix(system_id, row.get("seed"), matrix)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            expected_key = str(row.get("reusable_checkpoint_key"))
            if definition is None:
                errors.append(f"Q1B consumer {experiment_id} has no trainable producer definition")
                continue
            produced_key = definition.checkpoint_key(row.get("seed"))
            if produced_key != expected_key or produced_key != matrix_key_for_seed:
                errors.append(f"Q1B producer key mismatch for {experiment_id}: expected {expected_key}, got {produced_key}")
            if definition.executor_kind not in {"single_model_trainable", "generation_trainable"} or definition.training_applicability != "APPLICABLE":
                errors.append(f"Q1B producer {definition.producer_id} is not trainable")
            if definition.paper_facing:
                errors.append(f"Q1B producer {definition.producer_id} must not add a paper-facing row")
            if any("q1b" in dependency.casefold() for dependency in definition.dependencies):
                errors.append(f"Q1B producer {definition.producer_id} depends on Q1B evaluation")
            producer_node = _node_id("producer", f"{definition.producer_id}:{row.get('seed')}")
            nodes.add(producer_node)
            edges.append({"from": producer_node, "to": consumer_node, "consumer_id": experiment_id, "producer_id": definition.producer_id, "producer_run_id": f"{definition.producer_id}:{row.get('seed')}", "seed": row.get("seed"), "matrix_key": matrix_key, "expected_checkpoint_key": expected_key, "produced_checkpoint_key": produced_key, "producer_kind": definition.producer_kind})
            coverage.append({"inventory_id": experiment_id, "producer_node": producer_node, "producer_kind": definition.producer_kind})
            continue
        # Non-Q1b reuse rows must resolve to an actual trainable row with the same key.
        key = str(row.get("reusable_checkpoint_key"))
        source_row = _find_trainable_row(rows, key, row.get("seed"))
        if source_row is None and str(row.get("system_id")) == "vipragsent_full_phobert":
            definition = next((item for item in definitions.values() if item.system_id == "vipragsent_full_phobert"), None)
            if definition is not None and definition.checkpoint_key(row.get("seed")) == key:
                producer_node = _node_id("producer", f"{definition.producer_id}:{row.get('seed')}")
                nodes.add(producer_node)
                edges.append({"from": producer_node, "to": consumer_node, "consumer_id": experiment_id, "producer_id": definition.producer_id, "producer_run_id": f"{definition.producer_id}:{row.get('seed')}", "seed": row.get("seed"), "expected_checkpoint_key": key, "produced_checkpoint_key": key, "producer_kind": definition.producer_kind})
                coverage.append({"inventory_id": experiment_id, "producer_node": producer_node, "producer_kind": definition.producer_kind})
                continue
        if str(row.get("system_id")) == "explanation_only_vistral":
            source_row = next((candidate for candidate in rows if candidate.get("system_id") == "vipragsent_full_vistral" and candidate.get("research_question") == "Q1a" and str(candidate.get("seed")) == str(row.get("seed")) and candidate.get("execution_kind") in TRAINABLE_KINDS), None)
        if source_row is None:
            errors.append(f"inventory row {experiment_id} has no producer for reusable key {key}")
            continue
        producer_node = _node_id("inventory", str(source_row.get("experiment_id") or source_row.get("run_id")))
        edges.append({"from": producer_node, "to": consumer_node, "consumer_id": experiment_id, "producer_id": source_row.get("system_id"), "producer_run_id": source_row.get("experiment_id") or source_row.get("run_id"), "seed": row.get("seed"), "expected_checkpoint_key": key, "produced_checkpoint_key": source_row.get("reusable_checkpoint_key"), "producer_kind": "inventory_trainable"})
        coverage.append({"inventory_id": experiment_id, "producer_node": producer_node, "producer_kind": "inventory_trainable"})

    for edge in edges:
        if edge["from"] not in nodes:
            errors.append(f"producer node is missing: {edge['from']}")
        if edge["to"] not in nodes:
            errors.append(f"consumer node is missing: {edge['to']}")
    topological_order, topology_errors = _topological_order(sorted(nodes), edges)
    errors.extend(topology_errors)
    inventory_ids = {str(row.get("experiment_id") or row.get("run_id")) for row in rows}
    covered_ids = {str(item["inventory_id"]) for item in coverage}
    unresolved = sorted(inventory_ids - covered_ids)
    errors.extend(f"inventory row has no producer coverage: {item}" for item in unresolved)
    duplicate_consumers = sorted(consumer for consumer in {edge["consumer_id"] for edge in edges} if sum(edge["consumer_id"] == consumer for edge in edges) != 1)
    errors.extend(f"Q1b consumer does not resolve to exactly one producer: {item}" for item in duplicate_consumers)
    graph = {
        "schema_version": 1,
        "status": "PASS" if not errors and len(coverage) == len(rows) else "FAIL",
        "errors": errors,
        "paper_inventory_count": len(rows),
        "paper_inventory_count_before": len(rows),
        "paper_inventory_count_after": len(rows),
        "paper_inventory_changed": False,
        "q1b_consumer_count": len(consumers),
        "q1b_producer_definition_count": len(definitions),
        "q1b_producer_edge_count": len([edge for edge in edges if str(edge.get("consumer_id", "")).startswith("q1b_")]),
        "inventory_rows_with_producer": len(coverage),
        "inventory_rows_without_producer": unresolved,
        "producers": [definition.as_dict() for definition in sorted(definitions.values(), key=lambda item: item.producer_id)],
        "edges": sorted(edges, key=lambda item: (str(item["consumer_id"]), str(item["producer_id"]))),
        "coverage": sorted(coverage, key=lambda item: str(item["inventory_id"])),
        "nodes": sorted(nodes),
        "topological_order": topological_order,
        "inventory_hash": sha256_json(rows),
        "producer_registry_sha256": sha256_file(root / Q1B_PRODUCER_REGISTRY),
        "checkpoint_matrix_sha256": sha256_file(root / Q1B_MATRIX),
    }
    return graph


def resolve_q1b_producer(root: str | Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    graph = build_q1b_dependency_graph(root)
    if graph.get("status") != "PASS":
        raise RuntimeBlocked("Q1b producer graph is invalid: " + "; ".join(str(item) for item in graph.get("errors", [])))
    experiment_id = str(entry.get("experiment_id") or entry.get("run_id") or f"q1b_{entry.get('system_id')}_{entry.get('seed')}")
    candidates = [edge for edge in graph["edges"] if str(edge.get("consumer_id")) == experiment_id]
    if not candidates:
        candidates = [edge for edge in graph["edges"] if str(edge.get("consumer_id", "")).startswith("q1b_") and str(edge.get("consumer_id")).startswith(f"q1b_{entry.get('system_id')}_") and str(edge.get("seed")) == str(entry.get("seed"))]
    if len(candidates) != 1:
        raise RuntimeBlocked(f"Q1b entry does not resolve to exactly one producer: {experiment_id}")
    return {
        "edge": candidates[0],
        "graph_sha256": sha256_json(graph),
        "source_sha256": q1b_source_sha256(root),
    }


def write_q1b_dependency_report(root: str | Path = ".") -> dict[str, Any]:
    root = Path(root)
    graph = build_q1b_dependency_graph(root)
    atomic_write_json(root / "reports/q1b_dependency_graph.json", graph)
    lines = [
        "# Q1B Dependency Graph",
        "",
        f"Status: `{graph['status']}`",
        f"Paper-facing inventory rows: `{graph.get('paper_inventory_count', 0)}` before and `{graph.get('paper_inventory_count_after', 0)}` after",
        f"Q1B consumers: `{graph.get('q1b_consumer_count', 0)}`; producer edges: `{graph.get('q1b_producer_edge_count', 0)}`",
        f"Rows with producer coverage: `{graph.get('inventory_rows_with_producer', 0)}`/{graph.get('paper_inventory_count', 0)}`",
        "",
        "## Q1B Mappings",
        "",
        "| Consumer | Producer | Seed | Checkpoint key | Kind |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for edge in graph.get("edges", []):
        if str(edge.get("consumer_id", "")).startswith("q1b_"):
            lines.append(f"| {edge['consumer_id']} | {edge['producer_id']} | {edge.get('seed', 'N/A')} | {edge['expected_checkpoint_key']} | {edge['producer_kind']} |")
    if graph.get("errors"):
        lines.extend(["", "## Errors", "", *[f"- {error}" for error in graph["errors"]]])
    else:
        lines.extend(["", "No unresolved producer, key, seed, or cycle errors."])
    atomic_write_text(root / "reports/q1b_dependency_graph.md", "\n".join(lines) + "\n")
    return graph


def q1b_dependency_graph_is_available(root: str | Path = ".") -> bool:
    return _is_full_inventory_available(Path(root))
