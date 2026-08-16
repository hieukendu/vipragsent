"""Pure validation for the disjoint NAACL-balanced profile artifact.

This module reads the profile and the audited Q1b inputs, then builds the
existing dependency graph.  It does not execute experiments, inspect run
artifacts, or write files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from ..hashing import sha256_file, sha256_json
from ..orchestration.inventory import build_expected_runs
from ..orchestration.q1b_dependencies import (
    build_q1b_dependency_graph,
    load_q1b_producer_registry,
)


PROFILE_ID = "LUNA_NAACL_PROFILE"
PROFILE_CONFIG = "configs/experiments/naacl_balanced_runtime_profile.yaml"
PROFILE_REPORT = "reports/runtime_optimization/naacl_balanced_profile.json"
Q1B_SOURCE_FILES = (
    "src/vipragsent/orchestration/q1b_dependencies.py",
    "src/vipragsent/orchestration/inventory.py",
    "src/vipragsent/orchestration/system_registry.py",
    "src/vipragsent/protocol.py",
    "configs/experiments/master_matrix.yaml",
    "configs/experiments/system_execution_registry.yaml",
    "configs/models/model_registry.yaml",
    "configs/experiments/q4/checkpoint_resolution.yaml",
    "configs/experiments/q1b/producer_registry.yaml",
    "configs/experiments/q1b/checkpoint_matrix.yaml",
)
TRAINABLE_Q1B_PRODUCER_KIND = "trainable_checkpoint"
AZURE_Q1B_PRODUCER_KIND = "approved_azure_output"


class ProfileValidationError(ValueError):
    """Raised when the policy artifact no longer matches audited sources."""


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ProfileValidationError(f"expected mapping in {path}")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProfileValidationError(f"expected object in {path}")
    return payload


def _source_digest(root: Path) -> dict[str, Any]:
    files = []
    for relative in Q1B_SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise ProfileValidationError(f"missing Q1b profile source: {relative}")
        files.append({"path": relative, "sha256": sha256_file(path)})
    return {"files": files, "sha256": sha256_json(files)}


def _q1b_rows(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in inventory.get("rows", [])
        if str(row.get("research_question", "")).casefold() == "q1b"
    ]


def _graph_edges(graph: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        edge
        for edge in graph.get("edges", [])
        if str(edge.get("consumer_id", "")).startswith("q1b_")
    ]


def _edge_record(edge: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    return {
        "consumer_id": str(edge["consumer_id"]),
        "consumer_system_id": str(row["system_id"]),
        "producer_id": str(edge["producer_id"]),
        "producer_run_id": str(edge["producer_run_id"]),
        "producer_kind": str(edge["producer_kind"]),
        "checkpoint_key": str(edge["expected_checkpoint_key"]),
        "expected_checkpoint_key": str(edge["expected_checkpoint_key"]),
        "produced_checkpoint_key": str(edge["produced_checkpoint_key"]),
        "seed": edge.get("seed"),
        "graph_edge": True,
    }


def _build_binding(root: Path, graph: dict[str, Any], inventory: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    rows = _q1b_rows(inventory)
    rows_by_id = {str(row.get("experiment_id") or row.get("run_id")): row for row in rows}
    graph_edges = _graph_edges(graph)
    edges_by_consumer = {str(edge.get("consumer_id")): edge for edge in graph_edges}
    definitions = load_q1b_producer_registry(root)

    if graph.get("status") != "PASS":
        errors.append(f"Q1b graph status is {graph.get('status')!r}")
    if graph.get("q1b_consumer_count") != len(rows):
        errors.append(
            f"Q1b consumer count drift: graph={graph.get('q1b_consumer_count')}, inventory={len(rows)}"
        )
    if graph.get("q1b_producer_edge_count") != len(graph_edges):
        errors.append("Q1b graph edge count is internally inconsistent")

    trainable_rows = [row for row in rows if str(row.get("system_id")) != "azure_gpt41_mini"]
    azure_rows = [row for row in rows if str(row.get("system_id")) == "azure_gpt41_mini"]
    if len(azure_rows) != 1:
        errors.append(f"Q1b Azure consumer count drift: {len(azure_rows)}")
    if any(row.get("seed") is None for row in trainable_rows):
        errors.append("seeded Q1b consumer has null seed")
    if any(row.get("seed") is not None for row in azure_rows):
        errors.append("Azure Q1b consumer must have seed null")

    records: list[dict[str, Any]] = []
    expected_trainable_ids = {str(row["experiment_id"]) for row in trainable_rows}
    actual_trainable_ids = set(edges_by_consumer)
    if actual_trainable_ids != expected_trainable_ids:
        errors.append(
            "Q1b graph consumer ID drift: "
            f"missing={sorted(expected_trainable_ids - actual_trainable_ids)}, "
            f"extra={sorted(actual_trainable_ids - expected_trainable_ids)}"
        )

    for consumer_id in sorted(expected_trainable_ids):
        row = rows_by_id[consumer_id]
        edge = edges_by_consumer.get(consumer_id)
        if edge is None:
            continue
        expected_key = str(row.get("reusable_checkpoint_key"))
        expected_seed = row.get("seed")
        expected_definitions = [
            definition
            for definition in definitions.values()
            if definition.system_id == str(row.get("system_id"))
            and definition.producer_kind == TRAINABLE_Q1B_PRODUCER_KIND
        ]
        expected_producer_id = expected_definitions[0].producer_id if len(expected_definitions) == 1 else None
        if len(expected_definitions) != 1:
            errors.append(f"Q1b {consumer_id} producer definition drift")
        checks = {
            "producer_id": edge.get("producer_id") == expected_producer_id,
            "producer_run_id": edge.get("producer_run_id") == f"{expected_producer_id}:{expected_seed}",
            "producer_kind": edge.get("producer_kind") == TRAINABLE_Q1B_PRODUCER_KIND,
            "expected_checkpoint_key": edge.get("expected_checkpoint_key") == expected_key,
            "produced_checkpoint_key": edge.get("produced_checkpoint_key") == expected_key,
            "seed": str(edge.get("seed")) == str(expected_seed),
        }
        for field, passed in checks.items():
            if not passed:
                errors.append(f"Q1b {consumer_id} {field} drift")
        records.append(_edge_record(edge, row))

    # The current graph intentionally records the non-seeded Azure row as
    # self-covered rather than adding a graph edge.  Bind that coverage to the
    # approved-output producer definition explicitly and retain the null seed.
    if azure_rows:
        azure_row = azure_rows[0]
        azure_id = str(azure_row["experiment_id"])
        azure_coverage = [
            item
            for item in graph.get("coverage", [])
            if str(item.get("inventory_id")) == azure_id
        ]
        azure_definitions = [
            definition
            for definition in definitions.values()
            if definition.producer_kind == AZURE_Q1B_PRODUCER_KIND
        ]
        if len(azure_coverage) != 1:
            errors.append("Azure Q1b graph coverage drift")
        if len(azure_definitions) != 1:
            errors.append("Azure approved-output producer definition drift")
        if azure_coverage and azure_coverage[0].get("producer_kind") != "self":
            errors.append("Azure Q1b graph relation drift")
        if azure_definitions:
            azure_definition = azure_definitions[0]
            expected_key = str(azure_row.get("reusable_checkpoint_key"))
            produced_key = azure_definition.checkpoint_key(None)
            if produced_key != expected_key:
                errors.append("Azure Q1b approved-output checkpoint key drift")
            records.append(
                {
                    "consumer_id": azure_id,
                    "consumer_system_id": str(azure_row["system_id"]),
                    "producer_id": azure_definition.producer_id,
                    "producer_run_id": azure_definition.producer_id,
                    "producer_kind": azure_definition.producer_kind,
                    "checkpoint_key": expected_key,
                    "expected_checkpoint_key": expected_key,
                    "produced_checkpoint_key": produced_key,
                    "seed": None,
                    "graph_edge": False,
                    "graph_coverage_producer_kind": "self",
                    "relation": "approved_output_from_graph_coverage",
                }
            )

    if len(records) != len(rows):
        errors.append(f"Q1b profile edge count drift: records={len(records)}, consumers={len(rows)}")
    return {
        "consumer_count": len(rows),
        "graph_edge_count": len(graph_edges),
        "profile_edge_count": len(records),
        "seeded_consumer_count": len(trainable_rows),
        "seedless_consumer_count": len(azure_rows),
        "consumer_edges": sorted(records, key=lambda item: str(item["consumer_id"])),
    }, errors


def build_naacl_profile_snapshot(root: str | Path = ".") -> dict[str, Any]:
    """Build the read-only, current-source validation snapshot."""

    root = Path(root)
    config = _load_yaml(root / PROFILE_CONFIG)
    if config.get("profile_id") != PROFILE_ID:
        raise ProfileValidationError("NAACL profile ID drift")
    inventory = build_expected_runs(root)
    graph = build_q1b_dependency_graph(root, inventory_rows=inventory["rows"])
    binding, errors = _build_binding(root, graph, inventory)
    source = _source_digest(root)
    snapshot = {
        "schema_version": 1,
        "profile_id": PROFILE_ID,
        "graph": {
            "status": graph.get("status"),
            "sha256": sha256_json(graph),
            "inventory_hash": graph.get("inventory_hash"),
            "producer_registry_sha256": graph.get("producer_registry_sha256"),
            "checkpoint_matrix_sha256": graph.get("checkpoint_matrix_sha256"),
        },
        "source": source,
        "q1b": binding,
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
    }
    return snapshot


def validate_naacl_profile(root: str | Path = ".") -> dict[str, Any]:
    """Validate policy, graph binding, and the checked-in report fail-closed."""

    root = Path(root)
    config = _load_yaml(root / PROFILE_CONFIG)
    snapshot = build_naacl_profile_snapshot(root)
    errors = list(snapshot["errors"])
    activation = config.get("activation", {})
    scope = config.get("scope", {})
    q1b_policy = config.get("q1b", {}).get("dependency_binding", {})
    if activation.get("default_enabled") is not False:
        errors.append("profile default_enabled must be false")
    if activation.get("execution_enabled") is not False:
        errors.append("profile execution_enabled must be false")
    if activation.get("real_execution") != "PROHIBITED":
        errors.append("profile activation real_execution must be PROHIBITED")
    if scope.get("real_execution") != "PROHIBITED":
        errors.append("profile scope real_execution must be PROHIBITED")
    if config.get("source", {}).get("source_is_read_only") is not True:
        errors.append("profile source must be immutable")
    if config.get("source", {}).get("no_run_data_consumed") is not True:
        errors.append("profile must not consume run data")
    if q1b_policy.get("expected_profile_edge_count") != snapshot["q1b"]["profile_edge_count"]:
        errors.append("profile Q1b expected edge count drift")
    if q1b_policy.get("expected_consumer_count") != snapshot["q1b"]["consumer_count"]:
        errors.append("profile Q1b expected consumer count drift")
    if q1b_policy.get("expected_graph_edge_count") != snapshot["q1b"]["graph_edge_count"]:
        errors.append("profile Q1b expected graph edge count drift")
    report = _load_json(root / PROFILE_REPORT)
    report_binding = report.get("q1b", {}).get("dependency_binding")
    if report_binding != snapshot["q1b"]:
        errors.append("checked-in Q1b dependency binding differs from current graph")
    report_digests = report.get("q1b", {}).get("digests")
    expected_digests = {"graph_sha256": snapshot["graph"]["sha256"], "source_sha256": snapshot["source"]["sha256"]}
    if report_digests != expected_digests:
        errors.append("checked-in Q1b graph/source digest drift")
    if report.get("activation", {}).get("real_execution") != "PROHIBITED":
        errors.append("report real_execution exclusion must be PROHIBITED")
    if report.get("exclusions", {}).get("real_execution") != "PROHIBITED":
        errors.append("report real_execution exclusion parity is not PROHIBITED")
    if errors:
        raise ProfileValidationError("NAACL profile validation failed: " + "; ".join(errors))
    return snapshot


__all__ = [
    "PROFILE_ID",
    "ProfileValidationError",
    "build_naacl_profile_snapshot",
    "validate_naacl_profile",
]
