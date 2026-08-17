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

from ..constants import TRAINING_SEEDS
from ..hashing import sha256_file, sha256_json
from ..orchestration.inventory import build_expected_runs
from ..orchestration.q1b_dependencies import (
    build_q1b_dependency_graph,
    load_q1b_producer_registry,
)

PROFILE_ID = "LUNA_NAACL_PROFILE"
PROFILE_CONFIG = "configs/experiments/naacl_balanced_runtime_profile.yaml"
PROFILE_REPORT = "reports/runtime_optimization/naacl_balanced_profile.json"
Q3_LOCAL_SYSTEMS = (
    "phobert_pragmatic_finetune",
    "vistral_pragmatic_sft",
    "vipragsent_full_vistral",
)
Q3_RETAINED_BUDGETS = ("32", "128", "512", "full")
Q3_AZURE_SYSTEM = "azure_gpt41_mini_8shot"
Q3_AZURE_SOURCE_FILES = (
    "configs/experiments/q3/system_aliases.yaml",
    "configs/experiments/q3/protocol.yaml",
    "configs/experiments/system_execution_registry.yaml",
    "src/vipragsent/orchestration/inventory.py",
)
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
PROTOCOL_SOURCE_FILES = (
    "configs/experiments/q3/system_aliases.yaml",
    "configs/experiments/q3/protocol.yaml",
    "configs/experiments/q2/protocol.yaml",
    "src/vipragsent/constants.py",
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


def _digest_files(root: Path, relatives: tuple[str, ...], label: str) -> dict[str, Any]:
    files = []
    for relative in relatives:
        path = root / relative
        if not path.is_file():
            raise ProfileValidationError(f"missing {label} profile source: {relative}")
        files.append({"path": relative, "sha256": sha256_file(path)})
    return {"files": files, "sha256": sha256_json(files)}


def _source_digest(root: Path) -> dict[str, Any]:
    return _digest_files(root, Q1B_SOURCE_FILES, "Q1b")


def _protocol_source_digest(root: Path) -> dict[str, Any]:
    return _digest_files(root, PROTOCOL_SOURCE_FILES, "Q3/Q2 protocol")


def _q3_azure_source_digest(root: Path) -> dict[str, Any]:
    return _digest_files(root, Q3_AZURE_SOURCE_FILES, "Q3 Azure")


def _normalise_tokens(values: Any) -> list[str]:
    if not isinstance(values, list | tuple):
        return []
    return [str(value) for value in values]


def _normalise_seeds(values: Any) -> list[int]:
    if not isinstance(values, list | tuple):
        return []
    try:
        return [int(value) for value in values]
    except (TypeError, ValueError):
        return []


def _q3_row_key(row: dict[str, Any]) -> tuple[str, str, Any]:
    seed = row.get("seed")
    return str(row.get("system_id")), str(row.get("budget")), seed


def _q3_expected_keys(
    *,
    local_systems: list[str],
    local_budgets: list[str],
    seeds: list[int],
    azure_system: str,
    azure_budgets: list[str],
) -> tuple[set[tuple[str, str, Any]], set[tuple[str, str, Any]]]:
    local = {
        (system_id, budget, seed)
        for system_id in local_systems
        for budget in local_budgets
        for seed in seeds
    }
    azure = {(azure_system, budget, None) for budget in azure_budgets}
    return local, azure


def validate_q3_profile_rows(
    rows: list[dict[str, Any]],
    *,
    local_systems: list[str] | None = None,
    local_budgets: list[str] | None = None,
    seeds: list[int] | None = None,
    azure_system: str = Q3_AZURE_SYSTEM,
    azure_budgets: list[str] | None = None,
) -> dict[str, Any]:
    """Validate the exact Q3 rows allowed into profile aggregation.

    Local rows use the three-dimensional ``system/budget/seed`` Cartesian
    product.  Azure is a fixed-prompt comparison and therefore has one row per
    retained budget with no seed axis.  This function deliberately validates a
    candidate aggregation input rather than silently filtering it.
    """

    local_systems = list(local_systems or Q3_LOCAL_SYSTEMS)
    local_budgets = list(local_budgets or Q3_RETAINED_BUDGETS)
    seeds = list(seeds or [int(seed) for seed in TRAINING_SEEDS])
    azure_budgets = list(azure_budgets or Q3_RETAINED_BUDGETS)
    expected_local, expected_azure = _q3_expected_keys(
        local_systems=local_systems,
        local_budgets=local_budgets,
        seeds=seeds,
        azure_system=azure_system,
        azure_budgets=azure_budgets,
    )
    expected = expected_local | expected_azure
    actual = [_q3_row_key(row) for row in rows]
    errors: list[str] = []
    duplicates = {key for key in actual if actual.count(key) > 1}
    if duplicates:
        errors.append(f"duplicate Q3 profile rows: {sorted(duplicates, key=str)}")
    missing = expected - set(actual)
    missing_azure = sorted(missing & expected_azure, key=str)
    if missing_azure:
        errors.append(f"missing retained Azure Q3 row(s): {missing_azure}")
    missing_local = sorted(missing & expected_local, key=str)
    if missing_local:
        errors.append(f"missing retained local Q3 cell(s): {missing_local}")
    extra = set(actual) - expected
    if extra:
        errors.append(f"out-of-profile Q3 row(s): {sorted(extra, key=str)}")
    invented_azure_seeds = sorted(
        {key for key in actual if key[0] == azure_system and key[2] is not None},
        key=str,
    )
    if invented_azure_seeds:
        errors.append(f"Azure Q3 rows must not define a seed axis: {invented_azure_seeds}")
    if errors:
        raise ProfileValidationError("Q3 profile aggregation validation failed: " + "; ".join(errors))
    return {
        "local_cell_count": len(expected_local),
        "azure_row_count": len(expected_azure),
        "total_row_count": len(expected),
        "azure_seed": None,
    }


def _protocol_binding(
    root: Path,
    config: dict[str, Any],
    inventory: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    aliases_payload = _load_yaml(root / "configs/experiments/q3/system_aliases.yaml")
    q3_protocol = _load_yaml(root / "configs/experiments/q3/protocol.yaml")
    q2_protocol = _load_yaml(root / "configs/experiments/q2/protocol.yaml")
    execution_registry = _load_yaml(root / "configs/experiments/system_execution_registry.yaml")
    aliases = aliases_payload.get("q3_system_aliases")
    if not isinstance(aliases, list):
        errors.append("Q3 alias source must contain q3_system_aliases")
        aliases = []
    alias_by_system: dict[str, list[dict[str, Any]]] = {}
    for alias in aliases:
        if isinstance(alias, dict):
            alias_by_system.setdefault(str(alias.get("resolved_system_id")), []).append(alias)

    q3 = config.get("q3", {})
    q2 = config.get("q2", {})
    retained_systems = _normalise_tokens(q3.get("systems"))
    retained_budgets = _normalise_tokens(q3.get("budgets"))
    profile_seeds = _normalise_seeds(q3.get("seeds"))
    q2_seeds = _normalise_seeds(q2.get("seeds"))
    locked_seeds = _normalise_seeds(TRAINING_SEEDS)

    for system_id in retained_systems:
        matches = alias_by_system.get(system_id, [])
        if len(matches) != 1 or matches[0].get("resolution_status") != "RESOLVED":
            errors.append(f"retained Q3 system alias drift: {system_id}")

    excluded_systems = [
        str(item.get("q3_system"))
        for item in config.get("exclusions", [])
        if isinstance(item, dict) and "q3_system" in item
    ]
    if excluded_systems != ["xlmr_pragmatic_finetune"]:
        errors.append("Q3 excluded system policy drift")
    if "xlmr_pragmatic_finetune" not in excluded_systems:
        errors.append("XLM-R Q3 exclusion is missing")
    if "xlmr_pragmatic_finetune" in retained_systems:
        errors.append("XLM-R Q3 is retained")
    xlmr_alias = alias_by_system.get("xlmr_pragmatic_finetune", [])
    if len(xlmr_alias) != 1 or xlmr_alias[0].get("paper_label") != "XLM-R" or xlmr_alias[0].get("resolution_status") != "RESOLVED":
        errors.append("XLM-R alias source drift")
    if Q3_AZURE_SYSTEM in excluded_systems:
        errors.append("protocol-defined Azure Q3 comparison is excluded")
    azure_alias = alias_by_system.get(Q3_AZURE_SYSTEM, [])
    if len(azure_alias) != 1 or azure_alias[0].get("resolution_status") != "RESOLVED":
        errors.append("Azure Q3 alias source drift")

    registry_entries = {
        str(item.get("system_id")): item
        for item in execution_registry.get("systems", [])
        if isinstance(item, dict) and item.get("system_id")
    }
    azure_registry = registry_entries.get(Q3_AZURE_SYSTEM)
    if azure_registry is None:
        errors.append("Azure Q3 execution registry entry is missing")
    else:
        expected_registry = {
            "executor_kind": "azure",
            "variant_id": "azure_pragmatic_8shot",
            "checkpoint_semantics": "fixed_prompt_no_checkpoint",
            "rationale_training": False,
            "rationale_inference": False,
        }
        for field, expected in expected_registry.items():
            if azure_registry.get(field) != expected:
                errors.append(f"Azure Q3 execution registry drift: {field}")

    source_budgets = _normalise_tokens(q3_protocol.get("q3", {}).get("budgets"))
    excluded_budgets = [
        str(item.get("q3_budget"))
        for item in config.get("exclusions", [])
        if isinstance(item, dict) and "q3_budget" in item
    ]
    required_excluded_budgets = ["64", "256"]
    if excluded_budgets != required_excluded_budgets:
        errors.append("Q3 excluded budget policy drift")
    if set(required_excluded_budgets) - set(source_budgets):
        errors.append("Q3 protocol source is missing budget 64 or 256")
    expected_retained_budgets = [budget for budget in source_budgets if budget not in required_excluded_budgets]
    if set(retained_budgets) != set(expected_retained_budgets):
        errors.append(
            f"Q3 retained budget drift: profile={sorted(retained_budgets)}, source={sorted(expected_retained_budgets)}"
        )

    azure_config = q3.get("azure_comparison", {})
    azure_budgets = _normalise_tokens(azure_config.get("budgets"))
    if azure_config.get("system_id") != Q3_AZURE_SYSTEM:
        errors.append("Q3 Azure comparison system drift")
    if "seeds" in azure_config or azure_config.get("seed_axis") != "absent":
        errors.append("Azure Q3 comparison must not define a seed axis")
    if azure_config.get("seed") is not None:
        errors.append("Azure Q3 comparison seed must be null")
    if azure_budgets != expected_retained_budgets:
        errors.append("Q3 Azure comparison budget drift")
    if azure_config.get("expected_row_count") != len(expected_retained_budgets):
        errors.append("Q3 Azure comparison row count drift")

    q3_source_rows = [
        row
        for row in inventory.get("rows", [])
        if str(row.get("research_question", "")).casefold() == "q3"
    ]
    q3_azure_rows = [row for row in q3_source_rows if str(row.get("system_id")) == Q3_AZURE_SYSTEM]
    source_azure_keys = {_q3_row_key(row) for row in q3_azure_rows}
    expected_azure_keys = {
        (Q3_AZURE_SYSTEM, budget, None)
        for budget in expected_retained_budgets
    }
    missing_azure = sorted(expected_azure_keys - source_azure_keys, key=str)
    if missing_azure:
        errors.append(f"missing retained Azure Q3 source row(s): {missing_azure}")
    if any(row.get("seed") is not None for row in q3_azure_rows):
        errors.append("Azure Q3 source rows must not define a seed axis")
    selected_rows = [
        row
        for row in q3_source_rows
        if _q3_row_key(row) in expected_azure_keys
    ]
    if not errors:
        try:
            azure_row_summary = validate_q3_profile_rows(
                [
                    row
                    for row in q3_source_rows
                    if _q3_row_key(row)
                    in (
                        {
                            (system, budget, seed)
                            for system in retained_systems
                            for budget in expected_retained_budgets
                            for seed in profile_seeds
                        }
                        | expected_azure_keys
                    )
                ],
                local_systems=retained_systems,
                local_budgets=expected_retained_budgets,
                seeds=profile_seeds,
                azure_system=Q3_AZURE_SYSTEM,
                azure_budgets=expected_retained_budgets,
            )
        except ProfileValidationError as exc:
            errors.append(str(exc))
            azure_row_summary = {
                "local_cell_count": len(retained_systems) * len(expected_retained_budgets) * len(profile_seeds),
                "azure_row_count": len(expected_retained_budgets),
                "total_row_count": None,
                "azure_seed": None,
            }
    else:
        azure_row_summary = {
            "local_cell_count": len(retained_systems) * len(expected_retained_budgets) * len(profile_seeds),
            "azure_row_count": len(expected_retained_budgets),
            "total_row_count": None,
            "azure_seed": None,
        }

    if profile_seeds != locked_seeds:
        errors.append(f"Q3 profile seed drift: profile={profile_seeds}, locked={locked_seeds}")
    if q2_seeds != locked_seeds:
        errors.append(f"Q2 profile seed drift: profile={q2_seeds}, locked={locked_seeds}")
    if q2.get("expected_seed_count") != len(q2_seeds):
        errors.append("Q2 expected seed count drift")
    if q3.get("selection_metric") != q3_protocol.get("q3", {}).get("primary_metric"):
        errors.append("Q3 primary metric drift")
    q2_variants = q2_protocol.get("variants")
    if not isinstance(q2_variants, list) or not q2_variants:
        errors.append("Q2 protocol source must contain variants")
        q2_variants = []
    profile_q2_variants = _normalise_tokens(q2.get("variants"))
    if profile_q2_variants != _normalise_tokens(q2_variants):
        errors.append("Q2 retained variants drift")
    expected_cells = len(retained_systems) * len(retained_budgets) * len(profile_seeds)
    if q3.get("expected_cell_count") != expected_cells or expected_cells != 36:
        errors.append(f"Q3 expected cell count drift: {expected_cells}")
    if q3.get("local_cell_count") != expected_cells:
        errors.append("Q3 local cell count drift")
    if q3.get("expected_total_row_count") != expected_cells + len(expected_retained_budgets):
        errors.append("Q3 total profile row count drift")

    binding = {
        "q3": {
            "alias_systems": [str(alias.get("resolved_system_id")) for alias in aliases if isinstance(alias, dict)],
            "retained_systems": retained_systems,
            "excluded_systems": excluded_systems,
            "source_budgets": source_budgets,
            "retained_budgets": retained_budgets,
            "excluded_budgets": required_excluded_budgets,
            "seeds": profile_seeds,
            "expected_cell_count": expected_cells,
            "primary_metric": q3_protocol.get("q3", {}).get("primary_metric"),
            "local_cell_count": expected_cells,
            "azure_comparison": {
                "system_id": Q3_AZURE_SYSTEM,
                "budgets": expected_retained_budgets,
                "seed": None,
                "seed_axis": "absent",
                "expected_row_count": len(expected_retained_budgets),
                "row_source": azure_config.get("row_source"),
                "comparison_kind": azure_config.get("comparison_kind"),
                "rows": [
                    {
                        "system_id": Q3_AZURE_SYSTEM,
                        "budget": budget,
                        "seed": None,
                        "run_id": next(
                            row["run_id"]
                            for row in selected_rows
                            if _q3_row_key(row) == (Q3_AZURE_SYSTEM, budget, None)
                        ),
                    }
                    for budget in expected_retained_budgets
                    if any(_q3_row_key(row) == (Q3_AZURE_SYSTEM, budget, None) for row in selected_rows)
                ],
                "row_summary": azure_row_summary,
            },
            "expected_total_row_count": expected_cells + len(expected_retained_budgets),
            "inventory_q3_row_count": len(q3_source_rows),
        },
        "q2": {
            "source_variants": _normalise_tokens(q2_variants),
            "retained_variants": profile_q2_variants,
            "seeds": q2_seeds,
            "expected_variant_count": len(q2_variants),
        },
    }
    return binding, errors


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
    protocol_binding, protocol_errors = _protocol_binding(root, config, inventory)
    errors.extend(protocol_errors)
    source = _source_digest(root)
    protocol_source = _protocol_source_digest(root)
    q3_azure_source = _q3_azure_source_digest(root)
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
        "protocol_sources": protocol_source,
        "q3_azure_sources": q3_azure_source,
        "protocol_binding": protocol_binding,
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
    q2_policy = config.get("q2", {})
    if q2_policy.get("expected_variant_count") != snapshot["protocol_binding"]["q2"]["expected_variant_count"]:
        errors.append("profile Q2 expected variant count drift")
    report = _load_json(root / PROFILE_REPORT)
    report_binding = report.get("q1b", {}).get("dependency_binding")
    if report_binding != snapshot["q1b"]:
        errors.append("checked-in Q1b dependency binding differs from current graph")
    report_digests = report.get("q1b", {}).get("digests")
    expected_digests = {"graph_sha256": snapshot["graph"]["sha256"], "source_sha256": snapshot["source"]["sha256"]}
    if report_digests != expected_digests:
        errors.append("checked-in Q1b graph/source digest drift")
    if report.get("protocol_sources") != snapshot["protocol_sources"]:
        errors.append("checked-in Q3/Q2 protocol source digest drift")
    if report.get("q3_azure_sources") != snapshot["q3_azure_sources"]:
        errors.append("checked-in Q3 Azure source digest drift")
    if report.get("protocol_binding") != snapshot["protocol_binding"]:
        errors.append("checked-in Q3/Q2 protocol binding differs from source")
    report_q3 = report.get("q3", {})
    expected_q3 = snapshot["protocol_binding"]["q3"]
    report_exclusions = report.get("exclusions", {})
    report_excluded_systems = _normalise_tokens(report_exclusions.get("q3_systems"))
    report_excluded_budgets = _normalise_tokens(report_exclusions.get("q3_budgets"))
    if report_excluded_systems != expected_q3["excluded_systems"]:
        errors.append(
            "checked-in report Q3 excluded system parity drift: "
            f"report={report_excluded_systems}, policy={expected_q3['excluded_systems']}"
        )
    if report_excluded_budgets != expected_q3["excluded_budgets"]:
        errors.append(
            "checked-in report Q3 excluded budget parity drift: "
            f"report={report_excluded_budgets}, policy={expected_q3['excluded_budgets']}"
        )
    if "xlmr_pragmatic_finetune" not in report_excluded_systems:
        errors.append("checked-in report is missing required XLM-R Q3 exclusion")
    if {"64", "256"} - set(report_excluded_budgets):
        errors.append("checked-in report is missing required Q3 budget exclusion")
    if Q3_AZURE_SYSTEM in report_excluded_systems:
        errors.append("checked-in report excludes retained Azure Q3 comparison")
    if {
        "systems": report_q3.get("systems"),
        "budgets": [str(value) for value in report_q3.get("budgets", [])],
        "seeds": report_q3.get("seeds"),
        "expected_cell_count": report_q3.get("expected_cell_count"),
        "local_cell_count": report_q3.get("local_cell_count"),
        "expected_total_row_count": report_q3.get("expected_total_row_count"),
    } != {
        "systems": expected_q3["retained_systems"],
        "budgets": expected_q3["retained_budgets"],
        "seeds": expected_q3["seeds"],
        "expected_cell_count": expected_q3["expected_cell_count"],
        "local_cell_count": expected_q3["local_cell_count"],
        "expected_total_row_count": expected_q3["expected_total_row_count"],
    }:
        errors.append("checked-in Q3 profile scope differs from source")
    if report_q3.get("azure_comparison") != expected_q3.get("azure_comparison"):
        errors.append("checked-in Azure Q3 comparison scope differs from source")
    report_q2 = report.get("q2", {})
    if report_q2.get("variants") != snapshot["protocol_binding"]["q2"]["retained_variants"] or report_q2.get("seeds") != snapshot["protocol_binding"]["q2"]["seeds"] or report_q2.get("expected_seed_count") != len(snapshot["protocol_binding"]["q2"]["seeds"]):
        errors.append("checked-in Q2 profile scope differs from source")
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
    "validate_q3_profile_rows",
    "validate_naacl_profile",
]
