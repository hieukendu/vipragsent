from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..atomic import atomic_write_json
from ..config import load_yaml
from .status import HandlerResult, NodeStatus, ProtocolConflict, RunExitCode, RuntimeBlocked


@dataclass(frozen=True)
class DAGNode:
    node_id: str
    kind: str
    depends_on: tuple[str, ...]


class ExperimentDAG:
    def __init__(self, nodes: list[DAGNode]) -> None:
        self.nodes = {node.node_id: node for node in nodes}
        self._validate()

    def _validate(self) -> None:
        for node in self.nodes.values():
            missing = set(node.depends_on) - set(self.nodes)
            if missing:
                raise ValueError(f"Node {node.node_id} has missing dependencies: {sorted(missing)}")
        self.topological_order()

    def topological_order(self) -> list[DAGNode]:
        remaining = {key: set(node.depends_on) for key, node in self.nodes.items()}
        ordered: list[DAGNode] = []
        while remaining:
            ready = sorted(key for key, deps in remaining.items() if not deps)
            if not ready:
                raise ValueError("Experiment DAG contains a cycle")
            for key in ready:
                ordered.append(self.nodes[key])
                remaining.pop(key)
            for deps in remaining.values():
                deps.difference_update(ready)
        return ordered

    def plan_lines(self) -> list[str]:
        return [f"{index + 1:02d}. {node.node_id} [{node.kind}] depends_on={','.join(node.depends_on) or '-'}" for index, node in enumerate(self.topological_order())]

    def run(
        self,
        state_path: str | Path,
        handlers: Mapping[str, Callable[[DAGNode], HandlerResult]],
        *,
        resume: bool = False,
        force: bool = False,
        max_attempts: int = 2,
    ) -> dict[str, Any]:
        path = Path(state_path)
        state = json.loads(path.read_text(encoding="utf-8")) if resume and path.exists() else {
            "version": 2,
            "nodes": {},
            "status": NodeStatus.PENDING.value,
            "protocol_conflict": False,
        }
        state.setdefault("version", 2)
        state.setdefault("nodes", {})
        state.setdefault("protocol_conflict", False)
        for node in self.topological_order():
            previous = state["nodes"].get(node.node_id, {})
            if previous.get("status") == NodeStatus.PASS.value and not force:
                continue
            dependency_states = {dep: state["nodes"].get(dep, {}).get("status") for dep in node.depends_on}
            blocked_dependencies = [dep for dep, status in dependency_states.items() if status == NodeStatus.BLOCKED.value]
            failed_dependencies = [dep for dep, status in dependency_states.items() if status == NodeStatus.FAIL.value]
            if blocked_dependencies:
                state["nodes"][node.node_id] = HandlerResult.blocked(
                    f"Dependency blocked: {', '.join(blocked_dependencies)}"
                ).as_dict()
                atomic_write_json(path, state)
                continue
            if failed_dependencies:
                state["nodes"][node.node_id] = HandlerResult.failed(
                    f"Dependency failed: {', '.join(failed_dependencies)}"
                ).as_dict()
                state["status"] = NodeStatus.FAIL.value
                atomic_write_json(path, state)
                return state
            if any(status != NodeStatus.PASS.value for status in dependency_states.values()):
                state["nodes"][node.node_id] = HandlerResult.blocked(
                    f"Dependency is not complete: {dependency_states}"
                ).as_dict()
                atomic_write_json(path, state)
                continue
            if node.kind not in handlers:
                result = HandlerResult.failed(f"No production handler registered for DAG kind {node.kind!r}")
                state["nodes"][node.node_id] = result.as_dict()
                state["status"] = NodeStatus.FAIL.value
                atomic_write_json(path, state)
                return state
            state["nodes"][node.node_id] = {"status": NodeStatus.RUNNING.value, "attempt": 0}
            atomic_write_json(path, state)
            try:
                result = HandlerResult.failed("handler did not return a HandlerResult")
                for attempt in range(1, max_attempts + 1):
                    state["nodes"][node.node_id]["attempt"] = attempt
                    atomic_write_json(path, state)
                    try:
                        result = handlers[node.kind](node)
                    except (RuntimeBlocked, ProtocolConflict) as exc:
                        if isinstance(exc, ProtocolConflict):
                            state["protocol_conflict"] = True
                        result = HandlerResult.blocked(str(exc))
                    except Exception as exc:  # handlers must not leak untyped outcomes into the state file
                        result = HandlerResult.failed(f"{type(exc).__name__}: {exc}")
                    if result.status is not NodeStatus.FAIL or not result.retryable or attempt >= max_attempts:
                        break
                state["nodes"][node.node_id] = {"node_id": node.node_id, **result.as_dict()}
                atomic_write_json(path, state)
                if result.status is NodeStatus.FAIL:
                    state["status"] = NodeStatus.FAIL.value
                    atomic_write_json(path, state)
                    return state
            except Exception as exc:
                state["nodes"][node.node_id] = HandlerResult.failed(f"{type(exc).__name__}: {exc}").as_dict()
                state["status"] = NodeStatus.FAIL.value
                atomic_write_json(path, state)
                return state
        statuses = {item.get("status") for item in state["nodes"].values()}
        state["status"] = (
            NodeStatus.FAIL.value
            if NodeStatus.FAIL.value in statuses
            else NodeStatus.BLOCKED.value
            if NodeStatus.BLOCKED.value in statuses
            else NodeStatus.PASS.value
        )
        state["exit_code"] = RunExitCode.PROTOCOL_FAILURE if state.get("protocol_conflict") else {
            NodeStatus.PASS.value: RunExitCode.SUCCESS,
            NodeStatus.BLOCKED.value: RunExitCode.BLOCKED,
            NodeStatus.FAIL.value: RunExitCode.EXECUTION_FAILURE,
        }[state["status"]]
        atomic_write_json(path, state)
        return state


def load_master_dag(path: str | Path = "configs/experiments/master_matrix.yaml") -> ExperimentDAG:
    config = load_yaml(path)
    nodes = [DAGNode(item["id"], item["kind"], tuple(item.get("depends_on", []))) for item in config.get("nodes", [])]
    return ExperimentDAG(nodes)
