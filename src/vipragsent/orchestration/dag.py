from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..config import load_yaml


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

    def run(self, state_path: str | Path, handlers: dict[str, Callable[[DAGNode], Any]], *, resume: bool = False, force: bool = False) -> dict[str, Any]:
        path = Path(state_path)
        state = json.loads(path.read_text(encoding="utf-8")) if resume and path.exists() else {"nodes": {}, "status": "running"}
        for node in self.topological_order():
            previous = state["nodes"].get(node.node_id, {})
            if previous.get("status") == "PASS" and not force:
                continue
            if any(state["nodes"].get(dep, {}).get("status") != "PASS" for dep in node.depends_on):
                raise RuntimeError(f"Dependency not complete for {node.node_id}")
            try:
                result = handlers[node.kind](node)
                state["nodes"][node.node_id] = {"status": "PASS", "result": result}
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(state, indent=2, default=str) + "\n", encoding="utf-8")
            except Exception as exc:
                state["nodes"][node.node_id] = {"status": "FAIL", "error": str(exc)}
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(state, indent=2, default=str) + "\n", encoding="utf-8")
                raise
        state["status"] = "PASS"
        path.write_text(json.dumps(state, indent=2, default=str) + "\n", encoding="utf-8")
        return state


def load_master_dag(path: str | Path = "configs/experiments/master_matrix.yaml") -> ExperimentDAG:
    config = load_yaml(path)
    nodes = [DAGNode(item["id"], item["kind"], tuple(item.get("depends_on", []))) for item in config.get("nodes", [])]
    return ExperimentDAG(nodes)
