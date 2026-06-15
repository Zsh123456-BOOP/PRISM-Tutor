"""Small graph builder abstraction used until LangGraph is wired in."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .graph_state import TutorGraphState


NodeCallable = Callable[[TutorGraphState], TutorGraphState]


class SimpleGraph:
    def __init__(self, nodes: list[tuple[str, NodeCallable]]) -> None:
        self.nodes = nodes

    def invoke(self, state: TutorGraphState | dict[str, Any]) -> TutorGraphState:
        graph_state = state if isinstance(state, TutorGraphState) else TutorGraphState.model_validate(state)
        for _, node in self.nodes:
            if graph_state.termination_reason:
                break
            graph_state = node(graph_state)
        return graph_state

    __call__ = invoke


class GraphBuilder:
    def __init__(self) -> None:
        self._nodes: list[tuple[str, NodeCallable]] = []

    def add_node(self, name: str, node: NodeCallable) -> "GraphBuilder":
        self._nodes.append((name, node))
        return self

    def compile(self) -> SimpleGraph:
        return SimpleGraph(list(self._nodes))
