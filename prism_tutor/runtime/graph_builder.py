"""Runtime graph builder with an optional LangGraph backend."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .graph_state import TutorGraphState


NodeCallable = Callable[[TutorGraphState], TutorGraphState]


class SimpleGraph:
    backend = "simple_fallback"

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
    def __init__(self, *, prefer_langgraph: bool = True) -> None:
        self._nodes: list[tuple[str, NodeCallable]] = []
        self.prefer_langgraph = prefer_langgraph

    def add_node(self, name: str, node: NodeCallable) -> "GraphBuilder":
        self._nodes.append((name, node))
        return self

    def compile(self) -> Any:
        if self.prefer_langgraph:
            graph = self._compile_langgraph()
            if graph is not None:
                return graph
        return SimpleGraph(list(self._nodes))

    def _compile_langgraph(self) -> Any | None:
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError:
            return None

        graph = StateGraph(dict)
        for name, node in self._nodes:
            graph.add_node(name, _wrap_node(node))
        if not self._nodes:
            graph.add_edge(START, END)
        else:
            graph.add_edge(START, self._nodes[0][0])
            for (source, _), (target, _) in zip(self._nodes, self._nodes[1:], strict=False):
                graph.add_edge(source, target)
            graph.add_edge(self._nodes[-1][0], END)
        return LangGraphRuntime(graph.compile())


class LangGraphRuntime:
    backend = "langgraph"

    def __init__(self, compiled: Any) -> None:
        self.compiled = compiled

    def invoke(self, state: TutorGraphState | dict[str, Any]) -> TutorGraphState:
        graph_state = state if isinstance(state, TutorGraphState) else TutorGraphState.model_validate(state)
        result = self.compiled.invoke(graph_state.model_dump(mode="python"))
        if isinstance(result, TutorGraphState):
            return result
        return TutorGraphState.model_validate(result)

    __call__ = invoke


def _wrap_node(node: NodeCallable) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def wrapped(raw_state: dict[str, Any]) -> dict[str, Any]:
        graph_state = TutorGraphState.model_validate(raw_state)
        result = node(graph_state)
        return result.model_dump(mode="python")

    return wrapped
