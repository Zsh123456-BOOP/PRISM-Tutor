"""Callable PRISM-Tutor graph skeleton for M1/M2/M3."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from prism_tutor.agents.base_client import BaseLLMClient
from prism_tutor.agents.final_tutor import FinalTutorAgent
from prism_tutor.agents.hint import HintAgent
from prism_tutor.agents.misconception import MisconceptionAgent
from prism_tutor.agents.pedagogy import PedagogyAgent
from prism_tutor.agents.solver import SolverAgent
from prism_tutor.agents.state_manager import StateManagerAgent
from prism_tutor.agents.verifier import VerifierAgent

from .budget_controller import BudgetConfig, BudgetController
from .graph_state import TutorGraphState
from .qos_router import QoSRouter, RouterConfig
from .risk_estimator import RiskConfig, estimate_risk
from .state_commit import CommitConfig, StateCommitter


AGENT_REGISTRY = {
    "solver": SolverAgent(),
    "misconception": MisconceptionAgent(),
    "pedagogy": PedagogyAgent(),
    "hint": HintAgent(),
    "verifier": VerifierAgent(),
    "state_manager": StateManagerAgent(),
    "final_tutor": FinalTutorAgent(),
}


class PrismGraphConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk: RiskConfig = Field(default_factory=RiskConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    commit: CommitConfig = Field(default_factory=CommitConfig)


class PrismGraph:
    """Graph-like callable implementing M1, M2, and M3 behavior.

    M1: routing only.
    M2: routing plus budgeted deliberation.
    M3: routing, budgeted deliberation, and state commit.
    """

    def __init__(
        self,
        *,
        method: str = "M3",
        client: BaseLLMClient | None = None,
        config: PrismGraphConfig | None = None,
    ) -> None:
        if method not in {"M1", "M2", "M3"}:
            raise ValueError("method must be one of M1, M2, M3")
        self.method = method
        self.client = client or BaseLLMClient()
        self.config = config or PrismGraphConfig()
        self.router = QoSRouter(self.config.router)
        self.budget = BudgetController(self.config.budget)
        self.committer = StateCommitter(self.config.commit)

    def invoke(self, state: TutorGraphState | dict[str, Any]) -> TutorGraphState:
        graph_state = state if isinstance(state, TutorGraphState) else TutorGraphState.model_validate(state)
        graph_state.method = self.method
        risk = estimate_risk(graph_state, self.config.risk)
        graph_state.risk_scores.append(risk.model_dump(mode="json"))
        selected = self.router.select_agents(risk)
        graph_state.selected_agents.extend(selected)
        self._run_agents(graph_state, selected)

        if self.method in {"M2", "M3"}:
            while self.budget.should_continue(graph_state):
                next_agents = self.budget.next_agents(graph_state)
                if not next_agents:
                    break
                graph_state.rounds += 1
                graph_state.selected_agents.extend(next_agents)
                self._run_agents(graph_state, next_agents)

        if self.method == "M3":
            decision = self.committer.commit(graph_state)
            graph_state.agent_outputs.setdefault("state_commit", []).append(decision.model_dump(mode="json"))

        if graph_state.termination_reason is None:
            graph_state.termination_reason = "completed"
        return graph_state

    __call__ = invoke

    def _run_agents(self, state: TutorGraphState, agents: list[str]) -> None:
        for agent_name in agents:
            agent = AGENT_REGISTRY.get(agent_name)
            if agent is None:
                continue
            record = agent.invoke(
                sample=state.sample,
                state=self._agent_context(state),
                client=self.client,
                method=self.method,
            )
            state.add_call(record)

    @staticmethod
    def _agent_context(state: TutorGraphState) -> dict[str, Any]:
        return {
            "method": state.method,
            "rounds": state.rounds,
            "student_state": state.student_state.model_dump(mode="json"),
            "agent_outputs": state.agent_outputs,
            "risk_scores": state.risk_scores,
            "selected_agents": state.selected_agents,
            "total_tokens": state.total_tokens,
        }


def build_prism_graph(
    method: str = "M3",
    *,
    client: BaseLLMClient | None = None,
    config: PrismGraphConfig | None = None,
) -> PrismGraph:
    return PrismGraph(method=method, client=client, config=config)
