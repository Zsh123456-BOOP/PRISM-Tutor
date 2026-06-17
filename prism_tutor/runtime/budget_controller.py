"""Budgeted deliberation controller."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .graph_state import TutorGraphState


class BudgetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_rounds: int = Field(default=2, ge=1)
    max_tokens: int = Field(default=20000, ge=1)


class BudgetController:
    def __init__(self, config: BudgetConfig | None = None) -> None:
        self.config = config or BudgetConfig()

    def should_continue(self, state: TutorGraphState) -> bool:
        if state.rounds >= self.config.max_rounds:
            state.termination_reason = "max_rounds"
            return False
        if state.total_tokens >= self.config.max_tokens:
            state.termination_reason = "token_budget"
            state.budget_exhausted = True
            return False
        verifier_outputs = state.agent_outputs.get("verifier") or []
        latest = verifier_outputs[-1] if verifier_outputs else {}
        return not bool(latest.get("approved", False))

    def next_agents(self, state: TutorGraphState) -> list[str]:
        verifier_outputs = state.agent_outputs.get("verifier") or []
        latest = verifier_outputs[-1] if verifier_outputs else {}
        selected: list[str] = []
        for issue in latest.get("issues", []):
            recommended = issue.get("recommended_agent")
            if recommended:
                selected.append(recommended)
                continue
            issue_type = issue.get("issue_type")
            if issue_type == "leakage":
                selected.extend(["hint", "pedagogy"])
            elif issue_type == "incorrect_answer":
                selected.append("solver")
            elif issue_type == "misconception":
                selected.append("misconception")
            elif issue_type == "pedagogy":
                selected.append("pedagogy")
            elif issue_type == "state_conflict":
                selected.append("state_manager")
        if selected:
            selected.append("verifier")
        return list(dict.fromkeys(selected))
