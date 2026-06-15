"""Pedagogical QoS router."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from prism_tutor.agents.schemas import RiskEstimatorOutput


class RouterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    low_agents: list[str] = Field(default_factory=lambda: ["solver", "hint", "verifier", "final_tutor"])
    medium_agents: list[str] = Field(
        default_factory=lambda: ["solver", "misconception", "pedagogy", "hint", "verifier", "final_tutor"]
    )
    high_agents: list[str] = Field(
        default_factory=lambda: [
            "solver",
            "misconception",
            "pedagogy",
            "hint",
            "verifier",
            "state_manager",
            "final_tutor",
        ]
    )


class QoSRouter:
    def __init__(self, config: RouterConfig | None = None) -> None:
        self.config = config or RouterConfig()

    def select_agents(self, risk: RiskEstimatorOutput) -> list[str]:
        if risk.risk_bucket == "high":
            selected = list(self.config.high_agents)
        elif risk.risk_bucket == "medium":
            selected = list(self.config.medium_agents)
        else:
            selected = list(self.config.low_agents)

        if risk.leakage_risk >= 0.6:
            selected = self._ensure(selected, ["hint", "pedagogy", "verifier"])
        if risk.misconception_risk >= 0.6:
            selected = self._ensure(selected, ["misconception", "pedagogy"])
        if risk.state_conflict_risk >= 0.6:
            selected = self._ensure(selected, ["state_manager", "verifier"])
        if risk.answer_uncertainty >= 0.7:
            selected = self._ensure(selected, ["solver", "verifier"])
        return selected

    @staticmethod
    def _ensure(selected: list[str], required: list[str]) -> list[str]:
        for agent_name in required:
            if agent_name not in selected:
                selected.insert(max(0, len(selected) - 1), agent_name)
        return selected
