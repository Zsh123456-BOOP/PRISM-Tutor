"""Serializable runtime state shared by baselines and PRISM-Tutor."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from prism_tutor.agents.types import LLMCallRecord

from .errors import GraphErrorRecord


class StudentState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weak_skills: list[str] = Field(default_factory=list)
    active_misconceptions: list[str] = Field(default_factory=list)
    preferred_feedback: str | None = None
    recent_failures: list[str] = Field(default_factory=list)
    tentative_updates: list[dict[str, Any]] = Field(default_factory=list)


class StatePatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[dict[str, str]] = Field(default_factory=list)
    agent_outputs: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    risk_scores: list[dict[str, Any]] = Field(default_factory=list)
    selected_agents: list[str] = Field(default_factory=list)
    errors: list[GraphErrorRecord] = Field(default_factory=list)
    state_before: dict[str, Any] | None = None
    state_after: dict[str, Any] | None = None
    termination_reason: str | None = None


class TutorGraphState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample: dict[str, Any]
    method: str
    messages: list[dict[str, str]] = Field(default_factory=list)
    agent_outputs: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    llm_calls: list[dict[str, Any]] = Field(default_factory=list)
    risk_scores: list[dict[str, Any]] = Field(default_factory=list)
    selected_agents: list[str] = Field(default_factory=list)
    rounds: int = Field(default=0, ge=0)
    state_before: list[dict[str, Any]] = Field(default_factory=list)
    state_after: list[dict[str, Any]] = Field(default_factory=list)
    student_state: StudentState = Field(default_factory=StudentState)
    errors: list[GraphErrorRecord] = Field(default_factory=list)
    termination_reason: str | None = None
    total_tokens: int = Field(default=0, ge=0)
    budget_exhausted: bool = False

    def add_call(self, record: LLMCallRecord) -> None:
        call_dict = record.model_dump(mode="json")
        self.llm_calls.append(call_dict)
        self.total_tokens += record.usage.total_tokens
        if record.parsed_output is not None:
            self.agent_outputs.setdefault(record.agent_name, []).append(record.parsed_output)
        if record.error is not None:
            self.errors.append(
                GraphErrorRecord(
                    code="agent_failure",
                    message=record.error.message,
                    agent_name=record.agent_name,
                    round_index=self.rounds,
                    recoverable=record.error.retryable,
                )
            )

    def snapshot(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def apply_patch(self, patch: StatePatch) -> None:
        self.messages.extend(deepcopy(patch.messages))
        for agent_name, outputs in patch.agent_outputs.items():
            self.agent_outputs.setdefault(agent_name, []).extend(deepcopy(outputs))
        self.risk_scores.extend(deepcopy(patch.risk_scores))
        self.selected_agents.extend(patch.selected_agents)
        self.errors.extend(patch.errors)
        if patch.state_before is not None:
            self.state_before.append(deepcopy(patch.state_before))
        if patch.state_after is not None:
            self.state_after.append(deepcopy(patch.state_after))
        if patch.termination_reason is not None:
            self.termination_reason = patch.termination_reason
