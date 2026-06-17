"""Student State Commit module."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .graph_state import StudentState, TutorGraphState


class CommitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    commit_confidence_threshold: float = Field(default=0.7, ge=0, le=1)
    tentative_confidence_threshold: float = Field(default=0.4, ge=0, le=1)


class CommitDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    committed_updates: list[dict[str, Any]] = Field(default_factory=list)
    tentative_updates: list[dict[str, Any]] = Field(default_factory=list)
    rejected_updates: list[dict[str, Any]] = Field(default_factory=list)


class StateCommitter:
    def __init__(self, config: CommitConfig | None = None) -> None:
        self.config = config or CommitConfig()

    def commit(self, state: TutorGraphState) -> CommitDecision:
        state_manager_outputs = state.agent_outputs.get("state_manager") or []
        if not state_manager_outputs:
            return CommitDecision(status="no_updates")
        verifier_outputs = state.agent_outputs.get("verifier") or []
        verifier = verifier_outputs[-1] if verifier_outputs else {}
        if verifier.get("state_conflict_detected"):
            return self._tentative_all(state, "verifier_state_conflict")

        state.state_before.append(state.student_state.model_dump(mode="json"))
        decision = CommitDecision(status="committed")
        for update in state_manager_outputs[-1].get("proposed_updates", []):
            confidence = float(update.get("confidence", 0.0))
            if confidence >= self.config.commit_confidence_threshold:
                self._apply_update(state.student_state, update)
                decision.committed_updates.append(deepcopy(update))
            elif confidence >= self.config.tentative_confidence_threshold:
                state.student_state.tentative_updates.append(deepcopy(update))
                decision.tentative_updates.append(deepcopy(update))
            else:
                decision.rejected_updates.append(deepcopy(update))
        state.state_after.append(state.student_state.model_dump(mode="json"))
        if decision.tentative_updates and not decision.committed_updates:
            decision.status = "tentative"
        if decision.rejected_updates and not decision.committed_updates and not decision.tentative_updates:
            decision.status = "rejected"
        return decision

    def commit_naive(self, state: TutorGraphState) -> CommitDecision:
        """Apply the latest state-manager updates without verifier or confidence gating."""
        state_manager_outputs = state.agent_outputs.get("state_manager") or []
        if not state_manager_outputs:
            return CommitDecision(status="no_updates")
        state.state_before.append(state.student_state.model_dump(mode="json"))
        decision = CommitDecision(status="naive_committed")
        for update in state_manager_outputs[-1].get("proposed_updates", []):
            self._apply_update(state.student_state, update)
            decision.committed_updates.append(deepcopy(update))
        state.state_after.append(state.student_state.model_dump(mode="json"))
        return decision

    def _tentative_all(self, state: TutorGraphState, reason: str) -> CommitDecision:
        latest = (state.agent_outputs.get("state_manager") or [{}])[-1]
        updates = [deepcopy(update) for update in latest.get("proposed_updates", [])]
        state.student_state.tentative_updates.extend(updates)
        return CommitDecision(status="tentative", tentative_updates=updates)

    @staticmethod
    def _apply_update(student_state: StudentState, update: dict[str, Any]) -> None:
        field = update["field"]
        operation = update["operation"]
        value = update.get("value")
        current = getattr(student_state, field)
        if operation == "set":
            setattr(student_state, field, value)
        elif operation == "add":
            if isinstance(current, list):
                values = value if isinstance(value, list) else [value]
                for item in values:
                    if item not in current:
                        current.append(item)
            else:
                setattr(student_state, field, value)
        elif operation == "remove" and isinstance(current, list):
            values = value if isinstance(value, list) else [value]
            setattr(student_state, field, [item for item in current if item not in values])
