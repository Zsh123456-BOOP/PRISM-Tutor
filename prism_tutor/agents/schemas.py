"""Pydantic schemas for all structured agent outputs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .types import SCHEMA_VERSION


Confidence = float
RiskLevel = Literal["low", "medium", "high"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SchemaVersionMixin(StrictModel):
    schema_version: str = SCHEMA_VERSION


class SolverOutput(SchemaVersionMixin):
    answer: str = Field(min_length=1)
    reasoning: list[str] = Field(min_length=1)
    confidence: Confidence = Field(ge=0, le=1)
    uncertainty: Confidence = Field(ge=0, le=1)
    needs_more_info: bool


class MisconceptionOutput(SchemaVersionMixin):
    misconception_detected: bool
    misconception_labels: list[str]
    evidence: list[str]
    severity: RiskLevel
    confidence: Confidence = Field(ge=0, le=1)


class PedagogyOutput(SchemaVersionMixin):
    strategy: Literal["scaffold", "socratic", "worked_example", "conceptual", "corrective"]
    rationale: str = Field(min_length=1)
    target_skills: list[str]
    confidence: Confidence = Field(ge=0, le=1)


class HintOutput(SchemaVersionMixin):
    hint_text: str = Field(min_length=1)
    hint_level: int = Field(ge=1, le=3)
    answer_leakage_risk: Confidence = Field(ge=0, le=1)
    confidence: Confidence = Field(ge=0, le=1)


class VerifierIssue(StrictModel):
    issue_type: Literal[
        "incorrect_answer",
        "leakage",
        "misconception",
        "pedagogy",
        "state_conflict",
        "format",
        "other",
    ]
    severity: RiskLevel
    message: str = Field(min_length=1)
    recommended_agent: Literal[
        "solver",
        "misconception",
        "pedagogy",
        "hint",
        "state_manager",
        "final_tutor",
    ] | None = None

    @field_validator("recommended_agent", mode="before")
    @classmethod
    def normalize_recommended_agent(cls, value: Any) -> Any:
        if value is None or value == "":
            return None
        normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "corrective": "pedagogy",
            "conceptual": "pedagogy",
            "scaffold": "pedagogy",
            "scaffolding": "pedagogy",
            "socratic": "pedagogy",
            "worked_example": "pedagogy",
            "tutor": "hint",
            "critic": None,
            "verifier": None,
            "verification": None,
            "state": "state_manager",
            "state_proposer": "state_manager",
            "state_commit": "state_manager",
            "final": "final_tutor",
            "final_response": "final_tutor",
        }
        return aliases.get(normalized, normalized)


class VerifierOutput(SchemaVersionMixin):
    approved: bool
    issues: list[VerifierIssue]
    leakage_detected: bool
    state_conflict_detected: bool
    confidence: Confidence = Field(ge=0, le=1)


class StudentStateUpdate(StrictModel):
    field: Literal["weak_skills", "active_misconceptions", "preferred_feedback", "recent_failures"]
    operation: Literal["add", "remove", "set"]
    value: Any
    confidence: Confidence = Field(ge=0, le=1)
    evidence: str = Field(min_length=1)


class StateManagerOutput(SchemaVersionMixin):
    proposed_updates: list[StudentStateUpdate]
    conflicts: list[str]
    confidence: Confidence = Field(ge=0, le=1)


class FinalTutorOutput(SchemaVersionMixin):
    response: str = Field(min_length=1)
    withheld_answer: bool
    confidence: Confidence = Field(ge=0, le=1)
    safety_notes: list[str]


class RiskEstimatorOutput(SchemaVersionMixin):
    answer_uncertainty: Confidence = Field(ge=0, le=1)
    misconception_risk: Confidence = Field(ge=0, le=1)
    pedagogy_risk: Confidence = Field(ge=0, le=1)
    leakage_risk: Confidence = Field(ge=0, le=1)
    state_conflict_risk: Confidence = Field(ge=0, le=1)
    estimated_difficulty: Confidence = Field(ge=0, le=1)
    total_risk: Confidence = Field(ge=0, le=1)
    risk_bucket: RiskLevel
    recommended_mode: Literal["direct", "guided", "deliberative"]


AGENT_SCHEMAS: dict[str, type[BaseModel]] = {
    "solver": SolverOutput,
    "misconception": MisconceptionOutput,
    "pedagogy": PedagogyOutput,
    "hint": HintOutput,
    "verifier": VerifierOutput,
    "state_manager": StateManagerOutput,
    "final_tutor": FinalTutorOutput,
    "risk_estimator": RiskEstimatorOutput,
}
