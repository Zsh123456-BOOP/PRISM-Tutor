"""Pedagogical risk estimator for PRISM-Tutor routing."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from prism_tutor.agents.schemas import RiskEstimatorOutput

from .graph_state import TutorGraphState


class RiskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "answer_uncertainty": 1.0,
            "misconception_risk": 1.1,
            "pedagogy_risk": 0.8,
            "leakage_risk": 1.2,
            "state_conflict_risk": 1.0,
            "estimated_difficulty": 0.7,
        }
    )
    low_threshold: float = Field(default=0.33, ge=0, le=1)
    high_threshold: float = Field(default=0.66, ge=0, le=1)


def _latest(state: TutorGraphState, agent_name: str) -> dict[str, Any]:
    outputs = state.agent_outputs.get(agent_name) or []
    return outputs[-1] if outputs else {}


def _difficulty_from_sample(sample: dict[str, Any]) -> float:
    raw = sample.get("difficulty")
    if isinstance(raw, (int, float)):
        return max(0.0, min(1.0, float(raw)))
    if isinstance(raw, str):
        return {"easy": 0.2, "low": 0.2, "medium": 0.5, "hard": 0.8, "high": 0.8}.get(raw.lower(), 0.5)
    problem_text = " ".join(str(sample.get(key, "")) for key in ("question", "problem", "student_utterance"))
    return max(0.2, min(0.8, len(problem_text) / 800))


def estimate_risk(state: TutorGraphState, config: RiskConfig | None = None) -> RiskEstimatorOutput:
    cfg = config or RiskConfig()
    solver = _latest(state, "solver")
    misconception = _latest(state, "misconception")
    hint = _latest(state, "hint")
    verifier = _latest(state, "verifier")

    answer_uncertainty = float(solver.get("uncertainty", 1 - float(solver.get("confidence", 0.5))))
    severity = {"low": 0.2, "medium": 0.55, "high": 0.9}.get(str(misconception.get("severity", "medium")), 0.5)
    misconception_risk = severity * float(misconception.get("confidence", 0.5))
    if misconception.get("misconception_detected") is True:
        misconception_risk = max(misconception_risk, 0.6)

    pedagogy_risk = 0.35
    leakage_risk = float(hint.get("answer_leakage_risk", 0.2))
    state_conflict_risk = 0.2
    for issue in verifier.get("issues", []):
        issue_type = issue.get("issue_type")
        issue_severity = {"low": 0.3, "medium": 0.6, "high": 0.9}.get(issue.get("severity"), 0.5)
        if issue_type == "leakage":
            leakage_risk = max(leakage_risk, issue_severity)
        elif issue_type == "pedagogy":
            pedagogy_risk = max(pedagogy_risk, issue_severity)
        elif issue_type == "state_conflict":
            state_conflict_risk = max(state_conflict_risk, issue_severity)
        elif issue_type == "misconception":
            misconception_risk = max(misconception_risk, issue_severity)

    if verifier.get("leakage_detected"):
        leakage_risk = max(leakage_risk, 0.8)
    if verifier.get("state_conflict_detected") or state.student_state.tentative_updates:
        state_conflict_risk = max(state_conflict_risk, 0.7)

    values = {
        "answer_uncertainty": max(0.0, min(1.0, answer_uncertainty)),
        "misconception_risk": max(0.0, min(1.0, misconception_risk)),
        "pedagogy_risk": max(0.0, min(1.0, pedagogy_risk)),
        "leakage_risk": max(0.0, min(1.0, leakage_risk)),
        "state_conflict_risk": max(0.0, min(1.0, state_conflict_risk)),
        "estimated_difficulty": _difficulty_from_sample(state.sample),
    }
    weight_sum = sum(max(0.0, cfg.weights.get(key, 0.0)) for key in values) or 1.0
    total = sum(values[key] * max(0.0, cfg.weights.get(key, 0.0)) for key in values) / weight_sum
    if total >= cfg.high_threshold:
        bucket = "high"
        mode = "deliberative"
    elif total >= cfg.low_threshold:
        bucket = "medium"
        mode = "guided"
    else:
        bucket = "low"
        mode = "direct"
    return RiskEstimatorOutput(**values, total_risk=total, risk_bucket=bucket, recommended_mode=mode)
