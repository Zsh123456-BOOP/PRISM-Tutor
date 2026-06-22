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
    high_threshold: float = Field(default=0.60, ge=0, le=1)


def _latest(state: TutorGraphState, agent_name: str) -> dict[str, Any]:
    outputs = state.agent_outputs.get(agent_name) or []
    return outputs[-1] if outputs else {}


def _difficulty_from_sample(sample: dict[str, Any]) -> float:
    raw = sample.get("difficulty")
    if isinstance(raw, (int, float)):
        return max(0.0, min(1.0, float(raw)))
    if isinstance(raw, str):
        return {"easy": 0.2, "low": 0.2, "medium": 0.5, "hard": 0.8, "high": 0.8}.get(raw.lower(), 0.5)
    problem_text = " ".join(
        str(sample.get(key, ""))
        for key in ("question", "problem", "problem_text", "context", "student_utterance", "student_solution")
    )
    return max(0.2, min(0.8, len(problem_text) / 800))


def _has_value(value: Any) -> bool:
    if value is None or value == "":
        return False
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _has_any(record: dict[str, Any], keys: tuple[str, ...]) -> bool:
    return any(_has_value(record.get(key)) for key in keys)


def _visible_text(sample: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "question",
        "problem",
        "problem_text",
        "student_utterance",
        "student_answer",
        "student_solution",
        "dialogue_text",
        "context",
    ):
        value = sample.get(key)
        if _has_value(value):
            parts.append(str(value))
    return " ".join(parts).lower()


_AGREEMENT_TOKENS = (
    "thank", "thanks", "i see", "got it", "i get it", "makes sense", "make sense",
    "understand", "understood", "ok", "okay", "yes", "yeah", "yep", "right",
    "sure", "great", "cool", "alright", "all right", "that helps", "perfect",
)


_CONFUSION_MARKERS = (
    "not sure", "unsure", "confused", "don't", "do not", "dont", "how", "why",
    "what", "stuck", "lost", "help", "no idea", "can't", "cannot", "where",
)


def _is_low_stakes_turn(utterance: str) -> bool:
    """Infer (from the visible student turn only) whether this is a low-stakes turn
    -- a short social / acknowledgement turn with no fresh attempt, question, or
    computation. Such turns need no diagnosis, so the router can route a minimal
    agent set. Attempt / question / computation / confusion turns are NOT
    low-stakes (we bias toward diagnosis: a false negative only forgoes a saving,
    a false positive would skip needed diagnosis)."""
    text = (utterance or "").strip().lower()
    if not text:
        return False
    if "?" in text or any(char.isdigit() for char in text) or "=" in text:
        return False
    if any(marker in text for marker in _CONFUSION_MARKERS):
        return False
    words = text.split()
    if any(token in text for token in _AGREEMENT_TOKENS) and len(words) <= 14:
        return True
    return len(words) <= 5


def _sample_signals(sample: dict[str, Any]) -> dict[str, bool]:
    text = _visible_text(sample)
    has_student_work = _has_any(sample, ("student_utterance", "student_answer", "student_solution"))
    current_utterance = str(
        sample.get("student_utterance") or sample.get("student_answer") or sample.get("student_solution") or ""
    )
    low_stakes = has_student_work and _is_low_stakes_turn(current_utterance)
    math_like = any(char.isdigit() for char in text) or any(
        marker in text
        for marker in (
            "+",
            "-",
            "*",
            "/",
            "=",
            "solve",
            "equation",
            "fraction",
            "ratio",
            "percent",
            "area",
            "perimeter",
        )
    )
    confusion_like = has_student_work and any(
        marker in text
        for marker in (
            "i think",
            "i got",
            "not sure",
            "confused",
            "mistake",
            "wrong",
            "why",
            "how",
        )
    )
    dialogue_like = _has_any(sample, ("dialogue", "dialogue_text", "dialogue_history", "dialogue_turns"))
    return {
        "needs_solver": math_like and not low_stakes,
        "confusion": confusion_like,
        "has_student_work": has_student_work,
        "low_stakes": low_stakes,
        "dialogue_like": dialogue_like,
        "known_leakage": False,
    }


def estimate_risk(state: TutorGraphState, config: RiskConfig | None = None) -> RiskEstimatorOutput:
    cfg = config or RiskConfig()
    solver = _latest(state, "solver")
    misconception = _latest(state, "misconception")
    hint = _latest(state, "hint")
    verifier = _latest(state, "verifier")

    signals = _sample_signals(state.sample)
    difficulty = _difficulty_from_sample(state.sample)
    solver_ran = bool(solver)

    # Answer uncertainty: once the solver has run, trust its own (un)certainty;
    # before execution, grade by problem difficulty rather than forcing a near-
    # universal high floor (the old `needs_solver -> 0.72` collapsed every math
    # sample into "high risk", which removed all routing/budget adaptivity).
    if solver_ran:
        answer_uncertainty = float(solver.get("uncertainty", 1 - float(solver.get("confidence", 0.6))))
    else:
        answer_uncertainty = 0.32 + 0.6 * difficulty
    if not signals["needs_solver"]:
        answer_uncertainty = min(answer_uncertainty, 0.4)

    # Misconception risk: graded by difficulty, escalated only on an explicit
    # confusion signal or a detected misconception (no blanket high floor).
    if solver_ran or misconception:
        severity = {"low": 0.2, "medium": 0.55, "high": 0.9}.get(str(misconception.get("severity", "medium")), 0.5)
        misconception_risk = severity * float(misconception.get("confidence", 0.0))
    else:
        misconception_risk = 0.2 + 0.3 * difficulty
    if misconception.get("misconception_detected") is True:
        misconception_risk = max(misconception_risk, 0.66)
    if signals["confusion"]:
        misconception_risk = max(misconception_risk, 0.62)
    # Structural diagnosis floor: whenever a student has actually produced an
    # answer to a problem there is something to diagnose, so the misconception
    # agent should be routed (this is the case the Misconception Benchmark tests
    # and where confusion keywords are usually absent). Kept at a medium level so
    # it triggers routing (router threshold 0.45) WITHOUT forcing the deliberation
    # bucket to high; ablate_misconception_risk still zeroes this downstream.
    # The structural diagnosis floor applies only when there is an actual attempt
    # to diagnose -- NOT on low-stakes turns (short social / acknowledgement turns
    # like "I see, thanks"), which need no diagnosis. This is what lets the router
    # route a minimal agent set on the ~24% low-stakes turns of a full dialogue
    # (single-turn error datasets are never low-stakes, so they are unaffected).
    if signals["has_student_work"] and not signals["low_stakes"]:
        misconception_risk = max(misconception_risk, 0.5)

    # Pedagogy risk: graded by difficulty; escalated for dialogue / confused turns
    # and whenever we are responding to a student's answer.
    pedagogy_risk = 0.3 + 0.3 * difficulty
    if signals["dialogue_like"] or signals["confusion"]:
        pedagogy_risk = max(pedagogy_risk, 0.6)
    if signals["has_student_work"] and not signals["low_stakes"]:
        pedagogy_risk = max(pedagogy_risk, 0.5)
    leakage_risk = float(hint.get("answer_leakage_risk", 0.15))
    if signals["known_leakage"]:
        leakage_risk = max(leakage_risk, 0.65)
    state_conflict_risk = 0.15
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

    # Low-stakes turn: a short social / acknowledgement turn needs no diagnosis, so
    # keep the diagnosis-related risks low (this puts the turn in the low bucket and
    # routes a minimal agent set). Leakage / state risks are left untouched.
    if signals["low_stakes"] and not (misconception.get("misconception_detected") or verifier.get("issues")):
        answer_uncertainty = min(answer_uncertainty, 0.15)
        misconception_risk = min(misconception_risk, 0.15)
        pedagogy_risk = min(pedagogy_risk, 0.25)

    values = {
        "answer_uncertainty": max(0.0, min(1.0, answer_uncertainty)),
        "misconception_risk": max(0.0, min(1.0, misconception_risk)),
        "pedagogy_risk": max(0.0, min(1.0, pedagogy_risk)),
        "leakage_risk": max(0.0, min(1.0, leakage_risk)),
        "state_conflict_risk": max(0.0, min(1.0, state_conflict_risk)),
        "estimated_difficulty": difficulty,
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
