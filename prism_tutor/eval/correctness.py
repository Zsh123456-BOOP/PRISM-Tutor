"""Automatic correctness metrics for closed or lightly structured answers."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from prism_tutor.utils.answers import answers_match


def normalize_answer(value: Any) -> str:
    text = unicodedata.normalize("NFKC", "" if value is None else str(value))
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\s,.;:!?]+$", "", text)
    return text


def exact_or_normalized_match(candidate: Any, gold: Any) -> bool | None:
    if gold is None or gold == "":
        return None
    return normalize_answer(candidate) == normalize_answer(gold)


def _latest_solver_answer(record: dict[str, Any]) -> Any:
    state = record.get("state") if isinstance(record.get("state"), dict) else {}
    agent_outputs = state.get("agent_outputs") if isinstance(state.get("agent_outputs"), dict) else {}
    solver_outputs = agent_outputs.get("solver") if isinstance(agent_outputs.get("solver"), list) else []
    for output in reversed(solver_outputs):
        if isinstance(output, dict) and output.get("answer") not in (None, ""):
            return output.get("answer")
    llm_calls = state.get("llm_calls") if isinstance(state.get("llm_calls"), list) else record.get("llm_calls")
    if isinstance(llm_calls, list):
        for call in reversed(llm_calls):
            if not isinstance(call, dict) or call.get("agent_name") != "solver":
                continue
            parsed = call.get("parsed_output") if isinstance(call.get("parsed_output"), dict) else {}
            if parsed.get("answer") not in (None, ""):
                return parsed.get("answer")
    parsed = record.get("parsed_output") if isinstance(record.get("parsed_output"), dict) else {}
    if record.get("method") == "solver" and parsed.get("answer") not in (None, ""):
        return parsed.get("answer")
    return None


def _missing_correctness(reason: str) -> dict[str, Any]:
    return {
        "solver_correctness": None,
        "solver_correctness_numerator": 0,
        "solver_correctness_denominator": 0,
        "solver_correctness_coverage": 0.0,
        "solver_correctness_reason": reason,
        "internal_correctness": None,
        "internal_correctness_numerator": 0,
        "internal_correctness_denominator": 0,
        "internal_correctness_coverage": 0.0,
        "internal_correctness_reason": "solver_correctness_alias",
    }


def evaluate_solver_correctness(record: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    candidate = _latest_solver_answer(record)
    metadata = gold.get("metadata") if isinstance(gold.get("metadata"), dict) else {}
    # Prefer the extracted final answer; fall back to the full ground-truth
    # narrative (answers_match extracts the final number from either side).
    gold_answer = (
        gold.get("answer")
        or gold.get("final_answer")
        or gold.get("final_numeric_answer")
        or metadata.get("final_answer")
        or metadata.get("correct_answer")
        or gold.get("ground_truth")
        or metadata.get("ground_truth")
    )
    if gold_answer in (None, ""):
        return _missing_correctness("missing_gold")
    if candidate in (None, ""):
        return _missing_correctness("missing_solver_answer")
    match = answers_match(candidate, gold_answer)
    if match is None:
        return _missing_correctness("uncomparable_answer")
    return {
        "solver_correctness": float(match),
        "solver_correctness_numerator": int(match),
        "solver_correctness_denominator": 1,
        "solver_correctness_coverage": 1.0,
        "solver_correctness_reason": "normalized_match",
        "internal_correctness": float(match),
        "internal_correctness_numerator": int(match),
        "internal_correctness_denominator": 1,
        "internal_correctness_coverage": 1.0,
        "internal_correctness_reason": "solver_correctness_alias",
    }


def evaluate_internal_correctness(record: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    """Backward-compatible alias for solver correctness.

    Final tutor responses are scaffolded and should not be exact-matched
    against the gold answer.
    """
    return evaluate_solver_correctness(record, gold)
