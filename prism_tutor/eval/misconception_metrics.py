"""Misconception precision/recall/F1 helpers."""

from __future__ import annotations

from typing import Any

from prism_tutor.utils.answers import canonicalize_label


def _as_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value} if value else set()
    if isinstance(value, dict):
        return {str(k) for k, enabled in value.items() if enabled}
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value if item is not None and str(item) != ""}
    return {str(value)}


def precision_recall_f1(predicted: Any, gold: Any) -> dict[str, Any]:
    gold_set = _as_set(gold)
    pred_set = _as_set(predicted)
    if not gold_set:
        return {
            "precision": None,
            "recall": None,
            "f1": None,
            "tp": 0,
            "fp": len(pred_set),
            "fn": 0,
            "coverage": 0.0,
            "reason": "missing_gold",
        }
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "coverage": 1.0,
        "reason": "set_overlap",
    }


def evaluate_misconceptions(record: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    parsed = record.get("parsed_output") if isinstance(record.get("parsed_output"), dict) else {}
    state = record.get("state") if isinstance(record.get("state"), dict) else {}
    agent_outputs = state.get("agent_outputs") if isinstance(state.get("agent_outputs"), dict) else {}
    misconception_outputs = agent_outputs.get("misconception") if isinstance(agent_outputs.get("misconception"), list) else []
    predicted_from_agent: list[str] = []
    for output in misconception_outputs:
        if isinstance(output, dict):
            predicted_from_agent.extend(_as_set(output.get("misconception_labels")))
    predicted = (
        parsed.get("misconceptions")
        or parsed.get("misconception_labels")
        or record.get("predicted_misconceptions")
        or record.get("misconceptions")
        or predicted_from_agent
    )
    expected = (
        gold.get("misconceptions")
        or gold.get("gold_misconceptions")
        or gold.get("misconception_labels")
        or gold.get("misconception_label")
    )
    # Constrained-classification scoring: when the benchmark ships a fixed
    # candidate label space, map each free-text prediction onto its canonical
    # candidate before computing F1. Without a candidate set we fall back to the
    # raw set-overlap behavior so other datasets/tests are unaffected.
    candidates = gold.get("candidate_misconceptions")
    if isinstance(candidates, list) and candidates:
        mapped = {canonicalize_label(label, candidates) for label in _as_set(predicted)}
        predicted = {label for label in mapped if label}
    result = precision_recall_f1(predicted, expected)
    return {f"misconception_{key}": value for key, value in result.items()}
