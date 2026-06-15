"""Misconception precision/recall/F1 helpers."""

from __future__ import annotations

from typing import Any


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
    predicted = (
        parsed.get("misconceptions")
        or record.get("predicted_misconceptions")
        or record.get("misconceptions")
    )
    expected = (
        gold.get("misconceptions")
        or gold.get("gold_misconceptions")
        or gold.get("misconception_labels")
    )
    result = precision_recall_f1(predicted, expected)
    return {f"misconception_{key}": value for key, value in result.items()}
