"""Automatic correctness metrics for closed or lightly structured answers."""

from __future__ import annotations

import re
import unicodedata
from typing import Any


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


def evaluate_internal_correctness(record: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    candidate = (
        record.get("parsed_output", {}).get("answer")
        if isinstance(record.get("parsed_output"), dict)
        else None
    )
    if candidate is None:
        candidate = record.get("final_answer") or record.get("final_response")
    gold_answer = gold.get("answer") or gold.get("ground_truth") or gold.get("final_answer")
    match = exact_or_normalized_match(candidate, gold_answer)
    if match is None:
        return {
            "internal_correctness": None,
            "internal_correctness_numerator": 0,
            "internal_correctness_denominator": 0,
            "internal_correctness_coverage": 0.0,
            "internal_correctness_reason": "missing_gold",
        }
    return {
        "internal_correctness": float(match),
        "internal_correctness_numerator": int(match),
        "internal_correctness_denominator": 1,
        "internal_correctness_coverage": 1.0,
        "internal_correctness_reason": "normalized_match",
    }
