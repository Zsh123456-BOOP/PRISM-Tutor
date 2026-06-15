"""Merge judge answer-leakage scores with rule-detector outputs."""

from __future__ import annotations

from typing import Any


def merge_leakage(rule_row: dict[str, Any], judge_row: dict[str, Any]) -> dict[str, Any]:
    parsed = judge_row.get("parsed_score") if isinstance(judge_row.get("parsed_score"), dict) else {}
    rule_leakage = bool(rule_row.get("rule_leakage"))
    judge_leakage = bool(parsed.get("answer_leakage"))
    return {
        "sample_id": rule_row.get("sample_id") or judge_row.get("sample_id"),
        "dataset": rule_row.get("dataset") or judge_row.get("dataset"),
        "method": rule_row.get("method") or judge_row.get("method"),
        "rule_leakage": rule_leakage,
        "judge_leakage": judge_leakage,
        "final_leakage": rule_leakage or judge_leakage,
        "leakage_conflict": rule_leakage != judge_leakage,
        "matched_rules": rule_row.get("matched_rules") or rule_row.get("leakage_matched_rules"),
    }
