"""Aggregate automatic metrics from generation logs and gold rows."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any

from .correctness import evaluate_internal_correctness
from .leakage_detector import detect_leakage
from .misconception_metrics import evaluate_misconceptions
from .routing_metrics import evaluate_routing
from .state_metrics import evaluate_state_metrics
from .token_counter import count_agent_calls, record_token_count


def _key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("dataset", "")), str(row.get("sample_id", "")))


def compute_record_metrics(record: dict[str, Any], gold: dict[str, Any] | None = None) -> dict[str, Any]:
    gold = gold or {}
    token_info = record_token_count(record)
    latency = record.get("latency") or record.get("latency_s") or record.get("elapsed_seconds")
    rounds = record.get("rounds")
    if isinstance(rounds, list):
        rounds = len(rounds)
    final_response = record.get("final_response") or record.get("response") or ""
    leakage = detect_leakage(final_response, gold=gold, sample_id=record.get("sample_id"))
    parsed_success = record.get("parse_success")
    if parsed_success is None:
        parsed_success = record.get("error") in (None, "")

    output = {
        "sample_id": record.get("sample_id"),
        "dataset": record.get("dataset"),
        "split": record.get("split"),
        "method": record.get("method"),
        "parse_success": bool(parsed_success),
        "parse_success_numerator": int(bool(parsed_success)),
        "parse_success_denominator": 1,
        "agent_calls": count_agent_calls(record),
        "rounds": int(rounds or 0),
        "latency": float(latency) if isinstance(latency, (int, float)) else None,
        **token_info,
        **evaluate_internal_correctness(record, gold),
        **evaluate_misconceptions(record, gold),
        **evaluate_routing(record, gold),
        **evaluate_state_metrics(record),
        "rule_leakage": leakage["rule_leakage"],
        "leakage_matched_rules": "|".join(leakage["matched_rules"]),
        "leakage_hit_count": len(leakage["hits"]),
    }
    return output


def compute_auto_metrics(
    generation_rows: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    gold_index = {_key(row): row for row in gold_rows or []}
    record_metrics: list[dict[str, Any]] = []
    orphan_generations: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()

    for record in generation_rows:
        key = _key(record)
        seen_keys.add(key)
        gold = gold_index.get(key)
        if gold is None and gold_index:
            orphan_generations.append({"dataset": key[0], "sample_id": key[1], "method": record.get("method")})
        record_metrics.append(compute_record_metrics(record, gold))

    missing_samples = [
        {"dataset": dataset, "sample_id": sample_id}
        for dataset, sample_id in sorted(set(gold_index) - seen_keys)
    ]

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in record_metrics:
        grouped[(str(row.get("dataset")), str(row.get("method")))].append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for (dataset, method), rows in sorted(grouped.items()):
        aggregate_rows.append(_aggregate_group(dataset, method, rows))

    coverage = {
        "generation_count": len(generation_rows),
        "gold_count": len(gold_index),
        "orphan_generation_count": len(orphan_generations),
        "missing_sample_count": len(missing_samples),
        "metrics_with_missing_gold": _missing_gold_counts(record_metrics),
    }
    return {
        "record_metrics": record_metrics,
        "aggregate_metrics": aggregate_rows,
        "coverage_report": coverage,
        "orphan_generations": orphan_generations,
        "missing_samples": missing_samples,
    }


def _mean_present(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float, bool))]
    return mean(float(value) for value in values) if values else None


def _coverage(rows: list[dict[str, Any]], key: str) -> float:
    return sum(row.get(key) is not None for row in rows) / len(rows) if rows else 0.0


def _aggregate_group(dataset: str, method: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "method": method,
        "n": len(rows),
        "total_tokens_mean": _mean_present(rows, "total_tokens"),
        "agent_calls_mean": _mean_present(rows, "agent_calls"),
        "rounds_mean": _mean_present(rows, "rounds"),
        "latency_mean": _mean_present(rows, "latency"),
        "parse_success_rate": _mean_present(rows, "parse_success"),
        "internal_correctness_mean": _mean_present(rows, "internal_correctness"),
        "misconception_f1_mean": _mean_present(rows, "misconception_f1"),
        "routing_f1_mean": _mean_present(rows, "routing_f1"),
        "state_conflict_rate_mean": _mean_present(rows, "state_conflict_rate"),
        "rule_leakage_rate": _mean_present(rows, "rule_leakage"),
        "internal_correctness_coverage": _coverage(rows, "internal_correctness"),
        "misconception_coverage": _coverage(rows, "misconception_f1"),
        "routing_coverage": _coverage(rows, "routing_f1"),
    }


def _missing_gold_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "internal_correctness": sum(row.get("internal_correctness_denominator") == 0 for row in rows),
        "misconception": sum(row.get("misconception_coverage") == 0.0 for row in rows),
        "routing": sum(row.get("routing_coverage") == 0.0 for row in rows),
    }
