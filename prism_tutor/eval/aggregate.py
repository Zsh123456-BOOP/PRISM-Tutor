"""Aggregate automatic metrics from generation logs and gold rows."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any

from .correctness import evaluate_internal_correctness
from .generation_records import deduplicate_generation_rows
from .judge_records import deduplicate_judge_rows, judge_row_is_valid
from .judge_merge import merge_leakage
from .leakage_detector import detect_leakage
from .misconception_metrics import evaluate_misconceptions
from .routing_metrics import evaluate_routing
from .state_metrics import evaluate_state_metrics
from .token_counter import count_agent_calls, record_token_count


def _key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("dataset", "")), str(row.get("sample_id", "")))


def _method_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("dataset", "")), str(row.get("sample_id", "")), str(row.get("method", "")))


def _judge_for_record(record: dict[str, Any], judge_index: dict[tuple[str, ...], dict[str, Any]]) -> dict[str, Any] | None:
    return judge_index.get(_method_key(record)) or judge_index.get(_key(record))


def _build_judge_index(judge_rows: list[dict[str, Any]] | None) -> dict[tuple[str, ...], dict[str, Any]]:
    index: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in judge_rows or []:
        method_key = _method_key(row)
        sample_key = _key(row)
        if method_key[2]:
            index[method_key] = row
        index.setdefault(sample_key, row)
    return index


def _valid_judge_row(judge_row: dict[str, Any] | None) -> bool:
    return judge_row_is_valid(judge_row)


def compute_record_metrics(
    record: dict[str, Any],
    gold: dict[str, Any] | None = None,
    judge_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gold = gold or {}
    token_info = record_token_count(record)
    latency = (
        record.get("latency")
        or record.get("latency_seconds")
        or record.get("latency_s")
        or record.get("elapsed_seconds")
    )
    rounds = record.get("rounds")
    if isinstance(rounds, list):
        rounds = len(rounds)
    final_response = record.get("final_response") or record.get("response") or ""
    leakage = detect_leakage(final_response, gold=gold, sample_id=record.get("sample_id"))
    parsed_success = record.get("parse_success")
    if parsed_success is None:
        parsed_success = record.get("error") in (None, "")
    if parsed_success:
        correctness_metrics = evaluate_internal_correctness(record, gold)
        misconception_metrics = evaluate_misconceptions(record, gold)
    else:
        correctness_metrics = {
            "internal_correctness": None,
            "internal_correctness_numerator": 0,
            "internal_correctness_denominator": 0,
            "internal_correctness_coverage": 0.0,
            "internal_correctness_reason": "parse_failed",
        }
        misconception_metrics = {
            "misconception_precision": None,
            "misconception_recall": None,
            "misconception_f1": None,
            "misconception_tp": 0,
            "misconception_fp": 0,
            "misconception_fn": 0,
            "misconception_coverage": 0.0,
            "misconception_reason": "parse_failed",
        }

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
        **correctness_metrics,
        **misconception_metrics,
        **evaluate_routing(record, gold),
        **evaluate_state_metrics(record),
        "rule_leakage": leakage["rule_leakage"],
        "leakage_matched_rules": "|".join(leakage["matched_rules"]),
        "leakage_hit_count": len(leakage["hits"]),
    }
    if _valid_judge_row(judge_row):
        merged = merge_leakage(output, judge_row)
        output.update(
            {
                "judge_leakage": merged["judge_leakage"],
                "final_leakage": merged["final_leakage"],
                "leakage_conflict": merged["leakage_conflict"],
                "judge_leakage_coverage": 1.0,
                "judge_error": None,
                "judge_parse_success": True,
            }
        )
    else:
        judge_error = judge_row.get("error") if isinstance(judge_row, dict) else None
        output.update(
            {
                "judge_leakage": None,
                "final_leakage": output["rule_leakage"],
                "leakage_conflict": None,
                "judge_leakage_coverage": 0.0,
                "judge_error": judge_error or ("missing_parsed_score" if judge_row is not None else None),
                "judge_parse_success": False if judge_row is not None else None,
            }
        )
    return output


def compute_auto_metrics(
    generation_rows: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]] | None = None,
    judge_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    generation_rows, deduplication_report = deduplicate_generation_rows(generation_rows)
    judge_rows, judge_deduplication_report = deduplicate_judge_rows(judge_rows or [])
    gold_index = {_key(row): row for row in gold_rows or []}
    judge_index = _build_judge_index(judge_rows)
    record_metrics: list[dict[str, Any]] = []
    orphan_generations: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    seen_method_keys: set[tuple[str, str, str]] = set()

    for record in generation_rows:
        key = _key(record)
        seen_keys.add(key)
        seen_method_keys.add(_method_key(record))
        gold = gold_index.get(key)
        if gold is None and gold_index:
            orphan_generations.append({"dataset": key[0], "sample_id": key[1], "method": record.get("method")})
        record_metrics.append(compute_record_metrics(record, gold, _judge_for_record(record, judge_index)))

    orphan_judges = _orphan_judge_rows(judge_rows, seen_keys, seen_method_keys)
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
        **deduplication_report,
        "gold_count": len(gold_index),
        **judge_deduplication_report,
        "judge_matched_count": sum(row.get("judge_leakage_coverage") == 1.0 for row in record_metrics),
        "judge_invalid_count": sum(row.get("judge_parse_success") is False for row in record_metrics),
        "orphan_judge_count": len(orphan_judges),
        "parse_failure_count": sum(not bool(row.get("parse_success")) for row in record_metrics),
        "orphan_generation_count": len(orphan_generations),
        "missing_sample_count": len(missing_samples),
        "metrics_with_missing_gold": _missing_gold_counts(record_metrics),
    }
    return {
        "record_metrics": record_metrics,
        "aggregate_metrics": aggregate_rows,
        "coverage_report": coverage,
        "orphan_generations": orphan_generations,
        "orphan_judges": orphan_judges,
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
        "judge_leakage_rate": _mean_present(rows, "judge_leakage"),
        "final_leakage_rate": _mean_present(rows, "final_leakage"),
        "leakage_conflict_rate": _mean_present(rows, "leakage_conflict"),
        "internal_correctness_coverage": _coverage(rows, "internal_correctness"),
        "misconception_coverage": _coverage(rows, "misconception_f1"),
        "routing_coverage": _coverage(rows, "routing_f1"),
        "judge_leakage_coverage": _mean_present(rows, "judge_leakage_coverage"),
    }


def _missing_gold_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "internal_correctness": sum(row.get("internal_correctness_denominator") == 0 for row in rows),
        "misconception": sum(row.get("misconception_coverage") == 0.0 for row in rows),
        "routing": sum(row.get("routing_coverage") == 0.0 for row in rows),
    }


def _orphan_judge_rows(
    judge_rows: list[dict[str, Any]],
    generation_sample_keys: set[tuple[str, str]],
    generation_method_keys: set[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    orphans: list[dict[str, Any]] = []
    for row in judge_rows:
        method_key = _method_key(row)
        sample_key = _key(row)
        method = method_key[2]
        matched = method_key in generation_method_keys if method else sample_key in generation_sample_keys
        if not matched:
            orphans.append({"dataset": sample_key[0], "sample_id": sample_key[1], "method": row.get("method")})
    return orphans
