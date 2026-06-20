#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prism_tutor.eval.aggregate import compute_auto_metrics
from prism_tutor.eval.generation_records import deduplicate_generation_rows
from prism_tutor.eval.io import read_jsonl, write_csv, write_json, write_jsonl
from prism_tutor.eval.leakage_detector import detect_leakage


def _files(path: str | Path) -> list[Path]:
    if path in ("", None):
        return []
    p = Path(path)
    if not p.exists():
        return []
    if p.is_dir():
        return sorted(p.glob("*.jsonl"))
    return [p]


def _read_many(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return rows


def _key(row: dict) -> tuple[str, str]:
    return (str(row.get("dataset", "")), str(row.get("sample_id", "")))


def _subset_rows(rows: list[dict], fields: list[str]) -> list[dict]:
    return [{field: row.get(field) for field in fields} for row in rows]


def _leakage_hits(generation_rows: list[dict], gold_rows: list[dict]) -> list[dict]:
    gold_index = {_key(row): row for row in gold_rows}
    hits = []
    for record in generation_rows:
        detection = detect_leakage(
            record.get("final_response") or record.get("response") or "",
            gold=gold_index.get(_key(record), {}),
            sample_id=record.get("sample_id"),
        )
        for hit in detection["hits"]:
            hits.append(
                {
                    "dataset": record.get("dataset"),
                    "split": record.get("split"),
                    "sample_id": record.get("sample_id"),
                    "method": record.get("method"),
                    **hit,
                }
            )
    return hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute automatic metrics from generation JSONL logs.")
    parser.add_argument("--generations", default="outputs/generations")
    parser.add_argument("--gold", default="data/splits")
    parser.add_argument("--judge-scores", default="", help="Optional judge_scores.jsonl or directory for final leakage merge.")
    parser.add_argument("--output_dir", default="outputs/metrics")
    args = parser.parse_args(argv)

    generation_rows = _read_many(_files(args.generations))
    deduped_generation_rows, _ = deduplicate_generation_rows(generation_rows)
    gold_rows = _read_many(_files(args.gold))
    judge_rows = _read_many(_files(args.judge_scores))
    result = compute_auto_metrics(generation_rows, gold_rows, judge_rows)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    record_fields = sorted({key for row in result["record_metrics"] for key in row})
    aggregate_fields = sorted({key for row in result["aggregate_metrics"] for key in row})
    write_jsonl(out / "record_auto_metrics.jsonl", result["record_metrics"])
    write_csv(out / "main_auto_metrics.csv", result["aggregate_metrics"], aggregate_fields)
    routing_fields = [
        "dataset",
        "split",
        "sample_id",
        "method",
        "routing_precision",
        "routing_recall",
        "routing_f1",
        "routing_tp",
        "routing_fp",
        "routing_fn",
        "routing_coverage",
        "routing_reason",
    ]
    state_fields = [
        "dataset",
        "split",
        "sample_id",
        "method",
        "state_event_count",
        "state_conflict_rate",
        "incorrect_commit_rate",
        "tentative_update_rate",
        "unsafe_commit_rate",
        "tentative_when_conflict_rate",
        "commit_with_evidence_rate",
        "state_metric_coverage",
        "external_state_accuracy",
        "external_state_coverage",
        "incorrect_misconception_commit_rate",
        "misconception_commit_precision",
        "final_state_contradiction",
        "noisy_state_update_rejection_accuracy",
    ]
    leakage_fields = [
        "dataset",
        "split",
        "sample_id",
        "method",
        "rule_leakage",
        "judge_leakage",
        "final_leakage",
        "leakage_conflict",
        "judge_leakage_coverage",
        "leakage_matched_rules",
        "leakage_hit_count",
    ]
    correctness_fields = [
        "dataset",
        "split",
        "sample_id",
        "method",
        "solver_correctness",
        "solver_correctness_coverage",
        "solver_correctness_reason",
        "tutor_math_correctness",
        "next_turn_feedback_quality",
        "student_state_correctness",
    ]
    write_csv(out / "routing_metrics.csv", _subset_rows(result["record_metrics"], routing_fields), routing_fields)
    write_csv(out / "state_metrics.csv", _subset_rows(result["record_metrics"], state_fields), state_fields)
    write_csv(out / "leakage_metrics.csv", _subset_rows(result["record_metrics"], leakage_fields), leakage_fields)
    write_csv(out / "correctness_metrics.csv", _subset_rows(result["record_metrics"], correctness_fields), correctness_fields)
    write_jsonl(out / "leakage_rule_hits.jsonl", _leakage_hits(deduped_generation_rows, gold_rows))
    write_json(out / "metric_coverage_report.json", result["coverage_report"])
    write_json(
        out / "metric_alignment_report.json",
        {
            "orphan_generations": result["orphan_generations"],
            "orphan_judges": result["orphan_judges"],
            "missing_samples": result["missing_samples"],
        },
    )
    print(json.dumps({"generation_rows": len(generation_rows), "aggregate_rows": len(result["aggregate_metrics"]), "output_dir": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
