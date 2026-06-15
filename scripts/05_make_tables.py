#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prism_tutor.eval.io import write_csv, write_json
from prism_tutor.eval.significance import compare_methods, holm_correction
from prism_tutor.eval.table_builder import rows_to_latex, summarize_table


DEFAULT_METRICS = [
    "total_tokens",
    "agent_calls",
    "rounds",
    "parse_success",
    "internal_correctness",
    "misconception_f1",
    "routing_f1",
    "state_conflict_rate",
    "final_leakage",
    "judge_leakage",
    "leakage_conflict",
    "rule_leakage",
]

TABLE_SPECS = [
    {
        "stem": "table1_main_results",
        "caption": "Main end-to-end results",
        "methods": None,
    },
    {
        "stem": "table2_routing",
        "caption": "Routing experiment results",
        "methods": {"random_routing", "fixed_all_agents", "difficulty_routing", "generic_sparse", "oracle_routing", "ours_routing"},
    },
    {
        "stem": "table3_budget",
        "caption": "Budgeted deliberation results",
        "methods": {"single_round", "fixed_2_rounds", "fixed_3_rounds", "fixed_4_rounds", "debate", "generic_early_stopping", "ours_routing_budget"},
    },
    {
        "stem": "table4_state_commit",
        "caption": "Student state commit results",
        "methods": {"no_memory", "naive_shared_memory", "single_writer", "two_phase_commit", "ours_full"},
    },
    {
        "stem": "table5_ablation",
        "caption": "PRISM-Tutor ablation results",
        "methods": {
            "ours_full",
            "ablate_risk_estimator",
            "ablate_qos_routing",
            "ablate_budget_controller",
            "ablate_leakage_risk",
            "ablate_misconception_risk",
            "ablate_state_conflict_risk",
            "ablate_state_commit",
            "ablate_confidence_weighted_commit",
            "replace_pedagogical_risk_with_difficulty",
            "replace_two_phase_commit_with_naive_memory",
        },
    },
    {
        "stem": "table6_robustness",
        "caption": "Robustness experiment results",
        "methods": {"fixed_4", "debate", "generic_sparse", "ours_full"},
    },
]


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _coerce_numeric(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        item = dict(row)
        for key, value in list(item.items()):
            if value in ("", "None", None):
                item[key] = None
                continue
            try:
                item[key] = float(value)
            except (TypeError, ValueError):
                pass
        out.append(item)
    return out


def _filter_rows(rows: list[dict], methods: set[str] | None) -> list[dict]:
    if methods is None:
        return rows
    return [row for row in rows if _method_matches(str(row.get("method")), methods)]


def _method_matches(method: str, methods: set[str]) -> bool:
    return method in methods or any(method.startswith(f"{allowed}__") for allowed in methods)


def _write_table(out: Path, stem: str, caption: str, rows: list[dict], metrics: list[str]) -> dict[str, int | str]:
    table = summarize_table(rows, metrics, ["dataset", "method"]) if rows else []
    fields = sorted({key for row in table for key in row}) if table else ["dataset", "method", "n"]
    write_csv(out / f"{stem}.csv", table, fields)
    (out / f"{stem}.tex").write_text(rows_to_latex(table, fields, caption), encoding="utf-8")
    return {"stem": stem, "input_rows": len(rows), "table_rows": len(table)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build paper tables and significance summaries.")
    parser.add_argument("--record_metrics", default="outputs/metrics/record_auto_metrics.jsonl")
    parser.add_argument("--output_dir", default="outputs/tables")
    args = parser.parse_args(argv)

    record_path = Path(args.record_metrics)
    rows = _read_jsonl(record_path) if record_path.suffix == ".jsonl" else _read_csv(record_path)
    rows = _coerce_numeric(rows)
    metrics = [metric for metric in DEFAULT_METRICS if any(metric in row for row in rows)]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    table_outputs = [
        _write_table(out, spec["stem"], spec["caption"], _filter_rows(rows, spec["methods"]), metrics)
        for spec in TABLE_SPECS
    ]

    comparisons = []
    for metric in metrics:
        for a, b in [
            ("ours_full", "fixed_4"),
            ("ours_full", "generic_sparse"),
            ("ours_full", "debate"),
            ("ours_routing_budget", "difficulty_routing"),
        ]:
            comparisons.append(compare_methods(rows, metric, a, b, binary=metric.endswith("leakage") or metric == "leakage_conflict"))
    write_json(record_path.parent / "significance_tests.json", holm_correction(comparisons))
    print(json.dumps({"rows": len(rows), "tables": table_outputs, "output_dir": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
