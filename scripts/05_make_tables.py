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
    "rule_leakage",
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Build paper tables and significance summaries.")
    parser.add_argument("--record_metrics", default="outputs/metrics/record_auto_metrics.jsonl")
    parser.add_argument("--output_dir", default="outputs/tables")
    args = parser.parse_args()

    record_path = Path(args.record_metrics)
    rows = _read_jsonl(record_path) if record_path.suffix == ".jsonl" else _read_csv(record_path)
    rows = _coerce_numeric(rows)
    metrics = [metric for metric in DEFAULT_METRICS if any(metric in row for row in rows)]
    table = summarize_table(rows, metrics, ["dataset", "method"]) if rows else []
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in table for key in row}) if table else ["dataset", "method", "n"]
    write_csv(out / "table1_main_results.csv", table, fields)
    (out / "table1_main_results.tex").write_text(rows_to_latex(table, fields, "Main end-to-end results"), encoding="utf-8")

    comparisons = []
    for metric in metrics:
        for a, b in [
            ("ours_full", "fixed_4"),
            ("ours_full", "generic_sparse"),
            ("ours_full", "debate"),
            ("ours_routing_budget", "difficulty_routing"),
        ]:
            comparisons.append(compare_methods(rows, metric, a, b, binary=metric == "rule_leakage"))
    write_json(record_path.parent / "significance_tests.json", holm_correction(comparisons))
    print(json.dumps({"rows": len(rows), "table_rows": len(table), "output_dir": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
