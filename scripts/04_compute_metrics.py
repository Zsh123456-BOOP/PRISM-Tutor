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
from prism_tutor.eval.io import read_jsonl, write_csv, write_json, write_jsonl


def _files(path: str | Path) -> list[Path]:
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute automatic metrics from generation JSONL logs.")
    parser.add_argument("--generations", default="outputs/generations")
    parser.add_argument("--gold", default="data/splits")
    parser.add_argument("--output_dir", default="outputs/metrics")
    args = parser.parse_args()

    generation_rows = _read_many(_files(args.generations))
    gold_rows = _read_many(_files(args.gold))
    result = compute_auto_metrics(generation_rows, gold_rows)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    record_fields = sorted({key for row in result["record_metrics"] for key in row})
    aggregate_fields = sorted({key for row in result["aggregate_metrics"] for key in row})
    write_jsonl(out / "record_auto_metrics.jsonl", result["record_metrics"])
    write_csv(out / "main_auto_metrics.csv", result["aggregate_metrics"], aggregate_fields)
    write_json(out / "metric_coverage_report.json", result["coverage_report"])
    write_json(out / "metric_alignment_report.json", {"orphan_generations": result["orphan_generations"], "missing_samples": result["missing_samples"]})
    print(json.dumps({"generation_rows": len(generation_rows), "aggregate_rows": len(result["aggregate_metrics"]), "output_dir": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
