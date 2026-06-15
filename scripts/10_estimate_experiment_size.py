#!/usr/bin/env python
"""Estimate full experiment size from split files and optional smoke logs."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prism_tutor.utils.config import load_yaml


def _read_jsonl_dir(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    for file in files:
        with file.open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def _split_counts(split_dir: Path) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for path in sorted(split_dir.glob("*_*.jsonl")):
        dataset, split = path.stem.rsplit("_", 1)
        with path.open("r", encoding="utf-8") as handle:
            counts[(dataset, split)] = sum(1 for line in handle if line.strip())
    return counts


def _method_averages(smoke_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in smoke_rows:
        grouped.setdefault(str(row.get("method")), []).append(row)
    averages: dict[str, dict[str, float]] = {}
    for method, rows in grouped.items():
        averages[method] = {
            "agent_calls": statistics.mean(float(row.get("agent_calls") or 0) for row in rows),
            "tokens": statistics.mean(float(row.get("token_usage", {}).get("total_tokens") or 0) for row in rows),
        }
    return averages


def estimate(
    *,
    experiments_config: str,
    split_dir: str,
    smoke_generations: str | None,
) -> dict[str, Any]:
    experiments = load_yaml(experiments_config)["experiments"]
    counts = _split_counts(Path(split_dir))
    averages = _method_averages(_read_jsonl_dir(Path(smoke_generations))) if smoke_generations else {}

    experiment_rows = []
    total_records = 0
    total_agent_calls = 0.0
    total_tokens = 0.0
    for name, spec in experiments.items():
        split = spec.get("split", "test")
        records = sum(counts.get((dataset, split), 0) for dataset in spec["datasets"]) * len(spec["methods"])
        agent_calls = 0.0
        tokens = 0.0
        for dataset in spec["datasets"]:
            n = counts.get((dataset, split), 0)
            for method in spec["methods"]:
                avg = averages.get(method, {"agent_calls": 0.0, "tokens": 0.0})
                agent_calls += n * avg["agent_calls"]
                tokens += n * avg["tokens"]
        total_records += records
        total_agent_calls += agent_calls
        total_tokens += tokens
        experiment_rows.append(
            {
                "experiment": name,
                "datasets": spec["datasets"],
                "method_count": len(spec["methods"]),
                "records": records,
                "estimated_agent_calls": round(agent_calls),
                "estimated_tokens": round(tokens),
            }
        )

    return {
        "split_counts": {f"{dataset}/{split}": count for (dataset, split), count in sorted(counts.items())},
        "smoke_generations": smoke_generations,
        "experiments": experiment_rows,
        "total_records": total_records,
        "estimated_agent_calls": round(total_agent_calls),
        "estimated_tokens": round(total_tokens),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-config", default="configs/experiments.yaml")
    parser.add_argument("--split-dir", default="data/splits")
    parser.add_argument("--smoke-generations", default=None)
    parser.add_argument("--output", default="outputs/logs/full_experiment_estimate.json")
    args = parser.parse_args()

    result = estimate(
        experiments_config=args.experiments_config,
        split_dir=args.split_dir,
        smoke_generations=args.smoke_generations,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
