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

from prism_tutor.eval.human_agreement import (
    build_agreement_report,
    formal_gate_failures,
    resolve_pairwise_preferences,
)
from prism_tutor.eval.io import write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute human audit agreement metrics.")
    parser.add_argument("--input", default="outputs/human_audit/human_audit_labeled.csv")
    parser.add_argument("--output", default="outputs/human_audit/human_agreement_report.json")
    parser.add_argument("--allow-unlabeled", action="store_true")
    parser.add_argument(
        "--preference-mapping",
        default=None,
        help="Path to preference_mapping.json. Defaults to the input directory when present.",
    )
    parser.add_argument("--min-quality-pairs", type=int, default=2)
    parser.add_argument("--min-leakage-pairs", type=int, default=1)
    parser.add_argument("--min-preferences", type=int, default=1)
    args = parser.parse_args(argv)

    path = Path(args.input)
    if not path.exists():
        if not args.allow_unlabeled:
            raise SystemExit(f"Missing labeled audit file: {path}")
        fallback = path.with_name("human_audit_blind.csv")
        path = fallback
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    mapping_path = Path(args.preference_mapping) if args.preference_mapping else path.with_name("preference_mapping.json")
    mapping_rows = _read_mapping_rows(mapping_path) if mapping_path.exists() else []
    rows, mapping_report = resolve_pairwise_preferences(rows, mapping_rows)
    if mapping_report["ab_label_count"] and not mapping_path.exists():
        mapping_report["missing_mapping_file"] = str(mapping_path)
        mapping_report["unresolved_count"] = mapping_report["ab_label_count"] - mapping_report["tie_count"]
        mapping_report["unresolved_reason"] = "missing_preference_mapping_file"
    report = build_agreement_report(rows)
    report["preference_mapping"] = mapping_report
    failures = formal_gate_failures(
        report,
        min_quality_pairs=args.min_quality_pairs,
        min_leakage_pairs=args.min_leakage_pairs,
        min_preferences=args.min_preferences,
    )
    report["formal_gate"] = {
        "status": "failed" if failures else "passed",
        "failures": failures,
        "min_quality_pairs": args.min_quality_pairs,
        "min_leakage_pairs": args.min_leakage_pairs,
        "min_preferences": args.min_preferences,
    }
    report["status"] = report["formal_gate"]["status"]
    write_json(args.output, report)
    print(json.dumps({"rows": len(rows), "output": args.output}, indent=2))
    if failures and not args.allow_unlabeled:
        return 1
    return 0


def _read_mapping_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict) and isinstance(data.get("rows"), list):
        return [row for row in data["rows"] if isinstance(row, dict)]
    return []


if __name__ == "__main__":
    raise SystemExit(main())
