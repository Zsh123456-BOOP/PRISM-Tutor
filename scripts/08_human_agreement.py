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

from prism_tutor.eval.human_agreement import build_agreement_report, formal_gate_failures
from prism_tutor.eval.io import write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute human audit agreement metrics.")
    parser.add_argument("--input", default="outputs/human_audit/human_audit_labeled.csv")
    parser.add_argument("--output", default="outputs/human_audit/human_agreement_report.json")
    parser.add_argument("--allow-unlabeled", action="store_true")
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
    report = build_agreement_report(rows)
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


if __name__ == "__main__":
    raise SystemExit(main())
