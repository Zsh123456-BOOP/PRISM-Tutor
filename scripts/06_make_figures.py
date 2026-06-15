#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prism_tutor.eval.figure_builder import (
    FigureInputError,
    agent_call_distribution,
    require_pareto_points,
    risk_bucket_counts,
    state_conflict_case_lines,
    write_text_pdf,
)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build lightweight paper figure PDFs.")
    parser.add_argument("--record_metrics", default="outputs/metrics/record_auto_metrics.jsonl")
    parser.add_argument("--output_dir", default="outputs/figures")
    args = parser.parse_args(argv)

    rows = _read_jsonl(Path(args.record_metrics))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        pareto = require_pareto_points(rows, "internal_correctness", "total_tokens")
        buckets = risk_bucket_counts(rows)
        calls = agent_call_distribution(rows)
        state_conflicts = state_conflict_case_lines(rows)
    except FigureInputError as exc:
        print(json.dumps({"status": "failed", "reason": str(exc), "record_metrics": args.record_metrics}, indent=2), file=sys.stderr)
        return 2

    write_text_pdf(
        out / "figure1_system_overview.pdf",
        "Figure 1: PRISM-Tutor System Overview",
        ["Risk estimator -> router -> budget -> state commit."],
    )
    write_text_pdf(
        out / "figure2_quality_token_pareto.pdf",
        "Figure 2: Quality-token Pareto",
        [json.dumps(point, sort_keys=True) for point in pareto[:30]],
    )
    write_text_pdf(
        out / "figure3_risk_bucket_analysis.pdf",
        "Figure 3: Risk Bucket Analysis",
        [json.dumps(buckets, sort_keys=True)],
    )
    write_text_pdf(
        out / "figure4_agent_call_distribution.pdf",
        "Figure 4: Agent Call Distribution",
        [json.dumps(calls, sort_keys=True)],
    )
    write_text_pdf(
        out / "figure5_state_conflict_case_study.pdf",
        "Figure 5: State Conflict Case Study",
        state_conflicts,
    )
    manifest = {
        "status": "completed",
        "rows": len(rows),
        "record_metrics": args.record_metrics,
        "output_dir": str(out),
        "figures": {
            "figure1_system_overview.pdf": {"type": "system_overview"},
            "figure2_quality_token_pareto.pdf": {"type": "pareto", "points": len(pareto)},
            "figure3_risk_bucket_analysis.pdf": {"type": "risk_buckets", "buckets": buckets},
            "figure4_agent_call_distribution.pdf": {"type": "agent_calls", "distribution": calls},
            "figure5_state_conflict_case_study.pdf": {"type": "state_conflicts", "lines": len(state_conflicts)},
        },
    }
    (out / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
