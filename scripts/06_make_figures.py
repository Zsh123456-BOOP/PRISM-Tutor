#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prism_tutor.eval.figure_builder import pareto_points, risk_bucket_counts, write_text_pdf


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build lightweight paper figure PDFs.")
    parser.add_argument("--record_metrics", default="outputs/metrics/record_auto_metrics.jsonl")
    parser.add_argument("--output_dir", default="outputs/figures")
    args = parser.parse_args()

    rows = _read_jsonl(Path(args.record_metrics))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pareto = pareto_points(rows, "internal_correctness", "total_tokens")
    buckets = risk_bucket_counts(rows)
    write_text_pdf(out / "figure1_system_overview.pdf", "Figure 1: PRISM-Tutor System Overview", ["Risk estimator -> router -> budget -> state commit."])
    write_text_pdf(out / "figure2_quality_token_pareto.pdf", "Figure 2: Quality-token Pareto", [json.dumps(point) for point in pareto[:30]])
    write_text_pdf(out / "figure3_risk_bucket_analysis.pdf", "Figure 3: Risk Bucket Analysis", [json.dumps(buckets, sort_keys=True)])
    write_text_pdf(out / "figure4_agent_call_distribution.pdf", "Figure 4: Agent Call Distribution", [f"rows={len(rows)}"])
    write_text_pdf(out / "figure5_state_conflict_case_study.pdf", "Figure 5: State Conflict Case Study", ["Populate after full experiment."])
    print(json.dumps({"rows": len(rows), "output_dir": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
