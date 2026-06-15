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

from prism_tutor.eval.io import read_jsonl, write_json, write_jsonl
from prism_tutor.eval.judge_client import JudgeClientConfig, make_judge_client
from prism_tutor.utils.config import load_yaml


def _input_files(path: str | Path) -> list[Path]:
    p = Path(path)
    if p.is_dir():
        return sorted(p.glob("*.jsonl"))
    return [p]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run mock-safe LLM judge over generation logs.")
    parser.add_argument("--input", default="outputs/generations")
    parser.add_argument("--judge_config", default="configs/judge.yaml")
    parser.add_argument("--output_dir", default="outputs/judge_scores")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    cfg = load_yaml(args.judge_config) if Path(args.judge_config).exists() else {}
    dry_run = bool(cfg.get("dry_run", True))
    provider = "mock" if dry_run else str(cfg.get("provider", "deepseek"))
    client = make_judge_client(
        JudgeClientConfig(
            provider=provider,
            requested_model=str(cfg.get("requested_model", "deepseek-v4-pro")),
            temperature=float(cfg.get("temperature", 0.0)),
            top_p=float(cfg.get("top_p", 1.0)),
            max_tokens=int(cfg.get("max_tokens", 768)),
        )
    )

    rows = []
    for file in _input_files(args.input):
        rows.extend(read_jsonl(file))
    if args.limit is not None:
        rows = rows[: args.limit]

    judged = []
    for row in rows:
        case = {
            **row,
            "candidate_response": row.get("final_response"),
            "ground_truth": row.get("ground_truth") or row.get("answer"),
        }
        judged.append(client.judge(case))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "judge_scores.jsonl"
    raw_output = out_dir / "raw" / "judge_raw.jsonl"
    write_jsonl(output, judged)
    write_jsonl(raw_output, judged)
    metadata = judged[0]["metadata"] if judged else {"dry_run": dry_run, "requested_model": cfg.get("requested_model")}
    write_json(out_dir / "judge_metadata.json", metadata)

    print(json.dumps({"input_rows": len(rows), "output": str(output), "raw_output": str(raw_output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
