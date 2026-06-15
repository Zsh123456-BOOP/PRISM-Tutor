#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
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


def _sample_from_row(row: dict) -> dict:
    state = row.get("state") if isinstance(row.get("state"), dict) else {}
    sample = state.get("sample") if isinstance(state.get("sample"), dict) else {}
    return sample


def _metadata_from_sample(sample: dict) -> dict:
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    return metadata


def _stable_case_seed(row: dict, seed: int) -> int:
    material = json.dumps(
        {
            "seed": seed,
            "sample_id": row.get("sample_id"),
            "dataset": row.get("dataset"),
            "method": row.get("method"),
        },
        sort_keys=True,
    )
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16)


def _candidate_rows(row: dict, *, seed: int) -> list[dict]:
    raw_candidates = row.get("candidate_responses")
    if isinstance(raw_candidates, list) and raw_candidates:
        candidates = [
            {
                "candidate_label": str(candidate.get("label") or candidate.get("method") or f"candidate_{index + 1}"),
                "method": candidate.get("method") or row.get("method"),
                "candidate_response": candidate.get("candidate_response") or candidate.get("final_response") or candidate.get("response"),
            }
            for index, candidate in enumerate(raw_candidates)
            if isinstance(candidate, dict)
        ]
    else:
        candidates = [
            {
                "candidate_label": str(row.get("method") or "candidate_1"),
                "method": row.get("method"),
                "candidate_response": row.get("final_response"),
            }
        ]
    case_seed = _stable_case_seed(row, seed)
    rng = random.Random(case_seed)
    order = list(range(len(candidates)))
    rng.shuffle(order)
    display_order = [candidates[index]["candidate_label"] for index in order]
    rows = []
    for position, index in enumerate(order, 1):
        candidate = candidates[index]
        rows.append(
            {
                **row,
                "method": candidate["method"],
                "candidate_label": candidate["candidate_label"],
                "candidate_response": candidate["candidate_response"],
                "display_position": position,
                "display_order": display_order,
                "display_order_seed": case_seed,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run mock-safe LLM judge over generation logs.")
    parser.add_argument("--input", default="outputs/generations")
    parser.add_argument("--judge_config", default="configs/judge.yaml")
    parser.add_argument("--output_dir", default="outputs/judge_scores")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

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
            retries=int(cfg.get("retries", 1)),
            response_format_json=bool(cfg.get("response_format_json", False)),
            thinking_type=cfg.get("thinking_type", "disabled"),
        )
    )

    rows = []
    for file in _input_files(args.input):
        rows.extend(read_jsonl(file))
    if args.limit is not None:
        rows = rows[: args.limit]

    judged = []
    for row in rows:
        for candidate_row in _candidate_rows(row, seed=args.seed):
            sample = _sample_from_row(candidate_row)
            metadata = _metadata_from_sample(sample)
            case = {
                **candidate_row,
                "problem": candidate_row.get("problem") or candidate_row.get("problem_text") or sample.get("problem_text") or sample.get("question"),
                "student_answer": candidate_row.get("student_answer") or sample.get("student_utterance"),
                "ground_truth": (
                    candidate_row.get("ground_truth")
                    or candidate_row.get("answer")
                    or sample.get("ground_truth")
                    or metadata.get("ground_truth")
                    or metadata.get("correct_answer")
                ),
                "gold_context": candidate_row.get("gold_context") or sample.get("misconception_label") or sample.get("student_error"),
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
