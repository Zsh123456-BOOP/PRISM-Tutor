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

from prism_tutor.eval.io import read_jsonl, write_json
from prism_tutor.eval.generation_records import deduplicate_generation_rows
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
        candidates = []
        used_labels: dict[str, int] = {}
        for index, candidate in enumerate(raw_candidates):
            if not isinstance(candidate, dict):
                continue
            base_label = str(candidate.get("label") or candidate.get("method") or f"candidate_{index + 1}")
            label = _unique_candidate_label(base_label, used_labels)
            candidates.append(
                {
                    "candidate_label": label,
                    "method": candidate.get("method") or row.get("method") or label,
                    "candidate_response": candidate.get("candidate_response") or candidate.get("final_response") or candidate.get("response"),
                }
            )
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


def _unique_candidate_label(base_label: str, used_labels: dict[str, int]) -> str:
    count = used_labels.get(base_label, 0) + 1
    used_labels[base_label] = count
    return base_label if count == 1 else f"{base_label}_{count}"


def _judge_key(row: dict) -> tuple[str, str, str, str]:
    return (
        str(row.get("dataset", "")),
        str(row.get("sample_id", "")),
        str(row.get("method", "")),
        str(row.get("candidate_label") or row.get("method") or ""),
    )


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _run_metadata(judged: list[dict], *, dry_run: bool, requested_model: str | None) -> dict:
    if not judged:
        return {
            "dry_run": dry_run,
            "requested_model": requested_model,
            "actual_model": None,
            "actual_models": [],
            "input_rows": 0,
            "output_rows": 0,
            "parsed_count": 0,
            "error_count": 0,
        }
    metadata_rows = [row.get("metadata") for row in judged if isinstance(row.get("metadata"), dict)]
    base = dict(metadata_rows[0]) if metadata_rows else {"dry_run": dry_run, "requested_model": requested_model}
    actual_models = sorted({str(meta.get("actual_model")) for meta in metadata_rows if meta.get("actual_model")})
    base.update(
        {
            "actual_model": actual_models[0] if len(actual_models) == 1 else ",".join(actual_models),
            "actual_models": actual_models,
            "output_rows": len(judged),
            "parsed_count": sum(isinstance(row.get("parsed_score"), dict) for row in judged),
            "error_count": sum(bool(row.get("error")) for row in judged),
            "raw_response_count": sum(bool(row.get("raw_response")) for row in judged),
        }
    )
    return base


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run mock-safe LLM judge over generation logs.")
    parser.add_argument("--input", default="outputs/generations")
    parser.add_argument("--judge_config", default="configs/judge.yaml")
    parser.add_argument("--output_dir", default="outputs/judge_scores")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Append missing judge rows and skip existing outputs.")
    parser.add_argument(
        "--require-real",
        action="store_true",
        help="Fail if judge_config would use the mock/dry-run judge. Use for formal full-run finalization.",
    )
    args = parser.parse_args(argv)

    cfg = load_yaml(args.judge_config) if Path(args.judge_config).exists() else {}
    dry_run = bool(cfg.get("dry_run", True))
    provider = "mock" if dry_run else str(cfg.get("provider", "deepseek"))
    if args.require_real and (dry_run or provider == "mock"):
        raise SystemExit("Formal judge requires dry_run=false and a non-mock provider")
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
    rows, deduplication_report = deduplicate_generation_rows(rows)
    if args.limit is not None:
        rows = rows[: args.limit]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "judge_scores.jsonl"
    raw_output = out_dir / "raw" / "judge_raw.jsonl"
    existing_judged = read_jsonl(output) if args.resume and output.exists() else []
    completed_keys = {_judge_key(row) for row in existing_judged}
    if not args.resume:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("", encoding="utf-8")
        raw_output.parent.mkdir(parents=True, exist_ok=True)
        raw_output.write_text("", encoding="utf-8")

    judged = list(existing_judged)
    skipped_existing = 0
    for row in rows:
        for candidate_row in _candidate_rows(row, seed=args.seed):
            judge_key = _judge_key(candidate_row)
            if judge_key in completed_keys:
                skipped_existing += 1
                continue
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
            judged_row = client.judge(case)
            judged.append(judged_row)
            completed_keys.add(judge_key)
            _append_jsonl(output, judged_row)
            _append_jsonl(raw_output, judged_row)
    metadata = _run_metadata(judged, dry_run=dry_run, requested_model=cfg.get("requested_model"))
    metadata["input_rows"] = len(rows)
    metadata["generation_deduplication"] = deduplication_report
    metadata["resume"] = bool(args.resume)
    metadata["existing_output_rows"] = len(existing_judged)
    metadata["skipped_existing_rows"] = skipped_existing
    metadata["new_output_rows"] = len(judged) - len(existing_judged)
    write_json(out_dir / "judge_metadata.json", metadata)

    print(json.dumps({"input_rows": len(rows), "output": str(output), "raw_output": str(raw_output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
