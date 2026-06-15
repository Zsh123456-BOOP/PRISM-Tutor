#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prism_tutor.eval.human_audit_sampler import sample_human_audit
from prism_tutor.eval.io import read_jsonl, write_csv, write_json


def _jsonl_files(path: str | Path) -> list[Path]:
    p = Path(path)
    if not p.exists():
        return []
    if p.is_dir():
        return sorted(p.glob("*.jsonl"))
    return [p]


def _read_jsonl_many(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    for file in _jsonl_files(path):
        rows.extend(read_jsonl(file))
    return rows


def _artifact_exists(path: str | Path) -> bool:
    p = Path(path)
    if p.is_dir():
        return any(p.iterdir())
    return p.exists()


def _sample_from_generation(row: dict) -> dict:
    state = row.get("state") if isinstance(row.get("state"), dict) else {}
    sample = state.get("sample") if isinstance(state.get("sample"), dict) else {}
    return sample


def _row_key(row: dict) -> tuple[str, str, str]:
    return (str(row.get("dataset", "")), str(row.get("sample_id", "")), str(row.get("method", "")))


def _generation_context(row: dict) -> dict:
    sample = _sample_from_generation(row)
    return {
        "problem": row.get("problem") or row.get("problem_text") or sample.get("problem_text") or sample.get("question"),
        "student_answer": row.get("student_answer") or sample.get("student_utterance") or sample.get("student_answer"),
        "ground_truth": row.get("ground_truth") or row.get("answer") or sample.get("ground_truth") or sample.get("answer"),
        "dialogue_context": row.get("dialogue_context") or sample.get("dialogue_context") or sample.get("dialogue"),
        "candidate_response": row.get("candidate_response") or row.get("final_response") or row.get("response"),
    }


def _merge_records_with_generations(record_rows: list[dict], generation_rows: list[dict]) -> list[dict]:
    generation_index = {_row_key(row): row for row in generation_rows}
    merged = []
    for record in record_rows:
        row = dict(record)
        generation = generation_index.get(_row_key(record))
        if generation:
            for key, value in _generation_context(generation).items():
                if row.get(key) in (None, "") and value not in (None, ""):
                    row[key] = value
        merged.append(row)
    return merged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create blind human-audit sample after experiments.")
    parser.add_argument("--records", default="outputs/metrics/record_auto_metrics.jsonl")
    parser.add_argument("--generations", default="outputs/generations")
    parser.add_argument("--judge-scores", default="outputs/judge_scores/judge_scores.jsonl")
    parser.add_argument("--tables", default="outputs/tables")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--output_dir", default="outputs/human_audit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(argv)

    prerequisites = {
        "record_auto_metrics": _artifact_exists(args.records),
        "generation_logs": bool(_jsonl_files(args.generations)),
        "judge_scores": _artifact_exists(args.judge_scores),
        "tables": _artifact_exists(args.tables),
    }
    if not args.allow_incomplete:
        missing = [name for name, exists in prerequisites.items() if not exists]
        if missing:
            raise SystemExit(
                "Human audit sampling requires completed artifacts before formal sampling: "
                + ", ".join(missing)
                + ". Pass --allow-incomplete only for smoke checks."
            )

    gen_rows = _read_jsonl_many(args.generations)
    record_rows = _read_jsonl_many(args.records)
    audit_rows = _merge_records_with_generations(record_rows, gen_rows) if record_rows else gen_rows
    if not audit_rows and not args.allow_incomplete:
        raise SystemExit("No audit rows found; run experiments and metrics first or pass --allow-incomplete for smoke.")
    result = sample_human_audit(audit_rows, target_n=args.n, seed=args.seed)
    result["manifest"]["prerequisites"] = prerequisites
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "human_audit_blind.csv", result["blind_rows"], list(result["blind_rows"][0]) if result["blind_rows"] else ["audit_id"])
    write_json(out / "preference_mapping.json", result.get("preference_mapping", []))
    write_json(out / "sampling_manifest.json", result["manifest"])
    print(json.dumps({"actual_n": result["manifest"]["actual_n"], "output_dir": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
