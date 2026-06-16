from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "04_compute_metrics.py"
SPEC = importlib.util.spec_from_file_location("compute_metrics_script", SCRIPT_PATH)
compute_metrics = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(compute_metrics)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_compute_metrics_cli_writes_specialized_metric_outputs(tmp_path: Path) -> None:
    generations = tmp_path / "generations.jsonl"
    gold = tmp_path / "gold.jsonl"
    output = tmp_path / "metrics"
    _write_jsonl(
        generations,
        [
            {
                "sample_id": "s1",
                "dataset": "mathdial",
                "split": "test",
                "method": "ours_full",
                "token_usage": {"total_tokens": 10, "source": "api"},
                "selected_agents": ["solver", "hint", "final_tutor"],
                "rounds": 1,
                "latency": 0.5,
                "final_response": "The answer is 42.",
                "parse_success": True,
                "state": {"events": [{"type": "commit", "correct": True}, {"type": "conflict"}]},
            }
        ],
    )
    _write_jsonl(
        gold,
        [
            {
                "sample_id": "s1",
                "dataset": "mathdial",
                "answer": "42",
                "required_agents": ["solver", "hint", "verifier", "final_tutor"],
            }
        ],
    )

    rc = compute_metrics.main(["--generations", str(generations), "--gold", str(gold), "--output_dir", str(output)])

    assert rc == 0
    routing_rows = list(csv.DictReader((output / "routing_metrics.csv").open(encoding="utf-8")))
    state_rows = list(csv.DictReader((output / "state_metrics.csv").open(encoding="utf-8")))
    leakage_hits = [json.loads(line) for line in (output / "leakage_rule_hits.jsonl").read_text(encoding="utf-8").splitlines()]
    coverage = json.loads((output / "metric_coverage_report.json").read_text(encoding="utf-8"))

    assert routing_rows[0]["routing_f1"] != ""
    assert state_rows[0]["state_conflict_rate"] == "0.5"
    assert leakage_hits[0]["rule"] == "final_answer_match"
    assert leakage_hits[0]["evidence"] == "42"
    assert coverage["orphan_generation_count"] == 0


def test_compute_metrics_cli_merges_judge_leakage_outputs(tmp_path: Path) -> None:
    generations = tmp_path / "generations.jsonl"
    gold = tmp_path / "gold.jsonl"
    judge = tmp_path / "judge_scores.jsonl"
    output = tmp_path / "metrics"
    _write_jsonl(
        generations,
        [
            {
                "sample_id": "s1",
                "dataset": "mathdial",
                "split": "test",
                "method": "ours_full",
                "token_usage": {"total_tokens": 10, "source": "api"},
                "selected_agents": ["final_tutor"],
                "rounds": 1,
                "final_response": "Try again.",
                "parse_success": True,
            }
        ],
    )
    _write_jsonl(gold, [{"sample_id": "s1", "dataset": "mathdial", "answer": "42"}])
    _write_jsonl(
        judge,
        [
            {
                "sample_id": "s1",
                "dataset": "mathdial",
                "method": "ours_full",
                "parsed_score": {"answer_leakage": True},
            }
        ],
    )

    rc = compute_metrics.main(
        [
            "--generations",
            str(generations),
            "--gold",
            str(gold),
            "--judge-scores",
            str(judge),
            "--output_dir",
            str(output),
        ]
    )

    assert rc == 0
    record = json.loads((output / "record_auto_metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
    leakage_rows = list(csv.DictReader((output / "leakage_metrics.csv").open(encoding="utf-8")))
    coverage = json.loads((output / "metric_coverage_report.json").read_text(encoding="utf-8"))
    alignment = json.loads((output / "metric_alignment_report.json").read_text(encoding="utf-8"))

    assert record["rule_leakage"] is False
    assert record["judge_leakage"] is True
    assert record["final_leakage"] is True
    assert record["leakage_conflict"] is True
    assert leakage_rows[0]["final_leakage"] == "True"
    assert coverage["judge_count"] == 1
    assert coverage["judge_matched_count"] == 1
    assert coverage["orphan_judge_count"] == 0
    assert alignment["orphan_judges"] == []


def test_compute_metrics_cli_deduplicates_generation_recovery_rows(tmp_path: Path) -> None:
    generations = tmp_path / "generations.jsonl"
    gold = tmp_path / "gold.jsonl"
    output = tmp_path / "metrics"
    _write_jsonl(
        generations,
        [
            {
                "sample_id": "s1",
                "dataset": "mathdial",
                "split": "test",
                "method": "ours_full",
                "status": "failed",
                "final_response": "The answer is 42.",
                "parse_success": False,
            },
            {
                "sample_id": "s1",
                "dataset": "mathdial",
                "split": "test",
                "method": "ours_full",
                "status": "success",
                "final_response": "Try one more step.",
                "parse_success": True,
            },
        ],
    )
    _write_jsonl(gold, [{"sample_id": "s1", "dataset": "mathdial", "answer": "42"}])

    rc = compute_metrics.main(["--generations", str(generations), "--gold", str(gold), "--output_dir", str(output)])

    assert rc == 0
    records = [json.loads(line) for line in (output / "record_auto_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    leakage_hits = (output / "leakage_rule_hits.jsonl").read_text(encoding="utf-8").splitlines()
    coverage = json.loads((output / "metric_coverage_report.json").read_text(encoding="utf-8"))
    assert len(records) == 1
    assert records[0]["parse_success"] is True
    assert leakage_hits == []
    assert coverage["raw_generation_count"] == 2
    assert coverage["duplicate_generation_count"] == 1
