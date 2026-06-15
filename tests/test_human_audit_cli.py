from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "07_sample_human_audit.py"
SPEC = importlib.util.spec_from_file_location("sample_human_audit_script", SCRIPT_PATH)
sample_human_audit_script = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(sample_human_audit_script)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_human_audit_cli_refuses_missing_completed_artifacts(tmp_path: Path) -> None:
    records = tmp_path / "metrics" / "record_auto_metrics.jsonl"
    generations = tmp_path / "generations" / "rows.jsonl"
    _write_jsonl(records, [{"dataset": "mathdial", "sample_id": "s1", "method": "ours"}])
    _write_jsonl(generations, [{"dataset": "mathdial", "sample_id": "s1", "method": "ours", "final_response": "hint"}])

    with pytest.raises(SystemExit) as exc:
        sample_human_audit_script.main(
            [
                "--records",
                str(records),
                "--generations",
                str(generations.parent),
                "--judge-scores",
                str(tmp_path / "judge_scores" / "judge_scores.jsonl"),
                "--tables",
                str(tmp_path / "tables"),
                "--output_dir",
                str(tmp_path / "audit"),
            ]
        )

    assert "judge_scores" in str(exc.value)
    assert "table_manifest" in str(exc.value)
    assert "table1_main_results.csv" in str(exc.value)


def test_human_audit_cli_smoke_merges_records_with_generation_context(tmp_path: Path) -> None:
    records = tmp_path / "metrics" / "record_auto_metrics.jsonl"
    generations = tmp_path / "generations" / "rows.jsonl"
    output_dir = tmp_path / "audit"
    _write_jsonl(
        records,
        [
            {"dataset": "mathdial", "sample_id": "s1", "method": "ours_full", "risk_bucket": "high", "rule_leakage": True},
            {"dataset": "mathdial", "sample_id": "s1", "method": "fixed_4", "risk_bucket": "high", "rule_leakage": False},
        ],
    )
    _write_jsonl(
        generations,
        [
            {
                "dataset": "mathdial",
                "sample_id": "s1",
                "method": "ours_full",
                "final_response": "Use a smaller hint.",
                "state": {"sample": {"problem_text": "2+2?", "student_utterance": "5", "ground_truth": "4"}},
            },
            {
                "dataset": "mathdial",
                "sample_id": "s1",
                "method": "fixed_4",
                "final_response": "The answer is 4.",
                "state": {"sample": {"problem_text": "2+2?", "student_utterance": "5", "ground_truth": "4"}},
            },
        ],
    )

    rc = sample_human_audit_script.main(
        [
            "--records",
            str(records),
            "--generations",
            str(generations.parent),
            "--output_dir",
            str(output_dir),
            "--judge-scores",
            str(tmp_path / "judge_scores" / "judge_scores.jsonl"),
            "--tables",
            str(tmp_path / "tables"),
            "--n",
            "1",
            "--allow-incomplete",
        ]
    )

    rows = list(csv.DictReader((output_dir / "human_audit_blind.csv").open(encoding="utf-8")))
    manifest = json.loads((output_dir / "sampling_manifest.json").read_text(encoding="utf-8"))
    mapping = json.loads((output_dir / "preference_mapping.json").read_text(encoding="utf-8"))
    assert rc == 0
    assert rows[0]["candidate_response"] in {"Use a smaller hint.", "The answer is 4."}
    assert {rows[0]["candidate_a_response"], rows[0]["candidate_b_response"]} == {"Use a smaller hint.", "The answer is 4."}
    assert rows[0]["problem"] == "2+2?"
    assert "risk_bucket" not in rows[0]
    assert "candidate_a_method" not in rows[0]
    assert {mapping[0]["candidate_a_method"], mapping[0]["candidate_b_method"]} == {"ours_full", "fixed_4"}
    assert manifest["prerequisites"]["record_auto_metrics"] is True
    assert manifest["prerequisites"]["judge_scores"] is False
    assert manifest["prerequisites"]["table_manifest"] is False
    assert manifest["pairwise_preference_rows"] == 1


def _write_formal_table_artifacts(tables: Path) -> None:
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "table_manifest.json").write_text(json.dumps({"status": "passed"}), encoding="utf-8")
    for stem in [
        "table1_main_results",
        "table2_routing",
        "table3_budget",
        "table4_state_commit",
        "table5_ablation",
        "table6_robustness",
    ]:
        (tables / f"{stem}.csv").write_text("dataset,method,n\nmathdial,ours,1\n", encoding="utf-8")
        (tables / f"{stem}.tex").write_text("\\begin{table}\\end{table}\n", encoding="utf-8")


def test_human_audit_formal_refuses_rows_missing_annotation_context(tmp_path: Path) -> None:
    records = tmp_path / "metrics" / "record_auto_metrics.jsonl"
    generations = tmp_path / "generations" / "rows.jsonl"
    judge_scores = tmp_path / "judge_scores" / "judge_scores.jsonl"
    tables = tmp_path / "tables"
    _write_jsonl(records, [{"dataset": "mathdial", "sample_id": "s1", "method": "ours_full"}])
    _write_jsonl(generations, [{"dataset": "mathdial", "sample_id": "s1", "method": "ours_full", "final_response": ""}])
    _write_jsonl(judge_scores, [{"dataset": "mathdial", "sample_id": "s1", "method": "ours_full", "parsed_score": {"overall": 4}}])
    _write_formal_table_artifacts(tables)

    with pytest.raises(SystemExit) as exc:
        sample_human_audit_script.main(
            [
                "--records",
                str(records),
                "--generations",
                str(generations.parent),
                "--judge-scores",
                str(judge_scores),
                "--tables",
                str(tables),
                "--n",
                "1",
                "--output_dir",
                str(tmp_path / "audit"),
            ]
        )

    assert "missing required annotation context" in str(exc.value)


def test_human_audit_formal_requires_pairwise_preference_rows(tmp_path: Path) -> None:
    records = tmp_path / "metrics" / "record_auto_metrics.jsonl"
    generations = tmp_path / "generations" / "rows.jsonl"
    judge_scores = tmp_path / "judge_scores" / "judge_scores.jsonl"
    tables = tmp_path / "tables"
    _write_jsonl(records, [{"dataset": "mathdial", "sample_id": "s1", "method": "ours_full"}])
    _write_jsonl(
        generations,
        [
            {
                "dataset": "mathdial",
                "sample_id": "s1",
                "method": "ours_full",
                "final_response": "Try a hint.",
                "state": {"sample": {"problem_text": "2+2?", "student_utterance": "5"}},
            }
        ],
    )
    _write_jsonl(judge_scores, [{"dataset": "mathdial", "sample_id": "s1", "method": "ours_full", "parsed_score": {"overall": 4}}])
    _write_formal_table_artifacts(tables)

    with pytest.raises(SystemExit) as exc:
        sample_human_audit_script.main(
            [
                "--records",
                str(records),
                "--generations",
                str(generations.parent),
                "--judge-scores",
                str(judge_scores),
                "--tables",
                str(tables),
                "--n",
                "1",
                "--output_dir",
                str(tmp_path / "audit"),
            ]
        )

    assert "pairwise A/B preference" in str(exc.value)
