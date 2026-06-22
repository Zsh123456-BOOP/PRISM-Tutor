import json
import importlib.util
from pathlib import Path

import pytest

from prism_tutor.eval.judge_client import JudgeClientConfig, make_judge_client
from prism_tutor.eval.judge_merge import merge_leakage
from prism_tutor.eval.judge_prompts import PROMPT_VERSION, build_judge_prompt
from prism_tutor.eval.judge_schema import parse_score_json


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "03_run_judge.py"
SPEC = importlib.util.spec_from_file_location("run_judge_script", SCRIPT_PATH)
run_judge = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(run_judge)


def test_mock_judge_is_default_and_records_metadata(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("PRISM_TUTOR_ENABLE_REAL_JUDGE", raising=False)
    client = make_judge_client()

    result = client.judge(
        {
            "sample_id": "s1",
            "dataset": "mathdial",
            "method": "ours",
            "problem": "1+1",
            "candidate_response": "The answer is 2.",
            "ground_truth": "2",
        }
    )

    assert result["metadata"]["dry_run"] is True
    assert result["metadata"]["actual_model"] == "mock-judge"
    assert result["parsed_score"]["answer_leakage"] is True
    assert "api_key" not in str(result).lower()


def test_real_judge_requires_explicit_env(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("PRISM_TUTOR_ENABLE_REAL_JUDGE", raising=False)

    with pytest.raises(RuntimeError):
        make_judge_client(JudgeClientConfig(provider="deepseek"))


def test_judge_prompt_v2_anchors_math_to_gold_context():
    prompt = build_judge_prompt(
        {
            "problem": "A box has 10 spoons. How many spoons are there?",
            "ground_truth": "10",
            "candidate_response": "Set up x + 5 = 12, so x = 7.",
        }
    )

    assert PROMPT_VERSION == "judge-v2"
    assert "gold_context as the authoritative reference" in prompt
    assert "do not invent a different equation" in prompt
    assert "disagrees with gold_context" in prompt


def test_judge_prompt_v2_defines_answer_leakage_for_full_solution_paths():
    prompt = build_judge_prompt(
        {
            "problem": "Solve x + 5 = 12.",
            "ground_truth": "7",
            "candidate_response": "Subtract 5 from both sides; x = 7.",
        }
    )

    assert "answer_leakage=true" in prompt
    assert "provides enough complete calculation steps" in prompt
    assert "answer_leakage=false only when" in prompt


def test_run_judge_require_real_rejects_dry_run_config(tmp_path):
    config = tmp_path / "judge.yaml"
    config.write_text(
        "\n".join(
            [
                "provider: deepseek",
                "requested_model: deepseek-v4-pro",
                "dry_run: true",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="Formal judge requires"):
        run_judge.main(["--judge_config", str(config), "--require-real", "--input", str(tmp_path / "missing.jsonl")])


def test_parse_score_json_and_merge_leakage():
    parsed = parse_score_json(
        '{"mathematical_correctness": 4, "pedagogical_quality": 3, '
        '"scaffolding_quality": 2, "misconception_coverage": 1, '
        '"answer_leakage": false, "clarity": 5, '
        '"student_facing_appropriateness": 4, "overall": 3, "reason": "ok"}'
    )
    merged = merge_leakage(
        {"sample_id": "s1", "rule_leakage": True, "matched_rules": ["direct_answer_phrase"]},
        {"sample_id": "s1", "parsed_score": parsed.to_dict()},
    )

    assert merged["judge_leakage"] is False
    assert merged["final_leakage"] is True
    assert merged["leakage_conflict"] is True


def test_judge_schema_parses_boolean_strings_without_truthiness_bug():
    parsed = parse_score_json(
        '{"mathematical_correctness": 4, "pedagogical_quality": 3, '
        '"scaffolding_quality": 2, "misconception_coverage": 1, '
        '"answer_leakage": "false", "clarity": 5, '
        '"student_facing_appropriateness": 4, "overall": 3, "reason": "ok"}'
    )

    assert parsed.answer_leakage is False


def test_judge_schema_rejects_invalid_bool_and_bool_numeric_scores():
    with pytest.raises(ValueError, match="answer_leakage must be boolean"):
        parse_score_json(
            '{"mathematical_correctness": 4, "pedagogical_quality": 3, '
            '"scaffolding_quality": 2, "misconception_coverage": 1, '
            '"answer_leakage": "maybe", "clarity": 5, '
            '"student_facing_appropriateness": 4, "overall": 3, "reason": "ok"}'
        )

    with pytest.raises(ValueError, match="mathematical_correctness must be numeric"):
        parse_score_json(
            '{"mathematical_correctness": true, "pedagogical_quality": 3, '
            '"scaffolding_quality": 2, "misconception_coverage": 1, '
            '"answer_leakage": false, "clarity": 5, '
            '"student_facing_appropriateness": 4, "overall": 3, "reason": "ok"}'
        )


def test_real_judge_request_uses_json_response_format(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("PRISM_TUTOR_ENABLE_REAL_JUDGE", "1")
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps(
                {
                    "model": "deepseek-v4-pro",
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"mathematical_correctness": 4, "pedagogical_quality": 4, '
                                    '"scaffolding_quality": 4, "misconception_coverage": 4, '
                                    '"answer_leakage": false, "clarity": 4, '
                                    '"student_facing_appropriateness": 4, "overall": 4, "reason": "ok"}'
                                )
                            }
                        }
                    ],
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["authorization"] = request.headers.get("Authorization")
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = make_judge_client(JudgeClientConfig(provider="deepseek", retries=0, response_format_json=True))
    result = client.judge({"sample_id": "s1", "candidate_response": "Try again."})

    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert captured["body"]["thinking"] == {"type": "disabled"}
    assert result["parsed_score"]["overall"] == 4.0
    assert result["raw_attempts"][0]["error"] is None
    assert "sk-test" not in str(result)


def test_real_judge_retries_with_repair_instruction(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("PRISM_TUTOR_ENABLE_REAL_JUDGE", "1")
    bodies = []
    responses = [
        {"model": "deepseek-v4-pro", "choices": [{"message": {"content": '{"overall": 4,'}, "finish_reason": "length"}]},
        {
            "model": "deepseek-v4-pro",
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"mathematical_correctness": 4, "pedagogical_quality": 4, '
                            '"scaffolding_quality": 4, "misconception_coverage": 4, '
                            '"answer_leakage": false, "clarity": 4, '
                            '"student_facing_appropriateness": 4, "overall": 4, "reason": "ok"}'
                        )
                    },
                    "finish_reason": "stop",
                }
            ],
        },
    ]

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps(self.payload).encode("utf-8")

    def fake_urlopen(request, timeout):
        bodies.append(json.loads(request.data.decode("utf-8")))
        return FakeResponse(responses.pop(0))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = make_judge_client(JudgeClientConfig(provider="deepseek", retries=1))
    result = client.judge({"sample_id": "s1", "candidate_response": "Try again."})

    assert result["parsed_score"]["overall"] == 4.0
    assert len(result["raw_attempts"]) == 2
    assert result["raw_attempts"][0]["error"] is not None
    assert "previous response was invalid" in bodies[1]["messages"][-1]["content"].lower()


def test_judge_candidate_rows_have_stable_display_order():
    row = {
        "sample_id": "s1",
        "dataset": "mathdial",
        "method": "bundle",
        "candidate_responses": [
            {"method": "a", "final_response": "A"},
            {"method": "b", "final_response": "B"},
            {"method": "c", "final_response": "C"},
        ],
    }

    first = run_judge._candidate_rows(row, seed=42)
    second = run_judge._candidate_rows(row, seed=42)
    third = run_judge._candidate_rows(row, seed=43)

    assert [item["candidate_label"] for item in first] == [item["candidate_label"] for item in second]
    assert first[0]["display_order"] == second[0]["display_order"]
    assert first[0]["display_order_seed"] == second[0]["display_order_seed"]
    assert first[0]["display_order_seed"] != third[0]["display_order_seed"]
    assert sorted(first[0]["display_order"]) == ["a", "b", "c"]


def test_judge_candidate_rows_assign_unique_labels_when_missing_or_duplicate():
    duplicate_row = {
        "sample_id": "s1",
        "dataset": "mathdial",
        "method": "bundle",
        "candidate_responses": [
            {"method": "same", "final_response": "A"},
            {"method": "same", "final_response": "B"},
        ],
    }
    missing_row = {
        "sample_id": "s2",
        "dataset": "mathdial",
        "candidate_responses": [
            {"final_response": "A"},
            {"final_response": "B"},
        ],
    }

    duplicate_candidates = run_judge._candidate_rows(duplicate_row, seed=42)
    missing_candidates = run_judge._candidate_rows(missing_row, seed=42)

    assert sorted(item["candidate_label"] for item in duplicate_candidates) == ["same", "same_2"]
    assert sorted(duplicate_candidates[0]["display_order"]) == ["same", "same_2"]
    assert sorted(item["candidate_label"] for item in missing_candidates) == ["candidate_1", "candidate_2"]
    assert {item["method"] for item in missing_candidates} == {"candidate_1", "candidate_2"}


def test_run_judge_writes_display_order_fields(tmp_path):
    input_path = tmp_path / "generations.jsonl"
    output_dir = tmp_path / "judge"
    input_path.write_text(
        json.dumps(
            {
                "sample_id": "s1",
                "dataset": "mathdial",
                "method": "bundle",
                "candidate_responses": [
                    {"method": "a", "final_response": "The answer is 2."},
                    {"method": "b", "final_response": "Try again."},
                ],
                "state": {"sample": {"question": "1+1", "metadata": {"correct_answer": "2"}}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rc = run_judge.main(["--input", str(input_path), "--output_dir", str(output_dir), "--seed", "7"])

    rows = [json.loads(line) for line in (output_dir / "judge_scores.jsonl").read_text(encoding="utf-8").splitlines()]
    metadata = json.loads((output_dir / "judge_metadata.json").read_text(encoding="utf-8"))
    assert rc == 0
    assert len(rows) == 2
    assert rows[0]["display_order"] == rows[1]["display_order"]
    assert rows[0]["display_order_seed"] == rows[1]["display_order_seed"]
    assert {row["candidate_label"] for row in rows} == {"a", "b"}
    assert all(row["raw_response"] for row in rows)
    assert metadata["input_rows"] == 1
    assert metadata["output_rows"] == 2
    assert metadata["parsed_count"] == 2
    assert metadata["error_count"] == 0
    assert metadata["raw_response_count"] == 2
    assert metadata["actual_models"] == ["mock-judge"]


def test_run_judge_deduplicates_generation_recovery_rows(tmp_path):
    input_path = tmp_path / "generations.jsonl"
    output_dir = tmp_path / "judge"
    rows = [
        {
            "sample_id": "s1",
            "dataset": "mathdial",
            "split": "test",
            "method": "ours_full",
            "status": "failed",
            "final_response": "",
            "state": {"sample": {"question": "1+1", "metadata": {"correct_answer": "2"}}},
        },
        {
            "sample_id": "s1",
            "dataset": "mathdial",
            "split": "test",
            "method": "ours_full",
            "status": "success",
            "final_response": "Try adding one and one.",
            "state": {"sample": {"question": "1+1", "metadata": {"correct_answer": "2"}}},
        },
    ]
    input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    rc = run_judge.main(["--input", str(input_path), "--output_dir", str(output_dir)])

    judged = [json.loads(line) for line in (output_dir / "judge_scores.jsonl").read_text(encoding="utf-8").splitlines()]
    metadata = json.loads((output_dir / "judge_metadata.json").read_text(encoding="utf-8"))
    assert rc == 0
    assert len(judged) == 1
    assert metadata["input_rows"] == 1
    assert metadata["generation_deduplication"]["raw_generation_count"] == 2
    assert metadata["generation_deduplication"]["duplicate_generation_count"] == 1
    assert metadata["generation_deduplication"]["replaced_failed_with_success_count"] == 1


def test_run_judge_resume_skips_existing_rows(tmp_path):
    input_path = tmp_path / "generations.jsonl"
    output_dir = tmp_path / "judge"
    rows = [
        {
            "sample_id": "s1",
            "dataset": "mathdial",
            "split": "test",
            "method": "single_tutor",
            "final_response": "Try adding one and one.",
            "state": {"sample": {"question": "1+1", "metadata": {"correct_answer": "2"}}},
        },
        {
            "sample_id": "s2",
            "dataset": "mathdial",
            "split": "test",
            "method": "single_tutor",
            "final_response": "Try adding two and two.",
            "state": {"sample": {"question": "2+2", "metadata": {"correct_answer": "4"}}},
        },
    ]
    input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    output_dir.mkdir()
    existing = {
        "sample_id": "s1",
        "dataset": "mathdial",
        "method": "single_tutor",
        "candidate_label": "single_tutor",
        "raw_response": "{}",
        "parsed_score": {"overall": 3},
        "error": None,
        "metadata": {"actual_model": "mock-judge", "dry_run": True},
    }
    (output_dir / "judge_scores.jsonl").write_text(json.dumps(existing) + "\n", encoding="utf-8")

    rc = run_judge.main(["--input", str(input_path), "--output_dir", str(output_dir), "--resume"])

    judged = [json.loads(line) for line in (output_dir / "judge_scores.jsonl").read_text(encoding="utf-8").splitlines()]
    metadata = json.loads((output_dir / "judge_metadata.json").read_text(encoding="utf-8"))
    assert rc == 0
    assert len(judged) == 2
    assert [row["sample_id"] for row in judged] == ["s1", "s2"]
    assert metadata["existing_output_rows"] == 1
    assert metadata["skipped_existing_rows"] == 1
    assert metadata["new_output_rows"] == 1


def test_run_judge_shards_and_skips_resume_source(tmp_path):
    input_path = tmp_path / "generations.jsonl"
    rows = [
        {
            "sample_id": f"s{i}",
            "dataset": "mathdial",
            "split": "test",
            "method": "single_tutor",
            "final_response": f"Try case {i}.",
        }
        for i in range(4)
    ]
    input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    resume_dir = tmp_path / "existing"
    resume_dir.mkdir()
    existing = {
        "sample_id": "s1",
        "dataset": "mathdial",
        "method": "single_tutor",
        "candidate_label": "single_tutor",
        "raw_response": "{}",
        "parsed_score": {"overall": 3},
        "error": None,
        "metadata": {"actual_model": "mock-judge", "dry_run": True},
    }
    (resume_dir / "judge_scores.jsonl").write_text(json.dumps(existing) + "\n", encoding="utf-8")
    output_dir = tmp_path / "judge_shard"

    rc = run_judge.main(
        [
            "--input",
            str(input_path),
            "--output_dir",
            str(output_dir),
            "--resume",
            "--resume-from",
            str(resume_dir),
            "--num-shards",
            "2",
            "--shard-index",
            "1",
        ]
    )

    judged = [json.loads(line) for line in (output_dir / "judge_scores.jsonl").read_text(encoding="utf-8").splitlines()]
    metadata = json.loads((output_dir / "judge_metadata.json").read_text(encoding="utf-8"))
    assert rc == 0
    assert [row["sample_id"] for row in judged] == ["s3"]
    assert metadata["unsharded_input_rows"] == 4
    assert metadata["input_rows"] == 2
    assert metadata["existing_output_rows"] == 1
    assert metadata["skipped_existing_rows"] == 1
    assert metadata["new_output_rows"] == 1


def test_run_judge_metadata_summarizes_multiple_actual_models():
    judged = [
        {
            "metadata": {"actual_model": "judge-a", "requested_model": "deepseek-v4-pro"},
            "parsed_score": {"overall": 4},
            "raw_response": "{}",
            "error": None,
        },
        {
            "metadata": {"actual_model": "judge-b", "requested_model": "deepseek-v4-pro"},
            "parsed_score": None,
            "raw_response": "",
            "error": "timeout",
        },
    ]

    metadata = run_judge._run_metadata(judged, dry_run=False, requested_model="deepseek-v4-pro")

    assert metadata["actual_model"] == "judge-a,judge-b"
    assert metadata["actual_models"] == ["judge-a", "judge-b"]
    assert metadata["output_rows"] == 2
    assert metadata["parsed_count"] == 1
    assert metadata["error_count"] == 1
    assert metadata["raw_response_count"] == 1
