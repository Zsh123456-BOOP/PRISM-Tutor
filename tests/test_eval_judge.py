import os
import json

import pytest

from prism_tutor.eval.judge_client import JudgeClientConfig, make_judge_client
from prism_tutor.eval.judge_merge import merge_leakage
from prism_tutor.eval.judge_schema import parse_score_json


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
