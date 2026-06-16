from prism_tutor.eval.aggregate import compute_auto_metrics
from prism_tutor.eval.leakage_detector import detect_leakage


def test_compute_auto_metrics_with_gold_and_leakage():
    generations = [
        {
            "sample_id": "s1",
            "dataset": "mathdial",
            "split": "test",
            "method": "ours",
            "usage": {"prompt_tokens": 5, "completion_tokens": 7},
            "selected_agents": ["solver", "hint"],
            "rounds": [{"i": 1}],
            "latency": 1.5,
            "final_response": "The answer is 42.",
            "parsed_output": {"answer": "42", "misconceptions": ["sign"]},
            "parse_success": True,
            "state": {"events": [{"type": "commit", "correct": True}, {"type": "conflict"}]},
        }
    ]
    gold = [
        {
            "sample_id": "s1",
            "dataset": "mathdial",
            "answer": "42",
            "misconceptions": ["sign", "fraction"],
            "required_agents": ["solver", "hint", "verifier"],
        }
    ]

    result = compute_auto_metrics(generations, gold)
    row = result["record_metrics"][0]

    assert row["total_tokens"] == 12
    assert row["token_source"] == "usage.prompt_completion"
    assert row["internal_correctness"] == 1.0
    assert row["misconception_f1"] == 2 / 3
    assert row["routing_f1"] == 0.8
    assert row["state_conflict_rate"] == 0.5
    assert row["rule_leakage"] is True
    assert result["coverage_report"]["orphan_generation_count"] == 0


def test_leakage_detector_keeps_evidence_spans():
    result = detect_leakage(
        "First compute the equation. So the answer is x=3.",
        {"answer": "x=3"},
        sample_id="s2",
    )

    assert result["rule_leakage"] is True
    assert "final_answer_match" in result["matched_rules"]
    assert result["hits"][0]["sample_id"] == "s2"


def test_auto_metrics_use_unified_schema_gold_fields():
    generations = [
        {
            "sample_id": "m1",
            "dataset": "misconception",
            "split": "test",
            "method": "ours_full",
            "token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "source": "api"},
            "selected_agents": ["solver", "misconception", "verifier", "final_tutor"],
            "rounds": 1,
            "final_response": "The answer is 1/4.",
            "parse_success": True,
            "state": {
                "agent_outputs": {
                    "misconception": [
                        {"misconception_labels": ["students confuse numerator and denominator"]}
                    ]
                }
            },
        }
    ]
    gold = [
        {
            "sample_id": "m1",
            "dataset": "misconception",
            "problem_text": "What part is shaded?",
            "student_utterance": "1/3",
            "misconception_label": "students confuse numerator and denominator",
            "metadata": {"correct_answer": "1/4"},
        }
    ]

    row = compute_auto_metrics(generations, gold)["record_metrics"][0]

    assert row["internal_correctness_coverage"] == 1.0
    assert row["misconception_coverage"] == 1.0
    assert row["misconception_f1"] == 1.0
    assert row["routing_coverage"] == 1.0
    assert row["routing_recall"] > 0


def test_parse_failed_records_are_kept_but_structured_metrics_are_missing():
    generations = [
        {
            "sample_id": "s1",
            "dataset": "mathdial",
            "split": "test",
            "method": "ours",
            "token_usage": {"total_tokens": 11, "source": "api"},
            "selected_agents": ["solver", "final_tutor"],
            "rounds": 1,
            "final_response": "The answer is 42.",
            "parse_success": False,
            "state": {"events": [{"type": "conflict"}]},
        }
    ]
    gold = [
        {
            "sample_id": "s1",
            "dataset": "mathdial",
            "answer": "42",
            "misconceptions": ["sign"],
            "required_agents": ["solver", "hint", "final_tutor"],
        }
    ]

    result = compute_auto_metrics(generations, gold)
    row = result["record_metrics"][0]
    aggregate = result["aggregate_metrics"][0]

    assert result["coverage_report"]["generation_count"] == 1
    assert result["coverage_report"]["parse_failure_count"] == 1
    assert row["parse_success"] is False
    assert row["internal_correctness"] is None
    assert row["internal_correctness_reason"] == "parse_failed"
    assert row["misconception_f1"] is None
    assert row["misconception_reason"] == "parse_failed"
    assert row["routing_f1"] is not None
    assert row["state_metric_coverage"] == 1.0
    assert row["rule_leakage"] is True
    assert aggregate["n"] == 1
    assert aggregate["parse_success_rate"] == 0.0
    assert aggregate["internal_correctness_coverage"] == 0.0
    assert aggregate["misconception_coverage"] == 0.0


def test_generation_duplicates_prefer_latest_success_for_metrics():
    generations = [
        {
            "sample_id": "s1",
            "dataset": "mathdial",
            "split": "test",
            "method": "ours",
            "status": "failed",
            "parse_success": False,
            "final_response": "",
        },
        {
            "sample_id": "s1",
            "dataset": "mathdial",
            "split": "test",
            "method": "ours",
            "status": "success",
            "parse_success": True,
            "final_response": "The answer is 42.",
            "parsed_output": {"answer": "42"},
            "usage": {"prompt_tokens": 5, "completion_tokens": 7},
        },
        {
            "sample_id": "s1",
            "dataset": "mathdial",
            "split": "test",
            "method": "ours",
            "status": "success",
            "parse_success": True,
            "final_response": "The answer is 42.",
            "parsed_output": {"answer": "42"},
            "usage": {"prompt_tokens": 5, "completion_tokens": 8},
        },
    ]
    gold = [{"sample_id": "s1", "dataset": "mathdial", "answer": "42"}]

    result = compute_auto_metrics(generations, gold)
    row = result["record_metrics"][0]
    coverage = result["coverage_report"]

    assert len(result["record_metrics"]) == 1
    assert row["parse_success"] is True
    assert row["total_tokens"] == 13
    assert coverage["raw_generation_count"] == 3
    assert coverage["generation_count"] == 1
    assert coverage["duplicate_generation_count"] == 2
    assert coverage["duplicate_key_count"] == 1
    assert coverage["replaced_failed_with_success_count"] == 1
    assert coverage["repeated_success_duplicate_count"] == 1
    assert coverage["parse_failure_count"] == 0


def test_judge_leakage_merges_with_rule_leakage_for_final_label():
    generations = [
        {
            "sample_id": "s1",
            "dataset": "mathdial",
            "split": "test",
            "method": "ours",
            "token_usage": {"total_tokens": 5, "source": "api"},
            "selected_agents": ["final_tutor"],
            "rounds": 1,
            "final_response": "Try one more intermediate step.",
            "parse_success": True,
        }
    ]
    gold = [{"sample_id": "s1", "dataset": "mathdial", "answer": "42"}]
    judge = [
        {
            "sample_id": "s1",
            "dataset": "mathdial",
            "method": "ours",
            "parsed_score": {"answer_leakage": True},
        }
    ]

    result = compute_auto_metrics(generations, gold, judge)
    row = result["record_metrics"][0]
    aggregate = result["aggregate_metrics"][0]

    assert row["rule_leakage"] is False
    assert row["judge_leakage"] is True
    assert row["final_leakage"] is True
    assert row["leakage_conflict"] is True
    assert row["judge_leakage_coverage"] == 1.0
    assert result["coverage_report"]["judge_count"] == 1
    assert result["coverage_report"]["judge_matched_count"] == 1
    assert aggregate["rule_leakage_rate"] == 0.0
    assert aggregate["judge_leakage_rate"] == 1.0
    assert aggregate["final_leakage_rate"] == 1.0
    assert aggregate["leakage_conflict_rate"] == 1.0


def test_failed_judge_row_does_not_count_as_leakage_coverage():
    generations = [
        {
            "sample_id": "s1",
            "dataset": "mathdial",
            "split": "test",
            "method": "ours",
            "token_usage": {"total_tokens": 5, "source": "api"},
            "selected_agents": ["final_tutor"],
            "rounds": 1,
            "final_response": "Try one more intermediate step.",
            "parse_success": True,
        }
    ]
    gold = [{"sample_id": "s1", "dataset": "mathdial", "answer": "42"}]
    judge = [
        {
            "sample_id": "s1",
            "dataset": "mathdial",
            "method": "ours",
            "parsed_score": None,
            "error": "timeout",
        }
    ]

    result = compute_auto_metrics(generations, gold, judge)
    row = result["record_metrics"][0]
    aggregate = result["aggregate_metrics"][0]

    assert row["rule_leakage"] is False
    assert row["judge_leakage"] is None
    assert row["final_leakage"] is False
    assert row["leakage_conflict"] is None
    assert row["judge_leakage_coverage"] == 0.0
    assert row["judge_parse_success"] is False
    assert row["judge_error"] == "timeout"
    assert result["coverage_report"]["judge_count"] == 1
    assert result["coverage_report"]["judge_matched_count"] == 0
    assert result["coverage_report"]["judge_invalid_count"] == 1
    assert aggregate["judge_leakage_rate"] is None
    assert aggregate["final_leakage_rate"] == 0.0


def test_judge_duplicates_prefer_valid_recovery_row_for_metrics():
    generations = [
        {
            "sample_id": "s1",
            "dataset": "mathdial",
            "split": "test",
            "method": "ours",
            "token_usage": {"total_tokens": 5, "source": "api"},
            "selected_agents": ["final_tutor"],
            "rounds": 1,
            "final_response": "Try one more intermediate step.",
            "parse_success": True,
        }
    ]
    gold = [{"sample_id": "s1", "dataset": "mathdial", "answer": "42"}]
    judge = [
        {
            "sample_id": "s1",
            "dataset": "mathdial",
            "method": "ours",
            "parsed_score": None,
            "error": "timeout",
        },
        {
            "sample_id": "s1",
            "dataset": "mathdial",
            "method": "ours",
            "parsed_score": {"answer_leakage": True},
        },
    ]

    result = compute_auto_metrics(generations, gold, judge)
    row = result["record_metrics"][0]
    coverage = result["coverage_report"]

    assert row["judge_leakage"] is True
    assert row["judge_parse_success"] is True
    assert coverage["raw_judge_count"] == 2
    assert coverage["judge_count"] == 1
    assert coverage["duplicate_judge_count"] == 1
    assert coverage["duplicate_judge_key_count"] == 1
    assert coverage["replaced_invalid_with_valid_judge_count"] == 1
    assert coverage["judge_invalid_count"] == 0
    assert coverage["judge_matched_count"] == 1


def test_orphan_judge_rows_are_reported_when_method_does_not_match_generation():
    generations = [
        {
            "sample_id": "s1",
            "dataset": "mathdial",
            "split": "test",
            "method": "ours",
            "selected_agents": ["final_tutor"],
            "rounds": 1,
            "final_response": "Try one more intermediate step.",
            "parse_success": True,
        }
    ]
    gold = [{"sample_id": "s1", "dataset": "mathdial", "answer": "42"}]
    judge = [
        {
            "sample_id": "s1",
            "dataset": "mathdial",
            "method": "missing_method",
            "parsed_score": {"answer_leakage": True},
        }
    ]

    result = compute_auto_metrics(generations, gold, judge)

    assert result["coverage_report"]["orphan_judge_count"] == 1
    assert result["orphan_judges"] == [{"dataset": "mathdial", "sample_id": "s1", "method": "missing_method"}]
