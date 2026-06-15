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
