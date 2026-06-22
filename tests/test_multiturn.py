"""Tests for the multi-turn MathDial dialogue eval builder."""

from __future__ import annotations

from prism_tutor.data.multiturn import build_records_from_raw, parse_conversation
from prism_tutor.data.sample_view import assert_no_gold_fields, build_model_input, extract_evaluation_gold
from prism_tutor.runtime.graph_state import TutorGraphState
from prism_tutor.runtime.qos_router import QoSRouter
from prism_tutor.runtime.risk_estimator import estimate_risk


_RAW = {
    "qid": 42,
    "question": "Julia bought spoons; how many were in her package?",
    "ground_truth": "12+3=15 spoons.\nThen 15-5=10.\n 10",
    "student_incorrect_solution": "I think it is 9 because 12-3=9.",
    "student_profile": "7th grader; confuses relevant info.",
    "conversation": (
        "Teacher: (generic)Hi, please talk me through your solution|EOM|"
        "Student: I subtracted 3 from 12 to get 9.|EOM|"
        "Teacher: (focus)Which quantities should you add first?|EOM|"
        "Student: Oh, 12 and 3?|EOM|"
        "Teacher: (telling)Right, 12+3=15, then 15-5=10.|EOM|"
        "Student: I see, thanks!|EOM|"
        "Teacher: (generic)Great work!"
    ),
}


def test_parse_conversation_extracts_roles_and_moves():
    turns = parse_conversation(_RAW["conversation"])
    assert [t["role"] for t in turns] == ["teacher", "student", "teacher", "student", "teacher", "student", "teacher"]
    assert turns[0]["move"] == "generic"
    assert turns[2]["move"] == "focus"
    assert turns[4]["move"] == "telling"
    assert "(generic)" not in turns[0]["text"]  # move tag stripped from text


def test_build_records_one_per_teacher_turn():
    records = list(build_records_from_raw(_RAW, split="test"))
    assert len(records) == 4  # four teacher turns
    moves = [r["metadata"]["gold_teacher_move"] for r in records]
    assert moves == ["generic", "focus", "telling", "generic"]
    types = [r["metadata"]["gold_turn_type"] for r in records]
    assert types == ["low_stakes", "diagnosis", "telling", "low_stakes"]


def test_first_turn_uses_initial_student_solution_and_grows_history():
    records = list(build_records_from_raw(_RAW, split="test"))
    # first teacher turn (greeting): no prior student turn -> initial incorrect solution
    assert "12-3=9" in records[0]["student_utterance"]
    assert records[0]["dialogue_history"] == ""
    # a later turn carries the dialogue history and the latest student utterance
    assert "Oh, 12 and 3?" in records[2]["student_utterance"]
    assert "Teacher:" in records[2]["dialogue_history"]


def test_gold_is_isolated_from_model_input():
    record = list(build_records_from_raw(_RAW, split="test"))[2]
    model_input = build_model_input(record)
    assert_no_gold_fields(model_input)  # must not raise
    assert "problem_text" in model_input
    assert "dialogue_history" in model_input
    assert "student_utterance" in model_input
    # gold never reaches the model
    assert "ground_truth" not in str(model_input)
    assert "teacher_response" not in model_input.get("metadata", {})
    # but evaluation can still read it
    gold = extract_evaluation_gold(record)
    assert gold["metadata"]["final_answer"] == "10"
    assert gold["metadata"]["gold_teacher_move"] == "telling"


def test_low_stakes_turn_routes_minimal_agents():
    state = TutorGraphState(
        sample={"sample_id": "mt-low", "problem_text": "Julia bought spoons.", "student_utterance": "I see, thanks!"},
        method="M1",
    )
    risk = estimate_risk(state)
    selected = QoSRouter().select_agents(risk)
    assert risk.misconception_risk <= 0.2
    assert "solver" not in selected and "misconception" not in selected
    assert "final_tutor" in selected


def test_error_turn_still_routes_full_diagnosis():
    state = TutorGraphState(
        sample={
            "sample_id": "mt-err",
            "problem_text": "Julia bought spoons.",
            "student_utterance": "12 - 3 = 9, so she had 9 spoons left.",
        },
        method="M1",
    )
    risk = estimate_risk(state)
    selected = QoSRouter().select_agents(risk)
    assert risk.misconception_risk >= 0.45
    assert {"solver", "misconception"}.issubset(set(selected))
