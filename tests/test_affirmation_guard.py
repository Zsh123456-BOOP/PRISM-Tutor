"""Tests for the runtime false-affirmation guard + eval over-validation metric."""

from __future__ import annotations

from prism_tutor.agents.base_client import BaseLLMClient, LLMClientConfig
from prism_tutor.agents.parser import parse_agent_json
from prism_tutor.agents.schemas import AGENT_SCHEMAS
from prism_tutor.agents.types import LLMCallRecord, LLMUsage
from prism_tutor.eval.affirmation_metrics import detect_over_validation
from prism_tutor.runtime.affirmation_guard import (
    detect_false_affirmation,
    student_answer_disagrees_with_solver,
)
from prism_tutor.runtime.prism_graph import PrismGraphConfig, build_prism_graph


# --- detectors ---------------------------------------------------------------


def test_detect_false_affirmation_flags_bare_affirmation():
    assert detect_false_affirmation("Yes, that's correct! Well done.") is True
    assert detect_false_affirmation("Your answer is absolutely correct.") is True


def test_detect_false_affirmation_ignores_corrective_response():
    # affirmation token present but a correction cue is also present -> not a false affirmation
    assert detect_false_affirmation("You're right that 12 is involved, but let's re-check the next step.") is False
    assert detect_false_affirmation("What do you get when you add the two amounts?") is False


def test_student_answer_disagrees_with_solver():
    assert student_answer_disagrees_with_solver("I think it is 5.", "4") is True
    assert student_answer_disagrees_with_solver("It is 4.", "4") is False
    assert student_answer_disagrees_with_solver("not sure", "4") is False  # no number -> no signal


def test_eval_over_validation_on_wrong_student():
    gold = {"metadata": {"final_answer": "4", "student_incorrect_solution": "2+2=5"}}
    ov = detect_over_validation("Yes, that's correct, nice work!", gold=gold)
    assert ov["student_wrong"] is True
    assert ov["over_validation"] is True
    # a corrective response on the same wrong student is NOT over-validation
    ov2 = detect_over_validation("Not quite — let's re-check 2 + 2 together.", gold=gold)
    assert ov2["over_validation"] is False


# --- graph wiring ------------------------------------------------------------


class _AffirmThenCorrectClient:
    """misconception flags an error; the first final response affirms, the rewrite corrects."""

    def __init__(self) -> None:
        self.final_calls = 0

    def call(self, *, sample_id, method, agent_name, messages, schema):  # noqa: ANN001
        if agent_name == "solver":
            raw = '{"answer":"4","reasoning":["2+2=4"],"confidence":0.9,"uncertainty":0.1,"needs_more_info":false}'
        elif agent_name == "misconception":
            raw = '{"misconception_detected":true,"misconception_labels":["addition_error"],"evidence":["2+2 is not 5"],"severity":"high","confidence":0.9}'
        elif agent_name == "final_tutor":
            self.final_calls += 1
            if self.final_calls == 1:
                raw = '{"response":"Yes, that is correct! Great job.","withheld_answer":true,"confidence":0.8,"safety_notes":[]}'
            else:
                raw = '{"response":"Let us look again — what is 2 + 2?","withheld_answer":true,"confidence":0.8,"safety_notes":["regenerated"]}'
        else:
            raw = BaseLLMClient(LLMClientConfig(mock_mode=True))._mock_completion(agent_name, AGENT_SCHEMAS.get(agent_name))
        parsed = parse_agent_json(raw, schema)
        return LLMCallRecord(
            sample_id=sample_id,
            method=method,
            agent_name=agent_name,
            endpoint="mock://affirm",
            prompt=messages,
            raw_completion=raw,
            stripped_output=parsed.stripped_output,
            parsed_output=parsed.parsed_output,
            usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2, source="mock"),
            parse_success=parsed.parse_success,
        )


_SAMPLE = {"sample_id": "af-1", "problem_text": "What is 2 + 2?", "student_utterance": "I think it is 5."}


def test_affirmation_guard_fires_and_regenerates():
    client = _AffirmThenCorrectClient()
    graph = build_prism_graph("M1", client=client)
    result = graph.invoke({"sample": dict(_SAMPLE), "method": "M1"})
    final_calls = [c for c in result.llm_calls if c["agent_name"] == "final_tutor"]
    assert len(final_calls) == 2, "guard must regenerate the affirming response"
    assert result.agent_outputs.get("affirmation_guard")
    assert "what is 2 + 2" in final_calls[-1]["parsed_output"]["response"].lower()


def test_affirmation_guard_is_ablatable():
    client = _AffirmThenCorrectClient()
    graph = build_prism_graph("M1", client=client, config=PrismGraphConfig(disabled_modules=["affirmation_guard"]))
    result = graph.invoke({"sample": dict(_SAMPLE), "method": "M1"})
    final_calls = [c for c in result.llm_calls if c["agent_name"] == "final_tutor"]
    assert len(final_calls) == 1
    assert not result.agent_outputs.get("affirmation_guard")
