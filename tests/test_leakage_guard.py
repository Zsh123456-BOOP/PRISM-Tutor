"""Tests for the risk-gated leakage guard (runtime) and the eval telling rule.

The runtime guard is gold-free: it uses the solver's own computed answer to
detect when the student-facing response reveals it, and becomes stricter under
high leakage risk. It is a PRISM-only module (baselines never run it) and is
ablatable via disabled_modules=["leakage_guard"].
"""

from __future__ import annotations

from prism_tutor.agents.base_client import BaseLLMClient, LLMClientConfig
from prism_tutor.agents.parser import parse_agent_json
from prism_tutor.agents.schemas import AGENT_SCHEMAS
from prism_tutor.agents.types import LLMCallRecord, LLMUsage
from prism_tutor.eval.leakage_detector import detect_leakage
from prism_tutor.runtime.leakage_guard import detect_runtime_leakage
from prism_tutor.runtime.prism_graph import PrismGraphConfig, build_prism_graph


# --- runtime detector (gold-free) --------------------------------------------


def test_runtime_detector_flags_revealed_reference_answer():
    # The solver's OWN answer (not gold) is 42; an asserted "is 42" reveals it.
    result = detect_runtime_leakage("Great, the total is 42, nice work.", reference_answer="42")
    assert result["rule_leakage"] is True
    assert "runtime_reference_answer_revealed" in result["matched_rules"]


def test_runtime_detector_passes_guiding_question():
    result = detect_runtime_leakage(
        "What do you get when you combine the two amounts?",
        reference_answer="42",
        leakage_risk=0.2,
    )
    assert result["rule_leakage"] is False


def test_runtime_detector_flags_bare_reference_answer():
    # Bare statement of the solver's computed answer reveals it even without an
    # explicit assertion phrase, regardless of leakage risk.
    result = detect_runtime_leakage("The package contained 42 spoons in total.", reference_answer="42", leakage_risk=0.1)
    assert result["rule_leakage"] is True
    assert "runtime_reference_answer_present" in result["matched_rules"]


def test_runtime_detector_excludes_student_echo():
    # Echoing the student's own number is not leakage.
    result = detect_runtime_leakage("You said 42, but let's re-check that step together.", reference_answer="42")
    assert result["rule_leakage"] is False


def test_runtime_detector_noref_assertion_is_risk_gated():
    # With no reference answer, a generic asserted result only fires under high risk.
    text = "Therefore we get 28 at this stage."
    low = detect_runtime_leakage(text, reference_answer=None, leakage_risk=0.2)
    high = detect_runtime_leakage(text, reference_answer=None, leakage_risk=0.9)
    assert low["rule_leakage"] is False
    assert high["rule_leakage"] is True


def test_runtime_detector_keeps_direct_answer_phrase_rule():
    result = detect_runtime_leakage("The answer is whatever you computed.")
    assert result["rule_leakage"] is True
    assert "runtime_direct_answer_phrase" in result["matched_rules"]


# --- eval telling rule (gold-using) ------------------------------------------


def test_eval_detector_flags_telling_final_answer():
    gold = {"answer": "Buy spoons step by step ... = 12 spoons.\n 12"}
    told = detect_leakage("So the total is 12 spoons, well done.", gold=gold)
    assert told["rule_leakage"] is True
    assert "telling_final_answer" in told["matched_rules"]


def test_eval_detector_ignores_guiding_question():
    gold = {"answer": "... = 12 spoons.\n 12"}
    guiding = detect_leakage("What happens if you add the two groups together?", gold=gold)
    assert "telling_final_answer" not in guiding["matched_rules"]


# --- graph wiring: guard fires for M1 and is ablatable -----------------------


class _AnswerTellingClient:
    """Solver computes 42; the first final response tells it, the rewrite hides it."""

    def __init__(self) -> None:
        self.final_calls = 0

    def call(self, *, sample_id, method, agent_name, messages, schema):  # noqa: ANN001
        if agent_name == "solver":
            raw = '{"answer":"42","reasoning":["40+2=42"],"confidence":0.9,"uncertainty":0.1,"needs_more_info":false}'
        elif agent_name == "final_tutor":
            self.final_calls += 1
            if self.final_calls == 1:
                raw = '{"response":"Great, the total is 42, nice work.","withheld_answer":false,"confidence":0.8,"safety_notes":[]}'
            else:
                raw = '{"response":"What do you get when you combine the two amounts?","withheld_answer":true,"confidence":0.8,"safety_notes":["regenerated"]}'
        else:
            raw = BaseLLMClient(LLMClientConfig(mock_mode=True))._mock_completion(agent_name, AGENT_SCHEMAS.get(agent_name))
        parsed = parse_agent_json(raw, schema)
        return LLMCallRecord(
            sample_id=sample_id,
            method=method,
            agent_name=agent_name,
            endpoint="mock://answer-telling",
            prompt=messages,
            raw_completion=raw,
            stripped_output=parsed.stripped_output,
            parsed_output=parsed.parsed_output,
            usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2, source="mock"),
            parse_success=parsed.parse_success,
        )


_TELLING_SAMPLE = {
    "sample_id": "m1-leak",
    "problem": "What is 40 + 2?",
    "student_utterance": "I think it is 50.",
    "metadata": {"ground_truth": "42"},
}


def test_leakage_guard_fires_for_m1_using_solver_answer():
    client = _AnswerTellingClient()
    graph = build_prism_graph("M1", client=client)
    result = graph.invoke({"sample": dict(_TELLING_SAMPLE), "method": "M1"})

    final_calls = [call for call in result.llm_calls if call["agent_name"] == "final_tutor"]
    assert len(final_calls) == 2, "M1 must run the leakage guard and regenerate"
    assert result.agent_outputs.get("leakage_guard")
    assert final_calls[-1]["parsed_output"]["response"].endswith("?")


def test_leakage_guard_is_ablatable():
    client = _AnswerTellingClient()
    graph = build_prism_graph(
        "M1", client=client, config=PrismGraphConfig(disabled_modules=["leakage_guard"])
    )
    result = graph.invoke({"sample": dict(_TELLING_SAMPLE), "method": "M1"})

    final_calls = [call for call in result.llm_calls if call["agent_name"] == "final_tutor"]
    assert len(final_calls) == 1, "guard disabled => no regeneration"
    assert not result.agent_outputs.get("leakage_guard")
