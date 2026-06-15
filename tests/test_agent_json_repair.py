from prism_tutor.agents.base_client import BaseLLMClient, LLMClientConfig
from prism_tutor.agents.final_tutor import FinalTutorAgent
from prism_tutor.agents.parser import parse_agent_json
from prism_tutor.agents.runner import run_agent
from prism_tutor.agents.schemas import FinalTutorOutput, HintOutput, SolverOutput
from prism_tutor.agents.types import LLMCallRecord, LLMError, LLMUsage


def test_parse_strips_think_and_markdown_fence():
    raw = """<think>hidden</think>
```json
{"hint_text": "Look at the givens.", "hint_level": 1, "answer_leakage_risk": 0.1, "confidence": 0.8}
```"""
    result = parse_agent_json(raw, HintOutput)
    assert result.parse_success is True
    assert "hidden" not in result.stripped_output
    assert "think_block_removed" in result.warnings


def test_repair_single_quotes_and_trailing_comma():
    raw = "{'answer': '42', 'reasoning': ['compute'], 'confidence': 0.8, 'uncertainty': 0.2, 'needs_more_info': False,}"
    result = parse_agent_json(raw, SolverOutput)
    assert result.parse_success is True
    assert result.parsed_output["answer"] == "42"
    assert "trailing_commas_removed" in result.warnings


def test_repair_failure_is_not_silent_empty_json():
    result = parse_agent_json("not json", HintOutput)
    assert result.parse_success is False
    assert result.parsed_output is None
    assert result.error


class _RetryFakeClient:
    def __init__(self) -> None:
        self.calls = 0

    def call(self, *, sample_id, method, agent_name, messages, schema):
        self.calls += 1
        if self.calls == 1:
            return LLMCallRecord(
                sample_id=sample_id,
                method=method,
                agent_name=agent_name,
                endpoint="mock://retry",
                prompt=messages,
                raw_completion="not json",
                stripped_output="not json",
                usage=LLMUsage(source="mock"),
                parse_success=False,
                error=LLMError(code="parse_error", message="parse failed", retryable=True),
            )
        return LLMCallRecord(
            sample_id=sample_id,
            method=method,
            agent_name=agent_name,
            endpoint="mock://retry",
            prompt=messages,
            raw_completion='{"answer":"42","reasoning":["retry"],"confidence":0.8,"uncertainty":0.2,"needs_more_info":false}',
            stripped_output='{"answer":"42","reasoning":["retry"],"confidence":0.8,"uncertainty":0.2,"needs_more_info":false}',
            parsed_output={
                "schema_version": "0.1.0",
                "answer": "42",
                "reasoning": ["retry"],
                "confidence": 0.8,
                "uncertainty": 0.2,
                "needs_more_info": False,
            },
            usage=LLMUsage(source="mock"),
            parse_success=True,
        )


def test_run_agent_retries_after_parse_failure():
    client = _RetryFakeClient()

    record = run_agent(
        agent_name="solver",
        schema=SolverOutput,
        sample={"sample_id": "s1", "problem": "1+1"},
        state={},
        client=client,
        method="B0",
    )

    assert client.calls == 2
    assert record.parse_success is True
    assert record.parsed_output["answer"] == "42"
    assert "retry_after_parse_error" in record.warnings


def test_final_tutor_think_block_is_not_student_visible():
    raw = '<think>hidden chain</think>{"response":"Try the next step.","withheld_answer":true,"confidence":0.8,"safety_notes":[]}'
    client = BaseLLMClient(
        LLMClientConfig(
            endpoints=["mock://final"],
            mock_mode=True,
            mock_responses={"final_tutor": raw},
        )
    )

    record = FinalTutorAgent().invoke(
        sample={"sample_id": "s1", "problem": "1+1"},
        state={},
        client=client,
        method="B0",
    )

    assert record.parse_success is True
    assert "hidden chain" not in record.stripped_output
    assert record.parsed_output == FinalTutorOutput(
        response="Try the next step.",
        withheld_answer=True,
        confidence=0.8,
        safety_notes=[],
    ).model_dump(mode="json")
    assert "think_block_removed" in record.warnings
