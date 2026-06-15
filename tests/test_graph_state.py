from prism_tutor.agents.types import LLMCallRecord, LLMError, LLMUsage
from prism_tutor.runtime.graph_state import TutorGraphState


def test_graph_state_records_agent_failure_and_tokens():
    state = TutorGraphState(sample={"sample_id": "s1", "question": "q"}, method="M1")
    record = LLMCallRecord(
        sample_id="s1",
        method="M1",
        agent_name="solver",
        endpoint="mock://a",
        prompt=[{"role": "user", "content": "q"}],
        raw_completion="",
        usage=LLMUsage(prompt_tokens=2, completion_tokens=0, total_tokens=2, source="mock"),
        error=LLMError(code="empty_response", message="empty"),
    )
    state.add_call(record)
    assert state.total_tokens == 2
    assert state.errors[0].code == "agent_failure"


def test_graph_state_snapshot_is_serializable():
    state = TutorGraphState(sample={"sample_id": "s1"}, method="M1")
    snapshot = state.snapshot()
    assert snapshot["sample"]["sample_id"] == "s1"
    assert snapshot["student_state"]["weak_skills"] == []
