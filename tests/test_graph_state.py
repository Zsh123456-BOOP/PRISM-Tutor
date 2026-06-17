import json

from prism_tutor.agents.types import LLMCallRecord, LLMError, LLMUsage
from prism_tutor.runtime.budget_controller import BudgetConfig, BudgetController
from prism_tutor.runtime.checkpointing import CheckpointWriter
from prism_tutor.runtime.graph_builder import GraphBuilder
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


def test_graph_builder_fallback_uses_runtime_state_interface():
    def mark_node(state: TutorGraphState) -> TutorGraphState:
        state.selected_agents.append("marker")
        state.rounds += 1
        return state

    graph = GraphBuilder(prefer_langgraph=True).add_node("mark", mark_node).compile()
    result = graph.invoke({"sample": {"sample_id": "s1"}, "method": "baseline"})

    assert graph.backend in {"langgraph", "simple_fallback"}
    assert result.selected_agents == ["marker"]
    assert result.rounds == 1


def test_checkpoint_contains_runtime_audit_fields(tmp_path):
    state = TutorGraphState(sample={"sample_id": "s1"}, method="M3", rounds=2)
    state.selected_agents.append("solver")
    state.agent_outputs["solver"] = [{"answer": "x"}]
    state.state_before.append({"weak_skills": []})
    state.state_after.append({"weak_skills": ["fractions"]})

    path = CheckpointWriter(tmp_path).write(state)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["rounds"] == 2
    assert payload["selected_agents"] == ["solver"]
    assert payload["agent_outputs"]["solver"][0]["answer"] == "x"
    assert payload["state_before"]
    assert payload["state_after"]


def test_budget_controller_marks_max_rounds_and_token_budget():
    max_rounds_state = TutorGraphState(sample={"sample_id": "s1"}, method="M2", rounds=2)
    max_rounds = BudgetController(BudgetConfig(max_rounds=2, max_tokens=100))

    assert max_rounds.should_continue(max_rounds_state) is False
    assert max_rounds_state.termination_reason == "max_rounds"

    token_state = TutorGraphState(sample={"sample_id": "s2"}, method="M2", total_tokens=101)
    token_budget = BudgetController(BudgetConfig(max_rounds=3, max_tokens=100))

    assert token_budget.should_continue(token_state) is False
    assert token_state.termination_reason == "token_budget"
    assert token_state.budget_exhausted is True
    assert token_state.errors == []
