from prism_tutor.runtime.graph_state import TutorGraphState
from prism_tutor.runtime.prism_graph import build_prism_graph
from prism_tutor.runtime.qos_router import QoSRouter
from prism_tutor.runtime.risk_estimator import RiskConfig, estimate_risk
from prism_tutor.runtime.state_commit import StateCommitter


def test_router_selects_low_medium_high_agents():
    router = QoSRouter()
    low_state = TutorGraphState(sample={"sample_id": "l", "difficulty": "easy"}, method="M1")
    low = estimate_risk(low_state, RiskConfig(low_threshold=0.5, high_threshold=0.8))
    assert "final_tutor" in router.select_agents(low)

    high_state = TutorGraphState(sample={"sample_id": "h", "difficulty": "hard"}, method="M1")
    high_state.agent_outputs["verifier"] = [
        {
            "approved": False,
            "issues": [{"issue_type": "leakage", "severity": "high", "message": "leak"}],
            "leakage_detected": True,
            "state_conflict_detected": True,
            "confidence": 0.9,
        }
    ]
    high = estimate_risk(high_state)
    selected = router.select_agents(high)
    assert "hint" in selected
    assert "state_manager" in selected


def test_state_commit_tentative_on_conflict():
    state = TutorGraphState(sample={"sample_id": "s1"}, method="M3")
    state.agent_outputs["state_manager"] = [
        {
            "proposed_updates": [
                {
                    "field": "weak_skills",
                    "operation": "add",
                    "value": "fractions",
                    "confidence": 0.9,
                    "evidence": "missed denominator",
                }
            ],
            "conflicts": [],
            "confidence": 0.9,
        }
    ]
    state.agent_outputs["verifier"] = [
        {
            "approved": False,
            "issues": [],
            "leakage_detected": False,
            "state_conflict_detected": True,
            "confidence": 0.8,
        }
    ]
    decision = StateCommitter().commit(state)
    assert decision.status == "tentative"
    assert state.student_state.tentative_updates


def test_prism_graph_callable_runs_in_mock_mode():
    graph = build_prism_graph("M3")
    result = graph.invoke({"sample": {"sample_id": "s1", "question": "What is 1+1?"}, "method": "M3"})
    assert result.termination_reason == "completed"
    assert result.llm_calls
    assert "state_commit" in result.agent_outputs
