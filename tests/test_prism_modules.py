from prism_tutor.agents.base_client import BaseLLMClient, LLMClientConfig
from prism_tutor.agents.parser import parse_agent_json
from prism_tutor.agents.schemas import AGENT_SCHEMAS
from prism_tutor.agents.types import LLMCallRecord, LLMUsage
from prism_tutor.experiments.method_registry import MethodSpec
from prism_tutor.experiments.runner import _prism_graph_config_from_run_config, _run_live_baseline
from prism_tutor.runtime.graph_state import TutorGraphState
from prism_tutor.runtime.prism_graph import PrismGraphConfig, build_prism_graph
from prism_tutor.runtime.qos_router import QoSRouter
from prism_tutor.runtime.risk_estimator import RiskConfig, estimate_risk
from prism_tutor.runtime.state_commit import StateCommitter


def test_router_selects_low_medium_high_agents():
    router = QoSRouter()
    low_state = TutorGraphState(sample={"sample_id": "l", "difficulty": "easy"}, method="M1")
    low = estimate_risk(low_state, RiskConfig(low_threshold=0.5, high_threshold=0.8))
    assert "final_tutor" in router.select_agents(low)
    assert "hint" not in router.select_agents(low)

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
    assert "pedagogy" in selected
    assert "state_manager" in selected


def test_risk_estimator_uses_dataset_schema_signals_for_routing():
    state = TutorGraphState(
        sample={
            "sample_id": "bridge-1",
            "problem_text": "Solve 8 / 2.",
            "student_utterance": "I think it is 6.",
            "student_error": "operation_confusion",
            "remediation_strategy": "contrast",
            "teacher_intention": "clarify_misunderstanding",
        },
        method="M1",
    )

    risk = estimate_risk(state)
    selected = QoSRouter().select_agents(risk)

    assert risk.misconception_risk >= 0.6
    assert risk.pedagogy_risk >= 0.6
    assert risk.answer_uncertainty < 0.7
    assert {"misconception", "pedagogy", "verifier", "final_tutor"}.issubset(set(selected))
    assert "solver" not in selected
    assert "hint" not in selected


def test_prism_graph_routes_dynamically_from_sample_schema():
    graph = build_prism_graph("M1")
    result = graph.invoke(
        {
            "sample": {
                "sample_id": "mathdial-1",
                "problem_text": "What is 12 + 3?",
                "student_utterance": "I got 9.",
                "scaffolding": ["probing"],
                "metadata": {"ground_truth": "15"},
            },
            "method": "M1",
        }
    )

    assert result.risk_scores
    assert {"solver", "pedagogy", "verifier", "final_tutor"}.issubset(set(result.selected_agents))
    assert "hint" not in result.selected_agents


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
    assert state.errors == []


def test_prism_graph_callable_runs_in_mock_mode():
    graph = build_prism_graph("M3")
    result = graph.invoke({"sample": {"sample_id": "s1", "question": "What is 1+1?"}, "method": "M3"})
    assert result.termination_reason == "completed"
    assert result.llm_calls
    assert "state_commit" in result.agent_outputs


def test_prism_graph_m3_runs_state_manager_before_final_tutor():
    client = BaseLLMClient(
        LLMClientConfig(
            mock_responses={
                "state_manager": {
                    "proposed_updates": [
                        {
                            "field": "weak_skills",
                            "operation": "add",
                            "value": "fractions",
                            "confidence": 0.9,
                            "evidence": "student confused denominator",
                        }
                    ],
                    "conflicts": [],
                    "confidence": 0.9,
                }
            }
        )
    )
    graph = build_prism_graph("M3", client=client)

    result = graph.invoke(
        {
            "sample": {
                "sample_id": "s1",
                "problem_text": "Which fraction is larger, 1/3 or 1/4?",
                "student_utterance": "I think 1/4 is larger.",
                "misconception_label": "denominator_ordering",
                "metadata": {"ground_truth": "1/3"},
            },
            "method": "M3",
        }
    )

    assert "state_manager" in result.selected_agents
    assert result.selected_agents.index("state_manager") < result.selected_agents.index("final_tutor")
    assert result.selected_agents.count("final_tutor") == 1
    assert sum(call["agent_name"] == "final_tutor" for call in result.llm_calls) == 1
    assert result.agent_outputs["state_commit"][-1]["status"] == "committed"
    assert result.student_state.weak_skills == ["fractions"]


class _LeakageRetryClient:
    def __init__(self) -> None:
        self.final_calls = 0

    def call(self, *, sample_id, method, agent_name, messages, schema):
        if agent_name == "final_tutor":
            self.final_calls += 1
            if self.final_calls == 1:
                raw = '{"response":"The answer is 42.","withheld_answer":false,"confidence":0.8,"safety_notes":[]}'
            else:
                raw = '{"response":"Try checking the relationship in the problem first.","withheld_answer":true,"confidence":0.8,"safety_notes":["regenerated"]}'
        else:
            raw = BaseLLMClient(
                LLMClientConfig(mock_mode=True)
            )._mock_completion(agent_name, AGENT_SCHEMAS.get(agent_name))
        parsed = parse_agent_json(raw, schema)
        return LLMCallRecord(
            sample_id=sample_id,
            method=method,
            agent_name=agent_name,
            endpoint="mock://leakage-retry",
            prompt=messages,
            raw_completion=raw,
            stripped_output=parsed.stripped_output,
            parsed_output=parsed.parsed_output,
            usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2, source="mock"),
            parse_success=parsed.parse_success,
        )


def test_prism_graph_regenerates_leaky_final_response_once():
    client = _LeakageRetryClient()
    graph = build_prism_graph("M3", client=client)

    result = graph.invoke(
        {
            "sample": {
                "sample_id": "s1",
                "problem": "What is 40+2?",
                "metadata": {"ground_truth": "42"},
            },
            "method": "M3",
        }
    )

    final_calls = [call for call in result.llm_calls if call["agent_name"] == "final_tutor"]
    assert len(final_calls) == 2
    assert result.agent_outputs["leakage_guard"]
    assert final_calls[-1]["parsed_output"]["response"] == "Try checking the relationship in the problem first."


def test_live_state_baselines_commit_state_updates():
    client = BaseLLMClient(
        LLMClientConfig(
            mock_responses={
                "state_manager": {
                    "proposed_updates": [
                        {
                            "field": "active_misconceptions",
                            "operation": "add",
                            "value": "sign_error",
                            "confidence": 0.9,
                            "evidence": "student inverted the sign",
                        }
                    ],
                    "conflicts": [],
                    "confidence": 0.9,
                }
            }
        )
    )
    method = MethodSpec(
        "two_phase_commit",
        "state",
        "Two-phase state commit",
        ("state_proposer", "verifier", "state_commit", "final_tutor"),
    )

    result = _run_live_baseline({"sample_id": "s1", "dataset": "mathdial", "split": "test"}, method, client)

    assert result["state"]["agent_outputs"]["state_commit"][-1]["status"] == "committed"
    assert result["state"]["student_state"]["active_misconceptions"] == ["sign_error"]


def test_exp3_ours_full_uses_state_commit_focused_runtime():
    method = MethodSpec(
        "ours_full",
        "M3",
        "Full PRISM-Tutor runtime",
        ("risk_estimator", "qos_router", "budget_controller", "state_commit", "final_tutor"),
    )
    graph_config = _prism_graph_config_from_run_config(
        {
            "experiment": {"name": "exp3_state_commit"},
            "thresholds": {},
            "budget": {"max_rounds": 3, "max_tokens_per_case": 4000},
            "risk_weights": {},
        },
        method,
    )

    assert graph_config.variant["state_commit_focus"] is True
    assert "budget_controller" in graph_config.disabled_modules


def test_prism_graph_can_disable_state_commit_for_ablation():
    graph = build_prism_graph("M3", config=PrismGraphConfig(disabled_modules=["state_commit"]))
    result = graph.invoke({"sample": {"sample_id": "s1", "question": "What is 1+1?"}, "method": "M3"})

    assert result.termination_reason == "completed"
    assert "state_commit" not in result.agent_outputs
    assert result.agent_outputs["runtime_variant"] == [{}]


def test_prism_graph_can_bypass_qos_router_for_ablation():
    graph = build_prism_graph("M1", config=PrismGraphConfig(disabled_modules=["qos_routing"]))
    result = graph.invoke({"sample": {"sample_id": "s1", "difficulty": "easy"}, "method": "M1"})

    assert "state_manager" in result.selected_agents


def test_prism_graph_injects_noisy_agent_outputs_deterministically():
    graph = build_prism_graph("M1", config=PrismGraphConfig(noisy_agent_probability=1.0, noisy_agent_seed=7))
    result = graph.invoke(
        {
            "sample": {
                "sample_id": "s1",
                "question": "What is 1+1?",
                "scaffolding": ["probing"],
                "metadata": {"ground_truth": "2"},
            },
            "method": "M1",
        }
    )

    noisy_calls = [call for call in result.llm_calls if "noisy_agent_injected" in call.get("warnings", [])]
    assert noisy_calls
    assert any(call["request_payload"].get("robustness_noise", {}).get("seed") == 7 for call in noisy_calls)
