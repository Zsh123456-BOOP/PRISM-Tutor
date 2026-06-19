from prism_tutor.agents.base_client import BaseLLMClient, LLMClientConfig
from prism_tutor.agents.parser import parse_agent_json
from prism_tutor.agents.prompts import build_agent_messages
from prism_tutor.agents.schemas import AGENT_SCHEMAS
from prism_tutor.agents.types import LLMCallRecord, LLMUsage
from prism_tutor.experiments.method_registry import MethodSpec
from prism_tutor.experiments.runner import _prism_graph_config_from_run_config, _run_live_baseline
from prism_tutor.runtime.graph_state import TutorGraphState
from prism_tutor.runtime.prism_graph import PrismGraphConfig, build_prism_graph
from prism_tutor.runtime.qos_router import QoSRouter
from prism_tutor.runtime.risk_estimator import RiskConfig, estimate_risk
from prism_tutor.data.sample_view import build_model_input
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


def test_risk_estimator_uses_visible_student_signals_for_routing():
    state = TutorGraphState(
        sample={
            "sample_id": "bridge-1",
            "problem_text": "Solve 8 / 2.",
            "student_utterance": "I think it is 6.",
            "student_error": "forbidden_gold_error",
            "remediation_strategy": "forbidden_gold_strategy",
            "teacher_intention": "forbidden_gold_intention",
        },
        method="M1",
    )

    risk = estimate_risk(state)
    selected = QoSRouter().select_agents(risk)

    # A short, confused turn escalates misconception + pedagogy. Answer uncertainty
    # stays low (easy problem), but the solver is still routed because diagnosis
    # needs the reference solution -- the misconception agent compares the student
    # answer against the solver output (this closes the F1 gap to fixed_4).
    assert risk.misconception_risk >= 0.6
    assert risk.pedagogy_risk >= 0.6
    assert risk.answer_uncertainty < 0.7
    assert {"solver", "misconception", "pedagogy", "verifier", "final_tutor"}.issubset(set(selected))
    assert "hint" not in selected

    # A long / hard problem raises estimated difficulty -> answer uncertainty,
    # which routes the solver. Routing therefore varies with the visible signal.
    hard = TutorGraphState(
        sample={
            "sample_id": "hard-1",
            "problem_text": "Compute the result of the following multi-step word problem. " * 12,
            "student_utterance": "I am not sure how to start.",
        },
        method="M1",
    )
    hard_risk = estimate_risk(hard)
    assert hard_risk.answer_uncertainty >= 0.7
    assert "solver" in QoSRouter().select_agents(hard_risk)


def test_model_input_strips_gold_fields_from_agent_prompt():
    sample = {
        "sample_id": "s1",
        "problem": "What is 2+2?",
        "student_utterance": "I think it is 5.",
        "ground_truth": "4",
        "misconception_label": "addition_error",
        "metadata": {"correct_answer": "4", "topic": "addition"},
    }

    model_input = build_model_input(sample)
    messages = build_agent_messages("solver", sample, {})
    prompt = messages[-1]["content"]

    assert "ground_truth" not in model_input
    assert "misconception_label" not in model_input
    assert "correct_answer" not in prompt
    assert "addition_error" not in prompt
    assert "topic" in prompt


def test_risk_estimator_is_stable_under_fake_gold_changes():
    visible = {
        "sample_id": "s1",
        "problem": "What is 2+2?",
        "student_utterance": "I think it is 5.",
    }
    a = {**visible, "ground_truth": "4", "misconception_label": "addition_error"}
    b = {**visible, "ground_truth": "999", "misconception_label": "ratio_error"}

    risk_a = estimate_risk(TutorGraphState(sample=build_model_input(a), method="M1"))
    risk_b = estimate_risk(TutorGraphState(sample=build_model_input(b), method="M1"))

    assert risk_a.model_dump(mode="json") == risk_b.model_dump(mode="json")


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
    # Confused turn -> misconception + pedagogy routed; the solver is also routed so
    # diagnosis has a reference solution, and the graph orders it FIRST.
    assert {"solver", "misconception", "pedagogy", "verifier", "final_tutor"}.issubset(set(result.selected_agents))
    assert result.selected_agents.index("solver") < result.selected_agents.index("misconception")
    assert "hint" not in result.selected_agents


def test_state_commit_tentative_on_conflict():
    state = TutorGraphState(sample={"sample_id": "s1"}, method="M3")
    # Prior committed state exists, so a verifier-flagged conflict is genuine and
    # all proposed updates are held tentative.
    state.student_state.active_misconceptions = ["place_value_error"]
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


def test_state_commit_commits_on_first_turn_despite_verifier_conflict_flag():
    # No prior committed state (turn 1) -> a verifier state-conflict flag is a false
    # positive (nothing to conflict with), so a confident update is committed.
    state = TutorGraphState(sample={"sample_id": "s1"}, method="M3")
    state.agent_outputs["state_manager"] = [
        {
            "proposed_updates": [
                {
                    "field": "active_misconceptions",
                    "operation": "add",
                    "value": "denominator_confusion",
                    "confidence": 0.8,
                    "evidence": "student inverted the fraction",
                }
            ],
            "conflicts": [],
            "confidence": 0.8,
        }
    ]
    state.agent_outputs["verifier"] = [
        {"approved": False, "issues": [], "leakage_detected": False, "state_conflict_detected": True, "confidence": 0.8}
    ]
    decision = StateCommitter().commit(state)
    assert decision.status == "committed"
    assert state.student_state.active_misconceptions == ["denominator_confusion"]


def test_candidate_misconceptions_only_reach_the_misconception_agent():
    from prism_tutor.agents.prompts import build_agent_messages

    sample = {
        "sample_id": "m1",
        "problem_text": "What is 1/2 + 1/3?",
        "student_utterance": "2/5",
        "candidate_misconceptions": ["A misconception", "B misconception"],
    }
    misc_prompt = build_agent_messages("misconception", sample, {})[-1]["content"]
    solver_prompt = build_agent_messages("solver", sample, {})[-1]["content"]
    assert "candidate_misconceptions" in misc_prompt
    assert "candidate_misconceptions" not in solver_prompt


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


def test_prism_graph_budget_loop_prunes_repeated_agent_calls():
    client = BaseLLMClient(
        LLMClientConfig(
            mock_responses={
                "verifier": {
                    "approved": False,
                    "issues": [
                        {
                            "issue_type": "incorrect_answer",
                            "severity": "medium",
                            "message": "Needs a solver check.",
                            "recommended_agent": "solver",
                        },
                        {
                            "issue_type": "pedagogy",
                            "severity": "low",
                            "message": "Needs a pedagogy check.",
                            "recommended_agent": "pedagogy",
                        },
                    ],
                    "leakage_detected": False,
                    "state_conflict_detected": False,
                    "confidence": 0.7,
                }
            }
        )
    )
    graph = build_prism_graph(
        "M3",
        client=client,
        config=PrismGraphConfig(),
    )

    result = graph.invoke(
        {
            "sample": {
                "sample_id": "mathdial-repeat",
                "problem_text": "What is 12 + 3?",
                "student_utterance": "I got 9.",
                "metadata": {"ground_truth": "15"},
            },
            "method": "M3",
        }
    )

    call_counts = {
        agent: sum(call["agent_name"] == agent for call in result.llm_calls)
        for agent in {"solver", "pedagogy", "misconception", "state_manager", "verifier", "final_tutor"}
    }
    assert call_counts["solver"] == 1
    assert call_counts["pedagogy"] == 1
    assert call_counts["misconception"] == 1
    assert call_counts["state_manager"] == 1
    # Work agents are pruned to a single call; the verifier is intentionally
    # re-run after each deliberation round, so it may be called more than once.
    assert call_counts["verifier"] >= 1
    assert call_counts["final_tutor"] == 1


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


def test_exp4_ours_full_skips_budget_reverification_but_m2_keeps_it():
    config = {
        "experiment": {"name": "exp4_end_to_end"},
        "thresholds": {},
        "budget": {"max_rounds": 3, "max_tokens_per_case": 4000},
        "risk_weights": {},
    }
    ours_full = MethodSpec(
        "ours_full",
        "M3",
        "Full PRISM-Tutor runtime",
        ("risk_estimator", "qos_router", "budget_controller", "state_commit", "final_tutor"),
    )
    ours_routing_budget = MethodSpec(
        "ours_routing_budget",
        "M2",
        "PRISM routing and budget",
        ("risk_estimator", "qos_router", "budget_controller", "final_tutor"),
    )

    full_config = _prism_graph_config_from_run_config(config, ours_full)
    budget_config = _prism_graph_config_from_run_config(config, ours_routing_budget)

    assert full_config.budget.verify_after_new_agents is False
    assert budget_config.budget.verify_after_new_agents is True


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


def test_misconception_routed_on_student_answer_without_confusion_words():
    # Misconception-Benchmark style: a terse wrong answer, no "I think / confused"
    # words. The diagnosis agents must still be routed (this is what was missing).
    answered = TutorGraphState(
        sample={"sample_id": "mmb1", "problem_text": "What is 1/2 + 1/3?", "student_utterance": "2/5"},
        method="M1",
    )
    selected = set(QoSRouter().select_agents(estimate_risk(answered)))
    assert "misconception" in selected
    assert "pedagogy" in selected

    # With no student answer there is nothing to diagnose -> agents not forced.
    unanswered = TutorGraphState(
        sample={"sample_id": "q1", "problem_text": "What is 1/2 + 1/3?"},
        method="M1",
    )
    assert "misconception" not in set(QoSRouter().select_agents(estimate_risk(unanswered)))


def test_prism_rounds_reflect_actual_deliberation_not_nominal():
    from prism_tutor.experiments.method_registry import default_method_registry
    from prism_tutor.experiments.runner import _run_live_prism
    from prism_tutor.utils.config import load_config

    method = default_method_registry().get("ours_full")  # nominal method.rounds == 3
    result = _run_live_prism(
        {"sample_id": "s1", "question": "What is 1+1?", "dataset": "mathdial", "split": "test"},
        method,
        BaseLLMClient(),  # mock verifier approves -> no extra deliberation rounds
        load_config(),
    )
    # Was pinned to the nominal 3 before the fix; now reflects 1 initial pass.
    assert result["rounds"] == 1


def test_budget_rounds_are_risk_conditioned():
    from prism_tutor.runtime.budget_controller import BudgetConfig, BudgetController

    controller = BudgetController(BudgetConfig())
    low = TutorGraphState(sample={"sample_id": "l"}, method="M2")
    low.risk_scores.append({"risk_bucket": "low"})
    high = TutorGraphState(sample={"sample_id": "h"}, method="M2")
    high.risk_scores.append({"risk_bucket": "high"})

    assert controller._effective_max_rounds(low) == 1
    assert controller._effective_max_rounds(high) == 3

    low.rounds = 1
    assert controller.should_continue(low) is False  # low-risk stops after one round
    high.rounds = 1
    assert controller.should_continue(high) is True  # high-risk keeps deliberating


def test_prism_graph_injects_noisy_agent_outputs_deterministically():
    graph = build_prism_graph("M1", config=PrismGraphConfig(noisy_agent_probability=1.0, noisy_agent_seed=7))
    result = graph.invoke(
        {
            "sample": {
                "sample_id": "s1",
                "question": "What is 1+1?",
                "student_utterance": "I think it is 3.",
                "scaffolding": ["probing"],
                "metadata": {"ground_truth": "2"},
            },
            "method": "M1",
        }
    )

    noisy_calls = [call for call in result.llm_calls if "noisy_agent_injected" in call.get("warnings", [])]
    assert noisy_calls
    assert any(call["request_payload"].get("robustness_noise", {}).get("seed") == 7 for call in noisy_calls)
