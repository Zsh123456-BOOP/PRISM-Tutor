from prism_tutor.agents.base_client import BaseLLMClient
from prism_tutor.baselines.strategies import generic_sparse_plan, random_routing_plan
from prism_tutor.experiments.method_registry import default_method_registry
from prism_tutor.experiments.runner import _run_live_baseline


def test_random_routing_is_seeded_random_not_generic_sparse():
    sample = {"sample_id": "a", "problem_text": "What is 1/2 + 1/3?", "student_utterance": "2/5"}
    plan_a = random_routing_plan(sample)
    plan_a2 = random_routing_plan(sample)
    assert plan_a.agents == plan_a2.agents  # deterministic given the same sample id
    assert "final_tutor" in plan_a.agents
    assert plan_a.metadata["strategy"] == "random_routing"
    # different samples generally route to different agent subsets
    others = {tuple(random_routing_plan({"sample_id": f"s{i}", "problem_text": "p"}).agents) for i in range(12)}
    assert len(others) > 1
    # and it is not just an alias of the deterministic generic-sparse utility planner
    assert random_routing_plan(sample).metadata["strategy"] != generic_sparse_plan(sample).metadata["strategy"]


def test_fixed_round_baselines_deliberate_multiple_rounds():
    reg = default_method_registry()
    client = BaseLLMClient()  # mock
    sample = {"sample_id": "s1", "problem_text": "What is 12 + 3?", "student_utterance": "9", "dataset": "mathdial", "split": "test"}

    one = _run_live_baseline(sample, reg.get("fixed_4"), client)          # rounds=1
    debate = _run_live_baseline(sample, reg.get("debate"), client)        # rounds=2
    four = _run_live_baseline(sample, reg.get("fixed_4_rounds"), client)  # rounds=4

    assert one["rounds"] == 1
    assert debate["rounds"] == 2 and debate["agent_calls"] > one["agent_calls"]
    assert four["rounds"] == 4 and four["agent_calls"] > debate["agent_calls"]
    # the final student-facing response is produced exactly once regardless of rounds
    assert sum(c["agent_name"] == "final_tutor" for c in four["state"]["llm_calls"]) == 1
