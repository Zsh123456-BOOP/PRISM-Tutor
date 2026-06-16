from __future__ import annotations

import json
from pathlib import Path

from prism_tutor.agents.base_client import BaseLLMClient
from prism_tutor.baselines.strategies import difficulty_routing_plan, generic_sparse_plan, oracle_routing_plan
from prism_tutor.experiments.method_registry import default_method_registry
from prism_tutor.experiments.runner import RunnerOptions, _run_live_baseline, run_generation


def test_single_tutor_live_baseline_uses_one_agent_call() -> None:
    method = default_method_registry().get("single_tutor")
    result = _run_live_baseline(
        {"sample_id": "s1", "dataset": "mathdial", "split": "test", "problem": "What is 1 + 1?"},
        method,
        BaseLLMClient(),
    )

    assert result["selected_agents"] == ["final_tutor"]
    assert result["agent_calls"] == 1
    assert result["state"]["agent_outputs"]["baseline_plan"][0]["strategy"] == "single_tutor"


def test_generic_sparse_does_not_read_educational_risk_fields() -> None:
    sample = {"sample_id": "s1", "problem": "A train travels 30 miles in 2 hours. Find the speed."}
    with_risk_fields = {
        **sample,
        "risk_scores": [{"total_risk": 1.0}],
        "misconception_risk": 1.0,
        "leakage_risk": 1.0,
        "state_conflict_risk": 1.0,
        "pedagogy_risk": 1.0,
        "misconception_labels": ["forbidden"],
    }

    assert generic_sparse_plan(sample).agents == generic_sparse_plan(with_risk_fields).agents


def test_difficulty_routing_does_not_read_misconception_leakage_or_state_risk() -> None:
    sample = {"sample_id": "s1", "problem": "short problem", "difficulty": "medium"}
    with_forbidden_fields = {
        **sample,
        "misconception_risk": 1.0,
        "leakage_risk": 1.0,
        "state_conflict_risk": 1.0,
        "misconception_labels": ["forbidden"],
    }

    assert difficulty_routing_plan(sample).agents == difficulty_routing_plan(with_forbidden_fields).agents


def test_oracle_routing_handles_list_valued_gold_fields() -> None:
    sample = {
        "sample_id": "s1",
        "problem": "Explain the student error.",
        "misconception_label": ["ratio_confusion"],
    }

    plan = oracle_routing_plan(sample)

    assert plan.metadata["strategy"] == "oracle_routing"
    assert plan.metadata["upper_bound"] is True
    assert "misconception" in plan.agents


def test_baseline_dry_run_keeps_model_generation_config_and_samples_aligned(tmp_path: Path) -> None:
    result = run_generation(
        RunnerOptions(
            methods=["single_tutor", "fixed_2", "fixed_4", "debate", "generic_sparse", "difficulty_routing"],
            datasets=["mathdial"],
            split="test",
            limit=2,
            output_dir=str(tmp_path),
            run_id="baseline_fairness",
        )
    )
    rows = [json.loads(line) for line in Path(result["paths"]["generations"]).read_text(encoding="utf-8").splitlines()]

    by_method = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)
    sample_sequences = [[row["sample_id"] for row in method_rows] for method_rows in by_method.values()]
    generation_configs = {json.dumps(row["generation_config"], sort_keys=True) for row in rows}
    base_models = {row["base_model"] for row in rows}

    assert len(by_method) == 6
    assert all(sequence == sample_sequences[0] for sequence in sample_sequences)
    assert len(generation_configs) == 1
    assert base_models == {"Qwen/Qwen3-8B"}
