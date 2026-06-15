from __future__ import annotations

import json
from pathlib import Path

from prism_tutor.experiments.experiment_matrix import load_experiment_matrix
from prism_tutor.experiments.method_registry import default_method_registry
from prism_tutor.experiments.runner import RunnerOptions, run_generation


REQUIRED_FIELDS = {
    "sample_id",
    "dataset",
    "split",
    "method",
    "base_model",
    "endpoint",
    "generation_config",
    "selected_agents",
    "rounds",
    "risk_scores",
    "messages",
    "state",
    "token_usage",
    "latency_seconds",
    "final_response",
    "parse_success",
    "errors",
}


def test_default_method_registry_covers_task_card_methods() -> None:
    names = set(default_method_registry().names())
    assert {"single_tutor", "fixed_2", "fixed_4", "debate", "generic_sparse", "difficulty_routing"} <= names
    assert {"ours_routing", "ours_routing_budget", "ours_full"} <= names
    assert {"ablate_risk_estimator", "replace_two_phase_commit_with_naive_memory"} <= names


def test_experiment_matrix_loads_exp0_to_exp6() -> None:
    matrix = load_experiment_matrix()
    assert {
        "exp0_problem_diagnosis",
        "exp1_routing",
        "exp2_budget",
        "exp3_state_commit",
        "exp4_end_to_end",
        "exp5_ablation",
        "exp6_robustness",
    } <= set(matrix)
    assert matrix["exp6_robustness"].extra["noisy_agent_probabilities"] == [0.2, 0.4]


def test_runner_writes_smoke_generation_jsonl_and_manifest(tmp_path: Path) -> None:
    result = run_generation(
        RunnerOptions(
            methods=["single_tutor"],
            datasets=["mathdial"],
            split="test",
            limit=2,
            output_dir=str(tmp_path),
            run_id="smoke",
        )
    )
    assert result["status"] == "completed"
    generation_path = Path(result["paths"]["generations"])
    manifest_path = Path(result["paths"]["manifest"])
    records = [json.loads(line) for line in generation_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert REQUIRED_FIELDS <= set(records[0])
    assert records[0]["method"] == "single_tutor"
    assert records[0]["parse_success"] is True
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["run"]["counts"]["succeeded"] == 2
