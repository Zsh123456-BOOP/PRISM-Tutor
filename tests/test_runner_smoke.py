from __future__ import annotations

import json
from pathlib import Path

from prism_tutor.experiments.experiment_matrix import load_experiment_matrix
from prism_tutor.experiments.method_registry import default_method_registry
import pytest

from prism_tutor.experiments.runner import (
    RunnerOptions,
    _expand_method_specs,
    _llm_client_from_config,
    _prism_graph_config_from_run_config,
    load_samples,
    run_generation,
)
from prism_tutor.utils.config import load_config


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
    "agent_calls",
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


def test_exp6_methods_expand_across_noise_and_token_budgets() -> None:
    matrix = load_experiment_matrix()
    registry = default_method_registry()
    base_methods = registry.resolve(matrix["exp6_robustness"].methods)

    expanded = _expand_method_specs(base_methods, matrix["exp6_robustness"], {"seed": 42})

    assert len(expanded) == len(base_methods) * 2 * 3
    assert expanded[0].variant["base_method"] == "fixed_4"
    assert expanded[0].variant["robustness"] is True
    assert expanded[0].variant["noisy_agent_probability"] == 0.2
    assert expanded[0].variant["token_budget"] == 1000
    assert expanded[0].name.startswith("fixed_4__noise0p2__budget1000")


def test_prism_graph_config_uses_default_yaml_and_ablation_variant() -> None:
    registry = default_method_registry()
    method = registry.get("ablate_leakage_risk")
    config = _prism_graph_config_from_run_config(load_config(), method)

    assert config.risk.weights["misconception_risk"] == 0.30
    assert config.risk.weights["estimated_difficulty"] == 0.20
    assert config.risk.low_threshold == 0.38
    assert config.risk.high_threshold == 0.55
    assert config.budget.max_rounds == 3
    assert config.budget.max_tokens == 4000
    assert config.budget.rounds_by_bucket == {"low": 1, "medium": 2, "high": 3}
    assert config.disabled_risks == ["leakage_risk"]


def test_llm_client_uses_default_yaml_endpoints_and_generation_config() -> None:
    client = _llm_client_from_config(load_config())

    assert client.config.mock_mode is False
    assert client.config.model_name == "Qwen/Qwen3-8B"
    assert client.config.temperature == 0.2
    assert client.config.top_p == 0.8
    assert client.config.top_k == 20
    assert client.config.max_tokens == 768
    assert client.config.timeout_s == 120
    assert client.config.retries == 1
    assert [endpoint.base_url for endpoint in client._endpoints] == [
        "http://localhost:8000/v1",
        "http://localhost:8001/v1",
    ]
    assert [endpoint.model for endpoint in client._endpoints] == ["qwen3-8b-gpu0", "qwen3-8b-gpu1"]

    payload = client.build_payload([{"role": "user", "content": "hello"}], model_name=client._endpoints[0].model)
    assert payload["model"] == "qwen3-8b-gpu0"
    assert payload["chat_template_kwargs"]["enable_thinking"] is False


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
    assert manifest["schema_version"] == "0.2.0"
    assert manifest["manifest_type"] == "experiment_run"
    assert manifest["status"] == "completed"
    assert manifest["run"]["counts"]["succeeded"] == 2
    assert manifest["audit"]["datasets"] == ["mathdial"]
    assert manifest["audit"]["methods"] == ["single_tutor"]
    assert manifest["audit"]["model"] == "Qwen/Qwen3-8B"
    assert manifest["audit"]["generation_config"]["enable_thinking"] is False
    assert manifest["audit"]["input_paths"]["data_splits"] == "data/splits"
    assert manifest["audit"]["output_paths"]["generations"] == str(generation_path)
    assert manifest["audit"]["duration_seconds"] is not None
    assert "CUDA_VISIBLE_DEVICES" in manifest["reproducibility"]["environment"]


def test_load_samples_shards_are_disjoint_and_cover_fallback_samples() -> None:
    shard0 = load_samples("missing_dataset", "test", limit=6, num_shards=2, shard_index=0)
    shard1 = load_samples("missing_dataset", "test", limit=6, num_shards=2, shard_index=1)
    ids0 = {row["sample_id"] for row in shard0}
    ids1 = {row["sample_id"] for row in shard1}

    assert ids0.isdisjoint(ids1)
    assert ids0 | ids1 == {f"missing_dataset:test:{index:06d}" for index in range(6)}


def test_runner_manifest_records_shard_options(tmp_path: Path) -> None:
    result = run_generation(
        RunnerOptions(
            methods=["single_tutor"],
            datasets=["mathdial"],
            split="test",
            limit=4,
            output_dir=str(tmp_path),
            run_id="shard",
            num_shards=2,
            shard_index=1,
        )
    )
    manifest = json.loads(Path(result["paths"]["manifest"]).read_text(encoding="utf-8"))
    assert manifest["run"]["num_shards"] == 2
    assert manifest["run"]["shard_index"] == 1


def test_runner_rejects_invalid_shard_options(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_generation(
            RunnerOptions(
                output_dir=str(tmp_path),
                num_shards=2,
                shard_index=2,
            )
        )
