from __future__ import annotations

from prism_tutor.serving.endpoints import Endpoint, EndpointRegistry, strip_think_blocks
from serving.health_check import run_health_check
from serving.vllm_command import build_vllm_command


def test_strip_think_blocks_preserves_visible_answer() -> None:
    assert strip_think_blocks("<think>hidden</think>{\"answer\": 1}") == '{"answer": 1}'


def test_endpoint_registry_selects_samples_reproducibly() -> None:
    registry = EndpointRegistry(
        [
            Endpoint(base_url="http://localhost:8000/v1", model="qwen3-8b-gpu0"),
            Endpoint(base_url="http://localhost:8001/v1", model="qwen3-8b-gpu1"),
        ]
    )
    first = registry.select_for_sample("mathdial:test:1", method="single_tutor")
    second = registry.select_for_sample("mathdial:test:1", method="single_tutor")
    assert first == second


def test_health_check_dry_run_reports_usage_and_enable_thinking_false() -> None:
    config = {
        "model": {
            "generator": "Qwen/Qwen3-8B",
            "enable_thinking": False,
            "endpoints": [
                {"base_url": "http://localhost:8000/v1", "model": "qwen3-8b-gpu0"},
                {"base_url": "http://localhost:8001/v1", "model": "qwen3-8b-gpu1"},
            ],
        },
        "generation": {"timeout_seconds": 1, "max_tokens": {"verifier": 32}},
    }
    report = run_health_check(config, live=False)
    assert report["mode"] == "dry-run"
    assert report["enable_thinking"] is False
    assert report["all_ok"] is True
    assert len(report["endpoints"]) == 2
    assert report["endpoints"][0]["token_usage"]["total_tokens"] == 0


def test_vllm_command_uses_modelscope_and_profile_devices() -> None:
    config = {
        "generator": "Qwen/Qwen3-8B",
        "served_model_names": {"gpu0": "qwen3-8b-gpu0"},
        "ports": {"gpu0": 8000},
        "preferred_devices": {"gpu0": "2"},
        "max_model_len": {"single_gpu": 8192},
        "use_modelscope": True,
    }
    env, command = build_vllm_command(config, "gpu0")
    assert env["VLLM_USE_MODELSCOPE"] == "true"
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert "--max-model-len" in command
    assert "8192" in command
