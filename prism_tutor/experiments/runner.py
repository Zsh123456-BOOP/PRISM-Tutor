from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prism_tutor.agents.base_client import BaseLLMClient, LLMClientConfig, LLMEndpointConfig
from prism_tutor.experiments.experiment_matrix import ExperimentSpec, get_experiment
from prism_tutor.experiments.method_registry import MethodSpec, default_method_registry
from prism_tutor.logging.jsonl_logger import JsonlLogger
from prism_tutor.runtime.graph_state import TutorGraphState
from prism_tutor.runtime.prism_graph import AGENT_REGISTRY, build_prism_graph
from prism_tutor.logging.manifest import write_experiment_manifest
from prism_tutor.serving.endpoints import EndpointRegistry, strip_think_blocks
from prism_tutor.utils.config import deep_merge, load_config, load_yaml


@dataclass(frozen=True)
class RunnerOptions:
    config_path: str = "configs/default.yaml"
    experiments_config_path: str = "configs/experiments.yaml"
    methods: list[str] | None = None
    datasets: list[str] | None = None
    split: str | None = None
    experiment: str | None = None
    limit: int | None = None
    resume: bool = False
    output_dir: str = "outputs"
    run_id: str | None = None
    live_llm: bool = False
    num_shards: int = 1
    shard_index: int = 0


def _slug(values: list[str]) -> str:
    return "-".join(values).replace("/", "_")


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _validate_shard_options(num_shards: int, shard_index: int) -> None:
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")


def _apply_shard(samples: list[dict[str, Any]], *, num_shards: int, shard_index: int) -> list[dict[str, Any]]:
    _validate_shard_options(num_shards, shard_index)
    if num_shards == 1:
        return samples
    return [sample for index, sample in enumerate(samples) if index % num_shards == shard_index]


def load_samples(
    dataset: str,
    split: str,
    *,
    limit: int | None = None,
    num_shards: int = 1,
    shard_index: int = 0,
) -> list[dict[str, Any]]:
    candidates = [
        Path("data/splits") / dataset / f"{split}.jsonl",
        Path("data/splits") / f"{dataset}_{split}.jsonl",
        Path("data/processed") / dataset / f"{split}.jsonl",
        Path("data/processed") / f"{dataset}_{split}.jsonl",
    ]
    samples: list[dict[str, Any]] = []
    for path in candidates:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                record.setdefault("sample_id", record.get("id") or f"{dataset}:{split}:{len(samples):06d}")
                record.setdefault("dataset", dataset)
                record.setdefault("split", split)
                samples.append(record)
        sharded = _apply_shard(samples, num_shards=num_shards, shard_index=shard_index)
        return sharded[:limit] if limit is not None else sharded

    count = limit if limit is not None else 1
    synthetic = [
        {
            "sample_id": f"{dataset}:{split}:{index:06d}",
            "dataset": dataset,
            "split": split,
            "problem": f"Dry-run sample {index} for {dataset}/{split}.",
            "source": "synthetic_smoke_fallback",
        }
        for index in range(count)
    ]
    return _apply_shard(synthetic, num_shards=num_shards, shard_index=shard_index)


def _resolve_run_plan(options: RunnerOptions) -> tuple[dict[str, Any], ExperimentSpec | None, list[str], list[str], str]:
    config = load_config(options.config_path)
    experiment_spec: ExperimentSpec | None = None
    methods = options.methods or []
    datasets = options.datasets or []
    split = options.split
    if options.experiment:
        experiment_spec = get_experiment(options.experiment, options.experiments_config_path)
        methods = methods or experiment_spec.methods
        datasets = datasets or experiment_spec.datasets
        split = split or experiment_spec.split
        config = deep_merge(config, {"experiment": {"name": experiment_spec.name, "metrics": experiment_spec.metrics, **experiment_spec.extra}})
    if not methods:
        methods = ["single_tutor"]
    if not datasets:
        datasets = ["mathdial"]
    return config, experiment_spec, methods, datasets, split or "test"


def _output_paths(options: RunnerOptions, methods: list[str], datasets: list[str], split: str) -> dict[str, Path]:
    output_dir = Path(options.output_dir)
    run_id = options.run_id or f"{options.experiment or 'generation'}_{_slug(datasets)}_{split}_{_slug(methods)}_{_timestamp()}"
    if options.num_shards > 1 and options.run_id is None:
        run_id = f"{run_id}_shard{options.shard_index:03d}-of-{options.num_shards:03d}"
    return {
        "generations": output_dir / "generations" / f"{run_id}.jsonl",
        "errors": output_dir / "logs" / f"generation_errors_{run_id}.jsonl",
        "manifest": output_dir / "logs" / f"experiment_manifest_{run_id}.json",
    }


def _generation_config(config: dict[str, Any]) -> dict[str, Any]:
    generation = config.get("generation", {})
    model = config.get("model", {})
    return {
        "temperature": generation.get("temperature"),
        "top_p": generation.get("top_p"),
        "top_k": generation.get("top_k"),
        "max_tokens": generation.get("max_tokens"),
        "timeout_seconds": generation.get("timeout_seconds"),
        "retries": generation.get("retries"),
        "enable_thinking": bool(model.get("enable_thinking", False)),
    }


def _max_tokens_from_config(config: dict[str, Any]) -> int:
    value = config.get("generation", {}).get("max_tokens", 1024)
    if isinstance(value, dict):
        candidates = [int(item) for item in value.values() if isinstance(item, int | float)]
        return max(candidates) if candidates else 1024
    return int(value)


def _llm_client_from_config(config: dict[str, Any]) -> BaseLLMClient:
    model_config = config.get("model", {})
    generation = config.get("generation", {})
    endpoints = [
        LLMEndpointConfig(
            base_url=str(item["base_url"]),
            model=str(item.get("model") or model_config.get("generator", "Qwen3-8B")),
            name=item.get("name") or item.get("model"),
            timeout_seconds=float(item.get("timeout_seconds", generation.get("timeout_seconds", 120))),
        )
        for item in model_config.get("endpoints", [])
    ]
    if not endpoints:
        raise ValueError("live_llm requires model.endpoints in the run config")
    return BaseLLMClient(
        LLMClientConfig(
            endpoints=endpoints,
            model_name=str(model_config.get("generator", "Qwen3-8B")),
            temperature=float(generation.get("temperature", 0.2)),
            top_p=float(generation.get("top_p", 0.9)),
            top_k=int(generation.get("top_k", 20)),
            max_tokens=_max_tokens_from_config(config),
            timeout_s=float(generation.get("timeout_seconds", 120)),
            retries=int(generation.get("retries", 0)),
            mock_mode=False,
        )
    )


CONTROL_ONLY_AGENTS = {
    "risk_estimator",
    "qos_router",
    "budget_controller",
    "generic_router",
    "selected_agents",
    "difficulty_router",
    "oracle_router",
    "random_router",
    "generic_controller",
}


AGENT_ALIASES = {
    "tutor": "hint",
    "critic": "verifier",
    "solver_a": "solver",
    "solver_b": "solver",
    "solver_c": "solver",
    "judge": "verifier",
    "state": "state_manager",
    "state_proposer": "state_manager",
    "shared_memory": "state_manager",
    "single_writer": "state_manager",
}


def _callable_agent_name(selected_agent: str) -> str | None:
    if selected_agent in CONTROL_ONLY_AGENTS:
        return None
    return AGENT_ALIASES.get(selected_agent, selected_agent)


def _state_to_method_result(state: TutorGraphState, *, method: MethodSpec) -> dict[str, Any]:
    calls = state.llm_calls
    final_call = next((call for call in reversed(calls) if call.get("agent_name") == "final_tutor"), None)
    raw_completion = str(final_call.get("raw_completion", "")) if final_call else ""
    parsed_final = final_call.get("parsed_output") if final_call else None
    if isinstance(parsed_final, dict) and parsed_final.get("response"):
        final_response = str(parsed_final["response"])
    else:
        final_response = strip_think_blocks(raw_completion)

    token_usage = {
        "prompt_tokens": sum(int(call.get("usage", {}).get("prompt_tokens", 0)) for call in calls),
        "completion_tokens": sum(int(call.get("usage", {}).get("completion_tokens", 0)) for call in calls),
        "total_tokens": sum(int(call.get("usage", {}).get("total_tokens", 0)) for call in calls),
        "source": "api" if any(call.get("usage", {}).get("source") == "api" for call in calls) else "estimated",
    }
    call_errors = [
        {"agent_name": call.get("agent_name"), **call["error"]}
        for call in calls
        if isinstance(call.get("error"), dict)
    ]
    state_errors = [error.model_dump(mode="json") for error in state.errors]
    return {
        "selected_agents": state.selected_agents or list(method.selected_agents),
        "rounds": max(method.rounds, state.rounds),
        "risk_scores": state.risk_scores,
        "messages": [
            {"role": "agent", "agent_name": call.get("agent_name", ""), "content": call.get("stripped_output", "")}
            for call in calls
        ],
        "state": state.snapshot(),
        "token_usage": token_usage,
        "agent_calls": len(calls),
        "raw_completion": raw_completion,
        "final_response": final_response,
        "parse_success": bool(calls) and all(bool(call.get("parse_success")) for call in calls),
        "errors": state_errors + call_errors,
    }


def _run_live_prism(sample: dict[str, Any], method: MethodSpec, client: BaseLLMClient) -> dict[str, Any]:
    method_map = {"ours_routing": "M1", "ours_routing_budget": "M2", "ours_full": "M3"}
    graph_method = method_map.get(method.name, "M3")
    graph = build_prism_graph(method=graph_method, client=client)
    state = graph.invoke(TutorGraphState(sample=sample, method=method.name))
    return _state_to_method_result(state, method=method)


def _run_live_baseline(sample: dict[str, Any], method: MethodSpec, client: BaseLLMClient) -> dict[str, Any]:
    state = TutorGraphState(sample=sample, method=method.name, rounds=method.rounds)
    state.selected_agents.extend(method.selected_agents)
    for selected_agent in method.selected_agents:
        agent_name = _callable_agent_name(selected_agent)
        if agent_name is None:
            state.agent_outputs.setdefault(selected_agent, []).append({"control_only": True})
            continue
        agent = AGENT_REGISTRY.get(agent_name)
        if agent is None:
            state.agent_outputs.setdefault(selected_agent, []).append({"skipped_unknown_agent": True})
            continue
        record = agent.invoke(
            sample=sample,
            state={
                "method": method.name,
                "selected_agent": selected_agent,
                "agent_outputs": state.agent_outputs,
                "total_tokens": state.total_tokens,
            },
            client=client,
            method=method.name,
        )
        state.add_call(record)
    return _state_to_method_result(state, method=method)


def _run_live_method(sample: dict[str, Any], method: MethodSpec, client: BaseLLMClient) -> dict[str, Any]:
    if method.name.startswith("ours_") or method.family == "ablation":
        return _run_live_prism(sample, method, client)
    return _run_live_baseline(sample, method, client)


def _success_record(
    *,
    sample: dict[str, Any],
    method: MethodSpec,
    method_result: dict[str, Any],
    config: dict[str, Any],
    endpoint: Any,
    latency_seconds: float,
) -> dict[str, Any]:
    raw_completion = str(method_result.get("raw_completion", method_result.get("final_response", "")))
    final_response = str(method_result.get("final_response", strip_think_blocks(raw_completion)))
    token_usage = method_result.get("token_usage") or {
        "prompt_tokens": len(str(sample.get("problem", "")).split()),
        "completion_tokens": len(raw_completion.split()),
        "total_tokens": len(str(sample.get("problem", "")).split()) + len(raw_completion.split()),
        "source": "dry_run_estimate",
    }
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "success",
        "sample_id": sample["sample_id"],
        "dataset": sample["dataset"],
        "split": sample["split"],
        "method": method.name,
        "method_family": method.family,
        "base_model": config.get("model", {}).get("generator"),
        "endpoint": endpoint.as_dict(),
        "generation_config": _generation_config(config),
        "selected_agents": method_result.get("selected_agents", list(method.selected_agents)),
        "rounds": method_result.get("rounds", method.rounds),
        "risk_scores": method_result.get("risk_scores", {}),
        "messages": method_result.get("messages", []),
        "state": method_result.get("state", {}),
        "token_usage": token_usage,
        "agent_calls": int(method_result.get("agent_calls", len(method_result.get("selected_agents", [])))),
        "latency_seconds": latency_seconds,
        "raw_completion": raw_completion,
        "stripped_completion": strip_think_blocks(raw_completion),
        "final_response": final_response,
        "parse_success": bool(method_result.get("parse_success", True)),
        "errors": method_result.get("errors", []),
    }


def _failure_record(
    *,
    sample: dict[str, Any],
    method: MethodSpec,
    config: dict[str, Any],
    endpoint: Any | None,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
        "sample_id": sample.get("sample_id"),
        "dataset": sample.get("dataset"),
        "split": sample.get("split"),
        "method": method.name,
        "method_family": method.family,
        "base_model": config.get("model", {}).get("generator"),
        "endpoint": endpoint.as_dict() if endpoint else None,
        "generation_config": _generation_config(config),
        "selected_agents": [],
        "rounds": 0,
        "risk_scores": {},
        "messages": [],
        "state": {},
        "token_usage": {},
        "agent_calls": 0,
        "latency_seconds": 0.0,
        "raw_completion": "",
        "stripped_completion": "",
        "final_response": "",
        "parse_success": False,
        "errors": [{"type": type(exc).__name__, "message": str(exc)}],
    }


def run_generation(options: RunnerOptions) -> dict[str, Any]:
    _validate_shard_options(options.num_shards, options.shard_index)
    config, experiment_spec, method_names, datasets, split = _resolve_run_plan(options)
    registry = default_method_registry()
    methods = registry.resolve(method_names)
    endpoint_registry = EndpointRegistry.from_config(config)
    llm_client = _llm_client_from_config(config) if options.live_llm else None
    paths = _output_paths(options, method_names, datasets, split)

    generation_log = JsonlLogger(paths["generations"])
    error_log = JsonlLogger(paths["errors"])
    completed = generation_log.completed_keys() if options.resume else set()

    attempted = 0
    succeeded = 0
    failed = 0
    skipped = 0
    sample_counts: dict[str, int] = {}

    try:
        for dataset in datasets:
            samples = load_samples(
                dataset,
                split,
                limit=options.limit,
                num_shards=options.num_shards,
                shard_index=options.shard_index,
            )
            sample_counts[dataset] = len(samples)
            for sample_index, sample in enumerate(samples):
                for method in methods:
                    key = (str(sample["sample_id"]), dataset, split, method.name)
                    if key in completed:
                        skipped += 1
                        continue
                    endpoint = endpoint_registry.select_by_index(sample_index)
                    attempted += 1
                    try:
                        started = time.perf_counter()
                        if llm_client is not None:
                            result = _run_live_method(sample, method, llm_client)
                        else:
                            result = method.run(sample, {"config": config, "endpoint": endpoint, "sample_index": sample_index})
                        record = _success_record(
                            sample=sample,
                            method=method,
                            method_result=result,
                            config=config,
                            endpoint=endpoint,
                            latency_seconds=time.perf_counter() - started,
                        )
                        generation_log.append(record)
                        succeeded += 1
                    except Exception as exc:
                        record = _failure_record(sample=sample, method=method, config=config, endpoint=endpoint, exc=exc)
                        generation_log.append(record)
                        error_log.append(record)
                        failed += 1
        status = "completed" if failed == 0 else "completed_with_failures"
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        run_manifest = {
            "experiment": experiment_spec.name if experiment_spec else options.experiment,
            "datasets": datasets,
            "split": split,
            "methods": method_names,
            "limit": options.limit,
            "resume": options.resume,
            "live_llm": options.live_llm,
            "num_shards": options.num_shards,
            "shard_index": options.shard_index,
            "sample_counts": sample_counts,
            "paths": {key: str(value) for key, value in paths.items()},
            "counts": {
                "attempted": attempted,
                "succeeded": succeeded,
                "failed": failed,
                "skipped": skipped,
            },
        }
        write_experiment_manifest(path=paths["manifest"], config=config, run=run_manifest, status=locals().get("status", "interrupted"))

    return {
        "status": status,
        "paths": {key: str(value) for key, value in paths.items()},
        "counts": {"attempted": attempted, "succeeded": succeeded, "failed": failed, "skipped": skipped},
    }
