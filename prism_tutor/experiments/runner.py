from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prism_tutor.experiments.experiment_matrix import ExperimentSpec, get_experiment
from prism_tutor.experiments.method_registry import MethodSpec, default_method_registry
from prism_tutor.logging.jsonl_logger import JsonlLogger
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


def _slug(values: list[str]) -> str:
    return "-".join(values).replace("/", "_")


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_samples(dataset: str, split: str, *, limit: int | None = None) -> list[dict[str, Any]]:
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
                if limit is not None and len(samples) >= limit:
                    return samples
        return samples

    count = limit if limit is not None else 1
    return [
        {
            "sample_id": f"{dataset}:{split}:{index:06d}",
            "dataset": dataset,
            "split": split,
            "problem": f"Dry-run sample {index} for {dataset}/{split}.",
            "source": "synthetic_smoke_fallback",
        }
        for index in range(count)
    ]


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
    token_usage = {
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
        "latency_seconds": 0.0,
        "raw_completion": "",
        "stripped_completion": "",
        "final_response": "",
        "parse_success": False,
        "errors": [{"type": type(exc).__name__, "message": str(exc)}],
    }


def run_generation(options: RunnerOptions) -> dict[str, Any]:
    config, experiment_spec, method_names, datasets, split = _resolve_run_plan(options)
    registry = default_method_registry()
    methods = registry.resolve(method_names)
    endpoint_registry = EndpointRegistry.from_config(config)
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
            samples = load_samples(dataset, split, limit=options.limit)
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
