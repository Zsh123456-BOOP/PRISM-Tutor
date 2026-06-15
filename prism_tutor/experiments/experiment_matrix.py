from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prism_tutor.utils.config import load_yaml


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    datasets: list[str]
    split: str
    methods: list[str]
    metrics: list[str]
    output_table: str
    seeds: list[int]
    extra: dict[str, Any]


DEFAULT_METRICS = ["quality", "token_usage", "latency", "failure_rate"]


def _with_defaults(name: str, raw: dict[str, Any]) -> ExperimentSpec:
    return ExperimentSpec(
        name=name,
        datasets=list(raw.get("datasets", [])),
        split=str(raw.get("split", "test")),
        methods=list(raw.get("methods", [])),
        metrics=list(raw.get("metrics", DEFAULT_METRICS)),
        output_table=str(raw.get("output_table", f"outputs/tables/{name}.csv")),
        seeds=list(raw.get("seeds", [42])),
        extra={key: value for key, value in raw.items() if key not in {"datasets", "split", "methods", "metrics", "output_table", "seeds"}},
    )


def load_experiment_matrix(path: str | Path = "configs/experiments.yaml") -> dict[str, ExperimentSpec]:
    raw = load_yaml(path).get("experiments", {})
    if not isinstance(raw, dict):
        raise ValueError("experiments config must contain an experiments mapping")
    return {name: _with_defaults(name, value or {}) for name, value in raw.items()}


def get_experiment(name: str, path: str | Path = "configs/experiments.yaml") -> ExperimentSpec:
    matrix = load_experiment_matrix(path)
    try:
        return matrix[name]
    except KeyError as exc:
        raise KeyError(f"Unknown experiment: {name}") from exc
