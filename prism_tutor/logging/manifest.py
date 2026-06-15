from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from prism_tutor.utils.config import write_yaml_snapshot
from prism_tutor.utils.reproducibility import collect_reproducibility_metadata


MANIFEST_SCHEMA_VERSION = "0.2.0"


class ManifestAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment: str | None = None
    datasets: list[str] = Field(default_factory=list)
    split: str | None = None
    methods: list[str] = Field(default_factory=list)
    model: str | None = None
    generation_config: dict[str, Any] = Field(default_factory=dict)
    input_paths: dict[str, Any] = Field(default_factory=dict)
    output_paths: dict[str, Any] = Field(default_factory=dict)
    started_at_utc: str | None = None
    finished_at_utc: str
    duration_seconds: float | None = None


class ExperimentManifest(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: str
    manifest_type: str = "experiment_run"
    created_at_utc: str
    status: str
    run: dict[str, Any]
    audit: ManifestAudit
    reproducibility: dict[str, Any]


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


def _audit_record(config: dict[str, Any], run: dict[str, Any], *, finished_at_utc: str) -> ManifestAudit:
    paths = run.get("paths", {}) if isinstance(run.get("paths"), dict) else {}
    started_at = run.get("started_at_utc")
    duration = run.get("duration_seconds")
    return ManifestAudit(
        experiment=run.get("experiment"),
        datasets=list(run.get("datasets") or []),
        split=run.get("split"),
        methods=list(run.get("methods") or []),
        model=config.get("model", {}).get("generator"),
        generation_config=_generation_config(config),
        input_paths={"data_splits": config.get("paths", {}).get("data_splits"), "config": run.get("config_path")},
        output_paths={key: str(value) for key, value in paths.items()},
        started_at_utc=started_at,
        finished_at_utc=finished_at_utc,
        duration_seconds=duration,
    )


def write_experiment_manifest(
    *,
    path: str | Path,
    config: dict[str, Any],
    run: dict[str, Any],
    status: str,
) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    config_snapshot = target.with_suffix(".config.yaml")
    write_yaml_snapshot(config, config_snapshot)
    created_at = datetime.now(timezone.utc).isoformat()
    manifest = ExperimentManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        created_at_utc=created_at,
        status=status,
        run=run,
        audit=_audit_record(config, run, finished_at_utc=created_at),
        reproducibility=collect_reproducibility_metadata(str(config_snapshot)),
    ).model_dump(mode="json")
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
