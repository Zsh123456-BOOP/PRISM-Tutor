"""Export paper-facing artifact manifests and summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from prism_tutor.utils.config import load_yaml

from .reproducibility_checklist import build_reproducibility_checklist, checklist_to_markdown


DEFAULT_ARTIFACT_PREFIX = "outputs"


def default_required_paths(artifact_prefix: str = DEFAULT_ARTIFACT_PREFIX) -> list[str]:
    prefix = artifact_prefix.rstrip("/")
    return [
        f"{prefix}/logs",
        f"{prefix}/generations",
        f"{prefix}/judge_scores/judge_metadata.json",
        f"{prefix}/metrics",
        f"{prefix}/tables",
        f"{prefix}/figures",
        f"{prefix}/human_audit/human_agreement_report.json",
    ]


def export_paper_artifacts(
    root: str | Path,
    output_dir: str | Path,
    experiment_manifests: list[dict[str, Any]] | None = None,
    required_paths: list[str] | None = None,
    artifact_prefix: str = DEFAULT_ARTIFACT_PREFIX,
    shard_plan: dict[str, Any] | None = None,
) -> dict[str, Path]:
    root_path = Path(root)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    required_paths = required_paths or default_required_paths(artifact_prefix)
    manifest = build_experiment_manifest(experiment_manifests or [], root_path, shard_plan=shard_plan)
    checklist = build_reproducibility_checklist(root_path, required_paths, metadata={"inference_time_runtime": True})
    index = build_artifact_index(root_path, artifact_prefix=artifact_prefix)
    summary = build_experiment_summary(manifest, checklist, index)

    files = {
        "experiment_manifest": out / "experiment_manifest.json",
        "reproducibility_checklist": out / "reproducibility_checklist.md",
        "artifact_index": out / "artifact_index.md",
        "experiment_summary": out / "experiment_summary.md",
    }
    files["experiment_manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files["reproducibility_checklist"].write_text(checklist_to_markdown(checklist), encoding="utf-8")
    files["artifact_index"].write_text(index, encoding="utf-8")
    files["experiment_summary"].write_text(summary, encoding="utf-8")
    return files


def build_experiment_manifest(
    manifests: list[dict[str, Any]],
    root: Path,
    shard_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    plan_experiments = _experiments_from_shard_plan(root, shard_plan) if shard_plan else {}
    by_exp = {}
    for item in manifests:
        if not item:
            continue
        exp_name = item.get("experiment") or item.get("name")
        if exp_name:
            by_exp[str(exp_name)] = item
    by_exp = {**plan_experiments, **by_exp}
    expected = list(plan_experiments) if plan_experiments else [f"exp{i}" for i in range(7)]
    return {
        "experiments": by_exp,
        "expected_experiments": expected,
        "missing_experiments": [exp for exp in expected if exp not in by_exp],
        "root": str(root),
    }


def _experiments_from_shard_plan(root: Path, shard_plan: dict[str, Any]) -> dict[str, Any]:
    jobs = [job for job in shard_plan.get("jobs", []) if isinstance(job, dict)]
    config = _load_experiments_config(root, shard_plan.get("experiments_config"))
    output: dict[str, Any] = {}
    for job in jobs:
        name = str(job.get("experiment") or "")
        if not name:
            continue
        item = output.setdefault(
            name,
            {
                "experiment": name,
                "datasets": sorted(job.get("datasets") or []),
                "split": job.get("split"),
                "methods": config.get(name, {}).get("methods", []),
                "output_dir": shard_plan.get("output_dir"),
                "split_dir": shard_plan.get("split_dir"),
                "live_llm": shard_plan.get("live_llm"),
                "job_count": 0,
                "estimated_records": 0,
                "shards": [],
                "output_paths": {"generations": [], "manifests": [], "errors": []},
            },
        )
        item["job_count"] += 1
        item["estimated_records"] += int(job.get("estimated_records") or 0)
        item["shards"].append({"job_id": job.get("job_id"), "shard_index": job.get("shard_index"), "num_shards": job.get("num_shards")})
        paths = job.get("paths") if isinstance(job.get("paths"), dict) else {}
        for key, out_key in [("generations", "generations"), ("manifest", "manifests"), ("errors", "errors")]:
            if paths.get(key):
                item["output_paths"][out_key].append(paths[key])
    for item in output.values():
        for values in item["output_paths"].values():
            values.sort()
        item["shards"].sort(key=lambda row: (row.get("num_shards") or 0, row.get("shard_index") or 0, str(row.get("job_id"))))
        if not item.get("datasets"):
            item["datasets"] = config.get(item["experiment"], {}).get("datasets", [])
        if not item.get("split"):
            item["split"] = config.get(item["experiment"], {}).get("split")
    return dict(sorted(output.items()))


def _load_experiments_config(root: Path, config_path: Any) -> dict[str, Any]:
    if not config_path:
        return {}
    path = root / str(config_path)
    if not path.exists():
        return {}
    try:
        raw = load_yaml(path)
    except Exception:
        return {}
    experiments = raw.get("experiments") if isinstance(raw.get("experiments"), dict) else {}
    return experiments


def build_artifact_index(root: Path, artifact_prefix: str = DEFAULT_ARTIFACT_PREFIX) -> str:
    prefix = artifact_prefix.rstrip("/")
    entries = [
        (f"{prefix}/tables", "scripts/05_make_tables.py", f"{prefix}/metrics/*.csv and judge scores"),
        (f"{prefix}/figures", "scripts/06_make_figures.py", f"{prefix}/metrics/*.csv and experiment manifest"),
        (f"{prefix}/metrics/significance_tests.json", "prism_tutor.eval.significance", "paired metric rows"),
        (f"{prefix}/human_audit", "scripts/07_sample_human_audit.py and scripts/08_human_agreement.py", "judge, metrics, tables"),
    ]
    lines = ["# Artifact Index", ""]
    for rel, source_script, inputs in entries:
        status = "present" if (root / rel).exists() else "missing"
        lines.append(f"- `{rel}`: {status}; source `{source_script}`; inputs {inputs}")
    return "\n".join(lines) + "\n"


def build_experiment_summary(manifest: dict[str, Any], checklist: dict[str, Any], artifact_index: str) -> str:
    lines = [
        "# Experiment Summary",
        "",
        "PRISM-Tutor is evaluated as an inference-time runtime; no model training artifacts are expected.",
        "",
        f"Checklist status: **{checklist.get('status')}**",
        f"Missing experiments: {', '.join(manifest.get('missing_experiments', [])) or 'none'}",
        "",
        "## Artifact Traceability",
        "",
        artifact_index,
    ]
    return "\n".join(lines)
