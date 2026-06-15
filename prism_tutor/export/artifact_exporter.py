"""Export paper-facing artifact manifests and summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
) -> dict[str, Path]:
    root_path = Path(root)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    required_paths = required_paths or default_required_paths(artifact_prefix)
    manifest = build_experiment_manifest(experiment_manifests or [], root_path)
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


def build_experiment_manifest(manifests: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    by_exp = {str(item.get("experiment") or item.get("name")): item for item in manifests if item}
    expected = [f"exp{i}" for i in range(7)]
    return {
        "experiments": by_exp,
        "expected_experiments": expected,
        "missing_experiments": [exp for exp in expected if exp not in by_exp],
        "root": str(root),
    }


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
