"""Reproducibility checklist generation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from prism_tutor.utils.config import load_yaml
from prism_tutor.utils.env import run_nvidia_smi
from prism_tutor.utils.reproducibility import package_versions

SECRET_PATTERNS = ["api_key", "apikey", "secret", "password", "bearer "]
DEFAULT_CONFIG_PATH = "configs/default.yaml"
REQUIRED_JUDGE_METADATA_FIELDS = [
    "actual_model",
    "api_date",
    "temperature",
    "top_p",
    "max_tokens",
    "prompt_version",
    "dry_run",
    "parsed_count",
    "error_count",
    "output_rows",
]


def build_reproducibility_checklist(
    root: str | Path,
    required_paths: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    config_path: str = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    root_path = Path(root)
    required_paths = required_paths or []
    checks = []
    for rel in required_paths:
        path = root_path / rel
        checks.append({"name": rel, "status": "passed" if path.exists() else "failed", "path": rel})

    config = _load_config(root_path / config_path)
    config_summary = _config_summary(config, config_path)
    checks.extend(_config_checks(config_summary))

    git_commit = _run(["git", "rev-parse", "HEAD"], root_path)
    git_dirty = _run(["git", "status", "--short"], root_path)
    versions = {"python": sys.version.split()[0], **package_versions()}
    gpu_summary = run_nvidia_smi()
    judge_metadata = _load_optional_json(root_path / _judge_metadata_path(required_paths))
    checks.extend(_judge_checks(judge_metadata, required_paths))
    checks.extend(_content_checks(root_path, required_paths))
    secret_scan = scan_for_plaintext_secrets(root_path, required_paths)
    checks.append({"name": "plaintext_secret_scan", "status": "failed" if secret_scan else "passed", "hits": secret_scan})

    return {
        "status": "passed" if all(check["status"] == "passed" for check in checks) else "failed",
        "checks": checks,
        "git": {"commit": git_commit, "dirty_status": git_dirty},
        "package_versions": versions,
        "config": config_summary,
        "gpu": gpu_summary,
        "model": config_summary.get("model", {}),
        "judge": {
            "metadata_path": _judge_metadata_path(required_paths),
            "metadata_present": judge_metadata is not None,
            "metadata": judge_metadata or {},
        },
        "data_and_logs": _path_summary(root_path, required_paths),
        "metadata": metadata or {},
    }


def checklist_to_markdown(checklist: dict[str, Any]) -> str:
    lines = [
        "# Reproducibility Checklist",
        "",
        f"Overall status: **{checklist.get('status')}**",
        "",
        "## Config",
        "",
        f"- Config snapshot: `{checklist.get('config', {}).get('path') or 'unknown'}`",
        f"- Seed: `{checklist.get('config', {}).get('seed') if checklist.get('config', {}).get('seed') is not None else 'missing'}`",
        f"- Model: `{checklist.get('model', {}).get('generator') or 'missing'}`",
        f"- Enable thinking: `{checklist.get('model', {}).get('enable_thinking')}`",
        "",
        "## Git",
        "",
        f"- Commit: `{checklist.get('git', {}).get('commit') or 'unknown'}`",
        f"- Dirty status: `{checklist.get('git', {}).get('dirty_status') or 'clean'}`",
        "",
        "## GPU",
        "",
        f"- Available: `{checklist.get('gpu', {}).get('available')}`",
    ]
    for gpu in checklist.get("gpu", {}).get("gpus", []):
        lines.append(f"- GPU {gpu.get('index')}: {gpu.get('name')} ({gpu.get('memory_total_mb')} MB)")
    lines.extend(
        [
            "",
            "## Judge",
            "",
            f"- Metadata path: `{checklist.get('judge', {}).get('metadata_path')}`",
            f"- Metadata present: `{checklist.get('judge', {}).get('metadata_present')}`",
            "",
            "## Data And Logs",
            "",
        ]
    )
    for item in checklist.get("data_and_logs", []):
        lines.append(f"- {item.get('path')}: {item.get('status')} ({item.get('kind')})")
    lines.extend(
        [
            "",
            "## Checks",
            "",
        ]
    )
    for check in checklist.get("checks", []):
        lines.append(f"- {check.get('name')}: {check.get('status')}")
    lines.extend(["", "## Package Versions", ""])
    for name, version in checklist.get("package_versions", {}).items():
        lines.append(f"- {name}: {version}")
    return "\n".join(lines) + "\n"


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return load_yaml(path)
    except Exception:
        return {}


def _config_summary(config: dict[str, Any], config_path: str) -> dict[str, Any]:
    model = config.get("model") if isinstance(config.get("model"), dict) else {}
    generation = config.get("generation") if isinstance(config.get("generation"), dict) else {}
    cuda = config.get("cuda") if isinstance(config.get("cuda"), dict) else {}
    return {
        "path": config_path,
        "seed": config.get("seed"),
        "model": model,
        "generation": generation,
        "cuda": cuda,
    }


def _config_checks(config: dict[str, Any]) -> list[dict[str, Any]]:
    checks = [
        {"name": "config_snapshot", "status": "passed" if config.get("seed") is not None else "failed", "path": config.get("path")},
        {"name": "seed_fixed", "status": "passed" if config.get("seed") == 42 else "failed", "seed": config.get("seed")},
    ]
    model = config.get("model") if isinstance(config.get("model"), dict) else {}
    checks.extend(
        [
            {"name": "model_declared", "status": "passed" if model.get("generator") else "failed", "model": model.get("generator")},
            {
                "name": "enable_thinking_disabled",
                "status": "passed" if model.get("enable_thinking") is False else "failed",
                "enable_thinking": model.get("enable_thinking"),
            },
            {
                "name": "no_main_quantization",
                "status": "passed" if model.get("quantization") in (None, "null") else "failed",
                "quantization": model.get("quantization"),
            },
        ]
    )
    return checks


def _judge_checks(judge_metadata: dict[str, Any] | None, required_paths: list[str]) -> list[dict[str, Any]]:
    if not any("judge_scores" in rel or rel.endswith("judge_metadata.json") for rel in required_paths):
        return []
    if judge_metadata is None:
        return [{"name": "judge_metadata_schema", "status": "failed", "missing_fields": REQUIRED_JUDGE_METADATA_FIELDS}]
    missing = [field for field in REQUIRED_JUDGE_METADATA_FIELDS if judge_metadata.get(field) in (None, "")]
    dry_run = judge_metadata.get("dry_run")
    actual_model = str(judge_metadata.get("actual_model") or "")
    error_count = _as_int(judge_metadata.get("error_count"))
    parsed_count = _as_int(judge_metadata.get("parsed_count"))
    output_rows = _as_int(judge_metadata.get("output_rows"))
    invalid_reasons = []
    if dry_run is True:
        invalid_reasons.append("dry_run_true")
    if actual_model == "mock-judge" or actual_model.startswith("mock"):
        invalid_reasons.append("mock_actual_model")
    if error_count is not None and error_count != 0:
        invalid_reasons.append("judge_errors_present")
    if parsed_count is not None and output_rows is not None and parsed_count != output_rows:
        invalid_reasons.append("parsed_count_mismatch")
    return [
        {
            "name": "judge_metadata_schema",
            "status": "failed" if missing or invalid_reasons else "passed",
            "missing_fields": missing,
            "invalid_reasons": invalid_reasons,
        }
    ]


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _path_summary(root: Path, rel_paths: list[str]) -> list[dict[str, Any]]:
    summary = []
    for rel in rel_paths:
        path = root / rel
        kind = "directory" if path.is_dir() else "file" if path.is_file() else "missing"
        summary.append({"path": rel, "status": "passed" if path.exists() else "failed", "kind": kind})
    return summary


def _content_checks(root: Path, rel_paths: list[str]) -> list[dict[str, Any]]:
    checks = []
    for rel in rel_paths:
        path = root / rel
        if not path.exists() or not path.is_file():
            continue
        if rel.endswith("figure_manifest.json"):
            payload = _load_optional_json(path) or {}
            checks.append(
                {
                    "name": f"{rel}:content",
                    "status": "passed" if payload.get("status") == "completed" else "failed",
                    "expected_status": "completed",
                    "actual_status": payload.get("status"),
                }
            )
        elif rel.endswith("human_agreement_report.json"):
            payload = _load_optional_json(path) or {}
            checks.append(
                {
                    "name": f"{rel}:content",
                    "status": "passed" if payload.get("status") == "passed" else "failed",
                    "expected_status": "passed",
                    "actual_status": payload.get("status"),
                }
            )
        elif rel.endswith(".csv") and ("/tables/" in rel or rel.endswith("human_audit_blind.csv")):
            rows = _count_csv_rows(path)
            checks.append({"name": f"{rel}:content", "status": "passed" if rows > 0 else "failed", "rows": rows})
        elif rel.endswith("preference_mapping.json"):
            payload = _load_optional_json(path)
            rows = len(payload) if isinstance(payload, list) else 0
            checks.append({"name": f"{rel}:content", "status": "passed" if rows > 0 else "failed", "rows": rows})
        elif rel.endswith("sampling_manifest.json"):
            payload = _load_optional_json(path) or {}
            actual_n = _as_int(payload.get("actual_n")) or 0
            checks.append({"name": f"{rel}:content", "status": "passed" if actual_n > 0 else "failed", "actual_n": actual_n})
        elif rel.endswith("record_auto_metrics.jsonl") or rel.endswith("judge_raw.jsonl"):
            rows = _count_jsonl_rows(path)
            checks.append({"name": f"{rel}:content", "status": "passed" if rows > 0 else "failed", "rows": rows})
        elif rel.endswith("significance_tests.json"):
            payload = _load_optional_json(path)
            rows = len(payload) if isinstance(payload, list) else 0
            checks.append({"name": f"{rel}:content", "status": "passed" if rows > 0 else "failed", "rows": rows})
    return checks


def _count_csv_rows(path: Path) -> int:
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError:
        return 0
    return max(len(lines) - 1, 0)


def _count_jsonl_rows(path: Path) -> int:
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError:
        return 0


def _judge_metadata_path(required_paths: list[str]) -> str:
    for rel in required_paths:
        if rel.endswith("judge_metadata.json"):
            return rel
    return "outputs/judge_scores/judge_metadata.json"


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def scan_for_plaintext_secrets(root: Path, rel_paths: list[str]) -> list[dict[str, Any]]:
    hits = []
    for rel in rel_paths:
        path = root / rel
        for file_path in _iter_secret_scan_files(path):
            text = file_path.read_text(encoding="utf-8", errors="ignore").lower()
            rel_file = str(file_path.relative_to(root))
            for pattern in SECRET_PATTERNS:
                if pattern in text:
                    hits.append({"path": rel_file, "pattern": pattern})
    return hits


def _iter_secret_scan_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    if path.is_file():
        return [path] if path.stat().st_size <= 1_000_000 else []
    if not path.is_dir():
        return []
    files: list[Path] = []
    for child in sorted(path.rglob("*")):
        if child.is_file() and child.stat().st_size <= 1_000_000:
            files.append(child)
    return files


def _run(cmd: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.check_output(cmd, cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None
