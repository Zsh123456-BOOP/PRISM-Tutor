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


def _path_summary(root: Path, rel_paths: list[str]) -> list[dict[str, Any]]:
    summary = []
    for rel in rel_paths:
        path = root / rel
        kind = "directory" if path.is_dir() else "file" if path.is_file() else "missing"
        summary.append({"path": rel, "status": "passed" if path.exists() else "failed", "kind": kind})
    return summary


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
        if not path.exists() or not path.is_file() or path.stat().st_size > 1_000_000:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for pattern in SECRET_PATTERNS:
            if pattern in text:
                hits.append({"path": rel, "pattern": pattern})
    return hits


def _run(cmd: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.check_output(cmd, cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None
