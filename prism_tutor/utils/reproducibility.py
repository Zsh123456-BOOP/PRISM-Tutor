from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CORE_PACKAGES = [
    "pydantic",
    "pyyaml",
    "pandas",
    "numpy",
    "scipy",
    "scikit-learn",
    "jsonlines",
    "openai",
    "matplotlib",
    "transformers",
    "vllm",
    "langgraph",
    "modelscope",
]


def _run_git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(["git", *args], check=True, capture_output=True, text=True)
    except Exception:
        return None
    return result.stdout.strip()


def git_metadata() -> dict[str, Any]:
    status = _run_git(["status", "--short"])
    return {
        "commit": _run_git(["rev-parse", "HEAD"]),
        "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status),
        "status_short": status.splitlines() if status else [],
    }


def package_versions(packages: list[str] | None = None) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in packages or CORE_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def collect_reproducibility_metadata(config_snapshot_path: str | None = None) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "git": git_metadata(),
        "package_versions": package_versions(),
        "config_snapshot_path": config_snapshot_path,
    }


def write_json(data: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
