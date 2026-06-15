"""Reproducibility checklist generation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SECRET_PATTERNS = ["api_key", "apikey", "secret", "password", "bearer "]


def build_reproducibility_checklist(
    root: str | Path,
    required_paths: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root_path = Path(root)
    required_paths = required_paths or []
    checks = []
    for rel in required_paths:
        path = root_path / rel
        checks.append({"name": rel, "status": "passed" if path.exists() else "failed", "path": rel})

    git_commit = _run(["git", "rev-parse", "HEAD"], root_path)
    git_dirty = _run(["git", "status", "--short"], root_path)
    package_versions = {"python": sys.version.split()[0]}
    secret_scan = scan_for_plaintext_secrets(root_path, required_paths)
    checks.append({"name": "plaintext_secret_scan", "status": "failed" if secret_scan else "passed", "hits": secret_scan})

    return {
        "status": "passed" if all(check["status"] == "passed" for check in checks) else "failed",
        "checks": checks,
        "git": {"commit": git_commit, "dirty_status": git_dirty},
        "package_versions": package_versions,
        "metadata": metadata or {},
    }


def checklist_to_markdown(checklist: dict[str, Any]) -> str:
    lines = [
        "# Reproducibility Checklist",
        "",
        f"Overall status: **{checklist.get('status')}**",
        "",
        "## Git",
        "",
        f"- Commit: `{checklist.get('git', {}).get('commit') or 'unknown'}`",
        f"- Dirty status: `{checklist.get('git', {}).get('dirty_status') or 'clean'}`",
        "",
        "## Checks",
        "",
    ]
    for check in checklist.get("checks", []):
        lines.append(f"- {check.get('name')}: {check.get('status')}")
    lines.extend(["", "## Package Versions", ""])
    for name, version in checklist.get("package_versions", {}).items():
        lines.append(f"- {name}: {version}")
    return "\n".join(lines) + "\n"


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
