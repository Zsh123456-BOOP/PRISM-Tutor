from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    severity: str = "error"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def current_conda_env() -> str | None:
    return os.environ.get("CONDA_DEFAULT_ENV")


def check_conda_env() -> CheckResult:
    env = current_conda_env()
    if not env:
        return CheckResult("conda_env", False, "CONDA_DEFAULT_ENV is not set", "warning")
    if env == "base":
        return CheckResult("conda_env", False, "base conda environment is not allowed")
    return CheckResult("conda_env", True, env, "info")


def run_nvidia_smi() -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return {"available": False, "error": "nvidia-smi not found", "gpus": []}
    except subprocess.CalledProcessError as exc:
        return {"available": False, "error": exc.stderr.strip(), "gpus": []}

    gpus = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            gpus.append({"index": parts[0], "name": parts[1], "memory_total_mb": int(float(parts[2]))})
    return {"available": True, "error": None, "gpus": gpus}


def check_python(min_major: int = 3, min_minor: int = 11) -> CheckResult:
    version = sys.version_info
    ok = (version.major, version.minor) >= (min_major, min_minor)
    return CheckResult("python", ok, sys.version)
