from __future__ import annotations

import os
import re
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
    cuda_version = None
    try:
        banner = subprocess.run(["nvidia-smi"], check=True, capture_output=True, text=True)
        match = re.search(r"CUDA Version:\s*([0-9.]+)", banner.stdout)
        cuda_version = match.group(1) if match else None
    except Exception:
        cuda_version = None

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
        return {"available": False, "error": "nvidia-smi not found", "cuda_version": cuda_version, "gpus": []}
    except subprocess.CalledProcessError as exc:
        return {"available": False, "error": exc.stderr.strip(), "cuda_version": cuda_version, "gpus": []}

    gpus = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            gpus.append({"index": parts[0], "name": parts[1], "memory_total_mb": int(float(parts[2]))})
    return {"available": True, "error": None, "cuda_version": cuda_version, "gpus": gpus}


def check_cuda_visible_devices(expected_devices: str | None) -> CheckResult:
    actual = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not expected_devices:
        return CheckResult("cuda_visible_devices", True, f"CUDA_VISIBLE_DEVICES={actual!r}; no preferred devices configured", "info")
    if actual is None:
        return CheckResult(
            "cuda_visible_devices",
            False,
            f"CUDA_VISIBLE_DEVICES is not set; expected {expected_devices}",
            "warning",
        )
    expected = ",".join(part.strip() for part in expected_devices.split(",") if part.strip())
    normalized = ",".join(part.strip() for part in actual.split(",") if part.strip())
    if normalized != expected:
        return CheckResult(
            "cuda_visible_devices",
            False,
            f"CUDA_VISIBLE_DEVICES={actual!r}; expected {expected!r}",
            "warning",
        )
    return CheckResult("cuda_visible_devices", True, f"CUDA_VISIBLE_DEVICES={actual!r}", "info")


def check_python(min_major: int = 3, min_minor: int = 11) -> CheckResult:
    version = sys.version_info
    ok = (version.major, version.minor) >= (min_major, min_minor)
    return CheckResult("python", ok, sys.version)
