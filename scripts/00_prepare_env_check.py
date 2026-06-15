#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prism_tutor.utils.config import load_config, write_yaml_snapshot
from prism_tutor.utils.env import check_conda_env, check_cuda_visible_devices, check_python, run_nvidia_smi
from prism_tutor.utils.reproducibility import collect_reproducibility_metadata, package_versions, write_json


REQUIRED_IMPORTS = [
    "pydantic",
    "yaml",
    "pandas",
    "numpy",
    "scipy",
    "sklearn",
    "jsonlines",
    "openai",
]

OPTIONAL_SERVING_IMPORTS = ["transformers", "vllm", "langgraph", "modelscope"]


INSTALL_HINTS = {
    "jsonlines": "Install Python dependencies with `pip install -r requirements.txt` inside the prism_tutor conda env.",
    "openai": "Install Python dependencies with `pip install -r requirements.txt`; do not put API keys in dependency files.",
    "transformers": "Install/upgrade with `pip install 'transformers>=4.51.0'` for Qwen3 compatibility.",
    "vllm": "Install vLLM inside the experiment env, matching the server CUDA/PyTorch stack; do not modify system CUDA.",
    "langgraph": "Install LangGraph with `pip install langgraph` for the preferred runtime backend.",
    "modelscope": "Install ModelScope with `pip install modelscope` and set VLLM_USE_MODELSCOPE=true for Qwen serving.",
}


def check_imports(module_names: list[str]) -> list[dict[str, Any]]:
    results = []
    for module_name in module_names:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            results.append({"module": module_name, "ok": False, "error": repr(exc)})
        else:
            results.append({"module": module_name, "ok": True, "error": None})
    return results


def import_suggestions(import_results: list[dict[str, Any]]) -> list[dict[str, str]]:
    suggestions = []
    for result in import_results:
        if result.get("ok"):
            continue
        module = str(result.get("module"))
        suggestions.append(
            {
                "module": module,
                "suggestion": INSTALL_HINTS.get(module, "Install the missing package in the prism_tutor conda env."),
            }
        )
    return suggestions


def gpu_warnings(gpu_summary: dict[str, Any], *, min_memory_mb: int = 24000) -> list[str]:
    warnings = []
    if not gpu_summary.get("available"):
        return ["nvidia-smi unavailable; full experiment is not allowed until GPU visibility is fixed"]
    for gpu in gpu_summary.get("gpus", []):
        memory = int(gpu.get("memory_total_mb") or 0)
        name = str(gpu.get("name") or "")
        index = str(gpu.get("index") or "unknown")
        if memory < min_memory_mb:
            warnings.append(f"GPU {index} has {memory} MB memory, below {min_memory_mb} MB full-run expectation")
        if "4090" not in name and memory >= min_memory_mb:
            warnings.append(f"GPU {index} is {name!r}, not RTX 4090; memory is sufficient but hardware differs")
    return warnings


def validate_transformers_version(versions: dict[str, str | None]) -> dict[str, Any]:
    version = versions.get("transformers")
    if version is None:
        return {"ok": False, "version": None, "error": "transformers is not installed"}
    parts = tuple(int(part) for part in version.split(".")[:3] if part.isdigit())
    ok = parts >= (4, 51, 0)
    return {"ok": ok, "version": version, "error": None if ok else "transformers>=4.51.0 required for Qwen3"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="outputs/logs/env_check.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    snapshot_path = output.parent / "config_snapshot_env_check.yaml"
    write_yaml_snapshot(config, snapshot_path)

    versions = package_versions()
    gpu_summary = run_nvidia_smi()
    expected_gpu_count = int(config.get("cuda", {}).get("expected_gpu_count", 2))
    preferred_devices = str(config.get("cuda", {}).get("preferred_devices") or "")
    gpu_count_ok = len(gpu_summary.get("gpus", [])) >= expected_gpu_count if gpu_summary.get("available") else False
    required_imports = check_imports(REQUIRED_IMPORTS)
    serving_imports = check_imports(OPTIONAL_SERVING_IMPORTS)
    cuda_visible = check_cuda_visible_devices(preferred_devices).to_dict()
    suggestions = import_suggestions(required_imports + serving_imports)
    warnings = gpu_warnings(gpu_summary)
    errors: list[str] = []

    report: dict[str, Any] = {
        "status": "ok",
        "dry_run": args.dry_run,
        "config": str(Path(args.config)),
        "config_snapshot": str(snapshot_path),
        "checks": {
            "python": check_python().to_dict(),
            "conda_env": check_conda_env().to_dict(),
            "imports_required": required_imports,
            "imports_serving": serving_imports,
            "transformers_version": validate_transformers_version(versions),
            "cuda_visible_devices": cuda_visible,
            "gpu_count": {
                "ok": gpu_count_ok,
                "expected": expected_gpu_count,
                "detected": len(gpu_summary.get("gpus", [])),
                "preferred_devices": preferred_devices,
            },
        },
        "gpu_summary": gpu_summary,
        "warnings": warnings,
        "errors": errors,
        "fallback_suggestions": suggestions,
        "package_versions": versions,
        "reproducibility": collect_reproducibility_metadata(str(snapshot_path)),
    }

    if not report["checks"]["python"]["ok"]:
        errors.append(report["checks"]["python"]["detail"])
    if report["checks"]["conda_env"]["severity"] == "error" and not report["checks"]["conda_env"]["ok"]:
        errors.append(report["checks"]["conda_env"]["detail"])
    if not report["checks"]["transformers_version"]["ok"]:
        errors.append(report["checks"]["transformers_version"]["error"])
    if not cuda_visible["ok"]:
        warnings.append(cuda_visible["detail"])
    if not gpu_count_ok:
        warnings.append(
            f"detected {len(gpu_summary.get('gpus', []))} GPUs, expected at least {expected_gpu_count}; full experiment should not start"
        )

    hard_failures = [
        not report["checks"]["python"]["ok"],
        report["checks"]["conda_env"]["severity"] == "error" and not report["checks"]["conda_env"]["ok"],
        not report["checks"]["transformers_version"]["ok"],
    ]
    if any(hard_failures):
        report["status"] = "failed"
    elif not gpu_count_ok:
        report["status"] = "degraded"

    write_json(report, output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if args.dry_run or report["status"] in {"ok", "degraded"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
