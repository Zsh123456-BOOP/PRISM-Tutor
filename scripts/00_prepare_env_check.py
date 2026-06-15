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
from prism_tutor.utils.env import check_conda_env, check_python, run_nvidia_smi
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


def validate_transformers_version(versions: dict[str, str | None]) -> dict[str, Any]:
    version = versions.get("transformers")
    if version is None:
        return {"ok": False, "version": None, "error": "transformers is not installed"}
    parts = tuple(int(part) for part in version.split(".")[:3] if part.isdigit())
    ok = parts >= (4, 51, 0)
    return {"ok": ok, "version": version, "error": None if ok else "transformers>=4.51.0 required for Qwen3"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="outputs/logs/env_check.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    snapshot_path = output.parent / "config_snapshot_env_check.yaml"
    write_yaml_snapshot(config, snapshot_path)

    versions = package_versions()
    gpu_summary = run_nvidia_smi()
    expected_gpu_count = int(config.get("cuda", {}).get("expected_gpu_count", 2))
    gpu_count_ok = len(gpu_summary.get("gpus", [])) >= expected_gpu_count if gpu_summary.get("available") else False

    report: dict[str, Any] = {
        "status": "ok",
        "dry_run": args.dry_run,
        "config": str(Path(args.config)),
        "config_snapshot": str(snapshot_path),
        "checks": {
            "python": check_python().to_dict(),
            "conda_env": check_conda_env().to_dict(),
            "imports_required": check_imports(REQUIRED_IMPORTS),
            "imports_serving": check_imports(OPTIONAL_SERVING_IMPORTS),
            "transformers_version": validate_transformers_version(versions),
            "gpu_count": {
                "ok": gpu_count_ok,
                "expected": expected_gpu_count,
                "detected": len(gpu_summary.get("gpus", [])),
            },
        },
        "gpu_summary": gpu_summary,
        "package_versions": versions,
        "reproducibility": collect_reproducibility_metadata(str(snapshot_path)),
    }

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
