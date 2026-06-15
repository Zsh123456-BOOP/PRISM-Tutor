from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prism_tutor.utils.config import load_yaml


def _profile_value(config: dict[str, Any], key: str, profile: str, default: Any = None) -> Any:
    value = config.get(key, default)
    if isinstance(value, dict):
        return value.get(profile, default)
    return value


def build_vllm_command(config: dict[str, Any], profile: str) -> tuple[dict[str, str], list[str]]:
    if profile not in {"gpu0", "gpu1", "tp2"}:
        raise ValueError(f"Unknown vLLM profile: {profile}")

    model_name = str(config.get("generator", "Qwen/Qwen3-8B"))
    served_model_name = str(_profile_value(config, "served_model_names", profile, "qwen3-8b"))
    host = str(config.get("host", "0.0.0.0"))
    port = str(_profile_value(config, "ports", profile, 8000))
    dtype = str(config.get("dtype", "bfloat16"))
    use_modelscope = bool(config.get("use_modelscope", True))
    gpu_memory_utilization = str(config.get("gpu_memory_utilization", 0.88))

    max_model_len_key = "tensor_parallel" if profile == "tp2" else "single_gpu"
    max_model_len = str(_profile_value(config, "max_model_len", max_model_len_key, 8192))
    devices = str(_profile_value(config, "preferred_devices", profile, ""))

    env = {
        "VLLM_USE_MODELSCOPE": "true" if use_modelscope else "false",
        "CUDA_VISIBLE_DEVICES": devices,
    }
    for key, value in (config.get("extra_env") or {}).items():
        env[str(key)] = str(value)

    command = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_name,
        "--served-model-name",
        served_model_name,
        "--host",
        host,
        "--port",
        port,
        "--dtype",
        dtype,
        "--max-model-len",
        max_model_len,
        "--gpu-memory-utilization",
        gpu_memory_utilization,
    ]
    if profile == "tp2":
        command.extend(["--tensor-parallel-size", str(config.get("tensor_parallel_size", 2))])

    return env, command


def render_shell(env: dict[str, str], command: list[str]) -> str:
    exports = [f"export {key}={shlex.quote(value)}" for key, value in env.items() if value]
    return "\n".join([*exports, " ".join(shlex.quote(part) for part in command)])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a configured vLLM launch command.")
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--profile", choices=["gpu0", "gpu1", "tp2"], required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually exec the vLLM server. Default is dry-run command rendering.",
    )
    args = parser.parse_args(argv)

    config = load_yaml(Path(args.config))
    env, command = build_vllm_command(config, args.profile)

    if not args.execute:
        print(render_shell(env, command))
        return 0

    subprocess.run(command, check=True, env={**os.environ, **env})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
