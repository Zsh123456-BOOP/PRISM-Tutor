#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${PRISM_MODEL_CONFIG:-configs/model.yaml}"

python -m serving.vllm_command --config "$CONFIG_PATH" --profile tp2 "$@"
