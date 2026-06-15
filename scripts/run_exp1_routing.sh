#!/usr/bin/env bash
set -euo pipefail

python scripts/02_run_generation.py \
  --experiment exp1_routing \
  --limit "${PRISM_LIMIT:-2}" \
  --output_dir "${PRISM_OUTPUT_DIR:-outputs}" \
  "$@"
