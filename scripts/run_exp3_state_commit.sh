#!/usr/bin/env bash
set -euo pipefail

python scripts/02_run_generation.py \
  --experiment exp3_state_commit \
  --limit "${PRISM_LIMIT:-2}" \
  --output_dir "${PRISM_OUTPUT_DIR:-outputs}" \
  "$@"
