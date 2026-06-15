#!/usr/bin/env bash
set -euo pipefail

python scripts/02_run_generation.py \
  --experiment exp4_end_to_end \
  --limit "${PRISM_LIMIT:-2}" \
  --output_dir "${PRISM_OUTPUT_DIR:-outputs}" \
  "$@"
