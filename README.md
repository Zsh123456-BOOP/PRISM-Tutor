# PRISM-Tutor

PRISM-Tutor is an inference-time runtime for communication-efficient multi-agent math tutoring. The project fixes the generator to Qwen3-8B and compares baseline multi-agent workflows against pedagogical-risk-aware routing, budgeted deliberation, and student-state commit.

## Current Scope

The repository is organized around the task cards in `task_cards/`. Code should move through small, reproducible phases:

```bash
conda env create -f environment.yml
conda activate prism_tutor
python scripts/00_prepare_env_check.py --config configs/default.yaml
```

Do not commit raw datasets, model weights, runtime logs, API keys, or generated experiment outputs.

## Safety

Use environment variables for external services, for example `DEEPSEEK_API_KEY`. Never write credentials into configs, logs, task cards, or generated artifacts.
