#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prism_tutor.experiments.runner import RunnerOptions, run_generation


def _split_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parts = [item.strip() for item in value.split(",") if item.strip()]
    return parts or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run PRISM-Tutor generation experiments.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--experiments-config", default="configs/experiments.yaml")
    parser.add_argument("--methods", help="Comma-separated method names. Defaults to experiment methods or single_tutor.")
    parser.add_argument("--datasets", help="Comma-separated dataset names. Defaults to experiment datasets or mathdial.")
    parser.add_argument("--split")
    parser.add_argument("--experiment")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--run-id")
    parser.add_argument("--live-llm", action="store_true", help="Call configured vLLM endpoints instead of dry-run methods.")
    args = parser.parse_args(argv)

    result = run_generation(
        RunnerOptions(
            config_path=args.config,
            experiments_config_path=args.experiments_config,
            methods=_split_csv(args.methods),
            datasets=_split_csv(args.datasets),
            split=args.split,
            experiment=args.experiment,
            limit=args.limit,
            resume=args.resume,
            output_dir=args.output_dir,
            run_id=args.run_id,
            live_llm=args.live_llm,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] in {"completed", "completed_with_failures"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
