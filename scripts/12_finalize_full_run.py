#!/usr/bin/env python
"""Finalize a completed sharded full run into paper artifacts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SHARD_SCRIPT = ROOT / "scripts" / "11_plan_or_run_shards.py"
SPEC = importlib.util.spec_from_file_location("shard_planner_script", SHARD_SCRIPT)
shard_planner = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(shard_planner)


def _status_summary(plan_path: str | Path) -> dict[str, Any]:
    if not Path(plan_path).exists():
        raise SystemExit(f"Missing shard plan: {plan_path}")
    report = shard_planner.status_report(shard_planner.load_plan(plan_path))
    return {
        "job_count": report["job_count"],
        "by_status": report["by_status"],
        "generation_rows": report["generation_rows"],
        "error_rows": report["error_rows"],
        "estimated_records": report["estimated_records"],
    }


def _require_complete(summary: dict[str, Any], allow_incomplete: bool) -> None:
    incomplete = summary["by_status"].copy()
    completed = int(incomplete.pop("completed", 0))
    if incomplete and not allow_incomplete:
        raise SystemExit(
            "Full run is not complete: "
            + json.dumps({"completed": completed, "incomplete": incomplete}, ensure_ascii=False, sort_keys=True)
        )


def _completion_summary(summary: dict[str, Any]) -> dict[str, Any]:
    completed = int(summary["by_status"].get("completed", 0))
    total = int(summary["job_count"])
    incomplete = {key: value for key, value in summary["by_status"].items() if key != "completed" and value}
    return {
        "completed_jobs": completed,
        "total_jobs": total,
        "can_finalize": completed == total and not incomplete,
        "incomplete_jobs": incomplete,
    }


def _cmd(*parts: str) -> list[str]:
    return [sys.executable, *parts]


def build_commands(args: argparse.Namespace) -> list[dict[str, Any]]:
    generations = str(Path(args.output_dir) / "generations")
    metrics = str(Path(args.output_dir) / "metrics")
    tables = str(Path(args.output_dir) / "tables")
    figures = str(Path(args.output_dir) / "figures")
    human_audit = str(Path(args.output_dir) / "human_audit")
    paper_artifacts = str(Path(args.output_dir) / "paper_artifacts")
    logs = str(Path(args.output_dir) / "logs")
    commands = [
        {
            "name": "auto_metrics",
            "argv": _cmd("scripts/04_compute_metrics.py", "--generations", generations, "--gold", args.gold, "--output_dir", metrics),
        }
    ]
    if args.run_judge:
        commands.append(
            {
                "name": "llm_judge",
                "argv": _cmd("scripts/03_run_judge.py", "--input", generations, "--judge_config", args.judge_config, "--output_dir", str(Path(args.output_dir) / "judge_scores")),
            }
        )
    commands.extend(
        [
            {
                "name": "tables",
                "argv": _cmd("scripts/05_make_tables.py", "--record_metrics", str(Path(metrics) / "record_auto_metrics.jsonl"), "--output_dir", tables),
            },
            {
                "name": "figures",
                "argv": _cmd("scripts/06_make_figures.py", "--record_metrics", str(Path(metrics) / "record_auto_metrics.jsonl"), "--output_dir", figures),
            },
            {
                "name": "human_audit_sample",
                "argv": _cmd("scripts/07_sample_human_audit.py", "--records", str(Path(metrics) / "record_auto_metrics.jsonl"), "--generations", generations, "--n", str(args.audit_n), "--output_dir", human_audit),
            },
            {
                "name": "paper_artifacts",
                "argv": _cmd(
                    "scripts/09_export_paper_artifacts.py",
                    "--root",
                    ".",
                    "--output_dir",
                    paper_artifacts,
                    "--logs",
                    logs,
                    "--artifact-prefix",
                    args.output_dir,
                ),
            },
        ]
    )
    return commands


def _run_command(step: dict[str, Any], dry_run: bool) -> dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()
    if dry_run:
        return {"name": step["name"], "argv": step["argv"], "status": "planned", "started_at_utc": started}
    completed = subprocess.run(step["argv"], check=False)
    return {
        "name": step["name"],
        "argv": step["argv"],
        "status": "completed" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="outputs/full_run/shard_plan.json")
    parser.add_argument("--output-dir", default="outputs/full_run")
    parser.add_argument("--gold", default="data/splits")
    parser.add_argument("--judge-config", default="configs/judge.yaml")
    parser.add_argument("--audit-n", type=int, default=200)
    parser.add_argument("--run-judge", action="store_true", help="Run the configured judge. This may call an external API.")
    parser.add_argument("--allow-incomplete", action="store_true", help="Allow smoke finalization before all shards complete.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest", default="outputs/full_run/finalization_manifest.json")
    args = parser.parse_args(argv)

    summary = _status_summary(args.plan)
    _require_complete(summary, args.allow_incomplete)
    completion = _completion_summary(summary)
    commands = build_commands(args)
    steps = [_run_command(step, args.dry_run) for step in commands]
    status = "planned" if args.dry_run else ("completed" if all(step["status"] == "completed" for step in steps) else "failed")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "allow_incomplete": args.allow_incomplete,
        "run_judge": args.run_judge,
        "shard_status": summary,
        **completion,
        "steps": steps,
    }
    output = Path(args.manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": status,
                "manifest": str(output),
                "shard_status": summary,
                **completion,
                "planned_steps": [step["name"] for step in steps],
                "run_judge": args.run_judge,
                "allow_incomplete": args.allow_incomplete,
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if status in {"planned", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
