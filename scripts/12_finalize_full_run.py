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

from prism_tutor.utils.reproducibility import collect_reproducibility_metadata


def _status_summary(plan_path: str | Path) -> dict[str, Any]:
    if not Path(plan_path).exists():
        raise SystemExit(f"Missing shard plan: {plan_path}")
    report = shard_planner.status_report(shard_planner.load_plan(plan_path))
    return {
        "job_count": report["job_count"],
        "by_status": report["by_status"],
        "generation_rows": report["generation_rows"],
        "error_rows": report["error_rows"],
        "raw_error_rows": report.get("raw_error_rows", report["error_rows"]),
        "estimated_records": report["estimated_records"],
    }


def _generation_manifest_git_summary(plan_path: str | Path) -> dict[str, Any]:
    plan = shard_planner.load_plan(plan_path)
    commit_counts: dict[str, int] = {}
    dirty_count = 0
    missing_manifest_count = 0
    invalid_manifest_count = 0
    missing_commit_count = 0
    for job in plan.get("jobs", []):
        manifest_path = Path(job.get("paths", {}).get("manifest", ""))
        if not manifest_path.exists():
            missing_manifest_count += 1
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            invalid_manifest_count += 1
            continue
        git = manifest.get("reproducibility", {}).get("git", {})
        commit = git.get("commit")
        if commit:
            commit_counts[str(commit)] = commit_counts.get(str(commit), 0) + 1
        else:
            missing_commit_count += 1
        if git.get("dirty"):
            dirty_count += 1
    return {
        "manifest_count": sum(commit_counts.values()) + missing_commit_count,
        "commit_counts": commit_counts,
        "distinct_commit_count": len(commit_counts),
        "dirty_manifest_count": dirty_count,
        "missing_manifest_count": missing_manifest_count,
        "invalid_manifest_count": invalid_manifest_count,
        "missing_commit_count": missing_commit_count,
    }


def _require_complete(summary: dict[str, Any], allow_incomplete: bool) -> None:
    incomplete = summary["by_status"].copy()
    completed = int(incomplete.pop("completed", 0))
    if incomplete and not allow_incomplete:
        raise SystemExit(
            "Full run is not complete: "
            + json.dumps({"completed": completed, "incomplete": incomplete}, ensure_ascii=False, sort_keys=True)
        )
    error_rows = int(summary.get("error_rows") or 0)
    if error_rows and not allow_incomplete:
        raise SystemExit(
            "Full run has generation error rows: "
            + json.dumps({"error_rows": error_rows}, ensure_ascii=False, sort_keys=True)
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


def _require_formal_flag_consistency(args: argparse.Namespace) -> None:
    if args.allow_incomplete:
        return
    smoke_flags = []
    if args.allow_mock_judge:
        smoke_flags.append("--allow-mock-judge")
    if args.allow_unlabeled_agreement:
        smoke_flags.append("--allow-unlabeled-agreement")
    if smoke_flags:
        raise SystemExit(
            "Smoke-only finalization flags require --allow-incomplete: "
            + ", ".join(smoke_flags)
        )


def _require_generation_manifest_git_consistency(summary: dict[str, Any], args: argparse.Namespace) -> None:
    if args.allow_incomplete:
        return
    failures = []
    if int(summary.get("missing_manifest_count") or 0):
        failures.append({"reason": "missing_generation_manifests", "count": summary["missing_manifest_count"]})
    if int(summary.get("invalid_manifest_count") or 0):
        failures.append({"reason": "invalid_generation_manifests", "count": summary["invalid_manifest_count"]})
    if int(summary.get("missing_commit_count") or 0):
        failures.append({"reason": "missing_generation_commit", "count": summary["missing_commit_count"]})
    if int(summary.get("dirty_manifest_count") or 0):
        failures.append({"reason": "dirty_generation_manifest", "count": summary["dirty_manifest_count"]})
    if int(summary.get("distinct_commit_count") or 0) > 1 and not args.allow_mixed_generation_commits:
        failures.append(
            {
                "reason": "mixed_generation_commits",
                "distinct_commit_count": summary["distinct_commit_count"],
            }
        )
    if failures:
        raise SystemExit(
            "Formal finalization requires clean generation manifest git provenance: "
            + json.dumps(failures, ensure_ascii=False, sort_keys=True)
        )


def _require_human_agreement_inputs(args: argparse.Namespace) -> None:
    if not args.run_human_agreement or args.allow_unlabeled_agreement:
        return
    labeled_path = Path(args.output_dir) / "human_audit" / "human_audit_labeled.csv"
    if not labeled_path.exists():
        raise SystemExit(
            "Formal human agreement requires labeled audit CSV before finalization: "
            + str(labeled_path)
        )


def _require_formal_judge_source(args: argparse.Namespace) -> None:
    if getattr(args, "allow_incomplete", False) or getattr(args, "run_judge", False):
        return
    judge_scores = Path(args.output_dir) / "judge_scores" / "judge_scores.jsonl"
    if not judge_scores.exists():
        raise SystemExit(
            "Formal finalization requires --run-judge or existing judge scores: "
            + str(judge_scores)
        )


def _cmd(*parts: str) -> list[str]:
    return [sys.executable, *parts]


def _paper_artifact_argv(args: argparse.Namespace, paper_artifacts: str, logs: str) -> list[str]:
    argv = _cmd(
        "scripts/09_export_paper_artifacts.py",
        "--root",
        ".",
        "--output_dir",
        paper_artifacts,
        "--logs",
        logs,
        "--artifact-prefix",
        args.output_dir,
    )
    if getattr(args, "plan", None):
        argv.extend(["--shard-plan", args.plan])
    return argv


def build_commands(args: argparse.Namespace) -> list[dict[str, Any]]:
    generations = str(Path(args.output_dir) / "generations")
    metrics = str(Path(args.output_dir) / "metrics")
    judge_scores = str(Path(args.output_dir) / "judge_scores" / "judge_scores.jsonl")
    tables = str(Path(args.output_dir) / "tables")
    figures = str(Path(args.output_dir) / "figures")
    human_audit = str(Path(args.output_dir) / "human_audit")
    paper_artifacts = str(Path(args.output_dir) / "paper_artifacts")
    logs = str(Path(args.output_dir) / "logs")
    commands = []
    if args.run_judge:
        judge_argv = _cmd(
            "scripts/03_run_judge.py",
            "--input",
            generations,
            "--judge_config",
            args.judge_config,
            "--output_dir",
            str(Path(args.output_dir) / "judge_scores"),
        )
        if not getattr(args, "allow_mock_judge", False):
            judge_argv.append("--require-real")
        commands.append(
            {
                "name": "llm_judge",
                "argv": judge_argv,
            }
        )
    commands.append(
        {
            "name": "auto_metrics",
            "argv": _cmd(
                "scripts/04_compute_metrics.py",
                "--generations",
                generations,
                "--gold",
                args.gold,
                "--judge-scores",
                judge_scores,
                "--output_dir",
                metrics,
            ),
        }
    )
    commands.extend(
        [
            {
                "name": "tables",
                "argv": _cmd("scripts/05_make_tables.py", "--record_metrics", str(Path(metrics) / "record_auto_metrics.jsonl"), "--output_dir", tables)
                + (["--allow-incomplete-tables"] if getattr(args, "allow_incomplete", False) else []),
            },
            {
                "name": "figures",
                "argv": _cmd("scripts/06_make_figures.py", "--record_metrics", str(Path(metrics) / "record_auto_metrics.jsonl"), "--output_dir", figures),
            },
            {
                "name": "human_audit_sample",
                "argv": _cmd(
                    "scripts/07_sample_human_audit.py",
                    "--records",
                    str(Path(metrics) / "record_auto_metrics.jsonl"),
                    "--generations",
                    generations,
                    "--judge-scores",
                    judge_scores,
                    "--tables",
                    tables,
                    "--n",
                    str(args.audit_n),
                    "--output_dir",
                    human_audit,
                ),
            },
        ]
    )
    if args.run_human_agreement:
        agreement_argv = _cmd(
            "scripts/08_human_agreement.py",
            "--input",
            str(Path(human_audit) / "human_audit_labeled.csv"),
            "--output",
            str(Path(human_audit) / "human_agreement_report.json"),
            "--preference-mapping",
            str(Path(human_audit) / "preference_mapping.json"),
        )
        if args.allow_unlabeled_agreement:
            agreement_argv.append("--allow-unlabeled")
        commands.append({"name": "human_agreement", "argv": agreement_argv})
    commands.append(
        {
            "name": "paper_artifacts",
            "argv": _paper_artifact_argv(args, paper_artifacts, logs),
        }
    )
    return commands


def _step_log_paths(log_dir: Path, step_name: str) -> dict[str, Path]:
    safe_name = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in step_name)
    return {
        "stdout": log_dir / f"{safe_name}.stdout.log",
        "stderr": log_dir / f"{safe_name}.stderr.log",
    }


def _run_command(step: dict[str, Any], dry_run: bool, log_dir: str | Path | None = None) -> dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()
    if dry_run:
        return {"name": step["name"], "argv": step["argv"], "status": "planned", "started_at_utc": started}
    logs: dict[str, Path] | None = None
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        logs = _step_log_paths(log_path, str(step["name"]))
        with logs["stdout"].open("w", encoding="utf-8") as stdout, logs["stderr"].open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(step["argv"], check=False, stdout=stdout, stderr=stderr)
    else:
        completed = subprocess.run(step["argv"], check=False)
    return {
        "name": step["name"],
        "argv": step["argv"],
        "status": "completed" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "stdout_log": str(logs["stdout"]) if logs else None,
        "stderr_log": str(logs["stderr"]) if logs else None,
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _skipped_step(step: dict[str, Any], reason: str) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "name": step["name"],
        "argv": step["argv"],
        "status": "skipped",
        "skip_reason": reason,
        "started_at_utc": now,
        "finished_at_utc": now,
    }


def _invocation_metadata(args: argparse.Namespace, argv: list[str] | None) -> dict[str, Any]:
    return {
        "argv": list(argv) if argv is not None else sys.argv[1:],
        "cwd": str(Path.cwd()),
        "repo_root": str(ROOT),
        "plan": args.plan,
        "output_dir": args.output_dir,
        "manifest": args.manifest,
        "dry_run": args.dry_run,
    }


def _run_steps(commands: list[dict[str, Any]], dry_run: bool, log_dir: str | Path | None = None) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    failed_step: str | None = None
    for command in commands:
        if failed_step is not None and not dry_run:
            steps.append(_skipped_step(command, f"previous step failed: {failed_step}"))
            continue
        result = _run_command(command, dry_run, log_dir)
        steps.append(result)
        if result["status"] == "failed":
            failed_step = str(command["name"])
    return steps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="outputs/full_run/shard_plan.json")
    parser.add_argument("--output-dir", default="outputs/full_run")
    parser.add_argument("--gold", default="data/splits")
    parser.add_argument("--judge-config", default="configs/judge.yaml")
    parser.add_argument("--audit-n", type=int, default=200)
    parser.add_argument("--run-judge", action="store_true", help="Run the configured judge. This may call an external API.")
    parser.add_argument("--allow-mock-judge", action="store_true", help="Allow mock/dry-run judge in finalization. Use only for smoke checks.")
    parser.add_argument("--run-human-agreement", action="store_true", help="Compute agreement after labeled human audit CSV exists.")
    parser.add_argument("--allow-unlabeled-agreement", action="store_true", help="Allow agreement smoke from blind CSV when labels are not filled.")
    parser.add_argument(
        "--allow-mixed-generation-commits",
        action="store_true",
        help="Allow formal finalization when completed shard manifests record multiple git commits. Requires explicit provenance review.",
    )
    parser.add_argument("--allow-incomplete", action="store_true", help="Allow smoke finalization before all shards complete.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest", default="outputs/full_run/finalization_manifest.json")
    parser.add_argument("--step-log-dir", help="Directory for per-step stdout/stderr logs. Defaults to <output-dir>/logs/finalization.")
    args = parser.parse_args(argv)

    summary = _status_summary(args.plan)
    generation_manifest_git = _generation_manifest_git_summary(args.plan)
    _require_complete(summary, args.allow_incomplete)
    _require_generation_manifest_git_consistency(generation_manifest_git, args)
    _require_formal_flag_consistency(args)
    _require_human_agreement_inputs(args)
    _require_formal_judge_source(args)
    completion = _completion_summary(summary)
    commands = build_commands(args)
    step_log_dir = args.step_log_dir or str(Path(args.output_dir) / "logs" / "finalization")
    steps = _run_steps(commands, args.dry_run, step_log_dir)
    planned_steps = [step["name"] for step in steps]
    status = "planned" if args.dry_run else ("completed" if all(step["status"] == "completed" for step in steps) else "failed")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "allow_incomplete": args.allow_incomplete,
        "run_judge": args.run_judge,
        "allow_mock_judge": args.allow_mock_judge,
        "run_human_agreement": args.run_human_agreement,
        "allow_unlabeled_agreement": args.allow_unlabeled_agreement,
        "allow_mixed_generation_commits": args.allow_mixed_generation_commits,
        "invocation": _invocation_metadata(args, argv),
        "reproducibility": collect_reproducibility_metadata(),
        "generation_manifest_git": generation_manifest_git,
        "shard_status": summary,
        **completion,
        "planned_steps": planned_steps,
        "step_log_dir": step_log_dir,
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
                "planned_steps": planned_steps,
                "run_judge": args.run_judge,
                "run_human_agreement": args.run_human_agreement,
                "allow_unlabeled_agreement": args.allow_unlabeled_agreement,
                "allow_mixed_generation_commits": args.allow_mixed_generation_commits,
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
