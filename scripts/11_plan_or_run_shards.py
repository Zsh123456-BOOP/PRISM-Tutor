#!/usr/bin/env python
"""Plan, launch, and inspect sharded generation jobs."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prism_tutor.utils.config import load_yaml


def _split_counts(split_dir: Path) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for path in sorted(split_dir.glob("*_*.jsonl")):
        dataset, split = path.stem.rsplit("_", 1)
        with path.open("r", encoding="utf-8") as handle:
            counts[(dataset, split)] = sum(1 for line in handle if line.strip())
    return counts


def _shard_count(total: int, num_shards: int, shard_index: int) -> int:
    if total <= shard_index:
        return 0
    return (total - shard_index + num_shards - 1) // num_shards


def _filter_experiments(experiments: dict[str, Any], names: list[str] | None) -> dict[str, Any]:
    if not names:
        return experiments
    missing = [name for name in names if name not in experiments]
    if missing:
        raise KeyError(f"Unknown experiments: {', '.join(missing)}")
    return {name: experiments[name] for name in names}


def build_plan(
    *,
    experiments_config: str,
    split_dir: str,
    output_dir: str,
    num_shards: int,
    experiments: list[str] | None = None,
    live_llm: bool = True,
    resume: bool = True,
    limit: int | None = None,
) -> dict[str, Any]:
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    matrix = _filter_experiments(load_yaml(experiments_config)["experiments"], experiments)
    counts = _split_counts(Path(split_dir))
    jobs = []
    for experiment, spec in matrix.items():
        split = spec.get("split", "test")
        method_count = len(spec["methods"])
        for shard_index in range(num_shards):
            run_id = f"{experiment}_shard{shard_index:03d}-of-{num_shards:03d}"
            estimated_samples = sum(
                _shard_count(counts.get((dataset, split), 0), num_shards, shard_index)
                for dataset in spec["datasets"]
            )
            estimated_records = estimated_samples * method_count
            argv = [
                "python",
                "scripts/02_run_generation.py",
                "--experiment",
                experiment,
                "--num-shards",
                str(num_shards),
                "--shard-index",
                str(shard_index),
                "--run-id",
                run_id,
                "--output_dir",
                output_dir,
            ]
            if live_llm:
                argv.append("--live-llm")
            if resume:
                argv.append("--resume")
            if limit is not None:
                argv.extend(["--limit", str(limit)])
            jobs.append(
                {
                    "job_id": run_id,
                    "experiment": experiment,
                    "split": split,
                    "datasets": spec["datasets"],
                    "method_count": method_count,
                    "num_shards": num_shards,
                    "shard_index": shard_index,
                    "estimated_samples": estimated_samples,
                    "estimated_records": estimated_records,
                    "argv": argv,
                    "command": shlex.join(argv),
                    "paths": {
                        "generations": f"{output_dir}/generations/{run_id}.jsonl",
                        "errors": f"{output_dir}/logs/generation_errors_{run_id}.jsonl",
                        "manifest": f"{output_dir}/logs/experiment_manifest_{run_id}.json",
                        "stdout": f"{output_dir}/logs/shards/{run_id}.out",
                        "pid": f"{output_dir}/logs/shards/{run_id}.pid",
                    },
                }
            )
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiments_config": experiments_config,
        "split_dir": split_dir,
        "output_dir": output_dir,
        "num_shards": num_shards,
        "live_llm": live_llm,
        "resume": resume,
        "limit": limit,
        "split_counts": {f"{dataset}/{split}": count for (dataset, split), count in sorted(counts.items())},
        "job_count": len(jobs),
        "estimated_records": sum(job["estimated_records"] for job in jobs),
        "jobs": jobs,
    }


def write_plan(plan: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def load_plan(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _count_jsonl(path: str | Path) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    with p.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _read_pid(path: str | Path) -> int | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return int(p.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _pid_is_running(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def job_status(job: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(job["paths"]["manifest"])
    generation_rows = _count_jsonl(job["paths"]["generations"])
    error_rows = _count_jsonl(job["paths"]["errors"])
    manifest: dict[str, Any] | None = None
    pid = _read_pid(job["paths"].get("pid", ""))
    pid_running = _pid_is_running(pid)
    status = "pending"
    counts: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            status = str(manifest.get("status", "unknown"))
            counts = manifest.get("run", {}).get("counts", {})
        except json.JSONDecodeError:
            status = "manifest_invalid"
    elif pid_running:
        status = "running"
    elif generation_rows:
        status = "partial"
    if pid is not None and not pid_running and status in {"running", "partial"}:
        status = "stale_partial" if generation_rows else "stale"
    return {
        "job_id": job["job_id"],
        "experiment": job["experiment"],
        "shard_index": job["shard_index"],
        "num_shards": job["num_shards"],
        "status": status,
        "generation_rows": generation_rows,
        "error_rows": error_rows,
        "estimated_records": job.get("estimated_records"),
        "pid": pid,
        "pid_running": pid_running,
        "counts": counts,
    }


def status_report(plan: dict[str, Any]) -> dict[str, Any]:
    jobs = [job_status(job) for job in plan["jobs"]]
    by_status: dict[str, int] = {}
    for job in jobs:
        by_status[job["status"]] = by_status.get(job["status"], 0) + 1
    return {
        "job_count": len(jobs),
        "by_status": by_status,
        "generation_rows": sum(job["generation_rows"] for job in jobs),
        "error_rows": sum(job["error_rows"] for job in jobs),
        "estimated_records": sum(int(job.get("estimated_records") or 0) for job in jobs),
        "jobs": jobs,
    }


def _select_job(plan: dict[str, Any], job_id: str | None, launch_next: bool, selected_ids: set[str] | None = None) -> dict[str, Any]:
    selected_ids = selected_ids or set()
    if job_id:
        for job in plan["jobs"]:
            if job["job_id"] == job_id:
                return job
        raise KeyError(f"Unknown job_id: {job_id}")
    if launch_next:
        for job in plan["jobs"]:
            if job["job_id"] in selected_ids:
                continue
            status = job_status(job)["status"]
            if status not in {"completed", "running"}:
                return job
        raise RuntimeError("No incomplete jobs remain")
    raise ValueError("Provide --job-id or --next")


def launch_job(plan: dict[str, Any], *, job_id: str | None, launch_next: bool, background: bool, selected_ids: set[str] | None = None) -> dict[str, Any]:
    job = _select_job(plan, job_id, launch_next, selected_ids=selected_ids)
    argv = list(job["argv"])
    stdout_path = Path(job["paths"]["stdout"])
    pid_path = Path(job["paths"]["pid"])
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    if background:
        stdout = stdout_path.open("ab")
        process = subprocess.Popen(argv, stdout=stdout, stderr=subprocess.STDOUT)
        pid_path.write_text(str(process.pid), encoding="utf-8")
        return {"job_id": job["job_id"], "pid": process.pid, "stdout": str(stdout_path), "background": True}
    completed = subprocess.run(argv, check=False)
    return {"job_id": job["job_id"], "returncode": completed.returncode, "background": False}


def launch_jobs(plan: dict[str, Any], *, job_id: str | None, launch_next: bool, background: bool, count: int) -> dict[str, Any]:
    if count < 1:
        raise ValueError("count must be >= 1")
    if job_id and count != 1:
        raise ValueError("--job-id can only be used with --count 1")
    if not background and count != 1:
        raise ValueError("foreground launch supports only --count 1")
    launched = []
    selected_ids: set[str] = set()
    for _ in range(count):
        result = launch_job(
            plan,
            job_id=job_id,
            launch_next=launch_next,
            background=background,
            selected_ids=selected_ids,
        )
        selected_ids.add(result["job_id"])
        launched.append(result)
    return {"launched": launched, "count": len(launched), "background": background}


def maintain_jobs(plan: dict[str, Any], *, target_running: int, max_launches: int | None = None) -> dict[str, Any]:
    if target_running < 1:
        raise ValueError("target_running must be >= 1")
    report_before = status_report(plan)
    running = int(report_before["by_status"].get("running", 0))
    needed = max(0, target_running - running)
    if max_launches is not None:
        if max_launches < 0:
            raise ValueError("max_launches must be >= 0")
        needed = min(needed, max_launches)
    if needed == 0:
        launched: list[dict[str, Any]] = []
    else:
        launched = launch_jobs(
            plan,
            job_id=None,
            launch_next=True,
            background=True,
            count=needed,
        )["launched"]
    report_after = status_report(plan)
    return {
        "target_running": target_running,
        "requested_launches": needed,
        "launched": launched,
        "before": {
            "by_status": report_before["by_status"],
            "generation_rows": report_before["generation_rows"],
            "error_rows": report_before["error_rows"],
        },
        "after": {
            "by_status": report_after["by_status"],
            "generation_rows": report_after["generation_rows"],
            "error_rows": report_after["error_rows"],
        },
    }


def _append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _all_terminal(report: dict[str, Any]) -> bool:
    return set(report["by_status"]) <= {"completed"}


def _compact_status(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_count": report["job_count"],
        "by_status": report["by_status"],
        "generation_rows": report["generation_rows"],
        "error_rows": report["error_rows"],
        "estimated_records": report["estimated_records"],
    }


def _read_supervisor_events(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    log_path = Path(path)
    if not log_path.exists():
        return []
    events = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events[-limit:] if limit is not None else events


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def progress_report(plan: dict[str, Any], *, supervisor_log: str | Path | None = None, rate_window: int = 5) -> dict[str, Any]:
    if rate_window < 2:
        raise ValueError("rate_window must be >= 2")
    summary = _compact_status(status_report(plan))
    estimated = int(summary["estimated_records"] or 0)
    generated = int(summary["generation_rows"] or 0)
    remaining = max(0, estimated - generated)
    completion_fraction = (generated / estimated) if estimated else None
    supervisor_log = supervisor_log or Path(plan["output_dir"]) / "logs" / "shards" / "supervisor_compact.jsonl"
    events = _read_supervisor_events(supervisor_log, limit=rate_window)
    rate_rows_per_minute = None
    eta_hours = None
    if len(events) >= 2:
        first = events[0]
        last = events[-1]
        first_time = _parse_timestamp(first.get("timestamp_utc"))
        last_time = _parse_timestamp(last.get("timestamp_utc"))
        first_rows = int(first.get("status", {}).get("generation_rows") or 0)
        last_rows = int(last.get("status", {}).get("generation_rows") or 0)
        if first_time and last_time:
            elapsed_minutes = (last_time - first_time).total_seconds() / 60
            row_delta = last_rows - first_rows
            if elapsed_minutes > 0 and row_delta > 0:
                rate_rows_per_minute = row_delta / elapsed_minutes
                eta_hours = (remaining / rate_rows_per_minute) / 60 if rate_rows_per_minute else None
    return {
        **summary,
        "completion_fraction": completion_fraction,
        "remaining_records": remaining,
        "supervisor_log": str(supervisor_log),
        "rate_window_events": len(events),
        "recent_rows_per_minute": rate_rows_per_minute,
        "eta_hours": eta_hours,
    }


def supervise_jobs(
    plan_path: str | Path,
    *,
    target_running: int,
    interval_seconds: float,
    max_cycles: int | None = None,
    log_path: str | Path | None = None,
) -> dict[str, Any]:
    if interval_seconds < 1:
        raise ValueError("interval_seconds must be >= 1")
    if max_cycles is not None and max_cycles < 1:
        raise ValueError("max_cycles must be >= 1")
    plan = load_plan(plan_path)
    log_path = log_path or Path(plan["output_dir"]) / "logs" / "shards" / "supervisor.jsonl"
    cycles = 0
    last_event: dict[str, Any] | None = None
    while True:
        cycles += 1
        maintain_result = maintain_jobs(plan, target_running=target_running)
        report = status_report(plan)
        event = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "cycle": cycles,
            "maintain": maintain_result,
            "status": _compact_status(report),
        }
        _append_jsonl(log_path, event)
        last_event = event
        if _all_terminal(report):
            break
        if max_cycles is not None and cycles >= max_cycles:
            break
        time.sleep(interval_seconds)
    return {
        "cycles": cycles,
        "log_path": str(log_path),
        "last_status": last_event["status"] if last_event else None,
    }


def _split_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parts = [item.strip() for item in value.split(",") if item.strip()]
    return parts or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan_parser = sub.add_parser("plan")
    plan_parser.add_argument("--experiments-config", default="configs/experiments.yaml")
    plan_parser.add_argument("--split-dir", default="data/splits")
    plan_parser.add_argument("--output-dir", default="outputs/full_run")
    plan_parser.add_argument("--num-shards", type=int, default=128)
    plan_parser.add_argument("--experiments", help="Comma-separated subset of experiment names.")
    plan_parser.add_argument("--plan-output", default="outputs/full_run/shard_plan.json")
    plan_parser.add_argument("--dry-run", action="store_true", help="Plan commands without --live-llm.")
    plan_parser.add_argument("--no-resume", action="store_true")
    plan_parser.add_argument("--limit", type=int)

    status_parser = sub.add_parser("status")
    status_parser.add_argument("--plan", default="outputs/full_run/shard_plan.json")

    launch_parser = sub.add_parser("launch")
    launch_parser.add_argument("--plan", default="outputs/full_run/shard_plan.json")
    launch_parser.add_argument("--job-id")
    launch_parser.add_argument("--next", action="store_true")
    launch_parser.add_argument("--background", action="store_true")
    launch_parser.add_argument("--count", type=int, default=1)

    maintain_parser = sub.add_parser("maintain")
    maintain_parser.add_argument("--plan", default="outputs/full_run/shard_plan.json")
    maintain_parser.add_argument("--target-running", type=int, default=2)
    maintain_parser.add_argument("--max-launches", type=int)

    progress_parser = sub.add_parser("progress")
    progress_parser.add_argument("--plan", default="outputs/full_run/shard_plan.json")
    progress_parser.add_argument("--supervisor-log")
    progress_parser.add_argument("--rate-window", type=int, default=5)

    supervise_parser = sub.add_parser("supervise")
    supervise_parser.add_argument("--plan", default="outputs/full_run/shard_plan.json")
    supervise_parser.add_argument("--target-running", type=int, default=2)
    supervise_parser.add_argument("--interval-seconds", type=float, default=300)
    supervise_parser.add_argument("--max-cycles", type=int)
    supervise_parser.add_argument("--log-path")

    args = parser.parse_args(argv)
    if args.command == "plan":
        plan = build_plan(
            experiments_config=args.experiments_config,
            split_dir=args.split_dir,
            output_dir=args.output_dir,
            num_shards=args.num_shards,
            experiments=_split_csv(args.experiments),
            live_llm=not args.dry_run,
            resume=not args.no_resume,
            limit=args.limit,
        )
        write_plan(plan, args.plan_output)
        print(json.dumps({"plan": args.plan_output, "job_count": plan["job_count"], "estimated_records": plan["estimated_records"]}, indent=2))
        return 0
    if args.command == "status":
        report = status_report(load_plan(args.plan))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "launch":
        result = launch_jobs(
            load_plan(args.plan),
            job_id=args.job_id,
            launch_next=args.next,
            background=args.background,
            count=args.count,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return max(int(item.get("returncode", 0)) for item in result["launched"])
    if args.command == "maintain":
        result = maintain_jobs(
            load_plan(args.plan),
            target_running=args.target_running,
            max_launches=args.max_launches,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "progress":
        result = progress_report(
            load_plan(args.plan),
            supervisor_log=args.supervisor_log,
            rate_window=args.rate_window,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "supervise":
        result = supervise_jobs(
            args.plan,
            target_running=args.target_running,
            interval_seconds=args.interval_seconds,
            max_cycles=args.max_cycles,
            log_path=args.log_path,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
