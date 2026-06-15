#!/usr/bin/env python
"""Plan, launch, and inspect sharded generation jobs."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
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


def job_status(job: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(job["paths"]["manifest"])
    generation_rows = _count_jsonl(job["paths"]["generations"])
    error_rows = _count_jsonl(job["paths"]["errors"])
    manifest: dict[str, Any] | None = None
    status = "pending"
    counts: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            status = str(manifest.get("status", "unknown"))
            counts = manifest.get("run", {}).get("counts", {})
        except json.JSONDecodeError:
            status = "manifest_invalid"
    elif generation_rows:
        status = "partial"
    return {
        "job_id": job["job_id"],
        "experiment": job["experiment"],
        "shard_index": job["shard_index"],
        "num_shards": job["num_shards"],
        "status": status,
        "generation_rows": generation_rows,
        "error_rows": error_rows,
        "estimated_records": job.get("estimated_records"),
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


def _select_job(plan: dict[str, Any], job_id: str | None, launch_next: bool) -> dict[str, Any]:
    if job_id:
        for job in plan["jobs"]:
            if job["job_id"] == job_id:
                return job
        raise KeyError(f"Unknown job_id: {job_id}")
    if launch_next:
        for job in plan["jobs"]:
            status = job_status(job)["status"]
            if status not in {"completed"}:
                return job
        raise RuntimeError("No incomplete jobs remain")
    raise ValueError("Provide --job-id or --next")


def launch_job(plan: dict[str, Any], *, job_id: str | None, launch_next: bool, background: bool) -> dict[str, Any]:
    job = _select_job(plan, job_id, launch_next)
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
        result = launch_job(load_plan(args.plan), job_id=args.job_id, launch_next=args.next, background=args.background)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return int(result.get("returncode", 0))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
