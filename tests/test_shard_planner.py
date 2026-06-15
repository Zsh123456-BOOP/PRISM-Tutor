from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "11_plan_or_run_shards.py"
SPEC = importlib.util.spec_from_file_location("shard_planner_script", SCRIPT_PATH)
shard_planner = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(shard_planner)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_build_plan_estimates_records_by_shard(tmp_path: Path) -> None:
    experiments = tmp_path / "experiments.yaml"
    experiments.write_text(
        json.dumps(
            {
                "experiments": {
                    "exp0_problem_diagnosis": {
                        "datasets": ["mathdial", "bridge"],
                        "split": "test",
                        "methods": ["single_tutor", "fixed_2"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(tmp_path / "splits" / "mathdial_test.jsonl", [{"sample_id": f"m{i}"} for i in range(5)])
    _write_jsonl(tmp_path / "splits" / "bridge_test.jsonl", [{"sample_id": f"b{i}"} for i in range(3)])

    plan = shard_planner.build_plan(
        experiments_config=str(experiments),
        split_dir=str(tmp_path / "splits"),
        output_dir=str(tmp_path / "out"),
        num_shards=2,
        experiments=["exp0_problem_diagnosis"],
        live_llm=False,
        resume=True,
        limit=None,
    )

    assert plan["job_count"] == 2
    assert plan["estimated_records"] == 16
    assert [job["estimated_records"] for job in plan["jobs"]] == [10, 6]
    assert "--live-llm" not in plan["jobs"][0]["argv"]
    assert "--resume" in plan["jobs"][0]["argv"]


def test_status_report_reads_manifest_and_partial_rows(tmp_path: Path) -> None:
    plan = {
        "jobs": [
            {
                "job_id": "done",
                "experiment": "exp0",
                "shard_index": 0,
                "num_shards": 2,
                "estimated_records": 2,
                "paths": {
                    "generations": str(tmp_path / "done.jsonl"),
                    "errors": str(tmp_path / "done_errors.jsonl"),
                    "manifest": str(tmp_path / "done_manifest.json"),
                },
            },
            {
                "job_id": "partial",
                "experiment": "exp0",
                "shard_index": 1,
                "num_shards": 2,
                "estimated_records": 2,
                "paths": {
                    "generations": str(tmp_path / "partial.jsonl"),
                    "errors": str(tmp_path / "partial_errors.jsonl"),
                    "manifest": str(tmp_path / "partial_manifest.json"),
                },
            },
        ]
    }
    _write_jsonl(tmp_path / "done.jsonl", [{"sample_id": "a"}])
    (tmp_path / "done_manifest.json").write_text(json.dumps({"status": "completed", "run": {"counts": {"succeeded": 1}}}), encoding="utf-8")
    _write_jsonl(tmp_path / "partial.jsonl", [{"sample_id": "b"}])

    report = shard_planner.status_report(plan)

    assert report["by_status"] == {"completed": 1, "partial": 1}
    assert report["generation_rows"] == 2


def test_status_report_marks_running_pid(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "running.jsonl", [{"sample_id": "a"}])
    (tmp_path / "running.pid").write_text(str(os.getpid()), encoding="utf-8")
    plan = {
        "jobs": [
            {
                "job_id": "running",
                "experiment": "exp0",
                "shard_index": 0,
                "num_shards": 1,
                "estimated_records": 2,
                "paths": {
                    "generations": str(tmp_path / "running.jsonl"),
                    "errors": str(tmp_path / "running_errors.jsonl"),
                    "manifest": str(tmp_path / "running_manifest.json"),
                    "pid": str(tmp_path / "running.pid"),
                },
            }
        ]
    }

    report = shard_planner.status_report(plan)

    assert report["by_status"] == {"running": 1}
    assert report["jobs"][0]["pid_running"] is True


def test_launch_jobs_selects_distinct_next_jobs(monkeypatch) -> None:
    launched = []
    plan = {"jobs": [{"job_id": "a", "argv": ["echo", "a"], "paths": {}}, {"job_id": "b", "argv": ["echo", "b"], "paths": {}}]}

    monkeypatch.setattr(shard_planner, "job_status", lambda job: {"status": "pending"})

    def fake_launch_job(plan, *, job_id, launch_next, background, selected_ids):
        job = shard_planner._select_job(plan, job_id, launch_next, selected_ids=selected_ids)
        launched.append(job["job_id"])
        return {"job_id": job["job_id"], "pid": len(launched), "background": background}

    monkeypatch.setattr(shard_planner, "launch_job", fake_launch_job)

    result = shard_planner.launch_jobs(plan, job_id=None, launch_next=True, background=True, count=2)

    assert launched == ["a", "b"]
    assert result["count"] == 2


def test_maintain_jobs_does_not_launch_when_target_is_met(monkeypatch) -> None:
    plan = {"jobs": [{"job_id": "a", "argv": ["echo", "a"], "paths": {}}]}
    monkeypatch.setattr(
        shard_planner,
        "status_report",
        lambda plan: {"by_status": {"running": 2}, "generation_rows": 4, "error_rows": 0},
    )

    result = shard_planner.maintain_jobs(plan, target_running=2)

    assert result["requested_launches"] == 0
    assert result["launched"] == []
    assert result["before"]["by_status"] == {"running": 2}


def test_maintain_jobs_launches_missing_background_jobs(monkeypatch) -> None:
    plan = {"jobs": [{"job_id": "a", "argv": ["echo", "a"], "paths": {}}, {"job_id": "b", "argv": ["echo", "b"], "paths": {}}]}
    reports = [
        {"by_status": {"running": 1, "pending": 2}, "generation_rows": 3, "error_rows": 0},
        {"by_status": {"running": 3}, "generation_rows": 3, "error_rows": 0},
    ]
    launched_args = []
    monkeypatch.setattr(shard_planner, "status_report", lambda plan: reports.pop(0))

    def fake_launch_jobs(plan, *, job_id, launch_next, background, count):
        launched_args.append((job_id, launch_next, background, count))
        return {"launched": [{"job_id": "a", "pid": 123, "background": True}]}

    monkeypatch.setattr(shard_planner, "launch_jobs", fake_launch_jobs)

    result = shard_planner.maintain_jobs(plan, target_running=3, max_launches=1)

    assert result["requested_launches"] == 1
    assert result["launched"] == [{"job_id": "a", "pid": 123, "background": True}]
    assert launched_args == [(None, True, True, 1)]


def test_supervise_jobs_writes_cycle_log_and_stops_at_max_cycles(tmp_path: Path, monkeypatch) -> None:
    plan_path = tmp_path / "plan.json"
    log_path = tmp_path / "supervisor.jsonl"
    plan_path.write_text(json.dumps({"output_dir": str(tmp_path / "out"), "jobs": []}), encoding="utf-8")
    monkeypatch.setattr(
        shard_planner,
        "maintain_jobs",
        lambda plan, *, target_running: {
            "target_running": target_running,
            "requested_launches": 0,
            "launched": [],
        },
    )
    monkeypatch.setattr(
        shard_planner,
        "status_report",
        lambda plan: {
            "job_count": 2,
            "by_status": {"running": 1},
            "generation_rows": 2,
            "error_rows": 0,
            "estimated_records": 4,
            "jobs": [],
        },
    )

    result = shard_planner.supervise_jobs(
        plan_path,
        target_running=2,
        interval_seconds=1,
        max_cycles=1,
        log_path=log_path,
    )

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert result["cycles"] == 1
    assert result["last_status"]["by_status"] == {"running": 1}
    assert rows[0]["maintain"]["target_running"] == 2
    assert rows[0]["cycle"] == 1


def test_supervise_jobs_stops_when_all_jobs_completed(tmp_path: Path, monkeypatch) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"output_dir": str(tmp_path / "out"), "jobs": []}), encoding="utf-8")
    monkeypatch.setattr(shard_planner, "maintain_jobs", lambda plan, *, target_running: {"launched": []})
    monkeypatch.setattr(
        shard_planner,
        "status_report",
        lambda plan: {
            "job_count": 2,
            "by_status": {"completed": 2},
            "generation_rows": 4,
            "error_rows": 0,
            "estimated_records": 4,
            "jobs": [],
        },
    )

    result = shard_planner.supervise_jobs(plan_path, target_running=2, interval_seconds=1)

    assert result["cycles"] == 1
    assert result["last_status"]["by_status"] == {"completed": 2}


def test_progress_report_estimates_recent_rate_and_eta(tmp_path: Path) -> None:
    plan = {
        "output_dir": str(tmp_path / "out"),
        "jobs": [
            {
                "job_id": "done",
                "experiment": "exp0",
                "shard_index": 0,
                "num_shards": 2,
                "estimated_records": 10,
                "paths": {
                    "generations": str(tmp_path / "done.jsonl"),
                    "errors": str(tmp_path / "done_errors.jsonl"),
                    "manifest": str(tmp_path / "done_manifest.json"),
                },
            },
            {
                "job_id": "running",
                "experiment": "exp0",
                "shard_index": 1,
                "num_shards": 2,
                "estimated_records": 10,
                "paths": {
                    "generations": str(tmp_path / "running.jsonl"),
                    "errors": str(tmp_path / "running_errors.jsonl"),
                    "manifest": str(tmp_path / "running_manifest.json"),
                    "pid": str(tmp_path / "running.pid"),
                },
            },
        ],
    }
    _write_jsonl(tmp_path / "done.jsonl", [{"sample_id": f"a{i}"} for i in range(10)])
    (tmp_path / "done_manifest.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    _write_jsonl(tmp_path / "running.jsonl", [{"sample_id": f"b{i}"} for i in range(2)])
    (tmp_path / "running.pid").write_text(str(os.getpid()), encoding="utf-8")
    log = tmp_path / "supervisor.jsonl"
    _write_jsonl(
        log,
        [
            {"timestamp_utc": "2026-06-15T00:00:00+00:00", "status": {"generation_rows": 2}},
            {"timestamp_utc": "2026-06-15T00:10:00+00:00", "status": {"generation_rows": 12}},
        ],
    )

    report = shard_planner.progress_report(plan, supervisor_log=log)

    assert report["generation_rows"] == 12
    assert report["estimated_records"] == 20
    assert report["remaining_records"] == 8
    assert report["recent_rows_per_minute"] == 1.0
    assert round(report["eta_hours"], 3) == 0.133
    assert report["health"]["status"] == "ok"


def test_progress_report_works_without_supervisor_log(tmp_path: Path) -> None:
    plan = {
        "output_dir": str(tmp_path / "out"),
        "jobs": [
            {
                "job_id": "pending",
                "experiment": "exp0",
                "shard_index": 0,
                "num_shards": 1,
                "estimated_records": 5,
                "paths": {
                    "generations": str(tmp_path / "pending.jsonl"),
                    "errors": str(tmp_path / "pending_errors.jsonl"),
                    "manifest": str(tmp_path / "pending_manifest.json"),
                },
            }
        ],
    }

    report = shard_planner.progress_report(plan, supervisor_log=tmp_path / "missing.jsonl")

    assert report["completion_fraction"] == 0
    assert report["rate_window_events"] == 0
    assert report["recent_rows_per_minute"] is None
    assert report["eta_hours"] is None
    assert report["health"]["status"] == "ok"


def test_progress_report_marks_generation_errors_unhealthy(tmp_path: Path) -> None:
    plan = {
        "output_dir": str(tmp_path / "out"),
        "jobs": [
            {
                "job_id": "failed",
                "experiment": "exp0",
                "shard_index": 0,
                "num_shards": 1,
                "estimated_records": 2,
                "paths": {
                    "generations": str(tmp_path / "failed.jsonl"),
                    "errors": str(tmp_path / "failed_errors.jsonl"),
                    "manifest": str(tmp_path / "failed_manifest.json"),
                },
            }
        ],
    }
    _write_jsonl(tmp_path / "failed.jsonl", [{"sample_id": "a"}])
    _write_jsonl(tmp_path / "failed_errors.jsonl", [{"sample_id": "a", "status": "failed"}])
    (tmp_path / "failed_manifest.json").write_text(json.dumps({"status": "completed_with_failures"}), encoding="utf-8")

    report = shard_planner.progress_report(plan, supervisor_log=tmp_path / "missing.jsonl")

    assert report["error_rows"] == 1
    assert report["health"]["status"] == "error"
    assert "generation_errors_present" in report["health"]["issues"]
    assert "completed_with_failures" in report["health"]["bad_statuses"]
