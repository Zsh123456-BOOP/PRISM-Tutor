from __future__ import annotations

import importlib.util
import json
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
