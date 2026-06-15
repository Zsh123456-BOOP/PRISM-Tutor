from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "12_finalize_full_run.py"
SPEC = importlib.util.spec_from_file_location("finalize_full_run_script", SCRIPT_PATH)
finalize = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(finalize)


def _write_plan(tmp_path: Path, statuses: list[str]) -> Path:
    jobs = []
    for index, status in enumerate(statuses):
        job_id = f"job{index}"
        generation_path = tmp_path / f"{job_id}.jsonl"
        manifest_path = tmp_path / f"{job_id}.manifest.json"
        if status == "completed":
            generation_path.write_text(json.dumps({"sample_id": job_id}) + "\n", encoding="utf-8")
            manifest_path.write_text(json.dumps({"status": "completed", "run": {"counts": {"succeeded": 1}}}), encoding="utf-8")
        jobs.append(
            {
                "job_id": job_id,
                "experiment": "exp0",
                "shard_index": index,
                "num_shards": len(statuses),
                "estimated_records": 1,
                "paths": {
                    "generations": str(generation_path),
                    "errors": str(tmp_path / f"{job_id}.errors.jsonl"),
                    "manifest": str(manifest_path),
                    "pid": str(tmp_path / f"{job_id}.pid"),
                },
            }
        )
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"output_dir": str(tmp_path / "out"), "jobs": jobs}), encoding="utf-8")
    return plan


def test_finalize_refuses_incomplete_plan_without_override(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path, ["completed", "pending"])

    try:
        finalize.main(["--plan", str(plan), "--manifest", str(tmp_path / "manifest.json"), "--dry-run"])
    except SystemExit as exc:
        assert "Full run is not complete" in str(exc)
    else:
        raise AssertionError("Expected incomplete full run to fail")


def test_finalize_dry_run_writes_planned_manifest(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path, ["completed", "pending"])
    manifest = tmp_path / "manifest.json"

    rc = finalize.main(
        [
            "--plan",
            str(plan),
            "--output-dir",
            str(tmp_path / "out"),
            "--manifest",
            str(manifest),
            "--allow-incomplete",
            "--dry-run",
        ]
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["status"] == "planned"
    assert payload["shard_status"]["by_status"] == {"completed": 1, "pending": 1}
    assert [step["name"] for step in payload["steps"]] == [
        "auto_metrics",
        "tables",
        "figures",
        "human_audit_sample",
        "paper_artifacts",
    ]


def test_finalize_only_adds_judge_when_requested(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path, ["completed"])
    args = finalize.argparse.Namespace(
        output_dir=str(tmp_path / "out"),
        gold="data/splits",
        run_judge=True,
        judge_config="configs/judge.yaml",
        audit_n=200,
    )

    commands = finalize.build_commands(args)

    assert [command["name"] for command in commands][:2] == ["auto_metrics", "llm_judge"]
