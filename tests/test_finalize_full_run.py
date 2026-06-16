from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "12_finalize_full_run.py"
SPEC = importlib.util.spec_from_file_location("finalize_full_run_script", SCRIPT_PATH)
finalize = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(finalize)


def _write_plan(tmp_path: Path, statuses: list[str], *, error_jobs: set[int] | None = None) -> Path:
    error_jobs = error_jobs or set()
    jobs = []
    for index, status in enumerate(statuses):
        job_id = f"job{index}"
        generation_path = tmp_path / f"{job_id}.jsonl"
        manifest_path = tmp_path / f"{job_id}.manifest.json"
        error_path = tmp_path / f"{job_id}.errors.jsonl"
        if status == "completed":
            generation_path.write_text(json.dumps({"sample_id": job_id}) + "\n", encoding="utf-8")
            manifest_path.write_text(json.dumps({"status": "completed", "run": {"counts": {"succeeded": 1}}}), encoding="utf-8")
        if index in error_jobs:
            error_path.write_text(json.dumps({"sample_id": job_id, "error": "boom"}) + "\n", encoding="utf-8")
        jobs.append(
            {
                "job_id": job_id,
                "experiment": "exp0",
                "shard_index": index,
                "num_shards": len(statuses),
                "estimated_records": 1,
                "paths": {
                    "generations": str(generation_path),
                    "errors": str(error_path),
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


def test_finalize_refuses_error_rows_without_override(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path, ["completed"], error_jobs={0})

    try:
        finalize.main(["--plan", str(plan), "--manifest", str(tmp_path / "manifest.json"), "--dry-run"])
    except SystemExit as exc:
        assert "generation error rows" in str(exc)
    else:
        raise AssertionError("Expected full run with error rows to fail")


def test_finalize_refuses_smoke_only_flags_without_incomplete_override(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path, ["completed"])

    for flag in ["--allow-mock-judge", "--allow-unlabeled-agreement"]:
        try:
            finalize.main(["--plan", str(plan), "--manifest", str(tmp_path / f"{flag}.json"), flag, "--dry-run"])
        except SystemExit as exc:
            assert "Smoke-only finalization flags require --allow-incomplete" in str(exc)
        else:
            raise AssertionError(f"Expected {flag} without --allow-incomplete to fail")


def test_finalize_formal_human_agreement_requires_labeled_csv(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path, ["completed"])
    output_dir = tmp_path / "out"

    try:
        finalize.main(
            [
                "--plan",
                str(plan),
                "--output-dir",
                str(output_dir),
                "--manifest",
                str(tmp_path / "manifest.json"),
                "--run-human-agreement",
                "--dry-run",
            ]
        )
    except SystemExit as exc:
        assert "Formal human agreement requires labeled audit CSV" in str(exc)
    else:
        raise AssertionError("Expected formal human agreement without labels to fail")

    labeled = output_dir / "human_audit" / "human_audit_labeled.csv"
    labeled.parent.mkdir(parents=True)
    labeled.write_text(
        "sample_id,annotator_id,human_quality_score,human_leakage_label,human_preference\n",
        encoding="utf-8",
    )
    rc = finalize.main(
        [
            "--plan",
            str(plan),
            "--output-dir",
            str(output_dir),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--run-human-agreement",
            "--dry-run",
        ]
    )

    assert rc == 0


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
    assert payload["completed_jobs"] == 1
    assert payload["total_jobs"] == 2
    assert payload["can_finalize"] is False
    assert payload["incomplete_jobs"] == {"pending": 1}
    assert [step["name"] for step in payload["steps"]] == [
        "auto_metrics",
        "tables",
        "figures",
        "human_audit_sample",
        "paper_artifacts",
    ]
    assert payload["planned_steps"] == [
        "auto_metrics",
        "tables",
        "figures",
        "human_audit_sample",
        "paper_artifacts",
    ]
    assert payload["step_log_dir"] == str(tmp_path / "out" / "logs" / "finalization")


def test_finalize_run_command_preserves_stdout_stderr_logs(tmp_path: Path) -> None:
    step = {
        "name": "failing step",
        "argv": [
            sys.executable,
            "-c",
            "import sys; print('visible stdout'); print('visible stderr', file=sys.stderr); sys.exit(3)",
        ],
    }

    result = finalize._run_command(step, dry_run=False, log_dir=tmp_path / "finalization_logs")

    assert result["status"] == "failed"
    assert result["returncode"] == 3
    assert Path(result["stdout_log"]).read_text(encoding="utf-8").strip() == "visible stdout"
    assert Path(result["stderr_log"]).read_text(encoding="utf-8").strip() == "visible stderr"


def test_finalize_run_steps_fail_fast_and_marks_later_steps_skipped(tmp_path: Path) -> None:
    commands = [
        {
            "name": "first",
            "argv": [
                sys.executable,
                "-c",
                "import sys; print('first failed'); sys.exit(2)",
            ],
        },
        {
            "name": "second",
            "argv": [
                sys.executable,
                "-c",
                "from pathlib import Path; Path('should_not_exist').write_text('bad')",
            ],
        },
    ]

    results = finalize._run_steps(commands, dry_run=False, log_dir=tmp_path / "logs")

    assert [result["status"] for result in results] == ["failed", "skipped"]
    assert results[1]["skip_reason"] == "previous step failed: first"


def test_finalize_only_adds_judge_when_requested(tmp_path: Path) -> None:
    args = finalize.argparse.Namespace(
        output_dir=str(tmp_path / "out"),
        gold="data/splits",
        run_judge=True,
        allow_mock_judge=False,
        run_human_agreement=False,
        allow_unlabeled_agreement=False,
        judge_config="configs/judge.yaml",
        audit_n=200,
    )

    commands = finalize.build_commands(args)
    judge_command = next(command for command in commands if command["name"] == "llm_judge")
    metrics_command = next(command for command in commands if command["name"] == "auto_metrics")

    assert [command["name"] for command in commands][:2] == ["llm_judge", "auto_metrics"]
    assert "--require-real" in judge_command["argv"]
    assert metrics_command["argv"][metrics_command["argv"].index("--judge-scores") + 1] == str(
        tmp_path / "out" / "judge_scores" / "judge_scores.jsonl"
    )


def test_finalize_can_allow_mock_judge_for_smoke(tmp_path: Path) -> None:
    args = finalize.argparse.Namespace(
        output_dir=str(tmp_path / "out"),
        gold="data/splits",
        run_judge=True,
        allow_mock_judge=True,
        run_human_agreement=False,
        allow_unlabeled_agreement=False,
        judge_config="configs/judge.yaml",
        audit_n=200,
    )

    commands = finalize.build_commands(args)
    judge_command = next(command for command in commands if command["name"] == "llm_judge")

    assert "--require-real" not in judge_command["argv"]


def test_finalize_paper_artifacts_uses_full_run_prefix(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    args = finalize.argparse.Namespace(
        plan=str(plan),
        output_dir=str(tmp_path / "out"),
        gold="data/splits",
        run_judge=False,
        allow_mock_judge=False,
        run_human_agreement=False,
        allow_unlabeled_agreement=False,
        judge_config="configs/judge.yaml",
        audit_n=200,
    )

    commands = finalize.build_commands(args)
    paper_command = next(command for command in commands if command["name"] == "paper_artifacts")

    assert "--artifact-prefix" in paper_command["argv"]
    assert paper_command["argv"][paper_command["argv"].index("--artifact-prefix") + 1] == str(tmp_path / "out")
    assert "--shard-plan" in paper_command["argv"]
    assert paper_command["argv"][paper_command["argv"].index("--shard-plan") + 1] == str(plan)


def test_finalize_allows_incomplete_tables_only_for_smoke(tmp_path: Path) -> None:
    base = {
        "output_dir": str(tmp_path / "out"),
        "gold": "data/splits",
        "run_judge": False,
        "allow_mock_judge": False,
        "run_human_agreement": False,
        "allow_unlabeled_agreement": False,
        "judge_config": "configs/judge.yaml",
        "audit_n": 200,
    }
    formal_args = finalize.argparse.Namespace(**base, allow_incomplete=False)
    smoke_args = finalize.argparse.Namespace(**base, allow_incomplete=True)

    formal_tables = next(command for command in finalize.build_commands(formal_args) if command["name"] == "tables")
    smoke_tables = next(command for command in finalize.build_commands(smoke_args) if command["name"] == "tables")

    assert "--allow-incomplete-tables" not in formal_tables["argv"]
    assert "--allow-incomplete-tables" in smoke_tables["argv"]


def test_finalize_human_audit_uses_full_run_prerequisite_paths(tmp_path: Path) -> None:
    args = finalize.argparse.Namespace(
        output_dir=str(tmp_path / "out"),
        gold="data/splits",
        run_judge=True,
        allow_mock_judge=False,
        run_human_agreement=False,
        allow_unlabeled_agreement=False,
        judge_config="configs/judge.yaml",
        audit_n=200,
    )

    commands = finalize.build_commands(args)
    audit_command = next(command for command in commands if command["name"] == "human_audit_sample")

    assert audit_command["argv"][audit_command["argv"].index("--records") + 1] == str(
        tmp_path / "out" / "metrics" / "record_auto_metrics.jsonl"
    )
    assert audit_command["argv"][audit_command["argv"].index("--judge-scores") + 1] == str(
        tmp_path / "out" / "judge_scores" / "judge_scores.jsonl"
    )
    assert audit_command["argv"][audit_command["argv"].index("--tables") + 1] == str(tmp_path / "out" / "tables")


def test_finalize_can_insert_human_agreement_before_paper_artifacts(tmp_path: Path) -> None:
    args = finalize.argparse.Namespace(
        output_dir=str(tmp_path / "out"),
        gold="data/splits",
        run_judge=False,
        allow_mock_judge=False,
        run_human_agreement=True,
        allow_unlabeled_agreement=True,
        judge_config="configs/judge.yaml",
        audit_n=200,
    )

    commands = finalize.build_commands(args)
    names = [command["name"] for command in commands]
    agreement_command = next(command for command in commands if command["name"] == "human_agreement")

    assert names[-2:] == ["human_agreement", "paper_artifacts"]
    assert agreement_command["argv"][agreement_command["argv"].index("--input") + 1] == str(
        tmp_path / "out" / "human_audit" / "human_audit_labeled.csv"
    )
    assert agreement_command["argv"][agreement_command["argv"].index("--output") + 1] == str(
        tmp_path / "out" / "human_audit" / "human_agreement_report.json"
    )
    assert agreement_command["argv"][agreement_command["argv"].index("--preference-mapping") + 1] == str(
        tmp_path / "out" / "human_audit" / "preference_mapping.json"
    )
    assert "--allow-unlabeled" in agreement_command["argv"]
