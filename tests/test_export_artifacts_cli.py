from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "09_export_paper_artifacts.py"
SPEC = importlib.util.spec_from_file_location("export_paper_artifacts_script", SCRIPT_PATH)
export_script = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(export_script)


def test_export_cli_returns_nonzero_when_artifacts_are_incomplete(tmp_path: Path, capsys) -> None:
    output_dir = tmp_path / "paper_artifacts"

    rc = export_script.main(["--root", str(tmp_path), "--output_dir", str(output_dir)])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 2
    assert payload["status"] == "failed"
    assert (output_dir / "experiment_summary.md").exists()
    assert (output_dir / "reproducibility_checklist.json").exists()


def test_export_cli_allows_failed_checklist_only_for_smoke(tmp_path: Path, capsys) -> None:
    output_dir = tmp_path / "paper_artifacts"

    rc = export_script.main(
        [
            "--root",
            str(tmp_path),
            "--output_dir",
            str(output_dir),
            "--allow-failed-checklist",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 0
    assert payload["status"] == "failed"


def test_export_cli_fails_when_explicit_shard_plan_is_missing(tmp_path: Path) -> None:
    try:
        export_script.main(
            [
                "--root",
                str(tmp_path),
                "--output_dir",
                str(tmp_path / "paper_artifacts"),
                "--shard-plan",
                str(tmp_path / "missing_plan.json"),
            ]
        )
    except SystemExit as exc:
        assert "Missing shard plan" in str(exc)
    else:
        raise AssertionError("Expected missing shard plan to fail")


def test_export_cli_fails_when_explicit_shard_plan_is_invalid(tmp_path: Path) -> None:
    bad_plan = tmp_path / "bad_plan.json"
    bad_plan.write_text("{not json", encoding="utf-8")

    try:
        export_script.main(
            [
                "--root",
                str(tmp_path),
                "--output_dir",
                str(tmp_path / "paper_artifacts"),
                "--shard-plan",
                str(bad_plan),
            ]
        )
    except SystemExit as exc:
        assert "Invalid shard plan JSON" in str(exc)
    else:
        raise AssertionError("Expected invalid shard plan to fail")
