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
