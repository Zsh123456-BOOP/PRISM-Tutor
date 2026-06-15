from __future__ import annotations

import json
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "00_download_datasets.py"
SPEC = importlib.util.spec_from_file_location("download_datasets_script", SCRIPT_PATH)
download_datasets = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(download_datasets)


def test_download_datasets_dry_run_and_manual_report(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    config_path = tmp_path / "sources.yaml"
    config_path.write_text(
        json.dumps(
            {
                "report_path": str(report_path),
                "datasets": {
                    "mathdial": {
                        "source_type": "huggingface_files",
                        "repo_id": "eth-nlped/mathdial",
                        "repo_type": "dataset",
                        "files": ["train.jsonl"],
                        "raw_path": str(tmp_path / "raw" / "mathdial"),
                    },
                    "bridge": {
                        "source_type": "manual_required",
                        "raw_path": str(tmp_path / "raw" / "bridge"),
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    report = download_datasets.run_download(str(config_path), datasets=None, dry_run=True, strict=True)
    statuses = {item["dataset"]: item["status"] for item in report["datasets"]}
    assert statuses["mathdial"] == "dry_run"
    assert statuses["bridge"] == "manual_required"
    assert report["all_ready"] is False
    assert report_path.exists()
