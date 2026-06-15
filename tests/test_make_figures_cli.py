from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "06_make_figures.py"
SPEC = importlib.util.spec_from_file_location("make_figures_script", SCRIPT_PATH)
make_figures = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(make_figures)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_make_figures_writes_manifest_and_all_pdfs(tmp_path: Path) -> None:
    records = tmp_path / "record_auto_metrics.jsonl"
    _write_jsonl(
        records,
        [
            {
                "dataset": "mathdial",
                "sample_id": "s1",
                "method": "ours_full",
                "internal_correctness": 1.0,
                "total_tokens": 90,
                "risk_bucket": "low",
                "agent_calls": 2,
                "state_conflict_rate": 0.0,
            },
            {
                "dataset": "mathdial",
                "sample_id": "s2",
                "method": "fixed_4",
                "internal_correctness": 0.0,
                "total_tokens": 140,
                "risk_bucket": "high",
                "agent_calls": 5,
                "state_conflict_rate": 0.5,
            },
        ],
    )

    output = tmp_path / "figures"
    rc = make_figures.main(["--record_metrics", str(records), "--output_dir", str(output)])

    manifest = json.loads((output / "figure_manifest.json").read_text(encoding="utf-8"))
    assert rc == 0
    assert manifest["status"] == "completed"
    assert manifest["figures"]["figure2_quality_token_pareto.pdf"]["points"] == 2
    assert manifest["figures"]["figure4_agent_call_distribution.pdf"]["distribution"] == {"2": 1, "5": 1}
    for name in manifest["figures"]:
        if name.endswith(".pdf"):
            assert (output / name).read_bytes().startswith(b"%PDF")


def test_make_figures_fails_on_missing_required_columns(tmp_path: Path, capsys) -> None:
    records = tmp_path / "record_auto_metrics.jsonl"
    _write_jsonl(records, [{"dataset": "mathdial", "sample_id": "s1", "method": "ours_full"}])

    output = tmp_path / "figures"
    rc = make_figures.main(["--record_metrics", str(records), "--output_dir", str(output)])

    captured = capsys.readouterr()
    assert rc == 2
    assert "missing required columns" in captured.err
    assert not (output / "figure2_quality_token_pareto.pdf").exists()
    assert not (output / "figure_manifest.json").exists()
