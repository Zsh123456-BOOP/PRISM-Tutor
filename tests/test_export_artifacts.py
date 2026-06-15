import json

from prism_tutor.export.artifact_exporter import export_paper_artifacts
from prism_tutor.export.reproducibility_checklist import build_reproducibility_checklist


def test_reproducibility_checklist_marks_missing_and_secret_hits(tmp_path):
    (tmp_path / "outputs").mkdir()
    secret_file = tmp_path / "outputs" / "judge_metadata.json"
    secret_file.write_text('{"api_key": "bad"}', encoding="utf-8")

    checklist = build_reproducibility_checklist(
        tmp_path,
        ["outputs/judge_metadata.json", "outputs/missing.json"],
    )

    assert checklist["status"] == "failed"
    assert any(check["name"] == "outputs/missing.json" and check["status"] == "failed" for check in checklist["checks"])
    assert any(check.get("hits") for check in checklist["checks"] if check["name"] == "plaintext_secret_scan")


def test_export_paper_artifacts_writes_index_summary_and_manifest(tmp_path):
    (tmp_path / "outputs" / "metrics").mkdir(parents=True)
    out = tmp_path / "paper_artifacts"
    files = export_paper_artifacts(
        tmp_path,
        out,
        experiment_manifests=[{"experiment": "exp0", "methods": ["b0", "ours"]}],
        required_paths=["outputs/metrics"],
    )

    assert files["experiment_summary"].exists()
    assert "PRISM-Tutor" in files["experiment_summary"].read_text(encoding="utf-8")
    manifest = json.loads(files["experiment_manifest"].read_text(encoding="utf-8"))
    assert "exp0" in manifest["experiments"]
    assert "exp1" in manifest["missing_experiments"]
    assert "`outputs/metrics/significance_tests.json`" in files["artifact_index"].read_text(encoding="utf-8")


def test_export_paper_artifacts_uses_run_local_artifact_prefix(tmp_path):
    (tmp_path / "outputs" / "full_run" / "metrics").mkdir(parents=True)
    out = tmp_path / "paper_artifacts"

    files = export_paper_artifacts(
        tmp_path,
        out,
        required_paths=["outputs/full_run/metrics"],
        artifact_prefix="outputs/full_run",
    )

    checklist = files["reproducibility_checklist"].read_text(encoding="utf-8")
    artifact_index = files["artifact_index"].read_text(encoding="utf-8")
    assert "outputs/full_run/metrics: passed" in checklist
    assert "`outputs/full_run/metrics/significance_tests.json`" in artifact_index
