from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "08_human_agreement.py"
SPEC = importlib.util.spec_from_file_location("human_agreement_script", SCRIPT_PATH)
human_agreement_script = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(human_agreement_script)


def test_human_agreement_allow_unlabeled_uses_input_directory_blind_csv(tmp_path: Path) -> None:
    audit_dir = tmp_path / "outputs" / "full_run" / "human_audit"
    audit_dir.mkdir(parents=True)
    blind = audit_dir / "human_audit_blind.csv"
    blind.write_text(
        "\n".join(
            [
                "sample_id,annotator_id,human_quality_score,human_leakage_label,human_preference",
                "s1,a,4,no,ours",
                "s1,b,5,no,ours",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = audit_dir / "human_agreement_report.json"

    rc = human_agreement_script.main(
        [
            "--input",
            str(audit_dir / "human_audit_labeled.csv"),
            "--output",
            str(output),
            "--allow-unlabeled",
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert rc == 0
    assert report["leakage_kappa"]["n"] == 1


def test_human_agreement_cli_returns_nonzero_for_missing_core_columns(tmp_path: Path) -> None:
    audit_dir = tmp_path / "outputs" / "full_run" / "human_audit"
    audit_dir.mkdir(parents=True)
    labeled = audit_dir / "human_audit_labeled.csv"
    labeled.write_text(
        "\n".join(
            [
                "sample_id,annotator_id,human_quality_score",
                "s1,a,4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = audit_dir / "human_agreement_report.json"

    rc = human_agreement_script.main(["--input", str(labeled), "--output", str(output)])
    report = json.loads(output.read_text(encoding="utf-8"))

    assert rc == 1
    assert "schema_error" in report


def test_human_agreement_cli_allow_unlabeled_writes_schema_error_without_failing(tmp_path: Path) -> None:
    audit_dir = tmp_path / "outputs" / "full_run" / "human_audit"
    audit_dir.mkdir(parents=True)
    blind = audit_dir / "human_audit_blind.csv"
    blind.write_text("sample_id,annotator_id\ns1,a\n", encoding="utf-8")
    output = audit_dir / "human_agreement_report.json"

    rc = human_agreement_script.main(
        [
            "--input",
            str(audit_dir / "human_audit_labeled.csv"),
            "--output",
            str(output),
            "--allow-unlabeled",
        ]
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert rc == 0
    assert "schema_error" in report


def test_human_agreement_formal_fails_when_no_valid_labels(tmp_path: Path) -> None:
    audit_dir = tmp_path / "outputs" / "full_run" / "human_audit"
    audit_dir.mkdir(parents=True)
    labeled = audit_dir / "human_audit_labeled.csv"
    labeled.write_text(
        "\n".join(
            [
                "sample_id,annotator_id,human_quality_score,human_leakage_label,human_preference",
                "s1,a,,,",
                "s1,b,,,",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = audit_dir / "human_agreement_report.json"

    rc = human_agreement_script.main(["--input", str(labeled), "--output", str(output)])
    report = json.loads(output.read_text(encoding="utf-8"))

    assert rc == 1
    assert report["status"] == "failed"
    assert "too_few_quality_pairs" in report["formal_gate"]["failures"]
    assert "too_few_leakage_pairs" in report["formal_gate"]["failures"]
    assert "too_few_preferences" in report["formal_gate"]["failures"]


def test_human_agreement_formal_fails_when_no_two_annotator_overlap(tmp_path: Path) -> None:
    audit_dir = tmp_path / "outputs" / "full_run" / "human_audit"
    audit_dir.mkdir(parents=True)
    labeled = audit_dir / "human_audit_labeled.csv"
    labeled.write_text(
        "\n".join(
            [
                "sample_id,annotator_id,human_quality_score,human_leakage_label,human_preference",
                "s1,a,4,no,ours",
                "s2,a,5,no,tie",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = audit_dir / "human_agreement_report.json"

    rc = human_agreement_script.main(["--input", str(labeled), "--output", str(output)])
    report = json.loads(output.read_text(encoding="utf-8"))

    assert rc == 1
    assert report["status"] == "failed"
    assert "too_few_quality_pairs" in report["formal_gate"]["failures"]
    assert "too_few_leakage_pairs" in report["formal_gate"]["failures"]


def test_human_agreement_cli_resolves_pairwise_ab_preference_mapping(tmp_path: Path) -> None:
    audit_dir = tmp_path / "outputs" / "full_run" / "human_audit"
    audit_dir.mkdir(parents=True)
    labeled = audit_dir / "human_audit_labeled.csv"
    labeled.write_text(
        "\n".join(
            [
                "audit_id,sample_id,dataset,annotator_id,human_quality_score,human_leakage_label,human_preference,human_preference_ab",
                "A0001,s1,mathdial,a,4,no,,A",
                "A0001,s1,mathdial,b,5,no,,A",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (audit_dir / "preference_mapping.json").write_text(
        json.dumps(
            [
                {
                    "audit_id": "A0001",
                    "sample_id": "s1",
                    "dataset": "mathdial",
                    "candidate_a_is_ours": False,
                    "candidate_b_is_ours": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    output = audit_dir / "human_agreement_report.json"

    rc = human_agreement_script.main(
        [
            "--input",
            str(labeled),
            "--output",
            str(output),
            "--min-quality-pairs",
            "1",
            "--min-leakage-pairs",
            "1",
            "--min-preferences",
            "1",
        ]
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert rc == 0
    assert report["preference"]["n"] == 2
    assert report["preference"]["ours_win_rate"] == 0.0
    assert report["preference"]["candidate_a_rate"] == 1.0
    assert report["preference_mapping"]["mapped_count"] == 2
    assert report["status"] == "passed"


def test_human_agreement_cli_formal_fails_for_ab_labels_without_mapping(tmp_path: Path) -> None:
    audit_dir = tmp_path / "outputs" / "full_run" / "human_audit"
    audit_dir.mkdir(parents=True)
    labeled = audit_dir / "human_audit_labeled.csv"
    labeled.write_text(
        "\n".join(
            [
                "audit_id,sample_id,dataset,annotator_id,human_quality_score,human_leakage_label,human_preference,human_preference_ab",
                "A0001,s1,mathdial,a,4,no,,A",
                "A0001,s1,mathdial,b,5,no,,B",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = audit_dir / "human_agreement_report.json"

    rc = human_agreement_script.main(
        [
            "--input",
            str(labeled),
            "--output",
            str(output),
            "--min-quality-pairs",
            "1",
            "--min-leakage-pairs",
            "1",
            "--min-preferences",
            "1",
        ]
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert rc == 1
    assert report["preference_mapping"]["missing_mapping_file"].endswith("preference_mapping.json")
    assert "unresolved_pairwise_preference_mapping" in report["formal_gate"]["failures"]


def test_human_agreement_cli_allow_unlabeled_allows_missing_ab_mapping(tmp_path: Path) -> None:
    audit_dir = tmp_path / "outputs" / "full_run" / "human_audit"
    audit_dir.mkdir(parents=True)
    labeled = audit_dir / "human_audit_labeled.csv"
    labeled.write_text(
        "\n".join(
            [
                "audit_id,sample_id,dataset,annotator_id,human_quality_score,human_leakage_label,human_preference,human_preference_ab",
                "A0001,s1,mathdial,a,4,no,,A",
                "A0001,s1,mathdial,b,5,no,,B",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = audit_dir / "human_agreement_report.json"

    rc = human_agreement_script.main(
        [
            "--input",
            str(labeled),
            "--output",
            str(output),
            "--allow-unlabeled",
            "--min-quality-pairs",
            "1",
            "--min-leakage-pairs",
            "1",
            "--min-preferences",
            "1",
        ]
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert rc == 0
    assert report["status"] == "failed"
    assert "unresolved_pairwise_preference_mapping" in report["formal_gate"]["failures"]
