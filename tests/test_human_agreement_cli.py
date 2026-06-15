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
