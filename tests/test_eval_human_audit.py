import pytest

from prism_tutor.eval.human_agreement import (
    build_agreement_report,
    cohen_kappa,
    resolve_pairwise_preferences,
    spearman_correlation,
)
from prism_tutor.eval.human_audit_sampler import blind_row_content_issues, sample_human_audit, validate_blind_rows


def test_human_audit_sampler_blinds_forbidden_fields():
    rows = [
        {
            "sample_id": f"s{i}",
            "dataset": "mathdial" if i < 4 else "bridge",
            "problem": "p",
            "student_answer": "a",
            "ground_truth": "g",
            "dialogue_context": "ctx",
            "candidate_response": "hint",
            "method": "ours",
            "selected_agents": ["solver"],
            "risk_bucket": "high" if i % 2 else "low",
            "leakage_conflict": i == 1,
        }
        for i in range(8)
    ]

    result = sample_human_audit(rows, target_n=4, dataset_targets={"mathdial": 2, "bridge": 2}, seed=7)

    assert result["manifest"]["actual_n"] == 4
    assert result["manifest"]["display_order_seed"] == 7
    assert result["manifest"]["display_order_sample_ids"] == [row["sample_id"] for row in result["blind_rows"]]
    assert [row["display_order"] for row in result["blind_rows"]] == [1, 2, 3, 4]
    assert all("method" not in row for row in result["blind_rows"])
    assert all("risk_bucket" not in row for row in result["blind_rows"])


def test_human_audit_sampler_builds_pairwise_blind_preference_mapping():
    rows = [
        {
            "sample_id": "s1",
            "dataset": "mathdial",
            "problem": "p",
            "student_answer": "a",
            "ground_truth": "g",
            "dialogue_context": "ctx",
            "candidate_response": "ours hint",
            "method": "ours_full",
            "internal_correctness": 1.0,
        },
        {
            "sample_id": "s1",
            "dataset": "mathdial",
            "problem": "p",
            "student_answer": "a",
            "ground_truth": "g",
            "dialogue_context": "ctx",
            "candidate_response": "baseline hint",
            "method": "fixed_4",
            "internal_correctness": 0.0,
        },
    ]

    result = sample_human_audit(rows, target_n=1, dataset_targets={"mathdial": 1}, seed=5)
    blind = result["blind_rows"][0]
    mapping = result["preference_mapping"][0]

    assert blind["candidate_a_response"] in {"ours hint", "baseline hint"}
    assert blind["candidate_b_response"] in {"ours hint", "baseline hint"}
    assert blind["candidate_a_response"] != blind["candidate_b_response"]
    assert "candidate_a_method" not in blind
    assert "candidate_b_method" not in blind
    assert mapping["audit_id"] == blind["audit_id"]
    assert {mapping["candidate_a_method"], mapping["candidate_b_method"]} == {"ours_full", "fixed_4"}
    assert result["manifest"]["pairwise_preference_rows"] == 1


def test_human_audit_display_order_is_seeded_and_not_dataset_blocked():
    rows = [
        {
            "sample_id": f"m{i}",
            "dataset": "mathdial",
            "problem": "p",
            "student_answer": "a",
            "ground_truth": "g",
            "candidate_response": "hint",
        }
        for i in range(6)
    ] + [
        {
            "sample_id": f"b{i}",
            "dataset": "bridge",
            "problem": "p",
            "student_answer": "a",
            "ground_truth": "g",
            "candidate_response": "hint",
        }
        for i in range(6)
    ]

    first = sample_human_audit(rows, target_n=8, dataset_targets={"mathdial": 4, "bridge": 4}, seed=11)
    second = sample_human_audit(rows, target_n=8, dataset_targets={"mathdial": 4, "bridge": 4}, seed=11)

    first_order = first["manifest"]["display_order_sample_ids"]
    assert first_order == second["manifest"]["display_order_sample_ids"]
    assert [row["display_order"] for row in first["blind_rows"]] == list(range(1, 9))
    assert [row["sample_id"] for row in first["blind_rows"]] == first_order
    assert [row["dataset"] for row in first["blind_rows"]] != ["mathdial"] * 4 + ["bridge"] * 4


def test_validate_blind_rows_rejects_method_leak():
    with pytest.raises(ValueError):
        validate_blind_rows([{"sample_id": "s1", "display_order": 1, "method": "ours"}])


def test_validate_blind_rows_rejects_invalid_display_order():
    with pytest.raises(ValueError):
        validate_blind_rows([{"sample_id": "s1", "display_order": 2}])


def test_blind_row_content_issues_report_missing_annotation_context():
    issues = blind_row_content_issues(
        [
            {"audit_id": "A0001", "sample_id": "s1", "display_order": 1, "problem": "", "candidate_response": None}
        ]
    )

    assert issues == [{"row_index": 1, "audit_id": "A0001", "sample_id": "s1", "missing": ["problem", "candidate_response"]}]


def test_human_agreement_report():
    rows = [
        {"sample_id": "s1", "annotator_id": "a", "human_quality_score": "4", "human_leakage_label": "no", "human_preference": "ours"},
        {"sample_id": "s1", "annotator_id": "b", "human_quality_score": "5", "human_leakage_label": "no", "human_preference": "ours"},
        {"sample_id": "s2", "annotator_id": "a", "human_quality_score": "2", "human_leakage_label": "yes", "human_preference": "tie"},
        {"sample_id": "s2", "annotator_id": "b", "human_quality_score": "1", "human_leakage_label": "no", "human_preference": "baseline"},
    ]

    report = build_agreement_report(rows)

    assert report["leakage_kappa"]["n"] == 2
    assert report["quality_spearman"]["spearman"] == 1.0
    assert report["preference"]["n"] == 4
    assert cohen_kappa(["a"], ["a"])["kappa"] == 1.0
    assert spearman_correlation([1, 2], [2, 3])["spearman"] == 1.0


def test_human_agreement_report_accepts_pairwise_ab_preference():
    rows = [
        {
            "sample_id": "s1",
            "annotator_id": "a",
            "human_quality_score": "4",
            "human_leakage_label": "no",
            "human_preference": "",
            "human_preference_ab": "A",
        },
        {
            "sample_id": "s1",
            "annotator_id": "b",
            "human_quality_score": "5",
            "human_leakage_label": "no",
            "human_preference": "",
            "human_preference_ab": "B",
        },
    ]

    report = build_agreement_report(rows)

    assert report["preference"]["n"] == 2
    assert report["preference"]["candidate_a_rate"] == 0.5
    assert report["preference"]["candidate_b_rate"] == 0.5


def test_human_agreement_resolves_pairwise_ab_mapping_to_ours_baseline():
    rows = [
        {
            "audit_id": "A0001",
            "sample_id": "s1",
            "dataset": "mathdial",
            "annotator_id": "a",
            "human_quality_score": "4",
            "human_leakage_label": "no",
            "human_preference": "",
            "human_preference_ab": "A",
        },
        {
            "audit_id": "A0002",
            "sample_id": "s2",
            "dataset": "mathdial",
            "annotator_id": "a",
            "human_quality_score": "3",
            "human_leakage_label": "yes",
            "human_preference": "",
            "human_preference_ab": "B",
        },
        {
            "audit_id": "A0003",
            "sample_id": "s3",
            "dataset": "mathdial",
            "annotator_id": "a",
            "human_quality_score": "5",
            "human_leakage_label": "no",
            "human_preference": "",
            "human_preference_ab": "tie",
        },
    ]
    mapping_rows = [
        {"audit_id": "A0001", "candidate_a_is_ours": True, "candidate_b_is_ours": False},
        {"audit_id": "A0002", "candidate_a_is_ours": True, "candidate_b_is_ours": False},
    ]

    resolved, mapping_report = resolve_pairwise_preferences(rows, mapping_rows)
    report = build_agreement_report(resolved)

    assert mapping_report["mapped_count"] == 2
    assert mapping_report["tie_count"] == 1
    assert report["preference"]["n"] == 3
    assert report["preference"]["ours_win_rate"] == pytest.approx(1 / 3)
    assert report["preference"]["candidate_a_rate"] == pytest.approx(1 / 3)
    assert report["preference"]["candidate_b_rate"] == pytest.approx(1 / 3)


def test_human_agreement_report_requires_core_annotation_columns():
    report = build_agreement_report(
        [
            {
                "sample_id": "s1",
                "annotator_id": "a",
                "human_quality_score": "4",
            }
        ]
    )

    assert "schema_error" in report
    assert "human_leakage_label" in report["schema_error"]
    assert "human_preference" in report["schema_error"]
