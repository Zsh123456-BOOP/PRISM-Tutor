import pytest

from prism_tutor.eval.human_agreement import build_agreement_report, cohen_kappa, spearman_correlation
from prism_tutor.eval.human_audit_sampler import sample_human_audit, validate_blind_rows


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
