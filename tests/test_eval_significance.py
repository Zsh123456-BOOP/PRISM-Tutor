from prism_tutor.eval.figure_builder import pareto_points, risk_bucket_counts, write_text_pdf
from prism_tutor.eval.significance import compare_methods, holm_correction
from prism_tutor.eval.table_builder import rows_to_latex, summarize_table


def test_compare_methods_uses_only_paired_samples():
    rows = [
        {"sample_id": "s1", "method": "ours", "score": 4.0},
        {"sample_id": "s1", "method": "base", "score": 3.0},
        {"sample_id": "s2", "method": "ours", "score": 5.0},
        {"sample_id": "s2", "method": "base", "score": 2.0},
        {"sample_id": "s3", "method": "ours", "score": 1.0},
    ]

    result = compare_methods(rows, "score", "ours", "base")

    assert result["n"] == 2
    assert result["mean_diff"] == 2.0
    assert result["test"] == "wilcoxon_signed_rank"


def test_binary_compare_and_holm_correction():
    rows = [
        {"sample_id": "s1", "method": "ours", "leakage": 0},
        {"sample_id": "s1", "method": "base", "leakage": 1},
        {"sample_id": "s2", "method": "ours", "leakage": 0},
        {"sample_id": "s2", "method": "base", "leakage": 0},
    ]

    result = compare_methods(rows, "leakage", "ours", "base", binary=True)
    corrected = holm_correction([result])

    assert result["test"] == "mcnemar"
    assert corrected[0]["holm_p_value"] == result["p_value"]


def test_table_and_pdf_builders(tmp_path):
    rows = [
        {"dataset": "mathdial", "method": "ours", "score": 4.0, "tokens": 10, "risk_bucket": "low"},
        {"dataset": "mathdial", "method": "ours", "score": 2.0, "tokens": 20, "risk_bucket": "high"},
    ]
    table = summarize_table(rows, ["score"])
    latex = rows_to_latex(table, caption="Main results")
    pdf_path = tmp_path / "figure.pdf"
    write_text_pdf(pdf_path, "Figure 1", ["system overview placeholder"])

    assert table[0]["score_mean"] == 3.0
    assert "\\begin{table}" in latex
    assert pareto_points(rows, "score", "tokens")[0]["quality"] == 4.0
    assert risk_bucket_counts(rows) == {"high": 1, "low": 1}
    assert pdf_path.read_bytes().startswith(b"%PDF")
