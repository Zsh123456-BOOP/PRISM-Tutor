from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "05_make_tables.py"
SPEC = importlib.util.spec_from_file_location("make_tables_script", SCRIPT_PATH)
make_tables_script = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(make_tables_script)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_make_tables_writes_all_task_card_tables_and_significance(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics" / "record_auto_metrics.jsonl"
    rows = [
        {"dataset": "mathdial", "sample_id": "s1", "method": "ours_full", "internal_correctness": 1.0, "total_tokens": 100, "rule_leakage": 0, "judge_leakage": 0, "final_leakage": 0, "leakage_conflict": 0},
        {"dataset": "mathdial", "sample_id": "s1", "method": "fixed_4", "internal_correctness": 0.0, "total_tokens": 120, "rule_leakage": 0, "judge_leakage": 1, "final_leakage": 1, "leakage_conflict": 1},
        {"dataset": "mathdial", "sample_id": "s1", "method": "generic_sparse", "internal_correctness": 0.0, "total_tokens": 80, "rule_leakage": 0},
        {"dataset": "mathdial", "sample_id": "s1", "method": "debate", "internal_correctness": 0.0, "total_tokens": 160, "rule_leakage": 0},
        {"dataset": "mathdial", "sample_id": "s1", "method": "difficulty_routing", "internal_correctness": 0.0, "total_tokens": 90, "rule_leakage": 0},
        {"dataset": "mathdial", "sample_id": "s1", "method": "ours_routing_budget", "internal_correctness": 1.0, "total_tokens": 95, "rule_leakage": 0},
        {"dataset": "mathdial", "sample_id": "s2", "method": "ours_full", "internal_correctness": 1.0, "total_tokens": 110, "rule_leakage": 0, "judge_leakage": 0, "final_leakage": 0, "leakage_conflict": 0},
        {"dataset": "mathdial", "sample_id": "s2", "method": "fixed_4", "internal_correctness": 0.0, "total_tokens": 130, "rule_leakage": 0, "judge_leakage": 1, "final_leakage": 1, "leakage_conflict": 1},
        {"dataset": "mathdial", "sample_id": "s3", "method": "fixed_4__noise0p2__budget1000", "internal_correctness": 0.0, "total_tokens": 140, "final_leakage": 1},
        {"dataset": "mathdial", "sample_id": "s2", "method": "random_routing", "routing_f1": 0.1, "total_tokens": 70},
        {"dataset": "mathdial", "sample_id": "s2", "method": "single_round", "rounds": 1, "total_tokens": 50},
        {"dataset": "mathdial", "sample_id": "s2", "method": "two_phase_commit", "state_conflict_rate": 0.0, "total_tokens": 90},
        {"dataset": "mathdial", "sample_id": "s2", "method": "ablate_risk_estimator", "internal_correctness": 0.5, "total_tokens": 100},
    ]
    _write_jsonl(metrics, rows)
    out = tmp_path / "tables"

    rc = make_tables_script.main(["--record_metrics", str(metrics), "--output_dir", str(out)])

    assert rc == 0
    for stem in [
        "table1_main_results",
        "table2_routing",
        "table3_budget",
        "table4_state_commit",
        "table5_ablation",
        "table6_robustness",
    ]:
        assert (out / f"{stem}.csv").exists()
        assert (out / f"{stem}.tex").exists()
    routing_rows = list(csv.DictReader((out / "table2_routing.csv").open(encoding="utf-8")))
    routing_methods = {row["method"] for row in routing_rows}
    main_rows = list(csv.DictReader((out / "table1_main_results.csv").open(encoding="utf-8")))
    robustness_rows = list(csv.DictReader((out / "table6_robustness.csv").open(encoding="utf-8")))
    robustness_methods = {row["method"] for row in robustness_rows}
    assert "random_routing" in routing_methods
    assert "single_round" not in routing_methods
    assert "fixed_4__noise0p2__budget1000" in robustness_methods
    assert any(row["method"] == "fixed_4" and row["final_leakage_mean"] == "1.0" for row in main_rows)
    significance = json.loads((metrics.parent / "significance_tests.json").read_text(encoding="utf-8"))
    assert any(row["method_a"] == "ours_full" and row["method_b"] == "fixed_4" for row in significance)
    assert any(row["metric"] == "final_leakage" and row["test"] == "mcnemar" for row in significance)
