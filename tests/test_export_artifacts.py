import json

from prism_tutor.export.artifact_exporter import default_required_paths, export_paper_artifacts
from prism_tutor.export.reproducibility_checklist import build_reproducibility_checklist, checklist_to_markdown


def _formal_judge_metadata(**overrides):
    payload = {
        "actual_model": "deepseek-v4-pro",
        "api_date": "2026-06-16",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 768,
        "prompt_version": "judge-v1",
        "dry_run": False,
        "parsed_count": 1,
        "error_count": 0,
        "output_rows": 1,
    }
    payload.update(overrides)
    return payload


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


def test_reproducibility_checklist_scans_required_directories_for_secrets(tmp_path):
    generations = tmp_path / "outputs" / "full_run" / "generations"
    generations.mkdir(parents=True)
    (generations / "rows.jsonl").write_text('{"authorization": "bearer sk-test"}\n', encoding="utf-8")

    checklist = build_reproducibility_checklist(tmp_path, ["outputs/full_run/generations"])
    secret_check = next(check for check in checklist["checks"] if check["name"] == "plaintext_secret_scan")

    assert checklist["status"] == "failed"
    assert secret_check["hits"] == [{"path": "outputs/full_run/generations/rows.jsonl", "pattern": "bearer "}]


def test_reproducibility_checklist_records_config_judge_gpu_and_paths(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "outputs" / "full_run" / "judge_scores").mkdir(parents=True)
    (tmp_path / "outputs" / "full_run" / "logs").mkdir(parents=True)
    (tmp_path / "configs" / "default.yaml").write_text(
        "\n".join(
            [
                "seed: 42",
                "model:",
                "  generator: Qwen/Qwen3-8B",
                "  enable_thinking: false",
                "  quantization: null",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "outputs" / "full_run" / "judge_scores" / "judge_metadata.json").write_text(
        json.dumps(_formal_judge_metadata()),
        encoding="utf-8",
    )

    checklist = build_reproducibility_checklist(
        tmp_path,
        [
            "outputs/full_run/logs",
            "outputs/full_run/judge_scores/judge_metadata.json",
        ],
    )
    markdown = checklist_to_markdown(checklist)

    assert checklist["config"]["seed"] == 42
    assert checklist["model"]["generator"] == "Qwen/Qwen3-8B"
    assert checklist["judge"]["metadata_present"] is True
    assert checklist["judge"]["metadata"]["actual_model"] == "deepseek-v4-pro"
    assert any(item["path"] == "outputs/full_run/logs" and item["kind"] == "directory" for item in checklist["data_and_logs"])
    assert "## GPU" in markdown
    assert "Seed: `42`" in markdown
    assert "Metadata present: `True`" in markdown


def test_reproducibility_checklist_requires_judge_metadata_schema_and_raw_responses(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "default.yaml").write_text(
        "\n".join(
            [
                "seed: 42",
                "model:",
                "  generator: Qwen/Qwen3-8B",
                "  enable_thinking: false",
                "  quantization: null",
            ]
        ),
        encoding="utf-8",
    )
    judge_dir = tmp_path / "outputs" / "full_run" / "judge_scores"
    judge_dir.mkdir(parents=True)
    (judge_dir / "judge_metadata.json").write_text("{}", encoding="utf-8")

    failed = build_reproducibility_checklist(
        tmp_path,
        [
            "outputs/full_run/judge_scores/judge_metadata.json",
            "outputs/full_run/judge_scores/raw/judge_raw.jsonl",
        ],
    )
    schema_check = next(check for check in failed["checks"] if check["name"] == "judge_metadata_schema")

    assert failed["status"] == "failed"
    assert "actual_model" in schema_check["missing_fields"]
    assert any(
        check["name"] == "outputs/full_run/judge_scores/raw/judge_raw.jsonl" and check["status"] == "failed"
        for check in failed["checks"]
    )

    raw_dir = judge_dir / "raw"
    raw_dir.mkdir()
    (raw_dir / "judge_raw.jsonl").write_text('{"raw_response": "{}"}\n', encoding="utf-8")
    (judge_dir / "judge_metadata.json").write_text(
        json.dumps(_formal_judge_metadata()),
        encoding="utf-8",
    )
    passed = build_reproducibility_checklist(
        tmp_path,
        [
            "outputs/full_run/judge_scores/judge_metadata.json",
            "outputs/full_run/judge_scores/raw/judge_raw.jsonl",
        ],
    )

    assert next(check for check in passed["checks"] if check["name"] == "judge_metadata_schema")["status"] == "passed"
    assert all(check["status"] == "passed" for check in passed["checks"] if check["name"].startswith("outputs/full_run/judge_scores"))


def test_reproducibility_checklist_rejects_mock_or_failed_judge_metadata(tmp_path):
    judge_dir = tmp_path / "outputs" / "full_run" / "judge_scores"
    raw_dir = judge_dir / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "judge_raw.jsonl").write_text('{"raw_response": "{}"}\n', encoding="utf-8")
    (judge_dir / "judge_metadata.json").write_text(
        json.dumps(
            _formal_judge_metadata(
                actual_model="mock-judge",
                dry_run=True,
                parsed_count=9,
                error_count=1,
                output_rows=10,
            )
        ),
        encoding="utf-8",
    )

    checklist = build_reproducibility_checklist(
        tmp_path,
        [
            "outputs/full_run/judge_scores/judge_metadata.json",
            "outputs/full_run/judge_scores/raw/judge_raw.jsonl",
        ],
    )
    schema_check = next(check for check in checklist["checks"] if check["name"] == "judge_metadata_schema")

    assert checklist["status"] == "failed"
    assert schema_check["invalid_reasons"] == [
        "dry_run_true",
        "mock_actual_model",
        "judge_errors_present",
        "parsed_count_mismatch",
    ]


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
    summary = files["experiment_summary"].read_text(encoding="utf-8")
    assert "PRISM-Tutor" in summary
    assert "Final artifact status: **failed**" in summary
    assert "Experiment manifest status: **failed**" in summary
    manifest = json.loads(files["experiment_manifest"].read_text(encoding="utf-8"))
    assert manifest["artifact_status"] == "failed"
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


def test_default_paper_artifact_paths_require_concrete_tables_figures_and_metrics(tmp_path):
    required = default_required_paths("outputs/full_run")
    assert "outputs/full_run/metrics/record_auto_metrics.jsonl" in required
    assert "outputs/full_run/metrics/significance_tests.json" in required
    assert "outputs/full_run/tables/table1_main_results.csv" in required
    assert "outputs/full_run/tables/table6_robustness.tex" in required
    assert "outputs/full_run/figures/figure2_quality_token_pareto.pdf" in required
    assert "outputs/full_run/figures/figure_manifest.json" in required
    assert "outputs/full_run/human_audit/human_audit_blind.csv" in required
    assert "outputs/full_run/human_audit/preference_mapping.json" in required

    for rel in ["outputs/full_run/metrics", "outputs/full_run/tables", "outputs/full_run/figures"]:
        (tmp_path / rel).mkdir(parents=True, exist_ok=True)

    checklist = build_reproducibility_checklist(tmp_path, required)

    assert checklist["status"] == "failed"
    assert any(
        check["name"] == "outputs/full_run/figures/figure_manifest.json" and check["status"] == "failed"
        for check in checklist["checks"]
    )
    assert any(
        check["name"] == "outputs/full_run/tables/table1_main_results.csv" and check["status"] == "failed"
        for check in checklist["checks"]
    )


def test_checklist_fails_on_failed_artifact_content(tmp_path):
    table = tmp_path / "outputs" / "full_run" / "tables" / "table1_main_results.csv"
    figure_manifest = tmp_path / "outputs" / "full_run" / "figures" / "figure_manifest.json"
    blind_csv = tmp_path / "outputs" / "full_run" / "human_audit" / "human_audit_blind.csv"
    sampling_manifest = tmp_path / "outputs" / "full_run" / "human_audit" / "sampling_manifest.json"
    preference_mapping = tmp_path / "outputs" / "full_run" / "human_audit" / "preference_mapping.json"
    agreement = tmp_path / "outputs" / "full_run" / "human_audit" / "human_agreement_report.json"
    table.parent.mkdir(parents=True)
    figure_manifest.parent.mkdir(parents=True)
    agreement.parent.mkdir(parents=True)
    table.write_text("dataset,method,n\n", encoding="utf-8")
    figure_manifest.write_text(json.dumps({"status": "failed"}), encoding="utf-8")
    blind_csv.write_text("audit_id,sample_id\n", encoding="utf-8")
    sampling_manifest.write_text(json.dumps({"actual_n": 0}), encoding="utf-8")
    preference_mapping.write_text("[]", encoding="utf-8")
    agreement.write_text(json.dumps({"status": "failed"}), encoding="utf-8")

    checklist = build_reproducibility_checklist(
        tmp_path,
        [
            "outputs/full_run/tables/table1_main_results.csv",
            "outputs/full_run/figures/figure_manifest.json",
            "outputs/full_run/human_audit/human_audit_blind.csv",
            "outputs/full_run/human_audit/sampling_manifest.json",
            "outputs/full_run/human_audit/preference_mapping.json",
            "outputs/full_run/human_audit/human_agreement_report.json",
        ],
    )

    assert checklist["status"] == "failed"
    checks = {check["name"]: check for check in checklist["checks"]}
    assert checks["outputs/full_run/tables/table1_main_results.csv:content"]["status"] == "failed"
    assert checks["outputs/full_run/figures/figure_manifest.json:content"]["actual_status"] == "failed"
    assert checks["outputs/full_run/human_audit/human_audit_blind.csv:content"]["rows"] == 0
    assert checks["outputs/full_run/human_audit/sampling_manifest.json:content"]["actual_n"] == 0
    assert checks["outputs/full_run/human_audit/preference_mapping.json:content"]["rows"] == 0
    assert checks["outputs/full_run/human_audit/human_agreement_report.json:content"]["actual_status"] == "failed"


def test_export_paper_artifacts_populates_manifest_from_shard_plan(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "experiments.yaml").write_text(
        "\n".join(
            [
                "experiments:",
                "  exp0_problem_diagnosis:",
                "    datasets: [mathdial, bridge]",
                "    split: test",
                "    methods: [single_tutor, ours_full]",
                "  exp1_routing:",
                "    datasets: [mathdial]",
                "    split: test",
                "    methods: [difficulty_routing, ours_routing]",
            ]
        ),
        encoding="utf-8",
    )
    shard_plan = {
        "experiments_config": "configs/experiments.yaml",
        "output_dir": "outputs/full_run",
        "split_dir": "data/splits",
        "live_llm": True,
        "jobs": [
            {
                "job_id": "exp0_problem_diagnosis_shard000-of-002",
                "experiment": "exp0_problem_diagnosis",
                "datasets": ["mathdial", "bridge"],
                "split": "test",
                "estimated_records": 10,
                "shard_index": 0,
                "num_shards": 2,
                "paths": {
                    "generations": "outputs/full_run/generations/exp0.jsonl",
                    "manifest": "outputs/full_run/logs/experiment_manifest_exp0.json",
                    "errors": "outputs/full_run/logs/errors_exp0.jsonl",
                },
            },
            {
                "job_id": "exp1_routing_shard000-of-001",
                "experiment": "exp1_routing",
                "datasets": ["mathdial"],
                "split": "test",
                "estimated_records": 5,
                "shard_index": 0,
                "num_shards": 1,
                "paths": {
                    "generations": "outputs/full_run/generations/exp1.jsonl",
                    "manifest": "outputs/full_run/logs/experiment_manifest_exp1.json",
                    "errors": "outputs/full_run/logs/errors_exp1.jsonl",
                },
            },
        ],
    }
    out = tmp_path / "paper_artifacts"

    files = export_paper_artifacts(
        tmp_path,
        out,
        experiment_manifests=[{"created_at_utc": "missing experiment name should be ignored"}],
        shard_plan=shard_plan,
        artifact_prefix="outputs/full_run",
    )

    manifest = json.loads(files["experiment_manifest"].read_text(encoding="utf-8"))
    summary = files["experiment_summary"].read_text(encoding="utf-8")
    exp0 = manifest["experiments"]["exp0_problem_diagnosis"]
    assert "None" not in manifest["experiments"]
    assert manifest["artifact_status"] == "passed"
    assert manifest["missing_experiments"] == []
    assert "Experiment manifest status: **passed**" in summary
    assert exp0["methods"] == ["single_tutor", "ours_full"]
    assert exp0["job_count"] == 1
    assert exp0["estimated_records"] == 10
    assert exp0["output_paths"]["generations"] == ["outputs/full_run/generations/exp0.jsonl"]
