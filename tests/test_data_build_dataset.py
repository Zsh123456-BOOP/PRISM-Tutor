import csv
import json
from pathlib import Path

from prism_tutor.data.build_dataset import build_datasets
from prism_tutor.data.io import read_jsonl


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _config(tmp_path: Path) -> Path:
    config = {
        "seed": 42,
        "report_path": str(tmp_path / "reports" / "dataset_report.json"),
        "datasets": {
            "mathdial": {
                "raw_path": str(tmp_path / "raw" / "mathdial"),
                "processed_path": str(tmp_path / "processed" / "mathdial.jsonl"),
                "split_prefix": str(tmp_path / "splits" / "mathdial"),
            },
            "bridge": {
                "raw_path": str(tmp_path / "raw" / "bridge"),
                "processed_path": str(tmp_path / "processed" / "bridge.jsonl"),
                "split_prefix": str(tmp_path / "splits" / "bridge"),
            },
            "misconception": {
                "raw_path": str(tmp_path / "raw" / "misconception"),
                "processed_path": str(tmp_path / "processed" / "misconception.jsonl"),
                "split_prefix": str(tmp_path / "splits" / "misconception"),
            },
        },
    }
    path = tmp_path / "datasets.yaml"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_builds_processed_splits_and_report_from_local_files(tmp_path):
    _write_jsonl(
        tmp_path / "raw" / "mathdial" / "mathdial.jsonl",
        [
            {
                "id": "m1",
                "split": "train",
                "problem": "2+2?",
                "student_response": "5",
                "tutor_response": "Check by counting.",
                "scaffolding": ["hint"],
                "leakage": False,
            },
            {
                "id": "m2",
                "split": "test",
                "problem": "3+1?",
                "student_response": "3",
                "scaffolding": [],
                "leakage": None,
            },
        ],
    )

    bridge_dir = tmp_path / "raw" / "bridge"
    bridge_dir.mkdir(parents=True)
    with (bridge_dir / "bridge.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "problem",
                "student_answer",
                "student_error",
                "remediation_strategy",
                "teacher_intention",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "id": "b1",
                "problem": "10-6?",
                "student_answer": "16",
                "student_error": "operation_confusion",
                "remediation_strategy": "contrast",
                "teacher_intention": "diagnose",
            }
        )
        writer.writerow({"id": "b2", "problem": "8/2?", "student_answer": "6"})

    misconception_dir = tmp_path / "raw" / "misconception"
    misconception_dir.mkdir(parents=True)
    (misconception_dir / "misconception.json").write_text(
        json.dumps(
            [
                {"id": "x1", "question": "1/2 + 1/3?", "answer": "2/5", "misconception": "add_denominators"},
                {"id": "x2", "question": "5*0?", "answer": "5"},
            ]
        ),
        encoding="utf-8",
    )

    report = build_datasets(_config(tmp_path), strict=True)

    mathdial = read_jsonl(tmp_path / "processed" / "mathdial.jsonl")
    assert len(mathdial) == 2
    assert "tutor_response" in mathdial[1]["missing_fields"]

    bridge = read_jsonl(tmp_path / "processed" / "bridge.jsonl")
    assert len(bridge) == 2
    assert "student_error" in bridge[1]["missing_fields"]

    misconception_test = read_jsonl(tmp_path / "splits" / "misconception_test.jsonl")
    assert [row["bootstrap_index"] for row in misconception_test] == [0, 1]

    assert report["datasets"]["mathdial"]["processed_count"] == 2
    assert "split_hash" in report["datasets"]["bridge"]
    assert (tmp_path / "reports" / "dataset_report.json").exists()


def test_mathdial_conversation_split_keeps_conversations_together(tmp_path):
    rows = []
    for conversation in range(12):
        for turn in range(2):
            rows.append(
                {
                    "id": f"c{conversation}-t{turn}",
                    "conversation_id": f"c{conversation}",
                    "problem": f"{conversation}+{turn}",
                    "student_response": "wrong",
                    "tutor_response": "hint",
                    "scaffolding": "hint",
                    "leakage": False,
                }
            )
    _write_jsonl(tmp_path / "raw" / "mathdial" / "mathdial.jsonl", rows)
    _write_jsonl(tmp_path / "raw" / "bridge" / "bridge.jsonl", [])
    _write_jsonl(tmp_path / "raw" / "misconception" / "misconception.jsonl", [])

    build_datasets(_config(tmp_path), strict=True)

    conversation_to_split = {}
    for split in ("train", "dev", "test"):
        for row in read_jsonl(tmp_path / "splits" / f"mathdial_{split}.jsonl"):
            previous = conversation_to_split.setdefault(row["conversation_id"], split)
            assert previous == split
    assert len(conversation_to_split) == 12


def test_mathdial_official_string_conversation_is_expanded(tmp_path):
    _write_jsonl(
        tmp_path / "raw" / "mathdial" / "train.jsonl",
        [
            {
                "qid": 5000012,
                "question": "How much water is left?",
                "ground_truth": "54",
                "student_incorrect_solution": "72",
                "conversation": (
                    "Teacher: (probing)What is half of 36?|EOM|"
                    "Steven: 18.|EOM|"
                    "Teacher: (generic)Exactly correct!"
                ),
            }
        ],
    )
    _write_jsonl(tmp_path / "raw" / "bridge" / "bridge.jsonl", [])
    _write_jsonl(tmp_path / "raw" / "misconception" / "misconception.jsonl", [])

    report = build_datasets(_config(tmp_path), strict=True)
    mathdial = read_jsonl(tmp_path / "processed" / "mathdial.jsonl")

    assert len(mathdial) == 2
    assert mathdial[0]["conversation_id"] == 5000012
    assert mathdial[0]["student_utterance"] == "72"
    assert mathdial[0]["tutor_response"] == "What is half of 36?"
    assert mathdial[0]["scaffolding"] == ["probing"]
    assert mathdial[1]["student_utterance"] == "18."
    assert mathdial[0]["metadata"]["official_split"] == "train"
    assert mathdial[0]["metadata"]["ground_truth"] == "54"
    assert report["datasets"]["mathdial"]["field_completeness"]["student_utterance"] == 1.0
