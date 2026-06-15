"""Dataset report generation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .schema import SCHEMA_FIELDS, TASK_FIELDS


LABEL_FIELDS = {
    "mathdial": ["leakage"],
    "bridge": ["student_error", "remediation_strategy", "teacher_intention"],
    "misconception": ["misconception_label"],
}


def dataset_report(
    *,
    dataset: str,
    records: list[dict[str, Any]],
    raw_count: int,
    duplicates: list[Mapping[str, Any]],
    split_files: Mapping[str, str],
    split_strategy: str,
    missing_raw: bool,
) -> dict[str, Any]:
    split_hashes = {split: file_hash(path) for split, path in split_files.items()}
    return {
        "dataset": dataset,
        "raw_count": raw_count,
        "processed_count": len(records),
        "duplicate_count": len(duplicates),
        "duplicates": list(duplicates),
        "missing_raw": missing_raw,
        "field_completeness": _field_completeness(dataset, records),
        "empty_field_ratio": _empty_field_ratio(records),
        "label_distribution": _label_distribution(dataset, records),
        "missing_field_distribution": _missing_field_distribution(records),
        "splits": _split_summary(split_files),
        "split_strategy": split_strategy,
        "split_hash": _combined_hash(split_hashes),
        "split_file_hashes": split_hashes,
    }


def write_report(path: str | Path, report: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def file_hash(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.exists():
        return ""
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _field_completeness(dataset: str, records: list[dict[str, Any]]) -> dict[str, float]:
    fields = TASK_FIELDS[dataset]
    if not records:
        return {field: 0.0 for field in fields}
    return {
        field: round(sum(1 for record in records if _present(record.get(field))) / len(records), 6)
        for field in fields
    }


def _empty_field_ratio(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {field: 0.0 for field in SCHEMA_FIELDS}
    return {
        field: round(sum(1 for record in records if not _present(record.get(field))) / len(records), 6)
        for field in SCHEMA_FIELDS
    }


def _label_distribution(dataset: str, records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    distributions: dict[str, dict[str, int]] = {}
    for field in LABEL_FIELDS[dataset]:
        counter: Counter[str] = Counter()
        for record in records:
            value = record.get(field)
            key = "__missing__" if not _present(value) else str(value)
            counter[key] += 1
        distributions[field] = dict(sorted(counter.items()))
    return distributions


def _missing_field_distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        counter.update(record.get("missing_fields") or [])
    return dict(sorted(counter.items()))


def _split_summary(split_files: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for split, path in split_files.items():
        count = 0
        file_path = Path(path)
        if file_path.exists():
            with file_path.open("r", encoding="utf-8") as handle:
                count = sum(1 for line in handle if line.strip())
        summary[split] = {"path": str(path), "count": count}
    return summary


def _combined_hash(split_hashes: Mapping[str, str]) -> str:
    payload = json.dumps(split_hashes, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _present(value: Any) -> bool:
    return value is not None and value != "" and value != []
