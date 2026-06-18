"""Unified dataset schema helpers.

The loaders intentionally keep partially missing samples. Missing required
fields are represented in ``missing_fields`` instead of filtering records.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any


SCHEMA_FIELDS = [
    "dataset",
    "sample_id",
    "raw_record_id",
    "conversation_id",
    "problem_text",
    "student_utterance",
    "tutor_response",
    "scaffolding",
    "leakage",
    "student_error",
    "remediation_strategy",
    "teacher_intention",
    "misconception_label",
    "candidate_misconceptions",
    "sample_index",
    "source_file",
    "metadata",
    "missing_fields",
]

TASK_FIELDS = {
    "mathdial": [
        "conversation_id",
        "problem_text",
        "student_utterance",
        "tutor_response",
        "scaffolding",
        "leakage",
    ],
    "bridge": [
        "problem_text",
        "student_utterance",
        "student_error",
        "remediation_strategy",
        "teacher_intention",
    ],
    "misconception": [
        "problem_text",
        "student_utterance",
        "misconception_label",
    ],
}

LIST_FIELDS = {"scaffolding", "candidate_misconceptions"}


def stable_hash(value: Any, length: int = 16) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def stable_sample_id(dataset: str, raw_record_id: str, record: Mapping[str, Any]) -> str:
    key = raw_record_id or stable_hash(record, length=24)
    return f"{dataset}:{stable_hash(key, length=20)}"


def get_path(record: Mapping[str, Any], path: str) -> Any:
    current: Any = record
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def first_present(record: Mapping[str, Any], candidates: Iterable[str]) -> Any:
    for candidate in candidates:
        value = get_path(record, candidate) if "." in candidate else record.get(candidate)
        if value not in (None, ""):
            return value
    return None


def as_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def normalize_value(field: str, value: Any) -> Any:
    if field in LIST_FIELDS:
        return as_list(value)
    return value if value != "" else None


def missing_fields_for(dataset: str, record: Mapping[str, Any]) -> list[str]:
    missing: list[str] = []
    for field in TASK_FIELDS[dataset]:
        value = record.get(field)
        if field in LIST_FIELDS:
            if not value:
                missing.append(field)
        elif value is None or value == "":
            missing.append(field)
    return missing


def make_record(
    *,
    dataset: str,
    raw_record_id: str,
    values: Mapping[str, Any],
    source_file: str | None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized: dict[str, Any] = {field: None for field in SCHEMA_FIELDS}
    normalized["dataset"] = dataset
    normalized["raw_record_id"] = str(raw_record_id)
    normalized["source_file"] = source_file
    normalized["metadata"] = dict(metadata or {})

    for field, value in values.items():
        if field in SCHEMA_FIELDS:
            normalized[field] = normalize_value(field, value)

    normalized["sample_id"] = stable_sample_id(dataset, normalized["raw_record_id"], values)
    normalized["missing_fields"] = missing_fields_for(dataset, normalized)
    return normalized
