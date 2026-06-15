"""Raw and processed dataset I/O."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any


RAW_EXTENSIONS = {".json", ".jsonl", ".csv"}


def discover_raw_files(raw_path: str | Path) -> list[Path]:
    path = Path(raw_path)
    if path.is_file():
        return [path] if path.suffix.lower() in RAW_EXTENSIONS else []
    if not path.exists():
        return []
    return sorted(file for file in path.rglob("*") if file.suffix.lower() in RAW_EXTENSIONS)


def read_raw_records(raw_path: str | Path) -> Iterator[tuple[dict[str, Any], Path, int]]:
    for file_path in discover_raw_files(raw_path):
        suffix = file_path.suffix.lower()
        if suffix == ".jsonl":
            yield from _read_jsonl(file_path)
        elif suffix == ".json":
            yield from _read_json(file_path)
        elif suffix == ".csv":
            yield from _read_csv(file_path)


def _read_jsonl(path: Path) -> Iterator[tuple[dict[str, Any], Path, int]]:
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if isinstance(record, Mapping):
                yield dict(record), path, index
            else:
                yield {"value": record}, path, index


def _read_json(path: Path) -> Iterator[tuple[dict[str, Any], Path, int]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = _extract_records(payload)
    for index, record in enumerate(records):
        yield record, path, index


def _extract_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) if isinstance(item, Mapping) else {"value": item} for item in payload]
    if isinstance(payload, Mapping):
        for key in ("data", "records", "examples", "items", "rows", "dialogues", "conversations"):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(item) if isinstance(item, Mapping) else {"value": item} for item in value]
        return [dict(payload)]
    return [{"value": payload}]


def _read_csv(path: Path) -> Iterator[tuple[dict[str, Any], Path, int]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            yield dict(row), path, index


def write_jsonl(path: str | Path, records: list[Mapping[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records
