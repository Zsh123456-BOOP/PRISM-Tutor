from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {self.path}:{line_number}") from exc
        return records

    def completed_keys(self) -> set[tuple[str, str, str, str]]:
        keys: set[tuple[str, str, str, str]] = set()
        for record in self.read_all():
            if record.get("status") == "success":
                keys.add(
                    (
                        str(record.get("sample_id")),
                        str(record.get("dataset")),
                        str(record.get("split")),
                        str(record.get("method")),
                    )
                )
        return keys


def append_many(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    logger = JsonlLogger(path)
    for record in records:
        logger.append(record)
