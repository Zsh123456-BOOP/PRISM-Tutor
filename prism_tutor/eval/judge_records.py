from __future__ import annotations

from typing import Any


def judge_record_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("dataset", "")),
        str(row.get("sample_id", "")),
        str(row.get("method", "")),
    )


def judge_row_is_valid(row: dict[str, Any] | None) -> bool:
    if not isinstance(row, dict):
        return False
    return isinstance(row.get("parsed_score"), dict) and not row.get("error")


def deduplicate_judge_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: dict[tuple[str, str, str], tuple[int, dict[str, Any]]] = {}
    duplicate_keys: set[tuple[str, str, str]] = set()
    replaced_invalid_with_valid = 0
    repeated_valid = 0
    repeated_invalid = 0
    ignored_invalid_after_valid = 0

    for index, row in enumerate(rows):
        key = judge_record_key(row)
        previous = selected.get(key)
        if previous is None:
            selected[key] = (index, row)
            continue

        duplicate_keys.add(key)
        previous_row = previous[1]
        previous_valid = judge_row_is_valid(previous_row)
        current_valid = judge_row_is_valid(row)
        if current_valid and not previous_valid:
            replaced_invalid_with_valid += 1
            selected[key] = (index, row)
        elif current_valid == previous_valid:
            if current_valid:
                repeated_valid += 1
            else:
                repeated_invalid += 1
            selected[key] = (index, row)
        elif previous_valid and not current_valid:
            ignored_invalid_after_valid += 1

    deduped = [row for _, row in sorted(selected.values(), key=lambda item: item[0])]
    report = {
        "raw_judge_count": len(rows),
        "judge_count": len(deduped),
        "duplicate_judge_count": len(rows) - len(deduped),
        "duplicate_judge_key_count": len(duplicate_keys),
        "replaced_invalid_with_valid_judge_count": replaced_invalid_with_valid,
        "repeated_valid_judge_duplicate_count": repeated_valid,
        "repeated_invalid_judge_duplicate_count": repeated_invalid,
        "ignored_invalid_after_valid_judge_count": ignored_invalid_after_valid,
    }
    return deduped, report
