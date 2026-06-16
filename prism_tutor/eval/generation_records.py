from __future__ import annotations

from typing import Any


def generation_record_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("dataset", "")),
        str(row.get("sample_id", "")),
        str(row.get("split", "")),
        str(row.get("method", "")),
    )


def deduplicate_generation_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: dict[tuple[str, str, str, str], tuple[int, dict[str, Any]]] = {}
    duplicate_keys: set[tuple[str, str, str, str]] = set()
    replaced_failed_with_success = 0
    repeated_success = 0
    for index, row in enumerate(rows):
        key = generation_record_key(row)
        previous = selected.get(key)
        if previous is None:
            selected[key] = (index, row)
            continue
        duplicate_keys.add(key)
        previous_row = previous[1]
        previous_success = previous_row.get("status") == "success"
        current_success = row.get("status") == "success"
        if current_success and not previous_success:
            replaced_failed_with_success += 1
            selected[key] = (index, row)
        elif current_success == previous_success:
            if current_success:
                repeated_success += 1
            selected[key] = (index, row)

    deduped = [row for _, row in sorted(selected.values(), key=lambda item: item[0])]
    report = {
        "raw_generation_count": len(rows),
        "generation_count": len(deduped),
        "duplicate_generation_count": len(rows) - len(deduped),
        "duplicate_key_count": len(duplicate_keys),
        "replaced_failed_with_success_count": replaced_failed_with_success,
        "repeated_success_duplicate_count": repeated_success,
    }
    return deduped, report
