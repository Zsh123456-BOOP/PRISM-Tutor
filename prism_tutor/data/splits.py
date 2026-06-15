"""Deterministic split strategies."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any


def build_splits(dataset: str, records: list[dict[str, Any]], seed: int) -> tuple[dict[str, list[dict[str, Any]]], str]:
    if dataset == "mathdial":
        return _mathdial_splits(records, seed)
    if dataset == "bridge":
        return _bridge_splits(records, seed), "stratified_20_dev_80_test"
    if dataset == "misconception":
        return _misconception_splits(records), "all_test_with_bootstrap_index"
    raise ValueError(f"Unknown dataset: {dataset}")


def _mathdial_splits(records: list[dict[str, Any]], seed: int) -> tuple[dict[str, list[dict[str, Any]]], str]:
    official = [record.get("metadata", {}).get("official_split") for record in records]
    if records and all(split in {"train", "dev", "test"} for split in official):
        splits: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
        for record, split in zip(records, official, strict=True):
            splits[str(split)].append(record)
        return splits, "official"

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = str(record.get("conversation_id") or record["raw_record_id"])
        groups[key].append(record)

    keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)
    train_keys, dev_keys, test_keys = _ratio_partition(keys, train_ratio=0.8, dev_ratio=0.1)
    return (
        {
            "train": _collect_groups(groups, train_keys),
            "dev": _collect_groups(groups, dev_keys),
            "test": _collect_groups(groups, test_keys),
        },
        "conversation_80_10_10_seed_42",
    )


def _bridge_splits(records: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    has_label = False
    for record in records:
        label = record.get("student_error") or record.get("remediation_strategy")
        if label not in (None, ""):
            has_label = True
        groups[str(label) if label not in (None, "") else "__missing_label__"].append(record)

    if not has_label:
        ordered = sorted(records, key=lambda item: item["sample_id"])
        rng = random.Random(seed)
        rng.shuffle(ordered)
        dev_count = _count_for_ratio(len(ordered), 0.2)
        return {"dev": ordered[:dev_count], "test": ordered[dev_count:]}

    rng = random.Random(seed)
    dev: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for label in sorted(groups):
        group = sorted(groups[label], key=lambda item: item["sample_id"])
        rng.shuffle(group)
        dev_count = _count_for_ratio(len(group), 0.2)
        dev.extend(group[:dev_count])
        test.extend(group[dev_count:])
    return {
        "dev": sorted(dev, key=lambda item: item["sample_id"]),
        "test": sorted(test, key=lambda item: item["sample_id"]),
    }


def _misconception_splits(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    test = []
    for index, record in enumerate(sorted(records, key=lambda item: item["sample_id"])):
        enriched = dict(record)
        enriched["sample_index"] = index
        enriched["bootstrap_index"] = index
        test.append(enriched)
    return {"test": test}


def _ratio_partition(keys: list[str], train_ratio: float, dev_ratio: float) -> tuple[set[str], set[str], set[str]]:
    total = len(keys)
    train_count = int(total * train_ratio)
    dev_count = int(total * dev_ratio)
    if total >= 3:
        train_count = max(1, train_count)
        dev_count = max(1, dev_count)
    if train_count + dev_count > total:
        dev_count = max(0, total - train_count)
    train = set(keys[:train_count])
    dev = set(keys[train_count : train_count + dev_count])
    test = set(keys[train_count + dev_count :])
    return train, dev, test


def _count_for_ratio(total: int, ratio: float) -> int:
    if total <= 0:
        return 0
    if total == 1:
        return 0
    return max(1, int(round(total * ratio)))


def _collect_groups(groups: dict[str, list[dict[str, Any]]], keys: set[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for key in sorted(keys):
        records.extend(sorted(groups[key], key=lambda item: item["sample_id"]))
    return records
