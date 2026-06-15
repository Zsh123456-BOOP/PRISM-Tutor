"""Build processed datasets, deterministic splits, and dataset reports."""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .io import discover_raw_files, write_jsonl
from .loaders import load_bridge, load_mathdial, load_misconception
from .report import dataset_report, write_report
from .splits import build_splits


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "report_path": "outputs/logs/dataset_report.json",
    "datasets": {
        "mathdial": {
            "enabled": True,
            "raw_path": "data/raw/mathdial",
            "processed_path": "data/processed/mathdial.jsonl",
            "split_prefix": "data/splits/mathdial",
        },
        "bridge": {
            "enabled": True,
            "raw_path": "data/raw/bridge",
            "processed_path": "data/processed/bridge.jsonl",
            "split_prefix": "data/splits/bridge",
        },
        "misconception": {
            "enabled": True,
            "raw_path": "data/raw/misconception",
            "processed_path": "data/processed/misconception.jsonl",
            "split_prefix": "data/splits/misconception",
        },
    },
}

LOADERS = {
    "mathdial": load_mathdial,
    "bridge": load_bridge,
    "misconception": load_misconception,
}

EXPECTED_SPLITS = {
    "mathdial": ["train", "dev", "test"],
    "bridge": ["dev", "test"],
    "misconception": ["test"],
}


def build_datasets(config_path: str | Path = "configs/datasets.yaml", strict: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    seed = int(config.get("seed", 42))
    report: dict[str, Any] = {
        "seed": seed,
        "config_path": str(config_path),
        "datasets": {},
        "warnings": [],
    }

    for dataset, dataset_config in config.get("datasets", {}).items():
        if not dataset_config.get("enabled", True):
            continue
        if dataset not in LOADERS:
            report["warnings"].append(f"Unknown dataset skipped: {dataset}")
            continue

        raw_path = dataset_config["raw_path"]
        raw_files = discover_raw_files(raw_path)
        missing_raw = not raw_files
        if missing_raw:
            message = f"No raw JSON/JSONL/CSV files found for {dataset}: {raw_path}"
            report["warnings"].append(message)
            if strict:
                raise FileNotFoundError(message)

        raw_records = LOADERS[dataset](raw_path)
        records, duplicates = deduplicate_records(raw_records)
        processed_path = dataset_config["processed_path"]
        write_jsonl(processed_path, records)

        splits, split_strategy = build_splits(dataset, records, seed)
        split_files = write_split_files(dataset_config["split_prefix"], dataset, splits)

        report["datasets"][dataset] = dataset_report(
            dataset=dataset,
            records=records,
            raw_count=len(raw_records),
            duplicates=duplicates,
            split_files=split_files,
            split_strategy=split_strategy,
            missing_raw=missing_raw,
        )

    write_report(config["report_path"], report)
    return report


def deduplicate_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for record in records:
        key = f"{record['dataset']}::{record['raw_record_id']}"
        if key in seen:
            duplicates.append(
                {
                    "dataset": record["dataset"],
                    "raw_record_id": record["raw_record_id"],
                    "sample_id": record["sample_id"],
                }
            )
            continue
        seen.add(key)
        kept.append(record)
    return kept, duplicates


def write_split_files(
    split_prefix: str | Path,
    dataset: str,
    splits: Mapping[str, list[dict[str, Any]]],
) -> dict[str, str]:
    split_files: dict[str, str] = {}
    for split in EXPECTED_SPLITS[dataset]:
        records = []
        for index, record in enumerate(splits.get(split, [])):
            enriched = dict(record)
            enriched["split"] = split
            enriched["split_index"] = index
            records.append(enriched)
        path = f"{split_prefix}_{split}.jsonl"
        write_jsonl(path, records)
        split_files[split] = path
    return split_files


def load_config(config_path: str | Path) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    path = Path(config_path)
    if not path.exists():
        return config
    with path.open("r", encoding="utf-8") as handle:
        text = handle.read()
    loaded = _parse_config(text)
    if loaded:
        _deep_update(config, loaded)
    return config


def _parse_config(text: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        return loaded or {}
    except Exception:
        loaded = json.loads(text)
        return loaded or {}


def _deep_update(base: dict[str, Any], update: Mapping[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/datasets.yaml")
    parser.add_argument("--strict", action="store_true", help="Fail if a configured raw dataset is missing.")
    args = parser.parse_args()
    report = build_datasets(args.config, strict=args.strict)
    print(json.dumps({"report_path": load_config(args.config)["report_path"], "datasets": list(report["datasets"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
