from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def get_by_path(config: dict[str, Any], dotted_path: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def set_by_path(config: dict[str, Any], dotted_path: str, value: Any) -> None:
    current = config
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
        if not isinstance(current, dict):
            raise ValueError(f"Cannot set nested key through non-mapping: {dotted_path}")
    current[parts[-1]] = value


def apply_env_overrides(config: dict[str, Any], prefix: str = "PRISM_") -> dict[str, Any]:
    merged = deepcopy(config)
    for env_key, raw_value in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        key_path = env_key[len(prefix) :].lower().replace("__", ".")
        value: Any
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError:
            value = raw_value
        set_by_path(merged, key_path, value)
    return merged


def load_config(path: str | Path = "configs/default.yaml", *, env_overrides: bool = True) -> dict[str, Any]:
    config = load_yaml(path)
    if env_overrides:
        config = apply_env_overrides(config)
    return config


def write_yaml_snapshot(config: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True, allow_unicode=False)
