from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prism_tutor.utils.config import write_yaml_snapshot
from prism_tutor.utils.reproducibility import collect_reproducibility_metadata


def write_experiment_manifest(
    *,
    path: str | Path,
    config: dict[str, Any],
    run: dict[str, Any],
    status: str,
) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    config_snapshot = target.with_suffix(".config.yaml")
    write_yaml_snapshot(config, config_snapshot)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "run": run,
        "reproducibility": collect_reproducibility_metadata(str(config_snapshot)),
    }
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
