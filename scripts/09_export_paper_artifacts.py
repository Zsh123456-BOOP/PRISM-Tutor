#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prism_tutor.export.artifact_exporter import export_paper_artifacts


def _load_manifests(log_dir: Path) -> list[dict]:
    manifests = []
    if not log_dir.exists():
        return manifests
    for path in sorted(log_dir.glob("experiment_manifest_*.json")):
        try:
            manifests.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return manifests


def _load_shard_plan(path: str | None) -> dict | None:
    if not path:
        return None
    plan_path = Path(path)
    if not plan_path.exists():
        return None
    try:
        return json.loads(plan_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Export paper artifacts and reproducibility checklist.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output_dir", default="outputs/paper_artifacts")
    parser.add_argument("--logs", default="outputs/logs")
    parser.add_argument(
        "--artifact-prefix",
        default="outputs",
        help="Repository-relative output prefix to check in the reproducibility checklist and artifact index.",
    )
    parser.add_argument("--shard-plan", help="Optional full-run shard plan used to populate Exp0-Exp6 manifest metadata.")
    args = parser.parse_args()

    files = export_paper_artifacts(
        args.root,
        args.output_dir,
        _load_manifests(Path(args.logs)),
        artifact_prefix=args.artifact_prefix,
        shard_plan=_load_shard_plan(args.shard_plan),
    )
    print(json.dumps({key: str(value) for key, value in files.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
