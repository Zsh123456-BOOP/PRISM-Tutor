#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prism_tutor.export.artifact_exporter import export_paper_artifacts, final_artifact_status


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


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def main(argv: list[str] | None = None) -> int:
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
    parser.add_argument(
        "--allow-failed-checklist",
        action="store_true",
        help="Return success even when artifact manifest/checklist is failed. Use only for smoke or exploratory runs.",
    )
    args = parser.parse_args(argv)

    files = export_paper_artifacts(
        args.root,
        args.output_dir,
        _load_manifests(Path(args.logs)),
        artifact_prefix=args.artifact_prefix,
        shard_plan=_load_shard_plan(args.shard_plan),
    )
    manifest = _load_json(files["experiment_manifest"])
    checklist = _load_json(files["reproducibility_checklist_json"])
    status = final_artifact_status(manifest, checklist)
    payload = {"status": status, "files": {key: str(value) for key, value in files.items()}}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if status != "passed" and not args.allow_failed_checklist:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
