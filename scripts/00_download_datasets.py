#!/usr/bin/env python
"""Download verified public raw datasets and report manual data gaps."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


RAW_PATTERNS = ("*.jsonl", "*.json", "*.csv")


def load_config(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text) or {}
    except Exception:
        return json.loads(text)


def discover_raw_files(path: str | Path) -> list[str]:
    root = Path(path)
    if not root.exists():
        return []
    files = []
    for pattern in RAW_PATTERNS:
        files.extend(str(item) for item in sorted(root.rglob(pattern)) if item.is_file())
    return sorted(set(files))


def _matches_dataset(name: str, selected: set[str] | None) -> bool:
    return selected is None or name in selected


def _download_hf_files(name: str, spec: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    raw_path = Path(spec["raw_path"])
    files = list(spec.get("files", []))
    result: dict[str, Any] = {
        "dataset": name,
        "source_type": spec["source_type"],
        "repo_id": spec.get("repo_id"),
        "raw_path": str(raw_path),
        "requested_files": files,
        "downloaded_files": [],
        "status": "dry_run" if dry_run else "pending",
        "errors": [],
        "warnings": [],
    }
    if dry_run:
        return result

    hf_hub_download = None
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        result["warnings"].append(f"huggingface_hub unavailable, using direct HTTPS fallback: {exc}")

    raw_path.mkdir(parents=True, exist_ok=True)
    for filename in files:
        try:
            target = raw_path / Path(filename).name
            if hf_hub_download is None:
                _download_hf_file_direct(
                    repo_id=str(spec["repo_id"]),
                    repo_type=str(spec.get("repo_type", "dataset")),
                    filename=filename,
                    target=target,
                )
            else:
                cached = hf_hub_download(
                    repo_id=str(spec["repo_id"]),
                    repo_type=str(spec.get("repo_type", "dataset")),
                    filename=filename,
                )
                shutil.copy2(cached, target)
            result["downloaded_files"].append(str(target))
        except Exception as exc:
            result["warnings"].append(f"{filename}: hf_hub_download failed, trying direct HTTPS fallback: {exc}")
            try:
                target = raw_path / Path(filename).name
                _download_hf_file_direct(
                    repo_id=str(spec["repo_id"]),
                    repo_type=str(spec.get("repo_type", "dataset")),
                    filename=filename,
                    target=target,
                )
                result["downloaded_files"].append(str(target))
            except Exception as fallback_exc:
                result["errors"].append(f"{filename}: {fallback_exc}")
    result["status"] = "completed" if not result["errors"] else "failed"
    return result


def _download_hf_file_direct(*, repo_id: str, repo_type: str, filename: str, target: Path) -> None:
    repo_prefix = "datasets/" if repo_type == "dataset" else ""
    quoted_repo = "/".join(urllib.parse.quote(part) for part in repo_id.split("/"))
    quoted_filename = "/".join(urllib.parse.quote(part) for part in filename.split("/"))
    url = f"https://huggingface.co/{repo_prefix}{quoted_repo}/resolve/main/{quoted_filename}"
    request = urllib.request.Request(url, headers={"User-Agent": "prism-tutor-dataset-downloader/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def _check_manual_dataset(name: str, spec: dict[str, Any], *, strict: bool) -> dict[str, Any]:
    raw_path = Path(spec["raw_path"])
    raw_files = discover_raw_files(raw_path)
    status = "available" if raw_files else "manual_required"
    result = {
        "dataset": name,
        "source_type": spec["source_type"],
        "raw_path": str(raw_path),
        "expected_files": spec.get("expected_files"),
        "raw_files": raw_files,
        "status": status,
        "notes": spec.get("notes"),
        "errors": [],
    }
    if strict and not raw_files:
        result["errors"].append(f"manual dataset missing raw files: {raw_path}")
    return result


def run_download(config_path: str, *, datasets: list[str] | None, dry_run: bool, strict: bool) -> dict[str, Any]:
    config = load_config(config_path)
    selected = set(datasets) if datasets else None
    results = []
    for name, spec in config.get("datasets", {}).items():
        if not _matches_dataset(name, selected):
            continue
        source_type = spec.get("source_type")
        if source_type == "huggingface_files":
            results.append(_download_hf_files(name, spec, dry_run=dry_run))
        elif source_type == "manual_required":
            results.append(_check_manual_dataset(name, spec, strict=strict))
        else:
            results.append(
                {
                    "dataset": name,
                    "source_type": source_type,
                    "status": "failed",
                    "errors": [f"unsupported source_type: {source_type}"],
                }
            )

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": config_path,
        "dry_run": dry_run,
        "strict": strict,
        "datasets": results,
        "all_ready": all(item["status"] in {"completed", "available", "dry_run"} and not item.get("errors") for item in results),
    }
    report_path = Path(config.get("report_path", "outputs/logs/dataset_download_report.json"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _split_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parts = [item.strip() for item in value.split(",") if item.strip()]
    return parts or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data_sources.yaml")
    parser.add_argument("--datasets", help="Comma-separated dataset names. Defaults to all configured datasets.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if a manual dataset is missing.")
    args = parser.parse_args(argv)

    report = run_download(
        args.config,
        datasets=_split_csv(args.datasets),
        dry_run=args.dry_run,
        strict=args.strict,
    )
    print(json.dumps({"all_ready": report["all_ready"], "datasets": report["datasets"]}, ensure_ascii=False, indent=2))
    if args.strict and not report["all_ready"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
