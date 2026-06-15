#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prism_tutor.eval.human_audit_sampler import sample_human_audit
from prism_tutor.eval.io import read_jsonl, write_csv, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Create blind human-audit sample after experiments.")
    parser.add_argument("--records", default="outputs/metrics/record_auto_metrics.jsonl")
    parser.add_argument("--generations", default="outputs/generations")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--output_dir", default="outputs/human_audit")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    gen_rows = []
    gen_path = Path(args.generations)
    for file in sorted(gen_path.glob("*.jsonl")) if gen_path.is_dir() else [gen_path]:
        if file.exists():
            gen_rows.extend(read_jsonl(file))
    if not gen_rows and not args.allow_incomplete:
        raise SystemExit("No generation rows found; run experiments first or pass --allow-incomplete for smoke.")
    result = sample_human_audit(gen_rows, target_n=args.n)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "human_audit_blind.csv", result["blind_rows"], list(result["blind_rows"][0]) if result["blind_rows"] else ["audit_id"])
    write_json(out / "sampling_manifest.json", result["manifest"])
    print(json.dumps({"actual_n": result["manifest"]["actual_n"], "output_dir": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
