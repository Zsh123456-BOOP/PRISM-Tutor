"""JSON checkpoint writer for runtime state snapshots."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .graph_state import TutorGraphState


class CheckpointWriter:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def write(self, state: TutorGraphState, *, round_index: int | None = None) -> Path:
        sample_id = str(state.sample.get("sample_id") or state.sample.get("id") or "unknown")
        round_value = state.rounds if round_index is None else round_index
        path = self.root / state.method / f"{sample_id}.round{round_value}.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(state.snapshot(), ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            print(f"checkpoint_failed: {exc}", file=sys.stderr)
            raise
        return path
