"""State-related automatic metrics from runtime logs."""

from __future__ import annotations

from typing import Any


def _events(record: dict[str, Any]) -> list[dict[str, Any]]:
    state = record.get("state") if isinstance(record.get("state"), dict) else {}
    for key in ("events", "state_events", "commits"):
        value = state.get(key) if key in state else record.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def evaluate_state_metrics(record: dict[str, Any]) -> dict[str, Any]:
    events = _events(record)
    if not events:
        return {
            "state_event_count": 0,
            "state_conflict_rate": None,
            "incorrect_commit_rate": None,
            "tentative_update_rate": None,
            "state_metric_coverage": 0.0,
        }
    conflicts = sum(bool(event.get("conflict") or event.get("type") == "conflict") for event in events)
    commits = [event for event in events if event.get("type") in {"commit", "state_commit"} or "correct" in event]
    incorrect = sum(event.get("correct") is False or event.get("incorrect") is True for event in commits)
    tentative = sum(event.get("tentative") is True or event.get("type") == "tentative_update" for event in events)
    return {
        "state_event_count": len(events),
        "state_conflict_rate": conflicts / len(events),
        "incorrect_commit_rate": incorrect / len(commits) if commits else None,
        "tentative_update_rate": tentative / len(events),
        "state_metric_coverage": 1.0,
    }
