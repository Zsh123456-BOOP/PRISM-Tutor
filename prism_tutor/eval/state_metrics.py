"""State-related automatic metrics from runtime logs."""

from __future__ import annotations

from typing import Any


def _events(record: dict[str, Any]) -> list[dict[str, Any]]:
    state = record.get("state") if isinstance(record.get("state"), dict) else {}
    for key in ("events", "state_events", "commits"):
        value = state.get(key) if key in state else record.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    agent_outputs = state.get("agent_outputs") if isinstance(state.get("agent_outputs"), dict) else {}
    commit_outputs = agent_outputs.get("state_commit") or record.get("state_commit") or []
    if isinstance(commit_outputs, dict):
        commit_outputs = [commit_outputs]
    if isinstance(commit_outputs, list):
        return _events_from_commit_outputs([item for item in commit_outputs if isinstance(item, dict)])
    return []


def _events_from_commit_outputs(commit_outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for decision in commit_outputs:
        status = str(decision.get("status") or "")
        conflict = status == "tentative"
        for update in decision.get("committed_updates") or []:
            if isinstance(update, dict):
                events.append({"type": "state_commit", "status": status, "conflict": False, "tentative": False, **update})
        for update in decision.get("tentative_updates") or []:
            if isinstance(update, dict):
                events.append({"type": "tentative_update", "status": status, "conflict": conflict, "tentative": True, **update})
        for update in decision.get("rejected_updates") or []:
            if isinstance(update, dict):
                events.append({"type": "rejected_update", "status": status, "conflict": conflict, "tentative": False, **update})
        if status and not events:
            events.append({"type": "state_commit", "status": status, "conflict": conflict, "tentative": False})
    return events


def _gold_state(gold: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(gold, dict):
        return {}
    metadata = gold.get("metadata") if isinstance(gold.get("metadata"), dict) else {}
    for key in ("gold_state", "student_state", "state"):
        value = gold.get(key)
        if isinstance(value, dict):
            return value
        value = metadata.get(key)
        if isinstance(value, dict):
            return value
    misconception = (
        gold.get("misconception_label")
        or gold.get("gold_misconception")
        or metadata.get("misconception_label")
        or metadata.get("gold_misconception")
    )
    return {"active_misconceptions": misconception if isinstance(misconception, list) else [misconception]} if misconception else {}


def _final_student_state(record: dict[str, Any]) -> dict[str, Any]:
    state = record.get("state") if isinstance(record.get("state"), dict) else {}
    student_state = state.get("student_state")
    return student_state if isinstance(student_state, dict) else {}


def _as_set(value: Any) -> set[str]:
    if value in (None, "", [], {}):
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip().lower() for item in value if str(item).strip()}
    return {str(value).strip().lower()}


def _external_state_metrics(record: dict[str, Any], gold: dict[str, Any] | None) -> dict[str, Any]:
    target = _gold_state(gold)
    if not target:
        return {
            "external_state_accuracy": None,
            "external_state_coverage": 0.0,
            "incorrect_misconception_commit_rate": None,
            "final_state_contradiction": None,
            "noisy_state_update_rejection_accuracy": None,
        }
    final_state = _final_student_state(record)
    scored: list[bool] = []
    contradiction = False
    for field, gold_value in target.items():
        gold_set = _as_set(gold_value)
        pred_set = _as_set(final_state.get(field))
        if not gold_set:
            continue
        scored.append(bool(pred_set & gold_set))
        if pred_set and pred_set.isdisjoint(gold_set):
            contradiction = True
    active_gold = _as_set(target.get("active_misconceptions"))
    active_pred = _as_set(final_state.get("active_misconceptions"))
    incorrect_rate = None
    if active_pred and active_gold:
        incorrect_rate = len(active_pred - active_gold) / len(active_pred)
    return {
        "external_state_accuracy": sum(scored) / len(scored) if scored else None,
        "external_state_coverage": 1.0 if scored else 0.0,
        "incorrect_misconception_commit_rate": incorrect_rate,
        "final_state_contradiction": contradiction if scored else None,
        "noisy_state_update_rejection_accuracy": None,
    }


def evaluate_state_metrics(record: dict[str, Any], gold: dict[str, Any] | None = None) -> dict[str, Any]:
    events = _events(record)
    external = _external_state_metrics(record, gold)
    if not events:
        return {
            "state_event_count": 0,
            "state_conflict_rate": None,
            "incorrect_commit_rate": None,
            "tentative_update_rate": None,
            "unsafe_commit_rate": None,
            "tentative_when_conflict_rate": None,
            "commit_with_evidence_rate": None,
            "state_metric_coverage": 0.0,
            **external,
        }
    conflicts = sum(bool(event.get("conflict") or event.get("type") == "conflict") for event in events)
    commits = [event for event in events if event.get("type") in {"commit", "state_commit"} or "correct" in event]
    incorrect = sum(event.get("correct") is False or event.get("incorrect") is True for event in commits)
    tentative = sum(event.get("tentative") is True or event.get("type") == "tentative_update" for event in events)
    unsafe_commits = sum(
        bool(event.get("conflict")) and event.get("type") in {"commit", "state_commit"}
        for event in events
    )
    conflict_events = [event for event in events if bool(event.get("conflict") or event.get("type") == "conflict")]
    tentative_conflicts = sum(
        event.get("tentative") is True or event.get("type") == "tentative_update"
        for event in conflict_events
    )
    evidence_commits = sum(bool(event.get("evidence")) for event in commits)
    return {
        "state_event_count": len(events),
        "state_conflict_rate": conflicts / len(events),
        "incorrect_commit_rate": incorrect / len(commits) if commits else None,
        "tentative_update_rate": tentative / len(events),
        "unsafe_commit_rate": unsafe_commits / len(commits) if commits else None,
        "tentative_when_conflict_rate": tentative_conflicts / len(conflict_events) if conflict_events else None,
        "commit_with_evidence_rate": evidence_commits / len(commits) if commits else None,
        "state_metric_coverage": 1.0,
        **external,
    }
