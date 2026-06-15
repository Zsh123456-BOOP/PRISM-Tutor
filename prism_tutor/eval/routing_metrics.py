"""Routing precision/recall/F1 over selected agent sets."""

from __future__ import annotations

from .misconception_metrics import precision_recall_f1


def evaluate_routing(record: dict, gold: dict) -> dict:
    predicted = record.get("selected_agents") or record.get("routed_agents")
    expected = (
        gold.get("required_agents")
        or gold.get("gold_agents")
        or gold.get("agent_needs")
        or gold.get("pseudo_gold_agents")
    )
    result = precision_recall_f1(predicted, expected)
    return {f"routing_{key}": value for key, value in result.items()}
