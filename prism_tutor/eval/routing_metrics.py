"""Routing precision/recall/F1 over selected agent sets."""

from __future__ import annotations

from .misconception_metrics import precision_recall_f1


def _pseudo_gold_agents(gold: dict) -> list[str]:
    agents: set[str] = {"final_tutor"}
    metadata = gold.get("metadata") if isinstance(gold.get("metadata"), dict) else {}
    if gold.get("student_error") or gold.get("misconception_label") or gold.get("misconception_labels"):
        agents.add("misconception")
    if gold.get("remediation_strategy") or gold.get("teacher_intention") or gold.get("scaffolding"):
        agents.add("pedagogy")
    if metadata.get("ground_truth") or metadata.get("correct_answer"):
        agents.add("solver")
    if len(agents) > 1:
        agents.add("verifier")
    return sorted(agents)


def evaluate_routing(record: dict, gold: dict) -> dict:
    predicted = record.get("selected_agents") or record.get("routed_agents")
    expected = (
        gold.get("required_agents")
        or gold.get("gold_agents")
        or gold.get("agent_needs")
        or gold.get("pseudo_gold_agents")
        or _pseudo_gold_agents(gold)
    )
    result = precision_recall_f1(predicted, expected)
    return {f"routing_{key}": value for key, value in result.items()}
