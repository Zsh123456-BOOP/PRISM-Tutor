from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any


EDUCATIONAL_RISK_FIELDS = {
    "risk_scores",
    "misconception_risk",
    "leakage_risk",
    "state_conflict_risk",
    "pedagogy_risk",
    "misconception_labels",
}


@dataclass(frozen=True)
class BaselinePlan:
    agents: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


def _problem_text(sample: dict[str, Any]) -> str:
    return " ".join(str(sample.get(key, "")) for key in ("problem", "question", "student_utterance", "context")).strip()


def _token_count(sample: dict[str, Any]) -> int:
    return len(_problem_text(sample).split())


def _numeric_count(sample: dict[str, Any]) -> int:
    return sum(char.isdigit() for char in _problem_text(sample))


def _difficulty_score(sample: dict[str, Any]) -> float:
    raw = sample.get("difficulty")
    if isinstance(raw, (int, float)):
        return max(0.0, min(1.0, float(raw)))
    if isinstance(raw, str):
        mapped = {"easy": 0.2, "low": 0.2, "medium": 0.5, "hard": 0.8, "high": 0.8}
        if raw.lower() in mapped:
            return mapped[raw.lower()]
    length_score = min(1.0, _token_count(sample) / 80)
    numeric_score = min(1.0, _numeric_count(sample) / 12)
    return 0.7 * length_score + 0.3 * numeric_score


def single_tutor_plan(sample: dict[str, Any]) -> BaselinePlan:
    return BaselinePlan(["final_tutor"], {"strategy": "single_tutor"})


def fixed_2_reflection_plan(sample: dict[str, Any]) -> BaselinePlan:
    return BaselinePlan(["hint", "verifier", "final_tutor"], {"strategy": "fixed_2_reflection"})


def fixed_4_full_plan(sample: dict[str, Any]) -> BaselinePlan:
    return BaselinePlan(["solver", "misconception", "pedagogy", "verifier", "final_tutor"], {"strategy": "fixed_4_full"})


def debate_plan(sample: dict[str, Any]) -> BaselinePlan:
    return BaselinePlan(["solver_a", "solver_b", "solver_c", "judge", "final_tutor"], {"strategy": "multi_agent_debate"})


def generic_sparse_plan(sample: dict[str, Any], *, top_k: int = 3) -> BaselinePlan:
    text = _problem_text(sample)
    tokens = text.split()
    unique_ratio = len(set(tokens)) / max(1, len(tokens))
    length = len(tokens)
    candidates = [
        ("solver", min(1.0, length / 70), 0.7, 0.2),
        ("hint", 0.6, unique_ratio, 0.1),
        ("verifier", min(1.0, _numeric_count(sample) / 8), 0.6, 0.3),
        ("pedagogy", 0.5, 1.0 - min(0.8, length / 100), 0.2),
        ("misconception", 0.4, 0.5, 0.4),
    ]
    scored = [
        {
            "agent": agent,
            "score": round((0.45 * confidence) + (0.35 * novelty) - (0.20 * redundancy), 6),
            "confidence": confidence,
            "novelty": novelty,
            "redundancy": redundancy,
        }
        for agent, confidence, novelty, redundancy in candidates
    ]
    selected = [row["agent"] for row in sorted(scored, key=lambda row: row["score"], reverse=True)[:top_k]]
    if "final_tutor" not in selected:
        selected.append("final_tutor")
    return BaselinePlan(selected, {"strategy": "generic_sparse", "scores": scored, "forbidden_fields": sorted(EDUCATIONAL_RISK_FIELDS)})


# Exp3 state-commit baselines share ONE diagnosis pipeline so the comparison is
# purely about the COMMIT mechanism, not about whether a method diagnoses. Without
# this the baselines ran no solver/misconception agent, trivially scoring ~0 state
# accuracy and confounding "ours has better state" with "ours actually diagnoses".
_STATE_DIAGNOSIS_AGENTS = ["solver", "misconception", "pedagogy", "verifier"]


def no_memory_plan(sample: dict[str, Any]) -> BaselinePlan:
    # Diagnoses like the others but never persists student state (the lower bound).
    return BaselinePlan([*_STATE_DIAGNOSIS_AGENTS, "final_tutor"], {"strategy": "no_memory"})


def _state_commit_plan(strategy: str):
    def planner(sample: dict[str, Any]) -> BaselinePlan:
        return BaselinePlan([*_STATE_DIAGNOSIS_AGENTS, "state_manager", "final_tutor"], {"strategy": strategy})

    return planner


def random_routing_plan(sample: dict[str, Any]) -> BaselinePlan:
    """Routing baseline that selects a random agent subset (seeded by sample id for
    reproducibility). This is the negative control for Exp1: it isolates whether
    PRISM's gains come from routing the RIGHT agents vs merely routing SOME agents.
    It reads no gold and no educational-risk fields."""
    pool = ["solver", "misconception", "pedagogy", "hint", "verifier"]
    seed_key = str(sample.get("sample_id") or sample.get("id") or _problem_text(sample))
    seed = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:12], 16)
    rng = random.Random(seed)
    chosen = rng.sample(pool, rng.randint(1, len(pool)))
    if "final_tutor" not in chosen:
        chosen.append("final_tutor")
    return BaselinePlan(chosen, {"strategy": "random_routing", "seed": seed})


def difficulty_routing_plan(sample: dict[str, Any]) -> BaselinePlan:
    score = _difficulty_score(sample)
    if score < 0.35:
        agents = ["hint", "final_tutor"]
        bucket = "easy"
    elif score < 0.7:
        agents = ["solver", "hint", "verifier", "final_tutor"]
        bucket = "medium"
    else:
        agents = ["solver", "hint", "pedagogy", "verifier", "final_tutor"]
        bucket = "hard"
    return BaselinePlan(agents, {"strategy": "difficulty_routing", "difficulty_score": score, "difficulty_bucket": bucket})


def _has_value(value: Any) -> bool:
    if value is None or value == "":
        return False
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def oracle_routing_plan(sample: dict[str, Any], *, evaluation_gold: dict[str, Any] | None = None) -> BaselinePlan:
    gold = evaluation_gold or {}
    metadata = gold.get("metadata") if isinstance(gold.get("metadata"), dict) else {}
    has_gold = any(
        _has_value(value)
        for value in (
            gold.get("gold_answer"),
            gold.get("gold_misconception"),
            gold.get("misconception_label"),
            gold.get("teacher_response"),
            gold.get("tutor_response"),
            gold.get("student_error"),
            gold.get("remediation_strategy"),
            metadata.get("ground_truth"),
            metadata.get("correct_answer"),
            metadata.get("student_incorrect_solution"),
        )
    )
    if not has_gold:
        return BaselinePlan([], {"strategy": "oracle_routing", "skipped": True, "skip_reason": "missing_gold_label"})
    agents = ["solver", "misconception", "pedagogy", "verifier", "final_tutor"]
    return BaselinePlan(agents, {"strategy": "oracle_routing", "upper_bound": True})


def plan_baseline_agents(
    method_name: str,
    sample: dict[str, Any],
    *,
    evaluation_gold: dict[str, Any] | None = None,
) -> BaselinePlan:
    planners = {
        "single_tutor": single_tutor_plan,
        "fixed_2": fixed_2_reflection_plan,
        "fixed_4": fixed_4_full_plan,
        "debate": debate_plan,
        "generic_sparse": generic_sparse_plan,
        "difficulty_routing": difficulty_routing_plan,
        "oracle_routing": oracle_routing_plan,
        "random_routing": random_routing_plan,
        "fixed_all_agents": fixed_4_full_plan,
        "single_round": single_tutor_plan,
        "fixed_2_rounds": fixed_2_reflection_plan,
        "fixed_3_rounds": lambda sample: BaselinePlan(["hint", "verifier", "pedagogy", "final_tutor"], {"strategy": "fixed_3_rounds"}),
        "fixed_4_rounds": fixed_4_full_plan,
        "generic_early_stopping": generic_sparse_plan,
        "no_memory": no_memory_plan,
        "naive_shared_memory": _state_commit_plan("naive_shared_memory"),
        "single_writer": _state_commit_plan("single_writer"),
        "two_phase_commit": _state_commit_plan("two_phase_commit"),
    }
    base_method = method_name.split("__", 1)[0]
    planner = planners.get(base_method)
    if planner is None:
        return BaselinePlan([], {"strategy": "unknown_baseline", "base_method": base_method})
    if base_method == "oracle_routing":
        return oracle_routing_plan(sample, evaluation_gold=evaluation_gold)
    return planner(sample)
