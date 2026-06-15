"""Blind human-audit sampling."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

FORBIDDEN_BLIND_FIELDS = {
    "method",
    "selected_agents",
    "risk_score",
    "risk_scores",
    "risk_bucket",
    "judge_score",
}

BLIND_FIELDS = [
    "audit_id",
    "sample_id",
    "dataset",
    "problem",
    "student_answer",
    "ground_truth",
    "dialogue_context",
    "candidate_response",
    "human_quality_score",
    "human_leakage_label",
    "human_preference",
    "annotator_id",
    "notes",
]


def validate_prerequisites(paths: dict[str, bool]) -> None:
    missing = [name for name, exists in paths.items() if not exists]
    if missing:
        raise RuntimeError("human audit sampling requires completed artifacts: " + ", ".join(missing))


def sample_human_audit(
    rows: list[dict[str, Any]],
    target_n: int = 200,
    dataset_targets: dict[str, int] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    dataset_targets = dataset_targets or {"mathdial": 80, "bridge": 80, "misconception": 40}
    rng = random.Random(seed)
    samples: list[dict[str, Any]] = []
    shortages: dict[str, int] = {}
    for dataset, desired in dataset_targets.items():
        pool = [row for row in rows if str(row.get("dataset", "")).lower() == dataset.lower()]
        selected = _mixed_random_and_hard(pool, min(desired, target_n - len(samples)), rng)
        samples.extend(selected)
        if len(selected) < desired:
            shortages[dataset] = desired - len(selected)
    if len(samples) < target_n:
        used = {id(row) for row in samples}
        remainder = [row for row in rows if id(row) not in used]
        samples.extend(_mixed_random_and_hard(remainder, target_n - len(samples), rng))

    blind_rows = [_blind_row(row, idx + 1) for idx, row in enumerate(samples[:target_n])]
    validate_blind_rows(blind_rows)
    return {
        "blind_rows": blind_rows,
        "manifest": {
            "target_n": target_n,
            "actual_n": len(blind_rows),
            "seed": seed,
            "dataset_targets": dataset_targets,
            "shortages": shortages,
            "sampling_policy": "50pct random stratified + 50pct hard where available",
        },
    }


def validate_blind_rows(rows: list[dict[str, Any]]) -> None:
    for idx, row in enumerate(rows, 1):
        leaked = FORBIDDEN_BLIND_FIELDS & set(row)
        if leaked:
            raise ValueError(f"blind row {idx} contains forbidden fields: {sorted(leaked)}")


def _mixed_random_and_hard(pool: list[dict[str, Any]], n: int, rng: random.Random) -> list[dict[str, Any]]:
    if n <= 0 or not pool:
        return []
    hard = [row for row in pool if _hard_score(row) > 0]
    rng.shuffle(hard)
    by_stratum: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        by_stratum[(str(row.get("dataset")), str(row.get("risk_bucket", "missing")))].append(row)
    random_part: list[dict[str, Any]] = []
    for stratum_rows in by_stratum.values():
        rng.shuffle(stratum_rows)
        random_part.extend(stratum_rows[: max(1, round(n * len(stratum_rows) / max(len(pool), 1)))])
    rng.shuffle(random_part)
    combined: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in hard[: n // 2] + random_part:
        if id(row) not in seen:
            combined.append(row)
            seen.add(id(row))
        if len(combined) >= n:
            break
    if len(combined) < n:
        rest = [row for row in pool if id(row) not in seen]
        rng.shuffle(rest)
        combined.extend(rest[: n - len(combined)])
    rng.shuffle(combined)
    return combined[:n]


def _hard_score(row: dict[str, Any]) -> int:
    score = 0
    score += int(bool(row.get("leakage_conflict")))
    score += int(bool(row.get("state_conflict_rate") and row["state_conflict_rate"] > 0))
    score += int(str(row.get("risk_bucket", "")).lower() == "high")
    score += int(abs(float(row.get("ours_baseline_delta", 0) or 0)) >= 1.0)
    score += int(float(row.get("judge_variance", 0) or 0) > 0.5)
    return score


def _blind_row(row: dict[str, Any], audit_id: int) -> dict[str, Any]:
    return {
        "audit_id": f"A{audit_id:04d}",
        "sample_id": row.get("sample_id"),
        "dataset": row.get("dataset"),
        "problem": row.get("problem"),
        "student_answer": row.get("student_answer"),
        "ground_truth": row.get("ground_truth"),
        "dialogue_context": row.get("dialogue_context"),
        "candidate_response": row.get("candidate_response") or row.get("final_response"),
        "human_quality_score": "",
        "human_leakage_label": "",
        "human_preference": "",
        "annotator_id": "",
        "notes": "",
    }
