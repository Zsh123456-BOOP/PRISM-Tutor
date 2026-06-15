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
    "display_order",
    "sample_id",
    "dataset",
    "problem",
    "student_answer",
    "ground_truth",
    "dialogue_context",
    "candidate_response",
    "candidate_a_response",
    "candidate_b_response",
    "human_preference_ab",
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

    display_samples = samples[:target_n]
    rng.shuffle(display_samples)
    pair_candidates = _pair_candidates(rows, display_samples, rng)
    blind_rows = [
        _blind_row(row, idx + 1, pair_candidates.get(_pair_key(row)))
        for idx, row in enumerate(display_samples)
    ]
    validate_blind_rows(blind_rows)
    preference_mapping = [
        _mapping_row(blind_row, pair_candidates.get((str(blind_row.get("dataset")), str(blind_row.get("sample_id")))))
        for blind_row in blind_rows
        if pair_candidates.get((str(blind_row.get("dataset")), str(blind_row.get("sample_id"))))
    ]
    return {
        "blind_rows": blind_rows,
        "preference_mapping": preference_mapping,
        "manifest": {
            "target_n": target_n,
            "actual_n": len(blind_rows),
            "seed": seed,
            "dataset_targets": dataset_targets,
            "shortages": shortages,
            "sampling_policy": "50pct random stratified + 50pct hard where available",
            "display_order_seed": seed,
            "display_order_audit_ids": [row["audit_id"] for row in blind_rows],
            "display_order_sample_ids": [row.get("sample_id") for row in blind_rows],
            "pairwise_preference_rows": len(preference_mapping),
            "preference_mapping_file": "preference_mapping.json",
        },
    }


def validate_blind_rows(rows: list[dict[str, Any]]) -> None:
    for idx, row in enumerate(rows, 1):
        leaked = FORBIDDEN_BLIND_FIELDS & set(row)
        if leaked:
            raise ValueError(f"blind row {idx} contains forbidden fields: {sorted(leaked)}")
        if int(row.get("display_order", idx)) != idx:
            raise ValueError(f"blind row {idx} has invalid display_order")


def blind_row_content_issues(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    required = ["problem", "candidate_response"]
    for idx, row in enumerate(rows, 1):
        missing = [field for field in required if row.get(field) in (None, "")]
        if missing:
            issues.append({"row_index": idx, "audit_id": row.get("audit_id"), "sample_id": row.get("sample_id"), "missing": missing})
    return issues


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


def _pair_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("dataset")), str(row.get("sample_id")))


def _pair_candidates(all_rows: list[dict[str, Any]], selected_rows: list[dict[str, Any]], rng: random.Random) -> dict[tuple[str, str], dict[str, Any]]:
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        by_key[_pair_key(row)].append(row)
    pairs: dict[tuple[str, str], dict[str, Any]] = {}
    for selected in selected_rows:
        key = _pair_key(selected)
        candidates = [row for row in by_key.get(key, []) if _candidate_text(row)]
        if len(candidates) < 2:
            continue
        primary = _prefer_ours(candidates) or selected
        if not _candidate_text(primary):
            primary = candidates[0]
        baselines = [row for row in candidates if row is not primary and str(row.get("method")) != str(primary.get("method"))]
        if not baselines:
            baselines = [row for row in candidates if row is not primary]
        if not baselines:
            continue
        baseline = _strongest_baseline(baselines)
        a_is_primary = rng.random() < 0.5
        pairs[key] = {
            "candidate_a": primary if a_is_primary else baseline,
            "candidate_b": baseline if a_is_primary else primary,
            "preferred_method": primary.get("method"),
            "baseline_method": baseline.get("method"),
        }
    return pairs


def _prefer_ours(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for method_name in ("ours_full", "ours_routing_budget", "ours_routing"):
        match = next((row for row in rows if str(row.get("method")) == method_name), None)
        if match:
            return match
    return next((row for row in rows if str(row.get("method", "")).startswith("ours")), None)


def _strongest_baseline(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda row: float(row.get("internal_correctness") or row.get("overall") or 0))


def _candidate_text(row: dict[str, Any]) -> Any:
    return row.get("candidate_response") or row.get("final_response") or row.get("response")


def _mapping_row(blind_row: dict[str, Any], pair: dict[str, Any] | None) -> dict[str, Any]:
    assert pair is not None
    candidate_a = pair["candidate_a"]
    candidate_b = pair["candidate_b"]
    return {
        "audit_id": blind_row.get("audit_id"),
        "sample_id": blind_row.get("sample_id"),
        "dataset": blind_row.get("dataset"),
        "candidate_a_method": candidate_a.get("method"),
        "candidate_b_method": candidate_b.get("method"),
        "candidate_a_is_ours": str(candidate_a.get("method", "")).startswith("ours"),
        "candidate_b_is_ours": str(candidate_b.get("method", "")).startswith("ours"),
        "preferred_method": pair.get("preferred_method"),
        "baseline_method": pair.get("baseline_method"),
    }


def _blind_row(row: dict[str, Any], audit_id: int, pair: dict[str, Any] | None = None) -> dict[str, Any]:
    candidate_a = pair.get("candidate_a") if pair else None
    candidate_b = pair.get("candidate_b") if pair else None
    return {
        "audit_id": f"A{audit_id:04d}",
        "display_order": audit_id,
        "sample_id": row.get("sample_id"),
        "dataset": row.get("dataset"),
        "problem": row.get("problem"),
        "student_answer": row.get("student_answer"),
        "ground_truth": row.get("ground_truth"),
        "dialogue_context": row.get("dialogue_context"),
        "candidate_response": row.get("candidate_response") or row.get("final_response"),
        "candidate_a_response": _candidate_text(candidate_a) if isinstance(candidate_a, dict) else "",
        "candidate_b_response": _candidate_text(candidate_b) if isinstance(candidate_b, dict) else "",
        "human_quality_score": "",
        "human_leakage_label": "",
        "human_preference_ab": "",
        "human_preference": "",
        "annotator_id": "",
        "notes": "",
    }
