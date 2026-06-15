"""Human-audit agreement metrics."""

from __future__ import annotations

from collections import Counter
from math import sqrt
from typing import Any

CORE_HUMAN_AGREEMENT_COLUMNS = {
    "sample_id",
    "annotator_id",
    "human_quality_score",
    "human_leakage_label",
    "human_preference",
}


def cohen_kappa(labels_a: list[Any], labels_b: list[Any]) -> dict[str, Any]:
    pairs = [(a, b) for a, b in zip(labels_a, labels_b) if a not in (None, "") and b not in (None, "")]
    n = len(pairs)
    if n == 0:
        return {"n": 0, "kappa": None, "reason": "no_overlap"}
    observed = sum(a == b for a, b in pairs) / n
    count_a = Counter(a for a, _ in pairs)
    count_b = Counter(b for _, b in pairs)
    expected = sum((count_a[label] / n) * (count_b[label] / n) for label in set(count_a) | set(count_b))
    kappa = (observed - expected) / (1 - expected) if expected != 1 else 1.0
    return {"n": n, "kappa": kappa, "observed_agreement": observed, "expected_agreement": expected}


def spearman_correlation(values_a: list[Any], values_b: list[Any]) -> dict[str, Any]:
    pairs = []
    for a, b in zip(values_a, values_b):
        try:
            pairs.append((float(a), float(b)))
        except (TypeError, ValueError):
            continue
    if len(pairs) < 2:
        return {"n": len(pairs), "spearman": None, "reason": "too_few_pairs"}
    rank_a = _ranks([a for a, _ in pairs])
    rank_b = _ranks([b for _, b in pairs])
    return {"n": len(pairs), "spearman": _pearson(rank_a, rank_b)}


def preference_win_rate(rows: list[dict[str, Any]], ours_label: str = "ours") -> dict[str, Any]:
    labels = [str(row.get("human_preference", "")).lower() for row in rows]
    valid = [label for label in labels if label in {ours_label, "baseline", "tie"}]
    if not valid:
        return {"n": 0, "ours_win_rate": None, "tie_rate": None}
    return {
        "n": len(valid),
        "ours_win_rate": sum(label == ours_label for label in valid) / len(valid),
        "tie_rate": sum(label == "tie" for label in valid) / len(valid),
    }


def build_agreement_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    missing = CORE_HUMAN_AGREEMENT_COLUMNS - set(rows[0]) if rows else CORE_HUMAN_AGREEMENT_COLUMNS
    if missing:
        return {"schema_error": f"missing columns: {sorted(missing)}"}
    by_sample: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        by_sample.setdefault(row.get("sample_id"), []).append(row)
    paired = [items[:2] for items in by_sample.values() if len(items) >= 2]
    quality_a = [pair[0].get("human_quality_score") for pair in paired]
    quality_b = [pair[1].get("human_quality_score") for pair in paired]
    leakage_a = [pair[0].get("human_leakage_label") for pair in paired]
    leakage_b = [pair[1].get("human_leakage_label") for pair in paired]
    return {
        "quality_spearman": spearman_correlation(quality_a, quality_b),
        "leakage_kappa": cohen_kappa(leakage_a, leakage_b),
        "preference": preference_win_rate(rows),
    }


def _ranks(values: list[float]) -> list[float]:
    sorted_values = sorted((value, idx) for idx, value in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(sorted_values):
        j = i
        while j + 1 < len(sorted_values) and sorted_values[j + 1][0] == sorted_values[i][0]:
            j += 1
        rank = (i + j + 2) / 2
        for _, idx in sorted_values[i : j + 1]:
            ranks[idx] = rank
        i = j + 1
    return ranks


def _pearson(a_values: list[float], b_values: list[float]) -> float | None:
    n = len(a_values)
    mean_a = sum(a_values) / n
    mean_b = sum(b_values) / n
    cov = sum((a - mean_a) * (b - mean_b) for a, b in zip(a_values, b_values))
    var_a = sum((a - mean_a) ** 2 for a in a_values)
    var_b = sum((b - mean_b) ** 2 for b in b_values)
    denom = sqrt(var_a * var_b)
    return cov / denom if denom else 0.0
