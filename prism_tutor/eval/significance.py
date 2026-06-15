"""Dependency-light paired significance helpers."""

from __future__ import annotations

import math
import random
from statistics import mean, pstdev
from typing import Any, Callable


def paired_values(
    rows: list[dict[str, Any]],
    metric: str,
    method_a: str,
    method_b: str,
    sample_key: str = "sample_id",
) -> tuple[list[float], list[float]]:
    by_sample: dict[Any, dict[str, float]] = {}
    for row in rows:
        if row.get("method") not in {method_a, method_b}:
            continue
        value = row.get(metric)
        if not isinstance(value, (int, float, bool)):
            continue
        by_sample.setdefault(row.get(sample_key), {})[str(row.get("method"))] = float(value)
    a_values: list[float] = []
    b_values: list[float] = []
    for pair in by_sample.values():
        if method_a in pair and method_b in pair:
            a_values.append(pair[method_a])
            b_values.append(pair[method_b])
    return a_values, b_values


def paired_bootstrap_ci(
    a_values: list[float],
    b_values: list[float],
    iterations: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, Any]:
    if len(a_values) != len(b_values) or not a_values:
        return {"n": 0, "mean_diff": None, "ci_low": None, "ci_high": None, "reason": "missing_pairs"}
    rng = random.Random(seed)
    diffs = [a - b for a, b in zip(a_values, b_values)]
    boot = []
    for _ in range(max(1, iterations)):
        sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        boot.append(mean(sample))
    boot.sort()
    low_i = int((alpha / 2) * (len(boot) - 1))
    high_i = int((1 - alpha / 2) * (len(boot) - 1))
    return {
        "n": len(diffs),
        "mean_diff": mean(diffs),
        "ci_low": boot[low_i],
        "ci_high": boot[high_i],
        "reason": None,
    }


def wilcoxon_signed_rank(a_values: list[float], b_values: list[float]) -> dict[str, Any]:
    diffs = [a - b for a, b in zip(a_values, b_values) if a != b]
    n = len(diffs)
    if n < 2:
        return {"n": n, "statistic": None, "p_value": None, "reason": "too_few_nonzero_pairs"}
    ranked = sorted((abs(diff), diff) for diff in diffs)
    rank_sum_pos = sum(rank for rank, (_, diff) in enumerate(ranked, 1) if diff > 0)
    rank_sum_neg = sum(rank for rank, (_, diff) in enumerate(ranked, 1) if diff < 0)
    statistic = min(rank_sum_pos, rank_sum_neg)
    expected = n * (n + 1) / 4
    variance = n * (n + 1) * (2 * n + 1) / 24
    z = (statistic - expected) / math.sqrt(variance) if variance else 0.0
    p_value = 2 * _normal_cdf(-abs(z))
    return {"n": n, "statistic": statistic, "p_value": p_value, "reason": "normal_approximation"}


def mcnemar_test(a_values: list[float], b_values: list[float]) -> dict[str, Any]:
    b01 = sum(a == 0 and b == 1 for a, b in zip(a_values, b_values))
    b10 = sum(a == 1 and b == 0 for a, b in zip(a_values, b_values))
    n = b01 + b10
    if n == 0:
        return {"n": len(a_values), "statistic": None, "p_value": None, "b01": b01, "b10": b10, "reason": "no_discordant_pairs"}
    statistic = (abs(b01 - b10) - 1) ** 2 / n
    p_value = 1 - _chi_square_1_cdf(statistic)
    return {"n": len(a_values), "statistic": statistic, "p_value": p_value, "b01": b01, "b10": b10, "reason": "continuity_corrected"}


def cohens_d_paired(a_values: list[float], b_values: list[float]) -> float | None:
    diffs = [a - b for a, b in zip(a_values, b_values)]
    if len(diffs) < 2:
        return None
    sd = pstdev(diffs)
    return mean(diffs) / sd if sd else 0.0


def compare_methods(
    rows: list[dict[str, Any]],
    metric: str,
    method_a: str,
    method_b: str,
    binary: bool = False,
) -> dict[str, Any]:
    a_values, b_values = paired_values(rows, metric, method_a, method_b)
    ci = paired_bootstrap_ci(a_values, b_values)
    test = mcnemar_test(a_values, b_values) if binary else wilcoxon_signed_rank(a_values, b_values)
    return {
        "method_a": method_a,
        "method_b": method_b,
        "metric": metric,
        **ci,
        "test": "mcnemar" if binary else "wilcoxon_signed_rank",
        "p_value": test.get("p_value"),
        "test_statistic": test.get("statistic"),
        "effect_size": cohens_d_paired(a_values, b_values),
        "test_reason": test.get("reason"),
    }


def holm_correction(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [row for row in results if isinstance(row.get("p_value"), (int, float))]
    ordered = sorted(valid, key=lambda row: row["p_value"])
    m = len(ordered)
    adjusted: dict[int, float] = {}
    for idx, row in enumerate(ordered):
        adjusted[id(row)] = min(1.0, row["p_value"] * (m - idx))
    return [{**row, "holm_p_value": adjusted.get(id(row))} for row in results]


def _normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _chi_square_1_cdf(x: float) -> float:
    return math.erf(math.sqrt(max(x, 0.0) / 2))
