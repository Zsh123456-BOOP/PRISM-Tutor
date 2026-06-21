"""Rule-based answer leakage detector.

Rule hits are evidence, not the final leakage label. Task Card 12 combines rule
hits with judge output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any

from prism_tutor.utils.answers import extract_final_numeric

from .correctness import normalize_answer


# Asserting a computed result near the gold final answer => the tutor told it
# (MathDial's "Telling"). Used to catch leakage that exact-narrative matching
# misses (gold answers are often full worked-solution narratives).
_TELLING_ASSERTION_RE = re.compile(
    r"(=|\bequals?\b|\bis\b|\bare\b|\banswer\b|\bresult\b|\bget\b|\btotal\b|答案|结果|得到|一共|"
    r"\btherefore\b|\bthus\b|\bhence\b|\bso\b)",
    re.I,
)
_ALL_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


@dataclass(frozen=True)
class LeakageHit:
    sample_id: str | None
    rule: str
    evidence: str
    start: int
    end: int
    severity: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DIRECT_ANSWER_PATTERNS = [
    re.compile(r"\bthe\s+(?:final\s+)?answer\s+is\b", re.I),
    re.compile(r"\b答案是\b"),
    re.compile(r"\bso\s+the\s+answer\s+is\b", re.I),
]

STUDENT_ANSWER_REFERENCE_PATTERNS = [
    re.compile(r"\b(?:why|how)\s+(?:do\s+)?you\s+(?:think|believe|say)\s*$", re.I),
    re.compile(r"\byou\s+(?:think|believe|say|chose|selected)\s*$", re.I),
    re.compile(r"\byour\s*$", re.I),
]

SOLUTION_CHAIN_PATTERNS = [
    re.compile(r"\bstep\s*1\b.*\bstep\s*2\b.*\bstep\s*3\b", re.I | re.S),
    re.compile(r"\bcomplete\s+solution\b", re.I),
    re.compile(r"\bfull\s+solution\b", re.I),
]

KEY_STEP_PATTERNS = [
    re.compile(r"\bfirst,?\s+compute\b", re.I),
    re.compile(r"\bthen\s+solve\s+for\b", re.I),
    re.compile(r"\bplug\s+.*\s+into\s+.*\b", re.I),
]


def _span_hit(sample_id: str | None, rule: str, match: re.Match[str], severity: str) -> LeakageHit:
    start, end = match.span()
    return LeakageHit(sample_id, rule, match.string[start:end], start, end, severity)


def _is_student_answer_reference(text: str, match: re.Match[str]) -> bool:
    context_start = max(0, match.start() - 80)
    prefix = text[context_start : match.start()]
    return any(pattern.search(prefix) for pattern in STUDENT_ANSWER_REFERENCE_PATTERNS)


def detect_leakage(response: Any, gold: dict[str, Any] | None = None, sample_id: str | None = None) -> dict[str, Any]:
    text = "" if response is None else str(response)
    hits: list[LeakageHit] = []
    gold = gold or {}

    gold_answer = gold.get("answer") or gold.get("ground_truth") or gold.get("final_answer")
    if not gold_answer:
        # Several datasets (e.g. MathDial) store the gold answer under metadata,
        # not at the top level; without this the telling/final-answer rules can
        # never fire and leakage reads as ~0 for every method.
        meta = gold.get("metadata")
        if isinstance(meta, dict):
            gold_answer = meta.get("final_answer") or meta.get("ground_truth") or meta.get("answer")
    normalized_gold = normalize_answer(gold_answer)
    normalized_text = normalize_answer(text)
    if normalized_gold and normalized_gold in normalized_text:
        start = normalized_text.find(normalized_gold)
        hits.append(
            LeakageHit(
                sample_id=sample_id,
                rule="final_answer_match",
                evidence=str(gold_answer),
                start=max(start, 0),
                end=max(start, 0) + len(normalized_gold),
                severity="high",
            )
        )

    # Telling: the response states the gold FINAL numeric answer in an asserting
    # context. Catches leakage that exact-narrative matching misses because gold
    # answers are stored as full worked-solution narratives.
    gold_num = extract_final_numeric(gold_answer)
    if gold_num is not None:
        for match in _ALL_NUM_RE.finditer(text):
            token = match.group().replace(",", "").rstrip(".")
            if token != gold_num:
                continue
            context = text[max(0, match.start() - 30) : match.start()]
            if "=" in context or _TELLING_ASSERTION_RE.search(context):
                hits.append(
                    LeakageHit(
                        sample_id=sample_id,
                        rule="telling_final_answer",
                        evidence=match.group(),
                        start=match.start(),
                        end=match.end(),
                        severity="high",
                    )
                )
                break

    for pattern in DIRECT_ANSWER_PATTERNS:
        match = pattern.search(text)
        if match:
            if _is_student_answer_reference(text, match):
                continue
            hits.append(_span_hit(sample_id, "direct_answer_phrase", match, "medium"))
    for pattern in SOLUTION_CHAIN_PATTERNS:
        match = pattern.search(text)
        if match:
            hits.append(_span_hit(sample_id, "complete_solution_chain", match, "high"))
    for pattern in KEY_STEP_PATTERNS:
        match = pattern.search(text)
        if match:
            hits.append(_span_hit(sample_id, "key_step_disclosure", match, "medium"))

    return {
        "rule_leakage": bool(hits),
        "matched_rules": [hit.rule for hit in hits],
        "hits": [hit.to_dict() for hit in hits],
    }
