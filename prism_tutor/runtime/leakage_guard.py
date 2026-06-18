"""Runtime-only leakage guard.

This guard intentionally does not inspect gold answers. Evaluation uses the
independent detector in prism_tutor.eval.leakage_detector.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class GuardHit:
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

SOLUTION_CHAIN_PATTERNS = [
    re.compile(r"\bstep\s*1\b.*\bstep\s*2\b.*\bstep\s*3\b", re.I | re.S),
    re.compile(r"\bcomplete\s+solution\b", re.I),
    re.compile(r"\bfull\s+solution\b", re.I),
]

KEY_STEP_PATTERNS = [
    re.compile(r"\bfirst,?\s+compute\b", re.I),
    re.compile(r"\bthen\s+solve\s+for\b", re.I),
]

STUDENT_REFERENCE_PREFIXES = [
    re.compile(r"\bwhy\s+(?:do\s+)?you\s+(?:think|believe|say)\s*$", re.I),
    re.compile(r"\byour\s*$", re.I),
]


def _span_hit(sample_id: str | None, rule: str, match: re.Match[str], severity: str) -> GuardHit:
    start, end = match.span()
    return GuardHit(sample_id, rule, match.string[start:end], start, end, severity)


def _is_student_reference(text: str, match: re.Match[str]) -> bool:
    prefix = text[max(0, match.start() - 80) : match.start()]
    return any(pattern.search(prefix) for pattern in STUDENT_REFERENCE_PREFIXES)


def detect_runtime_leakage(response: Any, *, sample_id: str | None = None) -> dict[str, Any]:
    text = "" if response is None else str(response)
    hits: list[GuardHit] = []
    for pattern in DIRECT_ANSWER_PATTERNS:
        match = pattern.search(text)
        if match and not _is_student_reference(text, match):
            hits.append(_span_hit(sample_id, "runtime_direct_answer_phrase", match, "medium"))
    for pattern in SOLUTION_CHAIN_PATTERNS:
        match = pattern.search(text)
        if match:
            hits.append(_span_hit(sample_id, "runtime_complete_solution_chain", match, "high"))
    for pattern in KEY_STEP_PATTERNS:
        match = pattern.search(text)
        if match:
            hits.append(_span_hit(sample_id, "runtime_key_step_disclosure", match, "medium"))
    return {
        "rule_leakage": bool(hits),
        "matched_rules": [hit.rule for hit in hits],
        "hits": [hit.to_dict() for hit in hits],
    }
