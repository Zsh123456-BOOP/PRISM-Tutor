"""Runtime false-affirmation guard (detection).

Detects when the student-facing response affirms the student's work as correct
while the system's OWN diagnosis indicates the student is wrong (a "false
affirmation" / over-validation -- the failure highlighted as an educational
safety risk by recent benchmark work, e.g. Confirming-Correct (arXiv 2605.16207)
and Sycophancy-is-an-Educational-Safety-Risk (arXiv 2605.14604), neither of which
provides a runtime mitigation).

Gold-free: the "student is wrong" signal comes from internal agent outputs
(misconception detection, verifier verdict, solver-vs-student answer mismatch),
never from gold. Evaluation uses the gold-based over-validation metric in
prism_tutor.eval.affirmation_metrics.
"""

from __future__ import annotations

import re
from typing import Any

from prism_tutor.utils.answers import answers_match, extract_final_numeric

_AFFIRM_PATTERNS = [
    # "correct" (optionally with an intensifier); \b excludes "incorrect".
    re.compile(r"\b(?:absolutely\s+|exactly\s+|completely\s+|perfectly\s+|quite\s+)?correct\b", re.I),
    re.compile(
        r"\b(?:well done|good job|great job|nicely done|exactly right|spot on|perfect|"
        r"you got it|that'?s right|you are right|correct!|right!)\b",
        re.I,
    ),
    re.compile(r"\byes[,!\.]\s", re.I),
    re.compile(r"答案?(?:完全)?正确|做得(?:很)?好|完全正确|没错|对了"),
]

_CORRECTION_PATTERNS = [
    re.compile(
        r"\b(?:not\s+quite|not\s+correct|isn'?t\s+right|not\s+right|incorrect|a\s+mistake|"
        r"an\s+error|let'?s\s+(?:re-?check|re-?examine|reconsider|revisit)|however|but\b|"
        r"actually|wrong|double-?check|re-?check|reconsider)\b",
        re.I,
    ),
    re.compile(r"不(?:太)?对|有误|错误|再(?:检查|想想|看看|算)|重新|并不"),
]


def detect_false_affirmation(response: Any) -> bool:
    """True if the response affirms correctness with no correction cue present."""
    text = "" if response is None else str(response)
    if not any(pattern.search(text) for pattern in _AFFIRM_PATTERNS):
        return False
    if any(pattern.search(text) for pattern in _CORRECTION_PATTERNS):
        return False
    return True


def student_answer_disagrees_with_solver(student_text: Any, solver_answer: Any) -> bool:
    """Gold-free wrongness signal: the student's stated number differs from the
    solver's OWN computed answer (both model-side, never gold)."""
    student_num = extract_final_numeric(student_text)
    solver_num = extract_final_numeric(solver_answer)
    if student_num is None or solver_num is None:
        return False
    return answers_match(student_num, solver_num) is False
