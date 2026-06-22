"""Over-validation (false-affirmation) evaluation metric.

Over-validation = the tutor affirms the student's work as correct when the
student is actually wrong. Unlike the runtime guard, evaluation MAY use gold to
decide whether the student is wrong. Reuses the runtime affirmation detector for
the "affirms" signal so runtime and eval stay consistent.
"""

from __future__ import annotations

from typing import Any

from prism_tutor.runtime.affirmation_guard import detect_false_affirmation
from prism_tutor.utils.answers import answers_match


def _gold_field(gold: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if gold.get(key) not in (None, ""):
            return gold.get(key)
    meta = gold.get("metadata")
    if isinstance(meta, dict):
        for key in keys:
            if meta.get(key) not in (None, ""):
                return meta.get(key)
    return None


def detect_over_validation(response: Any, gold: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return whether the response over-validates an incorrect student.

    ``over_validation`` is None when the student is not (knowably) wrong, so it is
    excluded from the over-validation rate (which is computed only over wrong
    student cases).
    """
    gold = gold or {}
    final_answer = _gold_field(gold, "final_answer", "ground_truth", "answer")
    student = _gold_field(gold, "student_incorrect_solution", "student_solution", "student_answer")

    # The MathDial-style field is the incorrect solution by construction. Otherwise
    # compare the student's number against the gold answer.
    student_wrong = False
    if _gold_field(gold, "student_incorrect_solution") not in (None, ""):
        student_wrong = True
    elif student is not None and final_answer is not None:
        student_wrong = answers_match(student, final_answer) is False

    affirms = detect_false_affirmation(response)
    return {
        "student_wrong": student_wrong,
        "affirms_response": affirms,
        "over_validation": (affirms if student_wrong else None),
    }
