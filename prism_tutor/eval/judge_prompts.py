"""Prompt builder for judge evaluations."""

from __future__ import annotations

import json
from typing import Any

PROMPT_VERSION = "judge-v1"


def build_judge_prompt(case: dict[str, Any]) -> str:
    payload = {
        "problem": case.get("problem"),
        "student_answer": case.get("student_answer"),
        "gold_context": case.get("gold_context") or case.get("ground_truth"),
        "candidate_response": case.get("candidate_response") or case.get("final_response"),
    }
    schema = {
        "mathematical_correctness": "number 0-5",
        "pedagogical_quality": "number 0-5",
        "scaffolding_quality": "number 0-5",
        "misconception_coverage": "number 0-5",
        "answer_leakage": "boolean",
        "clarity": "number 0-5",
        "student_facing_appropriateness": "number 0-5",
        "overall": "number 0-5",
        "reason": "short string",
    }
    return (
        "You are an independent tutoring-response judge. "
        "Evaluate the candidate response for a math tutoring setting. "
        "Return only a JSON object matching this schema:\n"
        f"{json.dumps(schema, ensure_ascii=False)}\n\n"
        "Case:\n"
        f"{json.dumps(payload, ensure_ascii=False, sort_keys=True)}"
    )
