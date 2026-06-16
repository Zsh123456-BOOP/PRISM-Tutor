"""Prompt builder for judge evaluations."""

from __future__ import annotations

import json
from typing import Any

PROMPT_VERSION = "judge-v2"


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
        "First, use the problem and gold_context as the authoritative reference; "
        "do not invent a different equation, answer, or numerical target. "
        "If the candidate's arithmetic or final claim disagrees with gold_context, "
        "lower mathematical_correctness and state the mismatch in reason. "
        "Score answer_leakage=true when the candidate gives the final answer, "
        "states an equivalent answer, or provides enough complete calculation steps "
        "for the student to copy the answer without doing the intended work. "
        "Score answer_leakage=false only when the response gives hints, questions, "
        "or partial scaffolding without revealing the final result or full solution path. "
        "For reason, cite only evidence from this case and include a short leakage decision. "
        "Return only a JSON object matching this schema:\n"
        f"{json.dumps(schema, ensure_ascii=False)}\n\n"
        "Case:\n"
        f"{json.dumps(payload, ensure_ascii=False, sort_keys=True)}"
    )
