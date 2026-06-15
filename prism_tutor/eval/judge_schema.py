"""Judge score schema and validation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

SCORE_FIELDS = [
    "mathematical_correctness",
    "pedagogical_quality",
    "scaffolding_quality",
    "misconception_coverage",
    "answer_leakage",
    "clarity",
    "student_facing_appropriateness",
    "overall",
]


@dataclass(frozen=True)
class JudgeScore:
    mathematical_correctness: float
    pedagogical_quality: float
    scaffolding_quality: float
    misconception_coverage: float
    answer_leakage: bool
    clarity: float
    student_facing_appropriateness: float
    overall: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    raise ValueError(f"{field} must be boolean")


def validate_score(payload: dict[str, Any]) -> JudgeScore:
    missing = [field for field in SCORE_FIELDS + ["reason"] if field not in payload]
    if missing:
        raise ValueError(f"judge score missing fields: {', '.join(missing)}")
    values: dict[str, Any] = {}
    for field in SCORE_FIELDS:
        if field == "answer_leakage":
            values[field] = _parse_bool(payload[field], field)
            continue
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field} must be numeric")
        if not 0.0 <= float(value) <= 5.0:
            raise ValueError(f"{field} must be in [0, 5]")
        values[field] = float(value)
    values["reason"] = str(payload["reason"])
    return JudgeScore(**values)


def parse_score_json(raw: str) -> JudgeScore:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("judge response does not contain a JSON object")
    return validate_score(json.loads(text[start : end + 1]))


def default_mock_score(leakage: bool = False) -> JudgeScore:
    return JudgeScore(
        mathematical_correctness=3.0,
        pedagogical_quality=3.0,
        scaffolding_quality=3.0,
        misconception_coverage=3.0,
        answer_leakage=leakage,
        clarity=3.0,
        student_facing_appropriateness=3.0,
        overall=3.0,
        reason="mock judge score; no external API call was made",
    )
