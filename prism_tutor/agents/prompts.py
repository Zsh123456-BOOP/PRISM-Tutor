"""Prompt templates for JSON-only PRISM-Tutor agents."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from .schemas import AGENT_SCHEMAS


SYSTEM_PROMPTS: dict[str, str] = {
    "solver": "You solve the math task and return only valid JSON. Do not include markdown or prose outside JSON.",
    "misconception": "You diagnose student misconceptions from the sample and prior agent outputs. Return only valid JSON.",
    "pedagogy": "You choose a teaching strategy that helps learning without leaking the answer. Return only valid JSON.",
    "hint": "You write a scaffolded hint. Avoid revealing the final answer or key solution steps. Return only valid JSON.",
    "verifier": "You verify correctness, answer leakage, pedagogy fit, and state conflicts. Return only valid JSON.",
    "state_manager": "You propose student-state updates with evidence. Do not commit changes yourself. Return only valid JSON.",
    "final_tutor": "You write the student-facing response. Do not reveal the final answer or key steps unless the task explicitly allows it. Return only valid JSON.",
}


def schema_text(schema: type[BaseModel]) -> str:
    return json.dumps(schema.model_json_schema(), ensure_ascii=False, sort_keys=True)


def build_agent_messages(
    agent_name: str,
    sample: dict[str, Any],
    state: dict[str, Any] | None = None,
    error_summary: str | None = None,
) -> list[dict[str, str]]:
    schema = AGENT_SCHEMAS[agent_name]
    system = (
        f"{SYSTEM_PROMPTS[agent_name]}\n"
        "You must satisfy this JSON schema exactly:\n"
        f"{schema_text(schema)}"
    )
    payload = {"sample": sample, "runtime_state": state or {}}
    if error_summary:
        payload["previous_error"] = error_summary
        payload["retry_instruction"] = "Repair the output. Return one JSON object only."
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]
