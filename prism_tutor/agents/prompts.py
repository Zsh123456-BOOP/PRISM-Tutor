"""Prompt templates for JSON-only PRISM-Tutor agents."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from prism_tutor.data.sample_view import build_model_input

from .schemas import AGENT_SCHEMAS


SYSTEM_PROMPTS: dict[str, str] = {
    "solver": "You solve the math task and return only valid JSON. Do not include markdown or prose outside JSON.",
    "misconception": "You diagnose student misconceptions from the sample and prior agent outputs. Return only valid JSON.",
    "pedagogy": "You choose a teaching strategy that helps learning without leaking the answer. Return only valid JSON.",
    "hint": "You write a scaffolded hint. Avoid revealing the final answer or key solution steps. Return only valid JSON.",
    "verifier": "You verify correctness, answer leakage, pedagogy fit, and state conflicts. Return only valid JSON.",
    "state_manager": "You propose student-state updates with evidence. Do not commit changes yourself. Return only valid JSON.",
    "final_tutor": (
        "You write the student-facing response. Do not reveal the final answer, a complete solution path, "
        "or key computational steps unless the task explicitly allows it. Prefer one concise guiding question "
        "or next-step hint. If prior agents contain the answer, keep it internal and do not quote it. "
        "Return only valid JSON."
    ),
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
    model_input = build_model_input(sample)
    system = (
        f"{SYSTEM_PROMPTS[agent_name]}\n"
        "You must satisfy this JSON schema exactly:\n"
        f"{schema_text(schema)}"
    )
    if agent_name == "misconception" and model_input.get("candidate_misconceptions"):
        system += (
            "\nClassification constraint: first compare the student's answer to the reference "
            "solution in runtime_state.agent_outputs.solver (if present) to pinpoint the exact "
            "error, then select misconception_labels copied VERBATIM from "
            "sample.candidate_misconceptions (the fixed benchmark taxonomy). List the 1-3 most "
            "likely labels, MOST LIKELY FIRST. Return an empty list only if none apply. Do not "
            "invent new label text."
        )
    payload = {"sample": model_input, "runtime_state": state or {}}
    if error_summary:
        payload["previous_error"] = error_summary
        payload["retry_instruction"] = "Repair the output. Return one JSON object only."
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]
