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
    # The candidate misconception taxonomy (55 long labels) is only needed by the
    # misconception agent for constrained classification. Injecting it into every
    # agent's prompt was ~40% of per-sample tokens on the Misconception Benchmark.
    if agent_name != "misconception":
        model_input.pop("candidate_misconceptions", None)
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
    if agent_name == "final_tutor" and isinstance(state, dict):
        guard = (state.get("agent_outputs") or {}).get("leakage_guard")
        guard_entry = guard[-1] if isinstance(guard, list) and guard else guard
        instruction = guard_entry.get("instruction") if isinstance(guard_entry, dict) else None
        if instruction:
            system += "\nLEAKAGE GUARD (mandatory): " + str(instruction)
    payload = {"sample": model_input, "runtime_state": state or {}}
    if error_summary:
        payload["previous_error"] = error_summary
        payload["retry_instruction"] = "Repair the output. Return one JSON object only."
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]
