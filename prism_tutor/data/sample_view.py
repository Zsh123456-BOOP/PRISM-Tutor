"""Runtime-safe sample views.

The model-facing runtime must not receive evaluation labels or gold answers.
Keep the allowlist conservative; evaluation code can still read the original
gold rows from dataset files.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


MODEL_INPUT_FIELDS = {
    "sample_id",
    "id",
    "dataset",
    "split",
    "source",
    "problem",
    "problem_text",
    "question",
    "context",
    "dialogue",
    "dialogue_text",
    "dialogue_history",
    "dialogue_turns",
    "student_answer",
    "student_solution",
    "student_utterance",
    "student_profile",
    "past_student_profile",
    "difficulty",
    "candidate_misconceptions",
}

GOLD_FIELDS = {
    "answer",
    "correct_answer",
    "final_answer",
    "final_numeric_answer",
    "gold_answer",
    "gold_context",
    "gold_label",
    "gold_misconception",
    "gold_misconceptions",
    "gold_response",
    "gold_solution",
    "ground_truth",
    "misconception_label",
    "misconception_labels",
    "remediation_strategy",
    "required_agents",
    "scaffolding",
    "student_error",
    "teacher_intention",
    "teacher_response",
    "tutor_response",
    "gold_teacher_move",
    "gold_is_telling",
    "gold_turn_type",
}

SAFE_METADATA_FIELDS = {
    "source",
    "source_id",
    "difficulty",
    "grade",
    "topic",
    "skill",
    "unit",
    "turn_index",
}


def build_model_input(sample: dict[str, Any]) -> dict[str, Any]:
    """Return the sample fields allowed in model/runtime prompts."""
    model_input = {
        key: deepcopy(value)
        for key, value in sample.items()
        if key in MODEL_INPUT_FIELDS and key not in GOLD_FIELDS
    }
    metadata = sample.get("metadata")
    if isinstance(metadata, dict):
        safe_metadata = {
            key: deepcopy(value)
            for key, value in metadata.items()
            if key in SAFE_METADATA_FIELDS and key not in GOLD_FIELDS
        }
        if safe_metadata:
            model_input["metadata"] = safe_metadata
    return model_input


def extract_evaluation_gold(sample: dict[str, Any]) -> dict[str, Any]:
    """Return label/gold fields for evaluation only."""
    gold = {
        key: deepcopy(value)
        for key, value in sample.items()
        if key in GOLD_FIELDS and value not in (None, "", [], {})
    }
    metadata = sample.get("metadata")
    if isinstance(metadata, dict):
        metadata_gold = {
            key: deepcopy(value)
            for key, value in metadata.items()
            if key in GOLD_FIELDS and value not in (None, "", [], {})
        }
        if metadata_gold:
            gold["metadata"] = metadata_gold
    return gold


def assert_no_gold_fields(sample: dict[str, Any]) -> None:
    """Raise if a runtime-facing sample still contains known gold fields."""
    forbidden = sorted(key for key in sample if key in GOLD_FIELDS)
    metadata = sample.get("metadata")
    if isinstance(metadata, dict):
        forbidden.extend(f"metadata.{key}" for key in sorted(metadata) if key in GOLD_FIELDS)
    if forbidden:
        raise ValueError(f"model input contains evaluation gold fields: {', '.join(forbidden)}")
