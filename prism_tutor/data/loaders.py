"""Dataset-specific raw loaders."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from .io import read_raw_records
from .schema import first_present, make_record, stable_hash


ID_FIELDS = ("id", "record_id", "sample_id", "example_id", "uid", "question_id")
MATHDIAL_MOVE_RE = re.compile(r"^\((?P<move>[^)]+)\)\s*(?P<text>.*)$", flags=re.DOTALL)


def load_mathdial(raw_path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_record, source_file, source_index in read_raw_records(raw_path):
        for turn_index, sample in enumerate(_iter_mathdial_samples(raw_record)):
            raw_record_id = _raw_id(
                sample,
                source_file,
                source_index,
                suffix=turn_index if turn_index else None,
            )
            conversation_id = first_present(
                sample,
                (
                    "conversation_id",
                    "conversationId",
                    "dialogue_id",
                    "dialog_id",
                    "chat_id",
                    "id",
                ),
            )
            values = {
                "conversation_id": conversation_id or raw_record_id,
                "problem_text": first_present(
                    sample,
                    (
                        "problem_text",
                        "problem",
                        "question",
                        "math_problem",
                        "prompt",
                        "context.problem",
                        "scenario",
                    ),
                ),
                "student_utterance": first_present(
                    sample,
                    (
                        "student_utterance",
                        "student_response",
                        "student_answer",
                        "student",
                        "learner_utterance",
                        "learner_response",
                    ),
                ),
                "tutor_response": first_present(
                    sample,
                    (
                        "tutor_response",
                        "teacher_response",
                        "tutor",
                        "teacher",
                        "response",
                        "assistant_response",
                    ),
                ),
                "scaffolding": first_present(
                    sample,
                    ("scaffolding", "scaffold", "hints", "hint", "pedagogical_moves"),
                ),
                "leakage": first_present(
                    sample,
                    (
                        "leakage",
                        "answer_leakage",
                        "contains_answer",
                        "leaked_answer",
                        "answer_leak",
                    ),
                ),
            }
            metadata = _metadata(sample, official_split=_official_split(sample) or _split_from_source_file(source_file))
            for key in ("ground_truth", "student_incorrect_solution", "student_profile", "teacher_described_confusion"):
                value = first_present(sample, (key,))
                if value not in (None, ""):
                    metadata[key] = value
            records.append(
                make_record(
                    dataset="mathdial",
                    raw_record_id=raw_record_id,
                    values=values,
                    source_file=str(source_file),
                    metadata=metadata,
                )
            )
    return records


def load_bridge(raw_path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_record, source_file, source_index in read_raw_records(raw_path):
        raw_record_id = _raw_id(raw_record, source_file, source_index)
        values = {
            "conversation_id": first_present(
                raw_record,
                ("conversation_id", "dialogue_id", "dialog_id", "id", "c_id"),
            ),
            "problem_text": first_present(
                raw_record,
                ("problem_text", "problem", "question", "math_problem", "prompt", "context"),
            )
            or _bridge_problem_text(raw_record),
            "student_utterance": first_present(
                raw_record,
                (
                    "student_utterance",
                    "student_response",
                    "student_answer",
                    "student",
                    "learner_response",
                ),
            )
            or _last_bridge_utterance(raw_record, user="student"),
            "tutor_response": first_present(raw_record, ("tutor_response", "teacher_response", "gold_response"))
            or _join_bridge_messages(first_present(raw_record, ("c_r_", "c_revision", "c_r")), user="tutor"),
            "student_error": first_present(
                raw_record,
                ("student_error", "error_type", "error", "misstep", "bug", "label", "e"),
            ),
            "remediation_strategy": first_present(
                raw_record,
                (
                    "remediation_strategy",
                    "remediation",
                    "strategy",
                    "feedback_strategy",
                    "intervention",
                    "z_what",
                ),
            ),
            "teacher_intention": first_present(
                raw_record,
                ("teacher_intention", "teacher_intent", "intention", "goal", "teaching_goal", "z_why"),
            ),
        }
        metadata = _metadata(raw_record, official_split=_split_from_source_file(source_file))
        for key in ("lesson_topic", "c_h", "c_r", "c_r_", "c_revision"):
            value = raw_record.get(key)
            if value not in (None, ""):
                metadata[key] = value
        records.append(
            make_record(
                dataset="bridge",
                raw_record_id=raw_record_id,
                values=values,
                source_file=str(source_file),
                metadata=metadata,
            )
        )
    return records


def load_misconception(raw_path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_record, source_file, source_index in read_raw_records(raw_path):
        raw_record_id = _raw_id(raw_record, source_file, source_index)
        values = {
            "problem_text": first_present(
                raw_record,
                ("problem_text", "problem", "question", "Question", "math_problem", "prompt"),
            ),
            "student_utterance": first_present(
                raw_record,
                (
                    "student_utterance",
                    "student_response",
                    "student_answer",
                    "Incorrect Answer",
                    "answer",
                    "response",
                ),
            ),
            "misconception_label": first_present(
                raw_record,
                (
                    "misconception_label",
                    "misconception",
                    "Misconception",
                    "misconception_type",
                    "diagnosis",
                    "label",
                    "gold_label",
                    "Misconception ID",
                ),
            ),
            "sample_index": source_index,
        }
        metadata = _metadata(raw_record)
        metadata.update(
            {
                key: value
                for key, value in {
                    "misconception_id": first_present(raw_record, ("Misconception ID", "misconception_id")),
                    "topic": first_present(raw_record, ("Topic", "topic")),
                    "example_number": first_present(raw_record, ("Example Number", "example_number")),
                    "correct_answer": first_present(raw_record, ("Correct Answer", "correct_answer", "ground_truth")),
                    "explanation": first_present(raw_record, ("Explanation", "student_reasoning", "reasoning")),
                    "source": first_present(raw_record, ("Source", "source")),
                    "question_image": first_present(raw_record, ("Question image", "question_image")),
                    "learner_answer_image": first_present(raw_record, ("Learner Answer image", "learner_answer_image")),
                }.items()
                if value not in (None, "")
            }
        )
        records.append(
            make_record(
                dataset="misconception",
                raw_record_id=raw_record_id,
                values=values,
                source_file=str(source_file),
                metadata=metadata,
            )
        )
    return records


def _iter_mathdial_samples(record: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    turns = first_present(record, ("turns", "dialogue", "dialog", "messages", "conversation"))
    if isinstance(turns, str):
        yield from _iter_mathdial_string_samples(record, turns)
        return
    if not isinstance(turns, list):
        yield dict(record)
        return

    base = {key: value for key, value in record.items() if key not in {"turns", "dialogue", "dialog", "messages", "conversation"}}
    last_student: Any = first_present(base, ("student_utterance", "student_response", "student_answer"))
    emitted = False

    for index, turn in enumerate(turns):
        if not isinstance(turn, Mapping):
            continue
        role = str(first_present(turn, ("role", "speaker", "author")) or "").lower()
        text = first_present(turn, ("text", "content", "utterance", "message", "response"))
        if role in {"student", "learner", "user"}:
            last_student = text
            continue
        if role in {"tutor", "teacher", "assistant"}:
            sample = dict(base)
            sample.update(dict(turn))
            sample.setdefault("student_utterance", last_student)
            sample.setdefault("tutor_response", text)
            sample.setdefault("turn_index", index)
            yield sample
            emitted = True

    if not emitted:
        sample = dict(base)
        sample["dialogue_turns"] = turns
        yield sample


def _iter_mathdial_string_samples(record: Mapping[str, Any], conversation: str) -> Iterator[dict[str, Any]]:
    base = {key: value for key, value in record.items() if key != "conversation"}
    last_student: Any = first_present(base, ("student_utterance", "student_response", "student_answer", "student_incorrect_solution"))
    emitted = False
    for turn_index, raw_turn in enumerate(part.strip() for part in conversation.split("|EOM|")):
        if not raw_turn or ":" not in raw_turn:
            continue
        speaker, content = raw_turn.split(":", 1)
        role = _mathdial_role(speaker)
        content = content.strip()
        if role == "student":
            last_student = content
            continue
        if role != "tutor":
            continue
        move, text = _split_teacher_move(content)
        sample = dict(base)
        sample.update(
            {
                "conversation_id": first_present(record, ("qid", "id", "conversation_id")) or _raw_id(record, Path("mathdial"), 0),
                "student_utterance": last_student,
                "tutor_response": text,
                "scaffolding": [move] if move else [],
                "turn_index": turn_index,
            }
        )
        yield sample
        emitted = True

    if not emitted:
        sample = dict(base)
        sample["dialogue_text"] = conversation
        yield sample


def _bridge_problem_text(record: Mapping[str, Any]) -> str | None:
    topic = first_present(record, ("lesson_topic",))
    history = first_present(record, ("c_h", "dialogue_history"))
    history_text = _join_bridge_messages(history)
    if topic and history_text:
        return f"Lesson topic: {topic}\nDialogue history:\n{history_text}"
    if topic:
        return f"Lesson topic: {topic}"
    return history_text


def _last_bridge_utterance(record: Mapping[str, Any], *, user: str) -> str | None:
    history = first_present(record, ("c_h", "dialogue_history"))
    if not isinstance(history, list):
        return None
    for item in reversed(history):
        if isinstance(item, Mapping) and str(item.get("user", "")).lower() == user:
            text = item.get("text")
            return str(text) if text not in (None, "") else None
    return None


def _join_bridge_messages(messages: Any, *, user: str | None = None) -> str | None:
    if not isinstance(messages, list):
        return None
    parts: list[str] = []
    for item in messages:
        if not isinstance(item, Mapping):
            continue
        speaker = str(item.get("user", "")).lower()
        if user is not None and speaker != user:
            continue
        text = item.get("text")
        if text in (None, ""):
            continue
        prefix = speaker.title() if speaker else "Turn"
        parts.append(f"{prefix}: {text}")
    return "\n".join(parts) if parts else None


def _mathdial_role(speaker: str) -> str:
    normalized = speaker.strip().lower()
    if normalized in {"teacher", "tutor", "assistant"}:
        return "tutor"
    if normalized in {"student", "learner", "user"}:
        return "student"
    return "student"


def _split_teacher_move(text: str) -> tuple[str | None, str]:
    match = MATHDIAL_MOVE_RE.match(text.strip())
    if not match:
        return None, text.strip()
    return match.group("move").strip(), match.group("text").strip()


def _raw_id(record: Mapping[str, Any], source_file: Path, source_index: int, suffix: int | None = None) -> str:
    value = first_present(record, ID_FIELDS)
    if value is None:
        value = f"{source_file.name}:{source_index}"
    raw_id = str(value)
    if suffix is not None:
        raw_id = f"{raw_id}:{suffix}"
    return raw_id


def _official_split(record: Mapping[str, Any]) -> str | None:
    split = first_present(record, ("split", "official_split", "partition", "set"))
    if split is None:
        return None
    normalized = str(split).strip().lower()
    if normalized in {"validation", "valid", "val"}:
        return "dev"
    if normalized in {"train", "dev", "test"}:
        return normalized
    return normalized or None


def _split_from_source_file(source_file: Path) -> str | None:
    split = source_file.stem.lower()
    if split in {"train", "dev", "test"}:
        return split
    return None


def _metadata(record: Mapping[str, Any], official_split: str | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {"raw_fingerprint": stable_hash(record, length=16)}
    if official_split:
        metadata["official_split"] = official_split
    return metadata
