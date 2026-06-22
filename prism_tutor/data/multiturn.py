"""Multi-turn MathDial dialogue eval builder.

The single-turn split keeps only one (error) turn per conversation, which is why
risk routing / budget never vary: every retained sample needs full diagnosis.
MathDial's raw ``conversation`` field is a full multi-turn tutoring dialogue whose
teacher turns are tagged with a *move* (``generic`` / ``focus`` / ``probing`` /
``telling``). Running PRISM turn-by-turn over the whole dialogue exposes the
heterogeneity the adaptive modules are designed for:

- ``generic`` turns (greetings, encouragement, "go on") are low-stakes -> the QoS
  router should route a minimal agent set and the budget controller should stop
  early.
- ``focus`` / ``probing`` turns need full diagnosis.
- ``telling`` turns are the premature-answer-revelation failure the guard targets.

Gold (the teacher move, the reference teacher response, the answer) is kept under
``metadata`` so :func:`prism_tutor.data.sample_view.build_model_input` strips it
before the runtime sees it; evaluation reads it back from the gold rows.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Iterator

_MOVE_RE = re.compile(r"^\((?P<move>[^)]+)\)\s*")

# Map the MathDial teacher-move taxonomy onto a coarse turn type.
_LOW_STAKES_MOVES = {"generic"}
_DIAGNOSIS_MOVES = {"focus", "probing", "probe"}
_TELLING_MOVES = {"telling", "tell"}


def _turn_type(move: str) -> str:
    move = (move or "").strip().lower()
    if move in _TELLING_MOVES:
        return "telling"
    if move in _LOW_STAKES_MOVES:
        return "low_stakes"
    if move in _DIAGNOSIS_MOVES:
        return "diagnosis"
    return "other"


def parse_conversation(conversation: str) -> list[dict[str, str]]:
    """Split a raw ``conversation`` string into ordered role turns.

    Each turn is ``{"role": "teacher"|"student", "move": <str|"">, "text": <str>}``.
    """
    turns: list[dict[str, str]] = []
    for segment in str(conversation or "").split("|EOM|"):
        segment = segment.strip()
        if not segment:
            continue
        if segment.startswith("Teacher:"):
            role = "teacher"
            body = segment[len("Teacher:") :].strip()
        elif segment.startswith("Student:"):
            role = "student"
            body = segment[len("Student:") :].strip()
        else:
            # Continuation of the previous turn.
            if turns:
                turns[-1]["text"] = (turns[-1]["text"] + " " + segment).strip()
            continue
        move = ""
        match = _MOVE_RE.match(body)
        if match:
            move = match.group("move").strip()
            body = body[match.end() :].strip()
        turns.append({"role": role, "move": move, "text": body})
    return turns


def _format_history(turns: list[dict[str, str]]) -> str:
    lines = []
    for turn in turns:
        speaker = "Teacher" if turn["role"] == "teacher" else "Student"
        lines.append(f"{speaker}: {turn['text']}")
    return "\n".join(lines)


def build_records_from_raw(raw: dict[str, Any], *, split: str) -> Iterator[dict[str, Any]]:
    """Yield one eval record per teacher turn of a raw MathDial conversation."""
    qid = str(raw.get("qid") or raw.get("conversation_id") or raw.get("id") or "")
    question = str(raw.get("question") or raw.get("problem") or "")
    ground_truth = str(raw.get("ground_truth") or "")
    final_answer = _final_answer(ground_truth)
    profile = str(raw.get("student_profile") or "")
    initial_student = str(raw.get("student_incorrect_solution") or "")
    turns = parse_conversation(raw.get("conversation", ""))

    for index, turn in enumerate(turns):
        if turn["role"] != "teacher":
            continue
        history = turns[:index]
        prior_students = [t for t in history if t["role"] == "student"]
        student_utterance = prior_students[-1]["text"] if prior_students else initial_student
        move = turn["move"]
        yield {
            "sample_id": f"mathdial_mt:{qid}:t{index}",
            "dataset": "mathdial_multiturn",
            "split": split,
            "problem_text": question,
            "student_profile": profile,
            "student_utterance": student_utterance,
            "dialogue_history": _format_history(history),
            "metadata": {
                "conversation_id": qid,
                "turn_index": index,
                # gold (stripped from model input by build_model_input):
                "final_answer": final_answer,
                "ground_truth": ground_truth,
                "gold_teacher_move": move,
                "gold_turn_type": _turn_type(move),
                "gold_is_telling": _turn_type(move) == "telling",
                "teacher_response": turn["text"],
            },
        }


def _final_answer(ground_truth: str) -> str:
    """MathDial ground_truth ends with the final numeric answer on its own line."""
    tail = ground_truth.strip().splitlines()
    if not tail:
        return ""
    last = tail[-1].strip()
    nums = re.findall(r"-?\d[\d,]*\.?\d*", last)
    return nums[-1].replace(",", "") if nums else last


def build_multiturn_records(raw_records: Iterable[dict[str, Any]], *, split: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw in raw_records:
        records.extend(build_records_from_raw(raw, split=split))
    return records
