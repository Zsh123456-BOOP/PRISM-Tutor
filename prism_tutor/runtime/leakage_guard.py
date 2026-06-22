"""Runtime-only leakage guard (detection).

This guard intentionally does **not** inspect gold answers. It may use the
*model's own* reference solution (the solver agent's computed answer, which is a
model output, not a gold label) to detect when the student-facing response
reveals that answer. Detection becomes stricter when the estimated leakage risk
is high (risk-gated). Evaluation uses the independent detector in
``prism_tutor.eval.leakage_detector`` (which may use gold).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from prism_tutor.utils.answers import extract_final_numeric


@dataclass(frozen=True)
class GuardHit:
    sample_id: str | None
    rule: str
    evidence: str
    start: int
    end: int
    severity: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DIRECT_ANSWER_PATTERNS = [
    re.compile(r"\bthe\s+(?:final\s+)?answer\s+is\b", re.I),
    re.compile(r"\b答案是\b"),
    re.compile(r"\bso\s+the\s+answer\s+is\b", re.I),
]

SOLUTION_CHAIN_PATTERNS = [
    re.compile(r"\bstep\s*1\b.*\bstep\s*2\b.*\bstep\s*3\b", re.I | re.S),
    re.compile(r"\bcomplete\s+solution\b", re.I),
    re.compile(r"\bfull\s+solution\b", re.I),
]

KEY_STEP_PATTERNS = [
    re.compile(r"\bfirst,?\s+compute\b", re.I),
    re.compile(r"\bthen\s+solve\s+for\b", re.I),
]

STUDENT_REFERENCE_PREFIXES = [
    re.compile(r"\bwhy\s+(?:do\s+)?you\s+(?:think|believe|say)\s*$", re.I),
    re.compile(r"\byour\s*$", re.I),
]

# The tutor is echoing the student's own number (not revealing the solution), so a
# matching number nearby is not leakage.
STUDENT_ECHO_PATTERNS = [
    re.compile(
        r"\byou\s+(?:said|got|have|had|wrote|chose|selected|calculated|computed|answered|think|believe|claimed|mentioned)\b",
        re.I,
    ),
    re.compile(r"\byour\s+(?:answer|result|solution|calculation|response|estimate|guess|number)\b", re.I),
]


def _is_student_echo(context: str) -> bool:
    return any(pattern.search(context) for pattern in STUDENT_ECHO_PATTERNS)

# A computed result is being asserted (used together with the reference answer,
# or — in aggressive / high-leakage-risk mode — on its own).
RESULT_ASSERTION_PATTERNS = [
    re.compile(r"=\s*-?\d", re.I),
    re.compile(r"\b(?:equals?|is|are|gives?|得到|结果(?:是|为)?|答案(?:是|为)?)\s*-?\d", re.I),
    re.compile(r"\b(?:therefore|thus|hence|so)\b[^.?!]{0,40}-?\d", re.I | re.S),
]

_ALL_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _span_hit(sample_id: str | None, rule: str, match: re.Match[str], severity: str) -> GuardHit:
    start, end = match.span()
    return GuardHit(sample_id, rule, match.string[start:end], start, end, severity)


def _is_student_reference(text: str, match: re.Match[str]) -> bool:
    prefix = text[max(0, match.start() - 80) : match.start()]
    return any(pattern.search(prefix) for pattern in STUDENT_REFERENCE_PREFIXES)


def detect_runtime_leakage(
    response: Any,
    *,
    sample_id: str | None = None,
    reference_answer: Any = None,
    leakage_risk: float | None = None,
    aggressive_threshold: float = 0.55,
    problem_text: str = "",
) -> dict[str, Any]:
    """Detect leakage in a student-facing response (gold-free).

    ``reference_answer`` is the solver agent's OWN computed answer (a model
    output, never gold). When the estimated ``leakage_risk`` is at or above
    ``aggressive_threshold`` the detector also flags bare presence of that answer
    and any asserted numeric result.
    """
    text = "" if response is None else str(response)
    hits: list[GuardHit] = []

    for pattern in DIRECT_ANSWER_PATTERNS:
        match = pattern.search(text)
        if match and not _is_student_reference(text, match):
            hits.append(_span_hit(sample_id, "runtime_direct_answer_phrase", match, "medium"))
    for pattern in SOLUTION_CHAIN_PATTERNS:
        match = pattern.search(text)
        if match:
            hits.append(_span_hit(sample_id, "runtime_complete_solution_chain", match, "high"))
    for pattern in KEY_STEP_PATTERNS:
        match = pattern.search(text)
        if match:
            hits.append(_span_hit(sample_id, "runtime_key_step_disclosure", match, "medium"))

    aggressive = leakage_risk is not None and float(leakage_risk) >= aggressive_threshold

    # Reference-answer leakage: the response reveals the solver's own computed
    # final answer. High severity when asserted ("= 42", "is 42", "answer is 42");
    # medium when merely stated (bare presence), excluding echoes of the student's
    # own number. Bare presence fires regardless of leakage risk so the guard's
    # sensitivity matches the offline detector (which flags bare final-answer
    # presence); the leakage risk still gates the no-reference branch below.
    ref_num = extract_final_numeric(reference_answer) if reference_answer not in (None, "") else None
    if ref_num is not None and problem_text:
        # If the solver's answer also appears in the problem, a matching number in
        # the response can't be told apart from quoting the problem -> skip the
        # number-based reference rules for this sample.
        prob_nums = {m.group().replace(",", "").rstrip(".") for m in _ALL_NUM_RE.finditer(str(problem_text))}
        if ref_num in prob_nums:
            ref_num = None
    if ref_num is not None:
        asserted = False
        bare = False
        for match in _ALL_NUM_RE.finditer(text):
            token = match.group().replace(",", "").rstrip(".")
            if token != ref_num:
                continue
            context = text[max(0, match.start() - 40) : match.start()]
            if _is_student_echo(context):
                continue
            if "=" in context or any(p.search(context + match.group()) for p in RESULT_ASSERTION_PATTERNS):
                hits.append(
                    GuardHit(
                        sample_id,
                        "runtime_reference_answer_revealed",
                        match.group(),
                        match.start(),
                        match.end(),
                        "high",
                    )
                )
                asserted = True
                break
            bare = True
        if not asserted and bare:
            hits.append(GuardHit(sample_id, "runtime_reference_answer_present", str(ref_num), 0, 0, "medium"))

    # No reference answer available (e.g. solver not routed): a generic asserted
    # numeric result is only treated as leakage under high leakage risk.
    if aggressive and not hits:
        for pattern in RESULT_ASSERTION_PATTERNS:
            match = pattern.search(text)
            if match:
                hits.append(_span_hit(sample_id, "runtime_result_assertion_high_risk", match, "medium"))
                break

    return {
        "rule_leakage": bool(hits),
        "matched_rules": [hit.rule for hit in hits],
        "hits": [hit.to_dict() for hit in hits],
        "leakage_risk": leakage_risk,
        "aggressive": aggressive,
    }
