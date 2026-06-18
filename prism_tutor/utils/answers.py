"""Final-answer extraction and equivalence for closed-form correctness.

MathDial gold answers are stored as full worked-solution narratives whose final
line is the numeric answer (e.g. ``"... = 10 spoons.\\n 10"``). Solver agents emit
a clean answer (``"10"`` / ``"Julia bought 7 spoons."``). Matching the solver
answer against the *entire* narrative with exact string equality always fails,
which is why solver correctness was 0.0 for every method. These helpers extract
the final numeric answer from both sides and compare numerically, falling back to
normalized string equality for non-numeric answers.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _norm_str(value: Any) -> str:
    text = unicodedata.normalize("NFKC", "" if value is None else str(value)).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\s,.;:!?]+$", "", text)
    return text


def extract_final_numeric(value: Any) -> str | None:
    """Return the last numeric token in ``value`` (currency/comma stripped)."""
    if value is None:
        return None
    text = unicodedata.normalize("NFKC", str(value)).replace("$", "").replace("£", "").replace("€", "")
    matches = _NUM_RE.findall(text)
    if not matches:
        return None
    last = matches[-1].replace(",", "").rstrip(".")
    return last or None


def _to_float(token: Any) -> float | None:
    try:
        return float(str(token))
    except (TypeError, ValueError):
        return None


def numeric_equal(a: Any, b: Any, *, rel_tol: float = 1e-6) -> bool | None:
    fa, fb = _to_float(a), _to_float(b)
    if fa is None or fb is None:
        return None
    return abs(fa - fb) <= rel_tol * max(1.0, abs(fa), abs(fb))


def answers_match(candidate: Any, gold: Any) -> bool | None:
    """True/False when comparable; None when gold is missing.

    Numeric comparison is attempted first (extracting the final number from each
    side, so a full solution narrative still matches a clean numeric answer);
    otherwise normalized string equality is used.
    """
    if gold in (None, ""):
        return None
    cand_num = extract_final_numeric(candidate)
    gold_num = extract_final_numeric(gold)
    if cand_num is not None and gold_num is not None:
        eq = numeric_equal(cand_num, gold_num)
        if eq is not None:
            return eq
    if candidate in (None, ""):
        return False
    return _norm_str(candidate) == _norm_str(gold)


def canonicalize_label(predicted: Any, candidates: list[str]) -> str | None:
    """Map a free-text predicted label onto the closest candidate label.

    Used for constrained misconception classification: the agent is given the
    fixed candidate set and should copy a label; we still normalize/realign so a
    lightly reworded prediction maps to its canonical candidate. Returns the
    canonical candidate string, or None if no candidate is a confident match.
    """
    if predicted in (None, ""):
        return None
    pred = _norm_str(predicted)
    norm_candidates = [(cand, _norm_str(cand)) for cand in candidates if cand not in (None, "")]
    for cand, norm in norm_candidates:
        if pred == norm:
            return cand
    for cand, norm in norm_candidates:
        if norm and (norm in pred or pred in norm):
            return cand
    pred_tokens = set(pred.split())
    best: tuple[float, str | None] = (0.0, None)
    for cand, norm in norm_candidates:
        cand_tokens = set(norm.split())
        if not cand_tokens:
            continue
        overlap = len(pred_tokens & cand_tokens) / len(cand_tokens)
        if overlap > best[0]:
            best = (overlap, cand)
    return best[1] if best[0] >= 0.6 else None
