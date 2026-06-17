"""Deterministic JSON cleanup for common LLM formatting failures."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any


FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)
TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


@dataclass(frozen=True)
class RepairResult:
    text: str
    repaired: bool
    warnings: list[str] = field(default_factory=list)


def strip_markdown_fence(text: str) -> tuple[str, list[str]]:
    match = FENCE_RE.match(text.strip())
    if not match:
        return text.strip(), []
    return match.group(1).strip(), ["markdown_fence_removed"]


def extract_first_json(text: str) -> tuple[str, list[str]]:
    """Return the first balanced JSON object/array candidate."""

    warnings: list[str] = []
    start = min((i for i in [text.find("{"), text.find("[")] if i != -1), default=-1)
    if start == -1:
        return text.strip(), warnings
    if start > 0:
        warnings.append("prefix_removed")

    stack: list[str] = []
    in_string = False
    escape = False
    quote = ""
    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                in_string = False
            continue
        if char in {"\"", "'"}:
            in_string = True
            quote = char
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if not stack or stack[-1] != char:
                break
            stack.pop()
            if not stack:
                end = index + 1
                suffix = text[end:].strip()
                if suffix:
                    warnings.append("suffix_removed")
                    if "{" in suffix or "[" in suffix:
                        warnings.append("multiple_json_candidates")
                return text[start:end].strip(), warnings
    return text[start:].strip(), warnings


def _canonicalize_with_json_or_literal(text: str) -> str:
    try:
        return json.dumps(json.loads(text), ensure_ascii=False)
    except json.JSONDecodeError:
        pass
    parsed: Any = ast.literal_eval(text)
    if not isinstance(parsed, (dict, list)):
        raise ValueError("candidate is not a JSON object or array")
    return json.dumps(parsed, ensure_ascii=False)


def escape_control_chars_in_strings(text: str) -> tuple[str, list[str]]:
    output: list[str] = []
    warnings: list[str] = []
    in_string = False
    escape = False
    quote = ""
    changed = False
    for char in text:
        if in_string:
            if escape:
                output.append(char)
                escape = False
                continue
            if char == "\\":
                output.append(char)
                escape = True
                continue
            if char == quote:
                output.append(char)
                in_string = False
                continue
            if char == "\n":
                output.append("\\n")
                changed = True
                continue
            if char == "\r":
                output.append("\\r")
                changed = True
                continue
            if char == "\t":
                output.append("\\t")
                changed = True
                continue
            if ord(char) < 0x20:
                output.append("\\u%04x" % ord(char))
                changed = True
                continue
            output.append(char)
            continue
        output.append(char)
        if char in {"\"", "'"}:
            in_string = True
            quote = char
    if changed:
        warnings.append("control_chars_escaped")
    return "".join(output), warnings


def repair_json_text(text: str) -> RepairResult:
    warnings: list[str] = []
    candidate, fence_warnings = strip_markdown_fence(text)
    warnings.extend(fence_warnings)
    candidate, extract_warnings = extract_first_json(candidate)
    warnings.extend(extract_warnings)

    normalized = (
        candidate.replace("\u201c", "\"")
        .replace("\u201d", "\"")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )
    no_trailing_commas = TRAILING_COMMA_RE.sub(r"\1", normalized)
    if no_trailing_commas != candidate:
        warnings.append("trailing_commas_removed")
    escaped_control_chars, control_warnings = escape_control_chars_in_strings(no_trailing_commas)
    warnings.extend(control_warnings)

    try:
        canonical = _canonicalize_with_json_or_literal(escaped_control_chars)
    except Exception:
        return RepairResult(text=escaped_control_chars, repaired=bool(warnings), warnings=warnings)

    return RepairResult(text=canonical, repaired=canonical != text.strip(), warnings=warnings)
