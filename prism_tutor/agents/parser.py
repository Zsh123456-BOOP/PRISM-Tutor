"""Structured output parsing for PRISM-Tutor agents."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .json_repair import repair_json_text


THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class ParseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_output: str
    stripped_output: str
    repaired_output: str | None = None
    parsed_output: dict[str, Any] | None = None
    parse_success: bool
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)


def strip_think_blocks(text: str) -> tuple[str, list[str]]:
    stripped, count = THINK_RE.subn("", text)
    warnings = ["think_block_removed"] if count else []
    return stripped.strip(), warnings


def _validate(data: Any, schema: type[BaseModel]) -> dict[str, Any]:
    model = schema.model_validate(data)
    return model.model_dump(mode="json")


def parse_agent_json(raw_output: str, schema: type[BaseModel]) -> ParseResult:
    stripped, warnings = strip_think_blocks(raw_output or "")
    if not stripped:
        return ParseResult(
            raw_output=raw_output or "",
            stripped_output=stripped,
            parse_success=False,
            error="empty_response",
            warnings=warnings,
        )

    try:
        return ParseResult(
            raw_output=raw_output,
            stripped_output=stripped,
            parsed_output=_validate(json.loads(stripped), schema),
            parse_success=True,
            warnings=warnings,
        )
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as first_error:
        repair = repair_json_text(stripped)
        warnings.extend(repair.warnings)
        try:
            return ParseResult(
                raw_output=raw_output,
                stripped_output=stripped,
                repaired_output=repair.text,
                parsed_output=_validate(json.loads(repair.text), schema),
                parse_success=True,
                warnings=warnings,
            )
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as second_error:
            return ParseResult(
                raw_output=raw_output,
                stripped_output=stripped,
                repaired_output=repair.text,
                parse_success=False,
                error=f"{type(first_error).__name__}: {first_error}; after repair: {type(second_error).__name__}: {second_error}",
                warnings=warnings,
            )
