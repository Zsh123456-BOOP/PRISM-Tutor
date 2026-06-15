"""Shared invocation helper for JSON-only agents."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from .base_client import BaseLLMClient
from .prompts import build_agent_messages
from .types import LLMCallRecord


def run_agent(
    *,
    agent_name: str,
    schema: type[BaseModel],
    sample: dict[str, Any],
    state: dict[str, Any] | None,
    client: BaseLLMClient,
    method: str,
    retry_on_parse_error: bool = True,
) -> LLMCallRecord:
    sample_id = str(sample.get("sample_id") or sample.get("id") or "unknown")
    messages = build_agent_messages(agent_name, sample, state)
    record = client.call(
        sample_id=sample_id,
        method=method,
        agent_name=agent_name,
        messages=messages,
        schema=schema,
    )
    if record.parse_success or not retry_on_parse_error:
        return record

    retry_messages = build_agent_messages(
        agent_name,
        sample,
        state,
        error_summary=record.error.message if record.error else "parse failed",
    )
    retry_record = client.call(
        sample_id=sample_id,
        method=method,
        agent_name=agent_name,
        messages=retry_messages,
        schema=schema,
    )
    retry_record.warnings.append("retry_after_parse_error")
    return retry_record
