"""Token accounting with deterministic fallback estimates."""

from __future__ import annotations

import re
from typing import Any

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def estimate_tokens(text: Any) -> int:
    if text is None:
        return 0
    return len(TOKEN_RE.findall(str(text)))


def usage_total_tokens(usage: dict[str, Any] | None) -> tuple[int, str]:
    """Return total tokens and source.

    Logs may contain OpenAI-style `total_tokens`, split prompt/completion counts,
    or no usage at all. Missing usage is handled by callers using fallback text.
    """

    if not isinstance(usage, dict):
        return 0, "missing"
    total = usage.get("total_tokens")
    if isinstance(total, (int, float)):
        return int(total), "usage.total_tokens"
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if isinstance(prompt, (int, float)) or isinstance(completion, (int, float)):
        return int(prompt or 0) + int(completion or 0), "usage.prompt_completion"
    return 0, "missing"


def record_token_count(record: dict[str, Any]) -> dict[str, Any]:
    usage = record.get("usage")
    if not isinstance(usage, dict):
        usage = record.get("token_usage")
    total, source = usage_total_tokens(usage)
    if total:
        return {"total_tokens": total, "token_source": source}

    text_parts: list[str] = []
    for key in ("prompt", "raw_completion", "final_response"):
        if record.get(key):
            text_parts.append(str(record[key]))
    for message in record.get("messages") or []:
        if isinstance(message, dict):
            text_parts.append(str(message.get("content", "")))
        else:
            text_parts.append(str(message))
    return {"total_tokens": estimate_tokens("\n".join(text_parts)), "token_source": "fallback.regex"}


def count_agent_calls(record: dict[str, Any]) -> int:
    calls = record.get("agent_calls")
    if isinstance(calls, (int, float)):
        return int(calls)
    if isinstance(calls, list):
        return len(calls)
    messages = record.get("messages")
    if isinstance(messages, list):
        return sum(1 for item in messages if isinstance(item, dict) and item.get("role") == "assistant")
    selected = record.get("selected_agents")
    if isinstance(selected, list):
        return len(selected)
    return 1 if record.get("final_response") else 0
