"""Runtime error codes and helpers."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


GraphErrorCode = Literal[
    "agent_failure",
    "max_rounds",
    "token_budget",
    "checkpoint_failed",
    "state_conflict",
    "schema_error",
]


class GraphErrorRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: GraphErrorCode
    message: str
    agent_name: str | None = None
    round_index: int | None = None
    recoverable: bool = True
