"""Shared agent runtime types."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


SCHEMA_VERSION = "0.1.0"


class AgentName(StrEnum):
    SOLVER = "solver"
    MISCONCEPTION = "misconception"
    PEDAGOGY = "pedagogy"
    HINT = "hint"
    VERIFIER = "verifier"
    STATE_MANAGER = "state_manager"
    FINAL_TUTOR = "final_tutor"
    RISK_ESTIMATOR = "risk_estimator"


ErrorCode = Literal[
    "timeout",
    "http_error",
    "empty_response",
    "usage_missing",
    "parse_error",
    "schema_validation_error",
    "model_mismatch",
    "client_error",
]


class LLMUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    source: Literal["api", "estimated", "mock"] = "estimated"


class LLMError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: ErrorCode
    message: str
    retryable: bool = False


class LLMCallRecord(BaseModel):
    """Complete auditable record for one model call or mock call."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    method: str
    agent_name: str
    endpoint: str
    served_model_name: str | None = None
    prompt: list[dict[str, str]]
    request_payload: dict[str, Any] = Field(default_factory=dict)
    raw_completion: str = ""
    stripped_output: str = ""
    parsed_output: dict[str, Any] | None = None
    usage: LLMUsage = Field(default_factory=LLMUsage)
    usage_missing: bool = False
    latency_ms: float = Field(default=0.0, ge=0)
    parse_success: bool = False
    error: LLMError | None = None
    warnings: list[str] = Field(default_factory=list)
    schema_version: str = SCHEMA_VERSION
