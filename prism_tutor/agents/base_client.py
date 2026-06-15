"""OpenAI-compatible base client with first-class mock mode."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from itertools import cycle
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .parser import parse_agent_json
from .schemas import (
    FinalTutorOutput,
    HintOutput,
    MisconceptionOutput,
    PedagogyOutput,
    SolverOutput,
    StateManagerOutput,
    VerifierOutput,
)
from .types import LLMCallRecord, LLMError, LLMUsage


class LLMClientConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    endpoints: list[str] = Field(default_factory=lambda: ["mock://qwen3-8b"])
    model_name: str = "Qwen3-8B"
    temperature: float = Field(default=0.2, ge=0)
    top_p: float = Field(default=0.9, ge=0, le=1)
    top_k: int = Field(default=20, ge=0)
    max_tokens: int = Field(default=1024, ge=1)
    timeout_s: float = Field(default=30.0, gt=0)
    retries: int = Field(default=0, ge=0)
    mock_mode: bool = True
    mock_responses: dict[str, str | dict[str, Any]] = Field(default_factory=dict)

    @field_validator("endpoints")
    @classmethod
    def endpoints_not_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("at least one endpoint is required")
        return value


class BaseLLMClient:
    """Shared generator client for baselines and PRISM-Tutor.

    The default configuration never calls a real model. Set ``mock_mode=False``
    and provide HTTP endpoints to use an OpenAI-compatible server.
    """

    def __init__(self, config: LLMClientConfig | None = None) -> None:
        self.config = config or LLMClientConfig()
        self._endpoint_cycle = cycle(self.config.endpoints)

    def _next_endpoint(self) -> str:
        return next(self._endpoint_cycle)

    def build_payload(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        return {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "max_tokens": self.config.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }

    def call(
        self,
        *,
        sample_id: str,
        method: str,
        agent_name: str,
        messages: list[dict[str, str]],
        schema: type[BaseModel] | None = None,
    ) -> LLMCallRecord:
        endpoint = self._next_endpoint()
        payload = self.build_payload(messages)
        started = time.perf_counter()
        raw_completion = ""
        served_model_name = self.config.model_name
        usage = LLMUsage(source="mock")
        error: LLMError | None = None
        usage_missing = False

        try:
            if self.config.mock_mode or endpoint.startswith("mock://"):
                raw_completion = self._mock_completion(agent_name, schema)
                usage = self._estimate_usage(messages, raw_completion, source="mock")
            else:
                response = self._post_chat_completion(endpoint, payload)
                served_model_name = str(response.get("model") or self.config.model_name)
                if "qwen3-8b" not in served_model_name.lower():
                    raise ValueError(f"served model is not Qwen3-8B: {served_model_name}")
                choices = response.get("choices") or []
                raw_completion = str(choices[0]["message"]["content"]) if choices else ""
                if not raw_completion:
                    error = LLMError(code="empty_response", message="model returned empty completion")
                usage_payload = response.get("usage")
                if usage_payload:
                    usage = LLMUsage(
                        prompt_tokens=int(usage_payload.get("prompt_tokens", 0)),
                        completion_tokens=int(usage_payload.get("completion_tokens", 0)),
                        total_tokens=int(usage_payload.get("total_tokens", 0)),
                        source="api",
                    )
                else:
                    usage_missing = True
                    usage = self._estimate_usage(messages, raw_completion, source="estimated")
        except urllib.error.URLError as exc:
            error = LLMError(code="http_error", message=str(exc), retryable=True)
        except TimeoutError as exc:
            error = LLMError(code="timeout", message=str(exc), retryable=True)
        except ValueError as exc:
            error = LLMError(code="model_mismatch", message=str(exc), retryable=False)
        except Exception as exc:  # pragma: no cover - defensive boundary
            error = LLMError(code="client_error", message=str(exc), retryable=False)

        parse_success = False
        parsed_output: dict[str, Any] | None = None
        stripped_output = ""
        warnings: list[str] = []
        if schema and raw_completion:
            parse_result = parse_agent_json(raw_completion, schema)
            parse_success = parse_result.parse_success
            parsed_output = parse_result.parsed_output
            stripped_output = parse_result.stripped_output
            warnings = parse_result.warnings
            if not parse_success and error is None:
                error = LLMError(code="parse_error", message=parse_result.error or "parse failed")
        elif raw_completion:
            stripped_output = raw_completion

        return LLMCallRecord(
            sample_id=sample_id,
            method=method,
            agent_name=agent_name,
            endpoint=endpoint,
            served_model_name=served_model_name,
            prompt=messages,
            request_payload=payload,
            raw_completion=raw_completion,
            stripped_output=stripped_output,
            parsed_output=parsed_output,
            usage=usage,
            usage_missing=usage_missing,
            latency_ms=(time.perf_counter() - started) * 1000,
            parse_success=parse_success,
            error=error,
            warnings=warnings,
        )

    def _post_chat_completion(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = endpoint.rstrip("/") + "/v1/chat/completions"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.config.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    def _mock_completion(self, agent_name: str, schema: type[BaseModel] | None) -> str:
        configured = self.config.mock_responses.get(agent_name)
        if configured is not None:
            return configured if isinstance(configured, str) else json.dumps(configured, ensure_ascii=False)

        mock_by_schema: dict[type[BaseModel], dict[str, Any]] = {
            SolverOutput: {
                "answer": "mock answer",
                "reasoning": ["mock reasoning"],
                "confidence": 0.7,
                "uncertainty": 0.3,
                "needs_more_info": False,
            },
            MisconceptionOutput: {
                "misconception_detected": False,
                "misconception_labels": [],
                "evidence": [],
                "severity": "low",
                "confidence": 0.6,
            },
            PedagogyOutput: {
                "strategy": "scaffold",
                "rationale": "mock rationale",
                "target_skills": [],
                "confidence": 0.6,
            },
            HintOutput: {
                "hint_text": "Try identifying the known quantities first.",
                "hint_level": 1,
                "answer_leakage_risk": 0.1,
                "confidence": 0.7,
            },
            VerifierOutput: {
                "approved": True,
                "issues": [],
                "leakage_detected": False,
                "state_conflict_detected": False,
                "confidence": 0.8,
            },
            StateManagerOutput: {"proposed_updates": [], "conflicts": [], "confidence": 0.7},
            FinalTutorOutput: {
                "response": "Let's work from what the problem gives and take the next step carefully.",
                "withheld_answer": True,
                "confidence": 0.7,
                "safety_notes": [],
            },
        }
        payload = mock_by_schema.get(schema, {"ok": True})
        return json.dumps(payload, ensure_ascii=False)

    def _estimate_usage(self, messages: list[dict[str, str]], completion: str, source: str) -> LLMUsage:
        prompt_chars = sum(len(message.get("content", "")) for message in messages)
        prompt_tokens = max(1, prompt_chars // 4)
        completion_tokens = max(1, len(completion) // 4) if completion else 0
        return LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            source=source,  # type: ignore[arg-type]
        )
