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


def _salvage_solver_output(raw_completion: str) -> dict[str, Any] | None:
    """Recover a SolverOutput from a solver completion whose JSON was truncated by
    the thinking trace. The reasoning prose still states the answer (e.g. "...she
    bought 10 spoons"), so we extract the final answer deterministically rather
    than dropping the sample. Confidence is set low and the record is flagged
    ``solver_answer_salvaged`` so it stays auditable. This parses the model's own
    output -- it does not read gold and does not edit results."""
    from prism_tutor.utils.answers import extract_final_numeric

    answer = extract_final_numeric(raw_completion)
    if answer is None:
        return None
    tail = (raw_completion or "").strip()[-600:]
    candidate = {
        "answer": str(answer),
        "reasoning": [tail or "salvaged"],
        "confidence": 0.4,
        "uncertainty": 0.6,
        "needs_more_info": False,
    }
    try:
        return SolverOutput.model_validate(candidate).model_dump(mode="json")
    except Exception:
        return None


class LLMEndpointConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str
    model: str | None = None
    name: str | None = None
    timeout_seconds: float | None = Field(default=None, gt=0)

    @property
    def identifier(self) -> str:
        return self.name or self.model or self.base_url


class LLMClientConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    endpoints: list[str | LLMEndpointConfig] = Field(default_factory=lambda: ["mock://qwen3-8b"])
    model_name: str = "Qwen3-8B"
    temperature: float = Field(default=0.2, ge=0)
    top_p: float = Field(default=0.9, ge=0, le=1)
    top_k: int = Field(default=20, ge=0)
    max_tokens: int = Field(default=1024, ge=1)
    agent_max_tokens: dict[str, int] = Field(default_factory=dict)
    thinking_agents: list[str] = Field(default_factory=list)
    timeout_s: float = Field(default=30.0, gt=0)
    retries: int = Field(default=0, ge=0)
    mock_mode: bool = True
    mock_responses: dict[str, str | dict[str, Any]] = Field(default_factory=dict)

    @field_validator("endpoints")
    @classmethod
    def endpoints_not_empty(cls, value: list[str | LLMEndpointConfig]) -> list[str | LLMEndpointConfig]:
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
        self._endpoints = [self._normalize_endpoint(endpoint) for endpoint in self.config.endpoints]
        self._endpoint_cycle = cycle(self._endpoints)

    def _next_endpoint(self) -> LLMEndpointConfig:
        return next(self._endpoint_cycle)

    def build_payload(
        self,
        messages: list[dict[str, str]],
        *,
        model_name: str | None = None,
        agent_name: str | None = None,
    ) -> dict[str, Any]:
        max_tokens = self.config.agent_max_tokens.get(agent_name or "", self.config.max_tokens)
        # Thinking is enabled only for explicitly listed agents (the solver, so it
        # can actually reason through multi-step problems); all other agents stay
        # non-thinking to control cost and keep outputs clean.
        enable_thinking = bool(agent_name and agent_name in self.config.thinking_agents)
        return {
            "model": model_name or self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }

    @staticmethod
    def _normalize_endpoint(endpoint: str | LLMEndpointConfig) -> LLMEndpointConfig:
        if isinstance(endpoint, LLMEndpointConfig):
            return endpoint
        return LLMEndpointConfig(base_url=endpoint)

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
        endpoint_url = endpoint.base_url
        payload_model = endpoint.model or self.config.model_name
        payload = self.build_payload(messages, model_name=payload_model, agent_name=agent_name)
        started = time.perf_counter()
        raw_completion = ""
        served_model_name = payload_model
        usage = LLMUsage(source="mock")
        error: LLMError | None = None
        usage_missing = False
        request_warnings: list[str] = []

        for attempt in range(self.config.retries + 1):
            error = None
            try:
                if self.config.mock_mode or endpoint_url.startswith("mock://"):
                    raw_completion = self._mock_completion(agent_name, schema)
                    usage = self._estimate_usage(messages, raw_completion, source="mock")
                else:
                    response = self._post_chat_completion(endpoint, payload)
                    served_model_name = str(response.get("model") or payload_model)
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
                break
            except urllib.error.URLError as exc:
                error = LLMError(code="http_error", message=str(exc), retryable=True)
            except TimeoutError as exc:
                error = LLMError(code="timeout", message=str(exc), retryable=True)
            except ValueError as exc:
                error = LLMError(code="model_mismatch", message=str(exc), retryable=False)
                break
            except Exception as exc:  # pragma: no cover - defensive boundary
                error = LLMError(code="client_error", message=str(exc), retryable=False)
                break
            if error and error.retryable and attempt < self.config.retries:
                request_warnings.append(f"retry_after_{error.code}")
                continue
            break

        parse_success = False
        parsed_output: dict[str, Any] | None = None
        stripped_output = ""
        warnings: list[str] = list(request_warnings)
        if schema and raw_completion:
            parse_result = parse_agent_json(raw_completion, schema)
            parse_success = parse_result.parse_success
            parsed_output = parse_result.parsed_output
            stripped_output = parse_result.stripped_output
            warnings.extend(parse_result.warnings)
            if not parse_success and schema is SolverOutput:
                salvaged = _salvage_solver_output(raw_completion)
                if salvaged is not None:
                    parsed_output = salvaged
                    stripped_output = json.dumps(salvaged, ensure_ascii=False)
                    parse_success = True
                    warnings.append("solver_answer_salvaged")
            if not parse_success and error is None:
                error = LLMError(code="parse_error", message=parse_result.error or "parse failed")
        elif raw_completion:
            stripped_output = raw_completion

        return LLMCallRecord(
            sample_id=sample_id,
            method=method,
            agent_name=agent_name,
            endpoint=endpoint_url,
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

    def _post_chat_completion(self, endpoint: LLMEndpointConfig, payload: dict[str, Any]) -> dict[str, Any]:
        base_url = endpoint.base_url.rstrip("/")
        suffix = "/chat/completions" if base_url.endswith("/v1") else "/v1/chat/completions"
        url = base_url + suffix
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        timeout = endpoint.timeout_seconds or self.config.timeout_s
        with urllib.request.urlopen(request, timeout=timeout) as response:
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
