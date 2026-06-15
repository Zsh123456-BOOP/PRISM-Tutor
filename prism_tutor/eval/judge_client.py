"""Mock-safe judge client.

By default this module never calls an external API. Real DeepSeek-compatible
HTTP calls require both DEEPSEEK_API_KEY and PRISM_TUTOR_ENABLE_REAL_JUDGE=1.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from .judge_prompts import PROMPT_VERSION, build_judge_prompt
from .judge_schema import default_mock_score, parse_score_json
from .leakage_detector import detect_leakage


@dataclass(frozen=True)
class JudgeClientConfig:
    provider: str = "mock"
    requested_model: str = "deepseek-v4-pro"
    endpoint: str = "https://api.deepseek.com/chat/completions"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 768
    timeout_s: float = 30.0
    retries: int = 1


class MockJudgeClient:
    def __init__(self, config: JudgeClientConfig | None = None) -> None:
        self.config = config or JudgeClientConfig(provider="mock")

    def judge(self, case: dict[str, Any]) -> dict[str, Any]:
        started = time.time()
        prompt = build_judge_prompt(case)
        leakage = detect_leakage(
            case.get("candidate_response") or case.get("final_response"),
            {"answer": case.get("ground_truth")},
            sample_id=case.get("sample_id"),
        )
        score = default_mock_score(leakage=leakage["rule_leakage"])
        raw_response = json.dumps(score.to_dict(), ensure_ascii=False, sort_keys=True)
        return {
            "sample_id": case.get("sample_id"),
            "dataset": case.get("dataset"),
            "method": case.get("method"),
            "prompt": prompt,
            "raw_response": raw_response,
            "parsed_score": score.to_dict(),
            "latency": time.time() - started,
            "error": None,
            "metadata": self._metadata(actual_model="mock-judge", dry_run=True),
        }

    def _metadata(self, actual_model: str, dry_run: bool) -> dict[str, Any]:
        return {
            **asdict(self.config),
            "actual_model": actual_model,
            "api_date": datetime.now(timezone.utc).date().isoformat(),
            "prompt_version": PROMPT_VERSION,
            "dry_run": dry_run,
        }


class DeepSeekJudgeClient(MockJudgeClient):
    def __init__(self, config: JudgeClientConfig | None = None) -> None:
        super().__init__(config or JudgeClientConfig(provider="deepseek"))
        self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        enabled = os.environ.get("PRISM_TUTOR_ENABLE_REAL_JUDGE") == "1"
        if not self.api_key or not enabled:
            raise RuntimeError(
                "Real judge calls require DEEPSEEK_API_KEY and PRISM_TUTOR_ENABLE_REAL_JUDGE=1"
            )

    def judge(self, case: dict[str, Any]) -> dict[str, Any]:
        started = time.time()
        prompt = build_judge_prompt(case)
        messages = [
            {
                "role": "system",
                "content": "Return one complete JSON object only. Do not include markdown, comments, or partial JSON.",
            },
            {"role": "user", "content": prompt},
        ]
        body = {
            "model": self.config.requested_model,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
        }
        raw_response = ""
        error = None
        parsed_score = None
        actual_model = self.config.requested_model
        attempts: list[dict[str, Any]] = []
        for attempt in range(self.config.retries + 1):
            try:
                request = urllib.request.Request(
                    self.config.endpoint,
                    data=json.dumps(body).encode("utf-8"),
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=self.config.timeout_s) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                actual_model = payload.get("model") or actual_model
                choice = (payload.get("choices") or [{}])[0]
                raw_response = str(choice.get("message", {}).get("content") or "")
                parsed_score = parse_score_json(raw_response).to_dict()
                attempts.append(
                    {
                        "attempt": attempt,
                        "raw_response": raw_response,
                        "error": None,
                        "finish_reason": choice.get("finish_reason"),
                        "usage": payload.get("usage"),
                    }
                )
                error = None
                break
            except Exception as exc:  # pragma: no cover - network path is opt-in.
                error = f"attempt={attempt}: {type(exc).__name__}: {exc}"
                attempts.append(
                    {
                        "attempt": attempt,
                        "raw_response": raw_response,
                        "error": error,
                    }
                )
                body["messages"] = messages + [
                    {"role": "assistant", "content": raw_response or "{}"},
                    {
                        "role": "user",
                        "content": (
                            "The previous response was invalid or incomplete. "
                            "Return exactly one complete JSON object with all required score fields."
                        ),
                    },
                ]
        return {
            "sample_id": case.get("sample_id"),
            "dataset": case.get("dataset"),
            "method": case.get("method"),
            "prompt": prompt,
            "raw_response": raw_response,
            "raw_attempts": attempts,
            "parsed_score": parsed_score,
            "latency": time.time() - started,
            "error": error,
            "metadata": self._metadata(actual_model=actual_model, dry_run=False),
        }


def make_judge_client(config: JudgeClientConfig | None = None) -> MockJudgeClient:
    config = config or JudgeClientConfig()
    real_enabled = config.provider == "deepseek" or os.environ.get("PRISM_TUTOR_ENABLE_REAL_JUDGE") == "1"
    if real_enabled:
        return DeepSeekJudgeClient(config)
    return MockJudgeClient(config)
