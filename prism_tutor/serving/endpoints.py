from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Iterable


THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", flags=re.IGNORECASE | re.DOTALL)


def strip_think_blocks(text: str) -> str:
    """Remove Qwen thinking blocks while leaving token usage accounting untouched."""

    return THINK_BLOCK_RE.sub("", text).strip()


@dataclass(frozen=True)
class Endpoint:
    base_url: str
    model: str
    name: str | None = None
    timeout_seconds: float = 120.0

    @property
    def identifier(self) -> str:
        return self.name or self.model or self.base_url

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "base_url": self.base_url,
            "model": self.model,
            "timeout_seconds": self.timeout_seconds,
        }


class EndpointRegistry:
    """Deterministic endpoint selection with health-aware round-robin behavior."""

    def __init__(self, endpoints: Iterable[Endpoint]):
        self._endpoints = list(endpoints)
        if not self._endpoints:
            raise ValueError("At least one endpoint is required")
        self._unhealthy: set[str] = set()
        self._cursor = 0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "EndpointRegistry":
        model_config = config.get("model", config)
        timeout = float(config.get("generation", {}).get("timeout_seconds", 120))
        endpoints = [
            Endpoint(
                base_url=str(item["base_url"]).rstrip("/"),
                model=str(item["model"]),
                name=item.get("name") or item.get("model"),
                timeout_seconds=float(item.get("timeout_seconds", timeout)),
            )
            for item in model_config.get("endpoints", [])
        ]
        return cls(endpoints)

    @property
    def endpoints(self) -> list[Endpoint]:
        return list(self._endpoints)

    @property
    def healthy_endpoints(self) -> list[Endpoint]:
        healthy = [endpoint for endpoint in self._endpoints if endpoint.identifier not in self._unhealthy]
        if not healthy:
            raise RuntimeError("No healthy endpoints remain")
        return healthy

    def mark_unhealthy(self, endpoint: Endpoint | str, reason: str | None = None) -> dict[str, Any]:
        identifier = endpoint if isinstance(endpoint, str) else endpoint.identifier
        self._unhealthy.add(identifier)
        return {"endpoint": identifier, "reason": reason or "unhealthy"}

    def mark_healthy(self, endpoint: Endpoint | str) -> None:
        identifier = endpoint if isinstance(endpoint, str) else endpoint.identifier
        self._unhealthy.discard(identifier)

    def select_next(self) -> Endpoint:
        healthy = self.healthy_endpoints
        endpoint = healthy[self._cursor % len(healthy)]
        self._cursor += 1
        return endpoint

    def select_for_sample(self, sample_id: str, *, method: str | None = None) -> Endpoint:
        """Map a sample/method key to a stable slot so resume does not reshuffle endpoints."""

        healthy = self.healthy_endpoints
        key = f"{method or ''}:{sample_id}".encode("utf-8")
        index = int(hashlib.sha256(key).hexdigest()[:12], 16) % len(healthy)
        return healthy[index]

    def select_by_index(self, sample_index: int) -> Endpoint:
        healthy = self.healthy_endpoints
        return healthy[sample_index % len(healthy)]
