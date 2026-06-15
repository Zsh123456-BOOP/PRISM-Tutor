"""Pedagogy planning agent wrapper."""

from __future__ import annotations

from typing import Any

from .base_client import BaseLLMClient
from .runner import run_agent
from .schemas import PedagogyOutput
from .types import LLMCallRecord


class PedagogyAgent:
    name = "pedagogy"
    schema = PedagogyOutput

    def invoke(self, sample: dict[str, Any], state: dict[str, Any], client: BaseLLMClient, method: str) -> LLMCallRecord:
        return run_agent(agent_name=self.name, schema=self.schema, sample=sample, state=state, client=client, method=method)
