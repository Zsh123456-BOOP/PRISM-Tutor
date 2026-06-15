"""Uniform runtime node interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, Field

from prism_tutor.agents.types import LLMCallRecord

from .graph_state import StatePatch, TutorGraphState


class NodeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patch: StatePatch = Field(default_factory=StatePatch)
    calls: list[LLMCallRecord] = Field(default_factory=list)


class RuntimeNode(ABC):
    name: str

    @abstractmethod
    def invoke(self, state: TutorGraphState) -> NodeResult:
        raise NotImplementedError
