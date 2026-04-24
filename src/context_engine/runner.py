"""Provider-agnostic model runner contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .prompting import PromptPayload


@dataclass(frozen=True, slots=True)
class ModelResponse:
    answer: str
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int


class ModelRunner(Protocol):
    def run(self, payload: PromptPayload, *, model_name: str) -> ModelResponse:
        """Execute a model call for the assembled prompt payload."""


@dataclass(slots=True)
class StubModelRunner:
    """Deterministic placeholder runner for interface testing."""

    default_answer: str = "STUB_ANSWER"

    def run(self, payload: PromptPayload, *, model_name: str) -> ModelResponse:
        return ModelResponse(
            answer=self.default_answer,
            model_name=model_name,
            prompt_tokens=payload.estimated_prompt_tokens,
            completion_tokens=max(len(self.default_answer.split()), 1),
            latency_ms=0,
        )
