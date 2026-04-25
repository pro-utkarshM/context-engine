"""Provider-agnostic model runner contracts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Protocol
from urllib import error, request

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


@dataclass(slots=True)
class OpenAIResponsesRunner:
    """Minimal OpenAI Responses API runner using plain HTTP."""

    api_key: str | None = None
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")
        env_base_url = os.environ.get("OPENAI_BASE_URL")
        if env_base_url:
            self.base_url = env_base_url

    def run(self, payload: PromptPayload, *, model_name: str) -> ModelResponse:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIResponsesRunner")

        body = {
            "model": model_name,
            "input": payload.prompt,
        }
        encoded = json.dumps(body).encode("utf-8")
        req = request.Request(
            url=f"{self.base_url.rstrip('/')}/responses",
            data=encoded,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI API request failed with status {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"OpenAI API request failed: {exc.reason}") from exc

        payload_json = json.loads(response_body)
        answer = _extract_output_text(payload_json)
        usage = payload_json.get("usage", {}) if isinstance(payload_json, dict) else {}
        prompt_tokens = int(usage.get("input_tokens", payload.estimated_prompt_tokens))
        completion_tokens = int(usage.get("output_tokens", max(len(answer.split()), 1) if answer else 0))
        return ModelResponse(
            answer=answer,
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=0,
        )


def _extract_output_text(payload_json: dict) -> str:
    output_text = payload_json.get("output_text")
    if isinstance(output_text, str) and output_text:
        return output_text.strip()

    output = payload_json.get("output")
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for content_item in content:
                if not isinstance(content_item, dict):
                    continue
                text = content_item.get("text")
                if isinstance(text, str) and text:
                    parts.append(text.strip())
        return "\n".join(part for part in parts if part).strip()

    raise RuntimeError("Could not extract output text from OpenAI response payload")
