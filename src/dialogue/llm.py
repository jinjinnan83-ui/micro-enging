"""Minimal DeepSeek Chat Completions client with validated JSON responses."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from src.config import Settings


class LLMError(RuntimeError):
    """Raised when the configured language model cannot produce usable JSON."""


@dataclass(frozen=True)
class JSONCompletion:
    content: dict[str, Any]
    model: str
    usage: dict[str, Any] = field(default_factory=dict)


class JSONLLM(Protocol):
    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> JSONCompletion: ...


class DeepSeekChatClient:
    """Call DeepSeek's OpenAI-compatible `/chat/completions` endpoint."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> JSONCompletion:
        if not self.settings.llm_api_key:
            raise LLMError("LLM_API_KEY is not configured")

        payload: dict[str, Any] = {
            "model": self.settings.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": max_tokens,
            "thinking": {"type": self.settings.llm_thinking_type},
        }
        headers = {
            "Authorization": f"Bearer {self.settings.llm_api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.settings.llm_base_url.rstrip('/')}/chat/completions"
        last_error: Exception | None = None

        for _attempt in range(self.settings.llm_max_retries + 1):
            try:
                with httpx.Client(timeout=self.settings.llm_timeout_seconds) as client:
                    response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                body = response.json()
                raw = body["choices"][0]["message"].get("content") or ""
                if not raw.strip():
                    raise LLMError("DeepSeek returned empty JSON content")
                content = _parse_json_object(raw)
                return JSONCompletion(
                    content=content,
                    model=str(body.get("model") or self.settings.llm_model),
                    usage=dict(body.get("usage") or {}),
                )
            except (httpx.HTTPError, KeyError, TypeError, ValueError, LLMError) as exc:
                last_error = exc

        raise LLMError(f"DeepSeek request failed: {last_error}") from last_error


def _parse_json_object(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1]).strip()
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object")
    return parsed
