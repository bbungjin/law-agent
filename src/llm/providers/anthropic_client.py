"""Anthropic Claude API 클라이언트."""

from __future__ import annotations

import os

import anthropic

from ..base import BaseLLMClient, LLMResponse, ToolDefinition

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


class AnthropicClient(BaseLLMClient):
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def complete(
        self,
        messages: list[dict],
        system: str | None = None,
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in tools
            ]

        response = self._client.messages.create(**kwargs)

        text_parts = [b.text for b in response.content if hasattr(b, "text")]
        tool_calls = [
            {"name": b.name, "input": b.input}
            for b in response.content
            if b.type == "tool_use"
        ]

        return LLMResponse(
            text="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            raw=response.model_dump(),
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError(
            "Anthropic API는 임베딩을 제공하지 않습니다. "
            "EmbeddingModel(sentence-transformers)을 사용하세요."
        )
