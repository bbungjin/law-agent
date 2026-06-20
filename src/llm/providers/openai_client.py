"""OpenAI API 클라이언트."""

from __future__ import annotations

import json
import os

import openai

from ..base import BaseLLMClient, LLMResponse, ToolDefinition

DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIClient(BaseLLMClient):
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        self.model = model
        self._client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def complete(
        self,
        messages: list[dict],
        system: str | None = None,
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": all_messages,
        }
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in tools
            ]

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        text = choice.message.content

        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append({
                    "name": tc.function.name,
                    "input": json.loads(tc.function.arguments),
                })

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            raw=response.model_dump(),
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("OpenAI 임베딩은 별도 모델을 사용하세요.")
