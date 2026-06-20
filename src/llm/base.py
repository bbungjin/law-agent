"""LLM provider 추상화.

CLAUDE.md 규칙: 모든 LLM 호출은 이 인터페이스를 통해서만 이루어져야 하며,
provider(OpenAI / Anthropic / Ollama)를 교체해도 상위 코드는 변경되지 않아야 한다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolDefinition:
    """에이전트가 사용할 수 있는 도구(tool)의 스키마."""

    name: str
    description: str
    input_schema: dict = field(default_factory=dict)


@dataclass
class LLMResponse:
    text: str | None = None
    tool_calls: list[dict] = field(default_factory=list)
    raw: dict | None = None


class BaseLLMClient(ABC):
    """모든 LLM provider 구현체가 따라야 하는 인터페이스."""

    @abstractmethod
    def complete(
        self,
        messages: list[dict],
        system: str | None = None,
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """단일 턴 completion을 수행한다.

        Args:
            messages: [{"role": "user"|"assistant", "content": str}, ...]
            system: system prompt
            tools: tool calling을 사용할 경우 도구 정의 목록
            max_tokens: 최대 생성 토큰 수
        """
        raise NotImplementedError

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """텍스트 목록을 임베딩 벡터로 변환한다 (임베딩 모델이 별도라면 위임).

        주의: API LLM(OpenAI/Anthropic)은 자체 임베딩을 제공하지 않을 수 있으므로,
        실제 구현체에서는 sentence-transformers 등 별도 임베딩 모델을 호출하도록
        구성하는 것을 권장한다. 이 메서드는 일관된 인터페이스 제공 목적.
        """
        raise NotImplementedError


def get_llm_client(provider: str | None = None) -> BaseLLMClient:
    """환경변수 LLM_PROVIDER 또는 인자에 따라 적절한 클라이언트를 반환하는 팩토리."""
    import os
    p = (provider or os.environ.get("LLM_PROVIDER", "anthropic")).lower()
    if p == "anthropic":
        from .providers.anthropic_client import AnthropicClient
        return AnthropicClient()
    if p == "openai":
        from .providers.openai_client import OpenAIClient
        return OpenAIClient()
    raise ValueError(f"지원하지 않는 LLM provider: {p}. 사용 가능: anthropic, openai")
