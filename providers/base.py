"""AI Provider 的统一接口与安全配置。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OutputTokenBudgets:
    """各类模型请求的集中式输出上限。"""

    chunk_max_output_tokens: int = 1800
    reduce_max_output_tokens: int = 2200
    final_max_output_tokens: int = 2200
    health_check_max_output_tokens: int = 20


DEFAULT_OUTPUT_TOKEN_BUDGETS = OutputTokenBudgets()


@dataclass(frozen=True)
class ProviderConfig:
    provider_name: str
    model: str
    base_url: str
    api_key: str | None = None
    timeout: float = 60.0
    max_retries: int = 2
    input_budget_chars: int = 10500
    external_service: bool = True
    output_budgets: OutputTokenBudgets = DEFAULT_OUTPUT_TOKEN_BUDGETS


@dataclass(frozen=True)
class ProviderResponse:
    """Provider 统一响应；敏感请求头和密钥永远不进入此结构。"""

    content: str
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    response_chars: int | None = None
    provider_request_id: str | None = None
    model: str | None = None

    def __post_init__(self) -> None:
        if self.response_chars is None:
            object.__setattr__(self, "response_chars", len(self.content or ""))

    @classmethod
    def from_legacy_string(cls, content: str) -> "ProviderResponse":
        return cls(content=content, response_chars=len(content or ""))

    def __str__(self) -> str:
        return self.content

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.content == other
        if isinstance(other, ProviderResponse):
            return (
                self.content, self.finish_reason, self.input_tokens,
                self.output_tokens, self.total_tokens, self.response_chars,
                self.provider_request_id, self.model,
            ) == (
                other.content, other.finish_reason, other.input_tokens,
                other.output_tokens, other.total_tokens, other.response_chars,
                other.provider_request_id, other.model,
            )
        return NotImplemented


class ProviderError(RuntimeError):
    """用户可见的 Provider 错误，消息中不得包含密钥。"""

    def __init__(self, message: str, code: str | None = None, response: ProviderResponse | None = None):
        super().__init__(message)
        self.code = code
        self.response = response


def sanitize_error(error: Any, secret: str | None = None) -> str:
    message = str(error)
    if secret:
        message = message.replace(secret, "[已脱敏]")
    message = re.sub(r"(?i)(bearer\s+)[^\s,;]+", r"\1[已脱敏]", message)
    message = re.sub(r"(?i)(api[_ -]?key[=: ]+)[^\s,;]+", r"\1[已脱敏]", message)
    return message


class LLMProvider:
    """所有模型服务商必须实现的最小接口。"""

    config: ProviderConfig

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> ProviderResponse:
        raise NotImplementedError

    def generate_json(self, messages: list[dict[str, str]], schema: Any = None, **kwargs: Any) -> ProviderResponse:
        return self.generate(messages, **kwargs)

    def validate_config(self) -> None:
        raise NotImplementedError

    def health_check(self, max_output_tokens: int | None = None) -> ProviderResponse:
        self.validate_config()
        output_limit = (
            self.config.output_budgets.health_check_max_output_tokens
            if max_output_tokens is None else max_output_tokens
        )
        return self.generate(
            [{"role": "system", "content": "你是连接测试助手，只返回 OK。"},
             {"role": "user", "content": "返回 OK，不要处理任何论文文本。"}],
            temperature=0,
            max_output_tokens=output_limit,
        )
