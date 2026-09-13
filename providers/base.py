"""AI Provider 的统一接口与安全配置。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


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


class ProviderError(RuntimeError):
    """用户可见的 Provider 错误，消息中不得包含密钥。"""


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

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        raise NotImplementedError

    def generate_json(self, messages: list[dict[str, str]], schema: Any = None, **kwargs: Any) -> str:
        return self.generate(messages, **kwargs)

    def validate_config(self) -> None:
        raise NotImplementedError

    def health_check(self) -> str:
        self.validate_config()
        return self.generate(
            [{"role": "system", "content": "你是连接测试助手，只返回 OK。"},
             {"role": "user", "content": "返回 OK，不要处理任何论文文本。"}],
            temperature=0,
        )
