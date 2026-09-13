"""OpenAI-compatible Provider：DeepSeek、OpenAI 与自定义端点共用。"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from openai import OpenAI

from .base import LLMProvider, ProviderConfig, ProviderError, sanitize_error


class OpenAICompatibleProvider(LLMProvider):
    def __init__(self, config: ProviderConfig):
        self.config = config
        self._client: Any | None = None

    def validate_config(self) -> None:
        parsed = urlparse(self.config.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ProviderError("Base URL 无效：必须使用 HTTP(S) 地址")
        if parsed.username is not None or parsed.password is not None:
            raise ProviderError("Base URL 不允许包含用户名或密码")
        if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ProviderError("安全限制：远程 HTTP 地址被拒绝，仅允许 localhost 或 127.0.0.1")
        if not self.config.model.strip():
            raise ProviderError("模型名称不能为空")
        if self.config.provider_name not in {"Ollama"} and not (self.config.api_key or "").strip():
            raise ProviderError(f"缺少 {self.config.provider_name} API Key，请配置后再试")

    @property
    def client(self) -> Any:
        self.validate_config()
        if self._client is None:
            try:
                self._client = OpenAI(
                    api_key=self.config.api_key or "ollama",
                    base_url=self.config.base_url,
                    timeout=self.config.timeout,
                    max_retries=0,
                )
            except Exception as exc:
                raise ProviderError(f"无法初始化 {self.config.provider_name} 客户端：{sanitize_error(exc, self.config.api_key)}") from exc
        return self._client

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        try:
            max_output_tokens = kwargs.get(
                "max_output_tokens",
                self.config.output_budgets.reduce_max_output_tokens,
            )
            if not isinstance(max_output_tokens, int) or max_output_tokens <= 0:
                raise ProviderError("输出 Token 上限必须是正整数，不能静默忽略")
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=kwargs.get("temperature", 0.2),
                timeout=kwargs.get("timeout", self.config.timeout),
                max_tokens=max_output_tokens,
            )
            choice = response.choices[0]
            if getattr(choice, "finish_reason", None) == "length":
                raise ProviderError("output_limit_reached：模型输出达到 Token 上限，响应可能不完整")
            return choice.message.content or ""
        except ProviderError:
            raise
        except Exception as exc:
            text = sanitize_error(exc, self.config.api_key)
            lowered = text.casefold()
            if "401" in lowered or "unauthorized" in lowered:
                raise ProviderError(f"{self.config.provider_name} 认证失败（401），请检查 API Key") from exc
            if "429" in lowered or "rate limit" in lowered:
                raise ProviderError(f"{self.config.provider_name} 请求被限流或余额不足（429）") from exc
            if "timeout" in lowered or "timed out" in lowered:
                raise ProviderError(f"{self.config.provider_name} 连接超时") from exc
            if "model" in lowered and ("not found" in lowered or "does not exist" in lowered):
                raise ProviderError(f"模型不存在或不可用：{self.config.model}") from exc
            raise ProviderError(f"{self.config.provider_name} 调用失败：{text}") from exc
