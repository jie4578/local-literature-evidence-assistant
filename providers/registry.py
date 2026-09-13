"""Provider 注册表与预设配置。"""

from __future__ import annotations

import os
from typing import Any

from .base import LLMProvider, ProviderConfig, ProviderError
from .ollama import OllamaProvider
from .openai_compatible import OpenAICompatibleProvider


_DEFAULTS = {
    "DeepSeek": {"base_url": "https://api.deepseek.com", "model": "deepseek-chat", "env_key": "DEEPSEEK_API_KEY"},
    "OpenAI": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini", "env_key": "OPENAI_API_KEY"},
    "Ollama": {"base_url": "http://127.0.0.1:11434/v1", "model": "", "env_key": None},
    "Custom OpenAI-Compatible": {"base_url": "", "model": "", "env_key": None},
}


def provider_names() -> list[str]:
    return list(_DEFAULTS)


def provider_defaults(name: str) -> dict[str, Any]:
    if name not in _DEFAULTS:
        raise ProviderError(f"不支持的 AI Provider：{name}")
    return dict(_DEFAULTS[name])


def create_provider(
    provider_name: str,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float = 60.0,
    max_retries: int = 2,
    input_budget_chars: int = 10500,
) -> LLMProvider:
    defaults = provider_defaults(provider_name)
    resolved_key = api_key.strip() if api_key and api_key.strip() else None
    if resolved_key is None and defaults["env_key"]:
        resolved_key = os.environ.get(defaults["env_key"])
    config = ProviderConfig(
        provider_name=provider_name,
        model=(model or defaults["model"]).strip(),
        base_url=(base_url or defaults["base_url"]).strip(),
        api_key=resolved_key,
        timeout=timeout,
        max_retries=max_retries,
        input_budget_chars=input_budget_chars,
        external_service=True,
    )
    provider: LLMProvider = OllamaProvider(config) if provider_name == "Ollama" else OpenAICompatibleProvider(config)
    provider.validate_config()
    return provider
