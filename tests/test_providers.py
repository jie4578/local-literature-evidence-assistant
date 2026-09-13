import pytest

from providers import ProviderError, create_provider
from providers import openai_compatible


class FakeOpenAI:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.calls.append({"client": self.kwargs, "request": kwargs})
        message = type("Message", (), {"content": "OK"})
        choice = type("Choice", (), {"message": message})
        return type("Response", (), {"choices": [choice]})


def test_deepseek_and_openai_config_use_expected_environment(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-secret")
    deepseek = create_provider("DeepSeek")
    assert deepseek.config.base_url == "https://api.deepseek.com"
    assert deepseek.config.model == "deepseek-chat"
    assert deepseek.config.api_key == "deepseek-secret"

    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    openai = create_provider("OpenAI", model="gpt-test")
    assert openai.config.base_url == "https://api.openai.com/v1"
    assert openai.config.model == "gpt-test"
    assert openai.config.api_key == "openai-secret"


def test_ollama_without_key_and_custom_endpoint(monkeypatch):
    ollama = create_provider("Ollama", model="llama3")
    assert ollama.config.api_key is None
    assert ollama.config.base_url == "http://127.0.0.1:11434/v1"
    custom = create_provider("Custom OpenAI-Compatible", model="local", base_url="http://localhost:9000/v1", api_key="session-key")
    assert custom.config.model == "local"
    assert custom.config.external_service is True


@pytest.mark.parametrize("base_url", ["file:///tmp/key", "ftp://example.com/v1", "http://example.com/v1"])
def test_custom_endpoint_rejects_unsafe_urls(base_url):
    with pytest.raises(ProviderError):
        create_provider("Custom OpenAI-Compatible", model="test", base_url=base_url, api_key="secret")


def test_localhost_http_is_allowed_and_all_providers_share_generate(monkeypatch):
    FakeOpenAI.calls = []
    monkeypatch.setattr(openai_compatible, "OpenAI", FakeOpenAI)
    provider = create_provider("Custom OpenAI-Compatible", model="test", base_url="http://127.0.0.1:9000/v1", api_key="secret")
    assert provider.generate([{"role": "user", "content": "health only"}]) == "OK"
    assert provider.health_check() == "OK"
    assert len(FakeOpenAI.calls) == 2


def test_missing_key_is_clear_and_key_is_redacted(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="缺少 DeepSeek API Key"):
        create_provider("DeepSeek")
    class FailingOpenAI:
        def __init__(self, **kwargs):
            self.chat = self
            self.completions = self

        def create(self, **kwargs):
            raise RuntimeError("Authorization: Bearer secret-token")

    monkeypatch.setattr(openai_compatible, "OpenAI", FailingOpenAI)
    provider = create_provider("Custom OpenAI-Compatible", model="test", base_url="https://example.com/v1", api_key="secret-token")
    with pytest.raises(ProviderError) as error:
        provider.generate([{"role": "user", "content": "x"}])
    assert "secret-token" not in str(error.value)
