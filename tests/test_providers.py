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


@pytest.mark.parametrize("base_url", ["file:///tmp/key", "ftp://example.com/v1", "http://example.com/v1", "http://user:pass@localhost:9000/v1", "http://localhost.:9000/v1", "javascript:alert(1)"])
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
    assert FakeOpenAI.calls[0]["request"]["max_tokens"] == 2200
    assert FakeOpenAI.calls[1]["request"]["max_tokens"] == 20


@pytest.mark.parametrize("provider_name, model, base_url, api_key, expected_limit", [
    ("DeepSeek", "deepseek-chat", "https://api.deepseek.com", "deepseek-test", 1600),
    ("OpenAI", "gpt-test", "https://api.openai.com/v1", "openai-test", 1600),
    ("Ollama", "llama3", "http://127.0.0.1:11434/v1", None, 1600),
    ("Custom OpenAI-Compatible", "custom", "https://example.com/v1", "custom-test", 1600),
])
def test_all_provider_adapters_map_internal_output_limit_to_max_tokens(
    monkeypatch, provider_name, model, base_url, api_key, expected_limit
):
    FakeOpenAI.calls = []
    monkeypatch.setattr(openai_compatible, "OpenAI", FakeOpenAI)
    provider = create_provider(provider_name, model=model, base_url=base_url, api_key=api_key)
    assert provider.generate([{"role": "user", "content": "health"}], max_output_tokens=expected_limit) == "OK"
    assert FakeOpenAI.calls[-1]["request"]["max_tokens"] == expected_limit


def test_finish_reason_length_is_not_treated_as_success(monkeypatch):
    class TruncatedOpenAI(FakeOpenAI):
        def create(self, **kwargs):
            self.calls.append({"client": self.kwargs, "request": kwargs})
            message = type("Message", (), {"content": '{"partial":'})
            choice = type("Choice", (), {"message": message, "finish_reason": "length"})
            return type("Response", (), {"choices": [choice]})

    monkeypatch.setattr(openai_compatible, "OpenAI", TruncatedOpenAI)
    provider = create_provider("Ollama", model="llama3")
    with pytest.raises(ProviderError, match="output_limit_reached"):
        provider.generate([{"role": "user", "content": "health"}], max_output_tokens=20)


def test_two_sessions_keep_keys_instance_local_and_status_has_no_key(monkeypatch, tmp_path):
    FakeOpenAI.calls = []
    monkeypatch.setattr(openai_compatible, "OpenAI", FakeOpenAI)
    session_a = create_provider("DeepSeek", api_key="key_A")
    session_b = create_provider("OpenAI", api_key="key_B", model="custom-model")
    assert session_a.generate([{"role": "user", "content": "A"}]) == "OK"
    assert session_b.generate([{"role": "user", "content": "B"}]) == "OK"
    keys = [call["client"]["api_key"] for call in FakeOpenAI.calls]
    assert keys == ["key_A", "key_B"]
    assert session_a._client is not session_b._client

    from paper_pipeline import run_batch
    result = run_batch(session_a, [], tmp_path)
    status = (result.task_dir / "status.json").read_text(encoding="utf-8")
    assert '"provider": "DeepSeek"' in status and '"model": "deepseek-chat"' in status
    assert "key_A" not in status and "key_B" not in status


def test_connection_test_uses_minimal_request_and_local_mode_does_not_connect(monkeypatch):
    FakeOpenAI.calls = []
    monkeypatch.setattr(openai_compatible, "OpenAI", FakeOpenAI)
    import paper_claude
    result = paper_claude.test_provider_connection("DeepSeek", "deepseek-chat", "https://api.deepseek.com", "session-key")
    assert "连接测试成功" in result and "deepseek-chat" in result
    content = FakeOpenAI.calls[0]["request"]["messages"]
    request_text = str(content)
    assert "Controlled Validation Study" not in request_text
    assert "evidence_quote" not in request_text
    assert len(request_text) < 300
    assert "Local Offline" in paper_claude.test_provider_connection("Local Offline", "", "", "")


@pytest.mark.parametrize("remote_error, expected", [
    ("401 Unauthorized", "认证失败"),
    ("429 rate limit", "限流"),
    ("request timed out", "超时"),
    ("model does not exist", "模型不存在"),
    ("connection refused", "调用失败"),
])
def test_provider_errors_are_translated_to_chinese_without_real_requests(monkeypatch, remote_error, expected):
    class FailingOpenAI:
        def __init__(self, **kwargs):
            self.chat = self
            self.completions = self

        def create(self, **kwargs):
            raise RuntimeError(remote_error)

    monkeypatch.setattr(openai_compatible, "OpenAI", FailingOpenAI)
    provider = create_provider("Ollama", model="llama3")
    with pytest.raises(ProviderError, match=expected):
        provider.generate([{"role": "user", "content": "health"}])


def test_build_ui_does_not_initialize_any_ai_client(monkeypatch):
    def fail_if_initialized(**kwargs):
        raise AssertionError("构建 Local Offline UI 不应初始化 AI 客户端")

    monkeypatch.setattr(openai_compatible, "OpenAI", fail_if_initialized)
    import paper_claude
    paper_claude.build_ui()


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
