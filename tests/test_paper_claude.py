import pytest


def test_process_papers_reports_missing_api_key(monkeypatch):
    import paper_claude

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    result, output_path = paper_claude.process_papers([], "")

    assert "请先填写 DeepSeek API Key" in result
    assert output_path is None


def test_process_papers_does_not_call_api_when_key_is_missing(monkeypatch):
    import paper_claude

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    def fail_if_called(*args, **kwargs):
        pytest.fail("缺失 API Key 时不应调用 DeepSeek API")

    monkeypatch.setattr(paper_claude, "get_client", fail_if_called)
    result, output_path = paper_claude.process_papers([], "")

    assert result.startswith("❌")
    assert output_path is None
