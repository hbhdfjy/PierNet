from unittest.mock import Mock

import pytest

from PierNet.core.llm_client import LLMClient


@pytest.mark.parametrize(
    ("thinking", "expected"),
    [("enabled", True), ("disabled", False)],
)
def test_siliconflow_uses_boolean_thinking_switch(monkeypatch, thinking, expected):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "choices": [{"message": {"content": "ok"}}]
    }
    post = Mock(return_value=response)
    monkeypatch.setattr("PierNet.core.llm_client.requests.post", post)

    client = LLMClient(
        provider="siliconflow",
        model="deepseek-ai/DeepSeek-V3.2",
        api_key="test-key",
        thinking=thinking,
    )

    assert client.generate("hello", max_tokens=32) == "ok"
    payload = post.call_args.kwargs["json"]
    assert payload["enable_thinking"] is expected
    if expected:
        assert "thinking_budget" not in payload
    else:
        assert payload["thinking_budget"] == 128
    assert "thinking" not in payload


def test_siliconflow_omits_thinking_switch_when_unset(monkeypatch):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "choices": [{"message": {"content": "ok"}}]
    }
    post = Mock(return_value=response)
    monkeypatch.setattr("PierNet.core.llm_client.requests.post", post)

    client = LLMClient(
        provider="siliconflow",
        model="deepseek-ai/DeepSeek-V3.2",
        api_key="test-key",
    )

    assert client.generate("hello", max_tokens=32) == "ok"
    assert "enable_thinking" not in post.call_args.kwargs["json"]
