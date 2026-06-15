import json

from prism_tutor.agents.base_client import BaseLLMClient, LLMClientConfig, LLMEndpointConfig
from prism_tutor.agents.schemas import SolverOutput


def test_mock_client_returns_call_record_with_payload_and_usage():
    client = BaseLLMClient(LLMClientConfig(endpoints=["mock://a"], mock_mode=True))
    record = client.call(
        sample_id="s1",
        method="B0",
        agent_name="solver",
        messages=[{"role": "user", "content": "solve"}],
        schema=SolverOutput,
    )
    assert record.parse_success is True
    assert record.parsed_output["answer"] == "mock answer"
    assert record.usage.total_tokens > 0
    assert record.request_payload["chat_template_kwargs"]["enable_thinking"] is False


def test_client_round_robin_endpoint_selection():
    client = BaseLLMClient(LLMClientConfig(endpoints=["mock://a", "mock://b"], mock_mode=True))
    first = client.call(
        sample_id="s1",
        method="B0",
        agent_name="solver",
        messages=[{"role": "user", "content": "a"}],
        schema=SolverOutput,
    )
    second = client.call(
        sample_id="s2",
        method="B0",
        agent_name="solver",
        messages=[{"role": "user", "content": "b"}],
        schema=SolverOutput,
    )
    assert first.endpoint == "mock://a"
    assert second.endpoint == "mock://b"


def test_endpoint_specific_model_is_used_in_payload():
    client = BaseLLMClient(
        LLMClientConfig(
            endpoints=[LLMEndpointConfig(base_url="mock://gpu0", model="qwen3-8b-gpu0")],
            model_name="Qwen/Qwen3-8B",
            mock_mode=True,
        )
    )
    record = client.call(
        sample_id="s1",
        method="B0",
        agent_name="solver",
        messages=[{"role": "user", "content": "solve"}],
        schema=SolverOutput,
    )
    assert record.request_payload["model"] == "qwen3-8b-gpu0"
    assert record.endpoint == "mock://gpu0"


def test_v1_endpoint_url_is_not_double_prefixed(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps({"model": "qwen3-8b-gpu0", "choices": []}).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = BaseLLMClient(LLMClientConfig(mock_mode=False))
    client._post_chat_completion(
        LLMEndpointConfig(base_url="http://localhost:8000/v1", model="qwen3-8b-gpu0", timeout_seconds=7),
        {"model": "qwen3-8b-gpu0", "messages": []},
    )
    assert captured["url"] == "http://localhost:8000/v1/chat/completions"
    assert captured["timeout"] == 7
