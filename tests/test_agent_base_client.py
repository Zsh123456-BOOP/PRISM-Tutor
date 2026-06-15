from prism_tutor.agents.base_client import BaseLLMClient, LLMClientConfig
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
