from prism_tutor.agents.base_client import BaseLLMClient, LLMClientConfig


def test_build_payload_applies_per_agent_tokens_and_thinking():
    client = BaseLLMClient(
        LLMClientConfig(
            max_tokens=768,
            agent_max_tokens={"solver": 2048, "final_tutor": 384},
            thinking_agents=["solver"],
        )
    )
    messages = [{"role": "user", "content": "x"}]

    solver = client.build_payload(messages, agent_name="solver")
    final = client.build_payload(messages, agent_name="final_tutor")
    other = client.build_payload(messages, agent_name="pedagogy")

    assert solver["max_tokens"] == 2048
    assert solver["chat_template_kwargs"]["enable_thinking"] is True
    assert final["max_tokens"] == 384
    assert final["chat_template_kwargs"]["enable_thinking"] is False
    # agents without an override fall back to the global budget and stay non-thinking
    assert other["max_tokens"] == 768
    assert other["chat_template_kwargs"]["enable_thinking"] is False
