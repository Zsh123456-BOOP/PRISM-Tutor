from prism_tutor.agents.base_client import BaseLLMClient, LLMClientConfig, _salvage_solver_output


def test_salvage_solver_output_recovers_answer_from_truncated_reasoning():
    # Thinking trace truncated before the JSON -- the answer is still stated.
    raw = "12 + 3 = 15. Since her husband gave her 5, she must have bought 10 spoons"
    out = _salvage_solver_output(raw)
    assert out is not None
    assert out["answer"] == "10"
    assert out["reasoning"]
    assert out["confidence"] <= 0.5  # salvaged answers are flagged low-confidence
    # nothing numeric to recover -> no salvage
    assert _salvage_solver_output("no numeric answer here") is None


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
