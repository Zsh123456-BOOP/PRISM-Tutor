from prism_tutor.agents.parser import parse_agent_json
from prism_tutor.agents.schemas import HintOutput, SolverOutput


def test_parse_strips_think_and_markdown_fence():
    raw = """<think>hidden</think>
```json
{"hint_text": "Look at the givens.", "hint_level": 1, "answer_leakage_risk": 0.1, "confidence": 0.8}
```"""
    result = parse_agent_json(raw, HintOutput)
    assert result.parse_success is True
    assert "hidden" not in result.stripped_output
    assert "think_block_removed" in result.warnings


def test_repair_single_quotes_and_trailing_comma():
    raw = "{'answer': '42', 'reasoning': ['compute'], 'confidence': 0.8, 'uncertainty': 0.2, 'needs_more_info': False,}"
    result = parse_agent_json(raw, SolverOutput)
    assert result.parse_success is True
    assert result.parsed_output["answer"] == "42"
    assert "trailing_commas_removed" in result.warnings


def test_repair_failure_is_not_silent_empty_json():
    result = parse_agent_json("not json", HintOutput)
    assert result.parse_success is False
    assert result.parsed_output is None
    assert result.error
