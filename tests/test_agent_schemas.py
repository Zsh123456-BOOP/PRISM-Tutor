import pytest
from pydantic import ValidationError

from prism_tutor.agents.schemas import (
    FinalTutorOutput,
    HintOutput,
    SolverOutput,
    VerifierOutput,
)


def test_solver_schema_requires_confidence_range():
    with pytest.raises(ValidationError):
        SolverOutput(
            answer="x",
            reasoning=["step"],
            confidence=1.2,
            uncertainty=0.1,
            needs_more_info=False,
        )


def test_hint_schema_blocks_leakage_risk_out_of_range():
    with pytest.raises(ValidationError):
        HintOutput(hint_text="hint", hint_level=1, answer_leakage_risk=-0.1, confidence=0.5)


def test_verifier_schema_accepts_issue_object():
    output = VerifierOutput(
        approved=False,
        issues=[
            {
                "issue_type": "leakage",
                "severity": "high",
                "message": "Too much answer detail.",
                "recommended_agent": "hint",
            }
        ],
        leakage_detected=True,
        state_conflict_detected=False,
        confidence=0.9,
    )
    assert output.issues[0].issue_type == "leakage"


def test_final_tutor_requires_response():
    with pytest.raises(ValidationError):
        FinalTutorOutput(response="", withheld_answer=True, confidence=0.5, safety_notes=[])
