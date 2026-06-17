import pytest
from pydantic import ValidationError

from prism_tutor.agents.schemas import (
    AGENT_SCHEMAS,
    FinalTutorOutput,
    HintOutput,
    MisconceptionOutput,
    PedagogyOutput,
    RiskEstimatorOutput,
    SolverOutput,
    StateManagerOutput,
    StudentStateUpdate,
    VerifierIssue,
    VerifierOutput,
)


VALID_PAYLOADS = {
    "solver": {
        "answer": "x = 2",
        "reasoning": ["Isolate x."],
        "confidence": 0.8,
        "uncertainty": 0.2,
        "needs_more_info": False,
    },
    "misconception": {
        "misconception_detected": True,
        "misconception_labels": ["sign_error"],
        "evidence": ["Student flipped the inequality incorrectly."],
        "severity": "medium",
        "confidence": 0.7,
    },
    "pedagogy": {
        "strategy": "scaffold",
        "rationale": "Ask for the next algebraic operation.",
        "target_skills": ["linear_equations"],
        "confidence": 0.75,
    },
    "hint": {
        "hint_text": "Move the constant term first.",
        "hint_level": 1,
        "answer_leakage_risk": 0.1,
        "confidence": 0.8,
    },
    "verifier": {
        "approved": False,
        "issues": [
            {
                "issue_type": "leakage",
                "severity": "high",
                "message": "Too much answer detail.",
                "recommended_agent": "hint",
            }
        ],
        "leakage_detected": True,
        "state_conflict_detected": False,
        "confidence": 0.9,
    },
    "state_manager": {
        "proposed_updates": [
            {
                "field": "weak_skills",
                "operation": "add",
                "value": "linear_equations",
                "confidence": 0.8,
                "evidence": "Student missed isolation step.",
            }
        ],
        "conflicts": [],
        "confidence": 0.8,
    },
    "final_tutor": {
        "response": "Try isolating x before computing the value.",
        "withheld_answer": True,
        "confidence": 0.8,
        "safety_notes": [],
    },
    "risk_estimator": {
        "answer_uncertainty": 0.2,
        "misconception_risk": 0.4,
        "pedagogy_risk": 0.3,
        "leakage_risk": 0.1,
        "state_conflict_risk": 0.2,
        "estimated_difficulty": 0.5,
        "total_risk": 0.35,
        "risk_bucket": "medium",
        "recommended_mode": "guided",
    },
}

REQUIRED_FIELD_BY_SCHEMA = {
    "solver": "answer",
    "misconception": "misconception_detected",
    "pedagogy": "strategy",
    "hint": "hint_text",
    "verifier": "approved",
    "state_manager": "proposed_updates",
    "final_tutor": "response",
    "risk_estimator": "total_risk",
}

CONFIDENCE_FIELD_BY_SCHEMA = {
    "solver": "confidence",
    "misconception": "confidence",
    "pedagogy": "confidence",
    "hint": "confidence",
    "verifier": "confidence",
    "state_manager": "confidence",
    "final_tutor": "confidence",
    "risk_estimator": "total_risk",
}


@pytest.mark.parametrize("agent_name,schema", AGENT_SCHEMAS.items())
def test_all_agent_schemas_accept_valid_payloads(agent_name, schema):
    output = schema.model_validate(VALID_PAYLOADS[agent_name])
    assert output.schema_version


@pytest.mark.parametrize("agent_name,schema", AGENT_SCHEMAS.items())
def test_all_agent_schemas_reject_missing_required_fields(agent_name, schema):
    payload = dict(VALID_PAYLOADS[agent_name])
    payload.pop(REQUIRED_FIELD_BY_SCHEMA[agent_name])

    with pytest.raises(ValidationError):
        schema.model_validate(payload)


@pytest.mark.parametrize("agent_name,schema", AGENT_SCHEMAS.items())
def test_all_agent_schemas_reject_out_of_range_confidence(agent_name, schema):
    payload = dict(VALID_PAYLOADS[agent_name])
    payload[CONFIDENCE_FIELD_BY_SCHEMA[agent_name]] = 1.1

    with pytest.raises(ValidationError):
        schema.model_validate(payload)


@pytest.mark.parametrize("agent_name,schema", AGENT_SCHEMAS.items())
def test_all_agent_schemas_reject_extra_fields(agent_name, schema):
    payload = dict(VALID_PAYLOADS[agent_name])
    payload["unexpected"] = "not allowed"

    with pytest.raises(ValidationError):
        schema.model_validate(payload)


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


def test_verifier_issue_normalizes_common_recommended_agent_aliases():
    corrective = VerifierIssue.model_validate(
        {
            "issue_type": "pedagogy",
            "severity": "medium",
            "message": "Needs a corrective teaching move.",
            "recommended_agent": "corrector",
        }
    )
    formatter = VerifierIssue.model_validate(
        {
            "issue_type": "format",
            "severity": "low",
            "message": "Needs formatting cleanup.",
            "recommended_agent": "formatter",
        }
    )
    verifier = VerifierIssue.model_validate(
        {
            "issue_type": "other",
            "severity": "low",
            "message": "Needs another verification pass.",
            "recommended_agent": "verifier",
        }
    )

    assert corrective.recommended_agent == "pedagogy"
    assert formatter.recommended_agent == "final_tutor"
    assert verifier.recommended_agent is None


def test_final_tutor_requires_response():
    with pytest.raises(ValidationError):
        FinalTutorOutput(response="", withheld_answer=True, confidence=0.5, safety_notes=[])


@pytest.mark.parametrize(
    "schema,payload_update",
    [
        (MisconceptionOutput, {"severity": "critical"}),
        (PedagogyOutput, {"strategy": "lecture"}),
        (
            VerifierIssue,
            {
                "issue_type": "privacy",
                "severity": "high",
                "message": "Invalid issue type.",
                "recommended_agent": "hint",
            },
        ),
        (
            StudentStateUpdate,
            {
                "field": "grade",
                "operation": "add",
                "value": "A",
                "confidence": 0.7,
                "evidence": "Invalid state field.",
            },
        ),
        (RiskEstimatorOutput, {"risk_bucket": "critical"}),
        (RiskEstimatorOutput, {"recommended_mode": "oracle"}),
        (StateManagerOutput, {"confidence": -0.1}),
    ],
)
def test_schema_literal_and_nested_boundaries(schema, payload_update):
    if schema is MisconceptionOutput:
        payload = dict(VALID_PAYLOADS["misconception"])
    elif schema is PedagogyOutput:
        payload = dict(VALID_PAYLOADS["pedagogy"])
    elif schema is VerifierIssue:
        payload = payload_update
        payload_update = {}
    elif schema is StudentStateUpdate:
        payload = payload_update
        payload_update = {}
    elif schema is RiskEstimatorOutput:
        payload = dict(VALID_PAYLOADS["risk_estimator"])
    else:
        payload = dict(VALID_PAYLOADS["state_manager"])
    payload.update(payload_update)

    with pytest.raises(ValidationError):
        schema.model_validate(payload)
