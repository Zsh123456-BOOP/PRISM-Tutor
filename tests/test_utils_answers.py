from prism_tutor.utils.answers import (
    answers_match,
    canonicalize_label,
    extract_final_numeric,
)


def test_extract_final_numeric_from_solution_narrative():
    gt = "Julia's package contained 15-5=10 spoons.\n 10"
    assert extract_final_numeric(gt) == "10"
    assert extract_final_numeric("$500.\n 500") == "500"
    assert extract_final_numeric("no numbers here") is None


def test_answers_match_clean_answer_against_full_narrative():
    # The MathDial bug: solver emits a clean answer, gold is a full narrative.
    gold = "Karan has to pay $4015 / 5 = $803.\n 803"
    assert answers_match("803", gold) is True
    assert answers_match("Julia bought 7 spoons.", "... = 7 spoons.\n 7") is True
    assert answers_match("-99999", gold) is False


def test_answers_match_missing_gold_returns_none():
    assert answers_match("42", "") is None
    assert answers_match("42", None) is None


def test_canonicalize_label_maps_to_candidate_space():
    candidates = [
        "students confuse numerator and denominator",
        "students misunderstand proportional relationships",
    ]
    assert canonicalize_label("students confuse numerator and denominator", candidates) == candidates[0]
    # lightly reworded prediction still maps onto its canonical candidate
    assert canonicalize_label("the student confuses numerator and denominator here", candidates) == candidates[0]
    # an unrelated free-text label does not get force-mapped
    assert canonicalize_label("sign error when subtracting", candidates) is None
