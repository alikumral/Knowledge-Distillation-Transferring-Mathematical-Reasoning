"""Unit tests for answer extraction & matching -- the project's #1 silent-bug risk.

Run locally (no GPU/torch needed):  pytest -q
"""

import pytest

from mathdistill import answers as A
from mathdistill.prompts import FINAL_ANSWER_TAG


# --------------------------------------------------------------------------- #
# normalize_number
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("72", "72"),
        (" 72 ", "72"),
        ("1,000", "1000"),
        ("$5", "5"),
        ("5%", "5"),
        ("$1,234.50", "1234.50"),
        ("72.", "72"),
        ("-4", "-4"),
    ],
)
def test_normalize_number(raw, expected):
    assert A.normalize_number(raw) == expected


# --------------------------------------------------------------------------- #
# numbers_equal
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "a, b, eq",
    [
        ("72", "72", True),
        ("72", "72.0", True),
        ("3.0", "3", True),
        ("1,000", "1000", True),
        ("$5", "5", True),
        ("1/2", "0.5", True),
        ("72", "73", False),
        ("0.5", "0.50001", False),
    ],
)
def test_numbers_equal(a, b, eq):
    assert A.numbers_equal(a, b) is eq


# --------------------------------------------------------------------------- #
# GSM8K gold extraction
# --------------------------------------------------------------------------- #
def test_extract_gold_gsm8k():
    ans = (
        "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
        "Natalia sold 48+24 = <<48+24=72>>72 clips altogether.\n#### 72"
    )
    assert A.extract_gold_gsm8k(ans) == "72"


def test_extract_gold_gsm8k_with_commas():
    assert A.extract_gold_gsm8k("blah blah\n#### 1,000") == "1000"


# --------------------------------------------------------------------------- #
# Prediction extraction (priority order)
# --------------------------------------------------------------------------- #
def test_pred_from_marker():
    text = f"Step 1...\nStep 2...\n{FINAL_ANSWER_TAG} 72"
    assert A.extract_pred_number(text) == "72"


def test_pred_marker_beats_earlier_numbers():
    # earlier numbers in the reasoning must NOT win over the marked answer
    text = f"He had 5 apples and bought 10 more.\n{FINAL_ANSWER_TAG} 15"
    assert A.extract_pred_number(text) == "15"


def test_pred_marker_with_bold_and_period():
    text = f"reasoning...\n{FINAL_ANSWER_TAG} **72**."
    assert A.extract_pred_number(text) == "72"


def test_pred_from_boxed():
    text = r"so the result is \boxed{42} which is final"
    assert A.extract_pred_number(text) == "42"


def test_pred_from_hash():
    text = "some reasoning\n#### 18"
    assert A.extract_pred_number(text) == "18"


def test_pred_fallback_last_number():
    text = "I think the total comes to 5 then 9 then finally 13"
    assert A.extract_pred_number(text) == "13"


def test_pred_handles_currency_and_commas():
    text = f"{FINAL_ANSWER_TAG} $1,250"
    assert A.extract_pred_number(text) == "1250"


def test_pred_none_when_no_number():
    assert A.extract_pred_number("no numbers here at all") is None


# --------------------------------------------------------------------------- #
# End-to-end correctness (used by rejection sampling + eval)
# --------------------------------------------------------------------------- #
def test_is_correct_gsm8k_true():
    pred = f"work...\n{FINAL_ANSWER_TAG} 72"
    assert A.is_correct_gsm8k(pred, "long solution\n#### 72") is True


def test_is_correct_gsm8k_false():
    pred = f"work...\n{FINAL_ANSWER_TAG} 71"
    assert A.is_correct_gsm8k(pred, "long solution\n#### 72") is False


def test_is_correct_gsm8k_accepts_extracted_gold():
    pred = f"work...\n{FINAL_ANSWER_TAG} 1000"
    assert A.is_correct_gsm8k(pred, "1,000") is True


# --------------------------------------------------------------------------- #
# MATH (boxed) extraction + approximate matching
# --------------------------------------------------------------------------- #
def test_extract_boxed_simple():
    assert A.extract_boxed(r"answer is \boxed{42}") == "42"


def test_extract_boxed_nested():
    assert A.extract_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"


def test_extract_boxed_last_wins():
    assert A.extract_boxed(r"\boxed{1} then \boxed{2}") == "2"


def test_extract_gold_math():
    sol = r"We compute ... so the answer is $\boxed{\dfrac{1}{2}}$."
    assert A.extract_gold_math(sol) == r"\frac{1}{2}"


def test_is_correct_math_latex_equiv():
    pred = r"... therefore \boxed{\dfrac{1}{2}}"
    gold = r"\boxed{\frac{1}{2}}"
    assert A.is_correct_math(pred, gold) is True


def test_is_correct_math_numeric():
    pred = f"{FINAL_ANSWER_TAG} 0.5"
    assert A.is_correct_math(pred, r"\boxed{1/2}") is True


def test_is_correct_math_false():
    pred = r"\boxed{\frac{1}{3}}"
    assert A.is_correct_math(pred, r"\boxed{\frac{1}{2}}") is False
