"""Unit tests for the pure data-transformation helpers (no `datasets` needed)."""

import pytest

from mathdistill import data as D
from mathdistill.prompts import FINAL_ANSWER_TAG


GSM8K_ANSWER = (
    "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
    "Natalia sold 48+24 = <<48+24=72>>72 clips altogether.\n#### 72"
)


def test_strip_calculator_annotations():
    assert D.strip_calculator_annotations("48/2 = <<48/2=24>>24 clips") == "48/2 = 24 clips"


def test_gold_reasoning_has_no_annotations_or_hash():
    r = D.gold_reasoning(GSM8K_ANSWER)
    assert "<<" not in r and ">>" not in r
    assert "####" not in r
    assert "24 clips" in r


def test_build_sft_from_gold():
    rows = [{"question": "How many clips?", "answer": GSM8K_ANSWER}]
    recs = D.build_sft_from_gold(rows)
    assert len(recs) == 1
    rec = recs[0]
    assert rec["source"] == "gold"
    roles = [m["role"] for m in rec["messages"]]
    assert roles == ["system", "user", "assistant"]
    assistant = rec["messages"][-1]["content"]
    assert assistant.endswith(f"{FINAL_ANSWER_TAG} 72")


def test_build_sft_from_teacher_respects_cap():
    accepted = [
        {"problem_id": 0, "question": "Q0", "gold": "72",
         "completion": f"reasoning A\n{FINAL_ANSWER_TAG} 72"},
        {"problem_id": 0, "question": "Q0", "gold": "72",
         "completion": f"reasoning B\n{FINAL_ANSWER_TAG} 72"},
        {"problem_id": 0, "question": "Q0", "gold": "72",
         "completion": f"reasoning C\n{FINAL_ANSWER_TAG} 72"},
        {"problem_id": 1, "question": "Q1", "gold": "5",
         "completion": f"reasoning D\n{FINAL_ANSWER_TAG} 5"},
    ]
    one = D.build_sft_from_teacher(accepted, traces_per_problem=1)
    assert len(one) == 2  # 1 per problem, 2 problems

    three = D.build_sft_from_teacher(accepted, traces_per_problem=3)
    assert len(three) == 4  # 3 for problem 0 + 1 for problem 1


def test_teacher_target_standardized_ending():
    accepted = [{"problem_id": 0, "question": "Q0", "gold": "72",
                 "completion": f"some teacher reasoning here\n{FINAL_ANSWER_TAG} 72"}]
    rec = D.build_sft_from_teacher(accepted, 1)[0]
    assistant = rec["messages"][-1]["content"]
    # reasoning kept, ending standardized via build_target
    assert "some teacher reasoning here" in assistant
    assert assistant.endswith(f"{FINAL_ANSWER_TAG} 72")


def test_assert_no_leakage_raises():
    rows = [{"question": "shared question", "answer": GSM8K_ANSWER}]
    recs = D.build_sft_from_gold(rows)
    with pytest.raises(AssertionError):
        D.assert_no_leakage(recs, ["shared question"])


def test_assert_no_leakage_passes():
    rows = [{"question": "train question", "answer": GSM8K_ANSWER}]
    recs = D.build_sft_from_gold(rows)
    D.assert_no_leakage(recs, ["a totally different test question"])  # no raise
