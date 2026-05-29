"""Dataset loading and SFT-set construction (plan.md §6 Stage 0 & Stage 2).

Design: I/O (HF `datasets`) is kept separate from the pure transformation
helpers so the formatting logic is unit-testable without downloading anything.
`datasets` is lazy-imported only inside the loader functions.

All SFT records share the schema:
    {"problem_id": int, "source": str, "messages": [system, user, assistant]}
The assistant turn always ends with the standardized final-answer line
(prompts.build_target), so gold and teacher conditions differ ONLY in reasoning.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

from .answers import extract_gold_gsm8k, split_reasoning
from .prompts import build_messages, build_target

_CALC_ANNOTATION_RE = re.compile(r"<<[^>]*>>")


# --------------------------------------------------------------------------- #
# Pure transformation helpers (unit-tested)
# --------------------------------------------------------------------------- #
def strip_calculator_annotations(text: str) -> str:
    """Remove GSM8K's '<<48/2=24>>' calculator annotations."""
    return _CALC_ANNOTATION_RE.sub("", text)


def gold_reasoning(answer_field: str) -> str:
    """Reasoning body of a GSM8K gold solution: drop calc annotations and '#### x'."""
    body = answer_field.split("####")[0]
    return strip_calculator_annotations(body).strip()


def make_sft_record(problem_id: int, question: str, reasoning: str,
                    final_answer: str, source: str) -> dict:
    """Build one chat-formatted SFT record (system + user + assistant target)."""
    messages = build_messages(question)
    messages.append({"role": "assistant", "content": build_target(reasoning, final_answer)})
    return {"problem_id": problem_id, "source": source, "messages": messages}


def build_sft_from_gold(rows: Iterable[dict]) -> list[dict]:
    """Condition (ii): one record per problem from GSM8K's own gold CoT.

    `rows` items: {"question": str, "answer": str (with '#### x')}.
    """
    out: list[dict] = []
    for i, row in enumerate(rows):
        gold = extract_gold_gsm8k(row["answer"])
        if gold is None:
            continue
        reasoning = gold_reasoning(row["answer"])
        out.append(make_sft_record(i, row["question"], reasoning, gold, source="gold"))
    return out


def build_sft_from_teacher(accepted: Iterable[dict], traces_per_problem: int) -> list[dict]:
    """Conditions (iii)/(iv): up to `traces_per_problem` accepted teacher traces/problem.

    `accepted` items: {"problem_id": int, "question": str, "completion": str,
                       "gold": str}. The teacher's answer already equals gold
    (rejection sampling), so the target's answer line uses `gold` and only the
    reasoning body varies.
    """
    by_problem: dict[int, list[dict]] = defaultdict(list)
    for rec in accepted:
        by_problem[rec["problem_id"]].append(rec)

    out: list[dict] = []
    for pid in sorted(by_problem):
        for rec in by_problem[pid][:traces_per_problem]:
            reasoning = split_reasoning(rec["completion"])
            out.append(
                make_sft_record(pid, rec["question"], reasoning, rec["gold"], source="teacher")
            )
    return out


def assert_no_leakage(sft_records: Iterable[dict], test_questions: Iterable[str]) -> None:
    """Guard: no training question may appear in the GSM8K test split."""
    test_set = {q.strip() for q in test_questions}
    for rec in sft_records:
        user_turn = next(m["content"] for m in rec["messages"] if m["role"] == "user")
        if user_turn.strip() in test_set:
            raise AssertionError(
                f"Train/test leakage: problem_id={rec.get('problem_id')} is in the test split"
            )


# --------------------------------------------------------------------------- #
# Loaders (lazy-import `datasets`)
# --------------------------------------------------------------------------- #
def load_gsm8k(cfg: dict):
    """Return the GSM8K DatasetDict (train / test)."""
    from datasets import load_dataset

    return load_dataset(cfg["gsm8k"]["hf_id"], cfg["gsm8k"]["config"])


def load_math_ood(cfg: dict):
    """Return a frozen, level-stratified MATH-200 subset for OOD evaluation."""
    import random

    from datasets import Dataset, load_dataset

    mcfg = cfg["math"]
    ds = load_dataset(mcfg["hf_id"], split=mcfg["split"])

    buckets: dict[str, list[int]] = defaultdict(list)
    for idx, level in enumerate(ds[mcfg["stratify_by"]]):
        buckets[str(level)].append(idx)

    rng = random.Random(mcfg["ood_seed"])
    n_total = mcfg["ood_size"]
    per_bucket = max(1, n_total // max(1, len(buckets)))

    chosen: list[int] = []
    for level in sorted(buckets):
        idxs = buckets[level]
        rng.shuffle(idxs)
        chosen.extend(idxs[:per_bucket])
    rng.shuffle(chosen)
    chosen = chosen[:n_total]

    return ds.select(sorted(chosen))


def sft_to_hf_dataset(records: list[dict]):
    """Convert SFT records to a HF Dataset with a 'messages' column for SFTTrainer."""
    from datasets import Dataset

    return Dataset.from_list([{"messages": r["messages"]} for r in records])
