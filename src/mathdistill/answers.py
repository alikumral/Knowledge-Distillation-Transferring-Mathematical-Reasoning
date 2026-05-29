"""Answer extraction and equivalence checking.

This is the single most bug-prone part of the project: a flaky extractor
silently corrupts both rejection sampling (stage 2) and final accuracy
(stage 4). It is therefore pure, dependency-free, and heavily unit-tested
(tests/test_answers.py).

Two domains:
  * GSM8K  -> integer/decimal final answer (after '####' in gold).
  * MATH   -> a LaTeX expression inside \\boxed{...}; matched approximately.

Extraction priority for model predictions (most reliable first):
  1. text after our FINAL_ANSWER_TAG marker (what we instruct the model to emit)
  2. the last \\boxed{...}
  3. text after '####'
  4. fallback: the last number in the text
"""

from __future__ import annotations

import re

from .prompts import FINAL_ANSWER_TAG

# A number: optional sign, optional $, digits with optional thousands commas,
# optional decimal part. Matches "72", "-4", "$1,000", "3.50".
_NUMBER_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")

# Locate our marker (case-insensitive). Consumes an optional trailing "is" so
# that "final answer is: X" leaves only "X" -- otherwise the "is" leaks into
# string-based (MATH) extraction.
_FINAL_RE = re.compile(r"(?:final answer(?:\s+is)?|the answer is)\s*:?\s*", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #
def extract_boxed(text: str) -> str | None:
    r"""Return the content of the LAST ``\boxed{...}`` in ``text`` (brace-matched)."""
    if not text:
        return None
    idx = text.rfind(r"\boxed")
    if idx == -1:
        return None
    i = idx + len(r"\boxed")
    while i < len(text) and text[i] == " ":
        i += 1
    if i >= len(text) or text[i] != "{":
        return None
    depth = 0
    start = i
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : j]
    return None  # unbalanced braces


def normalize_number(s: str | None) -> str | None:
    """Strip currency/commas/percent/whitespace and a trailing period."""
    if s is None:
        return None
    s = s.strip().replace(",", "").replace("$", "").replace("%", "").replace(" ", "")
    s = s.rstrip(".")
    return s


def _to_float(s: str | None) -> float | None:
    """Best-effort float parse, including simple 'a/b' fractions."""
    if s is None:
        return None
    try:
        return float(s)
    except ValueError:
        if "/" in s:
            parts = s.split("/")
            if len(parts) == 2:
                try:
                    return float(parts[0]) / float(parts[1])
                except (ValueError, ZeroDivisionError):
                    return None
        return None


def numbers_equal(a: str | None, b: str | None, tol: float = 1e-6) -> bool:
    """Numeric equality with tolerance; falls back to normalized string match."""
    fa, fb = _to_float(normalize_number(a)), _to_float(normalize_number(b))
    if fa is not None and fb is not None:
        return abs(fa - fb) <= tol
    return normalize_number(a) == normalize_number(b)


# --------------------------------------------------------------------------- #
# GSM8K
# --------------------------------------------------------------------------- #
def extract_gold_gsm8k(answer_text: str) -> str | None:
    """Extract the gold final answer from a GSM8K 'answer' field ('... #### 72')."""
    if answer_text is None:
        return None
    if "####" in answer_text:
        tail = answer_text.split("####")[-1]
        m = _NUMBER_RE.search(tail)
        if m:
            return normalize_number(m.group())
    nums = _NUMBER_RE.findall(answer_text)
    return normalize_number(nums[-1]) if nums else None


def extract_pred_number(text: str) -> str | None:
    """Extract a numeric prediction from model output (priority order above)."""
    if not text:
        return None
    # 1. after our marker
    m = _FINAL_RE.search(text)
    if m:
        tail = text[m.end() :]
        boxed = extract_boxed(tail)
        if boxed:
            nums = _NUMBER_RE.findall(boxed)
            if nums:
                return normalize_number(nums[-1])
        nums = _NUMBER_RE.findall(tail)
        if nums:
            return normalize_number(nums[0])
    # 2. last \boxed{}
    boxed = extract_boxed(text)
    if boxed:
        nums = _NUMBER_RE.findall(boxed)
        if nums:
            return normalize_number(nums[-1])
    # 3. after ####
    if "####" in text:
        tail = text.split("####")[-1]
        nums = _NUMBER_RE.findall(tail)
        if nums:
            return normalize_number(nums[0])
    # 4. fallback: last number anywhere
    nums = _NUMBER_RE.findall(text)
    return normalize_number(nums[-1]) if nums else None


def split_reasoning(text: str) -> str:
    """Return the reasoning portion (everything before the final-answer line).

    Used to rebuild teacher traces through prompts.build_target so that every
    training target -- gold or teacher -- ends with the identical answer line.
    """
    if not text:
        return ""
    m = _FINAL_RE.search(text)
    if m:
        return text[: m.start()].rstrip()
    if "####" in text:
        return text.split("####")[0].rstrip()
    return text.rstrip()


def is_correct_gsm8k(pred_text: str, gold_answer: str) -> bool:
    """True iff the model's extracted number matches the gold answer.

    ``gold_answer`` may be a raw GSM8K answer field (with '####') or an
    already-extracted value.
    """
    gold = extract_gold_gsm8k(gold_answer) if "####" in str(gold_answer) else gold_answer
    pred = extract_pred_number(pred_text)
    if pred is None or gold is None:
        return False
    return numbers_equal(pred, gold)


# --------------------------------------------------------------------------- #
# MATH (approximate -- documented limitation in the report)
# --------------------------------------------------------------------------- #
def normalize_latex(s: str | None) -> str | None:
    r"""Normalize a LaTeX answer for approximate string comparison.

    Handles the common cosmetic differences (\\left/\\right, spacing macros,
    \\dfrac->\\frac, \\text{}, $, surrounding braces). This is intentionally
    approximate: a fully correct MATH grader needs a CAS; we accept some
    false negatives and disclose this in the report (plan.md §6).
    """
    if s is None:
        return None
    inner = extract_boxed(s)
    if inner is not None:
        s = inner
    s = s.strip()
    s = s.replace(r"\left", "").replace(r"\right", "")
    for macro in (r"\!", r"\,", r"\;", r"\:", r"\ "):
        s = s.replace(macro, "")
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.replace("$", "").replace(" ", "")
    s = s.rstrip(".")
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    return s


def extract_gold_math(solution_text: str) -> str | None:
    r"""Extract the boxed gold answer from a MATH 'solution' field."""
    boxed = extract_boxed(solution_text)
    return normalize_latex(boxed) if boxed is not None else None


def extract_pred_math(text: str) -> str | None:
    """Extract a MATH prediction: marker -> boxed -> last number."""
    if not text:
        return None
    m = _FINAL_RE.search(text)
    if m:
        tail = text[m.end() :]
        boxed = extract_boxed(tail)
        if boxed is not None:
            return normalize_latex(boxed)
        line = tail.splitlines()[0] if tail.splitlines() else tail
        if line.strip():
            return normalize_latex(line.strip())
    boxed = extract_boxed(text)
    if boxed is not None:
        return normalize_latex(boxed)
    nums = _NUMBER_RE.findall(text)
    return normalize_number(nums[-1]) if nums else None


def is_correct_math(pred_text: str, gold_answer: str) -> bool:
    """Approximate MATH correctness: numeric match if possible, else LaTeX string."""
    gold = extract_gold_math(gold_answer) if r"\boxed" in str(gold_answer) else normalize_latex(gold_answer)
    pred = extract_pred_math(pred_text)
    if pred is None or gold is None:
        return False
    if numbers_equal(pred, gold):
        return True
    return normalize_latex(pred) == normalize_latex(gold)
