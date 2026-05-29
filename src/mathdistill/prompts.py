"""Single source of truth for prompts and answer formatting.

Every stage (teacher generation, gold-CoT formatting, student training, and
evaluation) imports from here, so the ONLY thing that varies across the five
comparison conditions is the *content* of the reasoning -- never the prompt
wording or the final-answer format. This is what makes the H2 comparison
(teacher CoT vs gold CoT) fair: identical scaffolding, different reasoning.

The model is instructed to end with a fixed marker line so that answer
extraction (see answers.py) is reliable rather than guess-the-last-number.
"""

from __future__ import annotations

# Fixed marker the model is asked to end on. Keep in sync with answers.py.
FINAL_ANSWER_TAG = "The final answer is:"

SYSTEM_PROMPT = (
    "You are a careful math tutor. Solve the problem step by step, showing your "
    "reasoning clearly and concisely. Then, on a new line, give the final answer "
    f"in exactly this form:\n{FINAL_ANSWER_TAG} <answer>"
)


def build_messages(question: str, system: bool = True) -> list[dict]:
    """Chat messages for a math question (apply the tokenizer chat template to these).

    Used identically for teacher generation and student inference/eval.
    """
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": question.strip()})
    return messages


def build_target(reasoning: str, final_answer: str) -> str:
    """Assistant-side training target: reasoning + standardized final-answer line.

    Both the gold-CoT and teacher-CoT conditions are rebuilt through this helper
    so their endings are byte-identical; the reasoning body is the only variable.
    """
    reasoning = reasoning.strip()
    return f"{reasoning}\n{FINAL_ANSWER_TAG} {final_answer}"
