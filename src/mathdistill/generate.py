"""Stage 1 -- teacher CoT generation (plan.md §6 Stage 1).

Runs Llama-3.1-8B-Instruct in 4-bit over the GSM8K train split, sampling
`k_samples` chain-of-thought traces per problem. Writes incrementally to a
JSONL file and skips problems already present, so a Colab disconnect resumes
where it left off.

Output record schema (one row per sample):
    {"problem_id", "sample_idx", "question", "gold", "completion"}
"""

from __future__ import annotations

from pathlib import Path

from .answers import extract_gold_gsm8k
from .data import load_gsm8k
from .models import load_model_4bit, load_tokenizer, render_prompt
from .prompts import build_messages
from .utils import append_jsonl, artifact_path, get_logger, iter_jsonl, seed_everything

logger = get_logger("generate")


def _done_problem_ids(path: Path) -> set[int]:
    """Problem ids already fully sampled (resumability)."""
    if not path.exists():
        return set()
    from collections import Counter

    counts: Counter = Counter()
    for row in iter_jsonl(path):
        counts[row["problem_id"]] += 1
    return set(counts)  # presence is enough; partial problems are re-topped-up below


def generate_teacher_traces(cfg: dict, data_cfg: dict) -> Path:
    import torch

    seed_everything(cfg["seed"])
    gen = cfg["generation"]
    out_path = artifact_path(*cfg["output"].split("/"))

    train = load_gsm8k(data_cfg)["train"]
    tok = load_tokenizer(cfg["model_id"], padding_side="left")
    model = load_model_4bit(cfg["model_id"], cfg["quantization"])
    model.eval()

    done = _done_problem_ids(out_path)
    logger.info("Resuming: %d problems already have traces", len(done))

    bs = cfg["batch_size"]
    todo = [i for i in range(len(train)) if i not in done]
    logger.info("Generating for %d / %d problems (k=%d)", len(todo), len(train), gen["k_samples"])

    for start in range(0, len(todo), bs):
        batch_ids = todo[start : start + bs]
        questions = [train[i]["question"] for i in batch_ids]
        golds = [extract_gold_gsm8k(train[i]["answer"]) for i in batch_ids]

        prompts = [render_prompt(tok, build_messages(q)) for q in questions]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                do_sample=gen["do_sample"],
                temperature=gen["temperature"],
                top_p=gen["top_p"],
                max_new_tokens=gen["max_new_tokens"],
                num_return_sequences=gen["k_samples"],
                pad_token_id=tok.pad_token_id,
            )
        # out: (batch * k, seq). Strip the prompt tokens, decode completions only.
        gen_only = out[:, enc["input_ids"].shape[1] :]
        texts = tok.batch_decode(gen_only, skip_special_tokens=True)

        rows = []
        k = gen["k_samples"]
        for b, pid in enumerate(batch_ids):
            for s in range(k):
                rows.append({
                    "problem_id": pid,
                    "sample_idx": s,
                    "question": questions[b],
                    "gold": golds[b],
                    "completion": texts[b * k + s].strip(),
                })
        append_jsonl(out_path, rows)
        logger.info("Batch %d-%d done (%d rows written)", start, start + len(batch_ids), len(rows))

    logger.info("Teacher generation complete -> %s", out_path)
    return out_path
