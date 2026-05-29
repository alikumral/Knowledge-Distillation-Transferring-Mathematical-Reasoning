"""Stage 1 -- teacher CoT generation (plan.md §6 Stage 1).

Runs Llama-3.1-8B-Instruct in 4-bit over the GSM8K train split. Designed for
**multi-pass sampling**: each pass (pass_id 0, 1, 2) samples one trace per
problem with temperature, giving diverse traces without needing high k in a
single generate() call (which multiplies VRAM pressure by k).

  Pass 0: sample_idx=0 for all 7,473 problems  (~14h on T4, batch_size=16)
  Pass 1: sample_idx=1 for all 7,473 problems  (~14h, different random seed)
  Pass 2: sample_idx=2 for all 7,473 problems  (~14h)

All passes append to the same JSONL file. Resumability is at the
(problem_id, sample_idx) pair level, so a disconnect within a pass resumes
exactly where it left off.

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


def _done_pairs(path: Path) -> set[tuple[int, int]]:
    """Set of (problem_id, sample_idx) pairs already written (pair-level resume)."""
    if not path.exists():
        return set()
    return {(row["problem_id"], row["sample_idx"]) for row in iter_jsonl(path)}


def generate_teacher_traces(cfg: dict, data_cfg: dict) -> Path:
    import torch

    pass_id: int = int(cfg.get("pass_id", 0))
    k: int = int(cfg["generation"]["k_samples"])  # typically 1 in multi-pass mode
    seed_everything(cfg["seed"] + pass_id)         # different seed per pass for diversity

    gen = cfg["generation"]
    out_path = artifact_path(*cfg["output"].split("/"))

    train = load_gsm8k(data_cfg)["train"]
    n_problems = len(train)
    if cfg.get("limit"):
        n_problems = min(n_problems, int(cfg["limit"]))
        logger.info("LIMIT set -> pilot mode: first %d problems only", n_problems)

    done = _done_pairs(out_path)
    logger.info("Pass %d | resuming: %d (problem, sample) pairs already written",
                pass_id, len(done))

    tok = load_tokenizer(cfg["model_id"], padding_side="left")
    model = load_model_4bit(cfg["model_id"], cfg["quantization"])
    model.eval()

    bs = cfg["batch_size"]
    # Problems that still need sample_idx = pass_id * k + s for any s in range(k)
    todo = [
        i for i in range(n_problems)
        if any((i, pass_id * k + s) not in done for s in range(k))
    ]
    logger.info("Pass %d | generating for %d / %d problems (k=%d, batch_size=%d)",
                pass_id, len(todo), n_problems, k, bs)

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
                num_return_sequences=k,
                pad_token_id=tok.pad_token_id,
            )
        gen_only = out[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(gen_only, skip_special_tokens=True)

        rows = []
        for b, pid in enumerate(batch_ids):
            for s in range(k):
                sample_idx = pass_id * k + s
                if (pid, sample_idx) in done:
                    continue
                rows.append({
                    "problem_id": pid,
                    "sample_idx": sample_idx,
                    "question": questions[b],
                    "gold": golds[b],
                    "completion": texts[b * k + s].strip(),
                })
        append_jsonl(out_path, rows)
        logger.info("Pass %d | batch %d-%d done (%d rows written)",
                    pass_id, start, start + len(batch_ids), len(rows))

    logger.info("Pass %d complete -> %s", pass_id, out_path)
    return out_path
