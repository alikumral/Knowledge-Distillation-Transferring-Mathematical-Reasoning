"""Stage 4 -- evaluation (plan.md §6 Stage 4).

Runs a model (zero-shot 1B, a trained student adapter, or the 8B teacher) over
a split with greedy decoding, scores exact-match accuracy, and writes per-item
predictions for later metric/error analysis. Also measures inference latency
(batch size 1) and peak VRAM.

Prediction record schema:
    {"problem_id", "question", "gold", "completion", "pred", "correct", "gen_tokens"}
"""

from __future__ import annotations

from pathlib import Path

from .answers import (extract_pred_math, extract_pred_number, is_correct_gsm8k,
                      is_correct_math)
from .models import load_adapter, load_model_4bit, load_tokenizer, render_prompt
from .prompts import build_messages
from .utils import artifact_path, get_logger, seed_everything, write_jsonl

logger = get_logger("evaluate")


def build_gsm8k_test_items(data_cfg: dict) -> list[dict]:
    """Test items: {problem_id, question, gold} from the held-out GSM8K test split."""
    from .answers import extract_gold_gsm8k
    from .data import load_gsm8k

    test = load_gsm8k(data_cfg)["test"]
    return [{"problem_id": i, "question": r["question"], "gold": extract_gold_gsm8k(r["answer"])}
            for i, r in enumerate(test)]


def build_math_ood_items(data_cfg: dict) -> list[dict]:
    """OOD items: {problem_id, question, gold} from the frozen MATH-200 subset."""
    from .answers import extract_gold_math
    from .data import load_math_ood

    ds = load_math_ood(data_cfg)
    return [{"problem_id": i, "question": r["problem"], "gold": extract_gold_math(r["solution"])}
            for i, r in enumerate(ds)]


def load_eval_model(model_id: str, qcfg: dict, adapter_dir: str | None = None):
    """Load base model in 4-bit, optionally attaching a trained LoRA adapter."""
    tok = load_tokenizer(model_id, padding_side="left")
    model = load_model_4bit(model_id, qcfg)
    if adapter_dir:
        model = load_adapter(model, adapter_dir)
        logger.info("Attached adapter: %s", adapter_dir)
    model.eval()
    return model, tok


def run_predictions(model, tok, items: list[dict], mode: str,
                    decoding: dict, batch_size: int) -> list[dict]:
    """Generate + score predictions for a list of {problem_id, question, gold}."""
    import torch

    rows: list[dict] = []
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        prompts = [render_prompt(tok, build_messages(it["question"])) for it in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                do_sample=decoding["do_sample"],
                temperature=decoding["temperature"] if decoding["do_sample"] else None,
                max_new_tokens=decoding["max_new_tokens"],
                pad_token_id=tok.pad_token_id,
            )
        gen_only = out[:, enc["input_ids"].shape[1] :]
        texts = tok.batch_decode(gen_only, skip_special_tokens=True)
        for it, comp, g in zip(batch, texts, gen_only):
            comp = comp.strip()
            if mode == "gsm8k":
                pred = extract_pred_number(comp)
                correct = is_correct_gsm8k(comp, it["gold"])
            else:
                pred = extract_pred_math(comp)
                correct = is_correct_math(comp, it["gold"])
            rows.append({
                "problem_id": it["problem_id"], "question": it["question"],
                "gold": it["gold"], "completion": comp, "pred": pred,
                "correct": bool(correct), "gen_tokens": int((g != tok.pad_token_id).sum()),
            })
        logger.info("Scored %d / %d", min(start + batch_size, len(items)), len(items))
    return rows


def evaluate_split(model, tok, items: list[dict], mode: str, decoding: dict,
                   batch_size: int, out_name: str) -> Path:
    """Run a split end-to-end and persist predictions JSONL."""
    rows = run_predictions(model, tok, items, mode, decoding, batch_size)
    acc = sum(r["correct"] for r in rows) / max(1, len(rows))
    out_path = artifact_path("results", "predictions", out_name)
    write_jsonl(out_path, rows)
    logger.info("[%s] accuracy=%.4f (%d items) -> %s", out_name, acc, len(rows), out_path)
    return out_path


def measure_latency(model, tok, sample_questions: list[str], n_runs: int,
                    warmup: int, max_new_tokens: int) -> dict:
    """Median single-example generation latency (batch size 1) + peak VRAM."""
    import time

    import torch

    seed_everything(0)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    def _one(q):
        prompt = render_prompt(tok, build_messages(q))
        enc = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens,
                           pad_token_id=tok.pad_token_id)

    for q in sample_questions[:warmup]:
        _one(q)
    times = []
    for q in sample_questions[:n_runs]:
        t0 = time.perf_counter()
        _one(q)
        times.append(time.perf_counter() - t0)

    times.sort()
    peak_vram = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0
    return {
        "median_s": times[len(times) // 2] if times else 0.0,
        "mean_s": sum(times) / len(times) if times else 0.0,
        "peak_vram_gb": peak_vram, "n_runs": len(times),
    }
