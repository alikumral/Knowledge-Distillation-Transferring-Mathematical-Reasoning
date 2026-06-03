"""Stage 3 -- QLoRA fine-tuning of the 1B student (plan.md §6 Stage 3).

Loads the student in 4-bit NF4, attaches LoRA adapters, and trains with TRL's
SFTTrainer using completion-only loss where the TRL version supports it.
One call = one (condition, seed) run; the notebook loops seeds.

VERSION COMPATIBILITY: SFTConfig and SFTTrainer APIs drift between TRL versions.
This module uses inspect to detect what the installed version accepts and falls
back gracefully. Tested against TRL 0.9-0.12.
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path

from .models import build_lora_config, load_model_4bit, load_tokenizer
from .utils import get_logger, read_jsonl, seed_everything, write_json

logger = get_logger("train")


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #
def _load_sft_dataset(cfg: dict):
    from datasets import Dataset

    from .utils import artifact_path

    src = cfg["dataset"]["source"]
    n   = cfg["dataset"]["traces_per_problem"]
    name = "gold" if src == "gold" else ("teacher_1" if n == 1 else "teacher_3")

    path = artifact_path("sft", f"{name}.jsonl")
    records = read_jsonl(path)
    logger.info("Loaded SFT dataset '%s': %d records", name, len(records))
    return Dataset.from_list([{"messages": r["messages"]} for r in records])


# --------------------------------------------------------------------------- #
# Version-aware SFTConfig builder
# --------------------------------------------------------------------------- #
def _build_sft_config(out_dir: Path, s: dict, seed: int):
    from trl import SFTConfig

    valid = set(inspect.signature(SFTConfig.__init__).parameters)

    # Core params — present in all TRL versions that have SFTConfig
    kwargs: dict = dict(
        output_dir=str(out_dir),
        num_train_epochs=s["num_train_epochs"],
        learning_rate=float(s["learning_rate"]),
        lr_scheduler_type=s["lr_scheduler_type"],
        warmup_ratio=s["warmup_ratio"],
        per_device_train_batch_size=s["per_device_train_batch_size"],
        gradient_accumulation_steps=s["gradient_accumulation_steps"],
        bf16=s["bf16"],
        gradient_checkpointing=s["gradient_checkpointing"],
        logging_steps=s["logging_steps"],
        save_steps=s["save_steps"],
        optim=s["optim"],
        seed=seed,
        report_to="none",
    )

    # Version-sensitive params
    for key, val in [
        ("max_seq_length", s["max_seq_len"]),
        ("assistant_only_loss", s.get("assistant_only_loss", True)),
    ]:
        if key in valid:
            kwargs[key] = val
        else:
            logger.info("SFTConfig: '%s' not in this TRL version — skipping", key)

    return SFTConfig(**kwargs)


# --------------------------------------------------------------------------- #
# Completion-only collator fallback (when assistant_only_loss not in SFTConfig)
# --------------------------------------------------------------------------- #
def _maybe_collator(tok, s: dict):
    """Return DataCollatorForCompletionOnlyLM if assistant_only_loss unavailable."""
    from trl import SFTConfig
    if "assistant_only_loss" in inspect.signature(SFTConfig.__init__).parameters:
        return None  # SFTConfig handles it natively

    try:
        from trl import DataCollatorForCompletionOnlyLM
        # Llama-3 assistant header — marks where completion loss begins
        response_template = "<|start_header_id|>assistant<|end_header_id|>"
        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=tok,
            mlm=False,
        )
        logger.info("Using DataCollatorForCompletionOnlyLM for completion-only loss")
        return collator
    except Exception as e:
        logger.warning(
            "Completion-only loss unavailable (%s). Training on full sequence. "
            "Document this in the report as a minor limitation.", e
        )
        return None


# --------------------------------------------------------------------------- #
# Main training entry point
# --------------------------------------------------------------------------- #
def train_one(cfg: dict, data_cfg: dict, seed: int) -> Path:
    from trl import SFTTrainer

    from .utils import artifact_path

    seed_everything(seed)

    dataset = _load_sft_dataset(cfg)
    tok = load_tokenizer(cfg["student_model_id"], padding_side="right")
    # Set max length on the tokenizer regardless of SFTConfig version support
    tok.model_max_length = cfg["sft"]["max_seq_len"]

    model = load_model_4bit(cfg["student_model_id"], cfg["quantization"])
    model.config.use_cache = False

    peft_config = build_lora_config(cfg["lora"])

    s = cfg["sft"]
    out_dir = artifact_path(cfg["output_dir"], f"seed{seed}")

    sft_args = _build_sft_config(out_dir, s, seed)
    collator = _maybe_collator(tok, s)

    # SFTTrainer: processing_class (new) vs tokenizer (old)
    trainer_sig = set(inspect.signature(SFTTrainer.__init__).parameters)
    trainer_kwargs: dict = dict(
        model=model,
        args=sft_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    if "processing_class" in trainer_sig:
        trainer_kwargs["processing_class"] = tok
    else:
        trainer_kwargs["tokenizer"] = tok
    if collator is not None:
        trainer_kwargs["data_collator"] = collator

    logger.info("Training condition=%s seed=%d -> %s", cfg["condition"], seed, out_dir)
    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(str(out_dir))

    write_json(out_dir / "run_meta.json", {
        "condition": cfg["condition"], "seed": seed,
        "student_model_id": cfg["student_model_id"],
        "lora": cfg["lora"], "sft": s,
        "trl_version": __import__("trl").__version__,
    })
    logger.info("Saved adapter -> %s", out_dir)
    return out_dir


def train_all_seeds(cfg: dict, data_cfg: dict) -> list[Path]:
    return [train_one(cfg, data_cfg, seed) for seed in cfg["seeds"]]
