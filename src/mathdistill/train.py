"""Stage 3 -- QLoRA fine-tuning of the 1B student (plan.md §6 Stage 3).

Loads the student in 4-bit NF4, attaches LoRA adapters, and trains with TRL's
SFTTrainer using completion-only loss (only the assistant turn contributes to
the loss). One call = one (condition, seed) run; the notebook loops seeds.

VERSION-SENSITIVE: the TRL SFTConfig field names below (`max_seq_length`,
`assistant_only_loss`) drift between releases. Validate in the pilot run and
pin versions in requirements.txt before the full sweep.
"""

from __future__ import annotations

from pathlib import Path

from .models import build_lora_config, load_model_4bit, load_tokenizer
from .utils import get_logger, iter_jsonl, read_jsonl, seed_everything, write_json

logger = get_logger("train")


def _load_sft_dataset(cfg, data_cfg):
    """Return a HF Dataset with a 'messages' column for the configured condition."""
    from datasets import Dataset

    from .utils import artifact_path

    name = "gold" if cfg["dataset"]["source"] == "gold" else (
        "teacher_1" if cfg["dataset"]["traces_per_problem"] == 1 else "teacher_3"
    )
    path = artifact_path("sft", f"{name}.jsonl")
    records = read_jsonl(path)
    logger.info("Loaded SFT dataset '%s': %d records", name, len(records))
    return Dataset.from_list([{"messages": r["messages"]} for r in records])


def train_one(cfg: dict, data_cfg: dict, seed: int) -> Path:
    """Train one adapter for (condition, seed). Returns the adapter directory."""
    from trl import SFTConfig, SFTTrainer

    from .utils import artifact_path

    seed_everything(seed)

    dataset = _load_sft_dataset(cfg, data_cfg)
    tok = load_tokenizer(cfg["student_model_id"], padding_side="right")
    model = load_model_4bit(cfg["student_model_id"], cfg["quantization"])
    model.config.use_cache = False  # required with gradient checkpointing

    peft_config = build_lora_config(cfg["lora"])

    s = cfg["sft"]
    out_dir = artifact_path(cfg["output_dir"], f"seed{seed}")
    sft_args = SFTConfig(
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
        max_seq_length=s["max_seq_len"],
        assistant_only_loss=s["assistant_only_loss"],  # completion-only loss
        seed=seed,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tok,
    )

    logger.info("Training condition=%s seed=%d -> %s", cfg["condition"], seed, out_dir)
    trainer.train()
    trainer.save_model(str(out_dir))  # saves the LoRA adapter
    write_json(out_dir / "run_meta.json", {
        "condition": cfg["condition"], "seed": seed,
        "student_model_id": cfg["student_model_id"], "lora": cfg["lora"], "sft": s,
    })
    logger.info("Saved adapter -> %s", out_dir)
    return out_dir


def train_all_seeds(cfg: dict, data_cfg: dict) -> list[Path]:
    """Train every seed listed in the config."""
    return [train_one(cfg, data_cfg, seed) for seed in cfg["seeds"]]
