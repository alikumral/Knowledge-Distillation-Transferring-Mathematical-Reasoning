"""Model / tokenizer loading: 4-bit NF4 quantization, LoRA config, chat templates.

GPU-only; torch/transformers/peft are imported lazily so the rest of the package
stays importable on a CPU box. Validate the exact behavior in the Colab pilot.
"""

from __future__ import annotations


def _compute_dtype(name: str):
    import torch

    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def build_bnb_config(qcfg: dict):
    """BitsAndBytesConfig for 4-bit NF4 + double quantization."""
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=qcfg["load_in_4bit"],
        bnb_4bit_quant_type=qcfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=qcfg["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=_compute_dtype(qcfg["bnb_4bit_compute_dtype"]),
    )


def load_tokenizer(model_id: str, padding_side: str = "left"):
    """Load tokenizer. padding_side='left' for generation, 'right' for training."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = padding_side
    return tok


def load_model_4bit(model_id: str, qcfg: dict):
    """Load a causal LM in 4-bit NF4 across available GPU(s)."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=build_bnb_config(qcfg),
        device_map="auto",
        dtype=_compute_dtype(qcfg["bnb_4bit_compute_dtype"]),
    )
    model.config.use_cache = True  # generation; SFTTrainer disables this for training
    return model


def build_lora_config(lcfg: dict):
    """LoRA config for QLoRA fine-tuning."""
    from peft import LoraConfig

    return LoraConfig(
        r=lcfg["r"],
        lora_alpha=lcfg["alpha"],
        lora_dropout=lcfg["dropout"],
        target_modules=lcfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_adapter(base_model, adapter_dir: str):
    """Attach a trained LoRA adapter to a base model for evaluation."""
    from peft import PeftModel

    return PeftModel.from_pretrained(base_model, adapter_dir)


def render_prompt(tokenizer, messages: list[dict]) -> str:
    """Apply the chat template, adding the assistant generation prompt (for inference)."""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
