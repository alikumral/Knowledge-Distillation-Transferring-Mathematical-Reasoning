"""mathdistill — knowledge distillation of math reasoning (Llama-3.1-8B → Llama-3.2-1B).

Pipeline stages (see plan.md §3):
    1. generate  — teacher CoT generation
    2. reject    — rejection sampling + SFT dataset building
    3. train     — QLoRA fine-tuning of the 1B student
    4. evaluate  — accuracy / latency / VRAM / error analysis

The pure-Python foundation (prompts, answers, utils) has no heavy deps and is
unit-tested locally; the GPU stages lazy-import torch/transformers so this
package imports cleanly even without them installed.
"""

__version__ = "0.1.0"
