"""CLI: Stage 4 -- evaluate one model/condition on GSM8K test + MATH-200.

    # zero-shot 1B baseline (condition i):
    python scripts/run_eval.py --model-id meta-llama/Llama-3.2-1B-Instruct --name zeroshot
    # a trained student (condition iii, seed 0):
    python scripts/run_eval.py --model-id meta-llama/Llama-3.2-1B-Instruct \
        --adapter adapters/teacher_1/seed0 --name teacher_1_seed0
    # the 8B teacher upper bound (condition v):
    python scripts/run_eval.py --model-id meta-llama/Llama-3.1-8B-Instruct --name teacher
"""

import argparse

from mathdistill.evaluate import (build_gsm8k_test_items, build_math_ood_items,
                                  evaluate_split, load_eval_model, measure_latency)
from mathdistill.utils import artifact_path, load_config, write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--adapter", default=None, help="adapter dir under MATHDISTILL_HOME")
    ap.add_argument("--name", required=True, help="output tag, e.g. teacher_1_seed0")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dcfg = load_config(args.data_config)

    adapter_dir = str(artifact_path(*args.adapter.split("/"))) if args.adapter else None
    model, tok = load_eval_model(args.model_id, cfg["quantization"], adapter_dir)

    if cfg["splits"]["gsm8k_test"]:
        items = build_gsm8k_test_items(dcfg)
        evaluate_split(model, tok, items, "gsm8k", cfg["decoding"],
                       cfg["batch_size"], f"{args.name}_gsm8k.jsonl")

    if cfg["splits"]["math_ood"]:
        items = build_math_ood_items(dcfg)
        evaluate_split(model, tok, items, "math", cfg["decoding"],
                       cfg["batch_size"], f"{args.name}_math.jsonl")

    if cfg["latency"]["enabled"]:
        sample_qs = [it["question"] for it in build_gsm8k_test_items(dcfg)]
        lat = measure_latency(model, tok, sample_qs, cfg["latency"]["n_runs"],
                              cfg["latency"]["warmup"], cfg["decoding"]["max_new_tokens"])
        write_json(artifact_path("results", "tables", f"{args.name}_latency.json"), lat)
        print("latency/vram:", lat)


if __name__ == "__main__":
    main()
