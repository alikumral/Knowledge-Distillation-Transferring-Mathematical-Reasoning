"""CLI: Stage 3 -- QLoRA fine-tuning.

    # all seeds in the config:
    python scripts/run_train.py --config configs/train_teacher_1.yaml
    # a single seed (parallelize across sessions/accounts):
    python scripts/run_train.py --config configs/train_teacher_1.yaml --seed 0
"""

import argparse

from mathdistill.train import train_all_seeds, train_one
from mathdistill.utils import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--seed", type=int, default=None,
                    help="train only this seed (omit to train all seeds in config)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dcfg = load_config(args.data_config)
    if args.seed is not None:
        train_one(cfg, dcfg, args.seed)
    else:
        train_all_seeds(cfg, dcfg)


if __name__ == "__main__":
    main()
