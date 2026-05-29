"""CLI: Stage 1 -- teacher CoT generation.

    python scripts/run_generate.py --config configs/teacher_gen.yaml

Set MATHDISTILL_HOME to your Google Drive path so output survives disconnects.
"""

import argparse

from mathdistill.generate import generate_teacher_traces
from mathdistill.utils import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/teacher_gen.yaml")
    ap.add_argument("--data-config", default="configs/data.yaml")
    args = ap.parse_args()
    generate_teacher_traces(load_config(args.config), load_config(args.data_config))


if __name__ == "__main__":
    main()
