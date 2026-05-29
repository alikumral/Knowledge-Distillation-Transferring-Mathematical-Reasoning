"""CLI: Stage 2 -- rejection sampling + build the three SFT datasets.

    python scripts/run_reject.py
"""

import argparse

from mathdistill.reject import (acceptance_stats, build_all_sft_datasets,
                                rejection_sample)
from mathdistill.utils import artifact_path, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="traces/teacher_traces.jsonl",
                    help="path under MATHDISTILL_HOME")
    ap.add_argument("--data-config", default="configs/data.yaml")
    args = ap.parse_args()

    traces_path = artifact_path(*args.traces.split("/"))
    accepted_path = artifact_path("sft", "accepted.jsonl")

    rejection_sample(traces_path, accepted_path)
    acceptance_stats(accepted_path)
    build_all_sft_datasets(accepted_path, load_config(args.data_config))


if __name__ == "__main__":
    main()
