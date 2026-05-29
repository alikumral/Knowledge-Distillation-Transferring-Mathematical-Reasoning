"""Stage 2 -- rejection sampling + SFT dataset building (plan.md §6 Stage 2).

Keeps only teacher traces whose final answer matches the gold label, reports
dataset statistics (acceptance rate overall and by difficulty -- the Risk-1
signal), and materializes the three SFT datasets used in stages (ii)/(iii)/(iv).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .answers import is_correct_gsm8k
from .data import build_sft_from_gold, build_sft_from_teacher, load_gsm8k
from .utils import (artifact_path, get_logger, iter_jsonl, write_json, write_jsonl)

logger = get_logger("reject")


def rejection_sample(traces_path: str | Path, out_path: str | Path) -> Path:
    """Filter traces to those matching gold; write accepted.jsonl. Return its path."""
    out_path = Path(out_path)
    kept, total = 0, 0
    accepted_rows = []
    for row in iter_jsonl(traces_path):
        total += 1
        if is_correct_gsm8k(row["completion"], row["gold"]):
            kept += 1
            accepted_rows.append({
                "problem_id": row["problem_id"],
                "question": row["question"],
                "gold": row["gold"],
                "completion": row["completion"],
            })
    write_jsonl(out_path, accepted_rows)
    logger.info("Accepted %d / %d traces (%.1f%%) -> %s", kept, total,
                100 * kept / max(1, total), out_path)
    return out_path


def acceptance_stats(accepted_path: str | Path) -> dict:
    """Summary stats for the report (and the Risk-1 trigger check)."""
    per_problem: dict[int, int] = defaultdict(int)
    n_traces = 0
    for row in iter_jsonl(accepted_path):
        per_problem[row["problem_id"]] += 1
        n_traces += 1
    counts = list(per_problem.values())
    stats = {
        "n_accepted_traces": n_traces,
        "n_problems_with_ge1": len(per_problem),
        "n_problems_with_ge3": sum(1 for c in counts if c >= 3),
        "mean_traces_per_covered_problem": (sum(counts) / len(counts)) if counts else 0.0,
    }
    write_json(artifact_path("sft", "acceptance_stats.json"), stats)
    logger.info("Stats: %s", stats)
    return stats


def build_all_sft_datasets(accepted_path: str | Path, data_cfg: dict) -> dict[str, Path]:
    """Write the three SFT datasets (gold, teacher_1, teacher_3) as JSONL."""
    accepted = list(iter_jsonl(accepted_path))
    gsm8k = load_gsm8k(data_cfg)
    train_rows = [{"question": r["question"], "answer": r["answer"]} for r in gsm8k["train"]]
    test_questions = [r["question"] for r in gsm8k["test"]]

    from .data import assert_no_leakage

    datasets = {
        "gold": build_sft_from_gold(train_rows),
        "teacher_1": build_sft_from_teacher(accepted, traces_per_problem=1),
        "teacher_3": build_sft_from_teacher(accepted, traces_per_problem=3),
    }

    paths: dict[str, Path] = {}
    for name, records in datasets.items():
        assert_no_leakage(records, test_questions)  # leakage guard (plan.md §6)
        p = artifact_path("sft", f"{name}.jsonl")
        write_jsonl(p, records)
        paths[name] = p
        logger.info("SFT dataset '%s': %d records -> %s", name, len(records), p)
    return paths
