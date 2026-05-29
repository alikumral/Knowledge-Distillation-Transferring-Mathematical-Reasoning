"""Metrics: accuracy, bootstrap confidence intervals, difficulty buckets,
generation-length stats, multi-seed aggregation (plan.md §6 Stage 4, §7).

Pure-Python where possible (unit-testable); numpy is lazy-imported only for the
bootstrap.
"""

from __future__ import annotations

from typing import Sequence


def accuracy(correct: Sequence[bool]) -> float:
    return (sum(1 for c in correct if c) / len(correct)) if correct else 0.0


def bootstrap_ci(correct: Sequence[bool], n_boot: int = 10000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap CI for accuracy over the test items."""
    import numpy as np

    arr = np.asarray([1 if c else 0 for c in correct], dtype=float)
    if arr.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, arr.size, size=(n_boot, arr.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def assign_difficulty(lengths: Sequence[int]) -> list[str]:
    """Bucket items into easy/medium/hard by gold-CoT-length tertiles."""
    if not lengths:
        return []
    ordered = sorted(lengths)
    n = len(ordered)
    t1 = ordered[n // 3]
    t2 = ordered[2 * n // 3]
    out = []
    for x in lengths:
        if x <= t1:
            out.append("easy")
        elif x <= t2:
            out.append("medium")
        else:
            out.append("hard")
    return out


def accuracy_by_bucket(correct: Sequence[bool], buckets: Sequence[str]) -> dict[str, float]:
    groups: dict[str, list[bool]] = {"easy": [], "medium": [], "hard": []}
    for c, b in zip(correct, buckets):
        groups.setdefault(b, []).append(c)
    return {b: accuracy(v) for b, v in groups.items() if v}


def length_stats(lengths: Sequence[int]) -> dict:
    """Generation-length summary (verbosity-inheritance check)."""
    if not lengths:
        return {"n": 0}
    import numpy as np

    a = np.asarray(lengths, dtype=float)
    return {
        "n": int(a.size), "mean": float(a.mean()), "std": float(a.std()),
        "p50": float(np.median(a)), "p90": float(np.quantile(a, 0.9)),
        "min": float(a.min()), "max": float(a.max()),
    }


def aggregate_seeds(per_seed_acc: Sequence[float]) -> dict:
    """Mean +/- std of accuracy across seeds."""
    import numpy as np

    a = np.asarray(per_seed_acc, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std()), "n_seeds": int(a.size),
            "values": [float(x) for x in a]}
