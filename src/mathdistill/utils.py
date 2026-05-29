"""Shared utilities: seeding, JSONL I/O, paths, config loading, timing, logging.

Heavy libs (numpy/torch/transformers) are imported lazily inside functions so
this module loads in a GPU-free local environment for testing.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def seed_everything(seed: int) -> None:
    """Seed all RNGs we touch (stdlib, numpy, torch, transformers)."""
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        import transformers

        transformers.set_seed(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# JSONL I/O (artifacts are JSONL so stages are resumable on Colab)
# --------------------------------------------------------------------------- #
def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    """Stream rows without loading the whole file (large trace pools)."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    """Append rows (used for incremental, resumable generation)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(path: str | Path) -> dict:
    """Load a YAML config (all hyperparameters live in configs/*.yaml)."""
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
# Paths (local vs Google Drive resolved via env var)
# --------------------------------------------------------------------------- #
def base_dir() -> Path:
    """Root for all non-committed artifacts.

    On Colab, set MATHDISTILL_HOME to a Drive path, e.g.
    /content/drive/MyDrive/math-distillation. Defaults to ./outputs locally.
    """
    return Path(os.environ.get("MATHDISTILL_HOME", "outputs"))


def artifact_path(*parts: str) -> Path:
    p = base_dir().joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------- #
# Timing / logging
# --------------------------------------------------------------------------- #
@contextmanager
def timer(name: str = "block", logger: logging.Logger | None = None):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    msg = f"[timer] {name}: {dt:.2f}s"
    (logger.info(msg) if logger else print(msg))


def get_logger(name: str = "mathdistill") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
