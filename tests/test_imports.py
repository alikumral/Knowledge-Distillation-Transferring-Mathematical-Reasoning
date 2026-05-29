"""Smoke test: every submodule must import without torch/transformers/peft/trl
installed (heavy deps are lazy-imported inside functions). This guards the
'clean import on a CPU box' contract that keeps local testing fast.
"""

import importlib

import pytest

MODULES = [
    "mathdistill",
    "mathdistill.prompts",
    "mathdistill.answers",
    "mathdistill.utils",
    "mathdistill.data",
    "mathdistill.models",
    "mathdistill.generate",
    "mathdistill.reject",
    "mathdistill.train",
    "mathdistill.evaluate",
    "mathdistill.metrics",
]


@pytest.mark.parametrize("mod", MODULES)
def test_module_imports(mod):
    importlib.import_module(mod)
