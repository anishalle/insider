from __future__ import annotations

import ast
from pathlib import Path

import pytest


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def test_model_trainer_packages_parse_as_python39() -> None:
    for package in ("src/lr", "src/rnn", "src/lstm", "src/modeling_common"):
        for path in sorted(Path(package).rglob("*.py")):
            source = path.read_text()
            ast.parse(source, filename=str(path), feature_version=(3, 9))


@pytest.mark.skipif(not _has_torch(), reason="torch is not installed in the local test environment")
def test_recurrent_trainers_have_main_entrypoints() -> None:
    from lstm.train import main as lstm_main
    from rnn.train import main as rnn_main

    assert callable(rnn_main)
    assert callable(lstm_main)


def test_xgboost_trainer_has_main_entrypoint() -> None:
    from xgb_model.train import main as xgboost_main

    assert callable(xgboost_main)
