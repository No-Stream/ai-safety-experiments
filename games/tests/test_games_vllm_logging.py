"""TRL silences vLLM at import; games training must turn its warnings back on."""

from __future__ import annotations

import logging

import pytest

from games import train as gt


def test_restore_undoes_trls_error_only_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_LOGGING_LEVEL", "ERROR")
    vllm_logger = logging.getLogger("vllm")
    monkeypatch.setattr(vllm_logger, "level", logging.ERROR)

    gt.restore_vllm_logging()

    assert gt.os.environ["VLLM_LOGGING_LEVEL"] == "INFO"
    assert vllm_logger.level == logging.INFO
    assert vllm_logger.isEnabledFor(logging.WARNING)
