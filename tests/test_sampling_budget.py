from __future__ import annotations

import argparse
import logging

import pytest

from experiment_sampling import add_oversize_reason_argument, log_sampling_grid


def test_sampling_grid_logs_each_cell_and_total(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="experiment_sampling"):
        provenance = log_sampling_grid({"model-a/family-a": 3, "model-b/family-b": 5})

    assert "total episodes=8" in caplog.text
    assert "model-a/family-a: 3 episodes" in caplog.text
    assert "model-b/family-b: 5 episodes" in caplog.text
    assert provenance == {
        "max_episodes_per_cell": 8,
        "episodes_per_cell": {"model-a/family-a": 3, "model-b/family-b": 5},
        "total_episodes": 8,
        "oversize_reason": None,
    }


def test_sampling_grid_rejects_oversized_cells_with_names(caplog: pytest.LogCaptureFixture) -> None:
    with (
        caplog.at_level(logging.INFO, logger="experiment_sampling"),
        pytest.raises(ValueError, match=r"model-a/family-a.*9.*model-b/family-b.*12") as error,
    ):
        log_sampling_grid({"model-a/family-a": 9, "model-b/family-b": 12})

    assert "--oversize-reason" in str(error.value)
    assert "total episodes=21" in caplog.text


def test_oversize_reason_is_in_sampling_provenance() -> None:
    provenance = log_sampling_grid({"model-a/family-a": 9}, oversize_reason="publishing rerun")

    assert provenance["oversize_reason"] == "publishing rerun"
    assert provenance["episodes_per_cell"] == {"model-a/family-a": 9}


def test_oversize_reason_argument_rejects_blank_text() -> None:
    parser = argparse.ArgumentParser()
    add_oversize_reason_argument(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["--oversize-reason", "  "])
