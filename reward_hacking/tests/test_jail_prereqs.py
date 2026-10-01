"""Collection-time checks for the real episode-jail prerequisite probe."""

from __future__ import annotations

import os

import pytest
from conftest import (
    JAIL_AVAILABLE,
    JAIL_RESOURCE_LIMITS,
    JAIL_UNAVAILABLE_REASON,
    TEST_JAIL_ADVISORY_LIMITS,
)


@pytest.mark.skipif(
    not TEST_JAIL_ADVISORY_LIMITS,
    reason="set TEST_JAIL_ADVISORY_LIMITS=1 on an offline host to run the real jail prerequisite probe",
)
def test_explicit_advisory_opt_in_enables_real_unshare_runner() -> None:
    """The opt-in is visible in the resolved mode after the real probe succeeds."""
    assert os.environ["TEST_JAIL_ADVISORY_LIMITS"] == "1"
    assert JAIL_AVAILABLE, JAIL_UNAVAILABLE_REASON
    assert JAIL_RESOURCE_LIMITS is not None
    assert JAIL_RESOURCE_LIMITS.mode == "advisory"


def test_strict_default_does_not_select_advisory_limits() -> None:
    """The test-only escape hatch must never be implicit."""
    if TEST_JAIL_ADVISORY_LIMITS:
        pytest.skip("the explicit advisory test mode is enabled")
    assert JAIL_RESOURCE_LIMITS is not None
    assert JAIL_RESOURCE_LIMITS.mode == "enforced"
