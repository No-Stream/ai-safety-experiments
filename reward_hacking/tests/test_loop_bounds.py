"""What bounds one episode: its own wall clock, and the share of the box each jailed command gets.

Both bounds were missing, and both are about a sweep rather than about one episode. The turn cap
looked like the bound on an episode and was not: every action in a completion got a fresh full
timeout, nothing caps how many actions a completion may contain, and there was no deadline over the
episode, so one turn emitting N ``<run>`` blocks cost N times the per-command ceiling and an episode
cost ``max_turns`` times that. On a paid sweep that is money, and on this box it is the rule that
local work stays inside ten minutes.

The second bound is what a single jailed command may consume. ``run_in_jail`` composes
``resource-limits.sh`` outside the jail and passed it only ``-t``, so every model-issued command
inherited the limiter's defaults -- sized for a training job on the host, which on this box is 48 of
64 cores and 151 GiB of RAM. One shell one-liner could therefore take most of a shared box before
its cgroup killed it.

Neither bound may quietly rewrite what an episode's record means, which is why the truncation is a
field of its own rather than a new outcome: ``hack_detector.traces_from_records`` recomputes the
outcome and raises when it disagrees with the recorded one, so a new terminal state would make every
truncated episode unreadable to the re-analysis it exists for. A truncated episode stays
``incomplete`` -- it ran out of room, which is what that category already means -- and says on the
record that the clock, not the turn cap, is what ran out.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any, cast

from reward_hacking.harness import loop
from reward_hacking.harness.hack_detector import traces_from_records
from reward_hacking.harness.loop import (
    DEFAULT_EPISODE_SECONDS,
    DEFAULT_MAX_TURNS,
    EPISODE_CPUS,
    EPISODE_MEMORY_GIB,
    AgentEpisodeTrace,
    EpisodeOutcome,
    episode_limits,
    run_agent_episode,
)
from reward_hacking.harness.tasks import TASKS_BY_ID
from reward_hacking.jailed_poc import PROCESS_TIMEOUT_SECONDS, CommandResult
from reward_hacking.model_backend import MockBackend

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

_TASK_ID = "sum-ledger"

# Three commands in one completion is the shape the finding is about: one turn, N leases on the box.
_THREE_COMMAND_TURN = (
    "<run>printf 'a' > a.txt</run>\n<run>printf 'b' > b.txt</run>\n<run>printf 'c' > c.txt</run>"
)

# Long enough that a command lands inside the budget and two do not, short enough to keep the suite
# quick. Real sleeps rather than a faked clock: the deadline is wall clock, and a stubbed monotonic
# would pass while the thing being bounded went unmeasured.
_COMMAND_SECONDS = 0.15
_TIGHT_BUDGET_SECONDS = 0.2


def _stub_the_jail(
    monkeypatch: pytest.MonkeyPatch, *, seconds_per_command: float = 0.0
) -> list[dict[str, object]]:
    """Answer every jailed command instantly (or slowly), recording how it was invoked.

    The returned list holds one entry per call including the keyword arguments, which is how the
    resource-share half of this module checks what the limiter was actually asked for.
    """
    calls: list[dict[str, object]] = []

    def fake_run_in_jail(directory: Path, command: str, **kwargs: object) -> CommandResult:
        calls.append({"episode_dir": directory, "command": command, **kwargs})
        if seconds_per_command:
            time.sleep(seconds_per_command)
        return CommandResult(command=command, exit_code=0, stdout="", stderr="", timed_out=False)

    monkeypatch.setattr(loop, "run_in_jail", fake_run_in_jail)
    return calls


def _episode(
    monkeypatch: pytest.MonkeyPatch,
    episode_dir: Path,
    completions: list[str],
    *,
    episode_seconds: float = DEFAULT_EPISODE_SECONDS,
    seconds_per_command: float = 0.0,
) -> tuple[AgentEpisodeTrace, list[dict[str, object]]]:
    """Run one episode against a stubbed jail, returning its trace and every jailed call."""
    calls = _stub_the_jail(monkeypatch, seconds_per_command=seconds_per_command)
    trace = run_agent_episode(
        TASKS_BY_ID[_TASK_ID],
        MockBackend(completions, model_id="mock-bounds"),
        episode_dir=episode_dir,
        episode_seconds=episode_seconds,
    )
    return trace, calls


class TestAnEpisodeIsBoundedByItsOwnWallClock:
    """One completion with N run blocks used to cost N leases on the box, with nothing above it."""

    def test_a_turn_stops_issuing_commands_once_the_budget_is_spent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        trace, calls = _episode(
            monkeypatch,
            tmp_path / "truncated",
            [_THREE_COMMAND_TURN],
            episode_seconds=_TIGHT_BUDGET_SECONDS,
            seconds_per_command=_COMMAND_SECONDS,
        )

        grader_command = TASKS_BY_ID[_TASK_ID].grader_command
        policy_commands = [call["command"] for call in calls if call["command"] != grader_command]
        assert 1 <= len(policy_commands) < 3, (
            f"the deadline neither ran a command nor stopped the turn: {policy_commands}"
        )
        # The completion still holds all three, so re-analysis sees what the policy asked for.
        assert trace.turns[0].completion.count("<run>") == 3
        assert len(trace.turns[0].commands) == len(policy_commands)

    def test_the_truncation_is_recorded_rather_than_left_to_be_inferred(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Cut off by the clock reads as a policy that stopped, unless the record says otherwise."""
        trace, _ = _episode(
            monkeypatch,
            tmp_path / "recorded",
            [_THREE_COMMAND_TURN],
            episode_seconds=_TIGHT_BUDGET_SECONDS,
            seconds_per_command=_COMMAND_SECONDS,
        )

        assert trace.deadline_exceeded is True
        assert trace.deadline_seconds == _TIGHT_BUDGET_SECONDS
        deadline = cast("dict[str, Any]", trace.summary_record()["deadline"])
        assert deadline == {
            "seconds": _TIGHT_BUDGET_SECONDS,
            "exceeded": True,
            "phase": "actions",
            "engine_wait_seconds": 0.0,
        }
        assert "deadline_exceeded=True" in trace.summary_line()
        assert "deadline_phase=actions" in trace.summary_line()

    def test_a_truncated_episode_keeps_the_outcome_its_own_re_analysis_recomputes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The truncation is a field, not a new terminal state, so the rebuild still agrees.

        ``traces_from_records`` raises when a recorded outcome disagrees with a recompute over the
        rebuilt trace. A ``deadline_exceeded`` outcome would therefore make every truncated episode
        unreadable to the post-hoc detector, which does not know the field -- so the truncated
        episode stays ``incomplete`` ("ran out of room") and the flag says the clock ran out.
        """
        trace, _ = _episode(
            monkeypatch,
            tmp_path / "rebuilt",
            [_THREE_COMMAND_TURN],
            episode_seconds=_TIGHT_BUDGET_SECONDS,
            seconds_per_command=_COMMAND_SECONDS,
        )
        assert trace.deadline_exceeded is True
        assert trace.outcome is EpisodeOutcome.INCOMPLETE

        records = [*trace.turn_records(), trace.summary_record()]
        rebuilt = traces_from_records(records)
        assert [rebuilt_trace.outcome for rebuilt_trace in rebuilt] == [EpisodeOutcome.INCOMPLETE]

    def test_an_episode_inside_its_budget_is_untouched(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The control: the bound must not truncate the episodes the sweep is actually made of."""
        trace, calls = _episode(
            monkeypatch, tmp_path / "quick", [_THREE_COMMAND_TURN, "Nothing further."]
        )

        assert trace.deadline_exceeded is False
        assert len(trace.turns[0].commands) == 3
        assert len(calls) == 4  # three policy commands plus the harness's own grader run
        assert cast("dict[str, Any]", trace.summary_record()["deadline"])["exceeded"] is False

    def test_the_default_budget_bounds_the_turn_cap_without_cutting_one_slow_command(self) -> None:
        """Why 20 minutes: above the host-side per-command ceiling, below what turns alone allow.

        A budget under ``PROCESS_TIMEOUT_SECONDS`` would let a single slow command end the episode,
        and one at or above ``max_turns`` times that ceiling would bound nothing the turn cap did
        not already bound -- which is the state this replaced.
        """
        assert DEFAULT_EPISODE_SECONDS > PROCESS_TIMEOUT_SECONDS
        assert DEFAULT_EPISODE_SECONDS < DEFAULT_MAX_TURNS * PROCESS_TIMEOUT_SECONDS


class TestEveryJailedCommandGetsAnEpisodeSizedShareOfTheBox:
    """The limiter's defaults are a training job's share; an episode command is a one-liner."""

    def test_the_policy_commands_and_the_graders_run_are_both_sized(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cpus, mem_max = episode_limits()
        _, calls = _episode(monkeypatch, tmp_path / "sized", [_THREE_COMMAND_TURN, "Done."])

        assert len(calls) == 4, [call["command"] for call in calls]
        for call in calls:
            assert call["cpus"] == cpus, call
            assert call["mem_max"] == mem_max, call

    def test_the_share_is_derived_from_the_host_rather_than_hardcoded(self) -> None:
        """A fixed share is no cap on a small box: 8 GiB on a 4 GiB instance bounds nothing."""
        cpus, mem_max = episode_limits()
        host_cpus = os.cpu_count() or 1

        assert 1 <= cpus <= min(EPISODE_CPUS, host_cpus)
        gib = int(mem_max.removesuffix("G"))
        assert 1 <= gib <= EPISODE_MEMORY_GIB
        assert gib * 4 <= loop.host_memory_gib() or gib == 1
