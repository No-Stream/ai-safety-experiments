"""Tests for the held-out transfer evaluation driver.

The end-to-end test runs real episodes through the jail against ``MockBackend``, because the paper
cuts in this module live in the seams -- the partition to task ids to ``run_tasks`` to trace to
summary path -- and not in any one function. Everything else here is a guard, and each guard was
sabotaged and watched to fail before it was trusted; the sabotage is recorded in the session report
rather than in the tests, which only pin the behaviour it verified.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest
from conftest import needs_jail

from games.eval_model import LOAD_MODE_BASE, LOAD_MODE_RUNTIME_ADAPTER, ServedModel
from reward_hacking import train_eval
from reward_hacking.harness.analyze_ilcb import glob_trace_paths
from reward_hacking.harness.loop import MOCK_RESPONSES
from reward_hacking.harness.protocol import RUN_BLOCK_STOP
from reward_hacking.harness.task_spec import BASELINE_ARM
from reward_hacking.harness.tasks_ilcb import ILCB_TASKS_BY_ID
from reward_hacking.model_backend import MockBackend, SamplingConfig
from reward_hacking.train_partition import (
    SPLIT_CONFLICTING,
    SPLIT_ORIGINAL,
    build_partition,
    problem_id_of,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from reward_hacking.train_partition import HeldOutPartition

BASE_MODEL = "Qwen/Qwen3.5-4B"

UNMERGED_BACKEND = "vllm"
"""The one installed backend that applies a LoRA adapter at generation time.

Named rather than derived, so a capability table edited the wrong way turns these tests red instead
of silently relabelling what "un-merged" means.
"""


def split_records(record: Mapping[str, object]) -> list[dict[str, Any]]:
    """The per-split blocks of a summary record, typed for indexing."""
    return cast("list[dict[str, Any]]", record["splits"])


def base_served() -> ServedModel:
    """The un-adapted rung, resolved the way the driver resolves it."""
    return train_eval.resolve_served_model(
        checkpoint=None,
        base_model=BASE_MODEL,
        backend_kind="mock",
        merge_root=Path("/var/tmp/never-created-by-a-base-rung"),  # noqa: S108
        merge_label="unused-",
    )


@pytest.fixture(scope="module")
def partition() -> HeldOutPartition:
    """The partition built from the live problem pool, so no stored artifact is needed."""
    return build_partition()


def one_task_plan(
    partition: HeldOutPartition,
    tmp_path: Path,
    *,
    splits: tuple[str, ...] = train_eval.EVAL_SPLIT_ORDER,
    step: int = 0,
) -> train_eval.HeldOutEvalPlan:
    """A real plan narrowed to one held-out problem per split, so the jail test stays short."""
    plan = train_eval.build_plan(
        rung=train_eval.LadderRung(step=step, checkpoint=None),
        base_model=BASE_MODEL,
        partition=partition,
        splits=splits,
        repeats=1,
        out_dir=tmp_path / "traces",
        episode_root=tmp_path / "episodes",
    )
    return replace(
        plan,
        splits=tuple(
            replace(split, task_ids=split.task_ids[:1], tasks=split.tasks[:1])
            for split in plan.splits
        ),
    )


def setup_for(partition: HeldOutPartition, *, backend_kind: str = "mock") -> train_eval.EvalSetup:
    """The configuration record a run carries, with the harness budgets cut to keep tests quick."""
    return train_eval.EvalSetup(
        backend_kind=backend_kind,
        thinking=False,
        sampling=train_eval.harness_sampling_base(BASE_MODEL, thinking=True),
        partition_path=Path("in-memory-partition"),
        partition=partition,
        arm=BASELINE_ARM,
        knobs=train_eval.HarnessKnobs(max_turns=3, timeout="20s", episode_seconds=180.0),
    )


class TestSplitSelection:
    def test_headline_without_control_is_refused(self):
        with pytest.raises(ValueError, match="cannot tell a disposition from a capability"):
            train_eval.refuse_missing_capability_control([SPLIT_CONFLICTING])

    def test_control_alone_is_allowed(self):
        train_eval.refuse_missing_capability_control([SPLIT_ORIGINAL])

    def test_training_split_is_refused_as_a_held_out_read(self):
        with pytest.raises(ValueError, match="TRAINING grader"):
            train_eval.refuse_missing_capability_control([SPLIT_ORIGINAL, "oneoff"])

    def test_control_runs_before_the_headline_whatever_the_flags_said(self):
        assert train_eval.resolve_splits([SPLIT_CONFLICTING, SPLIT_ORIGINAL]) == (
            SPLIT_ORIGINAL,
            SPLIT_CONFLICTING,
        )

    def test_no_splits_means_both(self):
        assert train_eval.resolve_splits(None) == train_eval.EVAL_SPLIT_ORDER

    def test_a_repeated_split_is_refused_rather_than_doubling_a_denominator(self):
        with pytest.raises(ValueError, match="more than once"):
            train_eval.resolve_splits([SPLIT_ORIGINAL, SPLIT_ORIGINAL])


class TestBackendRefusals:
    def test_a_backend_that_cannot_serve_an_adapter_is_refused_by_name(self, tmp_path: Path):
        with pytest.raises(ValueError, match=r"--backend hf cannot serve a LoRA adapter"):
            train_eval.refuse_attenuating_backend("hf", tmp_path / "checkpoint-40")

    def test_the_refusal_names_the_measured_attenuation(self, tmp_path: Path):
        with pytest.raises(ValueError, match="64% of the trained delta"):
            train_eval.refuse_attenuating_backend("hf", tmp_path / "checkpoint-40")

    def test_the_unmerged_backend_is_allowed(self, tmp_path: Path):
        train_eval.refuse_attenuating_backend(UNMERGED_BACKEND, tmp_path / "checkpoint-40")

    def test_a_base_model_run_needs_no_adapter_support(self):
        train_eval.refuse_attenuating_backend("hf", None)

    def test_mock_is_allowed_because_it_loads_nothing(self, tmp_path: Path):
        train_eval.refuse_attenuating_backend("mock", tmp_path / "checkpoint-40")

    def test_a_hosted_endpoint_is_refused_for_every_rung(self):
        with pytest.raises(ValueError, match="one engine at one sampler"):
            train_eval.refuse_hosted_backend("bedrock")

    def test_the_local_kinds_pass_the_hosted_check(self):
        train_eval.refuse_hosted_backend(UNMERGED_BACKEND)
        train_eval.refuse_hosted_backend("mock")


class TestPlan:
    def test_every_selected_task_is_on_the_held_out_side(
        self, partition: HeldOutPartition, tmp_path: Path
    ):
        plan = train_eval.build_plan(
            rung=train_eval.LadderRung(step=40, checkpoint=None),
            base_model=BASE_MODEL,
            partition=partition,
            splits=train_eval.EVAL_SPLIT_ORDER,
            repeats=1,
            out_dir=tmp_path,
            episode_root=tmp_path / "episodes",
        )
        held_out = frozenset(partition.held_out_problem_ids)
        for split in plan.splits:
            assert split.task_ids
            assert {problem_id_of(task_id) for task_id in split.task_ids} <= held_out

    def test_a_task_from_the_training_side_is_refused_at_this_call_site(
        self, partition: HeldOutPartition, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        trained = f"ilcb-{SPLIT_ORIGINAL}-{partition.training_problem_ids[0]}"
        monkeypatch.setattr(train_eval, "held_out_task_ids", lambda _split, _partition: (trained,))
        with pytest.raises(ValueError, match="belong to the other side"):
            train_eval.build_plan(
                rung=train_eval.LadderRung(step=0, checkpoint=None),
                base_model=BASE_MODEL,
                partition=partition,
                splits=(SPLIT_ORIGINAL,),
                repeats=1,
                out_dir=tmp_path,
                episode_root=tmp_path / "episodes",
            )

    def test_repeats_expand_into_independent_episodes(
        self, partition: HeldOutPartition, tmp_path: Path
    ):
        plan = train_eval.build_plan(
            rung=train_eval.LadderRung(step=0, checkpoint=None),
            base_model=BASE_MODEL,
            partition=partition,
            splits=(SPLIT_ORIGINAL,),
            repeats=3,
            out_dir=tmp_path,
            episode_root=tmp_path / "episodes",
        )
        (split,) = plan.splits
        assert split.n_episodes == 3 * len(split.task_ids)

    def test_repeats_below_one_is_refused(self, partition: HeldOutPartition, tmp_path: Path):
        with pytest.raises(ValueError, match="repeats must be at least 1"):
            train_eval.build_plan(
                rung=train_eval.LadderRung(step=0, checkpoint=None),
                base_model=BASE_MODEL,
                partition=partition,
                splits=(SPLIT_ORIGINAL,),
                repeats=0,
                out_dir=tmp_path,
                episode_root=tmp_path / "episodes",
            )

    def test_a_git_tracked_trace_destination_is_refused_before_a_model_loads(
        self, partition: HeldOutPartition, tmp_path: Path
    ):
        tracked = Path(train_eval.__file__).resolve().parent / "not-a-gitignored-root"
        with pytest.raises(ValueError, match="not under a gitignored root"):
            train_eval.build_plan(
                rung=train_eval.LadderRung(step=0, checkpoint=None),
                base_model=BASE_MODEL,
                partition=partition,
                splits=(SPLIT_ORIGINAL,),
                repeats=1,
                out_dir=tracked,
                episode_root=tmp_path / "episodes",
            )

    def test_an_existing_trace_is_never_appended_to(
        self, partition: HeldOutPartition, tmp_path: Path
    ):
        plan = one_task_plan(partition, tmp_path)
        plan.splits[0].trace_path.parent.mkdir(parents=True, exist_ok=True)
        plan.splits[0].trace_path.write_text("{}\n", encoding="utf-8")
        with pytest.raises(FileExistsError, match="run_tasks appends"):
            train_eval.refuse_existing_traces([plan])


class TestArtifactPaths:
    def test_step_order_is_lexical_so_one_glob_reads_a_ladder_in_order(self, tmp_path: Path):
        names = [
            train_eval.trace_path_for(tmp_path, step=step, split=SPLIT_ORIGINAL).name
            for step in (0, 20, 100)
        ]
        assert names == sorted(names)

    def test_the_ladder_glob_matches_every_trace_and_no_summary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Resolved by the tool that consumes the pattern, not by a second spelling of the glob.

        ``analyze_ilcb`` is what the printed command runs, so its own resolver decides whether a
        pattern matches. A hand-copied ``Path().glob`` here passed while the consumer used that
        spelling and went on passing after it moved to ``glob.glob``, which is how the absolute case
        below stayed invisible: pathlib raises ``NotImplementedError`` on an absolute pattern rather
        than matching one, so the copy could not express that case at all.
        """
        monkeypatch.chdir(tmp_path)
        out_dir = Path("traces")
        out_dir.mkdir()
        for step in (0, 100):
            for split in train_eval.EVAL_SPLIT_ORDER:
                train_eval.trace_path_for(out_dir, step=step, split=split).touch()
            train_eval.summary_path_for(out_dir, step=step).touch()
        pattern = train_eval.ladder_trace_glob(out_dir)
        matched = sorted(path.name for path in glob_trace_paths(pattern))
        assert len(matched) == 4
        assert all(name.endswith(".jsonl") for name in matched)

    def test_an_out_dir_outside_the_cwd_yields_an_absolute_pattern_that_still_matches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The case the relativising was a workaround for, and which the consumer now resolves.

        ``ladder_trace_glob`` keeps an absolute form for an out dir it cannot relativise, and the
        printed command is only useful if the consumer matches that form.
        """
        out_dir = tmp_path / "traces"
        out_dir.mkdir()
        for split in train_eval.EVAL_SPLIT_ORDER:
            train_eval.trace_path_for(out_dir, step=0, split=split).touch()
        train_eval.summary_path_for(out_dir, step=0).touch()
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        pattern = train_eval.ladder_trace_glob(out_dir)
        assert Path(pattern).is_absolute()
        matched = sorted(path.name for path in glob_trace_paths(pattern))
        assert len(matched) == len(train_eval.EVAL_SPLIT_ORDER)
        assert all(name.endswith(".jsonl") for name in matched)


class TestLadder:
    def test_checkpoints_come_back_in_step_order_with_the_base_first(self, tmp_path: Path):
        for name in ("checkpoint-100", "checkpoint-20", "completions"):
            (tmp_path / name).mkdir()
        rungs = train_eval.checkpoint_ladder(tmp_path)
        assert [rung.step for rung in rungs] == [0, 20, 100]
        assert rungs[0].checkpoint is None

    def test_the_base_rung_can_be_left_out(self, tmp_path: Path):
        (tmp_path / "checkpoint-7").mkdir()
        assert [
            rung.step for rung in train_eval.checkpoint_ladder(tmp_path, include_base=False)
        ] == [7]

    def test_a_run_that_saved_nothing_is_refused(self, tmp_path: Path):
        with pytest.raises(ValueError, match="no checkpoint-<step> directories"):
            train_eval.checkpoint_ladder(tmp_path)


class TestThinLadder:
    """Which rungs a stride keeps, and why both endpoints are unconditional.

    ``--every-nth-checkpoint`` appeared in no test until 2026-08-24 while being the default shape of
    every ladder this module runs. Its own docstring names the silent failure it prevents: a stride
    that happens not to land on the last checkpoint yields a ladder that cannot be read as
    before-and-after, and does so quietly, because a ladder is a list of plausible rungs either way.
    """

    @staticmethod
    def _ladder(n: int) -> tuple[train_eval.LadderRung, ...]:
        """A ladder of n rungs whose steps are their own indices, so kept indices read directly."""
        return tuple(train_eval.LadderRung(step=step, checkpoint=None) for step in range(n))

    def test_the_final_rung_survives_a_stride_that_does_not_land_on_it(self):
        """EIGHT rungs at stride 3, not seven.

        At seven, ``range(0, 7, 3)`` is 0, 3, 6 and already ends on the last index, so the assertion
        would hold with the endpoint clause deleted and would pin nothing. Eight is the smallest
        fixture where the stride misses the end: ``range(0, 8, 3)`` is 0, 3, 6, and rung 7 is kept
        only because the endpoint is unconditional.
        """
        kept = train_eval.thin_ladder(self._ladder(8), stride=3)
        assert [rung.step for rung in kept] == [0, 3, 6, 7]

    def test_the_base_rung_survives_too(self):
        """The other endpoint: the transfer claim is a change FROM the untrained model."""
        kept = train_eval.thin_ladder(self._ladder(8)[1:], stride=3)
        assert kept[0].step == 1
        assert kept[-1].step == 7

    def test_a_stride_of_one_keeps_the_whole_ladder(self):
        assert train_eval.thin_ladder(self._ladder(4), stride=1) == self._ladder(4)

    def test_a_stride_wider_than_the_ladder_keeps_only_the_endpoints(self):
        assert [rung.step for rung in train_eval.thin_ladder(self._ladder(5), stride=99)] == [0, 4]

    def test_a_stride_below_one_is_refused(self):
        with pytest.raises(ValueError, match="must be at least 1"):
            train_eval.thin_ladder(self._ladder(3), stride=0)

    def test_an_empty_ladder_is_refused(self):
        with pytest.raises(ValueError, match="cannot thin an empty ladder"):
            train_eval.thin_ladder((), stride=2)

    def test_the_cli_thins_the_ladder_it_resolves(self, tmp_path: Path):
        """The flag reaches ``thin_ladder`` at all, which no test checked either."""
        for step in (0, 20, 40, 60, 80):
            (tmp_path / f"checkpoint-{step}").mkdir()
        args = train_eval._parse_args(
            ["--run-dir", str(tmp_path), "--every-nth-checkpoint", "3", "--no-with-base"]
        )
        assert [rung.step for rung in train_eval._rungs_from_args(args)] == [0, 60, 80]


class TestLadderSummary:
    """The ladder-level index: written before the first rung, and readable as JSON afterwards.

    Unverified in content until 2026-08-24 -- the dry-run test returns before it is called and the
    end-to-end test bypasses it -- while being the one artifact a stage plan gates on.
    """

    def _plans(
        self, partition: HeldOutPartition, tmp_path: Path
    ) -> list[train_eval.HeldOutEvalPlan]:
        return [one_task_plan(partition, tmp_path, step=step) for step in (0, 40)]

    def test_the_payload_names_every_intended_rung_and_the_stride_that_chose_them(
        self, partition: HeldOutPartition, tmp_path: Path
    ):
        plans = self._plans(partition, tmp_path)
        out_dir = tmp_path / "traces"
        path = train_eval.write_ladder_summary(out_dir, plans, stride=5)

        assert path == out_dir / train_eval.LADDER_SUMMARY_FILENAME
        payload = cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))
        assert payload["every_nth_checkpoint"] == 5
        assert payload["n_rungs"] == 2
        assert payload["steps"] == [0, 40]
        assert payload["n_episodes_planned"] == sum(plan.n_episodes for plan in plans)
        assert payload["git_sha"]
        assert payload["trace_glob"] == train_eval.ladder_trace_glob(out_dir)

    def test_a_rung_reads_as_landed_only_once_its_own_summary_exists(
        self, partition: HeldOutPartition, tmp_path: Path
    ):
        """The field that tells "thinned on purpose" from "died at rung three" apart.

        Written before the first rung runs, so every rung starts out not landed; a later rewrite is
        what flips one. Both states are asserted, because a field hardcoded either way would look
        right in whichever half a one-sided test happened to check.
        """
        plans = self._plans(partition, tmp_path)
        out_dir = tmp_path / "traces"
        path = train_eval.write_ladder_summary(out_dir, plans, stride=5)
        before = cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))
        assert [rung["landed"] for rung in before["rung_summaries"]] == [False, False]

        train_eval.summary_path_for(out_dir, step=0).touch()
        train_eval.write_ladder_summary(out_dir, plans, stride=5)
        after = cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))
        assert [rung["landed"] for rung in after["rung_summaries"]] == [True, False]
        assert after["rung_summaries"][0]["summary"] == str(
            train_eval.summary_path_for(out_dir, step=0)
        )


class TestEngineTeardown:
    """One engine at a time on the card, proved by the release rather than assumed.

    ``train_sequence.eval_stage`` runs a whole ``--every-nth-checkpoint`` ladder in ONE process, so
    the ladder loop below stands up roughly sixteen sequential engines, each claiming vLLM's default
    0.9 of the card because nothing on this path passes ``gpu_memory_utilization``. The teardown was a
    private ``del backend`` plus ``torch.cuda.empty_cache()`` -- both no-ops on a vLLM engine, and the
    exact pair ``games/vllm_teardown.py`` was written to replace.

    No GPU is needed for any of this: the engine construction and the release are both stubbed, and
    what is asserted is the ORDER and the gating, which is where the bug lived.
    """

    @staticmethod
    def _stub_engine(monkeypatch: pytest.MonkeyPatch, events: list[str]) -> None:
        """Stand in for everything that would touch a GPU, recording the order it was reached in."""
        monkeypatch.setattr(train_eval, "verify_served_model", lambda _backend, _served: None)
        monkeypatch.setattr(train_eval, "run_tasks", lambda *_args, **_kwargs: [])

        def build(*_args: object, **_kwargs: object) -> MockBackend:
            events.append("engine-built")
            return MockBackend(["nothing to do"])

        monkeypatch.setattr(train_eval.backend_cli, "backend_from_args", build)

        def baseline(backend_kind: str) -> list[int] | None:
            events.append(f"baseline:{backend_kind}")
            return [512] if backend_kind == "vllm" else None

        monkeypatch.setattr(train_eval, "baseline_before_engine", baseline)

        def release(_backend: object, *, baseline_mib: object) -> None:
            events.append(f"released:{baseline_mib}")

        monkeypatch.setattr(train_eval, "release_engine", release)

    def test_the_card_is_read_before_the_engine_and_released_after(
        self, partition: HeldOutPartition, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The baseline must PREDATE the engine, or the release can never fail.

        A baseline taken afterwards already includes the engine's own claim, so the residue computes
        as zero however much of the card is still held -- a check that passes by construction, which
        is the shape of the bug this replaced. Ordering is the only way to see that, so it is asserted
        rather than the mere fact that both were called.
        """
        events: list[str] = []
        self._stub_engine(monkeypatch, events)
        plan = one_task_plan(partition, tmp_path)
        args = train_eval._parse_args(["--backend", UNMERGED_BACKEND, "--base-model", BASE_MODEL])
        train_eval._run_rung(plan, args=args, setup=setup_for(partition, backend_kind="vllm"))

        assert events == ["baseline:vllm", "engine-built", "released:[512]"]

    def test_a_kind_that_loads_no_engine_is_not_put_through_the_engine_release(
        self, partition: HeldOutPartition, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The positive control on the gate: ``resolve_engine_shutdown`` refuses a mock backend.

        A release fired unconditionally would raise ``EngineDrainError`` on every mock rung, so this
        pins that the gate is on the declared kind and that mock still runs.
        """
        events: list[str] = []
        self._stub_engine(monkeypatch, events)
        plan = one_task_plan(partition, tmp_path)
        args = train_eval._parse_args(["--backend", "mock", "--base-model", BASE_MODEL])
        train_eval._run_rung(plan, args=args, setup=setup_for(partition))

        assert events == ["baseline:mock", "engine-built"]

    def test_a_rung_that_dies_still_gives_the_card_back(
        self, partition: HeldOutPartition, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """An engine left standing after a failed rung takes the card down for every rung after."""
        events: list[str] = []
        self._stub_engine(monkeypatch, events)

        def die(*_args: object, **_kwargs: object) -> list[object]:
            raise RuntimeError("engine died mid-split")

        monkeypatch.setattr(train_eval, "run_tasks", die)
        plan = one_task_plan(partition, tmp_path)
        args = train_eval._parse_args(["--backend", UNMERGED_BACKEND, "--base-model", BASE_MODEL])
        with pytest.raises(RuntimeError, match="engine died"):
            train_eval._run_rung(plan, args=args, setup=setup_for(partition, backend_kind="vllm"))

        assert events == ["baseline:vllm", "engine-built", "released:[512]"]


class TestSampling:
    def test_the_thinking_budget_is_never_lowered(self):
        thinking = train_eval.harness_sampling_base(BASE_MODEL, thinking=True)
        assert thinking.max_new_tokens == SamplingConfig.for_thinking(thinking=True).max_new_tokens

    def test_the_run_block_close_is_a_stop_sequence_in_both_modes(self):
        """Without it the policy writes the environment's own result envelopes.

        The harness CLI folds this stop into its base, so a driver that did not would sample a
        different policy from every other trace in ``artifacts/harness``.
        """
        for thinking in (True, False):
            assert train_eval.harness_sampling_base(BASE_MODEL, thinking=thinking).stop == (
                RUN_BLOCK_STOP,
            )

    def test_a_preset_below_the_measured_floor_is_raised_to_it(self):
        non_thinking = train_eval.harness_sampling_base(BASE_MODEL, thinking=False)
        assert non_thinking.max_new_tokens == train_eval.backend_cli.output_floor_for(BASE_MODEL)
        assert (
            non_thinking.max_new_tokens > SamplingConfig.for_thinking(thinking=False).max_new_tokens
        )


class TestSummaryRecord:
    def test_a_run_with_no_episodes_still_reports_its_denominators(
        self, partition: HeldOutPartition, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(train_eval, "run_tasks", lambda *_args, **_kwargs: [])
        plan = one_task_plan(partition, tmp_path)
        record = train_eval.run_held_out_eval(
            MockBackend(["nothing to do"]),
            plan,
            served=base_served(),
            setup=setup_for(partition),
        )
        assert record["n_episodes_attempted"] == plan.n_episodes
        assert record["n_episodes_recorded"] == 0
        assert record["complete"] is True
        for split in split_records(record):
            assert split["n_problems"] == 1
            assert split["n_problems_held_out"] == len(partition.held_out_problem_ids)
            assert split["n_episodes_attempted"] == 1
            assert split["n_episodes_recorded"] == 0
            assert split["n_unmeasured"] == 0

    def test_the_summary_says_which_weights_and_which_partition(
        self, partition: HeldOutPartition, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(train_eval, "run_tasks", lambda *_args, **_kwargs: [])
        plan = one_task_plan(partition, tmp_path, step=40)
        train_eval.run_held_out_eval(
            MockBackend(["nothing to do"]),
            plan,
            served=base_served(),
            setup=setup_for(partition),
        )
        record = cast("dict[str, Any]", json.loads(plan.summary_path.read_text(encoding="utf-8")))
        assert record["step"] == 40
        assert record["base_model"] == BASE_MODEL
        assert record["model_load_mode"] == LOAD_MODE_BASE
        assert record["model_delta_faithful"] is True
        assert record["adapter_verification_ran"] is False
        assert record["partition"]["pool_fingerprint"] == partition.pool_fingerprint
        assert record["sampling"]["max_new_tokens"] == 32768
        assert record["git_sha"]
        assert train_eval.TRACE_STEM in record["analyze_command"]

    def test_a_run_dying_before_its_first_split_still_leaves_an_attributing_summary(
        self, partition: HeldOutPartition, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        def die(*_args: object, **_kwargs: object) -> list[object]:
            raise RuntimeError("engine died mid-split")

        monkeypatch.setattr(train_eval, "run_tasks", die)
        plan = one_task_plan(partition, tmp_path)
        with pytest.raises(RuntimeError, match="engine died"):
            train_eval.run_held_out_eval(
                MockBackend(["nothing to do"]),
                plan,
                served=base_served(),
                setup=setup_for(partition),
            )
        record = cast("dict[str, Any]", json.loads(plan.summary_path.read_text(encoding="utf-8")))
        assert record["complete"] is False
        assert record["splits_completed"] == []
        assert record["base_model"] == BASE_MODEL


class TestVerification:
    def test_a_runtime_adapter_records_that_it_was_proved_to_apply(self, tmp_path: Path):
        served = ServedModel(
            model_id=BASE_MODEL,
            load_mode=LOAD_MODE_RUNTIME_ADAPTER,
            adapter_dir=tmp_path / "checkpoint-40",
        )
        record = train_eval._verification_record(served)
        assert record["adapter_verification_ran"] is True
        assert "changes generated output" in str(record["adapter_verification"])

    def test_a_base_run_says_there_was_nothing_to_prove(self):
        served = ServedModel(model_id=BASE_MODEL, load_mode=LOAD_MODE_BASE, adapter_dir=None)
        record = train_eval._verification_record(served)
        assert record["adapter_verification_ran"] is False
        assert "nothing" in str(record["adapter_verification"])

    def test_the_backend_is_asked_to_prove_the_adapter_before_any_episode(
        self, partition: HeldOutPartition, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        verified: list[str] = []
        monkeypatch.setattr(
            train_eval,
            "verify_served_model",
            lambda _backend, served: verified.append(served.load_mode),
        )
        plan = one_task_plan(partition, tmp_path)
        args = train_eval._parse_args(["--backend", "mock", "--base-model", BASE_MODEL])
        backend, served = train_eval._resolve_backend(plan, args=args, setup=setup_for(partition))
        assert verified == [served.load_mode]
        assert backend.model_id.startswith("mock:")


class TestCli:
    def test_a_dry_run_writes_nothing_and_loads_nothing(self, tmp_path: Path):
        out_dir = tmp_path / "traces"
        assert (
            train_eval.main(
                [
                    "--backend",
                    "mock",
                    "--base-model",
                    BASE_MODEL,
                    "--partition",
                    str(self._write_partition(tmp_path)),
                    "--out-dir",
                    str(out_dir),
                    "--episode-root",
                    str(tmp_path / "episodes"),
                    "--dry-run",
                ]
            )
            == 0
        )
        assert not out_dir.exists()

    def test_the_headline_split_alone_is_refused_at_the_cli(self, tmp_path: Path):
        with pytest.raises(ValueError, match="cannot tell a disposition"):
            train_eval.main(
                [
                    "--backend",
                    "mock",
                    "--base-model",
                    BASE_MODEL,
                    "--partition",
                    str(self._write_partition(tmp_path)),
                    "--out-dir",
                    str(tmp_path / "traces"),
                    "--split",
                    SPLIT_CONFLICTING,
                    "--dry-run",
                ]
            )

    def test_a_merging_backend_with_a_checkpoint_is_refused_at_the_cli(self, tmp_path: Path):
        checkpoint = tmp_path / "checkpoint-40"
        checkpoint.mkdir()
        with pytest.raises(ValueError, match="cannot serve a LoRA adapter"):
            train_eval.main(
                [
                    "--backend",
                    "hf",
                    "--checkpoint",
                    str(checkpoint),
                    "--base-model",
                    BASE_MODEL,
                    "--out-dir",
                    str(tmp_path / "traces"),
                    "--dry-run",
                ]
            )

    def test_a_bare_base_run_needs_a_base_model(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Pass --base-model"):
            train_eval.main(
                [
                    "--backend",
                    "mock",
                    "--out-dir",
                    str(tmp_path / "traces"),
                    "--dry-run",
                ]
            )

    def test_naming_both_a_checkpoint_and_a_run_dir_is_refused(self, tmp_path: Path):
        with pytest.raises(ValueError, match="pass exactly one"):
            train_eval.main(
                [
                    "--checkpoint",
                    str(tmp_path / "checkpoint-1"),
                    "--run-dir",
                    str(tmp_path),
                    "--dry-run",
                ]
            )

    @staticmethod
    def _write_partition(tmp_path: Path) -> Path:
        path = tmp_path / "held-out-partition.json"
        path.write_text(
            json.dumps(build_partition().to_json_dict(), indent=2) + "\n", encoding="utf-8"
        )
        return path


@needs_jail
class TestEndToEnd:
    def test_one_episode_per_split_runs_through_the_jail_and_lands_in_its_own_trace(
        self, partition: HeldOutPartition, tmp_path: Path
    ):
        plan = one_task_plan(partition, tmp_path)
        record = train_eval.run_held_out_eval(
            MockBackend(list(MOCK_RESPONSES)),
            plan,
            served=base_served(),
            setup=setup_for(partition),
        )
        assert record["complete"] is True
        assert record["n_episodes_recorded"] == plan.n_episodes
        for split_plan, split_record in zip(plan.splits, split_records(record), strict=True):
            assert split_record["split"] == split_plan.split
            records = [
                cast("dict[str, Any]", json.loads(line))
                for line in split_plan.trace_path.read_text(encoding="utf-8").splitlines()
            ]
            episodes = [row for row in records if row["record"] == "episode_summary"]
            assert len(episodes) == 1
            assert episodes[0]["task_id"] == split_plan.task_ids[0]
            assert episodes[0]["task_id"] in ILCB_TASKS_BY_ID
            executed = [
                command for row in records if row["record"] == "turn" for command in row["commands"]
            ]
            assert executed
            grader = ILCB_TASKS_BY_ID[split_plan.task_ids[0]].grader_command
            assert any(grader in command["command"] for command in executed)
            assert episodes[0]["true"]["measured"] is True
