"""Feed every stage's rendered argv to the module that stage invokes, using that module's own parser.

This is the test that catches a whole class mechanically rather than one bug at a time. A stage plan
renders a command as a tuple of strings and the stage runner executes it hours later on a rented box,
so a flag that was renamed upstream, or never existed, surfaces as ``unrecognized arguments`` at the
moment the command runs -- which for the evaluation stage is AFTER the entire training run has been
paid for. Two real defects were sitting here: the evaluation stage passed ``--splits a,b`` where the
parser defines ``--split`` singular and repeatable, and it passed an ``--every-nth-checkpoint`` that
existed nowhere in the repository.

Every argv also has to name a target the plan promises an artifact for, and the promise has to be a
path the target actually writes -- the third defect, where the stage promised ``eval_summary.json``
while the evaluator wrote only per-rung summaries, so the runner would have refused the stage after
the ladder had run.

Offline and CPU-only: parsers are called directly, nothing is executed, no model loads.
"""

from __future__ import annotations

import json
import shlex
from itertools import combinations
from typing import TYPE_CHECKING, ClassVar

import pytest

from games.plans import cap_hours
from reward_hacking import train as rh_train
from reward_hacking import train_eval, train_screen, train_sequence
from reward_hacking.train import MEASURED_MINUTES_PER_STEP
from reward_hacking.train_dataset import (
    ARM_LEGIBLE_SUBSET,
    ARM_MISSPECIFIED,
    SOLUTION_PARSER,
    TRAINABLE_ARMS,
)
from reward_hacking.train_partition import SPLIT_SUBSET3_STRATIFIED

if TYPE_CHECKING:
    from pathlib import Path

    from games.stage_runner import Stage

# What the 2026-08-24 timing probe projects for a 70-step arm: 32.7 hours at the measured 28.05
# minutes per step. Derived from the constant the trainer's own refusal cites rather than written down
# again, so a re-measurement moves the floor with it.
MEASURED_ARM_HOURS = 70 * MEASURED_MINUTES_PER_STEP / 60.0
# The floor a cap must clear, and deliberately the WORST case rather than that median -- matching how
# `games/tests/test_games_arm_sequence.py` pins `MEASURED_WORST_CASE_ARM_HOURS`. A reclaim was observed
# at roughly one every 35 minutes, and each one costs the interrupted step plus a fresh bootstrap and
# an S3 restore, so a reclaim-heavy arm runs well past the median projection. A median-derived floor
# would let a cap of ~36 h pass this guard while still killing such an arm at its most expensive point.
MEASURED_WORST_CASE_ARM_HOURS = 45.0
# One arm's gradient screen, read off the completed run's stage boundaries (an earlier ~65-minute
# mid-sweep extrapolation was wrong in the other direction).
MEASURED_SCREEN_HOURS_PER_ARM = 74.5 / 60.0
# How much headroom a screen cap needs over that measurement. Wider than an arm's proportionally,
# because `screen_timeout`'s own docstring records that a screen hitting its cap cannot be resumed,
# only redone -- so a kill costs the whole sweep rather than one step. The superseded 90m gave 1.21x,
# and a slower tail crossed it on 2026-08-24.
SCREEN_CAP_HEADROOM = 1.5

# The wrapper every stage argv begins with: `timeout <cap> uv run --frozen python -m <module> ...`.
# Stripped before a parser sees it, and asserted on, because a stage that lost its cap would bill
# overnight and a stage that lost `--frozen` would resolve a different environment than the lockfile.
_UV_PREFIX = ("uv", "run", "--frozen", "python", "-m")

# Which module each stage invokes, and the parser to hold its argv to. `analyze_ilcb` is another
# agent's module and its parser is private, so it is checked structurally instead -- see the test.
STAGE_PARSERS = {
    "screen": (train_screen.__name__, train_screen._parse_args),
    "eval": (train_eval.__name__, train_eval._parse_args),
}


@pytest.fixture(autouse=True)
def clean_plan_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the plan at a scratch artifact root, so no test reads the real artifact tree.

    Autouse, and named without a leading underscore to match every other autouse fixture in this
    repository (``clean_plan_environment`` in the two sibling stage-plan suites does this same job).
    The private name it had was flagged unused by basedpyright, which cannot see that pytest calls an
    autouse fixture -- and a private-and-unused fixture is indistinguishable from an unwired one, so
    the convention is what keeps "this setup is running" readable.
    """
    monkeypatch.setenv("RH_OPTION3_ARM", ARM_MISSPECIFIED)
    monkeypatch.setenv("RH_OPTION3_S3_BASE", "s3://example-bucket/option3")
    for name, directory in (
        ("SCREEN_DIR", tmp_path / "screen"),
        ("EVAL_DIR", tmp_path / "eval"),
        ("ANALYSIS_DIR", tmp_path / "analysis"),
        ("LOG_DIR", tmp_path / "logs"),
    ):
        monkeypatch.setattr(train_sequence, name, directory)


def _arm_module_and_argv(stage: Stage) -> tuple[str, list[str]]:
    """Strip the timeout and uv wrapper off a stage argv, returning its module and its arguments."""
    argv = list(stage.argv)
    assert argv[0] == "timeout", f"{stage.name} lost its wall-clock cap: {argv[:2]}"
    assert argv[1].endswith(("s", "m", "h")), f"{stage.name} has a malformed cap: {argv[1]!r}"
    body = argv[2:]
    assert tuple(body[: len(_UV_PREFIX)]) == _UV_PREFIX, f"{stage.name} argv: {body[:6]}"
    return body[len(_UV_PREFIX)], body[len(_UV_PREFIX) + 1 :]


def write_screen_verdict(
    artifact: Path,
    *,
    solution_parser: str | None = SOLUTION_PARSER,
    grader_exposure: str | None = None,
) -> None:
    """Write a screen verdict the launch gate accepts, or one it must refuse.

    ``solution_parser=None`` writes the shape every verdict produced before the parser change has: no
    such key at all. That is the case the gate has to treat as pre-fix rather than as "no opinion",
    and it is the shape of the two verdicts sitting in the live run's S3 prefix.

    ``grader_exposure=None`` likewise omits that key, which is every verdict written before the
    screen grew its ``--grader-exposure`` flag -- necessarily measured inline, so the gate accepts
    the absence.
    """
    artifact.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, object] = {"kind": "gradient-screen"}
    if solution_parser is not None:
        record["solution_parser"] = solution_parser
    if grader_exposure is not None:
        record["grader_exposure"] = grader_exposure
    artifact.write_text(json.dumps(record), encoding="utf-8")


def _stage(name: str) -> Stage:
    """Build one stage in isolation, writing the screen artifact first when the gate needs it."""
    if name != "screen":
        write_screen_verdict(train_sequence.screen_artifact())
    return train_sequence.STAGE_BUILDERS[name]()


class TestEveryStageArgvParses:
    """The whole point: a rendered command must be accepted by the parser that will receive it."""

    @pytest.mark.parametrize("stage_name", sorted(STAGE_PARSERS))
    def test_the_target_module_accepts_the_argv_the_plan_renders(self, stage_name: str) -> None:
        module, argv = _arm_module_and_argv(_stage(stage_name))
        expected_module, parse = STAGE_PARSERS[stage_name]
        assert module == expected_module
        # SystemExit is what argparse raises on an unrecognized or malformed argument; letting it
        # escape as a failure here is the entire value of this test.
        parse(argv)

    def test_the_training_stage_argv_builds_a_config(self) -> None:
        """The arm stage's argv is held to `reward_hacking.train`'s own parser AND its validation."""
        module, argv = _arm_module_and_argv(_stage("arm"))
        assert module == rh_train.__name__
        config = rh_train._parse_args(argv)
        assert config.arm == ARM_MISSPECIFIED
        # The retention and estimator settings the plan is responsible for carrying, read back off
        # the config rather than off the string, so a flag that parsed but bound nothing still fails.
        assert (config.save_steps, config.save_total_limit) == (1, 0)
        assert config.vllm_colocate is True
        assert config.vllm_importance_sampling_correction is False
        assert config.resume_from_checkpoint == rh_train.RESUME_LATEST
        assert config.s3_dest.endswith(config.arm_tag)

    def test_the_analysis_stage_names_a_module_and_a_glob(self) -> None:
        """Checked structurally: `analyze_ilcb` is another agent's module with a private parser."""
        module, argv = _arm_module_and_argv(_stage("analyze"))
        assert module.endswith("analyze_ilcb")
        assert "--glob" in argv
        assert "--out" in argv


class TestEveryStagePromisesSomethingItsTargetWrites:
    """A promised artifact the target never writes fails the stage after the work is done."""

    def test_the_eval_stage_promises_the_ladder_index_the_evaluator_writes(self) -> None:
        stage = _stage("eval")
        promised = {path.name for path in stage.artifacts}
        assert promised == {train_eval.LADDER_SUMMARY_FILENAME}

    def test_the_eval_stage_writes_its_promise_into_its_own_out_dir(self) -> None:
        """The promise and the ``--out-dir`` must agree, or the runner looks in the wrong place."""
        stage = _stage("eval")
        _, argv = _arm_module_and_argv(stage)
        out_dir = argv[argv.index("--out-dir") + 1]
        assert stage.artifacts[0].parent == train_sequence.Path(out_dir)

    @pytest.mark.parametrize("stage_name", ["screen", "arm", "eval", "analyze"])
    def test_every_stage_promises_at_least_one_artifact(self, stage_name: str) -> None:
        assert _stage(stage_name).artifacts


class TestTheEvalStageCarriesBothSplits:
    def test_the_splits_are_rendered_as_the_repeatable_singular_flag(self) -> None:
        """The defect this file was written for: ``--splits a,b`` is not a flag that exists."""
        _, argv = _arm_module_and_argv(_stage("eval"))
        assert "--splits" not in argv
        assert argv.count("--split") == len(train_sequence.EVAL_SPLITS)

    def test_the_capability_control_is_never_omitted(self) -> None:
        """A hack rate with no honest-solve rate beside it cannot be told from better coding."""
        _, argv = _arm_module_and_argv(_stage("eval"))
        named = {argv[index + 1] for index, token in enumerate(argv) if token == "--split"}
        assert named == set(train_sequence.EVAL_SPLITS)
        assert train_eval.CAPABILITY_CONTROL_SPLIT in named


class TestTheRenderedPlanIsReadable:
    def test_describe_plan_names_every_stage_and_no_argument_needs_word_splitting(self) -> None:
        """The rendered plan must be readable, and no argument may hide a word break.

        Deliberately NOT "nothing needs shell quoting": the analysis stage's ``--glob`` carries a
        ``*`` on purpose, which is correct because the stage runner executes an argv list through
        subprocess where no shell expands it -- and it is exactly why a reader copying the printed
        command into a terminal has to quote that one argument. Whitespace and quote characters are
        the real hazard, since those would silently split one argument into two in either setting.
        """
        write_screen_verdict(train_sequence.screen_artifact())
        rendered = train_sequence.describe_plan()
        for stage_name in ("screen", "arm", "eval", "analyze"):
            assert stage_name in rendered
        for stage in train_sequence.stages():
            for argument in stage.argv:
                assert argument == argument.strip(), argument
                assert not set(argument) & set(" \t\n'\""), argument
        globbed = [
            argument
            for stage in train_sequence.stages()
            for argument in stage.argv
            if shlex.quote(argument) != argument
        ]
        assert all("*" in argument for argument in globbed), globbed

    def test_every_trainable_arm_renders_disjoint_artifact_paths(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two arms sharing any artifact path is the collision the S3 refusal exists for."""
        paths: dict[str, set[str]] = {}
        for arm in TRAINABLE_ARMS:
            monkeypatch.setenv("RH_OPTION3_ARM", arm)
            write_screen_verdict(train_sequence.screen_artifact())
            paths[arm] = {
                str(path) for stage in train_sequence.stages() for path in stage.artifacts
            }
        for arm, other in combinations(TRAINABLE_ARMS, 2):
            assert paths[arm].isdisjoint(paths[other]), (arm, other)


class TestEveryTrainableArmIsLaunchable:
    """The plan accepts exactly the arms the corpus machinery can build, and refuses the rest.

    Until 2026-08-27 both entry points gated on the frozen flagship pair, which was correct while
    legible-subset training was ungated and wrong afterwards: the corpus, the screen and the reward
    all handled the arm while the two launchers refused to run it.
    """

    def test_the_legible_subset_arm_renders_a_full_plan(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RH_OPTION3_ARM", ARM_LEGIBLE_SUBSET)
        assert train_sequence.arm_name() == ARM_LEGIBLE_SUBSET
        rendered = train_sequence.describe_plan()
        assert ARM_LEGIBLE_SUBSET in rendered
        assert SPLIT_SUBSET3_STRATIFIED in rendered

    def test_the_legible_subset_arm_stage_argv_builds_a_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The rendered command must survive the trainer's own parser AND its validation."""
        monkeypatch.setenv("RH_OPTION3_ARM", ARM_LEGIBLE_SUBSET)
        module, argv = _arm_module_and_argv(_stage("arm"))
        assert module == rh_train.__name__
        config = rh_train._parse_args(argv)
        assert config.arm == ARM_LEGIBLE_SUBSET
        assert config.split == SPLIT_SUBSET3_STRATIFIED
        assert config.s3_dest.endswith(config.arm_tag)

    def test_an_unknown_arm_is_still_refused_and_the_menu_names_every_arm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sabotage: the likeliest slip is naming the SPLIT where the arm belongs."""
        monkeypatch.setenv("RH_OPTION3_ARM", SPLIT_SUBSET3_STRATIFIED)
        with pytest.raises(ValueError, match="RH_OPTION3_ARM") as caught:
            train_sequence.arm_name()
        for arm in TRAINABLE_ARMS:
            assert arm in str(caught.value)


class TestEveryStageCapIsOverridableFromTheEnvironment:
    """A cap that needs a code edit to raise is the one knob an operator wants at 3am.

    Live case, 2026-08-24: a screen stage was running against a 90-minute cap on rented capacity with
    no way to raise it from the launch environment. The two workarounds reachable without this were
    both worse than the wall -- cutting the sample count fits the cap by changing the statistic, since
    ``frac_groups_pure`` at the training group size IS the measurement, and re-invoking the stage from
    a wrapper script duplicates the stage runner's artifact verification.
    """

    CAPS: ClassVar[tuple[tuple[str, str, str], ...]] = (
        ("screen", "SCREEN_TIMEOUT", "240m"),
        ("arm", "ARM_TIMEOUT", "48h"),
        ("eval", "EVAL_TIMEOUT", "60h"),
        ("analyze", "ANALYZE_TIMEOUT", "90m"),
    )

    @pytest.mark.parametrize(("stage_name", "variable", "override"), CAPS)
    def test_the_override_reaches_the_rendered_argv(
        self, stage_name: str, variable: str, override: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(f"{train_sequence.ENV_PREFIX}{variable}", override)
        argv = list(_stage(stage_name).argv)
        assert argv[:2] == ["timeout", override]

    @pytest.mark.parametrize(("stage_name", "variable", "override"), CAPS)
    def test_the_default_stands_when_the_variable_is_unset(
        self, stage_name: str, variable: str, override: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nothing changes for anyone who does not set it."""
        del override
        monkeypatch.delenv(f"{train_sequence.ENV_PREFIX}{variable}", raising=False)
        expected = getattr(train_sequence, f"DEFAULT_{variable}")
        assert list(_stage(stage_name).argv)[:2] == ["timeout", expected]

    def test_the_caps_are_read_lazily_rather_than_frozen_at_import(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Module constants would have been frozen by whichever import came first, and would not
        even have imported: plan_setting is defined below where the caps used to sit."""
        monkeypatch.setenv(f"{train_sequence.ENV_PREFIX}SCREEN_TIMEOUT", "17m")
        assert train_sequence.screen_timeout() == "17m"
        monkeypatch.setenv(f"{train_sequence.ENV_PREFIX}SCREEN_TIMEOUT", "18m")
        assert train_sequence.screen_timeout() == "18m"


class TestWallClockCapsExceedTheMeasuredRunTime:
    """The guard whose absence let ``DEFAULT_ARM_TIMEOUT`` sit at 30h against a 32.7-hour arm.

    ``arm_timeout``'s own docstring already argued that a cap below the healthy run time is strictly
    worse than no cap -- it kills a working arm at the point where its artifacts are most expensive to
    recreate -- and nothing held the constant to it, so the correction lived only as an
    ``RH_OPTION3_ARM_TIMEOUT`` override in a gitignored launcher while the documented direct invocation
    still got 30h. Asserted against the measurement rather than against a number copied out of the
    module, so raising or lowering a cap without a measurement to justify it does not silently pass.
    """

    def test_the_arm_cap_clears_the_worst_measured_arm(self) -> None:
        assert cap_hours(train_sequence.DEFAULT_ARM_TIMEOUT) > MEASURED_WORST_CASE_ARM_HOURS

    def test_the_superseded_thirty_hour_cap_does_not_clear_it(self) -> None:
        """The in-suite proof that the assertion above can fail, and the value that made it fail.

        Without this the floor could be satisfied by any cap at all and nobody would know, which is
        this repository's standing complaint about checks nobody has watched go red. 30h fails the
        floor twice over: it is under the worst case AND under the plain median projection.
        """
        assert cap_hours("30h") < MEASURED_WORST_CASE_ARM_HOURS
        assert cap_hours("30h") < MEASURED_ARM_HOURS

    def test_the_screen_cap_clears_its_measurement_with_room_for_a_slow_tail(self) -> None:
        assert cap_hours(train_sequence.DEFAULT_SCREEN_TIMEOUT) > (
            MEASURED_SCREEN_HOURS_PER_ARM * SCREEN_CAP_HEADROOM
        )

    def test_the_superseded_ninety_minute_screen_cap_does_not_clear_it(self) -> None:
        """1.21x the measurement, on a stage a kill costs the whole sweep of. It was crossed live."""
        assert cap_hours("90m") < MEASURED_SCREEN_HOURS_PER_ARM * SCREEN_CAP_HEADROOM

    def test_the_arm_cap_is_the_longest_of_the_training_caps(self) -> None:
        """The screen is inference over one corpus; the arm is 70 optimizer steps over it."""
        assert cap_hours(train_sequence.DEFAULT_ARM_TIMEOUT) > cap_hours(
            train_sequence.DEFAULT_SCREEN_TIMEOUT
        )

    @pytest.mark.parametrize(
        "variable", ["SCREEN_TIMEOUT", "ARM_TIMEOUT", "EVAL_TIMEOUT", "ANALYZE_TIMEOUT"]
    )
    def test_every_default_cap_is_a_duration_timeout_understands(self, variable: str) -> None:
        """``timeout(1)`` rejects a bare number and ``72hr`` outright, hours into a rental."""
        assert cap_hours(getattr(train_sequence, f"DEFAULT_{variable}")) > 0


class TestTheStageSelectionIsReorderedIntoDependencyOrder:
    """The order an operator types is not a dependency order, and the arm gate depended on it.

    ``RH_OPTION3_STAGES=arm,screen`` satisfied ``assert_screen_precedes_the_arm`` -- which returns
    early whenever ``screen`` appears anywhere in the selection -- and then ran the 20-GPU-hour arm
    first, with no screen artifact on disk. The selection now routes through
    ``games.plans.selected_stages_from``, which re-orders into the plan's own declared order.
    """

    def test_a_reversed_selection_still_builds_the_screen_first(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RH_OPTION3_STAGES", "arm,screen")
        assert train_sequence.selected_stages() == ("screen", "arm")
        assert [stage.name.split("-")[0] for stage in train_sequence.stages()] == ["screen", "arm"]

    def test_a_scrambled_full_selection_is_normalised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RH_OPTION3_STAGES", "analyze,arm,eval,screen")
        assert train_sequence.selected_stages() == train_sequence.DEFAULT_STAGES

    def test_an_unknown_stage_is_still_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RH_OPTION3_STAGES", "screen,tune")
        with pytest.raises(ValueError, match="unknown stage"):
            train_sequence.selected_stages()

    def test_an_empty_selection_is_still_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RH_OPTION3_STAGES", " , ")
        with pytest.raises(ValueError, match="selected nothing"):
            train_sequence.selected_stages()

    def test_the_declared_order_covers_every_buildable_stage(self) -> None:
        """The selection validates against ``DEFAULT_STAGES``, so a stage missing from it is
        unselectable however well its builder works."""
        assert tuple(train_sequence.STAGE_BUILDERS) == train_sequence.DEFAULT_STAGES


class TestTheScreenGateChecksTheVerdictsProvenanceNotItsContents:
    """A verdict measured under another parser or another exposure says nothing about this arm.

    The gate still refuses to grade what the verdict FOUND -- that judgement is the owner's. It checks
    only that the verdict was measured the way the arm will train -- same solution parser, same
    grader exposure -- which are provenance questions with one right answer each.
    """

    def _arm_only(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("RH_OPTION3_ARM", ARM_MISSPECIFIED)
        monkeypatch.setenv("RH_OPTION3_STAGES", "arm")
        monkeypatch.setattr(train_sequence, "SCREEN_DIR", tmp_path / "screen")

    def test_a_verdict_from_the_shipped_parser_satisfies_it(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._arm_only(monkeypatch, tmp_path)
        write_screen_verdict(train_sequence.screen_artifact())
        assert len(train_sequence.stages()) == 1

    def test_a_verdict_with_no_parser_key_is_read_as_pre_fix_and_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The live case: both verdicts in the run's S3 prefix predate the parser change entirely."""
        self._arm_only(monkeypatch, tmp_path)
        write_screen_verdict(train_sequence.screen_artifact(), solution_parser=None)
        with pytest.raises(ValueError, match="solution_parser=None") as caught:
            train_sequence.stages()
        assert "RE-SCREEN" in str(caught.value)

    def test_a_verdict_from_the_retired_parser_is_refused_and_named(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._arm_only(monkeypatch, tmp_path)
        write_screen_verdict(train_sequence.screen_artifact(), solution_parser="tags-only")
        with pytest.raises(ValueError, match="tags-only") as caught:
            train_sequence.stages()
        # Both values named, so a reader can tell which side to re-run without opening the artifact.
        assert SOLUTION_PARSER in str(caught.value)

    def test_a_blind_screen_does_not_bless_an_inline_arm(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Sabotage: plant a --grader-exposure withheld verdict, which used to print GATE PASSED.

        The withheld rendering deletes the grader block from every prompt, so the verdict describes
        a corpus the arm will never train on; the parser comparison alone cannot see that.
        """
        self._arm_only(monkeypatch, tmp_path)
        write_screen_verdict(train_sequence.screen_artifact(), grader_exposure="withheld")
        with pytest.raises(ValueError, match="grader_exposure='withheld'") as caught:
            train_sequence.stages()
        assert "RE-SCREEN" in str(caught.value)

    def test_a_verdict_with_no_exposure_key_passes_because_it_was_necessarily_inline(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The asymmetry with the parser rule is deliberate: the key and the --grader-exposure
        flag landed in one commit, so an absent key proves the screen sampled inline."""
        self._arm_only(monkeypatch, tmp_path)
        write_screen_verdict(train_sequence.screen_artifact())
        assert len(train_sequence.stages()) == 1

    def test_an_explicitly_inline_verdict_passes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._arm_only(monkeypatch, tmp_path)
        write_screen_verdict(train_sequence.screen_artifact(), grader_exposure="inline")
        assert len(train_sequence.stages()) == 1

    def test_the_gate_still_refuses_to_grade_what_the_verdict_found(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """An all-pure screen is a judgement for a human; a matching parser is all this gate wants."""
        self._arm_only(monkeypatch, tmp_path)
        artifact = train_sequence.screen_artifact()
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(
            json.dumps(
                {
                    "solution_parser": SOLUTION_PARSER,
                    "groups": {"frac_groups_pure": 1.0, "all_groups_pure": True},
                    "visible_pass": {"count": 0, "denominator": 472, "rate": 0.0},
                }
            ),
            encoding="utf-8",
        )
        assert len(train_sequence.stages()) == 1

    def test_selecting_the_screen_in_the_same_invocation_skips_the_provenance_check(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """There is nothing to check yet: this invocation is about to write the verdict itself."""
        monkeypatch.setenv("RH_OPTION3_ARM", ARM_MISSPECIFIED)
        monkeypatch.setenv("RH_OPTION3_STAGES", "screen,arm")
        monkeypatch.setattr(train_sequence, "SCREEN_DIR", tmp_path / "screen")
        assert [stage.name.split("-")[0] for stage in train_sequence.stages()] == ["screen", "arm"]
