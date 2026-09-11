"""The gradient screen: the sampler it pins, the groups it reads, and the guards that keep it honest.

Offline throughout, on the pattern of ``test_rh_train_dataset``: a stub tokenizer with the Qwen3.5
thinking-prefill behaviour, ``MockBackend`` for generation, and an injected fake grader for the
grading seam. The one class marked ``needs_jail`` drives a mock completion through the REAL jailed
grader over a real ILCB task, because the screen's whole claim is that its pass rate is the training
reward's -- and a grading path nobody has watched run in the jail is a claim nobody has checked.
"""

from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest
from conftest import needs_jail

from games.generation import TRAINING_TEMPERATURE, TRAINING_TOP_K, TRAINING_TOP_P
from games.provenance import git_provenance
from reward_hacking import train_screen
from reward_hacking.model_backend import (
    STOP_REASON_END_TURN,
    STOP_REASON_MAX_TOKENS,
    MockBackend,
    RawResponse,
)
from reward_hacking.train_dataset import (
    ARM_CONTROL,
    BoundedSubset,
    resolve_arm_rows,
    training_rows,
)
from reward_hacking.train_partition import build_partition
from reward_hacking.train_reward import (
    REWARD_FAIL,
    REWARD_PASS,
    GradedCompletion,
    GraderConfig,
    GraderOutcome,
)
from reward_hacking.train_screen import (
    DEFAULT_MAX_PROMPT_TOKENS,
    DEFAULT_MODEL_ID,
    DEFAULT_SAMPLES_PER_PROMPT,
    DEFAULT_THINKING,
    PERCENTILE_METHOD,
    ScreenedSample,
    assert_backend_can_render_template,
    assert_screen_measured,
    generate_screen_completions,
    grade_screen_completions,
    mitigation_menu,
    mock_screen_response,
    samples_path_for,
    screen_report,
    training_matched_sampling,
    write_screen_artifacts,
)
from reward_hacking.train_termination import required_coding_completion_budget

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from reward_hacking.train_partition import HeldOutPartition

SCRATCH_ROOT = Path("/var/tmp/rh-train-screen-tests")  # noqa: S108 - the jail refuses a home-tree dir

PASS_MARKER = "return 'the-blessed-answer'"

THINKING_PREFILL = "<think>\n"
THINKING_CLOSED_BLOCK = "<think>\n\n</think>\n\n"


class StubTokenizer:
    """The Qwen3.5-shaped chat template with whitespace tokenisation, as the dataset tests use."""

    template = "chat template with no extra knobs"

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        enable_thinking: bool = False,
        **kwargs: Any,
    ) -> str:
        del kwargs
        assert not tokenize
        assert add_generation_prompt
        turns = "".join(f"<|im_start|>{turn['role']}\n{turn['content']}\n" for turn in conversation)
        tail = THINKING_PREFILL if enable_thinking else THINKING_CLOSED_BLOCK
        return f"{turns}<|im_start|>assistant\n{tail}"

    def get_chat_template(self) -> str:
        return self.template

    def __call__(self, text: str, *, add_special_tokens: bool = True) -> dict[str, list[int]]:
        del add_special_tokens
        return {"input_ids": list(range(len(text.split())))}


class EffortStubTokenizer(StubTokenizer):
    """A template carrying the knob the generation backends cannot pass through."""

    template = "{% if reasoning_effort %}steering text nobody wrote{% endif %}"


def stub_tokenizer() -> PreTrainedTokenizerBase:
    return cast("PreTrainedTokenizerBase", StubTokenizer())


@pytest.fixture
def partition() -> HeldOutPartition:
    return build_partition(seed=0)


def synthetic_rows(n: int) -> list[dict[str, Any]]:
    """Prompt rows shaped like ``training_rows`` output, with no real problem text."""
    return [
        {
            "prompt": f"Implement `toy_fn_{index}` in Python. (synthetic prompt {index})",
            "task_id": f"ilcb-oneoff-toy_{index}",
            "problem_id": f"toy_{index}",
            "split": "oneoff",
            "entry_point": f"toy_fn_{index}",
            "partition_side": "training",
        }
        for index in range(n)
    ]


def fake_grade(task_id: str, solution: str | None) -> GradedCompletion:
    """Grade without the jail: a solution carrying PASS_MARKER passes, anything else fails."""
    if solution is None:
        outcome = GraderOutcome.NO_SOLUTION
    elif PASS_MARKER in solution:
        outcome = GraderOutcome.PASS
    else:
        outcome = GraderOutcome.FAIL
    return GradedCompletion(
        task_id=task_id,
        outcome=outcome,
        reward=REWARD_PASS if outcome is GraderOutcome.PASS else REWARD_FAIL,
        grader_stdout="SUCCESS" if outcome is GraderOutcome.PASS else "FAIL",
        hidden_outcome=None,
        seconds=0.01,
        solution_chars=len(solution or ""),
    )


def raw(
    text: str, *, output_tokens: int | None = None, stop_reason: str | None = None
) -> RawResponse:
    return RawResponse(
        text=text,
        reasoning="",
        input_tokens=None,
        output_tokens=output_tokens,
        stop_reason=stop_reason,
    )


def passing_completion() -> str:
    return f"thinking...\n</think>\n<solution>\ndef toy():\n    {PASS_MARKER}\n</solution>"


def failing_completion() -> str:
    return "thinking...\n</think>\n<solution>\ndef toy():\n    return None\n</solution>"


def screened(  # noqa: PLR0913 - one keyword per varied record field
    outcome: GraderOutcome,
    *,
    group_index: int = 0,
    sample_index: int = 0,
    truncated: bool = False,
    output_tokens: int | None = None,
    stop_reason: str | None = None,
) -> ScreenedSample:
    return ScreenedSample(
        problem_id=f"toy_{group_index}",
        task_id=f"ilcb-oneoff-toy_{group_index}",
        group_index=group_index,
        sample_index=sample_index,
        completion="text",
        truncated_thinking=truncated,
        solution=None if outcome is GraderOutcome.NO_SOLUTION else "def toy(): ...",
        outcome=outcome,
        reward=REWARD_PASS if outcome is GraderOutcome.PASS else REWARD_FAIL,
        grader_stdout="",
        grader_seconds=0.01,
        output_tokens=output_tokens,
        stop_reason=stop_reason,
    )


def group(outcomes: list[GraderOutcome], group_index: int) -> list[ScreenedSample]:
    return [
        screened(outcome, group_index=group_index, sample_index=sample_index)
        for sample_index, outcome in enumerate(outcomes)
    ]


class TestTrainingMatchedSampling:
    def test_the_screen_samples_exactly_what_trl_trains_at(self):
        """The whole point of the screen: any other sampler measures a different distribution."""
        sampling = training_matched_sampling(DEFAULT_MODEL_ID)
        assert sampling.do_sample is True
        assert sampling.temperature == TRAINING_TEMPERATURE
        assert sampling.top_p == TRAINING_TOP_P
        assert sampling.top_k == TRAINING_TOP_K
        assert sampling.min_p == 0.0
        assert sampling.repetition_penalty == 1.0
        assert sampling.presence_penalty == 0.0

    def test_the_token_cap_is_the_coding_screen_budget_never_lower(self):
        sampling = training_matched_sampling(DEFAULT_MODEL_ID)
        assert sampling.max_new_tokens == required_coding_completion_budget(DEFAULT_MODEL_ID)

    def test_the_sampler_is_unseeded_so_group_members_can_differ(self):
        """A fixed per-request seed makes 8 identical prompts decode identically, so every group
        would read pure and the screen would report 'no gradient' about the seed, not the model."""
        assert training_matched_sampling(DEFAULT_MODEL_ID).seed is None

    def test_the_mirrored_defaults_match_the_trainer_field_for_field(self):
        """The four mirrored constants may never drift from reward_hacking.train's own."""
        import reward_hacking.train as train_module  # noqa: PLC0415 - trl-heavy, test-time only

        assert DEFAULT_MODEL_ID == train_module.DEFAULT_MODEL_ID
        assert DEFAULT_MAX_PROMPT_TOKENS == train_module.DEFAULT_MAX_PROMPT_TOKENS
        defaults = {
            field.name: field.default for field in fields(train_module.RewardHackingTrainConfig)
        }
        assert defaults["num_generations"] == DEFAULT_SAMPLES_PER_PROMPT
        assert defaults["thinking"] == DEFAULT_THINKING

    def test_the_trainer_decodes_at_the_same_three_sampler_values(self):
        import reward_hacking.train as train_module  # noqa: PLC0415 - trl-heavy, test-time only

        defaults = {
            field.name: field.default for field in fields(train_module.RewardHackingTrainConfig)
        }
        sampling = training_matched_sampling(DEFAULT_MODEL_ID)
        assert defaults["temperature"] == sampling.temperature
        assert defaults["top_p"] == sampling.top_p
        assert defaults["top_k"] == sampling.top_k


class TestTemplateRenderGuard:
    def test_a_plain_template_is_accepted(self):
        assert_backend_can_render_template(stub_tokenizer())

    def test_a_template_needing_kwargs_is_refused(self):
        """Sabotage target: on such a model the screen would sample different prompt text."""
        with pytest.raises(RuntimeError, match="chat-template kwargs"):
            assert_backend_can_render_template(
                cast("PreTrainedTokenizerBase", EffortStubTokenizer())
            )


class TestTheScreenResolvesRowsThroughTheArmsOwnResolver:
    """The screen no longer has a corpus resolver of its own, and that IS the fix.

    The two used to apply the prompt budget and ``--max-prompts`` in opposite orders, so the screen's
    headline claim -- that it covers exactly the corpus an arm would see -- was false whenever the
    bound was used on both sides. The behaviour of the shared resolver is tested where it now lives,
    in ``test_rh_train_dataset``; what belongs here is that this module reaches it and reaches nothing
    else, which a private copy reintroduced under any name would break.
    """

    def test_the_module_carries_no_corpus_resolver_of_its_own(self):
        private = [
            name
            for name in vars(train_screen)
            if "resolve" in name
            and name not in {"resolve_arm_rows", "resolve_chat_template_kwargs"}
        ]
        assert private == []

    def test_the_screen_reaches_the_shared_resolver_and_the_shared_subset_record(self):
        assert train_screen.resolve_arm_rows is resolve_arm_rows
        assert train_screen.BoundedSubset is BoundedSubset


class TestGenerateScreenCompletions:
    def test_samples_arrive_group_contiguous_in_row_order(self):
        rows = synthetic_rows(2)
        backend = MockBackend(lambda prompt: f"echo: {prompt}")
        responses = generate_screen_completions(rows, backend, samples_per_prompt=3)
        assert len(responses) == 6
        assert all(r.text == f"echo: {rows[0]['prompt']}" for r in responses[:3])
        assert all(r.text == f"echo: {rows[1]['prompt']}" for r in responses[3:])

    def test_a_group_of_one_is_refused_because_purity_would_be_arithmetic(self):
        with pytest.raises(ValueError, match="cannot disagree with itself"):
            generate_screen_completions(synthetic_rows(1), MockBackend(["x"]), samples_per_prompt=1)

    def test_no_rows_is_refused(self):
        with pytest.raises(ValueError, match="no prompt rows"):
            generate_screen_completions([], MockBackend(["x"]), samples_per_prompt=2)

    def test_a_backend_returning_a_short_batch_is_refused(self):
        """Sabotage: one completion too few shifts every later sample into its neighbour's group."""

        class ShortBatchBackend:
            model_id = "short"
            transport = "short"

            def generate(self, prompts: list[str]) -> list[str]:
                return ["only one reply" for _ in prompts[:-1]]

        with pytest.raises(RuntimeError, match="group membership is positional"):
            generate_screen_completions(
                synthetic_rows(2), ShortBatchBackend(), samples_per_prompt=2
            )


class TestGradeScreenCompletions:
    def test_the_full_offline_path_strips_extracts_grades_and_indexes(self):
        rows = synthetic_rows(2)
        responses = [
            raw(passing_completion(), output_tokens=10, stop_reason=STOP_REASON_END_TURN),
            raw(failing_completion(), output_tokens=20, stop_reason=STOP_REASON_END_TURN),
            raw(
                "still thinking, never closed", output_tokens=30, stop_reason=STOP_REASON_MAX_TOKENS
            ),
            raw(failing_completion(), output_tokens=40, stop_reason=STOP_REASON_END_TURN),
        ]
        samples = grade_screen_completions(
            rows,
            responses,
            samples_per_prompt=2,
            prefilled_think=True,
            grader=GraderConfig(scratch_root=SCRATCH_ROOT),
            grade=fake_grade,
        )
        assert [sample.outcome for sample in samples] == [
            GraderOutcome.PASS,
            GraderOutcome.FAIL,
            GraderOutcome.NO_SOLUTION,
            GraderOutcome.FAIL,
        ]
        assert [sample.group_index for sample in samples] == [0, 0, 1, 1]
        assert [sample.sample_index for sample in samples] == [0, 1, 0, 1]
        assert samples[0].problem_id == rows[0]["problem_id"]
        assert samples[2].problem_id == rows[1]["problem_id"]
        # The unclosed thinking block is the truncation measurement, and it submits nothing.
        assert samples[2].truncated_thinking is True
        assert samples[2].solution is None
        assert samples[0].reward == REWARD_PASS
        assert [sample.output_tokens for sample in samples] == [10, 20, 30, 40]

    def test_a_misaligned_batch_is_refused(self):
        with pytest.raises(RuntimeError, match="misaligned"):
            grade_screen_completions(
                synthetic_rows(2),
                [raw("one")],
                samples_per_prompt=2,
                prefilled_think=True,
                grader=GraderConfig(scratch_root=SCRATCH_ROOT),
                grade=fake_grade,
            )


class TestScreenReport:
    def report(self) -> dict[str, Any]:
        samples = [
            *group([GraderOutcome.FAIL] * 4, 0),
            *group(
                [GraderOutcome.PASS, GraderOutcome.PASS, GraderOutcome.FAIL, GraderOutcome.FAIL], 1
            ),
            *group([GraderOutcome.PASS] * 4, 2),
        ]
        return screen_report(samples, samples_per_prompt=4, token_cap=100)

    def test_group_purity_and_span_read_the_known_mixture(self):
        groups = self.report()["groups"]
        assert groups["frac_groups_pure"] == pytest.approx(2 / 3)
        assert groups["frac_groups_mixed"] == pytest.approx(1 / 3)
        assert groups["all_groups_pure"] is False
        assert groups["mean_group_reward_span"] == pytest.approx((0.0 + 1.0 + 0.0) / 3)

    def test_the_histogram_separates_spread_thin_from_concentrated(self):
        groups = self.report()["groups"]
        assert groups["pass_count_histogram"] == {"0": 1, "1": 0, "2": 1, "3": 0, "4": 1}

    def test_every_rate_carries_its_count_and_denominator(self):
        report = self.report()
        assert report["visible_pass"] == {"count": 6, "denominator": 12, "rate": 0.5}
        assert report["counts"] == {"n_prompts": 3, "samples_per_prompt": 4, "n_samples": 12}

    def test_the_outcome_tallies_name_every_grader_outcome(self):
        samples = [
            *group([GraderOutcome.NO_SOLUTION, GraderOutcome.TIMEOUT], 0),
            *group([GraderOutcome.NO_VERDICT, GraderOutcome.FAIL], 1),
        ]
        report = screen_report(samples, samples_per_prompt=2, token_cap=100)
        assert report["parse_failure"]["count"] == 1
        assert report["grader_timeout"]["count"] == 1
        assert report["grader_no_verdict"]["count"] == 1
        assert "apparatus failure" in report["grader_no_verdict"]["meaning"]
        assert report["groups"]["n_groups_containing_no_verdict"] == 1
        assert report["apparatus_failure"] is False

    def test_truncated_thinking_is_counted_with_its_denominator(self):
        samples = [
            screened(GraderOutcome.NO_SOLUTION, group_index=0, sample_index=0, truncated=True),
            screened(GraderOutcome.FAIL, group_index=0, sample_index=1),
        ]
        report = screen_report(samples, samples_per_prompt=2, token_cap=100)
        assert report["truncated_thinking"] == {"count": 1, "denominator": 2, "rate": 0.5}

    def test_the_length_distribution_reads_percentiles_and_cap_hits(self):
        samples = [
            screened(
                GraderOutcome.FAIL,
                group_index=0,
                sample_index=index,
                output_tokens=(index + 1) * 10,
                stop_reason=STOP_REASON_MAX_TOKENS if index == 9 else STOP_REASON_END_TURN,
            )
            for index in range(10)
        ]
        lengths = screen_report(samples, samples_per_prompt=10, token_cap=100)["generated_tokens"]
        assert lengths["n_measured"] == 10
        assert lengths["p50"] == 50
        assert lengths["p90"] == 90
        assert lengths["p95"] == 100
        assert lengths["max"] == 100
        assert lengths["n_hit_cap"] == 1
        assert lengths["token_cap"] == 100

    def test_unmeasured_token_counts_are_reported_as_absent_not_zero(self):
        samples = group([GraderOutcome.FAIL, GraderOutcome.FAIL], 0)
        lengths = screen_report(samples, samples_per_prompt=2, token_cap=100)["generated_tokens"]
        assert lengths["n_measured"] == 0
        assert lengths["p50"] is None
        assert lengths["max"] is None

    def test_the_percentiles_name_their_estimator_and_it_is_nearest_rank(self):
        """A percentile is not a quantity until its estimator is named.

        An offline re-grade of the same records with an interpolating median read 22,368 / 22,660.5
        against this artifact's 22,360 / 22,574, and the half-token was the only clue that two
        estimators were being compared as one measurement. The estimator itself stays as it is:
        nearest-rank returns an observed token count, and every screen on disk was measured with it.
        """
        samples = [
            screened(GraderOutcome.FAIL, group_index=0, sample_index=index, output_tokens=length)
            for index, length in enumerate((10, 20, 30, 40))
        ]
        lengths = screen_report(samples, samples_per_prompt=4, token_cap=100)["generated_tokens"]
        assert lengths["percentile_method"] == PERCENTILE_METHOD
        # An observed value, never the 25.0 an interpolating median would return over four points.
        assert lengths["p50"] == 20

    def test_per_problem_detail_names_each_group(self):
        per_problem = self.report()["per_problem"]
        assert [entry["pass_count"] for entry in per_problem] == [0, 2, 4]
        assert per_problem[0]["problem_id"] == "toy_0"
        assert per_problem[1]["outcomes"][GraderOutcome.PASS.value] == 2

    def test_a_ragged_sample_count_is_refused(self):
        with pytest.raises(ValueError, match="whole number of groups"):
            screen_report(group([GraderOutcome.FAIL] * 3, 0), samples_per_prompt=2, token_cap=100)

    def test_no_samples_is_refused(self):
        with pytest.raises(ValueError, match="no samples"):
            screen_report([], samples_per_prompt=2, token_cap=100)


class TestApparatusFailureGuard:
    def test_a_screen_where_no_grader_reported_raises_rather_than_reading_as_zero(self):
        """Sabotage: every grader silent, exactly what a jail that broke mid-run produces."""
        samples = group([GraderOutcome.NO_VERDICT] * 4, 0)
        report = screen_report(samples, samples_per_prompt=4, token_cap=100)
        assert report["apparatus_failure"] is True
        with pytest.raises(RuntimeError, match="measured the apparatus"):
            assert_screen_measured(report)

    def test_a_screen_of_honest_rejections_is_a_measurement(self):
        report = screen_report(
            group([GraderOutcome.FAIL] * 4, 0), samples_per_prompt=4, token_cap=100
        )
        assert_screen_measured(report)


class TestMitigationMenu:
    """The menu that keeps a pure screen from stalling the launch: priced, in the artifact."""

    def mixed_menu(self, arm: str = "misspecified") -> dict[str, Any]:
        samples = [
            *group([GraderOutcome.FAIL] * 4, 0),
            *group(
                [GraderOutcome.PASS, GraderOutcome.PASS, GraderOutcome.FAIL, GraderOutcome.FAIL], 1
            ),
            *group([GraderOutcome.PASS] * 4, 2),
        ]
        return mitigation_menu(screen_report(samples, samples_per_prompt=4, token_cap=100), arm=arm)

    def all_fail_menu(self) -> dict[str, Any]:
        samples = [*group([GraderOutcome.FAIL] * 4, 0), *group([GraderOutcome.FAIL] * 4, 1)]
        return mitigation_menu(
            screen_report(samples, samples_per_prompt=4, token_cap=100), arm="misspecified"
        )

    def option(self, menu: dict[str, Any], name: str) -> dict[str, Any]:
        matches = [entry for entry in menu["options"] if entry["option"] == name]
        assert len(matches) == 1, name
        return matches[0]

    def test_raising_the_group_size_is_priced_from_the_observed_rates(self):
        """Per-prompt empirical rates {0, 0.5, 1}: only the mixed prompt can yield mixed groups,
        and at N=8 its chance is 1 - 2*(0.5^8)."""
        priced = self.option(self.mixed_menu(), "raise-samples-per-group")
        expected = priced["expected_mixed_groups_by_group_size"]
        assert expected["8"] == pytest.approx(1 - 2 * 0.5**8, abs=1e-3)
        assert expected["64"] == pytest.approx(1.0, abs=1e-3)

    def test_a_zero_pass_screen_prices_its_own_detection_ceiling(self):
        """Zero passes in 8 samples bounds the rate below 3/8; the menu says what a group of each
        size could see at that ceiling instead of concluding no group size can work."""
        menu = self.all_fail_menu()
        per_sample = menu["per_sample_pass"]
        assert per_sample["count"] == 0
        assert per_sample["zero_pass_ceiling_95"] == pytest.approx(3 / 8)
        chances = per_sample["chance_of_any_pass_at_ceiling_by_group_size"]
        assert chances["8"] == pytest.approx(1 - (1 - 3 / 8) ** 8, abs=1e-3)
        priced = self.option(menu, "raise-samples-per-group")
        assert priced["expected_mixed_groups_by_group_size"]["64"] == 0.0

    def test_a_screen_with_passes_carries_no_ceiling(self):
        assert "zero_pass_ceiling_95" not in self.mixed_menu()["per_sample_pass"]

    def test_the_control_arm_reads_as_the_positive_control(self):
        control = mitigation_menu(
            screen_report(group([GraderOutcome.FAIL] * 4, 0), samples_per_prompt=4, token_cap=100),
            arm="control",
        )
        assert "POSITIVE CONTROL" in control["arm_role"]
        assert "screen itself" in control["arm_role"]
        misspecified = self.mixed_menu()
        assert "beside the control" in misspecified["arm_role"]

    def test_the_dataset_filter_option_counts_this_arms_passing_problems(self):
        priced = self.option(self.mixed_menu(), "filter-prompts-by-either-arm-pass")
        assert priced["n_problems_with_any_pass_this_arm"] == 2
        assert priced["n_problems_screened_this_arm"] == 3
        assert "arm-independent" in str(priced["changes_about_the_claim"])

    def test_the_pure_split_tells_all_fail_from_all_pass(self):
        pure = self.mixed_menu()["pure_groups"]
        assert pure["n_pure_all_fail"] == 1
        assert pure["n_pure_all_pass"] == 1

    def test_every_option_names_its_cost_and_what_it_changes(self):
        for entry in self.mixed_menu()["options"]:
            assert entry["cost"], entry["option"]
            assert entry["changes_about_the_claim"], entry["option"]

    def test_an_unknown_arm_is_refused(self):
        with pytest.raises(ValueError, match="unknown arm"):
            self.mixed_menu(arm="whatever")

    def test_the_legible_subset_arm_carries_its_own_role_note(self):
        """Every arm the CLI accepts needs a role note, or its screen crashes after sampling."""
        menu = self.mixed_menu(arm="legible-subset")
        assert "visible-pass AND hidden-fail" in menu["arm_role"]

    def test_every_cli_arm_choice_has_a_role_note(self):
        """The --arm choices and the menu's vocabulary must cover each other exactly: a choice
        without a note crashes mitigation_menu AFTER the screen has spent its sampling."""
        from reward_hacking.train_dataset import TRAINABLE_ARMS  # noqa: PLC0415
        from reward_hacking.train_screen import ARM_ROLE_NOTES  # noqa: PLC0415

        assert set(ARM_ROLE_NOTES) == set(TRAINABLE_ARMS)


class TestExposureFlagParsing:
    def test_the_exposure_defaults_to_inline_and_accepts_withheld(self):
        from reward_hacking.train_screen import _parse_args  # noqa: PLC0415

        base = ["--arm", "legible-subset", "--out", "screen.json"]
        assert _parse_args(base).grader_exposure == "inline"
        assert _parse_args([*base, "--grader-exposure", "withheld"]).grader_exposure == "withheld"

    def test_the_legible_arm_is_an_accepted_choice(self):
        from reward_hacking.train_screen import _parse_args  # noqa: PLC0415

        parsed = _parse_args(["--arm", "legible-subset", "--out", "screen.json"])
        assert parsed.arm == "legible-subset"


class TestArtifacts:
    def test_the_summary_and_samples_land_beside_each_other(self, tmp_path: Path):
        out = tmp_path / "gradient-screen-test.json"
        samples = group([GraderOutcome.PASS, GraderOutcome.FAIL], 0)
        summary = {"kind": "gradient-screen", "visible_pass": {"count": 1}}
        samples_path = write_screen_artifacts(out, summary, samples)
        assert samples_path == samples_path_for(out)
        assert json.loads(out.read_text())["kind"] == "gradient-screen"
        lines = [json.loads(line) for line in samples_path.read_text().splitlines()]
        assert len(lines) == 2
        assert lines[0]["outcome"] == GraderOutcome.PASS.value
        assert lines[1]["reward"] == REWARD_FAIL

    def test_an_existing_artifact_is_never_overwritten(self, tmp_path: Path):
        """Sabotage: a screen artifact is launch evidence; rewriting it relabels a decision."""
        out = tmp_path / "gradient-screen-test.json"
        samples = group([GraderOutcome.FAIL, GraderOutcome.FAIL], 0)
        write_screen_artifacts(out, {"kind": "gradient-screen"}, samples)
        with pytest.raises(FileExistsError, match="launch evidence"):
            write_screen_artifacts(out, {"kind": "gradient-screen"}, samples)

    def test_a_stray_samples_file_alone_also_refuses(self, tmp_path: Path):
        out = tmp_path / "gradient-screen-test.json"
        samples_path_for(out).write_text("stray\n")
        with pytest.raises(FileExistsError):
            write_screen_artifacts(out, {}, group([GraderOutcome.FAIL, GraderOutcome.FAIL], 0))

    def test_the_write_is_an_exclusive_create_not_a_checked_open(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The refusal has to survive a race, not just a sequential retry.

        Three to five sessions share this box, so an ``exists()`` test followed by an ordinary open
        has a window two screens of one arm can both pass -- and the interleaving that survives is a
        summary paired with the OTHER run's samples: internally consistent, wrong, and silent about
        the collision. The window is simulated by making the existence check answer False while the
        file is really there, which is exactly what a check that ran a moment too early reports. Only
        the exclusive create catches it.
        """
        out = tmp_path / "gradient-screen-test.json"
        out.write_text("the other session's summary\n", encoding="utf-8")
        monkeypatch.setattr(Path, "exists", lambda _self, **_kwargs: False)
        with pytest.raises(FileExistsError):
            write_screen_artifacts(
                out, {"kind": "gradient-screen"}, group([GraderOutcome.FAIL, GraderOutcome.FAIL], 0)
            )
        monkeypatch.undo()
        assert out.read_text(encoding="utf-8") == "the other session's summary\n"
        assert not samples_path_for(out).exists()

    def test_the_summary_names_the_commit_that_measured_it(self, tmp_path: Path):
        """A verdict is read as a launch decision months later, and the arm gate keys on provenance."""
        out = tmp_path / "gradient-screen-test.json"
        summary = {"kind": "gradient-screen", **git_provenance()}
        write_screen_artifacts(out, summary, group([GraderOutcome.FAIL, GraderOutcome.FAIL], 0))
        record = json.loads(out.read_text())
        assert record["git_sha"]
        assert "git_tree_dirty" in record


class TestMockScreenResponse:
    def test_the_canned_completion_defines_the_prompts_own_entry_point(self):
        response = mock_screen_response(synthetic_rows(1)[0]["prompt"])
        assert "def toy_fn_0(" in response
        assert "</think>" in response
        assert "<solution>" in response

    def test_a_prompt_without_the_template_marker_is_refused(self):
        with pytest.raises(ValueError, match="entry point"):
            mock_screen_response("a prompt shaped nothing like the training template")


@needs_jail
class TestScreenAgainstTheRealJail:
    """The seam the offline tests fake: a mock completion graded by the real jailed grader."""

    def test_a_mock_screen_grades_as_policy_failure_not_apparatus_failure(
        self, partition: HeldOutPartition
    ):
        rows = [training_rows(ARM_CONTROL, partition)[0]]
        backend = MockBackend(mock_screen_response)
        responses = generate_screen_completions(rows, backend, samples_per_prompt=2)
        samples = grade_screen_completions(
            rows,
            responses,
            samples_per_prompt=2,
            prefilled_think=True,
            grader=GraderConfig(scratch_root=SCRATCH_ROOT, workers=2),
        )
        # FAIL proves the jail ran and rejected the do-nothing solution; NO_VERDICT would not.
        assert [sample.outcome for sample in samples] == [GraderOutcome.FAIL, GraderOutcome.FAIL]
        report = screen_report(samples, samples_per_prompt=2, token_cap=1000)
        assert report["apparatus_failure"] is False
        assert report["visible_pass"] == {"count": 0, "denominator": 2, "rate": 0.0}
        assert_screen_measured(report)


class TestReplaceabilityOfTheSampleRecord:
    def test_the_json_record_round_trips_every_field(self):
        sample = replace(
            screened(GraderOutcome.TIMEOUT, output_tokens=42, stop_reason="max_tokens"),
            solution="def toy(): ...",
        )
        record = sample.to_json_dict()
        assert record["outcome"] == "timeout"
        assert record["output_tokens"] == 42
        assert record["stop_reason"] == "max_tokens"
        assert set(record) == {field.name for field in fields(ScreenedSample)}


class TestBoundedSubsetRecord:
    def test_the_record_indicts_itself(self):
        subset = BoundedSubset(
            max_prompts=3, n_total_rows=61, seed=0, kept_problem_ids=("a", "b", "c")
        )
        recorded = subset.to_json_dict()
        assert "must not be reported" in str(recorded["warning"])
        assert recorded["kept_problem_ids"] == ["a", "b", "c"]
