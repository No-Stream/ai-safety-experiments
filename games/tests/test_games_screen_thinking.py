"""Offline tests for the model-termination screen. No GPU, no model loads.

The screen exists because a leaderboard score cannot tell you whether a model stops deliberating
on a strategically underdetermined prompt, and that is the property that decides whether an arm can
run at all. So the tests care about the classification and the usability verdict.
"""

from __future__ import annotations

from typing import ClassVar

import pytest
import torch

from games import preflight as pf
from games import screen_thinking as st
from games.chunked_decode import THROUGHPUT_KNEE_SEQUENCE_TOKENS, knee_width_cap
from games.termination import MEASURED_TERMINATION_BUDGET

LABELS = {"label_a": "SIDE", "label_b": "LOOP", "coop_label": "SIDE"}


def rollout(budget: int, n_tokens: int, *, action: str | None, closed: bool = True) -> st.Rollout:
    """Build a Rollout directly, bypassing generation."""
    return st.Rollout(budget=budget, n_tokens=n_tokens, closed_think=closed, parsed_action=action)


class TestClassify:
    def test_a_terminated_completion_with_a_tag_parses(self):
        text = "reasoning here</think>\n\nI pick <action>SIDE</action>"
        result = st.classify(text, budget=1024, n_tokens=40, row=LABELS, prefilled=True)
        assert result.parsed_action == "C"
        assert result.parsed is True
        assert result.closed_think is True
        assert result.hit_cap is False

    def test_the_defecting_label_maps_to_d(self):
        text = "</think><action>LOOP</action>"
        assert (
            st.classify(text, budget=512, n_tokens=9, row=LABELS, prefilled=True).parsed_action
            == "D"
        )

    def test_an_unclosed_think_block_does_not_parse(self):
        # The observed failure: thinking runs into the cap, so there is no visible answer at all.
        text = "still deliberating about whether"
        result = st.classify(text, budget=64, n_tokens=64, row=LABELS, prefilled=True)
        assert result.parsed is False
        assert result.closed_think is False
        assert result.hit_cap is True

    def test_a_tag_inside_the_thinking_block_does_not_count(self):
        # A choice mentioned mid-reasoning is reasoning, not an answer.
        text = "maybe <action>SIDE</action> would work, but"
        assert st.classify(text, budget=64, n_tokens=64, row=LABELS, prefilled=True).parsed is False

    def test_hitting_the_cap_exactly_counts_as_capped(self):
        assert rollout(1024, 1024, action=None).hit_cap is True
        assert rollout(1024, 1023, action="C").hit_cap is False


class TestSummarise:
    def test_counts_and_rate(self):
        rollouts = [
            rollout(2048, 100, action="C"),
            rollout(2048, 120, action="D"),
            rollout(2048, 2048, action=None, closed=False),
            rollout(2048, 2048, action=None, closed=False),
        ]
        summary = st.summarise(rollouts, group_size=8)
        assert summary.budget == 2048
        assert summary.n_rollouts == 4
        assert summary.parsed == 2
        assert summary.parse_rate == pytest.approx(0.5)
        assert summary.hit_cap == 2
        assert summary.closed_think == 2
        assert summary.cooperated == 1
        assert summary.max_tokens == 2048

    def test_the_whole_batch_risk_is_what_decides_usability(self):
        # A 50% parse rate reads healthy but still fails a group of 8 about 0.4% of the time; a 3%
        # rate -- what Qwen3.5-0.8B gave at 2048 -- fails roughly four times in five.
        healthy = st.summarise([rollout(2048, 10, action="C")] * 8, group_size=8)
        assert healthy.whole_batch_failure_risk == pytest.approx(0.0)
        assert healthy.usable is True

        marginal = st.summarise(
            [rollout(2048, 10, action="C")] + [rollout(2048, 2048, action=None)] * 31,
            group_size=8,
        )
        assert marginal.parse_rate == pytest.approx(1 / 32)
        assert marginal.whole_batch_failure_risk > 0.7
        assert marginal.usable is False

    def test_a_model_that_never_parses_is_not_usable(self):
        dead = st.summarise([rollout(1024, 1024, action=None, closed=False)] * 16, group_size=8)
        assert dead.parse_rate == 0.0
        assert dead.whole_batch_failure_risk == pytest.approx(1.0)
        assert dead.usable is False

    def test_mixing_budgets_in_one_summary_is_refused(self):
        with pytest.raises(ValueError, match="one budget at a time"):
            st.summarise(
                [rollout(1024, 10, action="C"), rollout(2048, 10, action="C")], group_size=8
            )

    def test_nothing_to_summarise_is_refused(self):
        with pytest.raises(ValueError, match="no rollouts"):
            st.summarise([], group_size=8)

    def test_the_record_carries_the_derived_numbers(self):
        # The JSON artifact has to be readable without re-deriving anything.
        record = st.summarise([rollout(2048, 10, action="C")] * 4, group_size=8).as_record()
        assert record["parse_rate"] == pytest.approx(1.0)
        assert record["whole_batch_failure_risk"] == pytest.approx(0.0)
        assert record["usable"] is True
        assert record["budget"] == 2048


class TestFormatTable:
    def test_the_table_marks_an_unusable_budget(self):
        summaries = [
            st.summarise([rollout(1024, 1024, action=None, closed=False)] * 8, group_size=8),
            st.summarise([rollout(2048, 30, action="C")] * 8, group_size=8),
        ]
        table = st.format_table(summaries)
        assert "NO" in table
        assert "yes" in table
        assert "1024" in table
        assert "2048" in table


class TestParseArgs:
    def test_flags_map_onto_the_config(self):
        config = st.parse_args(
            [
                "--model",
                "Qwen/Qwen3.5-2B",
                "--budgets",
                "512",
                "1024",
                "--prompts",
                "3",
                "--samples",
                "2",
                "--group",
                "4",
                "--no-thinking",
                "--temperature",
                "0.7",
                "--top-p",
                "0.8",
                "--top-k",
                "20",
            ]
        )
        assert config.model_id == "Qwen/Qwen3.5-2B"
        assert config.budgets == (512, 1024)
        assert config.n_prompts == 3
        assert config.n_samples == 2
        assert config.group_size == 4
        assert config.thinking is False
        assert config.temperature == pytest.approx(0.7)
        assert config.top_p == pytest.approx(0.8)
        assert config.top_k == 20

    def test_thinking_is_on_by_default(self):
        assert st.parse_args(["--model", "x"]).thinking is True

    def test_an_unstated_budget_comes_from_the_measured_termination_table(self):
        """The old (1024, 2048) default is the pair that parsed 0 of 16: it screened its own cap."""
        assert st.parse_args(["--model", "Qwen/Qwen3.5-2B"]).budgets == (24576,)
        assert st.parse_args(["--model", "unscreened/model"]).budgets == (
            MEASURED_TERMINATION_BUDGET,
        )

    def test_a_budget_cannot_be_left_unstated_in_the_config_itself(self):
        with pytest.raises(TypeError, match="budgets"):
            st.ScreenConfig(model_id="x")  # pyright: ignore[reportCallIssue]

    def test_a_model_is_required(self):
        with pytest.raises(SystemExit):
            st.parse_args([])

    def test_an_empty_budget_list_is_refused(self):
        with pytest.raises(ValueError, match="at least one completion budget"):
            st.ScreenConfig(model_id="x", budgets=())

    def test_a_nonpositive_budget_is_refused(self):
        with pytest.raises(ValueError, match="must be positive"):
            st.ScreenConfig(model_id="x", budgets=(0,))


class TestScalarEosRepair:
    """A checkpoint declaring several stop tokens reads as our own bug's signature if left alone."""

    class StubTokenizer:
        """Enough of a tokenizer to exercise the repair without a download."""

        def __init__(self, eos: str, eos_id: int, pad: str, pad_id: int, names: dict[int, str]):
            self.eos_token = eos
            self._eos_id = eos_id
            self.pad_token = pad
            self._pad_id = pad_id
            self._names = names

        @property
        def eos_token_id(self) -> int:
            return next(i for i, name in self._names.items() if name == self.eos_token)

        @property
        def pad_token_id(self) -> int:
            return next(i for i, name in self._names.items() if name == self.pad_token)

        def convert_ids_to_tokens(self, token_id: int) -> str:
            return self._names[token_id]

    @staticmethod
    def stub_generation_config(declared: object):
        class Stub:
            eos_token_id = declared

            @classmethod
            def from_pretrained(cls, _model_id: str) -> Stub:
                return cls()

        return Stub

    def minicpm_like(self):
        # The verified openbmb/MiniCPM5-1B shape: eos and pad are the same old token, while the
        # template actually terminates on a different one that generation_config also declares.
        return self.StubTokenizer(
            eos="</s>", eos_id=1, pad="</s>", pad_id=1, names={1: "</s>", 130073: "<|im_end|>"}
        )

    def test_the_real_terminator_replaces_a_pad_shaped_eos(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(pf, "GenerationConfig", self.stub_generation_config([1, 130073]))
        tokenizer = self.minicpm_like()
        record = pf.repair_scalar_eos(tokenizer, "stub/multi-eos")  # pyright: ignore[reportArgumentType]
        assert record["repaired"] is True
        assert tokenizer.eos_token == "<|im_end|>"
        assert tokenizer.eos_token_id == 130073
        assert tokenizer.pad_token_id == 1

    def test_a_single_declared_stop_token_is_left_alone(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(pf, "GenerationConfig", self.stub_generation_config(1))
        tokenizer = self.minicpm_like()
        record = pf.repair_scalar_eos(tokenizer, "stub/scalar-eos")  # pyright: ignore[reportArgumentType]
        assert record["repaired"] is False
        assert tokenizer.eos_token == "</s>"

    def test_a_checkpoint_without_a_generation_config_is_left_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # Qwen3.5-2B ships none, and an unhandled OSError here took out a screening run.
        class Missing:
            @classmethod
            def from_pretrained(cls, _model_id: str) -> None:
                raise OSError("does not appear to have a file named generation_config.json")

        monkeypatch.setattr(pf, "GenerationConfig", Missing)
        tokenizer = self.minicpm_like()
        record = pf.repair_scalar_eos(tokenizer, "stub/no-generation-config")  # pyright: ignore[reportArgumentType]
        assert record["repaired"] is False
        assert record["reason"] == "checkpoint ships no generation_config.json"

    def test_stop_and_pad_collapsing_after_repair_raises(self, monkeypatch: pytest.MonkeyPatch):
        # If every declared terminator IS the pad token, "stopped" and "padded" are the same
        # observation and no parse rate off that model means anything.
        monkeypatch.setattr(pf, "GenerationConfig", self.stub_generation_config([1, 1]))
        tokenizer = self.minicpm_like()
        record = pf.repair_scalar_eos(tokenizer, "stub/all-pad")  # pyright: ignore[reportArgumentType]
        assert record["repaired"] is False
        assert "pad token" in str(record["reason"])


class TestTerminationCdf:
    """One generous budget has to answer every smaller one, or budget sizing needs N runs."""

    def sample(self) -> list[st.Rollout]:
        # Four that finished at various lengths, two that never did.
        return [
            rollout(32768, 400, action="C"),
            rollout(32768, 3000, action="D"),
            rollout(32768, 9000, action="C"),
            rollout(32768, 20000, action="C"),
            rollout(32768, 32768, action=None, closed=False),
            rollout(32768, 32768, action=None, closed=False),
        ]

    def test_a_shorter_budget_is_answered_by_the_longer_run(self):
        points = {
            p.threshold: p for p in st.termination_cdf(self.sample(), (1024, 4096, 16384, 32768))
        }
        assert points[1024].closed_think_by == 1
        assert points[4096].closed_think_by == 2
        assert points[16384].closed_think_by == 3
        assert points[32768].closed_think_by == 4

    def test_fractions_are_over_all_rollouts_not_just_the_terminating_ones(self):
        # The decision is "what fraction of a real batch would be usable", so the denominator is
        # every rollout, including the ones that never stopped.
        point = st.termination_cdf(self.sample(), (32768,))[0]
        assert point.n_rollouts == 6
        assert point.closed_think_fraction == pytest.approx(4 / 6)
        assert point.parsed_fraction == pytest.approx(4 / 6)

    def test_thresholds_are_reported_in_order_and_the_budget_closes_them(self):
        points = st.termination_cdf(self.sample(), (16384, 1024, 4096))
        assert [p.threshold for p in points] == [1024, 4096, 16384, 32768]
        assert {p.budget for p in points} == {32768}

    def test_the_record_carries_the_derived_fractions(self):
        record = st.termination_cdf(self.sample(), (4096,))[0].as_record()
        assert record["threshold"] == 4096
        assert record["closed_think_by"] == 2
        assert record["closed_think_fraction"] == pytest.approx(2 / 6)

    def test_the_table_renders_every_threshold(self):
        table = st.format_cdf(st.termination_cdf(self.sample(), (1024, 8192)))
        assert "1024" in table
        assert "8192" in table


class TestTerminationCdfAcrossBudgets:
    """A screen that ran several budgets must not report one budget's rollouts against another's."""

    def mixed(self) -> list[st.Rollout]:
        """Two rollouts cut off at a 1024-token cap, two that closed under an 8192-token one."""
        return [
            rollout(1024, 1024, action=None, closed=False),
            rollout(1024, 1024, action=None, closed=False),
            rollout(8192, 3000, action="C"),
            rollout(8192, 5000, action="D"),
        ]

    def test_a_smaller_budget_does_not_dilute_a_larger_ones_fraction(self):
        """Pooling put the 1024-capped pair in the 8192 denominator, halving that fraction."""
        points = {
            (p.budget, p.threshold): p
            for p in st.termination_cdf_by_budget(self.mixed(), (1024, 8192))
        }
        assert points[(8192, 8192)].n_rollouts == 2
        assert points[(8192, 8192)].closed_think_fraction == pytest.approx(1.0)
        assert points[(8192, 8192)].parsed_fraction == pytest.approx(1.0)
        assert points[(1024, 1024)].n_rollouts == 2
        assert points[(1024, 1024)].closed_think_fraction == pytest.approx(0.0)

    def test_a_budget_says_nothing_about_a_threshold_it_could_not_reach(self):
        reported = {
            (p.budget, p.threshold)
            for p in st.termination_cdf_by_budget(self.mixed(), (1024, 8192))
        }
        assert (1024, 1024) in reported
        assert (1024, 8192) not in reported

    def test_pooling_budgets_into_one_distribution_is_refused(self):
        with pytest.raises(ValueError, match="one budget at a time"):
            st.termination_cdf(self.mixed(), (8192,))

    def test_no_rollouts_is_refused(self):
        with pytest.raises(ValueError, match="no rollouts is undefined"):
            st.termination_cdf([], (1024,))

    def test_the_screened_budget_is_always_the_last_threshold(self):
        # 24,576 is the 2B's measured floor and is in no threshold table, so without this the CDF
        # would stop at 16,384 and never report the uncensored fraction that finished at all.
        points = st.termination_cdf([rollout(24576, 20000, action="C")], (8192, 16384, 32768))
        assert [p.threshold for p in points] == [8192, 16384, 24576]
        assert points[-1].closed_think_fraction == pytest.approx(1.0)


class TestTheScreenSizesItsOwnDecodeWidth:
    """A screen decodes `--samples` sequences of one prompt at once, at up to a 32,768-token budget.

    The one hardcoded width left in the repo after `games.chunked_decode` was extracted: the screen
    handed the whole sample list to `model.generate` in a single call, so `--samples 64` at the
    Qwen3.5 cards' recommended budget put 2,097,152 completion tokens in flight against a measured
    knee of 524,288 -- 4x over, which is the region where a decode returned zero completions in 93
    minutes on a rented card. The screen is the stage that runs BEFORE a budget is trusted, so it is
    exactly the stage that gets pointed at a budget nothing has measured yet.
    """

    RENTED_96_GIB_BYTES: ClassVar[tuple[int, int]] = (int(90.8 * 1024**3), int(95.0 * 1024**3))

    @pytest.fixture
    def pretend_rented_card(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fake the card, so the widths asserted below do not depend on which GPU is free.

        Without this the test reads whatever `torch.cuda.mem_get_info` reports on the box it runs on,
        which is the local L4 here, no card at all in CI, and a rented 96 GiB card on the machine
        this matters for -- three different answers to the same assertion.
        """
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: self.RENTED_96_GIB_BYTES)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "a faked rented card")

    def test_a_wide_screen_is_split_at_the_measured_throughput_knee(
        self, pretend_rented_card: None
    ):
        del pretend_rented_card
        widths = st.sample_widths(64, budget=32768, model_id="Qwen/Qwen3.5-2B")
        assert sum(widths) == 64
        assert max(widths) <= knee_width_cap(max_new_tokens=32768)
        assert max(widths) * 32768 <= THROUGHPUT_KNEE_SEQUENCE_TOKENS
        assert len(widths) > 1, "a 64-sample screen at this budget has to become several calls"

    def test_the_default_screen_still_decodes_in_one_call(self, pretend_rented_card: None):
        """Four samples at the recommended budget are well inside the knee; do not slow that down."""
        del pretend_rented_card
        assert st.sample_widths(4, budget=32768, model_id="Qwen/Qwen3.5-2B") == [4]

    def test_a_short_budget_leaves_a_wide_screen_whole(self, pretend_rented_card: None):
        """The cap is in in-flight COMPLETION tokens, so it does not bind on a small budget."""
        del pretend_rented_card
        assert st.sample_widths(64, budget=512, model_id="Qwen/Qwen3.5-2B") == [64]

    def test_every_width_is_positive_so_the_loop_terminates(self, pretend_rented_card: None):
        del pretend_rented_card
        for n_samples in (1, 2, 3, 7, 8, 33, 64, 100):
            widths = st.sample_widths(n_samples, budget=32768, model_id="Qwen/Qwen3.5-2B")
            assert sum(widths) == n_samples
            assert all(width >= 1 for width in widths)

    def test_a_screen_of_no_samples_is_refused(self):
        with pytest.raises(ValueError, match="measures nothing"):
            st.sample_widths(0, budget=32768, model_id="Qwen/Qwen3.5-2B")
