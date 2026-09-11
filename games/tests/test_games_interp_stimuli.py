"""Structural checks on the interp contrast stimuli.

The claims worth mechanising are the two the stimuli's value rests on: that each pair really is
matched-stem (so a direction cannot be riding on stem differences), and that the counterbalance is
exactly balanced (so it cannot be riding on the first-printed-label preference, which is the largest
single determinant of behaviour on this corpus). Both guards are also exercised in the failing
direction -- a doctored row, a doctored continuation, a template that does not prefill ``<think>`` --
because a guard nobody has watched go red is not yet a guard.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import TYPE_CHECKING, Any, cast

import pytest

from games import interp_stimuli as stim
from games.prompts import (
    LABEL_PRINT_ORDER_CANONICAL,
    LABEL_PRINT_ORDER_SWAPPED,
    format_points,
    generate_prompt_rows,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

THINK_PREFILL = "<think>\n"
THINK_CLOSED = "<think>\n\n</think>\n\n"

NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")


class StubTokenizer:
    """The Qwen3.5 template behaviour in miniature: one turn per line, ``<think>`` prefilled."""

    def __init__(self, *, prefill_think: bool = True) -> None:
        self.prefill_think = prefill_think
        self.extra_template_kwargs: list[dict[str, Any]] = []

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        enable_thinking: bool = False,
        **kwargs: Any,
    ) -> str:
        self.extra_template_kwargs.append(dict(kwargs))
        assert not tokenize, "stimuli must be templated to text, not token ids"
        assert add_generation_prompt, "the continuation needs the assistant turn opened"
        turns = "".join(f"<|im_start|>{turn['role']}\n{turn['content']}\n" for turn in conversation)
        tail = THINK_PREFILL if (enable_thinking and self.prefill_think) else THINK_CLOSED
        return f"{turns}<|im_start|>assistant\n{tail}"


def as_tokenizer(stub: StubTokenizer) -> PreTrainedTokenizerBase:
    return cast("PreTrainedTokenizerBase", stub)


@pytest.fixture(scope="module")
def pairs() -> list[stim.ContrastPair]:
    return stim.build_all_pairs()


@pytest.fixture(scope="module")
def rendered(pairs: list[stim.ContrastPair]) -> list[stim.RenderedStimulus]:
    return stim.render_pairs(pairs, as_tokenizer(StubTokenizer()), chat_template_kwargs={})


def game_pairs(pairs: list[stim.ContrastPair]) -> list[stim.ContrastPair]:
    return [pair for pair in pairs if "game_id" in pair.provenance]


class TestInventory:
    def test_every_set_is_present_and_non_trivial(self, pairs: list[stim.ContrastPair]) -> None:
        counts = Counter(pair.set_name for pair in pairs)
        assert set(counts) == set(stim.SET_NAMES)
        assert all(count >= 30 for count in counts.values()), counts

    def test_pair_and_stimulus_ids_are_unique(
        self, pairs: list[stim.ContrastPair], rendered: list[stim.RenderedStimulus]
    ) -> None:
        pair_ids = [pair.pair_id for pair in pairs]
        assert len(set(pair_ids)) == len(pair_ids)
        ids = [item.stimulus_id for item in rendered]
        assert len(set(ids)) == len(ids)
        assert len(rendered) == 2 * len(pairs)

    def test_both_sides_are_rendered_for_every_pair(
        self, rendered: list[stim.RenderedStimulus]
    ) -> None:
        sides_by_pair: dict[str, set[str]] = {}
        for item in rendered:
            sides_by_pair.setdefault(item.pair_id, set()).add(item.side)
        assert all(sides == set(stim.SIDES) for sides in sides_by_pair.values())

    def test_grouping_into_positive_negative_covers_every_set(
        self, rendered: list[stim.RenderedStimulus]
    ) -> None:
        grouped = stim.contrastive_pairs(rendered)
        assert set(grouped) == set(stim.SET_NAMES)
        assert sum(len(items) for items in grouped.values()) == len(rendered) // 2
        assert all(pair.positive != pair.negative for items in grouped.values() for pair in items)

    def test_grouping_refuses_a_pair_missing_a_side(
        self, rendered: list[stim.RenderedStimulus]
    ) -> None:
        orphaned = [item for item in rendered if item.side == stim.SIDE_A]
        with pytest.raises(ValueError, match="missing side"):
            stim.contrastive_pairs(orphaned)


class TestMatchedStem:
    def test_the_two_sides_share_the_stem_and_the_shared_reasoning_exactly(
        self, pairs: list[stim.ContrastPair]
    ) -> None:
        tokenizer = as_tokenizer(StubTokenizer())
        for pair in pairs:
            shared = (
                stim.templated_stem(tokenizer, pair.stem, chat_template_kwargs={})
                + pair.think_prefix
            )
            side_a, side_b = stim.render_pair(pair, tokenizer, chat_template_kwargs={})
            assert side_a.text.startswith(shared), pair.pair_id
            assert side_b.text.startswith(shared), pair.pair_id
            assert side_a.text != side_b.text
            assert side_a.text[len(shared) :] == pair.continuation_a
            assert side_b.text[len(shared) :] == pair.continuation_b

    def test_the_continuations_are_close_in_length(self, pairs: list[stim.ContrastPair]) -> None:
        """Length is the confound a matched stem does not remove, so it is bounded here."""
        for pair in pairs:
            shorter, longer = sorted((len(pair.continuation_a), len(pair.continuation_b)))
            assert longer <= 1.35 * shorter, pair.pair_id

    def test_every_number_a_game_continuation_quotes_is_a_cell_of_its_own_table(
        self, pairs: list[stim.ContrastPair]
    ) -> None:
        for pair in game_pairs(pairs):
            table = {
                format_points(value)
                for value in _payoffs_of(pair.provenance["prompt_id"], pair.provenance)
            }
            quoted = set(
                NUMBER_PATTERN.findall(
                    pair.think_prefix + pair.continuation_a + pair.continuation_b
                )
            )
            assert quoted <= table, (pair.pair_id, sorted(quoted - table))
            assert quoted, pair.pair_id


class TestCounterbalance:
    @pytest.mark.parametrize(
        "set_name", [stim.SET_COOPERATE_VS_DEFECT, stim.SET_CORRELATED_VS_INDEPENDENT]
    )
    def test_the_full_cross_of_nuisance_factors_is_uniform(
        self, pairs: list[stim.ContrastPair], set_name: str
    ) -> None:
        rows = [pair.provenance for pair in pairs if pair.set_name == set_name]
        cross = Counter(
            (
                row["payoff_variant"],
                row["label_print_order"],
                row["coop_label_prints_first"],
            )
            for row in rows
        )
        assert len(cross) == 8, cross
        assert len(set(cross.values())) == 1, cross

    def test_both_print_orders_appear_in_the_game_sets(
        self, pairs: list[stim.ContrastPair]
    ) -> None:
        orders = Counter(pair.provenance["label_print_order"] for pair in game_pairs(pairs))
        assert set(orders) == {LABEL_PRINT_ORDER_CANONICAL, LABEL_PRINT_ORDER_SWAPPED}
        assert len(set(orders.values())) == 1, orders

    @pytest.mark.parametrize(
        "set_name", [stim.SET_COOPERATE_VS_DEFECT, stim.SET_CORRELATED_VS_INDEPENDENT]
    )
    def test_both_game_sets_are_balanced_across_the_two_briefings(
        self, pairs: list[stim.ContrastPair], set_name: str
    ) -> None:
        framings = Counter(
            pair.provenance["counterpart_framing"] for pair in pairs if pair.set_name == set_name
        )
        assert framings["correlated-instance"] == framings["recorded-decision"] > 0

    def test_both_option_orders_appear_in_the_decision_set(
        self, pairs: list[stim.ContrastPair]
    ) -> None:
        orders = Counter(
            pair.provenance["option_order"]
            for pair in pairs
            if pair.set_name == stim.SET_CAUSAL_VS_FUNCTIONAL
        )
        assert orders[stim.OPTION_ORDER_FUNCTIONAL_FIRST] == orders[stim.OPTION_ORDER_CAUSAL_FIRST]
        assert min(orders.values()) == len(stim.DECISION_SCENARIOS)

    def test_a_row_whose_mapping_and_print_order_disagree_is_refused(self) -> None:
        """The guard, watched going red: flip the paid-for label and the derivation must object."""
        honest = {
            "prompt_id": f"twin-pd--frame--temptation-2--{stim.COOP_INDEX_FIRST}",
            "label_print_order": LABEL_PRINT_ORDER_CANONICAL,
            "label_a": "SHORT",
            "label_b": "LONG",
            "coop_label": "SHORT",
        }
        stim._assert_counterbalance_agrees(honest)
        doctored = {**honest, "coop_label": "LONG"}
        with pytest.raises(ValueError, match="counterbalance disagreement"):
            stim._assert_counterbalance_agrees(doctored)


class TestCommittedLabelIsNotAGiveaway:
    """The in-frame label a side names must not identify which side it is.

    The nuisance cross in ``TestCounterbalance`` is uniform at any ``CELL_OFFSET_PER_VARIANT``, so it
    cannot see this. What it misses is that the cross is balanced *between* frames while the labels
    are frame-specific: at an even offset each frame appears under one label mapping only, side A
    ends on one twenty-word vocabulary and side B on a disjoint one, and a direction read at the
    final position is satisfied by token identity alone. Both checks below fail at offset 2 and pass
    at 1, which is the sabotage this file's other guards get.
    """

    @pytest.mark.parametrize(
        "set_name", [stim.SET_COOPERATE_VS_DEFECT, stim.SET_CORRELATED_VS_INDEPENDENT]
    )
    def test_every_frame_contributes_both_of_its_label_mappings(
        self, pairs: list[stim.ContrastPair], set_name: str
    ) -> None:
        """The root-cause invariant: no frame may appear under one paid-for label only."""
        mappings: dict[tuple[str, str], set[str]] = {}
        for pair in pairs:
            if pair.set_name != set_name:
                continue
            frame = (pair.provenance["game_id"], pair.provenance["reskin_id"])
            mappings.setdefault(frame, set()).add(_coop_index_of(pair))
        assert mappings, set_name
        one_sided = sorted(frame for frame, seen in mappings.items() if len(seen) < 2)
        assert not one_sided, (set_name, one_sided)

    def test_the_two_sides_commit_to_the_same_labels_across_the_coarse_set(
        self, pairs: list[stim.ContrastPair]
    ) -> None:
        """The observable consequence: the committed-label multisets have to match exactly.

        Asserted on the coarse set only, where every continuation ends on a label. The sharpened
        set's continuations end on figures or pronouns by construction, so the same equality does
        not hold there and the per-frame check above is what covers it.
        """
        coarse = [pair for pair in pairs if pair.set_name == stim.SET_COOPERATE_VS_DEFECT]
        committed_a = Counter(_committed_label(pair.continuation_a) for pair in coarse)
        committed_b = Counter(_committed_label(pair.continuation_b) for pair in coarse)
        assert committed_a == committed_b, {
            "only-A": dict(committed_a - committed_b),
            "only-B": dict(committed_b - committed_a),
        }

    def test_the_offset_that_reintroduces_the_giveaway_is_rejected(self) -> None:
        """The constant carries a parity requirement, so state it where a future edit will see it."""
        assert stim.CELL_OFFSET_PER_VARIANT % 2 == 1, stim.CELL_OFFSET_PER_VARIANT


GAME_SET_PAIR_FLOOR = 80


class TestGameSetScale:
    """The two game sets have to be big enough, and built from one pool, to be worth comparing."""

    @pytest.mark.parametrize(
        "set_name", [stim.SET_COOPERATE_VS_DEFECT, stim.SET_CORRELATED_VS_INDEPENDENT]
    )
    def test_each_game_set_reaches_the_pair_count_that_cleared_the_noise_floor(
        self, pairs: list[stim.ContrastPair], set_name: str
    ) -> None:
        """40 pairs sat at the split-half floor at 4B; ~80 is where directions lifted off it."""
        count = sum(1 for pair in pairs if pair.set_name == set_name)
        assert count >= GAME_SET_PAIR_FLOOR, (set_name, count)

    def test_the_two_game_sets_are_built_from_the_same_stems(
        self, pairs: list[stim.ContrastPair]
    ) -> None:
        """Shared stems are what make the coarse direction residualisable against the sharpened one.

        Pair-by-pair residualising needs the same stem under both contrasts; if the sets ever drift
        apart the analysis silently becomes an unpaired comparison, so the pool identity is asserted.
        """
        pools = {
            set_name: {pair.provenance["prompt_id"] for pair in pairs if pair.set_name == set_name}
            for set_name in (stim.SET_COOPERATE_VS_DEFECT, stim.SET_CORRELATED_VS_INDEPENDENT)
        }
        coarse, sharpened = pools.values()
        assert coarse == sharpened, sorted(coarse ^ sharpened)


class TestCitedCells:
    """The cell-citation regime is a split an analysis conditions on, so it has to be real."""

    def test_the_coarse_set_is_split_evenly_between_the_two_citation_regimes(
        self, pairs: list[stim.ContrastPair]
    ) -> None:
        regimes = Counter(
            pair.provenance["cited_cells"]
            for pair in pairs
            if pair.set_name == stim.SET_COOPERATE_VS_DEFECT
        )
        assert regimes[stim.CITED_CELLS_MATCHED] == regimes[stim.CITED_CELLS_SPLIT] > 0

    def test_the_frozen_commitment_pairs_state_their_fixed_column_in_the_shared_prefix(
        self, pairs: list[stim.ContrastPair]
    ) -> None:
        """Stated once in text both sides share, which is what keeps them a matched-column pair."""
        frozen = [
            pair
            for pair in pairs
            if pair.set_name == stim.SET_COOPERATE_VS_DEFECT
            and pair.provenance.get("counterpart_framing") == "recorded-decision"
        ]
        assert frozen
        for pair in frozen:
            assert "already down" in pair.think_prefix, pair.pair_id

    def test_a_matched_column_pair_really_quotes_the_same_figures_on_both_sides(
        self, pairs: list[stim.ContrastPair]
    ) -> None:
        """Checked against the rendered numbers, not the label the builder attached.

        The provenance key is derived from the templates, so this closes the loop on the derivation:
        if a form were edited to cite an extra cell on one side only, the label would still say
        matched unless the numbers themselves are compared.
        """
        matched = [
            pair for pair in pairs if pair.provenance.get("cited_cells") == stim.CITED_CELLS_MATCHED
        ]
        assert matched
        for pair in matched:
            quoted_a = set(NUMBER_PATTERN.findall(pair.continuation_a))
            quoted_b = set(NUMBER_PATTERN.findall(pair.continuation_b))
            assert quoted_a == quoted_b, pair.pair_id

    def test_a_split_regime_pair_quotes_a_figure_one_side_does_not(
        self, pairs: list[stim.ContrastPair]
    ) -> None:
        """The other half of the split has to actually differ, or the regime label means nothing."""
        split = [
            pair for pair in pairs if pair.provenance.get("cited_cells") == stim.CITED_CELLS_SPLIT
        ]
        assert split
        for pair in split:
            quoted_a = set(NUMBER_PATTERN.findall(pair.continuation_a))
            quoted_b = set(NUMBER_PATTERN.findall(pair.continuation_b))
            assert quoted_a != quoted_b, pair.pair_id

    def test_the_regime_derivation_reports_split_when_the_sides_cite_different_cells(self) -> None:
        """Watched going red: a form pair that cites a cell on one side only is not matched-column."""
        matched = stim._cited_cells(
            "I take the {p_both_coop} over the {p_other_alone}, so {coop}.",
            "I take the {p_other_alone} over the {p_both_coop}, so {other}.",
        )
        assert matched == stim.CITED_CELLS_MATCHED
        lopsided = stim._cited_cells(
            "I take the {p_both_coop} over the {p_both_other}, so {coop}.",
            "I take the {p_other_alone} over the {p_both_coop}, so {other}.",
        )
        assert lopsided == stim.CITED_CELLS_SPLIT


class TestRenderingGuards:
    def test_a_template_that_does_not_prefill_think_is_refused(self) -> None:
        """Watched going red: without the prefill the continuation would not be model reasoning."""
        tokenizer = as_tokenizer(StubTokenizer(prefill_think=False))
        with pytest.raises(ValueError, match="does not open <think>"):
            stim.templated_stem(tokenizer, "anything", chat_template_kwargs={})

    def test_loaded_vocabulary_in_a_continuation_is_refused(self) -> None:
        """Watched going red: the no-giveaway-token rule has to survive a future edit."""
        leaky = stim.ContrastPair(
            pair_id="doctored",
            set_name=stim.SET_COOPERATE_VS_DEFECT,
            stem="A briefing with two labels.",
            think_prefix="",
            continuation_a="So I cooperate with them.",
            continuation_b="So I go with the other label.",
            provenance={},
        )
        with pytest.raises(ValueError, match="loaded vocabulary"):
            stim.render_pair(leaky, as_tokenizer(StubTokenizer()), chat_template_kwargs={})

    def test_pinned_template_kwargs_reach_the_template(
        self, pairs: list[stim.ContrastPair]
    ) -> None:
        stub = StubTokenizer()
        stim.render_pair(
            pairs[0], as_tokenizer(stub), chat_template_kwargs={"reasoning_effort": "medium"}
        )
        assert {"reasoning_effort": "medium"} in stub.extra_template_kwargs


class TestDecisionScenarios:
    def test_each_scenario_prints_both_of_its_options_in_the_order_named(self) -> None:
        """Read the order off the offered-options clause, since a setup may mention them first."""
        for scenario in stim.DECISION_SCENARIOS:
            for option_order, expected_first in (
                (stim.OPTION_ORDER_FUNCTIONAL_FIRST, scenario.functional_option),
                (stim.OPTION_ORDER_CAUSAL_FIRST, scenario.causal_option),
            ):
                offered = stim._decision_stem(scenario, option_order).split("You may ", 1)[1]
                assert scenario.functional_option in offered, scenario.scenario_id
                assert scenario.causal_option in offered, scenario.scenario_id
                assert offered.startswith(expected_first), (scenario.scenario_id, option_order)

    def test_each_reading_commits_to_its_own_option(self) -> None:
        """Every content word of the option has to appear in the reading that chooses it.

        Not literal containment: several readings shift person ("narrow your gate" chosen as "So I
        narrow my gate"), which is the natural first-person form and should not be flattened.
        """
        for scenario in stim.DECISION_SCENARIOS:
            for option, reading in (
                (scenario.functional_option, scenario.functional_reading),
                (scenario.causal_option, scenario.causal_reading),
            ):
                missing = [word for word in _content_words(option) if word not in reading.lower()]
                assert not missing, (scenario.scenario_id, missing)
                assert reading.rstrip().endswith("."), scenario.scenario_id

    def test_scenario_ids_are_unique(self) -> None:
        ids = [scenario.scenario_id for scenario in stim.DECISION_SCENARIOS]
        assert len(set(ids)) == len(ids)


class TestWriting:
    def test_the_written_records_carry_the_keys_the_harness_reads(
        self, rendered: list[stim.RenderedStimulus], tmp_path: Any
    ) -> None:
        stimuli_path, provenance_path = stim.write_stimuli(rendered[:4], tmp_path)
        records = [json.loads(line) for line in stimuli_path.read_text().splitlines()]
        assert all(
            set(record) == {"id", "set", "side", "pair_id", "text", "assistant_prefix"}
            for record in records
        )
        provenance = [json.loads(line) for line in provenance_path.read_text().splitlines()]
        assert [item["id"] for item in provenance] == [item["id"] for item in records]
        assert all(item["stance"] for item in provenance)

    def test_the_emitted_text_is_untemplated_so_the_harness_does_not_template_it_twice(
        self, rendered: list[stim.RenderedStimulus]
    ) -> None:
        """The record's ``text`` must be the raw stem, because the harness templates what it reads.

        A pre-rendered string here would be wrapped in a second user turn by
        ``capture.py:render_prompt``, burying the teacher-forced reasoning inside a user message
        instead of leaving it after the assistant turn's opening ``<think>``. Nothing downstream
        raises on that, so the shape of the emitted record is the only place it can be caught.
        """
        for item in rendered:
            record = item.record()
            assert "<|im_start|>" not in record["text"], item.stimulus_id
            assert "<think>" not in record["text"], item.stimulus_id
            assert record["assistant_prefix"], item.stimulus_id

    def test_the_harness_reconstructs_the_rendered_string_byte_for_byte(
        self, rendered: list[stim.RenderedStimulus]
    ) -> None:
        """Templating the emitted stem and appending the prefix must reproduce ``text`` exactly.

        This is the invariant that makes the split safe: the measured string is unchanged by emitting
        it in two pieces, so the harness's own render is the string these stimuli were designed as.
        """
        tokenizer = as_tokenizer(StubTokenizer())
        for item in rendered:
            record = item.record()
            reconstructed = (
                stim.templated_stem(tokenizer, record["text"], chat_template_kwargs={})
                + record["assistant_prefix"]
            )
            assert reconstructed == item.text, item.stimulus_id


POSSESSIVES = frozenset({"your", "their"})
MIN_CONTENT_WORD_LEN = 4


def _content_words(option: str) -> list[str]:
    """The option's distinctive words, dropping short function words and possessives."""
    return [
        word
        for word in option.lower().split()
        if len(word) >= MIN_CONTENT_WORD_LEN and word not in POSSESSIVES
    ]


def _coop_index_of(pair: stim.ContrastPair) -> str:
    """The ``coop0``/``coop1`` tag naming which of the frame's labels the reward pays for."""
    return pair.provenance["prompt_id"].rsplit("--", maxsplit=1)[-1]


def _committed_label(continuation: str) -> str:
    """The in-frame label a coarse-set continuation ends on, stripped of trailing punctuation."""
    return continuation.rstrip(". ").rsplit(maxsplit=1)[-1]


def _payoffs_of(prompt_id: str, provenance: Any) -> list[float]:
    """The four own-credit payoffs of the row this pair was built from."""
    rows = generate_prompt_rows(
        provenance["game_id"],
        stim.RENDER_GRADING,
        split=provenance["split"],
        label_print_order=provenance["label_print_order"],
    )
    row = next(row for row in rows if row["prompt_id"] == prompt_id)
    return [row["payoff_cc"], row["payoff_cd"], row["payoff_dc"], row["payoff_dd"]]
