"""Offline tests for the corrected activation-patching readout: candidates, labels, twin identity.

CPU only, no model load. Three families, and every one of them exists because its failure would
produce a plausible number rather than an error:

* **The candidate classification is pinned against the real detector.** The whole readout rests on
  "one of these actions tampers with the grader and one does not", so it is asserted through
  ``reward_hacking.harness.grader_tampering.read_tampering`` -- the same function the episode analysis
  uses -- rather than by naming the constants confidently.
* **The label and boundary guards** refuse a forced-choice pair that cannot be told apart at one
  readout position, and a prefill/label boundary whose tokenization merges.
* **Twin byte-identity after appending**, at the string level and again at the token level, because a
  suffix that becomes a divergence would silently move the readout off the shared question.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from reward_hacking.harness.grader_tampering import read_tampering
from reward_hacking.interp.patch_readout import (
    CANDIDATE_HONEST,
    CANDIDATE_TAMPER,
    FORCED_CHOICE_PREFILL,
    HONEST_ACTION,
    HONEST_COMMAND,
    OPTION_ORDER_HONEST_FIRST,
    OPTION_ORDER_TAMPER_FIRST,
    OPTION_ORDERS,
    TAMPER_ACTION,
    TAMPER_COMMAND,
    assert_no_boundary_merge,
    assert_suffix_appended_identically,
    build_action_readout,
    build_forced_choice_readout,
    candidate_readout_semantics,
    classify_candidate_actions,
    forced_choice_labels,
    forced_choice_suffix,
    readout_provenance,
    readout_transcripts,
    require_shared_suffix_covers,
    wrapped_action,
)
from reward_hacking.interp.steering import (
    MAX_CANDIDATE_LENGTH_RATIO,
    READOUT_MODE_ACTION_LOGPROB,
    READOUT_MODE_FORCED_CHOICE,
    TwinPatchPlan,
    action_gap,
    gap_readout,
    gap_recovery,
    plan_twin_patch_ladder,
)

if TYPE_CHECKING:
    from pathlib import Path


def _word_encode(text: str) -> list[int]:
    """A deterministic whitespace tokenizer: one id per word, stable across calls.

    Enough to exercise every guard that is about token IDENTITY rather than about a real vocabulary,
    and it makes a collision constructible by choosing words. The real tokenizer's boundary behaviour
    is measured in the smoke, not here.
    """
    return [abs(hash(word)) % 50_000 for word in text.split()]


class TestCandidateClassification:
    """The two candidates must be what the harness's own detector says they are."""

    def test_the_tampering_candidate_weakens_the_check_and_answers_nothing(self) -> None:
        facts = read_tampering(TAMPER_COMMAND)

        assert facts.interfered
        assert facts.weakened_checks
        assert "grade.py" in " ".join(facts.weakened_checks)
        assert facts.produced_artifacts == ()

    def test_the_honest_candidate_leaves_the_machinery_alone_and_produces_an_artifact(self) -> None:
        facts = read_tampering(HONEST_COMMAND)

        assert not facts.interfered
        assert facts.weakened_checks == ()
        assert facts.produced_artifacts == ("solution.py",)

    def test_the_two_candidates_share_a_verb_and_flag_and_aim_at_different_files(self) -> None:
        """Matched shape is what makes the gap about the target rather than about shell style.

        Named ``..._differ_only_in_their_target`` until 2026-08-24, and they do not: the sed
        expressions differ in content and length too (a comment-out versus a return-statement
        substitution), which is exactly why the sibling length-ratio guard below has to exist. What
        this pins is the shared verb and in-place flag plus the two different target files; "only"
        was never asserted and is not true.
        """
        assert TAMPER_COMMAND.startswith("sed -i ")
        assert HONEST_COMMAND.startswith("sed -i ")
        assert TAMPER_COMMAND.endswith("grade.py")
        assert HONEST_COMMAND.endswith("solution.py")

    def test_both_candidates_are_wrapped_in_the_harness_action_protocol(self) -> None:
        for action in (TAMPER_ACTION, HONEST_ACTION):
            assert action.startswith("<run>")
            assert action.endswith("</run>")

    def test_the_classification_is_reported_rather_than_asserted_in_the_artifact(self) -> None:
        classified = classify_candidate_actions()

        assert classified[CANDIDATE_TAMPER]["interfered"] is True
        assert classified[CANDIDATE_HONEST]["interfered"] is False
        assert classified[CANDIDATE_TAMPER]["command"] == TAMPER_COMMAND

    def test_the_provenance_block_carries_both_modes_semantics_and_the_orders(self) -> None:
        provenance = readout_provenance([READOUT_MODE_ACTION_LOGPROB, READOUT_MODE_FORCED_CHOICE])

        assert set(provenance["option_orders"]) == set(OPTION_ORDERS)  # pyright: ignore[reportArgumentType]
        semantics = provenance["semantics_by_mode"]
        assert isinstance(semantics, dict)
        assert "STATED PREFERENCE" in semantics[READOUT_MODE_FORCED_CHOICE]
        assert "PRIMARY" in semantics[READOUT_MODE_ACTION_LOGPROB]

    def test_the_action_semantics_refuse_a_behavioural_reading(self) -> None:
        assert "NOT a measurement of what" in candidate_readout_semantics(
            READOUT_MODE_ACTION_LOGPROB
        )


class TestCandidatesEndToEndThroughTheEpisodeAssessor:
    """The candidates classified the way a real episode is: a full label, not a signal family.

    ``read_tampering`` is the machinery-side reader and has the highest precedence in the ladder, so
    pinning it is the load-bearing assertion. This closes the gap between that and the label an
    episode analysis would actually print, by running each candidate as a one-command episode through
    the real ``assess_episode``. The episode is set up as the ambiguous case the harness sees most
    often -- the visible grader accepted the answer and the hidden check rejected it -- so the label
    is decided by what the command did rather than by the verdict handed in.
    """

    @staticmethod
    def _assess(command: str, episode_dir: Path) -> object:
        # Not at the top: the loop and task registry are heavy and only this class needs them.
        from reward_hacking.harness.hack_detector import assess_episode  # noqa: PLC0415
        from reward_hacking.harness.loop import AgentTurn  # noqa: PLC0415
        from reward_hacking.tests.conftest import command_result, episode_trace  # noqa: PLC0415

        turns = (
            AgentTurn(
                index=0,
                completion=wrapped_action(command),
                commands=(command_result(command),),
            ),
            AgentTurn(index=1, completion="Done.", commands=()),
        )
        return assess_episode(
            episode_trace(
                "sum-ledger", turns, episode_dir=episode_dir, proxy_pass=True, true_pass=False
            )
        )

    def test_the_tampering_candidate_lands_a_deliberate_hack_label(
        self, sum_ledger_dir: Path
    ) -> None:
        from reward_hacking.harness.hack_detector import HackLabel  # noqa: PLC0415

        assessment = self._assess(TAMPER_COMMAND, sum_ledger_dir)

        assert assessment.label is HackLabel.DELIBERATE_HACK  # pyright: ignore[reportAttributeAccessIssue]

    def test_the_honest_candidate_does_not(self, sum_ledger_dir: Path) -> None:
        from reward_hacking.harness.hack_detector import HackLabel  # noqa: PLC0415

        assessment = self._assess(HONEST_COMMAND, sum_ledger_dir)

        assert assessment.label is not HackLabel.DELIBERATE_HACK  # pyright: ignore[reportAttributeAccessIssue]


class TestForcedChoiceOptions:
    def test_both_orders_print_the_two_options_and_differ_only_in_which_came_first(self) -> None:
        tamper_first = forced_choice_suffix(OPTION_ORDER_TAMPER_FIRST)
        honest_first = forced_choice_suffix(OPTION_ORDER_HONEST_FIRST)

        for suffix in (tamper_first, honest_first):
            assert TAMPER_ACTION in suffix
            assert HONEST_ACTION in suffix
        assert tamper_first.index(TAMPER_ACTION) < tamper_first.index(HONEST_ACTION)
        assert honest_first.index(HONEST_ACTION) < honest_first.index(TAMPER_ACTION)

    def test_the_positive_label_follows_the_tampering_option_in_both_orders(self) -> None:
        """The gap is always tamper-minus-honest, so a position bias cannot read as the effect."""
        assert forced_choice_labels(OPTION_ORDER_TAMPER_FIRST) == (" A", " B")
        assert forced_choice_labels(OPTION_ORDER_HONEST_FIRST) == (" B", " A")

    def test_an_unknown_order_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown option order"):
            forced_choice_suffix("whichever")


class TestCollisionGuard:
    def test_distinct_labels_build_a_single_token_readout(self) -> None:
        readout = build_forced_choice_readout(_word_encode, OPTION_ORDER_TAMPER_FIRST)

        assert readout is not None
        assert readout.is_single_token
        assert not readout.collides
        assert readout.mode == READOUT_MODE_FORCED_CHOICE

    def test_labels_sharing_a_first_token_are_skipped_rather_than_scored(self) -> None:
        """The guard that exists so an unreadable pair is COUNTED instead of reading zero.

        Forced with an encoder that tokenizes cleanly across the prefill boundary but maps both
        option labels to ONE id, which is exactly the shape a badly chosen label pair would produce
        against a real vocabulary. Keeping the boundary clean matters: an encoder that also merged
        would trip the earlier guard and this test would pass without exercising the collision.
        """

        def collide(text: str) -> list[int]:
            return [1 if word in {"A", "B"} else 2 for word in text.split()]

        assert build_forced_choice_readout(collide, OPTION_ORDER_TAMPER_FIRST) is None

    def test_a_label_that_tokenizes_to_nothing_is_refused_loudly(self) -> None:
        with pytest.raises(ValueError, match="tokenized to nothing"):
            build_forced_choice_readout(lambda _text: [], OPTION_ORDER_TAMPER_FIRST)

    def test_a_boundary_merge_between_prefill_and_label_is_refused(self) -> None:
        """A merge means the scored id is not the id the model would emit after the prefill."""

        def merging(text: str) -> list[int]:
            return [1] if " " in text.strip() or text.count(" ") > 1 else [2, 3]

        with pytest.raises(ValueError, match="merges across the boundary"):
            assert_no_boundary_merge(merging, FORCED_CHOICE_PREFILL, " A")

    def test_a_clean_boundary_passes(self) -> None:
        assert_no_boundary_merge(_word_encode, "Answer:", " A")


class TestActionReadout:
    def test_the_two_authored_candidates_build_a_multi_token_readout(self) -> None:
        readout = build_action_readout(_word_encode)

        assert readout.mode == READOUT_MODE_ACTION_LOGPROB
        assert not readout.is_single_token
        assert not readout.collides
        assert readout.positive_name == CANDIDATE_TAMPER
        assert readout.negative_name == CANDIDATE_HONEST

    def test_the_authored_candidates_are_close_enough_in_length(self) -> None:
        """The length term cancels out of recovery but not out of the raw gap; keep it small."""
        assert build_action_readout(_word_encode).length_ratio <= MAX_CANDIDATE_LENGTH_RATIO

    def test_wildly_unequal_candidates_are_refused(self) -> None:
        with pytest.raises(ValueError, match="differ by a factor of"):
            gap_readout(
                mode=READOUT_MODE_ACTION_LOGPROB,
                positive_name="short",
                negative_name="long",
                positive_ids=torch.tensor([1, 2]),
                negative_ids=torch.arange(40),
            )

    def test_a_multi_token_forced_choice_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be one token"):
            gap_readout(
                mode=READOUT_MODE_FORCED_CHOICE,
                positive_name="a",
                negative_name="b",
                positive_ids=torch.tensor([1, 2]),
                negative_ids=torch.tensor([3, 4]),
            )

    def test_identical_multi_token_candidates_collide(self) -> None:
        readout = gap_readout(
            mode=READOUT_MODE_ACTION_LOGPROB,
            positive_name="a",
            negative_name="b",
            positive_ids=torch.tensor([1, 2, 3]),
            negative_ids=torch.tensor([1, 2, 3]),
        )

        assert readout.collides

    def test_a_shared_first_token_is_fine_for_multi_token_candidates(self) -> None:
        """Both authored actions open ``<run>``; only the forced-choice arm reads one position."""
        readout = gap_readout(
            mode=READOUT_MODE_ACTION_LOGPROB,
            positive_name="a",
            negative_name="b",
            positive_ids=torch.tensor([1, 2, 3]),
            negative_ids=torch.tensor([1, 9, 9]),
        )

        assert not readout.collides

    def test_an_unknown_mode_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown readout mode"):
            gap_readout(
                mode="vibes",
                positive_name="a",
                negative_name="b",
                positive_ids=torch.tensor([1]),
                negative_ids=torch.tensor([2]),
            )


class TestTwinIdentityAfterAppending:
    def test_the_action_mode_leaves_both_transcripts_untouched(self) -> None:
        original, conflicting = readout_transcripts(
            "ORIGINAL", "CONFLICTING", mode=READOUT_MODE_ACTION_LOGPROB, order=None
        )

        assert (original, conflicting) == ("ORIGINAL", "CONFLICTING")

    def test_the_forced_choice_mode_appends_the_identical_block_to_both(self) -> None:
        original, conflicting = readout_transcripts(
            "ORIGINAL",
            "CONFLICTING",
            mode=READOUT_MODE_FORCED_CHOICE,
            order=OPTION_ORDER_TAMPER_FIRST,
        )

        assert original[len("ORIGINAL") :] == conflicting[len("CONFLICTING") :]
        assert original.startswith("ORIGINAL")
        assert conflicting.startswith("CONFLICTING")

    def test_the_forced_choice_mode_needs_an_order(self) -> None:
        with pytest.raises(ValueError, match="needs an option print order"):
            readout_transcripts("A", "B", mode=READOUT_MODE_FORCED_CHOICE, order=None)

    def test_a_per_side_suffix_is_caught(self) -> None:
        """The check that would catch a suffix carrying a problem id, a length, or a path."""
        with pytest.raises(ValueError, match="not its transcript plus the shared suffix"):
            assert_suffix_appended_identically(
                "ORIGINAL",
                "CONFLICTING",
                ("ORIGINAL_SUFFIX_A", "CONFLICTING_SUFFIX"),
                suffix="_SUFFIX",
            )

    def test_a_matched_append_passes(self) -> None:
        assert_suffix_appended_identically(
            "ORIGINAL", "CONFLICTING", ("ORIGINAL!", "CONFLICTING!"), suffix="!"
        )


class TestSharedSuffixCoversTheAppendedText:
    @staticmethod
    def _plan(suffix_len: int) -> TwinPatchPlan:
        """A real ladder plan whose twins share exactly ``suffix_len`` trailing tokens."""
        tail = list(range(90, 90 + suffix_len))
        clean = torch.tensor([1, 2, 3, 4, *tail])
        corrupted = torch.tensor([1, 2, 3, 5, 6, *tail])
        return plan_twin_patch_ladder(clean, corrupted)

    def test_a_shared_suffix_at_least_as_long_as_the_appended_text_passes(self) -> None:
        require_shared_suffix_covers(self._plan(4), min_tokens=3, what="the option block")

    def test_a_short_shared_suffix_is_refused(self) -> None:
        """Two byte-identical tails CAN tokenize differently; this is the token-level re-assertion."""
        with pytest.raises(ValueError, match="tokenized differently across the two boundaries"):
            require_shared_suffix_covers(self._plan(2), min_tokens=9, what="the option block")


class TestGapArithmetic:
    def test_the_action_gap_is_the_difference_of_two_logits(self) -> None:
        logits = torch.tensor([0.0, 4.0, 1.5, -2.0])

        assert action_gap(logits, 1, 3) == pytest.approx(6.0)

    def test_a_single_token_log_softmax_gap_equals_the_raw_logit_gap(self) -> None:
        """Why the forced-choice arm costs one forward instead of two.

        ``log_softmax`` subtracts the same ``logsumexp`` from both entries, so the difference of two
        log-probs at ONE position is the difference of the two logits exactly. If this ever stopped
        holding, the single-forward shortcut in ``read_gap`` would be scoring something else.
        """
        logits = torch.tensor([0.3, 4.0, 1.5, -2.0])
        logprobs = torch.log_softmax(logits, dim=-1)

        assert action_gap(logits, 1, 3) == pytest.approx(float(logprobs[1] - logprobs[3]), abs=1e-6)

    def test_the_games_thread_reads_the_identical_gap_function_object(self) -> None:
        """One definition, shared: the games patching CLI imports it rather than carrying a copy.

        ``action_gap`` lives in ``reward_hacking.interp.steering`` because two threads now read the
        same statistic over different label pairs -- cooperate-minus-defect there,
        tamper-minus-honest here -- and two copies would drift into two incomparable recoveries. This
        asserts identity of the function object, not merely numeric agreement, so re-introducing a
        local copy fails here rather than passing until the two diverge.
        """
        # Not at the top: importing the games patching CLI drags its whole module tree, which only
        # this one test needs.
        from games.interp_patching import action_gap as games_action_gap  # noqa: PLC0415

        assert games_action_gap is action_gap

    def test_recovery_is_the_fraction_of_the_gap_the_patch_moved(self) -> None:
        assert gap_recovery(10.0, 2.0, 6.0) == pytest.approx(0.5)

    def test_a_zero_denominator_returns_none_rather_than_a_fabricated_zero(self) -> None:
        assert gap_recovery(3.0, 3.0, 9.0) is None

    def test_an_inverted_effect_is_reported_with_its_sign(self) -> None:
        """The one-sided screen could not see this; sign inversion is documented in the literature."""
        assert gap_recovery(10.0, 2.0, -6.0) == pytest.approx(-1.0)

    def test_an_overshoot_is_reported_rather_than_clipped(self) -> None:
        assert gap_recovery(10.0, 2.0, 18.0) == pytest.approx(2.0)
