"""Offline tests for the mixed-game, mixed-framing corpus builder and its stratified selection.

No model loads, no network, no GPU: the sweep is scripted per prompt, so a test can say "this far pair
comes back with exactly one cooperation in sixteen draws and that one comes back with none" and then
assert on which of them reached the corpus.

The counterpart clauses here are synthetic (`games.tests.test_framing_stimulus`), never the authored
ones: a tracked test may not carry stimulus text, and the properties under test are about the render
and the selection rather than about any particular wording.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from games.breadth_corpus import (
    BAND_FAR,
    BAND_MID,
    BAND_NEAR,
    FILL_BANDS,
    PLUMBING_SMOKE_SPEC,
    PROSOCIAL_BREADTH_SPEC,
    RULE_FAR_PAIR_ANY_COOPERATION,
    RULE_FILL_PURE_TO_QUOTA,
    RULE_STANDARD_BAND,
    TRUST_STRATUM_BAND,
    BreadthCandidates,
    BreadthDropReason,
    BreadthGame,
    BreadthSelection,
    BreadthSpec,
    build_candidates,
    main,
    select_breadth,
)
from games.framing_stimulus import load_framings
from games.prompts import (
    COUNTERPART_FRAMINGS,
    DECIDES_IN_STEP_SENTENCE,
    FRAMING_ANOTHER_AI,
    FRAMING_HUMAN,
    FRAMING_TWIN,
    FRAMING_UNSTATED,
    SPLIT_TRAIN,
    assert_counterpart_paragraph_is_the_only_insertion,
    render_matrix_rows_under_clause,
)
from games.rewards import FRAMING_ID_COLUMN, FRAMING_ID_UNSET, care_grading
from games.select_prompts import (
    DEFAULT_MIN_SPLIT_STD,
    GRADING_COLUMN,
    PROMPT_ID_COLUMN,
    DropReason,
    PromptSweepRecord,
    pair_identity,
    read_jsonl,
    rows_file_digest,
    sweep_meta,
    sweep_prompts,
    write_corpus,
    write_sweep_trace,
)
from games.select_prompts import _parse_args as parse_select_args
from games.tests.test_framing_stimulus import DEPENDENT, synthetic_clause, write_framings_file
from games.tests.test_games_select import ScriptedBackend

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

CARE_GRADING = care_grading(1)
SAMPLES_PER_PROMPT = 8

# One game, one payoff variant, one framing per band, no oversampling: small enough to script every
# completion by hand and still exercise every rule. The kept target is five pairs (ten rows), which is
# deliberately NOT a multiple of eight, so the step-multiple trim has something to do.
TEST_SPEC = BreadthSpec(
    games=(
        BreadthGame(
            group="pd-family",
            game_ids=("twin-pd",),
            payoff_variants=("temptation-2",),
            pairs_per_variant_by_band={BAND_NEAR: 2, BAND_MID: 1, BAND_FAR: 2},
        ),
    ),
    framings_by_band={
        BAND_NEAR: (FRAMING_TWIN,),
        BAND_MID: (FRAMING_ANOTHER_AI,),
        BAND_FAR: (FRAMING_HUMAN,),
    },
    grading=CARE_GRADING,
    oversample_by_band={BAND_NEAR: 1.0, BAND_MID: 1.0, BAND_FAR: 1.0},
)

# The same grid plus the trust sender at one announced rate, which is the only stratum whose target is
# a roster rather than a number the spec states. Its whole training roster renders, so the pool is
# sixteen singletons wider than TEST_SPEC's.
TRUST_TEST_SPEC = BreadthSpec(
    games=TEST_SPEC.games,
    framings_by_band=dict(TEST_SPEC.framings_by_band),
    grading=CARE_GRADING,
    oversample_by_band=dict(TEST_SPEC.oversample_by_band),
    trust_game_id="trust-vs-stated-return",
    trust_payoff_variants=("return-fifth",),
    step_prompt_multiple=2,
)


def defect_label(row: Mapping[str, Any]) -> str:
    """Return the label this row's grading column does NOT call cooperative."""
    return next(
        label for label in (str(row["label_a"]), str(row["label_b"])) if label != row["coop_label"]
    )


def picks(row: Mapping[str, Any], *, cooperates: bool) -> str:
    """Render one completion that parses as this row's cooperative or defecting answer.

    The trust sender's answer is an amount rather than one of two labels, so cooperating is the whole
    endowment against nothing: the send spread the standard band judges a trust prompt on is what the
    mix of the two produces.
    """
    if str(row[FRAMING_ID_COLUMN]) == FRAMING_ID_UNSET:
        sent = int(row["endowment"]) if cooperates else 0
        return f"<think>weighing</think><send>{sent}</send>"
    label = str(row["coop_label"]) if cooperates else defect_label(row)
    return f"<think>weighing</think><action>{label}</action>"


def scripted_sweep(
    rows: Sequence[Mapping[str, Any]], cooperations: Mapping[str, int]
) -> list[PromptSweepRecord]:
    """Sweep every row with a scripted number of cooperative draws out of `SAMPLES_PER_PROMPT`.

    The queue holds exactly one completion per draw, so nothing wraps and the cooperation count a test
    asks for is the count the trace records. A prompt left out of `cooperations` cooperates half the
    time, which is the middle of the standard band.
    """
    script = {
        str(row["prompt"]): [
            picks(row, cooperates=index < cooperations.get(str(row[PROMPT_ID_COLUMN]), 4))
            for index in range(SAMPLES_PER_PROMPT)
        ]
        for row in rows
    }
    return sweep_prompts(
        ScriptedBackend(script),
        [dict(row) for row in rows],
        samples_per_prompt=SAMPLES_PER_PROMPT,
        prefilled_think=False,
    )


def write_trace(
    path: Path, rows: Sequence[Mapping[str, Any]], records: Sequence[PromptSweepRecord]
) -> list[dict[str, Any]]:
    """Write a sweep trace the way `games.select_prompts` writes one, and read it back.

    Through the real writer and reader rather than a hand-built list of dicts, because the selection
    rebuilds its records from that JSON and a field that does not survive the round trip is exactly the
    failure this would otherwise miss.
    """
    rows_path = path.parent / "candidates.jsonl"
    write_corpus(rows_path, [dict(row) for row in rows])
    args = parse_select_args(
        [
            "--rows",
            str(rows_path),
            "--grading",
            CARE_GRADING,
            "--model",
            "mock/policy",
            "--backend",
            "mock",
        ]
    )
    write_sweep_trace(
        path,
        meta=sweep_meta(
            backend=ScriptedBackend({"unused": ["x"]}),
            args=args,
            prompt_ids=[record.prompt_id for record in records],
            samples_per_prompt=SAMPLES_PER_PROMPT,
            prefilled_think=False,
            rows_sha256=rows_file_digest(rows_path),
        ),
        records=records,
    )
    return [dict(entry) for entry in read_jsonl(path)]


def candidates_for(spec: BreadthSpec = TEST_SPEC) -> BreadthCandidates:
    """Build the candidate pool for a spec whose framings are all registered."""
    return build_candidates(spec)


class TestEveryCandidateIsItsStemPlusOneCounterpartParagraph:
    """The property every framing comparison rests on, asserted over the whole pool rather than a sample.

    A row under a framing has to differ from the same row with no counterpart paragraph by exactly one
    inserted paragraph. If a clause rendered two paragraphs, or the renderer moved anything else, a
    behavioural difference between framings would not be attributable to the clause.
    """

    def test_every_matrix_candidate_matches_its_unstated_stem(self) -> None:
        candidates = candidates_for()
        stems = {
            (str(row["reskin_id"]), str(row["payoff_variant"]), str(row["coop_label"])): str(
                row["prompt"]
            )
            for row in render_matrix_rows_under_clause(
                "twin-pd",
                CARE_GRADING,
                clause=None,
                framing_label=FRAMING_UNSTATED,
                split=SPLIT_TRAIN,
            )
        }
        framed = [
            row
            for row in candidates.rows
            if row[FRAMING_ID_COLUMN] not in (FRAMING_UNSTATED, FRAMING_ID_UNSET)
        ]
        assert framed
        for row in framed:
            key = (str(row["reskin_id"]), str(row["payoff_variant"]), str(row["coop_label"]))
            assert_counterpart_paragraph_is_the_only_insertion(
                stem=stems[key],
                rendered=str(row["prompt"]),
                prompt_id=str(row[PROMPT_ID_COLUMN]),
            )

    def test_the_builder_refuses_a_two_paragraph_clause(self, tmp_path: Path) -> None:
        """A clause carrying a paragraph break survives the audit's deletion, so the audit goes red.

        The loader refuses such a clause too, but only for a file it read; a caller passing clauses in
        directly reaches the audit first, and this is the audit failing rather than the loader.
        """
        del tmp_path
        spec = _spec_with_framings({BAND_NEAR: (DEPENDENT,), BAND_MID: (), BAND_FAR: ()})
        clause = f"{synthetic_clause(DEPENDENT)}\n\nA second paragraph the marker does not open."
        with pytest.raises(ValueError, match="not its stem plus one counterpart paragraph"):
            build_candidates(spec, runtime_clauses={DEPENDENT: clause})


def _spec_with_framings(framings_by_band: Mapping[str, tuple[str, ...]]) -> BreadthSpec:
    """Return the test spec with its framings replaced, quotas and games unchanged."""
    return BreadthSpec(
        games=TEST_SPEC.games,
        framings_by_band=dict(framings_by_band),
        grading=CARE_GRADING,
        oversample_by_band=dict(TEST_SPEC.oversample_by_band),
    )


class TestTheFramingTravelsWithThePair:
    """A counterbalanced pair is one scenario's two label orientations under ONE framing.

    The pair key is every column a label swap leaves alone, so the framing column being part of it is
    what keeps a frame's twin-framed and human-framed renderings from pairing with each other. Were
    they to pair, four rows would share a key, the coupling step would drop unrelated prompts together,
    and the far rule would decide on two framings' draws at once.
    """

    def test_every_pair_holds_two_rows_of_one_framing(self) -> None:
        candidates = candidates_for()
        groups: dict[tuple[object, ...], list[dict[str, Any]]] = {}
        for row in candidates.rows:
            groups.setdefault(pair_identity(row), []).append(dict(row))
        assert groups
        for group in groups.values():
            assert len(group) == 2
            assert len({str(row[FRAMING_ID_COLUMN]) for row in group}) == 1
            assert len({str(row["reskin_id"]) for row in group}) == 1
            assert {str(row["coop_label"]) for row in group} == {
                str(group[0]["label_a"]),
                str(group[0]["label_b"]),
            }

    def test_one_frame_under_two_framings_is_two_pairs(self) -> None:
        # The whole training roster in both bands, so every frame is certain to be drawn under both
        # framings rather than only where two independent draws happened to agree.
        whole_roster = BreadthSpec(
            games=(
                BreadthGame(
                    group="pd-family",
                    game_ids=("twin-pd",),
                    payoff_variants=("temptation-2",),
                    pairs_per_variant_by_band={BAND_NEAR: 16, BAND_MID: 16, BAND_FAR: 0},
                ),
            ),
            framings_by_band={
                BAND_NEAR: (FRAMING_TWIN,),
                BAND_MID: (FRAMING_ANOTHER_AI,),
                BAND_FAR: (),
            },
            grading=CARE_GRADING,
            oversample_by_band={BAND_NEAR: 1.0, BAND_MID: 1.0, BAND_FAR: 1.0},
        )
        candidates = candidates_for(whole_roster)
        by_frame: dict[str, set[str]] = {}
        for row in candidates.rows:
            by_frame.setdefault(str(row["reskin_id"]), set()).add(str(row[FRAMING_ID_COLUMN]))
        shared = [frame for frame, framings in by_frame.items() if len(framings) > 1]
        assert shared, "the draw put no frame under both framings, so this proves nothing"
        for frame in shared:
            rows = [row for row in candidates.rows if str(row["reskin_id"]) == frame]
            assert len({pair_identity(row) for row in rows}) == len(by_frame[frame])


class TestTheFarStratumRule:
    """The far framings keep a pair on one cooperation in sixteen, which the standard band would not.

    At 9B the base cooperates 1 to 3 percent under a decoupled non-AI counterpart, so the standard
    `[0.125, 0.875]` band keeps almost nothing there and the corpus would contain only the framings the
    reward already reaches. The rule is a real relaxation and the cost is real, which is why the
    stratum artifact records which rule it applied.
    """

    def far_rows(self, candidates: BreadthCandidates) -> list[dict[str, Any]]:
        return [dict(row) for row in candidates.rows if row[FRAMING_ID_COLUMN] == FRAMING_HUMAN]

    def test_a_pair_with_one_cooperation_in_sixteen_is_kept_and_one_with_none_is_dropped(
        self, tmp_path: Path
    ) -> None:
        candidates = candidates_for()
        far = self.far_rows(candidates)
        pairs = _pairs_of(far)
        assert len(pairs) == 2, "the far stratum should carry two candidate pairs"
        kept_pair, dropped_pair = pairs
        cooperations = {
            # One cooperative draw in the pair's sixteen: the whole point of the rule.
            str(kept_pair[0][PROMPT_ID_COLUMN]): 1,
            str(kept_pair[1][PROMPT_ID_COLUMN]): 0,
            str(dropped_pair[0][PROMPT_ID_COLUMN]): 0,
            str(dropped_pair[1][PROMPT_ID_COLUMN]): 0,
        }
        records = scripted_sweep(candidates.rows, cooperations)
        trace = write_trace(tmp_path / "sweep.jsonl", candidates.rows, records)
        selection = select_breadth(trace, list(candidates.rows), TEST_SPEC)

        written = {str(row[PROMPT_ID_COLUMN]) for row in selection.rows}
        assert {str(row[PROMPT_ID_COLUMN]) for row in kept_pair} <= written
        assert not {str(row[PROMPT_ID_COLUMN]) for row in dropped_pair} & written
        far_stratum = next(outcome for outcome in selection.strata if outcome.plan.band == BAND_FAR)
        assert far_stratum.plan.rule == RULE_FAR_PAIR_ANY_COOPERATION
        assert far_stratum.dropped_by_reason == {str(BreadthDropReason.FAR_PAIR_NO_COOPERATION): 1}

    def test_a_far_pair_the_standard_band_would_drop_is_kept(self, tmp_path: Path) -> None:
        """One cooperation in eight is 0.125 at the prompt, and its partner's zero would sink the pair.

        Under the standard band the pair's partner reads 0.0, below `DEFAULT_MIN_COOP`, and the
        coupling step drops both. The far rule decides on the pair's sixteen draws instead, which is
        the difference the whole stratum depends on.
        """
        candidates = candidates_for()
        far = self.far_rows(candidates)
        pairs = _pairs_of(far)
        cooperations = {
            str(row[PROMPT_ID_COLUMN]): (1 if index == 0 else 0)
            for pair in pairs
            for index, row in enumerate(pair)
        }
        records = scripted_sweep(candidates.rows, cooperations)
        trace = write_trace(tmp_path / "sweep.jsonl", candidates.rows, records)
        selection = select_breadth(trace, list(candidates.rows), TEST_SPEC)
        written = {str(row[PROMPT_ID_COLUMN]) for row in selection.rows}
        assert {str(row[PROMPT_ID_COLUMN]) for pair in pairs for row in pair} <= written

    def test_a_far_pair_whose_draws_mostly_failed_to_parse_is_dropped(self, tmp_path: Path) -> None:
        """A cooperation among two parseable draws of sixteen measures the format, not the policy.

        The parse floor is the one condition the far rule keeps from the standard band, and it is
        recorded under its own reason rather than folded into "no cooperation", because the two say
        different things about the prompt.
        """
        candidates = candidates_for()
        far_ids = {str(row[PROMPT_ID_COLUMN]) for row in self.far_rows(candidates)}
        script: dict[str, list[str]] = {}
        for row in candidates.rows:
            if str(row[PROMPT_ID_COLUMN]) in far_ids:
                script[str(row["prompt"])] = [
                    picks(row, cooperates=index == 0) if index < 2 else "<think>no answer"
                    for index in range(SAMPLES_PER_PROMPT)
                ]
            else:
                script[str(row["prompt"])] = [
                    picks(row, cooperates=index < 4) for index in range(SAMPLES_PER_PROMPT)
                ]
        records = sweep_prompts(
            ScriptedBackend(script),
            [dict(row) for row in candidates.rows],
            samples_per_prompt=SAMPLES_PER_PROMPT,
            prefilled_think=False,
        )
        trace = write_trace(tmp_path / "sweep.jsonl", candidates.rows, records)
        selection = select_breadth(trace, list(candidates.rows), TEST_SPEC)
        written = {str(row[PROMPT_ID_COLUMN]) for row in selection.rows}
        assert not far_ids & written
        far_stratum = next(outcome for outcome in selection.strata if outcome.plan.band == BAND_FAR)
        assert far_stratum.dropped_by_reason == {
            str(BreadthDropReason.FAR_PAIR_TOO_FEW_PARSEABLE): 2
        }

    def test_a_far_pair_present_in_only_one_orientation_is_refused(self, tmp_path: Path) -> None:
        """The far strata skip `judge_prompts`, so its orphan-orientation guard has to live in `_units`.

        Kept as a one-row unit the orientation carries its own verdict into the corpus, and the corpus
        then holds one label placement of a scenario -- exactly the position bias counterbalancing
        exists to cancel. The trust sender's singletons carry empty labels and must still pass.
        """
        candidates = candidates_for()
        far_ids = [str(row[PROMPT_ID_COLUMN]) for row in self.far_rows(candidates)]
        orphaned = [
            dict(row) for row in candidates.rows if str(row[PROMPT_ID_COLUMN]) != far_ids[1]
        ]
        records = scripted_sweep(orphaned, {})
        trace = write_trace(tmp_path / "sweep.jsonl", orphaned, records)
        with pytest.raises(ValueError, match="no counterbalanced partner in this pool"):
            select_breadth(trace, orphaned, TEST_SPEC)

    def test_the_trust_singletons_are_not_read_as_orphan_orientations(self, tmp_path: Path) -> None:
        """Sixteen unframed singletons and no partner between them, and the selection has to run."""
        candidates = candidates_for(TRUST_TEST_SPEC)
        records = scripted_sweep(candidates.rows, {})
        trace = write_trace(tmp_path / "sweep.jsonl", candidates.rows, records)
        selection = select_breadth(trace, list(candidates.rows), TRUST_TEST_SPEC)
        trust = [row for row in selection.rows if str(row[FRAMING_ID_COLUMN]) == FRAMING_ID_UNSET]
        assert len(trust) == 16


def _pairs_of(rows: Sequence[Mapping[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Group rows into counterbalanced pairs, in first-appearance order."""
    groups: dict[tuple[object, ...], list[dict[str, Any]]] = {}
    for row in rows:
        copied = dict(row)
        groups.setdefault(pair_identity(copied), []).append(copied)
    return [(group[0], group[1]) for group in groups.values() if len(group) == 2]


class TestQuotasAreTargets:
    """A quota bounds a stratum from above and promises nothing from below.

    A stratum whose sweep kept more pairs than the target is cut to it; a stratum that came in short
    stays short and is recorded, because topping it up from another stratum would silently reweight
    the composition the arm's whole reading is stated against.
    """

    def test_a_stratum_over_its_quota_is_cut_to_it(self, tmp_path: Path) -> None:
        spec = BreadthSpec(
            games=(
                BreadthGame(
                    group="pd-family",
                    game_ids=("twin-pd",),
                    payoff_variants=("temptation-2",),
                    pairs_per_variant_by_band={BAND_NEAR: 2, BAND_MID: 0, BAND_FAR: 0},
                ),
            ),
            framings_by_band={BAND_NEAR: (FRAMING_TWIN,), BAND_MID: (), BAND_FAR: ()},
            grading=CARE_GRADING,
            oversample_by_band={BAND_NEAR: 2.0, BAND_MID: 1.0, BAND_FAR: 1.0},
            step_prompt_multiple=2,
        )
        candidates = candidates_for(spec)
        assert len(candidates.rows) == 8, "four candidate pairs against a target of two"
        records = scripted_sweep(candidates.rows, {})
        trace = write_trace(tmp_path / "sweep.jsonl", candidates.rows, records)
        selection = select_breadth(trace, list(candidates.rows), spec)

        assert len(selection.rows) == 4
        stratum = selection.strata[0]
        assert stratum.plan.candidate_pairs == 4
        assert stratum.n_kept_units == 2
        assert stratum.dropped_by_reason == {str(BreadthDropReason.OVER_QUOTA): 2}

    def test_a_stratum_that_came_in_short_stays_short_and_is_recorded(self, tmp_path: Path) -> None:
        spec = BreadthSpec(
            games=(
                BreadthGame(
                    group="pd-family",
                    game_ids=("twin-pd",),
                    payoff_variants=("temptation-2",),
                    pairs_per_variant_by_band={BAND_NEAR: 4, BAND_MID: 0, BAND_FAR: 0},
                ),
            ),
            framings_by_band={BAND_NEAR: (FRAMING_TWIN,), BAND_MID: (), BAND_FAR: ()},
            grading=CARE_GRADING,
            oversample_by_band={BAND_NEAR: 1.0, BAND_MID: 1.0, BAND_FAR: 1.0},
            step_prompt_multiple=2,
        )
        candidates = candidates_for(spec)
        pairs = _pairs_of(candidates.rows)
        assert len(pairs) == 4
        unanimous = {str(row[PROMPT_ID_COLUMN]): SAMPLES_PER_PROMPT for row in pairs[0] + pairs[1]}
        records = scripted_sweep(candidates.rows, unanimous)
        trace = write_trace(tmp_path / "sweep.jsonl", candidates.rows, records)
        selection = select_breadth(trace, list(candidates.rows), spec)

        stratum = selection.strata[0]
        assert stratum.plan.quota_pairs == 4
        assert stratum.plan.rule == RULE_STANDARD_BAND
        assert stratum.n_kept_units == 2
        assert stratum.n_kept_rows == len(selection.rows) == 4
        assert stratum.dropped_by_reason == {"coop-fraction-above-max": 2}
        assert selection.realised_composition()["rows_by_framing_id"] == {FRAMING_TWIN: 4}

    def test_a_game_that_kept_too_few_pairs_drops_whole_and_is_recorded(
        self, tmp_path: Path
    ) -> None:
        """The hi-lo precedent: a game with almost nothing selectable is a datum, not a few prompts."""
        spec = BreadthSpec(
            games=(
                BreadthGame(
                    group="pd-family",
                    game_ids=("twin-pd",),
                    payoff_variants=("temptation-2",),
                    pairs_per_variant_by_band={BAND_NEAR: 4, BAND_MID: 0, BAND_FAR: 0},
                ),
                BreadthGame(
                    group="chicken",
                    game_ids=("chicken",),
                    payoff_variants=("standard",),
                    pairs_per_variant_by_band={BAND_NEAR: 8, BAND_MID: 0, BAND_FAR: 0},
                ),
            ),
            framings_by_band={BAND_NEAR: (FRAMING_TWIN,), BAND_MID: (), BAND_FAR: ()},
            grading=CARE_GRADING,
            oversample_by_band={BAND_NEAR: 1.0, BAND_MID: 1.0, BAND_FAR: 1.0},
            step_prompt_multiple=2,
        )
        candidates = candidates_for(spec)
        chicken = [row for row in candidates.rows if row["game_id"] == "chicken"]
        chicken_pairs = _pairs_of(chicken)
        unanimous = {
            str(row[PROMPT_ID_COLUMN]): SAMPLES_PER_PROMPT
            for pair in chicken_pairs[1:]
            for row in pair
        }
        records = scripted_sweep(candidates.rows, unanimous)
        trace = write_trace(tmp_path / "sweep.jsonl", candidates.rows, records)
        selection = select_breadth(trace, list(candidates.rows), spec)

        assert list(selection.dropped_groups) == ["chicken"]
        assert "kept 1 of 8 targeted pairs" in selection.dropped_groups["chicken"]
        assert {str(row["game_id"]) for row in selection.rows} == {"twin-pd"}
        chicken_stratum = next(
            outcome for outcome in selection.strata if outcome.plan.group == "chicken"
        )
        assert chicken_stratum.n_kept_units == 0
        assert (
            chicken_stratum.dropped_by_reason[str(BreadthDropReason.GAME_KEPT_TOO_FEW_PAIRS)] == 1
        )


def _fill_spec(*, mid_pairs: int = 2, min_kept_fraction_per_game: float = 0.0) -> BreadthSpec:
    """One stratum per band, each drawn at twice its target, so any band can come in short.

    Two pairs targeted against four drawn is the smallest grid where the fill has something to choose
    from in every band and the near band can be watched for NOT being filled. The per-game drop
    threshold is zero unless a test asks for it, so a stratum-level rule reads on its own rather than
    through the game-level rule that runs after it.
    """
    return BreadthSpec(
        games=(
            BreadthGame(
                group="pd-family",
                game_ids=("twin-pd",),
                payoff_variants=("temptation-2",),
                pairs_per_variant_by_band={BAND_NEAR: 2, BAND_MID: mid_pairs, BAND_FAR: 2},
            ),
        ),
        framings_by_band={
            BAND_NEAR: (FRAMING_TWIN,),
            BAND_MID: (FRAMING_ANOTHER_AI,),
            BAND_FAR: (FRAMING_HUMAN,),
        },
        grading=CARE_GRADING,
        oversample_by_band={BAND_NEAR: 2.0, BAND_MID: 2.0, BAND_FAR: 2.0},
        step_prompt_multiple=2,
        min_kept_fraction_per_game=min_kept_fraction_per_game,
    )


def _band_pairs(
    candidates: BreadthCandidates, framing_id: str
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Return one band's candidate pairs, in the order the pool rendered them."""
    return _pairs_of([dict(row) for row in candidates.rows if row[FRAMING_ID_COLUMN] == framing_id])


def _cooperations_by_orientation(
    candidates: BreadthCandidates,
    *,
    near: Sequence[tuple[int, int]],
    mid: Sequence[tuple[int, int]],
    far: Sequence[tuple[int, int]],
) -> dict[str, int]:
    """Script a cooperation count per orientation, one entry per pair each band drew.

    Per orientation rather than per pair because the fill's own order is decided per orientation: a
    pair holding one unanimous and one mixed orientation is a different thing from two unanimous ones,
    and only the first of them can produce a group GRPO's advantage can see. `strict=True` refuses a
    script that does not cover exactly the pairs the pool drew.
    """
    counts: dict[str, int] = {}
    for framing_id, per_pair in (
        (FRAMING_TWIN, near),
        (FRAMING_ANOTHER_AI, mid),
        (FRAMING_HUMAN, far),
    ):
        for pair, orientations in zip(_band_pairs(candidates, framing_id), per_pair, strict=True):
            for row, count in zip(pair, orientations, strict=True):
                counts[str(row[PROMPT_ID_COLUMN])] = count
    return counts


def _select_scripted(  # noqa: PLR0913 - one keyword per option under test
    tmp_path: Path,
    spec: BreadthSpec,
    cooperations: Mapping[str, int],
    *,
    fill_pure_to_quota: bool = False,
    trust_min_split_std: float | None = None,
    reverse_trace: bool = False,
) -> BreadthSelection:
    """Sweep a spec's pool under a scripted base model and select it, with the options under test."""
    candidates = candidates_for(spec)
    ordered = list(reversed(candidates.rows)) if reverse_trace else list(candidates.rows)
    records = scripted_sweep(ordered, cooperations)
    trace = write_trace(tmp_path / "sweep.jsonl", ordered, records)
    return select_breadth(
        trace,
        list(candidates.rows),
        spec,
        fill_pure_to_quota=fill_pure_to_quota,
        trust_min_split_std=trust_min_split_std,
    )


def _stratum(selection: BreadthSelection, band: str) -> Any:
    """Return the one stratum outcome of a band, refusing a fixture that grew a second one."""
    outcomes = [outcome for outcome in selection.strata if outcome.plan.band == band]
    assert len(outcomes) == 1, f"the fixture holds {len(outcomes)} {band} strata, not one"
    return outcomes[0]


class TestTheOptionalFillOfShortMidAndFarStrata:
    """The fill tops a short mid or far stratum up with pairs the base played unanimously.

    Off unless asked for, because the banked wave-4b corpus is the record of the registered rules. The
    argument for having it at all: training runs dynamic sampling at oversample 2, so a group whose
    rewards are all equal is dropped at step time, which means a pure-at-base prompt costs generation
    and never a gradient -- and those are exactly the prompts a care-weighted arm is meant to move.
    Under the registered rules the 9B mid band came in at 20 rows against a quota of 60 pairs.

    Near strata keep the mixed-only rule: they came in at 76 rows and have no shortage to fix, and
    relaxing the band where the reward already reaches would spend the corpus on prompts that are
    already movable.
    """

    def script(self, spec: BreadthSpec) -> dict[str, int]:
        """One mixed pair in the near band, one cooperation in the far band, the rest unanimous.

        The mid band is split between unanimous defection and unanimous cooperation on purpose: both
        are pure at base, they fail the standard band at opposite ends, and a rule that only looked at
        one end would keep half the mid band out for a reason nothing records.
        """
        return _cooperations_by_orientation(
            candidates_for(spec),
            near=[(4, 4), (0, 0), (0, 0), (0, 0)],
            mid=[(0, 0), (0, 0), (8, 8), (8, 8)],
            far=[(1, 0), (0, 0), (0, 0), (0, 0)],
        )

    def test_the_default_keeps_only_what_the_band_rules_kept(self, tmp_path: Path) -> None:
        spec = _fill_spec()
        selection = _select_scripted(tmp_path, spec, self.script(spec))

        assert len(selection.rows) == 4
        assert _stratum(selection, BAND_NEAR).n_kept_units == 1
        assert _stratum(selection, BAND_MID).n_kept_units == 0
        assert _stratum(selection, BAND_FAR).n_kept_units == 1
        artifact = selection.to_artifact()
        assert "selection_options" not in artifact
        assert not any("fill" in outcome for outcome in artifact["strata"])

    def test_a_short_mid_stratum_is_filled_to_quota_and_recorded(self, tmp_path: Path) -> None:
        spec = _fill_spec()
        selection = _select_scripted(tmp_path, spec, self.script(spec), fill_pure_to_quota=True)

        mid = _stratum(selection, BAND_MID)
        assert mid.plan.quota_pairs == 2
        assert mid.n_kept_units == 2
        assert mid.n_kept_rows == 4
        assert [unit.rule for unit in mid.filled_units] == [RULE_FILL_PURE_TO_QUOTA] * 2
        assert {unit.coop_fraction for unit in mid.filled_units} <= {0.0, 1.0}
        block = mid.to_json_dict()["fill"]
        assert block["rule"] == RULE_FILL_PURE_TO_QUOTA
        assert block["eligible"] is True
        assert block["n_filled_units"] == 2
        assert block["n_filled_rows"] == 4
        assert len(block["units"]) == 2
        assert all(len(unit["prompt_ids"]) == 2 for unit in block["units"])
        options = selection.to_artifact()["selection_options"]
        assert options["fill_rule"] == RULE_FILL_PURE_TO_QUOTA
        assert options["fill_bands"] == list(FILL_BANDS)
        assert options["n_filled_rows"] == 6

    def test_a_short_far_stratum_is_filled_on_top_of_what_its_own_rule_kept(
        self, tmp_path: Path
    ) -> None:
        """The far rule keeps a pair on one cooperation in sixteen; the fill covers the rest of the quota."""
        spec = _fill_spec()
        selection = _select_scripted(tmp_path, spec, self.script(spec), fill_pure_to_quota=True)

        far = _stratum(selection, BAND_FAR)
        assert far.plan.quota_pairs == 2
        assert far.n_kept_units == 2
        assert len(far.filled_units) == 1
        assert far.filled_units[0].coop_fraction == 0.0
        assert len(selection.rows) == 10

    def test_a_short_near_stratum_is_never_filled(self, tmp_path: Path) -> None:
        """The near band keeps the mixed-only rule, which is the whole reason the fill is banded."""
        spec = _fill_spec()
        selection = _select_scripted(tmp_path, spec, self.script(spec), fill_pure_to_quota=True)

        near = _stratum(selection, BAND_NEAR)
        assert near.plan.quota_pairs == 2
        assert near.n_kept_units == 1, "the near band has three unanimous pairs and must keep none"
        assert near.filled_units == ()
        block = near.to_json_dict()["fill"]
        assert block["eligible"] is False
        assert block["n_filled_units"] == 0

    def test_the_fill_stops_at_the_quota(self, tmp_path: Path) -> None:
        """Four unanimous mid pairs against a quota of two: the two that stay out keep their own reason."""
        spec = _fill_spec()
        selection = _select_scripted(tmp_path, spec, self.script(spec), fill_pure_to_quota=True)

        mid = _stratum(selection, BAND_MID)
        assert mid.plan.candidate_pairs == 4
        assert mid.n_kept_units == 2
        assert sum(mid.dropped_by_reason.values()) == 2
        assert set(mid.dropped_by_reason) <= {
            str(DropReason.COOP_FRACTION_BELOW_MIN),
            str(DropReason.COOP_FRACTION_ABOVE_MAX),
        }

    def test_a_pair_whose_draws_mostly_failed_to_parse_is_not_filled(self, tmp_path: Path) -> None:
        """The parse floor is the one condition the fill keeps: an unparsed pair measures the format."""
        spec = _fill_spec(mid_pairs=1)
        candidates = candidates_for(spec)
        mid_ids = {
            str(row[PROMPT_ID_COLUMN])
            for row in candidates.rows
            if row[FRAMING_ID_COLUMN] == FRAMING_ANOTHER_AI
        }
        script: dict[str, list[str]] = {}
        for row in candidates.rows:
            if str(row[PROMPT_ID_COLUMN]) in mid_ids:
                script[str(row["prompt"])] = [
                    picks(row, cooperates=False) if index < 2 else "<think>no answer"
                    for index in range(SAMPLES_PER_PROMPT)
                ]
            else:
                script[str(row["prompt"])] = [picks(row, cooperates=False)] * SAMPLES_PER_PROMPT
        records = sweep_prompts(
            ScriptedBackend(script),
            [dict(row) for row in candidates.rows],
            samples_per_prompt=SAMPLES_PER_PROMPT,
            prefilled_think=False,
        )
        trace = write_trace(tmp_path / "sweep.jsonl", candidates.rows, records)
        selection = select_breadth(trace, list(candidates.rows), spec, fill_pure_to_quota=True)

        mid = _stratum(selection, BAND_MID)
        assert mid.n_kept_units == 0
        assert mid.filled_units == ()
        assert _stratum(selection, BAND_FAR).n_kept_units == 2, (
            "the far band's own unanimous pairs parsed and must still fill"
        )

    def test_the_fill_takes_the_same_pairs_in_either_trace_order(self, tmp_path: Path) -> None:
        """The choice is a hash of prompt identity, so the trace's order cannot move it."""
        spec = _fill_spec()
        cooperations = self.script(spec)
        forward = _select_scripted(
            tmp_path / "forward", spec, cooperations, fill_pure_to_quota=True
        )
        backward = _select_scripted(
            tmp_path / "backward", spec, cooperations, fill_pure_to_quota=True, reverse_trace=True
        )

        assert [row[PROMPT_ID_COLUMN] for row in forward.rows] == [
            row[PROMPT_ID_COLUMN] for row in backward.rows
        ]
        assert [unit.prompt_ids for unit in _stratum(forward, BAND_MID).filled_units] == [
            unit.prompt_ids for unit in _stratum(backward, BAND_MID).filled_units
        ]

    def test_the_fill_prefers_a_pair_with_one_mixed_orientation(self, tmp_path: Path) -> None:
        """A pair whose two orientations are unanimous at opposite ends is pure twice over.

        Its pooled cooperation rate is exactly one half, which would sort it first on the pair's
        average and is the wrong reading: dynamic sampling drops a group per prompt, so both of its
        orientations are dropped at step time and neither ever reaches the gradient. The pair holding
        one orientation at an eighth has a group that can disagree, so it goes first.
        """
        spec = _fill_spec(mid_pairs=1)
        candidates = candidates_for(spec)
        mid_pairs = _band_pairs(candidates, FRAMING_ANOTHER_AI)
        assert len(mid_pairs) == 2
        cooperations = _cooperations_by_orientation(
            candidates,
            near=[(4, 4), (4, 4), (4, 4), (4, 4)],
            mid=[(0, 8), (0, 1)],
            far=[(0, 0), (0, 0), (0, 0), (0, 0)],
        )
        selection = _select_scripted(tmp_path, spec, cooperations, fill_pure_to_quota=True)

        mid = _stratum(selection, BAND_MID)
        assert len(mid.filled_units) == 1
        assert set(mid.filled_units[0].prompt_ids) == {
            str(row[PROMPT_ID_COLUMN]) for row in mid_pairs[1]
        }

    def test_the_fill_runs_before_the_game_level_drop(self, tmp_path: Path) -> None:
        """A game the band rules would drop whole survives on filled pairs, or the fill is unreachable.

        The order is the point: `_drop_thin_games` reads how many pairs a group kept, so a fill that
        ran after it would top up the strata of a game that had already been removed from the corpus.
        """
        spec = _fill_spec(min_kept_fraction_per_game=0.25)
        cooperations = _cooperations_by_orientation(
            candidates_for(spec),
            near=[(4, 4), (0, 0), (0, 0), (0, 0)],
            mid=[(0, 0), (0, 0), (0, 0), (0, 0)],
            far=[(0, 0), (0, 0), (0, 0), (0, 0)],
        )
        without_fill = _select_scripted(tmp_path / "without", spec, cooperations)
        with_fill = _select_scripted(tmp_path / "with", spec, cooperations, fill_pure_to_quota=True)

        assert list(without_fill.dropped_groups) == ["pd-family"]
        assert without_fill.rows == ()
        assert with_fill.dropped_groups == {}
        assert len(with_fill.rows) == 10


def _trust_floor_spec() -> BreadthSpec:
    """One near pair and the trust sender's whole roster, so the trust floor reads on its own."""
    return BreadthSpec(
        games=(
            BreadthGame(
                group="pd-family",
                game_ids=("twin-pd",),
                payoff_variants=("temptation-2",),
                pairs_per_variant_by_band={BAND_NEAR: 1, BAND_MID: 0, BAND_FAR: 0},
            ),
        ),
        framings_by_band={BAND_NEAR: (FRAMING_TWIN,), BAND_MID: (), BAND_FAR: ()},
        grading=CARE_GRADING,
        oversample_by_band={BAND_NEAR: 1.0, BAND_MID: 1.0, BAND_FAR: 1.0},
        trust_game_id="trust-vs-stated-return",
        trust_payoff_variants=("return-fifth",),
        step_prompt_multiple=2,
    )


def _trust_script(candidates: BreadthCandidates, *, n_with_spread: int) -> dict[str, int]:
    """Send the whole stock once in eight draws for `n_with_spread` rows and never for the rest.

    One send in eight gives a spread of 0.3307 and none gives exactly zero, which is the shape the 9B
    base sweep actually produced: 29 of its 32 trust rows sent the same amount every draw.
    """
    trust_ids = [
        str(row[PROMPT_ID_COLUMN])
        for row in candidates.rows
        if str(row[FRAMING_ID_COLUMN]) == FRAMING_ID_UNSET
    ]
    return {
        prompt_id: (1 if index < n_with_spread else 0) for index, prompt_id in enumerate(trust_ids)
    }


class TestTheTrustSpreadFloorIsItsOwnKnob:
    """The trust rows are judged on a send spread, and the floor judging them is now separately settable.

    In the 9B base sweep 29 of 32 trust rows returned the same amount in all eight draws, so their
    spread was exactly zero and the shared 0.05 floor dropped them, while the three that cleared it sat
    at 0.33 and above. No floor strictly between zero and a third admits anything the default did not,
    so the knob is an explicit value rather than a nudge: only zero, which keeps every row that parsed,
    changes the trust leg at all.
    """

    def _trust_rows(self, selection: BreadthSelection) -> list[str]:
        return [
            str(row[PROMPT_ID_COLUMN])
            for row in selection.rows
            if str(row[FRAMING_ID_COLUMN]) == FRAMING_ID_UNSET
        ]

    def _matrix_rows(self, selection: BreadthSelection) -> list[str]:
        return [
            str(row[PROMPT_ID_COLUMN])
            for row in selection.rows
            if str(row[FRAMING_ID_COLUMN]) != FRAMING_ID_UNSET
        ]

    @pytest.mark.parametrize(
        ("floor", "n_trust_rows"),
        [(None, 8), (0.0, 16), (DEFAULT_MIN_SPLIT_STD, 8), (0.4, 0)],
    )
    def test_each_floor_admits_exactly_the_rows_whose_spread_clears_it(
        self, tmp_path: Path, floor: float | None, n_trust_rows: int
    ) -> None:
        spec = _trust_floor_spec()
        cooperations = _trust_script(candidates_for(spec), n_with_spread=8)
        selection = _select_scripted(tmp_path, spec, cooperations, trust_min_split_std=floor)

        assert len(self._trust_rows(selection)) == n_trust_rows
        assert len(self._matrix_rows(selection)) == 2, (
            "the matrix strata are not the floor's business"
        )

    def test_naming_the_default_floor_explicitly_changes_nothing(self, tmp_path: Path) -> None:
        """The knob splits the trust rows into their own judging call, which must not move a verdict."""
        spec = _trust_floor_spec()
        cooperations = _trust_script(candidates_for(spec), n_with_spread=8)
        implicit = _select_scripted(tmp_path / "implicit", spec, cooperations)
        explicit = _select_scripted(
            tmp_path / "explicit", spec, cooperations, trust_min_split_std=DEFAULT_MIN_SPLIT_STD
        )

        assert implicit.rows == explicit.rows
        assert [outcome.to_json_dict() for outcome in implicit.strata] == [
            {key: value for key, value in outcome.to_json_dict().items() if key != "fill"}
            for outcome in explicit.strata
        ]

    def test_a_negative_floor_is_refused(self, tmp_path: Path) -> None:
        spec = _trust_floor_spec()
        cooperations = _trust_script(candidates_for(spec), n_with_spread=8)
        with pytest.raises(ValueError, match="trust_min_split_std"):
            _select_scripted(tmp_path, spec, cooperations, trust_min_split_std=-0.1)

    def test_the_floor_is_refused_on_a_spec_that_renders_no_trust_rows(
        self, tmp_path: Path
    ) -> None:
        """A knob that cannot change the corpus it is passed with is an operator mistake, not a no-op."""
        spec = _fill_spec()
        cooperations = _cooperations_by_orientation(
            candidates_for(spec),
            near=[(4, 4), (4, 4), (4, 4), (4, 4)],
            mid=[(4, 4), (4, 4), (4, 4), (4, 4)],
            far=[(4, 4), (4, 4), (4, 4), (4, 4)],
        )
        with pytest.raises(ValueError, match="carries no trust rows"):
            _select_scripted(tmp_path, spec, cooperations, trust_min_split_std=0.0)


def _asymmetric_near_spec() -> BreadthSpec:
    """Two near pairs and nothing else, so one asymmetric pair is the whole stratum's drop record."""
    return BreadthSpec(
        games=(
            BreadthGame(
                group="pd-family",
                game_ids=("twin-pd",),
                payoff_variants=("temptation-2",),
                pairs_per_variant_by_band={BAND_NEAR: 2, BAND_MID: 0, BAND_FAR: 0},
            ),
        ),
        framings_by_band={BAND_NEAR: (FRAMING_TWIN,), BAND_MID: (), BAND_FAR: ()},
        grading=CARE_GRADING,
        oversample_by_band={BAND_NEAR: 1.0, BAND_MID: 1.0, BAND_FAR: 1.0},
        step_prompt_multiple=2,
    )


class TestADroppedPairIsRecordedUnderItsBaseBehaviour:
    """`counterbalanced-partner-dropped` names the coupling, never why the pair was unusable.

    `judge_prompts` rewrites the acceptable orientation's verdict to the coupling reason, so a group
    holding one substantive verdict and one rewritten one has two reasons and only one of them answers
    the question the artifact is read for: how the base behaved, and why the stratum came in short.
    Which one is recorded must not depend on the order the trace happens to list the two orientations
    in, which is the builder's emission order and carries no meaning at all.
    """

    def _selection(self, tmp_path: Path, *, reverse: bool) -> Any:
        spec = _asymmetric_near_spec()
        candidates = candidates_for(spec)
        pairs = _pairs_of(candidates.rows)
        assert len(pairs) == 2
        unanimous = {str(pairs[0][1][PROMPT_ID_COLUMN]): SAMPLES_PER_PROMPT}
        ordered = list(reversed(candidates.rows)) if reverse else list(candidates.rows)
        records = scripted_sweep(ordered, unanimous)
        trace = write_trace(tmp_path / "sweep.jsonl", ordered, records)
        return select_breadth(trace, list(candidates.rows), spec)

    def test_the_acceptable_orientation_does_not_hide_the_reason(self, tmp_path: Path) -> None:
        selection = self._selection(tmp_path, reverse=False)
        assert selection.strata[0].dropped_by_reason == {str(DropReason.COOP_FRACTION_ABOVE_MAX): 1}

    def test_the_same_pair_reads_the_same_way_in_either_trace_order(self, tmp_path: Path) -> None:
        forward = self._selection(tmp_path / "forward", reverse=False)
        reversed_order = self._selection(tmp_path / "reversed", reverse=True)
        assert forward.strata[0].dropped_by_reason == reversed_order.strata[0].dropped_by_reason


class TestTheTrustStratumRecordsItsRealTarget:
    """Both artifacts of one run have to agree on what every stratum asked for.

    The stratum artifact is the target-versus-realised record banked beside the corpus, and a trust
    stratum recorded as targeting nothing reads as unplanned surplus: a roster that filled exactly
    cannot be told from one that came in short, and the artifact's targets no longer sum to the plan's
    corpus size.
    """

    def test_the_selection_artifact_and_the_manifest_agree_on_every_quota(
        self, tmp_path: Path
    ) -> None:
        candidates = candidates_for(TRUST_TEST_SPEC)
        records = scripted_sweep(candidates.rows, {})
        trace = write_trace(tmp_path / "sweep.jsonl", candidates.rows, records)
        selection = select_breadth(trace, list(candidates.rows), TRUST_TEST_SPEC)

        assert {plan.key: plan.quota_pairs for plan in candidates.strata} == {
            outcome.plan.key: outcome.plan.quota_pairs for outcome in selection.strata
        }
        trust = next(
            outcome for outcome in selection.strata if outcome.plan.band == TRUST_STRATUM_BAND
        )
        assert trust.plan.quota_pairs == 16
        assert trust.n_kept_units == 16

    def test_a_trust_stratum_that_came_in_short_reads_as_short(self, tmp_path: Path) -> None:
        """The reading the zero target made impossible: kept below target rather than surplus."""
        candidates = candidates_for(TRUST_TEST_SPEC)
        unanimous = {
            str(row[PROMPT_ID_COLUMN]): SAMPLES_PER_PROMPT
            for row in candidates.rows
            if str(row[FRAMING_ID_COLUMN]) == FRAMING_ID_UNSET
        }
        records = scripted_sweep(candidates.rows, dict(list(unanimous.items())[:4]))
        trace = write_trace(tmp_path / "sweep.jsonl", candidates.rows, records)
        selection = select_breadth(trace, list(candidates.rows), TRUST_TEST_SPEC)
        trust = next(
            outcome for outcome in selection.strata if outcome.plan.band == TRUST_STRATUM_BAND
        )
        assert trust.plan.quota_pairs == 16
        assert trust.n_kept_units < trust.plan.quota_pairs


class TestTheWrittenCorpusFillsWholeSteps:
    """TRL's sampler drops the incomplete chunk of every epoch, so a corpus has to be whole steps.

    Left unfilled, some prompt class is visited less often than another for the whole run and nothing
    in the artifacts says which. The trim removes whole units, never one orientation of a pair.
    """

    def test_the_corpus_is_a_multiple_of_the_step_prompt_count(self, tmp_path: Path) -> None:
        candidates = candidates_for()
        records = scripted_sweep(candidates.rows, {})
        trace = write_trace(tmp_path / "sweep.jsonl", candidates.rows, records)
        selection = select_breadth(trace, list(candidates.rows), TEST_SPEC)

        assert len(candidates.rows) == 10, "five candidate pairs, not a multiple of eight"
        assert len(selection.rows) == 8
        trimmed = sum(
            count
            for outcome in selection.strata
            for reason, count in outcome.dropped_by_reason.items()
            if reason == str(BreadthDropReason.TRIMMED_TO_STEP_MULTIPLE)
        )
        assert trimmed == 1
        assert len(_pairs_of(selection.rows)) == 4

    def test_the_trim_keeps_both_orientations_of_every_pair_it_keeps(self, tmp_path: Path) -> None:
        candidates = candidates_for()
        records = scripted_sweep(candidates.rows, {})
        trace = write_trace(tmp_path / "sweep.jsonl", candidates.rows, records)
        selection = select_breadth(trace, list(candidates.rows), TEST_SPEC)
        groups: dict[tuple[object, ...], int] = {}
        for row in selection.rows:
            key = pair_identity(row)
            groups[key] = groups.get(key, 0) + 1
        assert set(groups.values()) == {2}


class TestTheBuilderRefusesACouplingClauseOutsideTheTwinFraming:
    """A training row may not state that the counterpart decides as this side does, unless it is twin.

    Every care grading pays a completion against the group's realised mix and reads no framing, so a
    training prompt asserting the coupling would state one counterpart and pay another. On the eval
    split the same clause is the measurement, which is why the refusal is split-conditional.
    """

    def test_the_builder_refuses_the_twin_route_sentence_under_a_kin_label(self) -> None:
        spec = _spec_with_framings({BAND_NEAR: (DEPENDENT,), BAND_MID: (), BAND_FAR: ()})
        clause = f"{synthetic_clause(DEPENDENT)[:-1]}, {DECIDES_IN_STEP_SENTENCE}"
        with pytest.raises(ValueError, match="decision travels with this side's"):
            build_candidates(spec, runtime_clauses={DEPENDENT: clause})

    def test_the_twin_framing_itself_still_trains_under_its_own_clause(self) -> None:
        spec = _spec_with_framings({BAND_NEAR: (FRAMING_TWIN,), BAND_MID: (), BAND_FAR: ()})
        candidates = build_candidates(spec)
        clause = COUNTERPART_FRAMINGS[FRAMING_TWIN]
        assert clause is not None
        assert all(clause in str(row["prompt"]) for row in candidates.rows)

    def test_the_eval_split_still_renders_a_coupling_clause_under_any_label(self) -> None:
        rows = render_matrix_rows_under_clause(
            "twin-pd",
            CARE_GRADING,
            clause=COUNTERPART_FRAMINGS["different-ai-coupled"],
            framing_label="different-ai-coupled",
            split="eval",
        )
        assert rows
        assert all(row[FRAMING_ID_COLUMN] == "different-ai-coupled" for row in rows)


class TestTheDrawIsSeededFromIdentity:
    """The pool is drawn by hashing each frame's identity, never by walking an RNG stream.

    Two properties follow, and both matter for a corpus built once and swept expensively: the same spec
    draws the same frames however the strata were rendered, and adding a game to the spec leaves every
    other game's drawn frames exactly as they were. An RNG stream promises neither.
    """

    def test_two_builds_of_one_spec_draw_the_same_frames(self) -> None:
        first = candidates_for()
        second = candidates_for()
        assert [row[PROMPT_ID_COLUMN] for row in first.rows] == [
            row[PROMPT_ID_COLUMN] for row in second.rows
        ]

    def test_adding_a_game_does_not_reroll_the_other_games_frames(self) -> None:
        before = candidates_for()
        widened = BreadthSpec(
            games=(
                *TEST_SPEC.games,
                BreadthGame(
                    group="chicken",
                    game_ids=("chicken",),
                    payoff_variants=("standard",),
                    pairs_per_variant_by_band={BAND_NEAR: 1, BAND_MID: 1, BAND_FAR: 1},
                ),
            ),
            framings_by_band=dict(TEST_SPEC.framings_by_band),
            grading=CARE_GRADING,
            oversample_by_band=dict(TEST_SPEC.oversample_by_band),
        )
        after = candidates_for(widened)
        assert {str(row[PROMPT_ID_COLUMN]) for row in before.rows} <= {
            str(row[PROMPT_ID_COLUMN]) for row in after.rows
        }

    def test_a_different_seed_draws_a_different_pool(self) -> None:
        reseeded = BreadthSpec(
            games=TEST_SPEC.games,
            framings_by_band=dict(TEST_SPEC.framings_by_band),
            grading=CARE_GRADING,
            seed=17,
            oversample_by_band=dict(TEST_SPEC.oversample_by_band),
        )
        assert {str(row[PROMPT_ID_COLUMN]) for row in candidates_for().rows} != {
            str(row[PROMPT_ID_COLUMN]) for row in candidates_for(reseeded).rows
        }


class TestTheSelectionRefusesAMismatchedPool:
    """A verdict is only readable against the prompts it was measured on.

    Two cases, both silent otherwise: a candidate the sweep never measured would shrink the corpus with
    nothing saying why, and a candidate whose text changed under the same id would carry a verdict about
    a prompt the corpus no longer holds.
    """

    def test_a_candidate_the_sweep_never_measured_is_refused(self, tmp_path: Path) -> None:
        candidates = candidates_for()
        records = scripted_sweep(candidates.rows[:-2], {})
        trace = write_trace(tmp_path / "sweep.jsonl", candidates.rows[:-2], records)
        with pytest.raises(ValueError, match="candidates were never swept"):
            select_breadth(trace, list(candidates.rows), TEST_SPEC)

    def test_a_candidate_whose_prompt_changed_is_refused(self, tmp_path: Path) -> None:
        candidates = candidates_for()
        records = scripted_sweep(candidates.rows, {})
        trace = write_trace(tmp_path / "sweep.jsonl", candidates.rows, records)
        edited = [dict(row) for row in candidates.rows]
        edited[0]["prompt"] = f"{edited[0]['prompt']}\n\nAn afterthought."
        with pytest.raises(ValueError, match="'prompt'"):
            select_breadth(trace, edited, TEST_SPEC)

    def test_a_candidate_whose_reward_columns_changed_is_refused(self, tmp_path: Path) -> None:
        """A rebuild can move the columns that decide the reward and leave the prompt text alone.

        The grading, the framing, the game and the payoff cells pick the stratum, the selection rule
        and the reward branch, so a pool rebuilt under an edited spec carries verdicts about prompts
        that would train differently -- and a check on the rendered text alone cannot see it.
        """
        candidates = candidates_for()
        records = scripted_sweep(candidates.rows, {})
        trace = write_trace(tmp_path / "sweep.jsonl", candidates.rows, records)
        for column, value in (("payoff_cc", 99.0), (GRADING_COLUMN, "group-mix")):
            edited = [dict(row) for row in candidates.rows]
            edited[0][column] = value
            with pytest.raises(ValueError, match=f"'{column}'"):
                select_breadth(trace, edited, TEST_SPEC)

    def test_a_row_of_a_game_the_spec_does_not_cover_is_refused(self, tmp_path: Path) -> None:
        """The pool and its trace agree here, so the spec is the only thing that can object."""
        rows = [dict(row) for row in candidates_for().rows]
        rows[0]["game_id"] = "hi-lo"
        records = scripted_sweep(rows, {})
        trace = write_trace(tmp_path / "sweep.jsonl", rows, records)
        with pytest.raises(ValueError, match="belongs to no stratum of this spec"):
            select_breadth(trace, rows, TEST_SPEC)


class TestTheSpecRefusesAGridItCannotRender:
    def test_a_band_left_out_of_the_framings_is_refused(self) -> None:
        with pytest.raises(ValueError, match="every band of"):
            BreadthSpec(
                games=TEST_SPEC.games,
                framings_by_band={BAND_NEAR: (FRAMING_TWIN,)},
                grading=CARE_GRADING,
            )

    def test_a_framing_in_two_bands_is_refused(self) -> None:
        with pytest.raises(ValueError, match="bands; a framing has exactly one distance"):
            BreadthSpec(
                games=TEST_SPEC.games,
                framings_by_band={
                    BAND_NEAR: (FRAMING_TWIN,),
                    BAND_MID: (FRAMING_TWIN,),
                    BAND_FAR: (),
                },
                grading=CARE_GRADING,
            )

    def test_an_oversample_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="below 1"):
            BreadthSpec(
                games=TEST_SPEC.games,
                framings_by_band=dict(TEST_SPEC.framings_by_band),
                grading=CARE_GRADING,
                oversample_by_band={BAND_NEAR: 0.5, BAND_MID: 1.0, BAND_FAR: 1.0},
            )

    def test_a_quota_larger_than_the_roster_is_refused(self) -> None:
        spec = BreadthSpec(
            games=(
                BreadthGame(
                    group="chicken",
                    game_ids=("chicken",),
                    payoff_variants=("standard",),
                    pairs_per_variant_by_band={BAND_NEAR: 99, BAND_MID: 0, BAND_FAR: 0},
                ),
            ),
            framings_by_band={BAND_NEAR: (FRAMING_TWIN,), BAND_MID: (), BAND_FAR: ()},
            grading=CARE_GRADING,
        )
        with pytest.raises(ValueError, match="pooled training roster holds"):
            build_candidates(spec)

    def test_an_unknown_framing_names_both_sources(self) -> None:
        spec = _spec_with_framings({BAND_NEAR: ("no-such-framing",), BAND_MID: (), BAND_FAR: ()})
        with pytest.raises(ValueError, match="nor one of the runtime clauses supplied"):
            build_candidates(spec)

    def test_a_spec_round_trips_through_json(self) -> None:
        payload = json.loads(json.dumps(PROSOCIAL_BREADTH_SPEC.to_json_dict()))
        assert BreadthSpec.from_json_dict(payload).to_json_dict() == (
            PROSOCIAL_BREADTH_SPEC.to_json_dict()
        )


class TestTheWaveSpecAndTheTrustRows:
    """The plan's own grid, and the one game in it that carries no counterpart framing at all."""

    def test_the_wave_spec_targets_the_planned_corpus_size(self) -> None:
        matrix_pairs = sum(
            plan.quota_pairs
            for plan in _wave_candidates().strata
            if plan.band != TRUST_STRATUM_BAND
        )
        trust_rows = sum(
            plan.quota_pairs
            for plan in _wave_candidates().strata
            if plan.band == TRUST_STRATUM_BAND
        )
        assert matrix_pairs * 2 + trust_rows == 336

    def test_the_trust_rows_carry_the_unset_framing_marker(self) -> None:
        trust = [
            row
            for row in _wave_candidates().rows
            if row["game_id"] == PROSOCIAL_BREADTH_SPEC.trust_game_id
        ]
        assert len(trust) == 32
        assert {row[FRAMING_ID_COLUMN] for row in trust} == {FRAMING_ID_UNSET}

    def test_every_candidate_carries_the_framing_column(self) -> None:
        assert all(FRAMING_ID_COLUMN in row for row in _wave_candidates().rows)

    def test_the_pool_is_larger_than_the_target_in_every_band(self) -> None:
        for plan in _wave_candidates().strata:
            if plan.band == TRUST_STRATUM_BAND or plan.quota_pairs == 0:
                continue
            assert plan.candidate_pairs > plan.quota_pairs, plan.key


_WAVE_CANDIDATES: dict[str, BreadthCandidates] = {}


def _wave_candidates() -> BreadthCandidates:
    """Build the plan's grid once per session; it renders 632 prompts and nothing mutates them."""
    if "candidates" not in _WAVE_CANDIDATES:
        clauses = {
            framing_id: synthetic_clause(framing_id)
            for framing_id in PROSOCIAL_BREADTH_SPEC.framing_ids
            if framing_id not in COUNTERPART_FRAMINGS
        }
        _WAVE_CANDIDATES["candidates"] = build_candidates(
            PROSOCIAL_BREADTH_SPEC, runtime_clauses=clauses
        )
    return _WAVE_CANDIDATES["candidates"]


class TestTheCommandLine:
    """The two subcommands end to end, which is the only check that covers the writes together."""

    def test_build_then_select_writes_a_corpus_and_its_stratum_artifact(
        self, tmp_path: Path
    ) -> None:
        spec_path = tmp_path / "spec.json"
        spec_path.write_text(json.dumps(TEST_SPEC.to_json_dict()), encoding="utf-8")
        build_dir = tmp_path / "build"
        assert main(["build", "--spec", str(spec_path), "--out-dir", str(build_dir)]) == 0
        candidate_path = build_dir / "breadth-candidates.jsonl"
        manifest = json.loads(
            (build_dir / "breadth-candidates-manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["n_rows"] == 10
        assert manifest["spec"] == TEST_SPEC.to_json_dict()
        assert manifest["runtime_framing_ids"] == []
        assert manifest["runtime_clauses_digest"] is None

        rows = [dict(row) for row in read_jsonl(candidate_path)]
        records = scripted_sweep(rows, {})
        sweep_path = tmp_path / "sweep" / "sweep.jsonl"
        sweep_path.parent.mkdir(parents=True, exist_ok=True)
        write_trace(sweep_path, rows, records)

        select_dir = tmp_path / "select"
        assert (
            main(
                [
                    "select",
                    "--spec",
                    str(spec_path),
                    "--sweep",
                    str(sweep_path),
                    "--candidates",
                    str(candidate_path),
                    "--out-dir",
                    str(select_dir),
                ]
            )
            == 0
        )
        corpus = read_jsonl(select_dir / "breadth-corpus.jsonl")
        assert len(corpus) == 8
        artifact = json.loads((select_dir / "breadth-strata.json").read_text(encoding="utf-8"))
        assert artifact["n_rows"] == 8
        assert artifact["pool_hash"]
        assert artifact["corpus_sha256"]
        assert {outcome["rule"] for outcome in artifact["strata"]} == {
            RULE_STANDARD_BAND,
            RULE_FAR_PAIR_ANY_COOPERATION,
        }
        assert all("base_cooperation" in outcome for outcome in artifact["strata"])

    def _swept(self, tmp_path: Path, spec: BreadthSpec, cooperations: Mapping[str, int]) -> Path:
        """Build a spec's pool through the CLI, sweep it as scripted, and return the build directory."""
        spec_path = tmp_path / "spec.json"
        spec_path.write_text(json.dumps(spec.to_json_dict()), encoding="utf-8")
        build_dir = tmp_path / "build"
        assert main(["build", "--spec", str(spec_path), "--out-dir", str(build_dir)]) == 0
        rows = [dict(row) for row in read_jsonl(build_dir / "breadth-candidates.jsonl")]
        write_trace(tmp_path / "sweep.jsonl", rows, scripted_sweep(rows, cooperations))
        return build_dir

    def test_the_fill_flag_reaches_the_selection_and_is_off_without_it(
        self, tmp_path: Path
    ) -> None:
        """Both readings of one swept pool, so the flag's whole effect is the diff between them."""
        spec = _fill_spec()
        cooperations = _cooperations_by_orientation(
            candidates_for(spec),
            near=[(4, 4), (0, 0), (0, 0), (0, 0)],
            mid=[(0, 0), (0, 0), (8, 8), (8, 8)],
            far=[(1, 0), (0, 0), (0, 0), (0, 0)],
        )
        build_dir = self._swept(tmp_path, spec, cooperations)
        common = [
            "select",
            "--spec",
            str(tmp_path / "spec.json"),
            "--sweep",
            str(tmp_path / "sweep.jsonl"),
            "--candidates",
            str(build_dir / "breadth-candidates.jsonl"),
        ]
        assert main([*common, "--out-dir", str(tmp_path / "registered")]) == 0
        assert main([*common, "--fill-pure-to-quota", "--out-dir", str(tmp_path / "filled")]) == 0

        registered = read_jsonl(tmp_path / "registered" / "breadth-corpus.jsonl")
        filled = read_jsonl(tmp_path / "filled" / "breadth-corpus.jsonl")
        assert len(registered) == 4
        assert len(filled) == 10
        registered_artifact = json.loads(
            (tmp_path / "registered" / "breadth-strata.json").read_text(encoding="utf-8")
        )
        filled_artifact = json.loads(
            (tmp_path / "filled" / "breadth-strata.json").read_text(encoding="utf-8")
        )
        assert "selection_options" not in registered_artifact
        assert filled_artifact["selection_options"]["fill_rule"] == RULE_FILL_PURE_TO_QUOTA
        assert filled_artifact["selection_options"]["trust_min_split_std"] is None

    def test_the_trust_floor_flag_reaches_the_selection(self, tmp_path: Path) -> None:
        spec = _trust_floor_spec()
        cooperations = _trust_script(candidates_for(spec), n_with_spread=8)
        build_dir = self._swept(tmp_path, spec, cooperations)
        assert (
            main(
                [
                    "select",
                    "--spec",
                    str(tmp_path / "spec.json"),
                    "--sweep",
                    str(tmp_path / "sweep.jsonl"),
                    "--candidates",
                    str(build_dir / "breadth-candidates.jsonl"),
                    "--trust-min-split-std",
                    "0",
                    "--out-dir",
                    str(tmp_path / "select"),
                ]
            )
            == 0
        )
        artifact = json.loads(
            (tmp_path / "select" / "breadth-strata.json").read_text(encoding="utf-8")
        )
        assert artifact["selection_options"]["trust_min_split_std"] == 0.0
        assert artifact["selection_options"]["fill_rule"] is None
        assert artifact["realised_composition"]["rows_by_band"][TRUST_STRATUM_BAND] == 16

    def test_naming_both_a_spec_and_a_preset_is_refused(self, tmp_path: Path) -> None:
        spec_path = tmp_path / "spec.json"
        spec_path.write_text(json.dumps(TEST_SPEC.to_json_dict()), encoding="utf-8")
        with pytest.raises(ValueError, match="exactly one of --spec and --preset"):
            main(
                [
                    "build",
                    "--spec",
                    str(spec_path),
                    "--preset",
                    "plumbing-smoke",
                    "--out-dir",
                    str(tmp_path / "out"),
                ]
            )

    def test_selecting_under_a_grading_the_rows_do_not_carry_is_refused(
        self, tmp_path: Path
    ) -> None:
        """`--grading` is shared by both subcommands and only the build renders under it.

        On `select` the rows are copied verbatim, so an override there relabels the artifact alone and
        banks a stratum record naming a reward the corpus does not carry. The refusal names the pass
        that does change a corpus's grading.
        """
        spec_path = tmp_path / "spec.json"
        spec_path.write_text(json.dumps(TEST_SPEC.to_json_dict()), encoding="utf-8")
        build_dir = tmp_path / "build"
        assert main(["build", "--spec", str(spec_path), "--out-dir", str(build_dir)]) == 0
        candidate_path = build_dir / "breadth-candidates.jsonl"
        rows = [dict(row) for row in read_jsonl(candidate_path)]
        sweep_path = tmp_path / "sweep.jsonl"
        write_trace(sweep_path, rows, scripted_sweep(rows, {}))
        argv = [
            "select",
            "--spec",
            str(spec_path),
            "--sweep",
            str(sweep_path),
            "--candidates",
            str(candidate_path),
            "--out-dir",
            str(tmp_path / "select"),
        ]
        with pytest.raises(ValueError, match=r"games\.regrade_corpus"):
            main([*argv, "--grading", "group-mix"])
        assert main([*argv, "--grading", CARE_GRADING]) == 0

    def test_the_plumbing_smoke_preset_carries_the_trust_leg(self, tmp_path: Path) -> None:
        """Four games, three framings and the trust sender's roster, no sweep behind any of them.

        The corpus a 2B plumbing smoke trains on: the point is that the arm path executes over a mixed
        corpus, so the prompts are not selected and the log says so. The trust rows are there because
        the care family's send branch is a separate reward path, and a smoke that omits it never runs
        that path through the trainer at all.
        """
        framings = write_framings_file(tmp_path / "framings.json")
        corpus_path = tmp_path / "corpus-2b-smoke.jsonl"
        assert (
            main(
                [
                    "build",
                    "--preset",
                    "plumbing-smoke",
                    "--framings-file",
                    str(framings),
                    "--out-dir",
                    str(tmp_path / "build"),
                    "--corpus-out",
                    str(corpus_path),
                ]
            )
            == 0
        )
        rows = [dict(row) for row in read_jsonl(corpus_path)]
        assert len(rows) == 24
        assert len(rows) % PLUMBING_SMOKE_SPEC.step_prompt_multiple == 0
        assert len({str(row["game_id"]) for row in rows}) == 4
        assert len({str(row[FRAMING_ID_COLUMN]) for row in rows}) == 4
        assert {str(row["grading"]) for row in rows} == {PLUMBING_SMOKE_SPEC.grading}
        trust = [row for row in rows if str(row[FRAMING_ID_COLUMN]) == FRAMING_ID_UNSET]
        assert {str(row["game_id"]) for row in trust} == {PLUMBING_SMOKE_SPEC.trust_game_id}
        assert len(trust) == 16

    def test_an_unselected_corpus_that_is_not_whole_steps_is_refused(self, tmp_path: Path) -> None:
        spec_path = tmp_path / "spec.json"
        spec_path.write_text(json.dumps(TEST_SPEC.to_json_dict()), encoding="utf-8")
        with pytest.raises(ValueError, match="is not a multiple of"):
            main(
                [
                    "build",
                    "--spec",
                    str(spec_path),
                    "--out-dir",
                    str(tmp_path / "build"),
                    "--corpus-out",
                    str(tmp_path / "corpus.jsonl"),
                ]
            )

    def test_a_runtime_framings_file_is_digested_and_never_quoted(self, tmp_path: Path) -> None:
        """The manifest pins the clauses it rendered without carrying a word of them."""
        framings = write_framings_file(tmp_path / "framings.json")
        loaded = load_framings(framings)
        build_dir = tmp_path / "build"
        assert (
            main(
                [
                    "build",
                    "--preset",
                    "plumbing-smoke",
                    "--framings-file",
                    str(framings),
                    "--out-dir",
                    str(build_dir),
                ]
            )
            == 0
        )
        text = (build_dir / "breadth-candidates-manifest.json").read_text(encoding="utf-8")
        manifest = json.loads(text)
        assert manifest["runtime_framing_ids"] == [DEPENDENT]
        assert manifest["runtime_clauses_digest"]
        assert loaded.clauses[DEPENDENT] not in text
