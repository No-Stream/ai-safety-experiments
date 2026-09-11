"""Build and select the mixed-game, mixed-framing training corpus a breadth arm trains on.

Every training corpus banked before this one was one game under one counterpart paragraph, selected
by one band on the base model's cooperation rate. A breadth arm asks a different question -- does an
other-regarding disposition reach games and counterpart identities the training never covered -- and
that question needs the training corpus itself to span a grid: five matrix games over two rosters and
several payoff variants, plus the trust sender, each rendered under one of several counterpart
framings from "a copy of your weights" to "a human".

Three things in that grid are not what a single-game selector does, and each is why this module
exists rather than a flag on `games.select_prompts`.

**The pool is drawn before it is swept, from prompt identity.** The full grid is a few thousand
prompts and the sweep is the expensive half, so each stratum is oversampled against its target by a
factor that reflects how often the standard band keeps a prompt there, and the draw is a hash of the
prompt's own identity rather than a walk of an RNG stream. Two builds of the same spec draw the same
frames whatever order anything ran in, and adding a game to the spec does not re-roll the frames the
other games drew.

**The far framings need their own rule.** At 9B the base cooperates 1 to 3 percent under
`different-ai` and `human`, so the standard `[0.125, 0.875]` band keeps almost no counterbalanced pair
there and the corpus would silently contain only the near framings -- exactly the breadth the arm is
supposed to train. The far rule is pair-level and much weaker: keep the pair when its two orientations'
draws together hold at least one cooperation. That admits prompts whose groups will mostly be pure,
which is a real cost paid deliberately, and the trainer logs the pure-group fraction per framing so
the cost is measured rather than assumed.

**Quotas are targets and the realised composition is the record.** A stratum that comes in short is a
recorded fact rather than a silent reweighting, and a game whose strata keep too few pairs drops out
of the corpus whole (the hi-lo precedent: a capability datum, not a plumbing failure). Everything the
selection did lands in a stratum artifact beside the corpus: candidates, kept, dropped by reason, the
rule applied, the base cooperation distribution, and the sweep's own pool hash.

The corpus is written under the arm's grading directly, because selection is action-based and
grading-independent; the queued control regrades the same file rather than re-sweeping it.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import statistics
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from games.framing_stimulus import load_framings
from games.prompts import (
    COUNTERPART_FRAMINGS,
    FRAMING_ANOTHER_AI,
    FRAMING_DIFFERENT_AI,
    FRAMING_HUMAN,
    FRAMING_TWIN,
    FRAMING_UNSTATED,
    LABEL_PRINT_ORDER_CANONICAL,
    LABEL_PRINT_ORDERS,
    PAYOFF_VARIANTS,
    SINGLE_VARIANT,
    SPLIT_TRAIN,
    STAG_HUNT_PAYOFF_VARIANTS,
    TRUST_PAYOFF_VARIANTS,
    Scenario,
    assert_counterpart_paragraph_is_the_only_insertion,
    generate_prompt_rows,
    matrix_frames_for_split,
    render_matrix_rows_under_clause,
)
from games.rewards import FRAMING_ID_COLUMN, FRAMING_ID_UNSET, grading_cli_value, is_grading
from games.select_prompts import (
    COLUMNS_VARYING_WITHIN_A_PAIR,
    DEFAULT_MAX_COOP,
    DEFAULT_MIN_COOP,
    DEFAULT_MIN_PARSEABLE_FRACTION,
    DEFAULT_MIN_SPLIT_STD,
    GRADING_COLUMN,
    LABEL_ORIENTATIONS_PER_SCENARIO,
    META_RECORD_KIND,
    PROMPT_COLUMN,
    PROMPT_ID_COLUMN,
    SWEEP_RECORD_KIND,
    DropReason,
    PromptSweepRecord,
    SampleOutcome,
    is_counterbalanced,
    judge_prompt,
    judge_prompts,
    pair_identity,
    read_jsonl,
    rows_file_digest,
    write_corpus,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)

type Row = dict[str, Any]

BAND_NEAR = "near"
BAND_MID = "mid"
BAND_FAR = "far"
FRAMING_BANDS: tuple[str, ...] = (BAND_NEAR, BAND_MID, BAND_FAR)
"""How far a counterpart framing is from "a copy of your own weights", which is what the corpus varies.

The band is the axis the arm's whole reading runs along, and it decides three separate things: how many
skin pairs a game contributes under a framing, how hard the candidate pool is oversampled there, and
which selection rule the stratum is judged by. A framing belongs to exactly one band.
"""

# The far strata are judged by the pair rule rather than the standard band, because at 9B the base
# cooperates 1 to 3 percent under a decoupled non-AI counterpart: eight draws show any cooperation
# about a fifth of the time and a counterbalanced pair clears [0.125, 0.875] about a twentieth, so the
# standard band would empty the far strata and the corpus would carry only the framings the reward
# already reaches. Which strata take which rule is recorded per stratum in the artifact.
RULE_STANDARD_BAND = "standard-band"
RULE_FAR_PAIR_ANY_COOPERATION = "far-pair-any-cooperation"

# An optional third rule, off unless a caller asks for it, and never part of the registered selection:
# top a short mid or far stratum up with pairs the base played unanimously. Two facts argue for it and
# both come from the 9B base sweep the registered corpus was selected from. The base is near
# deterministic in most cells -- on the PD-type games every framing outside twin and dependent defected
# in 55 to 70 percent of its strata's prompts with no cooperation at all in eight draws, and on stag
# hunt and chicken the near framings cooperated unanimously -- so the standard band kept nothing in
# most mid strata and the mid band came in at 20 rows. And training runs dynamic sampling at oversample
# 2, which drops a group whose rewards are all equal at step time: a pure-at-base prompt therefore
# costs generation and never a gradient, so admitting one cannot dilute the step's signal, while the
# prompts it admits are exactly the ones whose behaviour a care-weighted arm is supposed to move.
RULE_FILL_PURE_TO_QUOTA = "fill-pure-to-quota"

# Which bands the fill may top up. Near strata keep the mixed-only rule: they realised 76 rows against
# the registered quotas and have no shortage to fix, and relaxing the band where the reward already
# reaches would spend corpus on prompts that are already movable. The trust stratum is out for a
# different reason -- its target is its whole roster rather than a pair count, and its own knob is the
# send-spread floor (`trust_min_split_std`), where a floor of zero is the same relaxation.
FILL_BANDS: tuple[str, ...] = (BAND_MID, BAND_FAR)

# A cooperation rate of one half is the most a prompt's eight draws can disagree, so distance from it
# is how close an orientation came to a group GRPO's advantage can see.
MIXED_COOP_FRACTION = 0.5

# The candidate pool per stratum, as a multiple of the kept target. Near strata keep about three
# quarters of their pairs at the 9B twin base (0.38 to 0.45 cooperation, the 9B twin sweep kept 72
# percent), mid strata roughly half, and a far pair passes its own rule with probability 0.15 to 0.4 at
# a 1 to 3 percent base. A factor below 1 would draw fewer candidates than the target it feeds.
DEFAULT_OVERSAMPLE_BY_BAND: dict[str, float] = {BAND_NEAR: 1.4, BAND_MID: 2.0, BAND_FAR: 3.0}

# TRL's RepeatSampler chunks the shuffled prompt indices into groups of `prompts_per_step` and drops
# the incomplete chunk at the end of each epoch, so a corpus whose size is not a whole number of steps
# drops some prompt class more often than another for the whole run. Eight is the reference arm's
# prompts per step.
DEFAULT_STEP_PROMPT_MULTIPLE = 8

# A game keeping less than this share of its total pair quota drops from the corpus whole rather than
# contributing a handful of prompts nothing can be read off. The threshold is a spec field so the
# number is recorded with the run rather than buried here, and dropping is per game because the pairs
# a game keeps are what its own transfer reading rests on.
DEFAULT_MIN_KEPT_FRACTION_PER_GAME = 0.25


class BreadthDropReason(StrEnum):
    """Why a candidate pair or singleton did not reach the corpus, beyond the sweep's own verdicts.

    `games.select_prompts.DropReason` covers everything the standard band decides. These are the four
    decisions this module adds on top, and they are separate values rather than one "not selected"
    because they answer different questions: whether the base behaviour was usable at all, whether the
    stratum was already full, whether the game was worth carrying, and whether the corpus had to lose
    a unit to fill a whole number of steps.
    """

    FAR_PAIR_NO_COOPERATION = "far-pair-no-cooperation"
    FAR_PAIR_TOO_FEW_PARSEABLE = "far-pair-too-few-parseable"
    OVER_QUOTA = "over-quota"
    GAME_KEPT_TOO_FEW_PAIRS = "game-kept-too-few-pairs"
    TRIMMED_TO_STEP_MULTIPLE = "trimmed-to-step-multiple"


def _assert_the_framing_column_is_part_of_the_pair_key() -> None:
    """Refuse at import if the framing column ever joins the columns a label swap is allowed to change.

    `pair_identity` keys a counterbalanced pair on every column outside
    `COLUMNS_VARYING_WITHIN_A_PAIR`, so the framing column being outside that set is what keeps one
    scenario's two orientations under one framing from pairing with its orientations under another. If
    it were ever added there, four rows would share a pair key, `judge_prompts` would raise on the
    group size, and this module's own far rule would pool two framings' draws into one decision. The
    check is here rather than in a test because it is the assumption every pair-level rule below rests
    on.
    """
    if FRAMING_ID_COLUMN in COLUMNS_VARYING_WITHIN_A_PAIR:
        raise RuntimeError(
            f"{FRAMING_ID_COLUMN!r} is in COLUMNS_VARYING_WITHIN_A_PAIR, so two framings' renderings "
            f"of one scenario now share a counterbalanced-pair key. A breadth corpus renders every "
            f"frame under several framings, so the framing has to be part of the pair key."
        )


_assert_the_framing_column_is_part_of_the_pair_key()


@dataclass(frozen=True)
class BreadthGame:
    """One game group of the grid: the games whose training rosters pool into one pair quota.

    A group is not always one game id. The prisoner's dilemma family renders one payoff table over two
    rosters, the house skins as `twin-pd` and the register-diverse bank as `pd-reskin`, and the plan
    weights the family as a whole at about half the corpus. Pooling them into one quota is what lets
    that weight be stated once; each row still carries the game id of the roster it came from, so the
    held-out claim of each bank stays readable.

    `pairs_per_variant_by_band` is counterbalanced pairs per payoff variant per framing in that band,
    which is the level the plan's composition table states its weights at.
    """

    group: str
    game_ids: tuple[str, ...]
    payoff_variants: tuple[str, ...]
    pairs_per_variant_by_band: Mapping[str, int]

    def __post_init__(self) -> None:
        """Refuse a group nothing could render, before a frame is drawn."""
        if not self.group:
            raise ValueError("a breadth game group needs a name; it keys every stratum record")
        if not self.game_ids:
            raise ValueError(f"group {self.group!r} names no games")
        if len(set(self.game_ids)) != len(self.game_ids):
            raise ValueError(f"group {self.group!r} names a game twice: {self.game_ids}")
        if not self.payoff_variants:
            raise ValueError(f"group {self.group!r} names no payoff variants")
        missing = [band for band in FRAMING_BANDS if band not in self.pairs_per_variant_by_band]
        unknown = sorted(set(self.pairs_per_variant_by_band) - set(FRAMING_BANDS))
        if missing or unknown:
            raise ValueError(
                f"group {self.group!r} states pair quotas for "
                f"{sorted(self.pairs_per_variant_by_band)}; every band of {list(FRAMING_BANDS)} "
                f"needs one (missing {missing}, unknown {unknown}). A band left out would read as a "
                f"quota of zero, which is how a corpus loses a whole framing distance silently."
            )
        negative = sorted(
            band for band, pairs in self.pairs_per_variant_by_band.items() if pairs < 0
        )
        if negative:
            raise ValueError(f"group {self.group!r} states a negative pair quota for {negative}")
        if not any(self.pairs_per_variant_by_band.values()):
            raise ValueError(
                f"group {self.group!r} states a quota of zero in every band, so it contributes "
                f"nothing; drop it from the spec instead."
            )

    def to_json_dict(self) -> dict[str, Any]:
        """Flatten for the manifest and the stratum artifact."""
        return {
            "group": self.group,
            "game_ids": list(self.game_ids),
            "payoff_variants": list(self.payoff_variants),
            "pairs_per_variant_by_band": dict(self.pairs_per_variant_by_band),
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> BreadthGame:
        """Rebuild from a spec file, naming a missing key rather than defaulting it."""
        required = ("group", "game_ids", "payoff_variants", "pairs_per_variant_by_band")
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"breadth game entry lacks {missing}; keys present: {sorted(payload)}")
        return cls(
            group=str(payload["group"]),
            game_ids=tuple(str(game_id) for game_id in payload["game_ids"]),
            payoff_variants=tuple(str(variant) for variant in payload["payoff_variants"]),
            pairs_per_variant_by_band={
                str(band): int(pairs)
                for band, pairs in dict(payload["pairs_per_variant_by_band"]).items()
            },
        )


@dataclass(frozen=True)
class BreadthSpec:
    """The whole grid: which games, which framings at which distance, how many pairs, how hard drawn.

    One value that a run's whole composition can be rebuilt from, and the thing both halves of this
    module read: `build_candidates` renders the oversampled pool from it, `select_breadth` reads the
    quotas, the bands and the per-game drop threshold back out of it. It is recorded verbatim in both
    artifacts, because the realised composition only means something beside the composition asked for.

    `trust_game_id` is optional and separate from `games` because the trust sender is not frameable:
    its counterpart paragraph is the announced return rule the game's own mechanics state, so it
    contributes every training skin at both announced rates as singletons under no framing at all.
    """

    games: tuple[BreadthGame, ...]
    framings_by_band: Mapping[str, tuple[str, ...]]
    grading: str
    seed: int = 0
    oversample_by_band: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_OVERSAMPLE_BY_BAND)
    )
    trust_game_id: str | None = None
    trust_payoff_variants: tuple[str, ...] = ()
    label_print_order: str = LABEL_PRINT_ORDER_CANONICAL
    min_kept_fraction_per_game: float = DEFAULT_MIN_KEPT_FRACTION_PER_GAME
    step_prompt_multiple: int = DEFAULT_STEP_PROMPT_MULTIPLE

    def __post_init__(self) -> None:
        """Refuse a spec whose grid could not be rendered or judged, before anything is drawn."""
        if not self.games:
            raise ValueError("a breadth spec needs at least one game group")
        groups = [game.group for game in self.games]
        if len(set(groups)) != len(groups):
            raise ValueError(f"two game groups share a name: {sorted(groups)}")
        self._assert_bands_are_covered()
        self._assert_framings_are_distinct()
        if not is_grading(self.grading):
            raise ValueError(f"unknown grading {self.grading!r} in the breadth spec")
        if self.label_print_order not in LABEL_PRINT_ORDERS:
            raise ValueError(
                f"unknown label_print_order {self.label_print_order!r}; known orders: "
                f"{sorted(LABEL_PRINT_ORDERS)}"
            )
        if self.step_prompt_multiple < 1:
            raise ValueError(
                f"step_prompt_multiple must be positive, got {self.step_prompt_multiple}"
            )
        if not 0.0 <= self.min_kept_fraction_per_game <= 1.0:
            raise ValueError(
                f"min_kept_fraction_per_game must lie in [0, 1], got "
                f"{self.min_kept_fraction_per_game}"
            )
        if (self.trust_game_id is None) != (not self.trust_payoff_variants):
            raise ValueError(
                f"trust_game_id={self.trust_game_id!r} and trust_payoff_variants="
                f"{self.trust_payoff_variants} disagree about whether the corpus carries trust rows; "
                f"a game with no variants renders nothing and variants with no game render nowhere."
            )

    def _assert_bands_are_covered(self) -> None:
        for name, mapping in (
            ("framings_by_band", self.framings_by_band),
            ("oversample_by_band", self.oversample_by_band),
        ):
            missing = [band for band in FRAMING_BANDS if band not in mapping]
            unknown = sorted(set(mapping) - set(FRAMING_BANDS))
            if missing or unknown:
                raise ValueError(
                    f"{name} covers {sorted(mapping)}; every band of {list(FRAMING_BANDS)} needs an "
                    f"entry (missing {missing}, unknown {unknown})"
                )
        thin = sorted(band for band, factor in self.oversample_by_band.items() if factor < 1.0)
        if thin:
            raise ValueError(
                f"oversample factors below 1 for {thin} would draw fewer candidates than the target "
                f"they feed, so the stratum could never reach its quota"
            )

    def _assert_framings_are_distinct(self) -> None:
        seen: dict[str, str] = {}
        for band in FRAMING_BANDS:
            for framing_id in self.framings_by_band[band]:
                if framing_id in seen:
                    raise ValueError(
                        f"framing {framing_id!r} is in both the {seen[framing_id]!r} and {band!r} "
                        f"bands; a framing has exactly one distance, and two would give it two "
                        f"quotas and two selection rules"
                    )
                seen[framing_id] = band
        if not seen:
            raise ValueError("a breadth spec needs at least one counterpart framing")

    @property
    def band_by_framing(self) -> dict[str, str]:
        """Return which band each framing belongs to, which is how a swept row finds its rule."""
        return {
            framing_id: band for band in FRAMING_BANDS for framing_id in self.framings_by_band[band]
        }

    @property
    def group_by_game_id(self) -> dict[str, str]:
        """Return which group each matrix game id belongs to."""
        return {game_id: game.group for game in self.games for game_id in game.game_ids}

    @property
    def framing_ids(self) -> tuple[str, ...]:
        """Return every framing the grid renders, near band first."""
        return tuple(
            framing_id for band in FRAMING_BANDS for framing_id in self.framings_by_band[band]
        )

    def game(self, group: str) -> BreadthGame:
        """Return one group by name, naming the groups present when it is absent."""
        for game in self.games:
            if game.group == group:
                return game
        raise KeyError(
            f"no game group {group!r} in this spec; groups: {[g.group for g in self.games]}"
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Flatten the whole spec, so both artifacts can record what was asked for."""
        return {
            "games": [game.to_json_dict() for game in self.games],
            "framings_by_band": {band: list(ids) for band, ids in self.framings_by_band.items()},
            "grading": self.grading,
            "seed": self.seed,
            "oversample_by_band": dict(self.oversample_by_band),
            "trust_game_id": self.trust_game_id,
            "trust_payoff_variants": list(self.trust_payoff_variants),
            "label_print_order": self.label_print_order,
            "min_kept_fraction_per_game": self.min_kept_fraction_per_game,
            "step_prompt_multiple": self.step_prompt_multiple,
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> BreadthSpec:
        """Rebuild a spec from its JSON form, naming a missing required key."""
        missing = [key for key in ("games", "framings_by_band", "grading") if key not in payload]
        if missing:
            raise ValueError(f"breadth spec lacks {missing}; keys present: {sorted(payload)}")
        trust_game_id = payload.get("trust_game_id")
        return cls(
            games=tuple(
                BreadthGame.from_json_dict(dict(entry)) for entry in list(payload["games"])
            ),
            framings_by_band={
                str(band): tuple(str(framing_id) for framing_id in ids)
                for band, ids in dict(payload["framings_by_band"]).items()
            },
            grading=str(payload["grading"]),
            seed=int(payload.get("seed", 0)),
            oversample_by_band=(
                {
                    str(band): float(factor)
                    for band, factor in dict(payload["oversample_by_band"]).items()
                }
                if "oversample_by_band" in payload
                else dict(DEFAULT_OVERSAMPLE_BY_BAND)
            ),
            trust_game_id=None if trust_game_id is None else str(trust_game_id),
            trust_payoff_variants=tuple(
                str(variant) for variant in payload.get("trust_payoff_variants", ())
            ),
            label_print_order=str(payload.get("label_print_order", LABEL_PRINT_ORDER_CANONICAL)),
            min_kept_fraction_per_game=float(
                payload.get("min_kept_fraction_per_game", DEFAULT_MIN_KEPT_FRACTION_PER_GAME)
            ),
            step_prompt_multiple=int(
                payload.get("step_prompt_multiple", DEFAULT_STEP_PROMPT_MULTIPLE)
            ),
        )


# The section 5.4 grid of the wave-4b plan, as the default spec: the prisoner's dilemma family about
# half the corpus over two rosters and both temptations, stag hunt a fifth over its four rungs,
# chicken and public goods the rest, and the trust sender's whole training roster at both announced
# rates. Seven training framings across the three distance bands.
PROSOCIAL_BREADTH_SPEC = BreadthSpec(
    games=(
        BreadthGame(
            group="pd-family",
            game_ids=("twin-pd", "pd-reskin"),
            payoff_variants=PAYOFF_VARIANTS,
            pairs_per_variant_by_band={BAND_NEAR: 9, BAND_MID: 6, BAND_FAR: 4},
        ),
        BreadthGame(
            group="stag-hunt",
            game_ids=("stag-hunt",),
            payoff_variants=STAG_HUNT_PAYOFF_VARIANTS,
            pairs_per_variant_by_band={BAND_NEAR: 2, BAND_MID: 1, BAND_FAR: 1},
        ),
        BreadthGame(
            group="chicken",
            game_ids=("chicken",),
            payoff_variants=(SINGLE_VARIANT,),
            pairs_per_variant_by_band={BAND_NEAR: 3, BAND_MID: 2, BAND_FAR: 1},
        ),
        BreadthGame(
            group="public-goods",
            game_ids=("public-goods",),
            payoff_variants=(SINGLE_VARIANT,),
            pairs_per_variant_by_band={BAND_NEAR: 3, BAND_MID: 2, BAND_FAR: 1},
        ),
    ),
    framings_by_band={
        BAND_NEAR: (FRAMING_TWIN, "dependent"),
        BAND_MID: ("sibling-adapter", FRAMING_ANOTHER_AI, FRAMING_UNSTATED),
        BAND_FAR: (FRAMING_DIFFERENT_AI, FRAMING_HUMAN),
    },
    grading="care-alpha-1",
    trust_game_id="trust-vs-stated-return",
    trust_payoff_variants=TRUST_PAYOFF_VARIANTS,
)

# A grid small enough to smoke the whole arm path on the local card: four matrix pairs across three
# games and three framings, one of them loaded at runtime, plus the trust sender's roster at one
# announced rate. Its oversample factors are 1, so the candidates ARE the corpus and no sweep stands
# between the build and a training run -- a plumbing smoke measures whether the path executes, and a
# selection on two dozen prompts would measure nothing either way.
#
# The trust rows are here because the care reward reaches a send amount through a different branch from
# a binary action, and a smoke corpus of matrix games alone never runs that branch through the trainer
# at all -- which is the class of seam a plumbing smoke exists to catch. Sixteen of them rather than
# one: the trust stratum's quota IS its whole training roster, by the same argument that keeps it out of
# the pair trims, so the roster is what a build renders. One announced rate rather than both, because
# the second exercises the same branch at a different constant, and the smoke reads nothing either way.
PLUMBING_SMOKE_SPEC = BreadthSpec(
    games=(
        BreadthGame(
            group="pd-family",
            game_ids=("twin-pd",),
            payoff_variants=("temptation-2",),
            pairs_per_variant_by_band={BAND_NEAR: 1, BAND_MID: 0, BAND_FAR: 0},
        ),
        BreadthGame(
            group="stag-hunt",
            game_ids=("stag-hunt",),
            payoff_variants=("favoured-hunt",),
            pairs_per_variant_by_band={BAND_NEAR: 0, BAND_MID: 0, BAND_FAR: 1},
        ),
        BreadthGame(
            group="chicken",
            game_ids=("chicken",),
            payoff_variants=(SINGLE_VARIANT,),
            pairs_per_variant_by_band={BAND_NEAR: 0, BAND_MID: 0, BAND_FAR: 1},
        ),
    ),
    framings_by_band={
        BAND_NEAR: (FRAMING_TWIN, "dependent"),
        BAND_MID: (),
        BAND_FAR: (FRAMING_HUMAN,),
    },
    grading="care-alpha-1",
    oversample_by_band={BAND_NEAR: 1.0, BAND_MID: 1.0, BAND_FAR: 1.0},
    trust_game_id="trust-vs-stated-return",
    trust_payoff_variants=TRUST_PAYOFF_VARIANTS[:1],
)

SPEC_PRESETS: dict[str, BreadthSpec] = {
    "prosocial-breadth": PROSOCIAL_BREADTH_SPEC,
    "plumbing-smoke": PLUMBING_SMOKE_SPEC,
}
"""Named grids, so a spec can be given on the command line instead of written to a file each time."""


@dataclass(frozen=True, slots=True)
class StratumPlan:
    """One cell of the grid: how many pairs it targets and how many candidates were drawn for it."""

    group: str
    framing_id: str
    band: str
    payoff_variant: str
    quota_pairs: int
    candidate_pairs: int
    rule: str

    @property
    def key(self) -> str:
        """Return the stratum's name in every artifact: group, framing and payoff variant."""
        return f"{self.group}/{self.framing_id}/{self.payoff_variant}"

    def to_json_dict(self) -> dict[str, Any]:
        """Flatten for the manifest and the stratum artifact."""
        return {
            "stratum": self.key,
            "group": self.group,
            "framing_id": self.framing_id,
            "band": self.band,
            "payoff_variant": self.payoff_variant,
            "quota_pairs": self.quota_pairs,
            "candidate_pairs": self.candidate_pairs,
            "rule": self.rule,
        }


TRUST_STRATUM_BAND = "trust"
"""The trust sender's own stratum band: no counterpart framing, so no framing distance either.

Kept out of `FRAMING_BANDS` rather than folded into one of them, because every meaning a band carries
(the oversample factor, the pair quota, the far rule) is about a counterpart paragraph the trust rows
do not have. Their rule is the standard one on the send spread.
"""


@dataclass(frozen=True, slots=True)
class BreadthCandidates:
    """The oversampled pool: the rows to sweep, and the plan that says which stratum each belongs to."""

    rows: tuple[Row, ...]
    strata: tuple[StratumPlan, ...]
    spec: BreadthSpec
    runtime_framing_ids: tuple[str, ...]
    runtime_clauses_digest: str | None

    def to_manifest(self) -> dict[str, Any]:
        """Describe the pool without quoting a clause: ids and a digest, never the authored text."""
        return {
            "spec": self.spec.to_json_dict(),
            "strata": [plan.to_json_dict() for plan in self.strata],
            "n_rows": len(self.rows),
            "n_matrix_rows": sum(
                1 for row in self.rows if row[FRAMING_ID_COLUMN] != FRAMING_ID_UNSET
            ),
            "n_trust_rows": sum(
                1 for row in self.rows if row[FRAMING_ID_COLUMN] == FRAMING_ID_UNSET
            ),
            "runtime_framing_ids": list(self.runtime_framing_ids),
            "runtime_clauses_digest": self.runtime_clauses_digest,
        }


def _draw_digest(*parts: object) -> str:
    """Hash a prompt's identity into the value its draw order is sorted by.

    Sorting frames by a hash of their own identity rather than shuffling a list is what makes the pool
    reproducible without depending on execution order: adding a game to the spec, or rendering the
    strata in another order, leaves every other stratum's drawn frames exactly as they were. An RNG
    stream cannot promise that, which is the same reason the sweep records its pool hash.
    """
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _pool_for_group(game: BreadthGame) -> tuple[tuple[str, Scenario], ...]:
    """Return every (game id, training frame) a group can draw on, in registry order."""
    return tuple(
        (game_id, scenario)
        for game_id in game.game_ids
        for scenario in matrix_frames_for_split(game_id, SPLIT_TRAIN)
    )


def _candidate_pairs(quota_pairs: int, oversample: float) -> int:
    """Return how many pairs a stratum draws for a target of `quota_pairs`.

    Rounded up, because a fractional factor on a small quota would otherwise round the oversampling
    away entirely: a target of one pair at 1.4 draws two candidates, not one.
    """
    if quota_pairs == 0:
        return 0
    return max(quota_pairs, math.ceil(quota_pairs * oversample))


def _draw_frames(
    game: BreadthGame,
    *,
    framing_id: str,
    payoff_variant: str,
    n_pairs: int,
    seed: int,
) -> tuple[tuple[str, Scenario], ...]:
    """Draw `n_pairs` frames for one stratum out of the group's pooled roster, by identity hash."""
    pool = _pool_for_group(game)
    if n_pairs > len(pool):
        raise ValueError(
            f"stratum {game.group}/{framing_id}/{payoff_variant} asks for {n_pairs} frames but its "
            f"pooled training roster holds {len(pool)} ({list(game.game_ids)}). Lower the quota or "
            f"the oversample factor; drawing with replacement would put one frame in the corpus twice "
            f"under one prompt_id."
        )
    ordered = sorted(
        pool,
        key=lambda entry: _draw_digest(
            seed, game.group, framing_id, payoff_variant, entry[0], entry[1].scenario_id
        ),
    )
    return tuple(ordered[:n_pairs])


def _resolve_clause(framing_id: str, runtime_clauses: Mapping[str, str] | None) -> str | None:
    """Return one framing's counterpart clause: the registry first, the runtime file second.

    The registry answers first so a loaded file can add framings and never shadow one, which is the
    same order `games.framing_stimulus.resolve_framing_clause` uses for the eval sweep. `unstated` is a
    registered framing whose clause is None, which is why membership rather than truthiness decides.
    """
    if framing_id in COUNTERPART_FRAMINGS:
        return COUNTERPART_FRAMINGS[framing_id]
    clauses = {} if runtime_clauses is None else runtime_clauses
    if framing_id in clauses:
        return clauses[framing_id]
    raise ValueError(
        f"framing {framing_id!r} is neither a registered counterpart framing "
        f"({list(COUNTERPART_FRAMINGS)}) nor one of the runtime clauses supplied "
        f"({sorted(clauses)}). A breadth corpus renders the two kin framings and the identity ladder's "
        f"rungs from a runtime file; pass it."
    )


def _audit_one_insertion(framed: Sequence[Row], stems: Sequence[Row]) -> None:
    """Assert every framed row is its own clause-free stem plus exactly one counterpart paragraph.

    The property the whole design rests on: a row under a framing differs from the same row under no
    counterpart paragraph by one inserted paragraph, so a behavioural difference between framings is
    attributable to the clause and to nothing else. Audited at build time on every candidate rather
    than in a test over a sample, because a clause that renders two paragraphs, or that the vocabulary
    guard rewrote, would otherwise reach the GPU.
    """
    for framed_row, stem_row in zip(framed, stems, strict=True):
        if framed_row[PROMPT_ID_COLUMN] == stem_row[PROMPT_ID_COLUMN]:
            raise ValueError(
                f"the stem and the framed render share prompt_id "
                f"{framed_row[PROMPT_ID_COLUMN]!r}, so the audit is comparing a row with itself"
            )
        assert_counterpart_paragraph_is_the_only_insertion(
            stem=str(stem_row[PROMPT_COLUMN]),
            rendered=str(framed_row[PROMPT_COLUMN]),
            prompt_id=str(framed_row[PROMPT_ID_COLUMN]),
        )


def _matrix_candidate_rows(
    spec: BreadthSpec, *, runtime_clauses: Mapping[str, str] | None
) -> tuple[list[Row], list[StratumPlan]]:
    """Render the matrix half of the pool: every group under every framing at every payoff variant."""
    band_by_framing = spec.band_by_framing
    rows: list[Row] = []
    strata: list[StratumPlan] = []
    for game in spec.games:
        for band in FRAMING_BANDS:
            for framing_id in spec.framings_by_band[band]:
                clause = _resolve_clause(framing_id, runtime_clauses)
                drawn_by_variant: dict[str, tuple[tuple[str, Scenario], ...]] = {}
                for payoff_variant in game.payoff_variants:
                    quota = game.pairs_per_variant_by_band[band]
                    n_pairs = _candidate_pairs(quota, spec.oversample_by_band[band])
                    drawn_by_variant[payoff_variant] = _draw_frames(
                        game,
                        framing_id=framing_id,
                        payoff_variant=payoff_variant,
                        n_pairs=n_pairs,
                        seed=spec.seed,
                    )
                    strata.append(
                        StratumPlan(
                            group=game.group,
                            framing_id=framing_id,
                            band=band_by_framing[framing_id],
                            payoff_variant=payoff_variant,
                            quota_pairs=quota,
                            candidate_pairs=n_pairs,
                            rule=(
                                RULE_FAR_PAIR_ANY_COOPERATION
                                if band == BAND_FAR
                                else RULE_STANDARD_BAND
                            ),
                        )
                    )
                rows.extend(
                    _render_group_under_one_framing(
                        spec,
                        game,
                        framing_id=framing_id,
                        clause=clause,
                        drawn_by_variant=drawn_by_variant,
                    )
                )
    return rows, strata


def _render_group_under_one_framing(
    spec: BreadthSpec,
    game: BreadthGame,
    *,
    framing_id: str,
    clause: str | None,
    drawn_by_variant: Mapping[str, tuple[tuple[str, Scenario], ...]],
) -> list[Row]:
    """Render one group's drawn frames under one clause, one call per game id in the group.

    One render per game id rather than per stratum, because the renderer walks every payoff variant of
    the arm whatever subset of frames it is given; the strata then keep the (frame, variant) pairs each
    of them actually drew. The clause-free stems for the one-insertion audit come from the same call
    with the clause dropped, so the two row lists line up index for index.
    """
    wanted: set[tuple[str, str, str]] = {
        (game_id, scenario.scenario_id, payoff_variant)
        for payoff_variant, drawn in drawn_by_variant.items()
        for game_id, scenario in drawn
    }
    rows: list[Row] = []
    for game_id in game.game_ids:
        frames = tuple(
            dict.fromkeys(
                scenario
                for drawn in drawn_by_variant.values()
                for drawn_game_id, scenario in drawn
                if drawn_game_id == game_id
            )
        )
        if not frames:
            continue
        rendered = render_matrix_rows_under_clause(
            game_id,
            spec.grading,
            clause=clause,
            framing_label=framing_id,
            split=SPLIT_TRAIN,
            label_print_order=spec.label_print_order,
            scenarios=frames,
        )
        if clause is not None:
            _audit_one_insertion(
                rendered,
                render_matrix_rows_under_clause(
                    game_id,
                    spec.grading,
                    clause=None,
                    framing_label=FRAMING_UNSTATED,
                    split=SPLIT_TRAIN,
                    label_print_order=spec.label_print_order,
                    scenarios=frames,
                ),
            )
        rows.extend(
            row
            for row in rendered
            if (game_id, str(row["reskin_id"]), str(row["payoff_variant"])) in wanted
        )
    return rows


def _trust_candidate_rows(spec: BreadthSpec) -> tuple[list[Row], list[StratumPlan]]:
    """Render the trust sender's whole training roster at the announced rates the spec names."""
    if spec.trust_game_id is None:
        return [], []
    rendered = generate_prompt_rows(
        spec.trust_game_id,
        spec.grading,
        split=SPLIT_TRAIN,
        label_print_order=spec.label_print_order,
    )
    rows: list[Row] = []
    strata: list[StratumPlan] = []
    for payoff_variant in spec.trust_payoff_variants:
        kept = [row for row in rendered if row["payoff_variant"] == payoff_variant]
        if not kept:
            present = sorted({str(row["payoff_variant"]) for row in rendered})
            raise ValueError(
                f"{spec.trust_game_id!r} renders no rows at payoff variant {payoff_variant!r}; "
                f"variants present: {present}"
            )
        for row in kept:
            row[FRAMING_ID_COLUMN] = FRAMING_ID_UNSET
        rows.extend(kept)
        strata.append(
            StratumPlan(
                group=spec.trust_game_id,
                framing_id=FRAMING_ID_UNSET,
                band=TRUST_STRATUM_BAND,
                payoff_variant=payoff_variant,
                # Singletons, not pairs: the answer is an amount rather than one of two labels, so
                # there is no label orientation to counterbalance and the quota is the whole roster.
                quota_pairs=len(kept),
                candidate_pairs=len(kept),
                rule=RULE_STANDARD_BAND,
            )
        )
    return rows, strata


def build_candidates(
    spec: BreadthSpec, *, runtime_clauses: Mapping[str, str] | None = None
) -> BreadthCandidates:
    """Render the oversampled candidate pool for one breadth spec.

    The rows are the sweep's input, not the corpus: each stratum carries more pairs than it targets, by
    the factor its band's keep rate justifies, and `select_breadth` cuts them down against the sweep's
    verdicts. Frames are drawn by a hash of their own identity, so this is reproducible in the sense
    the sweep's pool hash needs: the same spec draws the same pool whatever ran before it.

    `runtime_clauses` supplies the counterpart paragraphs the tracked registry does not carry, which is
    every framing authored for a wave: the two kin clauses and the identity ladder's rungs. The
    registry answers first, so a loaded file can never shadow a registered framing.
    """
    matrix_rows, matrix_strata = _matrix_candidate_rows(spec, runtime_clauses=runtime_clauses)
    trust_rows, trust_strata = _trust_candidate_rows(spec)
    rows = [*matrix_rows, *trust_rows]
    if not rows:
        raise ValueError("the breadth spec rendered no candidate rows at all")
    prompt_ids = [str(row[PROMPT_ID_COLUMN]) for row in rows]
    duplicates = sorted({pid for pid in prompt_ids if prompt_ids.count(pid) > 1})
    if duplicates:
        raise ValueError(
            f"candidate pool holds duplicate prompt_ids {duplicates}; the sweep keys every verdict "
            f"and the reward every group on prompt_id, so a collision would pool two prompts' draws"
        )
    used_runtime = tuple(
        framing_id for framing_id in spec.framing_ids if framing_id not in COUNTERPART_FRAMINGS
    )
    logger.info(
        f"built breadth candidates, n_rows={len(rows)} n_matrix={len(matrix_rows)} "
        f"n_trust={len(trust_rows)} n_strata={len(matrix_strata) + len(trust_strata)} "
        f"grading={spec.grading!r} seed={spec.seed} runtime_framings={list(used_runtime)}"
    )
    return BreadthCandidates(
        rows=tuple(rows),
        strata=(*matrix_strata, *trust_strata),
        spec=spec,
        runtime_framing_ids=used_runtime,
        runtime_clauses_digest=_clauses_digest(used_runtime, runtime_clauses),
    )


def _clauses_digest(
    framing_ids: Sequence[str], runtime_clauses: Mapping[str, str] | None
) -> str | None:
    """Digest the runtime clauses this pool rendered, so the artifact pins them without quoting them.

    The clauses are authored stimulus and cannot land in an artifact this repository publishes, but a
    corpus built under an edited clause is a different corpus, so the digest is what a later reader
    compares. None when the grid used registered framings only.
    """
    if not framing_ids or runtime_clauses is None:
        return None
    digest = hashlib.sha256()
    for framing_id in sorted(framing_ids):
        digest.update(f"{framing_id}\n{runtime_clauses[framing_id]}\n".encode())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SelectionUnit:
    """One indivisible thing the selection keeps or drops: a counterbalanced pair, or a singleton.

    A pair is indivisible because keeping one label orientation and dropping the other walks the
    position bias the counterbalancing exists to cancel straight into the corpus. Every rule below
    therefore decides per unit, and the multiple-of-eight trim removes whole units too.
    """

    stratum: str
    band: str
    prompt_ids: tuple[str, ...]
    coop_count: int
    coop_total: int
    coop_fractions: tuple[float, ...]
    score_stds: tuple[float, ...]
    parseable_fractions: tuple[float, ...]
    draw_digest: str

    @property
    def n_rows(self) -> int:
        """Return how many corpus rows this unit contributes."""
        return len(self.prompt_ids)


@dataclass(frozen=True, slots=True)
class FilledUnit:
    """One pair a short stratum took under the fill rule, at the base behaviour it was taken at.

    Recorded per unit rather than as a count so a readout can separate fill rows from mixed rows in the
    corpus: the prompt ids are the join, since the corpus rows are copied verbatim and carry no
    selection column of their own. The counts are kept rather than the rate they imply, so a later
    reader re-derives the fraction from the draws it was measured on.
    """

    prompt_ids: tuple[str, ...]
    coop_count: int
    coop_total: int
    rule: str

    @property
    def n_rows(self) -> int:
        """Return how many corpus rows this filled unit contributes."""
        return len(self.prompt_ids)

    @property
    def coop_fraction(self) -> float:
        """Return the pair's pooled cooperation rate over both orientations' parsed draws."""
        return self.coop_count / self.coop_total

    def to_json_dict(self) -> dict[str, Any]:
        """Flatten for the stratum artifact, the rule named on the unit itself."""
        return {
            "prompt_ids": list(self.prompt_ids),
            "coop_count": self.coop_count,
            "coop_total": self.coop_total,
            "coop_fraction": round(self.coop_fraction, 6),
            "rule": self.rule,
        }


@dataclass(frozen=True, slots=True)
class StratumOutcome:
    """What one stratum asked for, what its sweep showed, and what reached the corpus."""

    plan: StratumPlan
    n_candidate_rows: int
    n_kept_units: int
    n_kept_rows: int
    dropped_by_reason: Mapping[str, int]
    base_coop_fractions: tuple[float, ...]
    base_score_stds: tuple[float, ...]
    filled_units: tuple[FilledUnit, ...] = ()
    fill_rule: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        """Flatten, with the base behaviour summarised beside the raw per-prompt values.

        The fill block appears only when a run asked for the fill, so the artifact of a run under the
        registered rules is byte-identical to the ones already banked. It is written for every stratum
        of such a run rather than only the filled ones, because "this stratum was not eligible" and
        "this stratum was eligible and found nothing to take" are different facts about the corpus.
        """
        payload = {
            **self.plan.to_json_dict(),
            "n_candidate_rows": self.n_candidate_rows,
            "n_kept_units": self.n_kept_units,
            "n_kept_rows": self.n_kept_rows,
            "dropped_by_reason": dict(sorted(self.dropped_by_reason.items())),
            "base_cooperation": _distribution(self.base_coop_fractions),
            "base_score_spread": _distribution(self.base_score_stds),
        }
        if self.fill_rule is not None:
            payload["fill"] = {
                "rule": self.fill_rule,
                "eligible": self.plan.band in FILL_BANDS,
                "n_filled_units": len(self.filled_units),
                "n_filled_rows": sum(unit.n_rows for unit in self.filled_units),
                "base_cooperation": _distribution(
                    [unit.coop_fraction for unit in self.filled_units]
                ),
                "units": [unit.to_json_dict() for unit in self.filled_units],
            }
        return payload


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    """Summarise a stratum's base behaviour, keeping the per-prompt values beside the aggregates.

    The values themselves are kept because the aggregate cannot answer the question the far rule
    raises: whether a stratum was kept on a handful of prompts each showing one cooperation, or on
    prompts genuinely near the middle. Cooperation rates over eight draws are aggregates of a base
    model's behaviour rather than item text, so they are safe to record.
    """
    if not values:
        return {"n": 0, "mean": None, "min": None, "max": None, "values": []}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "values": [round(value, 6) for value in values],
    }


@dataclass(frozen=True, slots=True)
class BreadthSelection:
    """The corpus a breadth arm trains on, plus the per-stratum record of how it was cut down."""

    rows: tuple[Row, ...]
    strata: tuple[StratumOutcome, ...]
    dropped_groups: Mapping[str, str]
    pool_hash: str | None
    spec: BreadthSpec
    fill_rule: str | None = None
    trust_min_split_std: float | None = None

    def to_artifact(self) -> dict[str, Any]:
        """Build the stratum artifact: the spec asked for, the composition realised, per stratum.

        `selection_options` appears only when a run reached for one of the optional rules, so an
        artifact written under the registered ones stays byte-identical to the banked wave-4b record.
        An option that ran, in exchange, is named in the file rather than only in the invocation, since
        the corpus is what a training run reads and the artifact is the only thing beside it.
        """
        payload = {
            "spec": self.spec.to_json_dict(),
            "pool_hash": self.pool_hash,
            "n_rows": len(self.rows),
            "step_prompt_multiple": self.spec.step_prompt_multiple,
            "strata": [outcome.to_json_dict() for outcome in self.strata],
            "dropped_groups": dict(sorted(self.dropped_groups.items())),
            "realised_composition": self.realised_composition(),
        }
        if self.fill_rule is not None or self.trust_min_split_std is not None:
            filled = [unit for outcome in self.strata for unit in outcome.filled_units]
            payload["selection_options"] = {
                "fill_rule": self.fill_rule,
                "fill_bands": list(FILL_BANDS) if self.fill_rule is not None else [],
                "n_filled_units": len(filled),
                "n_filled_rows": sum(unit.n_rows for unit in filled),
                "trust_min_split_std": self.trust_min_split_std,
            }
        return payload

    def realised_composition(self) -> dict[str, Any]:
        """Count the written rows per game, per framing and per band, which is the corpus's own weights.

        The plan's table is a target; this is what a reader of the run has to be able to compare it
        with. Computed off the rows rather than off the verdicts, so it describes the file that was
        written even if a later trim moved something.
        """
        band_by_framing = self.spec.band_by_framing
        by_game: dict[str, int] = {}
        by_framing: dict[str, int] = {}
        by_band: dict[str, int] = {}
        for row in self.rows:
            game_id = str(row["game_id"])
            framing_id = str(row[FRAMING_ID_COLUMN])
            by_game[game_id] = by_game.get(game_id, 0) + 1
            by_framing[framing_id] = by_framing.get(framing_id, 0) + 1
            band = band_by_framing.get(framing_id, TRUST_STRATUM_BAND)
            by_band[band] = by_band.get(band, 0) + 1
        return {
            "rows_by_game_id": dict(sorted(by_game.items())),
            "rows_by_framing_id": dict(sorted(by_framing.items())),
            "rows_by_band": dict(sorted(by_band.items())),
        }


def _sample_from_json(payload: Mapping[str, Any]) -> SampleOutcome:
    """Rebuild one sampled completion from a written sweep trace.

    The trace is the sweep's own `dataclasses.asdict`, so every field round-trips except the tuples,
    which JSON writes as lists. Rebuilt into the dataclass rather than read field by field, because
    then the aggregates the verdicts key on come from the same properties the sweep computed them with
    rather than from a second implementation here.
    """
    sequence = payload.get("action_sequence")
    levels = payload.get("levels")
    return SampleOutcome(
        completion=str(payload["completion"]),
        visible_text=str(payload["visible_text"]),
        truncated_thinking=bool(payload["truncated_thinking"]),
        parsed=bool(payload["parsed"]),
        selection_score=(
            None if payload["selection_score"] is None else float(payload["selection_score"])
        ),
        action=None if payload.get("action") is None else str(payload["action"]),
        action_sequence=None if sequence is None else tuple(str(move) for move in sequence),
        kept=None if payload.get("kept") is None else int(payload["kept"]),
        claim=None if payload.get("claim") is None else int(payload["claim"]),
        contribution=(
            None if payload.get("contribution") is None else int(payload["contribution"])
        ),
        sent=None if payload.get("sent") is None else int(payload["sent"]),
        return_percentage=(
            None if payload.get("return_percentage") is None else int(payload["return_percentage"])
        ),
        level=None if payload.get("level") is None else int(payload["level"]),
        levels=None if levels is None else tuple(int(value) for value in levels),
    )


def records_from_trace(trace: Sequence[Mapping[str, Any]]) -> list[PromptSweepRecord]:
    """Rebuild the policy sweep's records from a written trace, skipping the meta and opponent lines.

    A frozen-opponent record describes a different model on the same prompts, so folding it in would
    judge this corpus on the opponent's behaviour; the meta line carries no samples at all.
    """
    records = [
        PromptSweepRecord(
            prompt_id=str(entry[PROMPT_ID_COLUMN]),
            grading=str(entry["grading"]),
            row=dict(entry["row"]),
            samples=tuple(_sample_from_json(dict(sample)) for sample in list(entry["samples"])),
        )
        for entry in trace
        if entry.get("record_kind") == SWEEP_RECORD_KIND
    ]
    if not records:
        raise ValueError(
            f"the sweep trace holds no {SWEEP_RECORD_KIND!r} records, so there is nothing to select "
            f"from. A trace carrying only its meta line is a sweep that wrote provenance and failed."
        )
    return records


def pool_hash_from_trace(trace: Sequence[Mapping[str, Any]]) -> str | None:
    """Read the sweep's pool hash off its meta line, which is what dates a verdict to a pool."""
    for entry in trace:
        if entry.get("record_kind") == META_RECORD_KIND:
            value = entry.get("prompt_id_order_sha256")
            return None if value is None else str(value)
    return None


def _assert_trace_matches_candidates(
    records: Sequence[PromptSweepRecord], candidates: Sequence[Row]
) -> None:
    """Refuse a trace and a candidate pool that are not the same rows, column for column.

    Two failures this catches, both silent otherwise. A candidate with no record would shrink the
    corpus with nothing saying why, and a record for a prompt the pool no longer holds means the pool
    was rebuilt between the sweep and the selection -- at which point the verdicts describe prompts
    the corpus does not contain. Every column is compared rather than the id or the rendered text
    alone: a frame edited under the same id is what an id check cannot see, and a rebuild that moved
    the grading, the framing, the game or a payoff cell while leaving the prompt byte-identical is
    what a text check cannot see, though those are exactly the columns that decide the stratum, the
    selection rule and the reward branch.
    """
    swept = {record.prompt_id: record for record in records}
    pool = {str(row[PROMPT_ID_COLUMN]): row for row in candidates}
    unswept = sorted(set(pool) - set(swept))
    unknown = sorted(set(swept) - set(pool))
    if unswept or unknown:
        raise ValueError(
            f"the sweep trace and the candidate pool are not the same prompts: "
            f"{len(unswept)} candidates were never swept ({unswept[:5]}) and {len(unknown)} swept "
            f"prompts are not in the pool ({unknown[:5]}). Select against the trace of THIS pool; a "
            f"rebuilt pool needs its own sweep."
        )
    differing = {
        prompt_id: sorted(
            column
            for column in set(row) | set(swept[prompt_id].row)
            if row.get(column) != swept[prompt_id].row.get(column)
        )
        for prompt_id, row in pool.items()
    }
    changed = sorted(prompt_id for prompt_id, columns in differing.items() if columns)
    if changed:
        columns = sorted({column for prompt_id in changed for column in differing[prompt_id]})
        raise ValueError(
            f"{len(changed)} candidate rows differ from the ones the sweep measured "
            f"({changed[:5]}), in columns {columns}, so their verdicts describe prompts this corpus "
            f"does not hold. Re-sweep after editing a frame, a clause, a payoff or the render."
        )


def _stratum_of(row: Mapping[str, Any], spec: BreadthSpec) -> StratumPlan:
    """Return which stratum plan one swept row belongs to, refusing a row the spec does not cover."""
    game_id = str(row["game_id"])
    framing_id = str(row.get(FRAMING_ID_COLUMN, FRAMING_ID_UNSET))
    payoff_variant = str(row["payoff_variant"])
    if framing_id == FRAMING_ID_UNSET:
        if spec.trust_game_id is None or game_id != spec.trust_game_id:
            raise ValueError(
                f"row of game {game_id!r} carries no counterpart framing, but the spec's unframed "
                f"game is {spec.trust_game_id!r}; every matrix row of a breadth corpus is stamped "
                f"with the framing it was rendered under."
            )
        return StratumPlan(
            group=game_id,
            framing_id=FRAMING_ID_UNSET,
            band=TRUST_STRATUM_BAND,
            payoff_variant=payoff_variant,
            quota_pairs=0,
            candidate_pairs=0,
            rule=RULE_STANDARD_BAND,
        )
    group = spec.group_by_game_id.get(game_id)
    band = spec.band_by_framing.get(framing_id)
    if group is None or band is None:
        raise ValueError(
            f"row of game {game_id!r} under framing {framing_id!r} belongs to no stratum of this "
            f"spec (games {sorted(spec.group_by_game_id)}, framings {sorted(spec.band_by_framing)}). "
            f"Select against the spec the pool was built from."
        )
    game = spec.game(group)
    return StratumPlan(
        group=group,
        framing_id=framing_id,
        band=band,
        payoff_variant=payoff_variant,
        quota_pairs=game.pairs_per_variant_by_band[band],
        candidate_pairs=0,
        rule=RULE_FAR_PAIR_ANY_COOPERATION if band == BAND_FAR else RULE_STANDARD_BAND,
    )


def _plans_by_prompt(
    records: Sequence[PromptSweepRecord], spec: BreadthSpec
) -> dict[str, StratumPlan]:
    """Map every swept prompt to its stratum plan, the trust strata's target filled from the roster.

    A trust stratum targets its whole training roster rather than a number the spec states, so no
    single row carries the target and `_stratum_of` cannot know it. The pool answers instead, on the
    same argument `_stratum_outcomes` makes for `candidate_pairs`: every candidate was swept
    (`_assert_trace_matches_candidates`), so the trust rows here ARE the roster the build rendered.
    Left at zero, the stratum artifact records a stratum that asked for nothing and delivered sixteen,
    which cannot be compared against the build manifest's own target and hides a roster that came in
    short.
    """
    plans = {record.prompt_id: _stratum_of(record.row, spec) for record in records}
    roster: dict[str, int] = {}
    for plan in plans.values():
        if plan.band == TRUST_STRATUM_BAND:
            roster[plan.key] = roster.get(plan.key, 0) + 1
    return {
        prompt_id: (
            dataclasses.replace(plan, quota_pairs=roster[plan.key])
            if plan.band == TRUST_STRATUM_BAND
            else plan
        )
        for prompt_id, plan in plans.items()
    }


def _assert_the_rows_carry_the_specs_grading(candidates: Sequence[Row], spec: BreadthSpec) -> None:
    """Refuse a pool whose rows disagree with the grading the stratum artifact would record.

    Selection copies rows verbatim, so `spec.grading` reaches the artifact while the rows keep whatever
    they were rendered under. `--grading` is shared by both subcommands and on `build` it IS the
    render's input, so reaching for it on `select` is the ordinary mistake and it banks an artifact
    pair that contradicts itself: a corpus one arm's trainer refuses and the other arm's trainer
    accepts while every record beside it names the wrong reward.
    """
    carried = sorted({str(row[GRADING_COLUMN]) for row in candidates})
    if carried != [spec.grading]:
        raise ValueError(
            f"the candidate rows are graded {carried} but this spec's grading is {spec.grading!r}. "
            f"Selection copies rows verbatim, so the stratum artifact would name a reward the corpus "
            f"does not carry; select under the grading the pool was built with, and rewrite a "
            f"corpus's grading with `python -m games.regrade_corpus` afterwards."
        )


@dataclass(frozen=True, slots=True)
class _JudgedUnit:
    """One unit with its verdict, before the quota and step-multiple trims run."""

    unit: SelectionUnit
    keep: bool
    reason: str
    filled: bool = False


def _log_trust_spread_ladder(records: Sequence[PromptSweepRecord], *, floor: float) -> None:
    """Log what every floor value would admit among the trust rows, not only the one in use.

    A floor admits exactly the rows whose send spread reaches it, so every value between two observed
    spreads is the same choice and the observed spreads are the whole answer. Logged because the
    alternative is re-reading a 180 MB trace to learn whether lowering the floor would admit anything:
    in the 9B base sweep it would not until the floor reached zero, since 29 of the 32 trust rows sent
    the same amount in all eight draws and the other three sat at a third of the endowment and above.
    """
    spreads = sorted(record.score_std for record in records)
    admitted_by_floor = {
        f"{value:g}": sum(1 for spread in spreads if spread >= value)
        for value in sorted({0.0, *spreads})
    }
    logger.info(
        f"trust send spread: n_rows={len(spreads)} floor_in_use={floor:g} "
        f"admitted_at_floor={sum(1 for spread in spreads if spread >= floor)} "
        f"admitted_by_floor={admitted_by_floor}"
    )


def _standard_verdicts(  # noqa: PLR0913 - one keyword per selection threshold
    standard: Sequence[PromptSweepRecord],
    plan_by_prompt: Mapping[str, StratumPlan],
    *,
    min_coop: float,
    max_coop: float,
    min_split_std: float,
    trust_min_split_std: float | None,
    min_parseable_fraction: float,
) -> dict[str, tuple[bool, str]]:
    """Judge the near, mid and trust records by the standard band, the trust rows on their own floor.

    Two calls rather than one, because the send-spread floor is settable for the trust rows alone: they
    are the only records it decides at all (a binary action is judged on its cooperation rate), and the
    9B base sweep left 29 of its 32 trust rows at exactly zero spread, so no floor short of zero admits
    anything the shared 0.05 did not. Splitting the call cannot move a matrix verdict: `judge_prompts`
    couples counterbalanced pairs, and the trust rows carry empty labels and are singletons by
    construction, so no pair spans the two calls.
    """
    trust = [
        record for record in standard if plan_by_prompt[record.prompt_id].band == TRUST_STRATUM_BAND
    ]
    matrix = [
        record for record in standard if plan_by_prompt[record.prompt_id].band != TRUST_STRATUM_BAND
    ]
    trust_floor = min_split_std if trust_min_split_std is None else trust_min_split_std
    if trust:
        _log_trust_spread_ladder(trust, floor=trust_floor)
    verdicts: dict[str, tuple[bool, str]] = {}
    for group, floor in ((matrix, min_split_std), (trust, trust_floor)):
        if not group:
            continue
        for record, verdict in zip(
            group,
            judge_prompts(
                group,
                min_coop=min_coop,
                max_coop=max_coop,
                min_split_std=floor,
                min_parseable_fraction=min_parseable_fraction,
            ),
            strict=True,
        ):
            verdicts[record.prompt_id] = (verdict.keep, str(verdict.reason))
    return verdicts


def _judge_units(  # noqa: PLR0913 - one keyword per selection threshold
    records: Sequence[PromptSweepRecord],
    spec: BreadthSpec,
    plan_by_prompt: Mapping[str, StratumPlan],
    *,
    min_coop: float,
    max_coop: float,
    min_split_std: float,
    trust_min_split_std: float | None,
    min_parseable_fraction: float,
) -> list[_JudgedUnit]:
    """Judge every counterbalanced pair and singleton, by the rule its stratum's band takes.

    The near, mid and trust strata go through `judge_prompts` unchanged, which is the standard band
    plus the pair coupling every banked corpus was selected by. The far strata take the pair rule, and
    their parse floor still comes from `judge_prompt`, because a pair whose draws mostly failed to
    parse is a measurement of the format rather than of the policy however many cooperations it holds.
    """
    standard = [record for record in records if plan_by_prompt[record.prompt_id].band != BAND_FAR]
    far = [record for record in records if plan_by_prompt[record.prompt_id].band == BAND_FAR]
    verdicts = _standard_verdicts(
        standard,
        plan_by_prompt,
        min_coop=min_coop,
        max_coop=max_coop,
        min_split_std=min_split_std,
        trust_min_split_std=trust_min_split_std,
        min_parseable_fraction=min_parseable_fraction,
    )

    judged: list[_JudgedUnit] = []
    for group_records in _units(standard):
        keep = all(verdicts[record.prompt_id][0] for record in group_records)
        judged.append(
            _JudgedUnit(
                unit=_unit_of(group_records, plan_by_prompt, spec),
                keep=keep,
                reason=_pair_drop_reason(group_records, verdicts),
            )
        )
    for group_records in _units(far):
        unit = _unit_of(group_records, plan_by_prompt, spec)
        parse_floor_failed = [
            record
            for record in group_records
            if judge_prompt(
                record,
                min_coop=min_coop,
                max_coop=max_coop,
                min_split_std=min_split_std,
                min_parseable_fraction=min_parseable_fraction,
            ).reason
            == DropReason.TOO_FEW_PARSEABLE
        ]
        if parse_floor_failed:
            judged.append(
                _JudgedUnit(
                    unit=unit,
                    keep=False,
                    reason=str(BreadthDropReason.FAR_PAIR_TOO_FEW_PARSEABLE),
                )
            )
            continue
        keep = unit.coop_count >= 1
        judged.append(
            _JudgedUnit(
                unit=unit,
                keep=keep,
                reason=(
                    str(DropReason.KEPT_MIXED)
                    if keep
                    else str(BreadthDropReason.FAR_PAIR_NO_COOPERATION)
                ),
            )
        )
    return judged


def _pair_drop_reason(
    group_records: Sequence[PromptSweepRecord], verdicts: Mapping[str, tuple[bool, str]]
) -> str:
    """Name why a coupled unit is not usable, preferring the base behaviour over the coupling.

    `judge_prompts` rewrites the acceptable orientation's verdict to `counterbalanced-partner-dropped`,
    so an asymmetric pair carries two reasons and taking the first in trace order records the coupling
    about half the time -- and the builder always emits `coop0` first, so which half depends on nothing
    the artifact's reader can see. The coupling reason answers no question the artifact is read for
    ("how did the base behave under this framing", "why did this stratum come in short"), so it is
    recorded only when every orientation carries it. `judge_prompts` rewrites a verdict only when some
    orientation of the pair failed on its own, so that case cannot arise today and the fallback exists
    to keep a reason rather than to be reached.
    """
    failed = [
        verdicts[record.prompt_id][1]
        for record in group_records
        if not verdicts[record.prompt_id][0]
    ]
    if not failed:
        return str(DropReason.KEPT_MIXED)
    own = [reason for reason in failed if reason != str(DropReason.PARTNER_DROPPED)]
    return own[0] if own else failed[0]


def _units(records: Sequence[PromptSweepRecord]) -> list[tuple[PromptSweepRecord, ...]]:
    """Group records into the units selection decides on, in first-appearance order.

    `pair_identity` is the sweep's own key, so a unit here is exactly the unit `judge_prompts` couples,
    and the framing column being part of that key is what keeps one frame's renderings under two
    framings from landing in one unit (checked at import).
    """
    groups: dict[tuple[object, ...], list[PromptSweepRecord]] = {}
    for record in records:
        groups.setdefault(pair_identity(record.row), []).append(record)
    oversized = {
        identity: [record.prompt_id for record in group]
        for identity, group in groups.items()
        if len(group) > LABEL_ORIENTATIONS_PER_SCENARIO
    }
    if oversized:
        raise ValueError(
            f"{len(oversized)} counterbalanced-pair keys hold more than "
            f"{LABEL_ORIENTATIONS_PER_SCENARIO} rows: {list(oversized.values())[:3]}. A scenario has "
            f"two label orientations, so a larger group means the pool repeats a rendering or two "
            f"framings share a pair key."
        )
    orphans = sorted(
        group[0].prompt_id
        for group in groups.values()
        if len(group) == 1 and is_counterbalanced(group[0].row)
    )
    if orphans:
        raise ValueError(
            f"{len(orphans)} prompts have no counterbalanced partner in this pool: {orphans[:5]}. "
            f"Both label orientations of a scenario are judged and kept together precisely so that a "
            f"preference for the first-listed option cannot masquerade as a preference for "
            f"cooperating, and a lone orientation keeps its own verdict and walks that position bias "
            f"into the corpus. `judge_prompts` raises on this for the standard band, which the far "
            f"strata never reach, so the guard lives here and covers both. The trust sender's rows "
            f"carry empty labels and are singletons by construction, so they pass."
        )
    return [tuple(group) for group in groups.values()]


def _unit_of(
    records: Sequence[PromptSweepRecord],
    plan_by_prompt: Mapping[str, StratumPlan],
    spec: BreadthSpec,
) -> SelectionUnit:
    """Summarise one unit's base behaviour, and derive the digest its trim order is keyed on."""
    plan = plan_by_prompt[records[0].prompt_id]
    strata = {plan_by_prompt[record.prompt_id].key for record in records}
    if len(strata) != 1:
        raise ValueError(
            f"a counterbalanced unit spans strata {sorted(strata)}; the pair key carries the game, "
            f"the framing and the payoff variant, so this cannot happen without one of them moving "
            f"between the two orientations"
        )
    coop_counts = [record.coop_count for record in records if record.coop_count is not None]
    coop_totals = [record.coop_total for record in records if record.coop_total is not None]
    prompt_ids = tuple(record.prompt_id for record in records)
    return SelectionUnit(
        stratum=plan.key,
        band=plan.band,
        prompt_ids=prompt_ids,
        coop_count=sum(coop_counts),
        coop_total=sum(coop_totals),
        coop_fractions=tuple(
            record.coop_fraction for record in records if record.coop_fraction is not None
        ),
        score_stds=tuple(record.score_std for record in records),
        parseable_fractions=tuple(record.parseable_fraction for record in records),
        draw_digest=_draw_digest(spec.seed, "trim", *sorted(prompt_ids)),
    )


def select_breadth(  # noqa: PLR0913 - one keyword per selection threshold
    sweep_trace: Sequence[Mapping[str, Any]],
    candidates: Sequence[Row],
    spec: BreadthSpec,
    *,
    min_coop: float = DEFAULT_MIN_COOP,
    max_coop: float = DEFAULT_MAX_COOP,
    min_split_std: float = DEFAULT_MIN_SPLIT_STD,
    min_parseable_fraction: float = DEFAULT_MIN_PARSEABLE_FRACTION,
    fill_pure_to_quota: bool = False,
    trust_min_split_std: float | None = None,
) -> BreadthSelection:
    """Cut the candidate pool down to the corpus a breadth arm trains on, and record every decision.

    Two refusals first, because both would otherwise bank a self-contradictory artifact pair: the trace
    has to describe these exact rows, and the rows have to carry the grading the artifact will record.
    Then five steps in order, and the order matters: judge each unit by its band's rule, cut each
    stratum to its quota, top a short mid or far stratum up if the caller asked for the fill, drop a
    game whose strata kept too little to read anything off, then trim whole units until the corpus
    fills a whole number of training steps. Everything dropped is counted by reason per stratum, so the
    artifact says what the sweep found rather than only what survived.

    The two optional rules are off by default and neither changes a corpus selected without them.
    `fill_pure_to_quota` is `RULE_FILL_PURE_TO_QUOTA` above, and it runs before the game-level drop on
    purpose: a game whose bands kept almost nothing is exactly the case the fill exists for, and running
    after would top up the strata of a game already removed from the corpus. `trust_min_split_std`
    replaces `min_split_std` for the trust rows alone, where zero keeps every row that parsed.
    """
    if trust_min_split_std is not None:
        if trust_min_split_std < 0:
            raise ValueError(
                f"trust_min_split_std must not be negative, got {trust_min_split_std}; zero already "
                f"admits every trust row that cleared the parse floor"
            )
        if spec.trust_game_id is None:
            raise ValueError(
                f"trust_min_split_std={trust_min_split_std} was passed, but this spec carries no trust "
                f"rows to judge with it (trust_game_id is None). The floor decides the trust sender's "
                f"send spread and nothing else, so a matrix-only corpus would be selected exactly as "
                f"it is without it."
            )
    records = records_from_trace(sweep_trace)
    _assert_trace_matches_candidates(records, candidates)
    _assert_the_rows_carry_the_specs_grading(candidates, spec)
    plan_by_prompt = _plans_by_prompt(records, spec)
    judged = _judge_units(
        records,
        spec,
        plan_by_prompt,
        min_coop=min_coop,
        max_coop=max_coop,
        min_split_std=min_split_std,
        trust_min_split_std=trust_min_split_std,
        min_parseable_fraction=min_parseable_fraction,
    )
    dropped: dict[str, str] = {
        prompt_id: entry.reason
        for entry in judged
        if not entry.keep
        for prompt_id in entry.unit.prompt_ids
    }
    dropped.update(_trim_to_quota(judged, plan_by_prompt, dropped))
    if fill_pure_to_quota:
        judged = _fill_short_strata(
            judged, plan_by_prompt, dropped, min_parseable_fraction=min_parseable_fraction
        )
    dropped_groups = _drop_thin_games(judged, plan_by_prompt, dropped, spec)
    dropped.update(_trim_to_step_multiple(judged, dropped, spec))

    kept_ids = {
        prompt_id
        for entry in judged
        for prompt_id in entry.unit.prompt_ids
        if prompt_id not in dropped
    }
    rows = tuple(row for row in candidates if str(row[PROMPT_ID_COLUMN]) in kept_ids)
    if len(rows) % spec.step_prompt_multiple:
        raise RuntimeError(
            f"the selected corpus holds {len(rows)} rows, not a multiple of "
            f"{spec.step_prompt_multiple}; the trim that fills a whole number of steps did not."
        )
    fill_rule = RULE_FILL_PURE_TO_QUOTA if fill_pure_to_quota else None
    outcomes = _stratum_outcomes(
        judged, plan_by_prompt, dropped, candidates, spec, fill_rule=fill_rule
    )
    logger.info(
        f"selected breadth corpus, n_rows={len(rows)} n_candidates={len(candidates)} "
        f"n_strata={len(outcomes)} dropped_groups={sorted(dropped_groups)} "
        f"fill_rule={fill_rule} trust_min_split_std={trust_min_split_std}"
    )
    return BreadthSelection(
        rows=rows,
        strata=outcomes,
        dropped_groups=dropped_groups,
        pool_hash=pool_hash_from_trace(sweep_trace),
        spec=spec,
        fill_rule=fill_rule,
        trust_min_split_std=trust_min_split_std,
    )


def _kept_units(judged: Sequence[_JudgedUnit], dropped: Mapping[str, str]) -> list[_JudgedUnit]:
    """Return the units still standing: kept by their rule and not dropped by a later step."""
    return [
        entry
        for entry in judged
        if entry.keep and not any(prompt_id in dropped for prompt_id in entry.unit.prompt_ids)
    ]


def _trim_to_quota(
    judged: Sequence[_JudgedUnit],
    plan_by_prompt: Mapping[str, StratumPlan],
    dropped: Mapping[str, str],
) -> dict[str, str]:
    """Cut every stratum that kept more units than it targets down to its quota.

    The units that go are the ones whose identity digest sorts last, which makes the trim reproducible
    and independent of execution order. The trust stratum has no pair quota (it contributes every
    training skin), so it is never trimmed here.
    """
    by_stratum: dict[str, list[_JudgedUnit]] = {}
    for entry in _kept_units(judged, dropped):
        by_stratum.setdefault(entry.unit.stratum, []).append(entry)
    quota_by_stratum = {
        plan.key: plan.quota_pairs
        for plan in plan_by_prompt.values()
        if plan.band != TRUST_STRATUM_BAND
    }
    trimmed: dict[str, str] = {}
    for stratum, entries in by_stratum.items():
        quota = quota_by_stratum.get(stratum)
        if quota is None or len(entries) <= quota:
            continue
        for entry in sorted(entries, key=lambda item: item.unit.draw_digest)[quota:]:
            for prompt_id in entry.unit.prompt_ids:
                trimmed[prompt_id] = str(BreadthDropReason.OVER_QUOTA)
    return trimmed


def _fill_order(entry: _JudgedUnit) -> tuple[float, str]:
    """Order one stratum's fill candidates: the pair with the most mixed orientation first.

    The key is the distance from mixed of the unit's CLOSEST orientation rather than the pair's pooled
    rate, because dynamic sampling drops a group per prompt: a pair whose two orientations are
    unanimous at opposite ends pools to exactly one half and would sort first on the average, while
    neither of its orientations can ever produce a group that disagrees. A fully pure pair sits at the
    maximum distance, so a pair with any spread at all outranks every pure one, and the identity digest
    decides among the pure ones -- which is the whole of the order in most mid strata of the 9B sweep,
    and is what keeps the choice independent of the order anything ran in.
    """
    return (
        min(abs(fraction - MIXED_COOP_FRACTION) for fraction in entry.unit.coop_fractions),
        entry.unit.draw_digest,
    )


def _measures_base_behaviour(entry: _JudgedUnit, min_parseable_fraction: float) -> bool:
    """Say whether a dropped unit measured the policy well enough for the fill to take it.

    The parse floor is the one condition the fill keeps, on the same argument the far rule keeps it: a
    pair whose draws mostly failed to parse measures the answer format rather than the policy, so it is
    a prompt to reword rather than one to train on. A unit missing a cooperation rate for either
    orientation is that same case reached from the other side, and the two are checked separately
    because only the first of them has a threshold.
    """
    return (
        len(entry.unit.coop_fractions) == entry.unit.n_rows
        and entry.unit.coop_total > 0
        and all(fraction >= min_parseable_fraction for fraction in entry.unit.parseable_fractions)
    )


def _fill_short_strata(
    judged: Sequence[_JudgedUnit],
    plan_by_prompt: Mapping[str, StratumPlan],
    dropped: dict[str, str],
    *,
    min_parseable_fraction: float,
) -> list[_JudgedUnit]:
    """Top every short mid and far stratum up to its quota, and return the units re-judged.

    Optional and off by default; the argument for it is at `RULE_FILL_PURE_TO_QUOTA`. A promoted unit
    loses its band rule's drop reason in both places that record one, the verdict list and `dropped`,
    so every later step -- the game-level drop, the step-multiple trim, the per-stratum counts --
    reads it as kept and names the fill as what kept it.

    Only mid and far strata are eligible (`FILL_BANDS`), only up to the quota the spec already states,
    and only among units whose draws parsed. Nothing here reaches for a candidate outside the stratum
    that is short: a stratum stays short when its own pool is exhausted, which is the same recorded
    fact it would have been without the fill.
    """
    quota_by_stratum = {
        plan.key: plan.quota_pairs for plan in plan_by_prompt.values() if plan.band in FILL_BANDS
    }
    kept_by_stratum: dict[str, int] = {}
    for entry in _kept_units(judged, dropped):
        kept_by_stratum[entry.unit.stratum] = kept_by_stratum.get(entry.unit.stratum, 0) + 1
    available: dict[str, list[_JudgedUnit]] = {}
    for entry in judged:
        if entry.keep or entry.unit.stratum not in quota_by_stratum:
            continue
        if _measures_base_behaviour(entry, min_parseable_fraction):
            available.setdefault(entry.unit.stratum, []).append(entry)
    promoted: dict[tuple[str, ...], _JudgedUnit] = {}
    filled_by_stratum: dict[str, int] = {}
    for stratum in sorted(available):
        short = quota_by_stratum[stratum] - kept_by_stratum.get(stratum, 0)
        if short <= 0:
            continue
        taken = sorted(available[stratum], key=_fill_order)[:short]
        filled_by_stratum[stratum] = len(taken)
        for entry in taken:
            promoted[entry.unit.prompt_ids] = entry
    for prompt_ids in promoted:
        for prompt_id in prompt_ids:
            del dropped[prompt_id]
    logger.info(
        f"{RULE_FILL_PURE_TO_QUOTA}: n_units={len(promoted)} "
        f"n_rows={sum(entry.unit.n_rows for entry in promoted.values())} "
        f"by_stratum={dict(sorted(filled_by_stratum.items()))}"
    )
    return [
        dataclasses.replace(entry, keep=True, reason=RULE_FILL_PURE_TO_QUOTA, filled=True)
        if entry.unit.prompt_ids in promoted
        else entry
        for entry in judged
    ]


def _pairs_by_group(
    judged: Sequence[_JudgedUnit],
    plan_by_prompt: Mapping[str, StratumPlan],
    dropped: Mapping[str, str],
) -> tuple[dict[str, int], dict[str, int]]:
    """Total the pairs each game group targeted and still holds, the trust singletons excluded.

    The trust rows are excluded from both totals because they are singletons whose quota is the whole
    roster: folding them into a group's pair count would compare a send-spread verdict against a pair
    target and could drop the giving game for arithmetic reasons.
    """
    group_by_stratum = {plan.key: plan.group for plan in plan_by_prompt.values()}
    quota_by_group: dict[str, int] = {}
    for plan in {plan.key: plan for plan in plan_by_prompt.values()}.values():
        if plan.band == TRUST_STRATUM_BAND:
            continue
        quota_by_group[plan.group] = quota_by_group.get(plan.group, 0) + plan.quota_pairs
    kept_by_group: dict[str, int] = dict.fromkeys(quota_by_group, 0)
    for entry in _kept_units(judged, dropped):
        group = group_by_stratum[entry.unit.stratum]
        if group in kept_by_group:
            kept_by_group[group] += 1
    return quota_by_group, kept_by_group


def _drop_thin_games(
    judged: Sequence[_JudgedUnit],
    plan_by_prompt: Mapping[str, StratumPlan],
    dropped: dict[str, str],
    spec: BreadthSpec,
) -> dict[str, str]:
    """Drop whole any game group that kept too small a share of its pair quota, and say so.

    The hi-lo precedent: a game whose base behaviour leaves almost nothing selectable is a capability
    datum rather than a plumbing failure, and carrying two of its pairs would put a game in the corpus
    that no per-game reading could rest on while its quota went unused. Per group rather than per
    stratum, because a group's own transfer reading pools its strata.
    """
    group_by_stratum = {plan.key: plan.group for plan in plan_by_prompt.values()}
    quota_by_group, kept_by_group = _pairs_by_group(judged, plan_by_prompt, dropped)
    reasons: dict[str, str] = {}
    for group, quota in quota_by_group.items():
        kept = kept_by_group[group]
        if quota == 0 or kept >= math.ceil(quota * spec.min_kept_fraction_per_game):
            continue
        reasons[group] = (
            f"kept {kept} of {quota} targeted pairs, below the "
            f"{spec.min_kept_fraction_per_game:g} share this spec requires of a game"
        )
        for entry in _kept_units(judged, dropped):
            if group_by_stratum[entry.unit.stratum] != group:
                continue
            for prompt_id in entry.unit.prompt_ids:
                dropped[prompt_id] = str(BreadthDropReason.GAME_KEPT_TOO_FEW_PAIRS)
    if reasons:
        logger.warning(f"breadth corpus drops game groups: {reasons}")
    return reasons


def _trim_order(kept: Sequence[_JudgedUnit]) -> list[_JudgedUnit]:
    """Order the kept units for the step-multiple trim: largest strata first, then by identity digest.

    Largest first so the composition moves least in relative terms, and by digest within a stratum so
    which unit goes does not depend on the order anything ran in.
    """
    size_by_stratum: dict[str, int] = {}
    for entry in kept:
        size_by_stratum[entry.unit.stratum] = size_by_stratum.get(entry.unit.stratum, 0) + 1
    return sorted(
        kept, key=lambda item: (-size_by_stratum[item.unit.stratum], item.unit.draw_digest)
    )


def _trim_to_step_multiple(
    judged: Sequence[_JudgedUnit], dropped: Mapping[str, str], spec: BreadthSpec
) -> dict[str, str]:
    """Drop whole units until the corpus fills a whole number of training steps.

    TRL's sampler drops the incomplete chunk at the end of every epoch, so a corpus whose size is not a
    multiple of the step's prompt count silently visits some prompt class less often than another for
    the whole run. Units go from the largest strata first, so the composition moves least in relative
    terms, and by identity digest within a stratum so the choice is reproducible. A singleton is
    removed where an odd number of rows has to go, which the trust rows always make possible: the row
    count is odd only when the number of singletons is.
    """
    kept = _kept_units(judged, dropped)
    excess = sum(entry.unit.n_rows for entry in kept) % spec.step_prompt_multiple
    if not excess:
        return {}
    order = _trim_order(kept)
    trimmed: dict[str, str] = {}
    remaining = excess

    def take(entry: _JudgedUnit) -> None:
        for prompt_id in entry.unit.prompt_ids:
            trimmed[prompt_id] = str(BreadthDropReason.TRIMMED_TO_STEP_MULTIPLE)

    # Exactly one singleton for an odd remainder, then pairs, so a corpus loses as few whole scenarios
    # as the arithmetic allows rather than several trust rows where one pair would have done.
    if remaining % LABEL_ORIENTATIONS_PER_SCENARIO:
        singleton = next((entry for entry in order if entry.unit.n_rows == 1), None)
        if singleton is None:
            raise RuntimeError(
                f"{remaining} rows have to go to fill whole steps of {spec.step_prompt_multiple}, an "
                f"odd number, and every selected unit is a counterbalanced pair. A pair cannot be "
                f"half dropped, so this corpus cannot be trimmed without breaking counterbalancing."
            )
        take(singleton)
        remaining -= 1
    for entry in order:
        if remaining < LABEL_ORIENTATIONS_PER_SCENARIO:
            break
        if entry.unit.n_rows != LABEL_ORIENTATIONS_PER_SCENARIO:
            continue
        take(entry)
        remaining -= LABEL_ORIENTATIONS_PER_SCENARIO
    if remaining:
        raise RuntimeError(
            f"could not trim {excess} rows down to a multiple of {spec.step_prompt_multiple}: "
            f"{remaining} still to go with no unit of the right size left. The selected units are "
            f"{sorted({entry.unit.n_rows for entry in kept})} rows each."
        )
    logger.info(f"trimmed {excess} rows to fill whole steps of {spec.step_prompt_multiple}")
    return trimmed


def _stratum_outcomes(  # noqa: PLR0913 - the fill rule joins the five inputs a record is built from
    judged: Sequence[_JudgedUnit],
    plan_by_prompt: Mapping[str, StratumPlan],
    dropped: Mapping[str, str],
    candidates: Sequence[Row],
    spec: BreadthSpec,
    *,
    fill_rule: str | None = None,
) -> tuple[StratumOutcome, ...]:
    """Build one outcome record per stratum, in the order the spec's grid names them.

    `fill_rule` is None for a selection under the registered rules, which is what keeps the fill block
    out of an artifact that never ran one.
    """
    plans = {plan.key: plan for plan in plan_by_prompt.values()}
    candidate_rows_by_stratum: dict[str, int] = {}
    for row in candidates:
        key = _stratum_of(row, spec).key
        candidate_rows_by_stratum[key] = candidate_rows_by_stratum.get(key, 0) + 1
    units_by_stratum: dict[str, list[_JudgedUnit]] = {}
    for entry in judged:
        units_by_stratum.setdefault(entry.unit.stratum, []).append(entry)
    outcomes: list[StratumOutcome] = []
    for key, plan in sorted(plans.items()):
        entries = units_by_stratum.get(key, [])
        # The plan rebuilt from a swept row cannot know the pool's planned draw, and does not have
        # to: every candidate was swept (`_assert_trace_matches_candidates`), so the units observed
        # here ARE the candidates drawn.
        plan = dataclasses.replace(plan, candidate_pairs=len(entries))  # noqa: PLW2901
        kept: list[_JudgedUnit] = []
        reasons: dict[str, int] = {}
        for entry in entries:
            # A later step's reason wins over the rule's own verdict, so a pair the band kept and the
            # quota then cut is counted as over-quota rather than twice as kept.
            first_dropped = next(
                (dropped[pid] for pid in entry.unit.prompt_ids if pid in dropped), None
            )
            if entry.keep and first_dropped is None:
                kept.append(entry)
                continue
            reason = first_dropped if first_dropped is not None else entry.reason
            reasons[reason] = reasons.get(reason, 0) + 1
        outcomes.append(
            StratumOutcome(
                plan=plan,
                n_candidate_rows=candidate_rows_by_stratum.get(key, 0),
                n_kept_units=len(kept),
                n_kept_rows=sum(entry.unit.n_rows for entry in kept),
                dropped_by_reason=reasons,
                base_coop_fractions=tuple(
                    fraction for entry in entries for fraction in entry.unit.coop_fractions
                ),
                base_score_stds=tuple(std for entry in entries for std in entry.unit.score_stds),
                # In the order the fill itself took them, with each pair's ids sorted, so the record is
                # derived from content alone: the trace's order is the pool's emission order and
                # carries no meaning, which is the same reason `draw_digest` sorts before hashing.
                filled_units=tuple(
                    FilledUnit(
                        prompt_ids=tuple(sorted(entry.unit.prompt_ids)),
                        coop_count=entry.unit.coop_count,
                        coop_total=entry.unit.coop_total,
                        rule=entry.reason,
                    )
                    for entry in sorted((entry for entry in kept if entry.filled), key=_fill_order)
                ),
                fill_rule=fill_rule,
            )
        )
    return tuple(outcomes)


def write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON artifact beside a corpus, parents made and a trailing newline kept."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _spec_from_args(args: argparse.Namespace) -> BreadthSpec:
    """Resolve the spec: a preset named on the command line, or a small JSON file, plus overrides."""
    if (args.spec is None) == (args.preset is None):
        raise ValueError(
            f"pass exactly one of --spec and --preset (presets: {sorted(SPEC_PRESETS)}); the grid has "
            f"to come from one place, because both artifacts record it as what was asked for"
        )
    if args.preset is not None:
        spec = SPEC_PRESETS[args.preset]
    else:
        payload = json.loads(Path(args.spec).read_text(encoding="utf-8"))
        spec = BreadthSpec.from_json_dict(dict(payload))
    overrides: dict[str, Any] = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.grading is not None:
        overrides["grading"] = args.grading
    if not overrides:
        return spec
    return dataclasses.replace(spec, **overrides)


def _runtime_clauses(args: argparse.Namespace) -> Mapping[str, str] | None:
    """Load the runtime counterpart clauses when a file was named, and none when it was not."""
    if args.framings_file is None:
        return None
    loaded = load_framings(Path(args.framings_file))
    logger.info(
        f"loaded runtime framings, path={loaded.path} digest={loaded.digest} "
        f"framing_ids={list(loaded.framing_ids)}"
    )
    return loaded.clauses


def _build(args: argparse.Namespace) -> int:
    """Render the candidate pool, and optionally write it as an unselected corpus."""
    spec = _spec_from_args(args)
    candidates = build_candidates(spec, runtime_clauses=_runtime_clauses(args))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.out_dir / "breadth-candidates.jsonl"
    write_corpus(rows_path, candidates.rows)
    manifest = {
        **candidates.to_manifest(),
        "candidates_path": str(rows_path),
        "candidates_sha256": rows_file_digest(rows_path),
    }
    write_manifest(args.out_dir / "breadth-candidates-manifest.json", manifest)
    logger.info(f"wrote {rows_path} and its manifest, n_rows={len(candidates.rows)}")
    if args.corpus_out is not None:
        if len(candidates.rows) % spec.step_prompt_multiple:
            raise ValueError(
                f"an unselected corpus of {len(candidates.rows)} rows is not a multiple of "
                f"{spec.step_prompt_multiple}, so TRL's sampler would drop part of every epoch. "
                f"Adjust the spec's quotas; nothing here silently drops rows a caller asked for."
            )
        write_corpus(args.corpus_out, candidates.rows)
        logger.info(
            f"wrote the UNSELECTED corpus {args.corpus_out}: every candidate, no sweep, no band. "
            f"Only a plumbing smoke reads a corpus built this way."
        )
    return 0


def _select(args: argparse.Namespace) -> int:
    """Judge a swept pool against the spec and write the corpus plus its stratum artifact."""
    spec = _spec_from_args(args)
    trace = read_jsonl(args.sweep)
    candidates = read_jsonl(args.candidates)
    selection = select_breadth(
        trace,
        candidates,
        spec,
        min_coop=args.min_coop,
        max_coop=args.max_coop,
        min_split_std=args.min_split_std,
        min_parseable_fraction=args.min_parseable_fraction,
        fill_pure_to_quota=args.fill_pure_to_quota,
        trust_min_split_std=args.trust_min_split_std,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = args.out_dir / "breadth-corpus.jsonl"
    write_corpus(corpus_path, selection.rows)
    write_manifest(
        args.out_dir / "breadth-strata.json",
        {
            **selection.to_artifact(),
            "corpus_path": str(corpus_path),
            "corpus_sha256": rows_file_digest(corpus_path),
            "candidates_path": str(args.candidates),
            "candidates_sha256": rows_file_digest(args.candidates),
            "sweep_path": str(args.sweep),
        },
    )
    logger.info(
        f"wrote {corpus_path} and its stratum artifact, n_rows={len(selection.rows)} "
        f"composition={selection.realised_composition()}"
    )
    return 0


def _add_spec_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--spec", type=Path, default=None, help="Path to a JSON breadth spec (see BreadthSpec)."
    )
    parser.add_argument(
        "--preset",
        choices=sorted(SPEC_PRESETS),
        default=None,
        help="Use a named grid instead of a spec file.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override the spec's draw seed.")
    parser.add_argument(
        "--grading",
        type=grading_cli_value,
        default=None,
        help=(
            "Override the grading the rows are written under. On `select` it only names the recorded "
            "spec, so it has to agree with the grading the candidate rows already carry; rewrite a "
            "corpus's grading with games.regrade_corpus."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/games/breadth"),
        help="Directory for the rows and the JSON artifact.",
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the oversampled candidate pool for a mixed-game, mixed-framing training corpus, "
            "and select the corpus from a sweep of that pool."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Render the candidate pool for a spec.")
    _add_spec_args(build)
    build.add_argument(
        "--framings-file",
        type=Path,
        default=None,
        help="Runtime counterpart clauses for the framings the tracked registry does not carry.",
    )
    build.add_argument(
        "--corpus-out",
        type=Path,
        default=None,
        help=(
            "Also write the candidates as a corpus, UNSELECTED. For a plumbing smoke only: no sweep "
            "stands behind it, so its prompts are not known to be answered in a mixed way."
        ),
    )
    build.set_defaults(handler=_build)

    select = subparsers.add_parser("select", help="Select the corpus from a swept pool.")
    _add_spec_args(select)
    select.add_argument(
        "--sweep", type=Path, required=True, help="The sweep trace over the candidate pool."
    )
    select.add_argument(
        "--candidates", type=Path, required=True, help="The candidate rows that sweep measured."
    )
    select.add_argument(
        "--min-coop",
        type=float,
        default=DEFAULT_MIN_COOP,
        help="Lowest cooperation rate a near or mid prompt may show.",
    )
    select.add_argument(
        "--max-coop",
        type=float,
        default=DEFAULT_MAX_COOP,
        help="Highest cooperation rate a near or mid prompt may show.",
    )
    select.add_argument(
        "--min-split-std",
        type=float,
        default=DEFAULT_MIN_SPLIT_STD,
        help="Smallest send spread a kept trust prompt may show.",
    )
    select.add_argument(
        "--min-parseable-fraction",
        type=float,
        default=DEFAULT_MIN_PARSEABLE_FRACTION,
        help="Share of completions that must parse for a prompt to be judged at all.",
    )
    select.add_argument(
        "--fill-pure-to-quota",
        action="store_true",
        help=(
            "OPTIONAL, off by default and not part of the registered selection: after the band rules "
            f"run, top every {' and '.join(FILL_BANDS)} stratum still short of its quota up with pairs "
            "the base played unanimously, preferring any pair with spread over a fully pure one. Near "
            "strata keep the mixed-only rule. Training drops a pure group at step time, so these "
            "prompts cost generation and never a gradient, and they are the ones the arm should move."
        ),
    )
    select.add_argument(
        "--trust-min-split-std",
        type=float,
        default=None,
        help=(
            "OPTIONAL: the send-spread floor for the trust strata alone, replacing --min-split-std "
            "there. Zero keeps every trust row that cleared the parse floor. In the 9B base sweep 29 "
            "of 32 trust rows sent the same amount in all eight draws, so their spread was exactly "
            "zero and nothing between zero and a third of the endowment admits any of them; the log "
            "line names what each observed value would admit."
        ),
    )
    select.set_defaults(handler=_select)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand: build the pool, or select the corpus from a sweep of it."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    args = _parse_args(argv)
    handler: Any = args.handler
    result: int = handler(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
