"""Check a corpus against the consumers that will read it, on CPU, before a card is reserved.

The failure this exists for: a 9B arm booted its vLLM engine on a rented card, ran one generation
batch, and died in the reward function because its corpus predated nine of the columns
`games.rewards` rebuilds each row from. Everything up to that point was green, because no earlier
consumer looks at those columns -- the training loader checks two of them, the frames evaluator
fourteen -- and the reward schema is the widest of the three. So the whole cost of a wrong corpus
was paid at the one place that could not be reached without a GPU.

Every check here is the consumer's own function, imported and called rather than reproduced:

*   `games.eval_training_frames.load_corpus_rows` -- the downstream evaluator. Reads the file,
    refuses an empty one and requires its per-row corpus fields. Which games the file may hold is not
    that loader's rule any more (a breadth arm's corpus is several games, and the set belongs to the
    arm), and it is not repeated here either: `load_corpus` below applies the same set per row and
    refuses harder, since it checks the grading with it.
*   `games.train.load_corpus` plus `filter_payoff_variants`, `filter_corpus_partition` and
    `assert_opponent_distribution_sampled` -- the trainer's own corpus path, in the order
    `games.train.prepare_rows` runs them, so an arm's payoff or partition pin is applied here too
    and the row count reported is the one the run would actually train on.
*   `games.train.assert_corpus_selected_for_model`, when `--model` names the checkpoint the run
    will train, which is the other half of a corpus's identity next to its game and grading.
*   `games.rewards.REQUIRED_REWARD_COLUMNS` -- the tuple the reward function itself indexes rows
    with, derived from its own `_Row` dataclass fields. Imported rather than restated, because a
    restated list is exactly how the 9B corpus came to be nine columns short.

The reward function is not itself invoked: it needs completions that parse into an action, and the
answer shape is per game (`<action>LABEL</action>`, `<claim>N</claim>`, `<contribute>N</contribute>`
and so on), so synthesising them here would be a second copy of every game's answer format --
exactly the kind of copy that drifts. The column tuple cannot drift, since adding a field to `_Row`
extends it.

Refusals raise rather than printing, so the process exits non-zero with the offending columns or
path named in the traceback, which is what a launch runner needs to stop on.

    uv run python -m games.corpus_preflight --corpus artifacts/games/select/pd-reskin-self.jsonl
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from games.arms import ARMS, arm_game_ids
from games.eval_training_frames import load_corpus_rows
from games.rewards import REQUIRED_REWARD_COLUMNS
from games.train import (
    assert_corpus_selected_for_model,
    assert_opponent_distribution_sampled,
    assert_stated_match_mixture,
    filter_corpus_partition,
    filter_payoff_variants,
    load_corpus,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from games.arms import GameArm

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CorpusPreflight:
    """What a passing preflight establishes about one corpus file.

    `n_rows_after_arm_pins` is reported separately from `n_rows` because an arm pinning one payoff
    variant or one side of the group-mix boundary trains a subset of the file, and "this arm covers
    the ladder" versus "this arm trains one rung" is a difference nothing else in the artifacts
    records.

    `game_id` is the arm's lead game and `corpus_game_ids` is what the file actually holds, which are
    the same one game for every arm before wave 4b and are not for a breadth arm: a report naming the
    lead game alone would read as a single-game corpus whatever the file contained.
    """

    path: Path
    arm_name: str
    game_id: str
    corpus_game_ids: tuple[str, ...]
    grading: str
    n_rows: int
    n_rows_after_arm_pins: int


def resolve_arm(rows: Sequence[dict[str, Any]], *, arm_name: str | None) -> tuple[str, GameArm]:
    """Name the registered arm whose corpus this file claims to be.

    An explicit `arm_name` wins and is checked against the registry. Without one the arm is derived
    from the corpus's own game and grading columns, which is unambiguous for most arms and is not
    for the (game, grading) pairs several arms share -- the stag ladder's rungs differ only in their
    payoff pin. Guessing one of those would apply pins the run does not have and report a row count
    for a corpus nobody is training, so an ambiguous corpus asks for `--arm` instead.

    The derivation asks for an arm whose game SET covers every game the file holds
    (`games.arms.arm_game_ids`), not just row 0's game: a breadth arm's corpus is several games, and a
    rule reading one row would either pick an arm by whichever game came first or find none at all.
    That widening can make a corpus ambiguous where it was not -- a single-game file could be a
    breadth arm's corpus with rows missing -- and an ambiguous corpus asks for `--arm`, which is the
    same answer the payoff-pin case already gets.
    """
    if arm_name is not None:
        arm = ARMS.get(arm_name)
        if arm is None:
            raise ValueError(f"{arm_name!r} is not a registered arm; known arms: {sorted(ARMS)}.")
        return arm_name, arm

    corpus_games = sorted({str(row["game_id"]) for row in rows})
    grading = str(rows[0]["grading"])
    covers = {name: arm_game_ids(arm) for name, arm in ARMS.items()}
    candidates = sorted(
        name
        for name, arm in ARMS.items()
        if arm.grading == grading and set(corpus_games) <= set(covers[name])
    )
    if not candidates:
        raise ValueError(
            f"no registered arm trains {corpus_games} under {grading!r}, so there is no arm whose "
            f"pins this corpus could be checked against. Registered arms carrying any of "
            f"{corpus_games}: "
            f"{sorted(name for name, games in covers.items() if set(corpus_games) & set(games))}. "
            f"Pass --arm to check it against a specific arm anyway."
        )
    if len(candidates) > 1:
        raise ValueError(
            f"{len(candidates)} arms train {corpus_games} under {grading!r}: {candidates}. They "
            f"differ in the games, payoff variants and corpus partition they pin, so which one this "
            f"corpus belongs to decides how many of its rows a run would actually train. Pass "
            f"--arm with one of them."
        )
    resolved = candidates[0]
    logger.info(f"derived arm from the corpus itself, {resolved=} {corpus_games=} {grading=}")
    return resolved, ARMS[resolved]


def assert_reward_columns(rows: Sequence[dict[str, Any]], *, path: Path) -> None:
    """Refuse a corpus the reward function could not rebuild its rows from.

    Checked across every row rather than the first, so a file concatenated from two corpus vintages
    is caught as well as one that is uniformly short. `games.rewards` does raise on this by itself,
    but only once TRL has handed it a batch -- after the weights are resident and one generation
    pass has been paid for.
    """
    missing = sorted(
        {column for row in rows for column in REQUIRED_REWARD_COLUMNS if column not in row}
    )
    if missing:
        raise ValueError(
            f"corpus {path} is missing reward columns {missing}; the reward function rebuilds "
            f"every row from {list(REQUIRED_REWARD_COLUMNS)} and raises on a missing one, inside "
            f"the training loop, after the engine has loaded and a generation batch has been "
            f"spent. Re-sweep or re-render this corpus against the current row schema."
        )


def preflight_corpus(
    path: Path, *, arm_name: str | None = None, model_id: str | None = None
) -> CorpusPreflight:
    """Run one corpus through every consumer's own checks, and report what passed.

    Raises on the first refusal. `model_id` enables the selected-for-model check, which is skipped
    rather than faked when the launch's checkpoint is not supplied.
    """
    if not path.is_file():
        raise FileNotFoundError(f"--corpus {path} does not exist.")

    frame_rows = load_corpus_rows(path)
    resolved_arm_name, arm = resolve_arm(frame_rows, arm_name=arm_name)

    rows = load_corpus(str(path), arm)
    if model_id is not None:
        assert_corpus_selected_for_model(rows, path=str(path), model_id=model_id)
    else:
        logger.info(
            "skipping the selected-for-model check: pass --model with the checkpoint this run "
            "will train to enable it"
        )
    kept = filter_payoff_variants(rows, arm.payoff_variants)
    kept = filter_corpus_partition(kept, arm.corpus_partition)
    assert_opponent_distribution_sampled(kept, arm)
    assert_stated_match_mixture(kept, arm)
    assert_reward_columns(kept, path=path)

    report = CorpusPreflight(
        path=path,
        arm_name=resolved_arm_name,
        game_id=arm.game_id,
        corpus_game_ids=tuple(sorted({str(row["game_id"]) for row in kept})),
        grading=arm.grading,
        n_rows=len(rows),
        n_rows_after_arm_pins=len(kept),
    )
    logger.info(
        "corpus preflight passed, %s",
        f"path={report.path} arm={report.arm_name} game_id={report.game_id} "
        f"corpus_game_ids={list(report.corpus_game_ids)} "
        f"grading={report.grading} n_rows={report.n_rows} "
        f"n_rows_after_arm_pins={report.n_rows_after_arm_pins}",
    )
    return report


def main(argv: Sequence[str] | None = None) -> None:
    """Preflight one corpus file, raising on the first check it fails."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path, help="Corpus JSONL to check.")
    parser.add_argument(
        "--arm",
        default=None,
        help=(
            "Arm whose pins to check the corpus against. Derived from the corpus's own game and "
            "grading when omitted, which is unambiguous unless several arms share that pair."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Checkpoint this run will train. Enables the selected-for-model check, which refuses "
            "a corpus swept against a different model."
        ),
    )
    args = parser.parse_args(argv)
    preflight_corpus(args.corpus, arm_name=args.arm, model_id=args.model)


if __name__ == "__main__":
    main()
