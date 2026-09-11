"""Re-emit a selected prompt corpus under a different grading, when that is provably sound.

The `twin-pd-group` and `twin-pd-self` arms are the plan's causal contrast: identical prompts, two
grading rules, so that any behavioural difference is attributable to the grading correlation
structure and nothing else. Sweeping twice would undermine exactly that, because the baseline sweep
samples a stochastic policy and the two sweeps would select overlapping-but-different prompt sets.

Re-grading one corpus is sound for two independent reasons, and this module checks the first rather
than assuming it:

*   **The prompts do not depend on the grading.** `games.prompts` threads `grading` into the row as
    a column and never into any rendering call, so for a given game the two gradings produce rows
    that differ in that one column. `assert_grading_is_prompt_independent` regenerates both and
    refuses if anything else differs -- so a game that ever does condition its text on the grading
    fails here instead of silently producing a corpus whose prompt and grading disagree.
*   **The selection criterion is grading-independent.** `games.select_prompts` keeps prompts whose
    *action* distribution is mixed. Which actions a policy samples is a property of the model and
    the prompt; the grading only decides what reward those actions earn afterwards. So the prompts
    worth training on are the same set either way.

    uv run python -m games.regrade_corpus --corpus <selected.jsonl> \
        --grading self --out <self.jsonl>
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from games.prompts import generate_prompt_rows
from games.rewards import grading_cli_value, is_grading, unknown_grading_message

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

GRADING_COLUMN = "grading"
GAME_ID_COLUMN = "game_id"


def assert_grading_is_prompt_independent(game_id: str, source: str, target: str) -> None:
    """Refuse to re-grade a game whose prompt text depends on which grading it is scored by.

    The check is generative rather than declarative: it renders the game's whole train split under
    both gradings and requires that `grading` is the only column that moves. That keeps this
    module correct for games it was never tested against, including ones not yet written.
    """
    source_rows = generate_prompt_rows(game_id, source, split="train")
    target_rows = generate_prompt_rows(game_id, target, split="train")
    if len(source_rows) != len(target_rows):
        raise ValueError(
            f"{game_id!r} renders {len(source_rows)} rows under {source!r} but "
            f"{len(target_rows)} under {target!r}, so the two gradings are not the same corpus "
            f"and re-grading would change which prompts the arm trains on."
        )
    differing: set[str] = set()
    for before, after in zip(source_rows, target_rows, strict=True):
        if set(before) != set(after):
            raise ValueError(
                f"{game_id!r} renders different columns under {source!r} and {target!r}: "
                f"{sorted(set(before) ^ set(after))}"
            )
        differing.update(key for key in before if before[key] != after[key])
    if differing != {GRADING_COLUMN}:
        raise ValueError(
            f"{game_id!r} prompts are not grading-independent: re-grading {source!r} -> {target!r} "
            f"would need to change {sorted(differing)}, not just {GRADING_COLUMN!r}. Sweep the "
            f"target grading directly instead of transforming this corpus."
        )
    logger.info(
        "grading-independence check passed, %s",
        f"{game_id=} {source=} {target=} n_rows={len(source_rows)}",
    )


def regrade_rows(rows: Sequence[dict[str, Any]], *, target_grading: str) -> list[dict[str, Any]]:
    """Return the same rows carrying `target_grading`, or raise if that is not a sound transform.

    Every other column is passed through untouched, including whatever the baseline sweep added,
    so the re-graded corpus stays the same prompt set with the same provenance.

    A corpus may carry SEVERAL games, which is what the wave-4b breadth arms need: one mixed corpus
    of five matrix games plus the trust sender, regraded from `care-alpha-1` to `care-alpha-0` for the
    control. The grading-independence check then runs once per distinct game in the file rather than
    once, because the property it checks is a property of a game's renderer and each game has to
    answer for itself. The single-GRADING requirement stays: a file mixing gradings is already
    unusable, since the reward function needs one grading per group.
    """
    if not is_grading(target_grading):
        raise ValueError(unknown_grading_message(target_grading))
    if not rows:
        raise ValueError("Nothing to re-grade: the corpus holds no rows.")
    for column in (GRADING_COLUMN, GAME_ID_COLUMN):
        if column not in rows[0]:
            raise ValueError(f"Corpus rows carry no {column!r} column; columns: {sorted(rows[0])}")

    source_gradings = {str(row[GRADING_COLUMN]) for row in rows}
    if len(source_gradings) != 1:
        raise ValueError(
            f"Corpus mixes gradings {sorted(source_gradings)}. The reward function requires one "
            f"grading per group, so a mixed corpus is already unusable."
        )
    game_ids = sorted({str(row[GAME_ID_COLUMN]) for row in rows})

    source_grading = next(iter(source_gradings))
    if source_grading == target_grading:
        raise ValueError(
            f"Corpus is already graded {target_grading!r}; re-grading would be a no-op copy that "
            f"later reads as a second, independent selection."
        )
    for game_id in game_ids:
        assert_grading_is_prompt_independent(game_id, source_grading, target_grading)
    logger.info(
        "re-graded corpus, %s",
        f"{game_ids=} {source_grading=} -> {target_grading=} n_rows={len(rows)}",
    )
    return [{**row, GRADING_COLUMN: target_grading} for row in rows]


def read_corpus(path: Path) -> list[dict[str, Any]]:
    """Read a headerless corpus JSONL, the shape `games.select_prompts` writes."""
    rows: list[dict[str, Any]] = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    if not rows:
        raise ValueError(f"Corpus {path} holds no rows.")
    return rows


def write_corpus(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write a headerless corpus JSONL, matching what the training loader expects."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("wrote %s", path)


def main(argv: Sequence[str] | None = None) -> None:
    """Re-grade one corpus file into another."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path, help="Selected corpus JSONL to read.")
    # `type=` rather than `choices=`: the care family's alpha is a number, so no fixed list can hold
    # it, and argparse's own "invalid choice" would name a vocabulary the family is missing from.
    parser.add_argument("--grading", required=True, type=grading_cli_value)
    parser.add_argument(
        "--out", required=True, type=Path, help="Where to write the re-graded copy."
    )
    args = parser.parse_args(argv)
    write_corpus(args.out, regrade_rows(read_corpus(args.corpus), target_grading=args.grading))


if __name__ == "__main__":
    main()
