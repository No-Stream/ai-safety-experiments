"""Render one game's whole prompt corpus, before any baseline selection narrows it.

`games.select_prompts` writes the corpus an arm trains on: the prompts whose baseline action
distribution was mixed enough to carry a gradient. That is the right file to train on and the wrong
file to read an amplification effect off. The prompts selection swept out are exactly the ones whose
cooperation sat at zero or one at step 0, and "did this skin's cooperation rise" is a question about
those too -- gains confined to skins with baseline cooperative mass read as amplification, gains on
near-zero-mass skins read as creation, and a corpus holding only the middle band cannot tell the two
apart. So the read needs the pre-selection render, and until now no artifact held one.

Nothing here builds a row or serialises one. The rows come from
`games.prompts.generate_prompt_rows`, the same deterministic call the baseline sweep and
`games.train --generate-fresh` make, and the file comes from `games.select_prompts.write_corpus`. A
local copy of either would be free to drift, and a drifted schema would mean the amplification read
and the trained arm were run over two different prompt sets with nothing saying so. For the same
reason this module validates nothing about the game, grading, split or print order:
`games.prompts` already refuses each bad combination by name, including the eval-only games that
have no train split, and a second copy of those checks here could only answer with a worse message.

    uv run python -m games.render_corpus --game pd-reskin --grading self --split train \
        --out artifacts/games/render/pd-reskin-self-train.jsonl
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from games.prompts import LABEL_PRINT_ORDER_CANONICAL, generate_prompt_rows
from games.select_prompts import write_corpus

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


def render_corpus(
    out: Path,
    *,
    game_id: str,
    grading: str,
    split: str,
    label_print_order: str = LABEL_PRINT_ORDER_CANONICAL,
) -> list[dict[str, Any]]:
    """Write every prompt row for one game, grading and split, in the selection-artifact schema.

    Refuses an `--out` that exists rather than overwriting it. A corpus file is what a later
    analysis resolves a `prompt_id` against, so silently replacing one would leave every artifact
    that named it pointing at a different prompt set.
    """
    if out.exists():
        raise FileExistsError(
            f"{out} already exists; refusing to overwrite it. A corpus file is what a later "
            f"analysis resolves its prompt_ids against, so replacing one in place would "
            f"re-point every artifact that names it. Render to a new path, or delete this one."
        )
    rows = generate_prompt_rows(game_id, grading, split=split, label_print_order=label_print_order)
    write_corpus(out, rows)
    logger.info(
        "rendered full corpus, %s",
        f"{game_id=} {grading=} {split=} {label_print_order=} n_rows={len(rows)} out={out}",
    )
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    """Render one game/grading/split to a corpus JSONL."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", required=True, help="Game id to render, e.g. pd-reskin.")
    parser.add_argument("--grading", required=True, help="Grading to stamp on every row.")
    parser.add_argument("--split", required=True, help="Which split to render: train or eval.")
    parser.add_argument(
        "--label-print-order",
        default=LABEL_PRINT_ORDER_CANONICAL,
        help=(
            "Which action label the prompts print first. Defaults to the canonical order every "
            "corpus so far was built in; a swapped render is a different prompt set."
        ),
    )
    parser.add_argument(
        "--out", required=True, type=Path, help="Where to write the corpus JSONL (must not exist)."
    )
    args = parser.parse_args(argv)
    render_corpus(
        args.out,
        game_id=args.game,
        grading=args.grading,
        split=args.split,
        label_print_order=args.label_print_order,
    )


if __name__ == "__main__":
    main()
