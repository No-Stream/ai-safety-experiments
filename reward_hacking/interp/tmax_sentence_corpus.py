"""The 290 contrastive sentence pairs of `reward_hacking.interp.stimuli` as a capture stimulus file.

The capture driver (`reward_hacking.interp.tmax_capture_ladder`) reads a five-key stimulus file and
names every cell's stimulus sets and sides after it; the direction readers (`tmax_directions`) select a
set and two sides by name. Until this module nothing wrote the sentence pairs in that form, so the name
a reader expected was an assumption. Here each pair list becomes one stimulus set named by its
`stimuli` key (`eval_awareness`, `shortcut`, `deception`, `contradiction`), with the two sentences of
pair `i` as sides `positive` and `negative` of pair id `<set>--<i>`; the set and side names are
`stimuli.SENTENCE_STIMULUS_SETS` and `stimuli.SENTENCE_SIDES`, imported, so the writer and every
reader spell them from one place.

The rows are bare stems: the capture templates them as one user turn (`--stimulus-render
templated_here`). The games render check refuses a bare stem under `verbatim`, so the other convention
is an error rather than a different measurement. The file carries no spans and needs no sidecar; its
digest is the cells' identity.

    uv run python -m reward_hacking.interp.tmax_sentence_corpus --out-dir artifacts/.../sentence-corpus
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast

from games.interp_cells import (
    STIMULUS_RENDER_TEMPLATED,
    Stimulus,
    load_stimuli,
    short_digest,
    stimuli_digest,
)
from reward_hacking.interp.stimuli import (
    SENTENCE_SIDE_NEGATIVE,
    SENTENCE_SIDE_POSITIVE,
    SENTENCE_STIMULUS_SETS,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

logger = logging.getLogger(__name__)

STIMULI_FILENAME = "stimuli.jsonl"
DEFAULT_OUT_DIR = Path("artifacts/reward_hacking/tmax-interp/sentence-corpus")

STIMULUS_RENDER = STIMULUS_RENDER_TEMPLATED
"""Bare stems: the ladder templates them, so it runs with `--stimulus-render templated_here`."""

PAIR_SEPARATOR = "--"


class SentenceCorpusError(ValueError):
    """The sentence corpus cannot be written as asked, or did not read back as written."""


def assert_known_sentence_sets(names: Iterable[str]) -> None:
    """Refuse a set name `stimuli` does not spell, listing the ones it does."""
    unknown = sorted(set(names) - set(SENTENCE_STIMULUS_SETS))
    if unknown:
        raise SentenceCorpusError(
            f"{unknown} are not sentence stimulus sets; the sets are {sorted(SENTENCE_STIMULUS_SETS)}, "
            f"spelled as reward_hacking.interp.stimuli names them"
        )


def pair_id(stimulus_set: str, index: int) -> str:
    """Name pair `index` of a set: `<set>--<index>`, zero-padded so the ids sort in corpus order."""
    return f"{stimulus_set}{PAIR_SEPARATOR}{index:03d}"


def sentence_stimuli(
    stimulus_sets: Sequence[str] = tuple(SENTENCE_STIMULUS_SETS),
) -> list[Stimulus]:
    """Every pair of the named sets as two five-key rows, positive then negative, in list order."""
    assert_known_sentence_sets(stimulus_sets)
    rows: list[Stimulus] = []
    for stimulus_set in stimulus_sets:
        for index, pair in enumerate(SENTENCE_STIMULUS_SETS[stimulus_set]):
            pair_name = pair_id(stimulus_set, index)
            for side, text in (
                (SENTENCE_SIDE_POSITIVE, pair.positive),
                (SENTENCE_SIDE_NEGATIVE, pair.negative),
            ):
                rows.append(
                    Stimulus(
                        stimulus_id=f"{pair_name}{PAIR_SEPARATOR}{side}",
                        stimulus_set=stimulus_set,
                        side=side,
                        pair_id=pair_name,
                        text=text,
                    )
                )
    return rows


def write_sentence_stimuli(
    out_dir: Path, stimulus_sets: Sequence[str] = tuple(SENTENCE_STIMULUS_SETS)
) -> Path:
    """Write the stimulus file and read it back through the capture's own loader; return its path.

    Refuses to overwrite: the file's digest is the identity every cell captured on it joins on.
    """
    path = out_dir / STIMULI_FILENAME
    if path.exists():
        raise FileExistsError(
            f"{path} already exists; a stimulus file is the identity every capture joins on, so write "
            f"a new directory or delete this one deliberately"
        )
    rows = sentence_stimuli(stimulus_sets)
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                {
                    "id": row.stimulus_id,
                    "set": row.stimulus_set,
                    "side": row.side,
                    "pair_id": row.pair_id,
                    "text": row.text,
                }
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    loaded = load_stimuli(path)
    if loaded != rows:
        raise SentenceCorpusError(f"{path} did not read back as written")
    logger.info(
        f"wrote the sentence corpus, {path=} n_stimuli={len(rows)} sets={list(stimulus_sets)} "
        f"digest={short_digest(stimuli_digest(rows))}"
    )
    return path


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--sets",
        default=",".join(SENTENCE_STIMULUS_SETS),
        help=f"Comma-separated subset of {sorted(SENTENCE_STIMULUS_SETS)}; default all of them.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Write the sentence stimulus file under the given directory."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    names = [part.strip() for part in cast("str", args.sets).split(",") if part.strip()]
    write_sentence_stimuli(cast("Path", args.out_dir), names)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
