"""Capture corpus for the framing x interp cross: twin-pd prompts under every counterpart framing.

The behavioral framing sweep found that at 9B the twin clause carries all the cooperation, and the
interp arc found a base-native correlated-counterpart ("lead") direction that steers it. This
corpus is the join between the two: the same twin-pd eval rows the steering leg generates on,
rendered under each registered counterpart framing, shaped as `games.interp_cells` stimuli so
`games.interp_capture` can cache activations for them and a CPU read can ask where each framing's
prompts sit along the lead axis.

The shape is the measurement. `side` holds the framing id -- the grouping key of the projection
read -- and `pair_id` holds the underlying game row, shared by exactly the framings rendered for
that row, so framing contrasts are within-prompt paired differences rather than across-prompt ones.
Rows carry no `assistant_prefix`: the read pools at the prompt end (the assistant turn's opening
``<think>``), the state the model starts thinking from, which is the earliest point a stated
counterpart belief could be carried forward. The twin rows must be byte-identical to the trained
rendering (`generate_framing_prompt_rows` guarantees the text; the tests pin it), so the twin
framing is the reference cell every other framing is read against.

Both print orders are rendered, mirroring the framing sweep and the standing rule that the
first-printed-label confound is split, never averaged away.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from games.interp_cells import Stimulus
from games.prompts import (
    COUNTERPART_FRAMING_IDS,
    LABEL_PRINT_ORDERS,
    SPLIT_EVAL,
    generate_framing_prompt_rows,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

GAME_ID = "twin-pd"
RENDER_GRADING = "self"
"""Prompt text is identical across gradings (a reward-time fact, not a prompt fact); this names
which registry entry rendered the rows, matching `games.interp_steering.RENDER_GRADING`."""

FRAMING_CAPTURE_SET = "framing-twin-pd"
STIMULI_FILENAME = "framing-stimuli.jsonl"
PROVENANCE_FILENAME = "framing-stimuli-provenance.jsonl"
DEFAULT_OUT_DIR = Path("artifacts/games/framing-interp-cross")

_FRAMING_SEGMENT_TEMPLATE = "--framing-{framing_id}"


def _base_prompt_id(prompt_id: str, framing_id: str) -> str:
    """Strip the framing segment, leaving the id of the underlying game row.

    `generate_framing_prompt_rows` writes exactly one ``--framing-<id>`` segment into every
    prompt_id; the stripped id is what every framing of one row shares, so it is the pair key
    of the within-prompt contrast. Anything but exactly one occurrence means the id convention
    changed underneath this corpus, and the corpus refuses rather than pairing rows wrongly.
    """
    segment = _FRAMING_SEGMENT_TEMPLATE.format(framing_id=framing_id)
    if prompt_id.count(segment) != 1:
        raise ValueError(
            f"prompt_id {prompt_id!r} carries {prompt_id.count(segment)} copies of {segment!r}; "
            f"the pair key needs exactly one, so the prompt_id convention has changed underneath "
            f"this corpus."
        )
    return prompt_id.replace(segment, "")


def _render_corpus(
    framings: Sequence[str],
) -> list[tuple[Stimulus, dict[str, Any]]]:
    """One render pass: each stimulus beside the provenance fields the analysis splits on."""
    rendered: list[tuple[Stimulus, dict[str, Any]]] = []
    for framing_id in framings:
        for order in LABEL_PRINT_ORDERS:
            for row in generate_framing_prompt_rows(
                GAME_ID,
                RENDER_GRADING,
                framing_id=framing_id,
                split=SPLIT_EVAL,
                label_print_order=order,
            ):
                pair_id = _base_prompt_id(str(row["prompt_id"]), framing_id)
                stimulus = Stimulus(
                    stimulus_id=str(row["prompt_id"]),
                    stimulus_set=FRAMING_CAPTURE_SET,
                    side=framing_id,
                    pair_id=pair_id,
                    text=str(row["prompt"]),
                )
                provenance = {
                    "id": row["prompt_id"],
                    "framing": framing_id,
                    "pair_id": pair_id,
                    "reskin_id": row["reskin_id"],
                    "payoff_variant": row["payoff_variant"],
                    "label_print_order": row["label_print_order"],
                    "coop_label": row["coop_label"],
                }
                rendered.append((stimulus, provenance))
    _assert_pairs_complete([stimulus for stimulus, _ in rendered], framings)
    return rendered


def framing_capture_stimuli(
    framings: Sequence[str] = COUNTERPART_FRAMING_IDS,
) -> list[Stimulus]:
    """Render the corpus: every requested framing of every twin-pd eval row, both print orders."""
    return [stimulus for stimulus, _ in _render_corpus(framings)]


def _assert_pairs_complete(stimuli: Sequence[Stimulus], framings: Sequence[str]) -> None:
    """Refuse a corpus where any pair is missing a framing: the paired read would silently thin."""
    sides_by_pair: dict[str, set[str]] = {}
    for stimulus in stimuli:
        sides_by_pair.setdefault(stimulus.pair_id, set()).add(stimulus.side)
    incomplete = {
        pair: sorted(set(framings) - sides)
        for pair, sides in sides_by_pair.items()
        if sides != set(framings)
    }
    if incomplete:
        raise ValueError(
            f"{len(incomplete)} pairs are missing framings ({dict(list(incomplete.items())[:3])}); "
            f"a within-prompt framing contrast over an incomplete pair set is a different corpus."
        )


def write_framing_stimuli(
    out_dir: Path, framings: Sequence[str] = COUNTERPART_FRAMING_IDS
) -> tuple[Path, Path]:
    """Write the five-key stimuli file and its provenance sidecar. Returns both paths."""
    rendered = _render_corpus(framings)
    stimuli = [stimulus for stimulus, _ in rendered]
    out_dir.mkdir(parents=True, exist_ok=True)
    stimuli_path = out_dir / STIMULI_FILENAME
    provenance_path = out_dir / PROVENANCE_FILENAME
    stimuli_path.write_text(
        "".join(
            json.dumps(
                {
                    "id": stimulus.stimulus_id,
                    "set": stimulus.stimulus_set,
                    "side": stimulus.side,
                    "pair_id": stimulus.pair_id,
                    "text": stimulus.text,
                }
            )
            + "\n"
            for stimulus in stimuli
        ),
        encoding="utf-8",
    )
    provenance_path.write_text(
        "".join(f"{json.dumps(provenance)}\n" for _, provenance in rendered),
        encoding="utf-8",
    )
    logger.info(
        f"wrote {len(stimuli)} framing capture stimuli to {stimuli_path} "
        f"(provenance {provenance_path})"
    )
    return stimuli_path, provenance_path


def main(argv: Sequence[str] | None = None) -> None:
    """Render the corpus and write it with its provenance sidecar."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    write_framing_stimuli(args.out_dir)


if __name__ == "__main__":
    main()
