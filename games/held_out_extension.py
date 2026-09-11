"""Runtime loader for the held-out scenario extension: four banks of eval-only authored frames.

The tracked rosters in :mod:`games.prompts` are two to four held-out frames wide per bank, so every
per-frame rate a battery reports is a mean over a handful of fictions. Widening them means authoring
new frames, and an authored frame is benchmark material: it will be run against future models, and a
model that memorised it invalidates every measurement made with it. So the new frames live in one
gitignored staging file that this module reads at runtime and refuses to run without, exactly as
:mod:`sociology.transfer_stimulus` reads its stimulus. No frame prose, no label and no resource name
appears in tracked code; this module knows the file's SHAPE and the gates its entries must pass.

Everything the file carries is held out entirely. The staging schema forbids an `eval_only` field
precisely so that this loader is the only thing that decides: it constructs every entry
``eval_only=True``, which is the tracked mechanism that keeps a frame out of every training corpus.
An entry that carried the field would be an author asserting the opposite of the file's purpose, so
it is refused rather than overridden.

Four checks run at load, each a property a reading depends on rather than tidiness:

- **The tracked vocabulary gates, on the authored prose and on every render.**
  :func:`games.prompts.assert_no_loaded_vocabulary` keeps the literature's names out of the prompts,
  and the renderers call it on their own output, so the frames are rendered here through the same
  public renderers the row builders use. A leak found at load costs nothing; the same leak found
  mid-battery costs the cell.
- **No decision-coupling language in a matrix frame.**
  :func:`games.prompts.assert_no_coupling_claims`. A skin's fiction may imply that a counterpart
  exists and must say nothing about how it decides, or the twin framing comes back in through the
  story and a self-graded arm trains on prompts that lie about the reward's coupling.
- **Ids unique against every tracked roster and within the file.** A `scenario_id` is a `prompt_id`
  segment and an artifact filename segment, so a reused one pools two frames' draws.
- **Label strings globally unique.** The pin `TestScenarioRoster` holds the tracked matrix and
  responder rosters to, extended over the new frames: a label appearing in two frames would pool
  them in every per-label analysis without anything looking wrong.

The authoring-quality checks (label form, resource plurality, figures stated in prose that the
renderer is supposed to print) live in the package's own ``validate_extension.py`` beside the staging
file, together with the sabotage suite that watches each of them go red. This loader re-runs the
gates that TRACKED code owns, because those are the ones a future roster edit can break.

**A matrix game renders only the extension frames its cap lists.** Eleven registered matrix games
draw on the shared house roster, so appending the whole matrix bank to that roster would widen every
one of their eval splits at once: about five times the game-behaviour section's render count, and a
different held-out bank for every game whose numbers are compared across waves. The caps file beside
the staging file therefore names, per game, the frames that game renders, by id and never by
position, and a matrix game absent from it renders none of them. The resource banks need no such
statement: dictator, trust and trustee each have exactly one game reading exactly one roster.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from games.payoffs import DictatorSpec, TrustSpec, trust_return_fraction, twin_pd
from games.prompts import (
    COOP_LABEL_INDICES,
    DICTATOR_ENDOWMENTS,
    DICTATOR_GAME_ID,
    DICTATOR_SCENARIOS,
    FRAMEABLE_GAME_IDS,
    LABEL_PRINT_ORDERS,
    MATRIX_SCENARIOS,
    MIN_EFFORT_SCENARIOS,
    NASH_DEMAND_SCENARIOS,
    NO_EXTRA_EVAL_FRAMES,
    PAYOFF_VARIANTS,
    RESKIN_SCENARIOS,
    RESPONDER_SCENARIOS,
    THRESHOLD_GOODS_SCENARIOS,
    TRUST_ENDOWMENT,
    TRUST_MULTIPLIER,
    TRUST_PAYOFF_VARIANTS,
    TRUST_ROSTER_GAME_IDS,
    TRUST_SCENARIOS,
    TRUST_STATED_RETURN_GAME_ID,
    TRUST_STRATEGY_METHOD_GAME_ID,
    TRUSTEE_RETURN_GAME_ID,
    TRUSTEE_SCENARIOS,
    DictatorScenario,
    ExtraEvalFrames,
    Scenario,
    TrustScenario,
    assert_no_coupling_claims,
    assert_no_loaded_vocabulary,
    render_dictator_prompt,
    render_game_prompt,
    render_trust_stated_return_prompt,
    render_trust_strategy_prompt,
    render_trustee_prompt,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

logger = logging.getLogger(__name__)

STAGING_PATH = Path("docs/scratch/held-out-extension-2026-09-02/staging/extension-all.json")
"""Where the authored staging file sits on a machine that has it; a fresh clone does not."""

EXTENSION_VERSION = "games-held-out-extension-v1"
CAPS_VERSION = "games-held-out-extension-caps-v1"

CAPS_FILENAME = "extension-caps.json"
"""The caps sidecar's name, read from the staging file's own directory so the two travel together."""

MATRIX_GAME_CAPS_KEY = "matrix_game_caps"

FAMILY_MATRIX = "matrix"
FAMILY_DICTATOR = "dictator"
FAMILY_TRUST = "trust"
FAMILY_TRUSTEE = "trustee"
EXTENSION_FAMILIES: tuple[str, ...] = (
    FAMILY_MATRIX,
    FAMILY_DICTATOR,
    FAMILY_TRUST,
    FAMILY_TRUSTEE,
)

MATRIX_ENTRY_FIELDS: frozenset[str] = frozenset({"scenario_id", "frame", "label_a", "label_b"})
RESOURCE_ENTRY_FIELDS: frozenset[str] = frozenset({"scenario_id", "frame", "resource"})

SCENARIO_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
"""The spelling every tracked id already uses; it lands in `prompt_id`s and artifact filenames."""

# Every tracked roster an id may not collide with, named so a refusal says where the collision is.
TRACKED_ROSTERS: dict[str, tuple[Scenario | DictatorScenario | TrustScenario, ...]] = {
    "MATRIX_SCENARIOS": MATRIX_SCENARIOS,
    "RESKIN_SCENARIOS": RESKIN_SCENARIOS,
    "RESPONDER_SCENARIOS": RESPONDER_SCENARIOS,
    "DICTATOR_SCENARIOS": DICTATOR_SCENARIOS,
    "TRUST_SCENARIOS": TRUST_SCENARIOS,
    "TRUSTEE_SCENARIOS": TRUSTEE_SCENARIOS,
}
# The claim, undertaking and effort rosters carry no extension family, and their ids share the same
# `prompt_id` namespace, so they are collision sources even though nothing here appends to them.
TRACKED_ID_ONLY_ROSTERS: dict[str, tuple[Any, ...]] = {
    "NASH_DEMAND_SCENARIOS": NASH_DEMAND_SCENARIOS,
    "THRESHOLD_GOODS_SCENARIOS": THRESHOLD_GOODS_SCENARIOS,
    "MIN_EFFORT_SCENARIOS": MIN_EFFORT_SCENARIOS,
}

# The two rosters `TestScenarioRoster` pins label strings across, which the new matrix frames join.
LABELLED_TRACKED_ROSTERS: tuple[tuple[Scenario, ...], ...] = (
    MATRIX_SCENARIOS,
    RESKIN_SCENARIOS,
    RESPONDER_SCENARIOS,
)


@dataclass(frozen=True, slots=True)
class ExtensionRosters:
    """The four loaded banks, the per-game matrix caps, and the digest of the file they came from.

    ``digest`` covers the whole staging payload and the caps together, because both decide what a
    cell renders: it reaches the trace meta and the cell identity through
    `games.evals.EvalConfig.as_record`, so a cell run with the extension can never share a bank
    entry or a resumed trace with one run without it, or with one run under a different cap.
    """

    matrix: tuple[Scenario, ...]
    dictator: tuple[DictatorScenario, ...]
    trust: tuple[TrustScenario, ...]
    trustee: tuple[TrustScenario, ...]
    matrix_ids_by_game: tuple[tuple[str, tuple[str, ...]], ...]
    digest: str

    def matrix_for_game(self, game_id: str) -> tuple[Scenario, ...]:
        """Return the extension frames one matrix game renders, in the staging file's own order.

        File order rather than the caps file's listing order, so re-ordering a cap's list is not a
        different measurement while adding or removing an id is.
        """
        capped = dict(self.matrix_ids_by_game).get(game_id)
        if capped is None:
            return ()
        return tuple(scenario for scenario in self.matrix if scenario.scenario_id in set(capped))

    def extra_eval_frames_for(self, game_id: str) -> ExtraEvalFrames:
        """Return exactly the family one game's row builder reads, and nothing else.

        One family rather than all four, because `games.prompts.generate_prompt_rows` refuses a
        family its chosen renderer never iterates: a game handed the wrong bank must be a loud
        refusal, not a cell that quietly measured the narrow roster.
        """
        if game_id == DICTATOR_GAME_ID:
            return ExtraEvalFrames(dictator=self.dictator)
        if game_id == TRUSTEE_RETURN_GAME_ID:
            return ExtraEvalFrames(trustee=self.trustee)
        if game_id in TRUST_ROSTER_GAME_IDS:
            return ExtraEvalFrames(trust=self.trust)
        if game_id in FRAMEABLE_GAME_IDS:
            return ExtraEvalFrames(matrix=self.matrix_for_game(game_id))
        return NO_EXTRA_EVAL_FRAMES

    def as_record(self) -> dict[str, Any]:
        """Return what the trace meta and the cell identity record about this extension."""
        return {
            "version": EXTENSION_VERSION,
            "digest": self.digest,
            "n_frames_by_family": {
                FAMILY_MATRIX: len(self.matrix),
                FAMILY_DICTATOR: len(self.dictator),
                FAMILY_TRUST: len(self.trust),
                FAMILY_TRUSTEE: len(self.trustee),
            },
            MATRIX_GAME_CAPS_KEY: {game_id: list(ids) for game_id, ids in self.matrix_ids_by_game},
        }


def _read_json(path: Path, *, what: str) -> dict[str, Any]:
    """Read one JSON object, refusing absence loudly rather than defaulting to an empty roster."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{what} {path} is missing. It is gitignored on purpose -- every frame it carries is "
            f"authored benchmark material that would become training data if committed -- so a "
            f"fresh clone does not contain it. Recreate it from the package under docs/scratch/, "
            f"or drop --held-out-extension and run the tracked rosters."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{what} {path} is a {type(payload).__name__}, expected a JSON object.")
    return payload


def _assert_version(payload: Mapping[str, Any], *, path: Path, expected: str) -> None:
    """Refuse a file written to another schema, which would map onto the dataclasses differently."""
    version = payload.get("version")
    if version != expected:
        raise ValueError(
            f"{path} has version {version!r}, expected {expected!r}; the field-for-field mapping "
            f"onto the tracked dataclasses is what a version pins."
        )


def _assert_no_unknown_keys(
    payload: Mapping[str, Any], *, path: Path, allowed: frozenset[str]
) -> None:
    """Refuse a top-level key nothing reads: an author's note here is a note nothing enforces."""
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(
            f"{path} carries top-level keys {unknown} that nothing reads. Authoring notes belong "
            f"in a separate file; a key here reads as material the loader honours and it does not."
        )


def _entry_fields(entry: Mapping[str, Any], *, family: str, path: Path, index: int) -> None:
    """Refuse an entry whose field set is not exactly the family's, `eval_only` named explicitly.

    `eval_only` gets its own sentence because it is the one field an author might add on purpose,
    and adding it asserts the opposite of what the file is for: the loader constructs every frame
    held out, and a file that could say otherwise would put new prose into a training corpus.
    """
    expected = MATRIX_ENTRY_FIELDS if family == FAMILY_MATRIX else RESOURCE_ENTRY_FIELDS
    actual = frozenset(entry)
    if "eval_only" in actual:
        raise ValueError(
            f"{path} {family} entry {index} carries an eval_only field. The staging schema forbids "
            f"it: every extension frame is held out entirely and this loader is what constructs "
            f"them that way, so a writable field could only ever be used to make one trainable."
        )
    if actual != expected:
        raise ValueError(
            f"{path} {family} entry {index} has fields {sorted(actual)}, expected "
            f"{sorted(expected)}. Each entry maps field for field onto a tracked dataclass with no "
            f"renaming and no computed field, which is what makes the landing mechanical."
        )
    for name in sorted(expected):
        value = entry[name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{path} {family} entry {index} has a blank or non-string {name}: {value!r}."
            )


def _assert_id_well_formed(scenario_id: str, *, family: str, path: Path) -> None:
    """Refuse an id that is not the slug every tracked id is, and one that names the literature."""
    if SCENARIO_ID_RE.match(scenario_id) is None:
        raise ValueError(
            f"{path} {family} scenario_id {scenario_id!r} does not match "
            f"{SCENARIO_ID_RE.pattern}; the id is a prompt_id segment and an artifact filename "
            f"segment, so it is held to the lowercase hyphenated spelling every tracked id uses."
        )
    assert_no_loaded_vocabulary(scenario_id)


def _tracked_roster_of(scenario_id: str) -> str | None:
    """Return the tracked roster already carrying this id, or None."""
    for name, roster in (*TRACKED_ROSTERS.items(), *TRACKED_ID_ONLY_ROSTERS.items()):
        if any(scenario.scenario_id == scenario_id for scenario in roster):
            return name
    return None


def _assert_ids_unique(entries: Sequence[tuple[str, str]], *, path: Path) -> None:
    """Refuse an id used twice in the file or already used by a tracked frame.

    Two frames under one id would render two different prompts whose `prompt_id`s collide, and
    every rate is a mean over draws grouped by `prompt_id`, so the collision pools them.
    """
    ids = [scenario_id for _, scenario_id in entries]
    repeated = sorted({scenario_id for scenario_id in ids if ids.count(scenario_id) > 1})
    if repeated:
        raise ValueError(
            f"{path} uses the scenario ids {repeated} more than once. An id keys every prompt_id "
            f"and every record, so two frames under one id pool their draws."
        )
    for family, scenario_id in entries:
        roster = _tracked_roster_of(scenario_id)
        if roster is not None:
            raise ValueError(
                f"{path} {family} scenario_id {scenario_id!r} is already a tracked frame in "
                f"{roster}; every roster's ids are pinned unique across the whole namespace."
            )


def _assert_labels_unique(matrix: Sequence[Scenario], *, path: Path) -> None:
    """Refuse a matrix label string used by another frame, tracked or new.

    The pin `games/tests/test_games_prompts.py`'s `TestScenarioRoster` holds the tracked labelled
    rosters to, extended over the loaded frames: a label in two frames pools them in every
    per-label reading, and nothing about the run would look wrong.
    """
    tracked: dict[str, str] = {
        label.casefold(): scenario.scenario_id
        for roster in LABELLED_TRACKED_ROSTERS
        for scenario in roster
        for label in scenario.labels
    }
    seen: dict[str, str] = {}
    for scenario in matrix:
        for label in scenario.labels:
            folded = label.casefold()
            owner = tracked.get(folded) or seen.get(folded)
            if owner is not None:
                raise ValueError(
                    f"{path} matrix frame {scenario.scenario_id!r} uses the label {label!r}, "
                    f"already used by {owner!r}. A label appearing in two frames pools them in "
                    f"every per-label rate."
                )
            seen[folded] = scenario.scenario_id


def _matrix_renders(scenario: Scenario) -> Iterator[str]:
    """Render one matrix frame every way a matrix row builder would, for the vocabulary gate.

    Both payoff variants, both label mappings and both print orders, under the twin counterpart
    paragraph the trained games carry. The renderer runs
    :func:`games.prompts.assert_no_loaded_vocabulary` on its own output, so iterating this is the
    check; the spec is the twin prisoner's dilemma's because a matrix render differs between games
    only in printed numbers, which cannot introduce a word.
    """
    for payoff_variant in PAYOFF_VARIANTS:
        spec = twin_pd(payoff_variant)
        for coop_label_index in COOP_LABEL_INDICES:
            for label_print_order in LABEL_PRINT_ORDERS:
                yield render_game_prompt(
                    spec,
                    scenario,
                    coop_label_index=coop_label_index,
                    twin_framing=True,
                    label_print_order=label_print_order,
                )


def _dictator_renders(scenario: DictatorScenario) -> Iterator[str]:
    """Render one unilateral-split frame at every registered endowment, as `_dictator_rows` does."""
    for endowment in DICTATOR_ENDOWMENTS:
        yield render_dictator_prompt(
            DictatorSpec(game_id=DICTATOR_GAME_ID, endowment=endowment), scenario
        )


def _trust_renders(scenario: TrustScenario) -> Iterator[str]:
    """Render one trustor frame at every stated return rate, then under the strategy method."""
    for variant in TRUST_PAYOFF_VARIANTS:
        yield render_trust_stated_return_prompt(
            TrustSpec(
                game_id=TRUST_STATED_RETURN_GAME_ID,
                endowment=TRUST_ENDOWMENT,
                multiplier=TRUST_MULTIPLIER,
                stated_return_fraction=trust_return_fraction(variant),
            ),
            scenario,
        )
    yield render_trust_strategy_prompt(
        TrustSpec(
            game_id=TRUST_STRATEGY_METHOD_GAME_ID,
            endowment=TRUST_ENDOWMENT,
            multiplier=TRUST_MULTIPLIER,
        ),
        scenario,
    )


def _trustee_renders(scenario: TrustScenario) -> Iterator[str]:
    """Render one receiving-side frame, the one variant that game carries."""
    yield render_trustee_prompt(
        TrustSpec(
            game_id=TRUSTEE_RETURN_GAME_ID,
            endowment=TRUST_ENDOWMENT,
            multiplier=TRUST_MULTIPLIER,
        ),
        scenario,
    )


def _matrix_scenario(entry: Mapping[str, Any], *, path: Path, index: int) -> Scenario:
    """Build one held-out matrix frame from its staging entry, gates and renders included."""
    _entry_fields(entry, family=FAMILY_MATRIX, path=path, index=index)
    scenario_id = str(entry["scenario_id"])
    _assert_id_well_formed(scenario_id, family=FAMILY_MATRIX, path=path)
    frame = str(entry["frame"])
    assert_no_loaded_vocabulary(frame)
    assert_no_coupling_claims(frame)
    for label in (str(entry["label_a"]), str(entry["label_b"])):
        assert_no_loaded_vocabulary(label)
    scenario = Scenario(
        scenario_id=scenario_id,
        frame=frame,
        label_a=str(entry["label_a"]),
        label_b=str(entry["label_b"]),
        eval_only=True,
    )
    for prompt in _matrix_renders(scenario):
        assert_no_loaded_vocabulary(prompt)
    return scenario


def _resource_entry_parts(
    entry: Mapping[str, Any], *, family: str, path: Path, index: int
) -> tuple[str, str, str]:
    """Check one numeric-answer entry and return its id, frame and resource name.

    The coupling gate does not run on these families, which is the reading
    ``validate_extension.py`` takes too (it reports them advisory there): a consignment frame
    legitimately says what the other side will do with the goods, so the phrases that name a matrix
    counterpart's DECISION process mean something else here. The vocabulary gate is fatal on all
    four families alike.
    """
    _entry_fields(entry, family=family, path=path, index=index)
    scenario_id = str(entry["scenario_id"])
    _assert_id_well_formed(scenario_id, family=family, path=path)
    frame = str(entry["frame"])
    resource = str(entry["resource"])
    assert_no_loaded_vocabulary(frame)
    assert_no_loaded_vocabulary(resource)
    return scenario_id, frame, resource


def _dictator_scenario(entry: Mapping[str, Any], *, path: Path, index: int) -> DictatorScenario:
    """Build one held-out unilateral-split frame from its staging entry, renders included."""
    scenario_id, frame, resource = _resource_entry_parts(
        entry, family=FAMILY_DICTATOR, path=path, index=index
    )
    scenario = DictatorScenario(
        scenario_id=scenario_id, frame=frame, resource=resource, eval_only=True
    )
    for prompt in _dictator_renders(scenario):
        assert_no_loaded_vocabulary(prompt)
    return scenario


def _trust_scenario(
    entry: Mapping[str, Any], *, family: str, path: Path, index: int
) -> TrustScenario:
    """Build one held-out consignment frame, rendered through the side its family names.

    One dataclass, two rosters: a trust frame puts the consignment in the model's own hands and a
    trustee frame puts it in the other side's, so the two render through different renderers.
    """
    scenario_id, frame, resource = _resource_entry_parts(
        entry, family=family, path=path, index=index
    )
    scenario = TrustScenario(
        scenario_id=scenario_id, frame=frame, resource=resource, eval_only=True
    )
    renders = _trust_renders(scenario) if family == FAMILY_TRUST else _trustee_renders(scenario)
    for prompt in renders:
        assert_no_loaded_vocabulary(prompt)
    return scenario


def _matrix_game_caps(
    staging_path: Path, matrix: Sequence[Scenario]
) -> tuple[dict[str, Any], tuple[tuple[str, tuple[str, ...]], ...]]:
    """Read the caps sidecar beside the staging file, returning its payload and the parsed caps.

    The payload comes back too because it goes into the digest: a cap decides what a cell renders
    just as much as a frame does, so two caps may not produce one identity.
    """
    caps_path = staging_path.parent / CAPS_FILENAME
    payload = _read_json(caps_path, what="held-out extension caps file")
    _assert_version(payload, path=caps_path, expected=CAPS_VERSION)
    _assert_no_unknown_keys(
        payload, path=caps_path, allowed=frozenset({"version", MATRIX_GAME_CAPS_KEY})
    )
    raw = payload.get(MATRIX_GAME_CAPS_KEY, {})
    if not isinstance(raw, dict):
        raise TypeError(
            f"{caps_path} {MATRIX_GAME_CAPS_KEY} is a {type(raw).__name__}, expected an object "
            f"mapping each matrix game id to the extension frame ids it renders."
        )
    available = {scenario.scenario_id for scenario in matrix}
    caps: list[tuple[str, tuple[str, ...]]] = []
    for game_id in sorted(raw):
        if game_id not in FRAMEABLE_GAME_IDS:
            raise ValueError(
                f"{caps_path} caps {game_id!r}, which renders no matrix frame roster; the games "
                f"whose renderer reads one are {sorted(FRAMEABLE_GAME_IDS)}."
            )
        listed = raw[game_id]
        if not isinstance(listed, list) or not listed:
            raise ValueError(
                f"{caps_path} caps {game_id!r} with {listed!r}, expected a non-empty list of "
                f"scenario ids. A game that should render no extension frame is simply absent."
            )
        ids = [str(value) for value in listed]
        repeated = sorted({value for value in ids if ids.count(value) > 1})
        if repeated:
            raise ValueError(f"{caps_path} lists {repeated} more than once under {game_id!r}.")
        missing = sorted(set(ids) - available)
        if missing:
            raise ValueError(
                f"{caps_path} caps {game_id!r} to {missing}, which the staging file "
                f"{staging_path} does not carry. A cap names frames by id and never by position, "
                f"so a renamed or dropped frame fails here rather than shrinking the cap."
            )
        caps.append((game_id, tuple(ids)))
    return payload, tuple(caps)


def _digest(staging: Mapping[str, Any], caps: Mapping[str, Any]) -> str:
    """One sha256 over the staging payload and the caps together, canonically serialised."""
    return hashlib.sha256(
        json.dumps(
            {"staging": staging, "caps": caps},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def load_extension(path: Path = STAGING_PATH) -> ExtensionRosters:
    """Load and validate the held-out extension, refusing absence loudly rather than defaulting.

    A default of empty rosters would let a battery launched with ``--held-out-extension`` run the
    narrow tracked banks while its trace recorded the wider ones, which is the one failure this
    whole file exists to make impossible.
    """
    payload = _read_json(path, what="held-out extension staging file")
    _assert_version(payload, path=path, expected=EXTENSION_VERSION)
    _assert_no_unknown_keys(payload, path=path, allowed=frozenset({"version", *EXTENSION_FAMILIES}))
    matrix: list[Scenario] = []
    dictator: list[DictatorScenario] = []
    trust: list[TrustScenario] = []
    trustee: list[TrustScenario] = []
    identified: list[tuple[str, str]] = []
    for family in EXTENSION_FAMILIES:
        entries = payload.get(family, [])
        if not isinstance(entries, list):
            raise TypeError(
                f"{path} {family} is a {type(entries).__name__}, expected a list of entries."
            )
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise TypeError(
                    f"{path} {family} entry {index} is a {type(entry).__name__}, expected an object."
                )
            identified.append((family, str(entry.get("scenario_id", ""))))
            if family == FAMILY_MATRIX:
                matrix.append(_matrix_scenario(entry, path=path, index=index))
            elif family == FAMILY_DICTATOR:
                dictator.append(_dictator_scenario(entry, path=path, index=index))
            elif family == FAMILY_TRUST:
                trust.append(_trust_scenario(entry, family=family, path=path, index=index))
            else:
                trustee.append(_trust_scenario(entry, family=family, path=path, index=index))
    _assert_ids_unique(identified, path=path)
    _assert_labels_unique(matrix, path=path)
    caps_payload, caps = _matrix_game_caps(path, matrix)
    rosters = ExtensionRosters(
        matrix=tuple(matrix),
        dictator=tuple(dictator),
        trust=tuple(trust),
        trustee=tuple(trustee),
        matrix_ids_by_game=caps,
        digest=_digest(payload, caps_payload),
    )
    logger.info(
        f"loaded held-out extension, {path=} digest={rosters.digest[:16]} "
        f"n_matrix={len(rosters.matrix)} n_dictator={len(rosters.dictator)} "
        f"n_trust={len(rosters.trust)} n_trustee={len(rosters.trustee)} "
        f"matrix_caps={ {game_id: len(ids) for game_id, ids in caps} }"
    )
    return rosters
