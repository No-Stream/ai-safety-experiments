"""Load every authored declaration artifact through the checks the grader will hold it to.

A load-time refusal is a claim about what every artifact on disk already looks like, and nothing was
checking that claim: ruff skips these artifacts because they are gitignored, the type checker excludes
the path by name, and no consumer loads them yet. Load-level failures only, never style, inheriting
that line from `tests/test_scratch_compiles.py`.

The design record is `docs/declaration-gate.md`: why the gate is prospective, why the schema
translation is its own failure surface, why the untranslated key set is the tripwire for the gate's
own incompleteness, why both denominators are floored rather than printed, and why it calls a private
validator.
"""

import json
import logging
from pathlib import Path

import pytest

from reward_hacking.recoverybench.answers import AnswerShape
from reward_hacking.recoverybench.arms import Arm
from reward_hacking.recoverybench.decision import ConstantMeaning, SymbolDomain
from reward_hacking.recoverybench.items import (
    Domain,
    FlawType,
    GradingMode,
    ItemValidationError,
    RecoveryItem,
    _validate_declaration,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DECLARATIONS_ROOT = REPO_ROOT / "docs" / "scratch" / "grading-redesign" / "declarations"

# Floors rather than exact counts; see docs/declaration-gate.md on floored denominators.
KNOWN_AUTHORED_FILES = 3
KNOWN_AUTHORED_ENTRIES = 47

# Measured across all three files, not assumed; see docs/declaration-gate.md on the untranslated key set.
KNOWN_ENTRY_KEYS = frozenset(
    {
        "default",
        "per_symbol",
        "constants",
        "why",
        "parser_artifact_symbols",
        "auth_defect",
        "declaration_licensed_by",
    }
)

# The only default the grader can honour; see docs/declaration-gate.md on `default`.
REPRESENTABLE_DEFAULT = SymbolDomain.REAL

UNTRANSLATED_KEY_HEADLINE = (
    "authored entries carry keys this gate does not translate, so a consumer reading one of them "
    "would be unchecked here. Either extend the translation and its sabotage, or add the key to "
    "KNOWN_ENTRY_KEYS as prose:"
)

logger = logging.getLogger(__name__)


def _authored_files() -> list[Path]:
    """Every authored declaration artifact, sorted so a failure list reads the same each run."""
    return sorted(DECLARATIONS_ROOT.glob("authored_*.json"))


def _structural_failure(key: str, entry: object) -> str | None:
    """The shape complaints that precede any coercion: `None` when the entry is worth coercing."""
    if not isinstance(entry, dict):
        return f"entry is {type(entry).__name__}, not an object"
    if "|" not in key:
        return "key carries no 'corpus|item_id' separator, so a loader's split-unpack would raise"
    if entry.get("default") is None:
        return "missing required key 'default'"
    if not isinstance(entry.get("per_symbol"), dict):
        return "per_symbol is missing or not an object"
    if not isinstance(entry.get("constants", {}), dict):
        return "constants is not an object"
    return None


def _default_failure(default_raw: object) -> str | None:
    """Why this entry's `default` cannot be honoured, or `None` when it is the representable one."""
    try:
        default = SymbolDomain(default_raw)
    except ValueError as exc:
        return f"default: {exc}"
    if default is not REPRESENTABLE_DEFAULT:
        return (
            f"default {default.value!r} has no RecoveryItem representation: declaration() never "
            f"passes a default, so the grader would read undeclared symbols as "
            f"{REPRESENTABLE_DEFAULT.value!r} whatever this artifact says"
        )
    return None


def _coerced_maps(
    per_symbol_raw: dict[object, object], constants_raw: dict[object, object]
) -> tuple[str | None, dict[str, SymbolDomain], dict[str, ConstantMeaning]]:
    """Coerce both maps through their enums, one name at a time so a failure names the symbol.

    A comprehension would report only the offending value, and an entry can declare a dozen
    symbols; `'positive_rael' is not a valid SymbolDomain` then leaves a reader grepping.
    """
    per_symbol: dict[str, SymbolDomain] = {}
    constants: dict[str, ConstantMeaning] = {}
    for name, domain in per_symbol_raw.items():
        try:
            per_symbol[str(name)] = SymbolDomain(domain)
        except ValueError as exc:
            return f"per_symbol[{str(name)!r}]: {exc}", per_symbol, constants
    for name, meaning in constants_raw.items():
        try:
            constants[str(name)] = ConstantMeaning(meaning)
        except ValueError as exc:
            return f"constants[{str(name)!r}]: {exc}", per_symbol, constants
    return None, per_symbol, constants


def _entry_failure(key: str, entry: object) -> str | None:
    """Load one declaration entry as the grader would: `None` if it loads, else why it cannot.

    Mirrors the loading a consumer must perform -- coerce the domain and constant strings through
    their enums (an illegal name raises rather than coercing), then hold the result to the
    production validator. The enum coercion is a transcription of the scratch prototype's loader
    (`grading-redesign/proto/declarations.py`), which a tracked test cannot import because the
    scratch tree is absent on a fresh clone; if that loader's coercion ever changes shape, this is
    the transcription to re-check.
    """
    structural = _structural_failure(key, entry)
    if structural is not None or not isinstance(entry, dict):
        return structural
    default_failure = _default_failure(entry["default"])
    if default_failure is not None:
        return default_failure
    coercion, per_symbol, constants = _coerced_maps(entry["per_symbol"], entry.get("constants", {}))
    if coercion is not None:
        return coercion
    item = RecoveryItem(
        item_id=key.split("|", 1)[1],
        domain=Domain.SCIENCE,
        flaw_type=FlawType.WRONG_METHOD,
        grading_mode=GradingMode.CLOSED_ANSWER,
        arms={Arm.BARE: "bare prompt", Arm.CORRUPTED: "corrupted prompt"},
        source="declaration-data-gate",
        license_terms="research use",
        answer_shape=AnswerShape.EXPRESSION,
        symbol_domains=per_symbol,
        reserved_name_meanings=constants,
    )
    try:
        _validate_declaration(item)
    except ItemValidationError as exc:
        return str(exc)
    return None


def _file_failures(path: Path) -> tuple[int, list[str]]:
    """Sweep one artifact: (entries examined, one report line per entry that cannot load)."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return 0, [f"{path.name}: unreadable as JSON: {exc}"]
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, dict):
        return 0, [
            f"{path.name}: no top-level 'items' object, so a loader would raise before any entry"
        ]
    failures = [
        f"{path.name}: {key}: {reason}"
        for key, entry in sorted(items.items())
        if (reason := _entry_failure(key, entry)) is not None
    ]
    return len(items), failures


def _untranslated_keys(path: Path) -> dict[str, list[str]]:
    """Per entry, the keys this gate does not know about, keyed by the entry's own key.

    Separate from :func:`_file_failures` on purpose. An unknown key is not a reason to call an
    artifact unloadable -- today's four untranslated ones load fine -- it is a reason to make
    somebody look at whether the translation should grow. So it reports as its own gate rather than
    as a load failure, and an unreadable file is silently no contribution here because
    :func:`_file_failures` is already the thing that fails loudly on it.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, dict):
        return {}
    unknown: dict[str, list[str]] = {}
    for key, entry in sorted(items.items()):
        if not isinstance(entry, dict):
            continue
        extra = sorted(set(entry) - KNOWN_ENTRY_KEYS)
        if extra:
            unknown[f"{path.name}: {key}"] = extra
    return unknown


# One synthetic entry per violation class, each the minimal shape its validator rejects.
CLEAN_ENTRY = {"default": "real", "per_symbol": {"m": "positive_real"}, "constants": {}}
CONTRADICTION_ENTRY = {
    "default": "real",
    "per_symbol": {"e": "positive_real"},
    "constants": {"e": "euler_number"},
}
ILLEGAL_DOMAIN_ENTRY = {"default": "real", "per_symbol": {"m": "positive_rael"}, "constants": {}}
ILLEGAL_MEANING_ENTRY = {"default": "real", "per_symbol": {}, "constants": {"e": "eulers_number"}}
UNREPRESENTABLE_DEFAULT_ENTRY = {"default": "positive_real", "per_symbol": {}, "constants": {}}
MISSING_KEY_ENTRY = {"default": "real"}


class TestTheGateRejectsWhatItExistsToCatch:
    """The gate's own sabotage, kept in the suite instead of run once and remembered.

    Every rejection drives `_entry_failure` / `_file_failures` rather than a private copy, on
    test_scratch_compiles.py's reasoning: neuter the real code path and these are what go red. The
    contradiction case additionally pins the *message* to the production validator's wording, which
    is what makes the schema translation's completeness a tested property -- drop the
    `constants` -> `reserved_name_meanings` half and the entry stops reaching the check, the
    production wording never comes back, and this test fails; same for the `per_symbol` half.
    """

    def test_a_clean_entry_loads(self) -> None:
        assert _entry_failure("corpus|clean", CLEAN_ENTRY) is None

    def test_the_fifth_gap_contradiction_is_refused_by_the_production_check(self) -> None:
        reason = _entry_failure("corpus|fifth-gap", CONTRADICTION_ENTRY)
        assert reason is not None, (
            "the e-declared-both-ways artifact loaded, so one half of the per_symbol/constants "
            "translation has been dropped and the contradiction check is no longer reachable"
        )
        assert "constant meaning and a symbol domain" in reason, reason
        assert "'e'" in reason, reason

    def test_an_illegal_domain_name_is_refused(self) -> None:
        reason = _entry_failure("corpus|typo-domain", ILLEGAL_DOMAIN_ENTRY)
        assert reason is not None
        assert "positive_rael" in reason, reason

    def test_an_illegal_constant_meaning_is_refused(self) -> None:
        reason = _entry_failure("corpus|typo-meaning", ILLEGAL_MEANING_ENTRY)
        assert reason is not None
        assert "eulers_number" in reason, reason

    def test_an_unrepresentable_default_is_refused(self) -> None:
        reason = _entry_failure("corpus|wide-default", UNREPRESENTABLE_DEFAULT_ENTRY)
        assert reason is not None
        assert "no RecoveryItem representation" in reason, reason

    def test_a_missing_required_key_is_refused(self) -> None:
        reason = _entry_failure("corpus|truncated", MISSING_KEY_ENTRY)
        assert reason is not None
        assert "per_symbol" in reason, reason

    def test_a_key_without_a_separator_is_refused(self) -> None:
        assert _entry_failure("no-separator", CLEAN_ENTRY) is not None

    def test_a_coercion_failure_names_the_symbol_not_only_the_value(self) -> None:
        reason = _entry_failure("corpus|typo-domain", ILLEGAL_DOMAIN_ENTRY)
        assert reason is not None
        assert "per_symbol['m']" in reason, reason

    def test_an_untranslated_key_is_reported(self, tmp_path: Path) -> None:
        """The tripwire's own sabotage: the range key that would be silently unchecked."""
        grown = tmp_path / "authored_grown_a_range.json"
        entry = dict(CLEAN_ENTRY) | {"per_symbol_range": {"m": [1.0, 5.0]}}
        grown.write_text(json.dumps({"items": {"corpus|grown": entry}}), encoding="utf-8")

        unknown = _untranslated_keys(grown)

        assert unknown, (
            "an entry carrying per_symbol_range was reported as fully translated, so a consumer "
            "field this gate never passes would go unchecked -- both sampling-range refusals are "
            "unreachable from here by construction and this tripwire is what makes that visible"
        )
        assert unknown == {"authored_grown_a_range.json: corpus|grown": ["per_symbol_range"]}, (
            unknown
        )

    def test_todays_untranslated_keys_are_not_reported(self, tmp_path: Path) -> None:
        """The opposite vacuity: a tripwire that fires on the prose keys would be turned off."""
        prose = tmp_path / "authored_prose.json"
        entry = dict(CLEAN_ENTRY) | {
            "why": {"m": "a mass"},
            "parser_artifact_symbols": [],
            "auth_defect": "none",
            "declaration_licensed_by": "somewhere",
        }
        prose.write_text(json.dumps({"items": {"corpus|prose": entry}}), encoding="utf-8")

        assert _untranslated_keys(prose) == {}

    def test_an_unparseable_file_is_reported_not_raised(self, tmp_path: Path) -> None:
        mangled = tmp_path / "authored_mangled.json"
        mangled.write_text('{"items": {', encoding="utf-8")
        examined, failures = _file_failures(mangled)
        assert examined == 0
        assert len(failures) == 1, failures
        assert "unreadable as JSON" in failures[0], failures

    def test_a_file_without_an_items_object_is_reported(self, tmp_path: Path) -> None:
        hollow = tmp_path / "authored_hollow.json"
        hollow.write_text('{"authored_by": "nobody"}', encoding="utf-8")
        examined, failures = _file_failures(hollow)
        assert examined == 0
        assert len(failures) == 1, failures
        assert "'items'" in failures[0], failures


@pytest.mark.skipif(
    not DECLARATIONS_ROOT.is_dir(),
    reason=(
        "docs/scratch/grading-redesign/declarations is gitignored and absent here, so there are "
        "0 authored declaration artifacts to sweep. -ra in addopts prints this on every run, so a "
        "sweep that covers nothing stays visible."
    ),
)
class TestEveryAuthoredDeclarationLoads:
    """The sweep itself, over whatever the local declarations tree happens to hold.

    Local, hand-edited data reddening the shared suite is this gate's cost, accepted knowingly:
    the failure tier is "cannot load", which is broken on any box, not a style opinion about this
    one. A red here on an entry another session is mid-authoring is a real finding about that
    entry, to be reported to its author -- never grounds to loosen the gate.
    """

    def test_the_sweep_has_files_to_check(self) -> None:
        files = _authored_files()
        assert len(files) >= KNOWN_AUTHORED_FILES, (
            f"only {len(files)} authored_*.json under {DECLARATIONS_ROOT}, where "
            f"{KNOWN_AUTHORED_FILES} existed when this gate landed. A moved or renamed artifact "
            f"is invisible to the sweep, which would otherwise pass green over nothing; found: "
            f"{[path.name for path in files]}"
        )

    def test_no_authored_entry_carries_an_untranslated_key(self) -> None:
        unknown: dict[str, list[str]] = {}
        for path in _authored_files():
            unknown.update(_untranslated_keys(path))
        assert not unknown, "\n".join(
            [
                UNTRANSLATED_KEY_HEADLINE,
                *(f"  {where}: {sorted(keys)}" for where, keys in sorted(unknown.items())),
            ]
        )

    def test_every_authored_declaration_loads(self) -> None:
        files = _authored_files()
        examined = 0
        failures: list[str] = []
        for path in files:
            entries, file_failures = _file_failures(path)
            examined += entries
            failures.extend(file_failures)
        assert examined >= KNOWN_AUTHORED_ENTRIES, (
            f"only {examined} declaration entries examined across {len(files)} files, where "
            f"{KNOWN_AUTHORED_ENTRIES} existed when this gate landed. A truncated or emptied "
            f"artifact collapses the denominator this test reports, which would otherwise let it "
            f"pass green over almost nothing"
        )
        logger.info(
            "declaration data gate: %d files, %d entries examined, %d cannot load",
            len(files),
            examined,
            len(failures),
        )
        headline = (
            f"{len(failures)} declaration(s) cannot load, out of {examined} entries "
            f"examined across {len(files)} files:"
        )
        assert not failures, "\n".join(
            [
                headline,
                *failures,
            ]
        )
