"""Find training items whose statement duplicates a held-out item across source datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import tempfile
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from reward_hacking.aisi_rh_envs_data import (
    AISI_SOURCES,
)
from reward_hacking.aisi_rh_envs_data import (
    DATA_PATH as AISI_DATA_PATH,
)
from reward_hacking.aisi_rh_envs_data import (
    load_problems as load_aisi_problems,
)
from reward_hacking.aisi_rh_envs_split import deterministic_source_splits
from reward_hacking.ilcb_data import load_problems as load_ilcb_problems
from reward_hacking.ilcb_split import deterministic_problem_split

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

DedupSource = Literal["aisi:apps", "aisi:codecontests", "aisi:mbpp", "aisi:humaneval", "ilcb"]
DataPartition = Literal["train", "held-out"]
KNOWN_SOURCES: tuple[DedupSource, ...] = (
    "aisi:apps",
    "aisi:codecontests",
    "aisi:mbpp",
    "aisi:humaneval",
    "ilcb",
)
NORMALIZATION_ALGORITHM = "nfkc-casefold-word-space-v1"
HASH_ALGORITHM = "sha256(normalized-full-or-400-prefix-statement-v2)"
PREFIX_LENGTH = 400
DEFAULT_EXCLUSION_LIST_PATH = AISI_DATA_PATH.parent / "statement_dedup_exclusions.json"


@dataclass(frozen=True, slots=True)
class StatementItem:
    """One source item with its stable ID, split membership and statement text."""

    source: DedupSource
    item_id: str
    partition: DataPartition
    statement: str


@dataclass(frozen=True, slots=True)
class ExclusionReport:
    """IDs removed from training and counts that downstream cards can report."""

    excluded_item_ids: frozenset[str]
    excluded_count_by_source: dict[str, int]
    training_item_count: int
    held_out_item_count: int
    ilcb_aisi_match_count: int
    apps_codecontests_shared_prefix_count: int
    apps_codecontests_prefix_pair_count: int
    apps_codecontests_matched_codecontests_items: int


def normalize_statement(statement: str) -> str:
    """Normalize text and punctuation consistently before hashing a task statement."""
    if not statement.strip():
        raise ValueError("statement must be a nonempty string")
    compatibility_text = unicodedata.normalize("NFKC", statement).casefold()
    alphanumeric_text = "".join(
        character if character.isalnum() or character == "_" or character.isspace() else " "
        for character in compatibility_text
    )
    normalized_statement = " ".join(alphanumeric_text.split())
    if not normalized_statement:
        raise ValueError("statement must contain normalized word characters")
    return normalized_statement


def statement_hash(statement: str) -> str:
    """Return the full normalized-statement fingerprint used for exclusion decisions."""
    return _hash_normalized_statement(normalize_statement(statement))


def _hash_normalized_statement(normalized_statement: str) -> str:
    """Hash text that has already passed through :func:`normalize_statement`."""
    return hashlib.sha256(normalized_statement.encode("utf-8")).hexdigest()


def analyze_training_exclusions(items: Sequence[StatementItem]) -> ExclusionReport:
    """Exclude train items matching held-out full statements or normalized prefixes."""
    if not items:
        raise ValueError("statement items must not be empty")

    item_ids: set[str] = set()
    full_hash_by_item_id: dict[str, str] = {}
    prefix_hash_by_item_id: dict[str, str] = {}
    held_out_hashes: set[str] = set()
    held_out_prefix_hashes: set[str] = set()
    statement_hashes_by_source: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    prefix_hashes_by_source: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    training_item_count = 0
    held_out_item_count = 0

    for item in items:
        if item.source not in KNOWN_SOURCES:
            raise ValueError(f"unsupported statement source {item.source!r}")
        if item.partition not in {"train", "held-out"}:
            raise ValueError(f"unsupported statement partition {item.partition!r}")
        if not item.item_id or item.item_id in item_ids:
            raise ValueError("statement item IDs must be nonempty and unique")
        item_ids.add(item.item_id)

        normalized_statement = normalize_statement(item.statement)
        full_hash = _hash_normalized_statement(normalized_statement)
        prefix_hash = _hash_normalized_statement(normalized_statement[:PREFIX_LENGTH])
        full_hash_by_item_id[item.item_id] = full_hash
        prefix_hash_by_item_id[item.item_id] = prefix_hash
        statement_hashes_by_source[item.source][full_hash].add(item.item_id)
        prefix_hashes_by_source[item.source][prefix_hash].add(item.item_id)
        if item.partition == "held-out":
            held_out_hashes.add(full_hash)
            held_out_prefix_hashes.add(prefix_hash)
            held_out_item_count += 1
        else:
            training_item_count += 1

    excluded_items = tuple(
        item
        for item in items
        if item.partition == "train"
        and (
            full_hash_by_item_id[item.item_id] in held_out_hashes
            or prefix_hash_by_item_id[item.item_id] in held_out_prefix_hashes
        )
    )
    excluded_item_ids = frozenset(item.item_id for item in excluded_items)
    excluded_count_by_source = {
        source: sum(item.source == source for item in excluded_items) for source in KNOWN_SOURCES
    }

    aisi_hashes: dict[str, set[str]] = defaultdict(set)
    for source, hashes in statement_hashes_by_source.items():
        if source.startswith("aisi:"):
            for full_hash, source_item_ids in hashes.items():
                aisi_hashes[full_hash].update(source_item_ids)
    ilcb_hashes = statement_hashes_by_source["ilcb"]
    ilcb_aisi_match_count = sum(
        len(ilcb_hashes[full_hash]) * len(aisi_item_ids)
        for full_hash, aisi_item_ids in aisi_hashes.items()
        if full_hash in ilcb_hashes
    )

    apps_prefixes = prefix_hashes_by_source["aisi:apps"]
    codecontests_prefixes = prefix_hashes_by_source["aisi:codecontests"]
    shared_prefix_hashes = apps_prefixes.keys() & codecontests_prefixes.keys()
    apps_codecontests_prefix_pair_count = sum(
        len(apps_prefixes[prefix_hash]) * len(codecontests_prefixes[prefix_hash])
        for prefix_hash in shared_prefix_hashes
    )
    apps_codecontests_matched_codecontests_items = len(
        set().union(*(codecontests_prefixes[prefix_hash] for prefix_hash in shared_prefix_hashes))
    )

    return ExclusionReport(
        excluded_item_ids=excluded_item_ids,
        excluded_count_by_source=excluded_count_by_source,
        training_item_count=training_item_count,
        held_out_item_count=held_out_item_count,
        ilcb_aisi_match_count=ilcb_aisi_match_count,
        apps_codecontests_shared_prefix_count=len(shared_prefix_hashes),
        apps_codecontests_prefix_pair_count=apps_codecontests_prefix_pair_count,
        apps_codecontests_matched_codecontests_items=apps_codecontests_matched_codecontests_items,
    )


def build_local_exclusion_report() -> ExclusionReport:
    """Load all AISI and ILCB sources, assign their existing splits and analyze overlap."""
    aisi_problems = load_aisi_problems()
    dedup_source_by_aisi_source: dict[str, DedupSource] = {
        "apps": "aisi:apps",
        "codecontests": "aisi:codecontests",
        "mbpp": "aisi:mbpp",
        "humaneval": "aisi:humaneval",
    }
    aisi_ids_by_source = {
        source: tuple(
            sorted(problem.problem_id for problem in aisi_problems if problem.source == source)
        )
        for source in AISI_SOURCES
    }
    aisi_splits = deterministic_source_splits(aisi_ids_by_source)
    aisi_items = tuple(
        StatementItem(
            source=dedup_source_by_aisi_source[problem.source],
            item_id=f"aisi:{problem.problem_id}",
            partition=(
                "held-out"
                if problem.problem_id in aisi_splits[problem.source]["held-out"]
                else "train"
            ),
            statement=problem.statement,
        )
        for problem in aisi_problems
    )

    ilcb_statements: dict[str, str] = {}
    for problem in load_ilcb_problems():
        existing_statement = ilcb_statements.setdefault(problem.task_id, problem.prompt)
        if normalize_statement(existing_statement) != normalize_statement(problem.prompt):
            raise ValueError("ILCB variants for one item must share a normalized statement")
    ilcb_splits = deterministic_problem_split(tuple(sorted(ilcb_statements)))
    ilcb_items = tuple(
        StatementItem(
            source="ilcb",
            item_id=f"ilcb:{item_id}",
            partition="held-out" if item_id in ilcb_splits["held-out"] else "train",
            statement=ilcb_statements[item_id],
        )
        for item_id in sorted(ilcb_statements)
    )
    return analyze_training_exclusions((*aisi_items, *ilcb_items))


def filter_excluded_task_ids(
    task_ids: Sequence[str],
    *,
    families_by_task: Mapping[str, str],
    item_ids_by_task: Mapping[str, str],
    excluded_item_ids: frozenset[str],
) -> tuple[str, ...]:
    """Filter selected generator task IDs using the stable source-qualified item IDs."""
    selected_task_ids: list[str] = []
    for task_id in task_ids:
        if task_id not in families_by_task:
            raise KeyError(f"selected task {task_id!r} is missing its family")
        family = families_by_task[task_id]
        if family not in {"aisi", "ilcb"}:
            selected_task_ids.append(task_id)
            continue
        if task_id not in item_ids_by_task:
            raise KeyError(f"selected task {task_id!r} is missing its item identity")
        item_id = item_ids_by_task[task_id]
        if not item_id:
            raise ValueError(f"selected task {task_id!r} has an empty item identity")
        excluded_id = f"{family}:{item_id}"
        if excluded_id not in excluded_item_ids:
            selected_task_ids.append(task_id)
    return tuple(selected_task_ids)


def write_exclusion_ids(
    output_path: Path,
    excluded_item_ids: Sequence[str] | frozenset[str],
) -> None:
    """Atomically write a sorted JSON list containing item IDs and no statement text."""
    if any(not item_id for item_id in excluded_item_ids):
        raise ValueError("excluded item IDs must be nonempty")
    sorted_item_ids = sorted(excluded_item_ids)
    if len(sorted_item_ids) != len(set(sorted_item_ids)):
        raise ValueError("excluded item IDs must be unique")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)
            output_file.write(json.dumps(sorted_item_ids, indent=2) + "\n")
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_exclusion_ids(input_path: Path = DEFAULT_EXCLUSION_LIST_PATH) -> frozenset[str]:
    """Load and validate the generated ID-only exclusion list."""
    try:
        value: object = json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"cannot read the statement dedup exclusion list at {input_path}"
        ) from error
    if not isinstance(value, list) or any(
        not isinstance(item_id, str) or not item_id for item_id in value
    ):
        raise ValueError("statement dedup exclusion list must be a list of nonempty item IDs")
    item_ids = tuple(value)
    if tuple(sorted(set(item_ids))) != item_ids:
        raise ValueError("statement dedup exclusion list must contain sorted unique item IDs")
    return frozenset(item_ids)


def prepare_training_exclusions(
    output_path: Path = DEFAULT_EXCLUSION_LIST_PATH,
) -> ExclusionReport:
    """Build the current exclusion list and log its counts for generation and dataset cards."""
    report = build_local_exclusion_report()
    write_exclusion_ids(output_path, report.excluded_item_ids)
    logger.info(
        "statement dedup normalization=%s hash=%s train_items=%d held_out_items=%d "
        "excluded=%d by_source=%s "
        "ilcb_aisi_matches=%d apps_codecontests_shared_400_prefixes=%d "
        "apps_codecontests_prefix_pairs=%d",
        NORMALIZATION_ALGORITHM,
        HASH_ALGORITHM,
        report.training_item_count,
        report.held_out_item_count,
        len(report.excluded_item_ids),
        report.excluded_count_by_source,
        report.ilcb_aisi_match_count,
        report.apps_codecontests_shared_prefix_count,
        report.apps_codecontests_prefix_pair_count,
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Generate the local-only ID exclusion list from the baked case files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_EXCLUSION_LIST_PATH)
    arguments = parser.parse_args(argv)
    prepare_training_exclusions(arguments.output)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
