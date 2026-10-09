"""Explicit channel arms and configurable named channel transfer splits."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import TYPE_CHECKING, cast

from reward_hacking.harness.leak_channels import (
    ALL_COMPATIBLE_CHANNELS,
    CHANNEL_CONFLICTS,
    CHANNELS,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@dataclass(frozen=True)
class _LeakChannelSplit:
    channels_by_partition: dict[str, tuple[str, ...]]
    eval_pairs: tuple[tuple[str, str], ...]


_EVAL_PAIR_LENGTH = 2


def parse_channels(value: str) -> tuple[str, ...]:
    """Resolve the open set, closed control, or an explicit channel arm."""
    if value == "open":
        return tuple(sorted(ALL_COMPATIBLE_CHANNELS))
    if value == "closed":
        return ()
    requested = value.split(",")
    if any(not channel for channel in requested) or len(set(requested)) != len(requested):
        raise argparse.ArgumentTypeError("leak-channels requires unique comma-separated channels")
    if set(requested) - set(CHANNELS):
        raise argparse.ArgumentTypeError("leak-channels contains unknown channels")
    if any(
        channel in requested and excluded & set(requested)
        for channel, excluded in CHANNEL_CONFLICTS.items()
    ):
        raise argparse.ArgumentTypeError("leak-channels contains channels that cannot coexist")
    return tuple(sorted(requested))


class _LeakChannelsAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        del option_string
        if not isinstance(values, str):
            parser.error("leak-channels must be a string")
        try:
            parsed_channels = parse_channels(values)
        except argparse.ArgumentTypeError as error:
            parser.error(str(error))
        setattr(namespace, self.dest, parsed_channels)
        namespace.leak_channels_open = values == "open"


def parse_repos(value: str) -> tuple[str, ...]:
    """Parse the repository allowlist used for round-robin task selection."""
    repos = value.split(",")
    if any(not repo for repo in repos) or len(set(repos)) != len(repos):
        raise argparse.ArgumentTypeError("leak-repos requires unique comma-separated repositories")
    return tuple(repos)


def add_leak_arguments(parser: argparse.ArgumentParser) -> None:
    """Install the same leak arm interface in all organism entrypoints."""
    default_max_turns = parser.get_default("max_turns")
    if default_max_turns is not None:
        parser.set_defaults(leak_non_leak_max_turns_default=default_max_turns, max_turns=None)
    parser.add_argument(
        "--leak-channels",
        action=_LeakChannelsAction,
        default=None,
        metavar="open|closed|CHANNEL,...",
    )
    parser.set_defaults(leak_channels_open=False)
    parser.add_argument("--leak-instruction", choices=("none", "vague", "specific"), default="none")
    parser.add_argument("--leak-repos", type=parse_repos)
    parser.add_argument(
        "--leak-assets",
        type=Path,
        default=None,
        help="explicit path to the prepared leak asset index (required for r2e_leak)",
    )
    parser.add_argument("--leak-channel-split", help="named split in --leak-channel-splits")
    parser.add_argument(
        "--leak-channel-splits",
        type=Path,
        help="JSON object mapping split names to train/held-out channel lists",
    )
    parser.add_argument(
        "--leak-channel-per-episode",
        action="store_true",
        help="select each held-out task/channel pair from eval_pairs in the named split",
    )


def resolve_leak_options(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, partition: str
) -> None:
    """Validate the named split and prevent training arms from opening held-out channels."""
    _resolve_family_defaults(args, parser)
    args.leak_channels_open = args.leak_channels is None or args.leak_channels_open
    if (args.leak_channel_split is None) != (args.leak_channel_splits is None):
        parser.error("leak-channel-split and leak-channel-splits must be provided together")
    args.leak_eval_pairs = ()
    _validate_per_episode_option(args, parser, partition)
    if args.leak_channel_split is None:
        args.leak_channels = (
            parse_channels("open") if args.leak_channels is None else args.leak_channels
        )
        return
    split = _load_channel_split(args.leak_channel_splits, args.leak_channel_split, parser)
    channels_by_partition = split.channels_by_partition
    if set(channels_by_partition["train"]) & set(channels_by_partition["held-out"]):
        parser.error("train and held-out leak channel sets must be disjoint")
    if args.leak_channel_per_episode:
        _select_eval_pair_channels(args, parser, split)
        return
    selected = _selected_named_channels(args, parser, channels_by_partition[partition], partition)
    try:
        args.leak_channels = parse_channels(",".join(selected)) if selected else ()
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))


def _resolve_family_defaults(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    includes_leak_family = "r2e_leak" in (args.family_entries or ())
    if includes_leak_family and args.leak_assets is None:
        parser.error("--leak-assets is required when selecting the r2e_leak family")
    if not hasattr(args, "max_turns") or args.max_turns is not None:
        vars(args).pop("leak_non_leak_max_turns_default", None)
        return
    if includes_leak_family:
        args.max_turns = 30
    elif "leak_non_leak_max_turns_default" in vars(args):
        args.max_turns = vars(args)["leak_non_leak_max_turns_default"]
    vars(args).pop("leak_non_leak_max_turns_default", None)


def _validate_per_episode_option(
    args: argparse.Namespace, parser: argparse.ArgumentParser, partition: str
) -> None:
    if not args.leak_channel_per_episode:
        return
    if partition != "held-out":
        parser.error("leak-channel-per-episode is only supported for held-out leak tasks")
    if args.leak_channel_split is None:
        parser.error("leak-channel-per-episode requires a named leak channel split")


def _select_eval_pair_channels(
    args: argparse.Namespace, parser: argparse.ArgumentParser, split: _LeakChannelSplit
) -> None:
    if not split.eval_pairs:
        parser.error("leak-channel-per-episode requires at least one eval_pairs entry")
    args.leak_eval_pairs = split.eval_pairs
    selected_pair_channels = tuple(sorted({channel for _task_id, channel in split.eval_pairs}))
    if args.leak_channels is not None and set(args.leak_channels) != set(selected_pair_channels):
        parser.error("explicit leak channels must match the named split's eval_pairs")
    args.leak_channels = selected_pair_channels
    args.leak_channels_open = False


def _selected_named_channels(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    allowed_channels: tuple[str, ...],
    partition: str,
) -> tuple[str, ...]:
    if args.leak_channels_open:
        selected = tuple(
            channel for channel in allowed_channels if channel in ALL_COMPATIBLE_CHANNELS
        )
    else:
        if args.leak_channels is None:
            parser.error("explicit leak channel selection is missing")
        selected = args.leak_channels
    if set(selected) - set(allowed_channels):
        parser.error(f"leak channel arm includes channels outside the {partition} split")
    return selected


def _load_channel_split(
    config_path: Path, split_name: str, parser: argparse.ArgumentParser
) -> _LeakChannelSplit:
    raw: object = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or split_name not in raw:
        parser.error("leak-channel-splits must contain the requested split name")
    raw_object = cast("dict[str, object]", raw)
    split: object = raw_object[split_name]
    if not isinstance(split, dict) or set(split) != {"train", "held-out"}:
        parser.error("each leak channel split requires train and held-out lists")
    channels_by_partition: dict[str, tuple[str, ...]] = {}
    for key, values in cast("dict[str, object]", split).items():
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            parser.error("leak channel split values must be lists of channel names")
        channel_names = cast("list[str]", values)
        if len(set(channel_names)) != len(channel_names) or set(channel_names) - set(CHANNELS):
            parser.error("leak channel split lists require unique known channel names")
        channels_by_partition[key] = tuple(sorted(channel_names))
    eval_pairs = _parse_eval_pairs(
        raw_object.get("eval_pairs", []), channels_by_partition["held-out"], parser
    )
    return _LeakChannelSplit(channels_by_partition, eval_pairs)


def _parse_eval_pairs(
    raw_eval_pairs: object, heldout_channels: tuple[str, ...], parser: argparse.ArgumentParser
) -> tuple[tuple[str, str], ...]:
    if not isinstance(raw_eval_pairs, list):
        parser.error("eval_pairs must be a list of [task_id, channel] pairs")
    eval_pairs: list[tuple[str, str]] = []
    for value in raw_eval_pairs:
        if (
            not isinstance(value, list)
            or len(value) != _EVAL_PAIR_LENGTH
            or any(not isinstance(part, str) or not part for part in value)
        ):
            parser.error("eval_pairs must be a list of [task_id, channel] pairs")
        task_id, channel = cast("list[str]", value)
        if channel not in heldout_channels:
            parser.error("eval_pairs channels must belong to the held-out leak split")
        if any(existing_task_id == task_id for existing_task_id, _channel in eval_pairs):
            parser.error("eval_pairs must contain at most one channel per task")
        eval_pairs.append((task_id, channel))
    return tuple(eval_pairs)


def round_robin_leak_tasks(
    task_ids: Sequence[str], environments: Mapping[str, str]
) -> tuple[str, ...]:
    """Cycle repositories using the trusted environment identity, never task-id heuristics."""
    grouped: dict[str, list[str]] = {}
    for task_id in sorted(task_ids):
        repo = environments[task_id].rsplit("/", 1)[0]
        grouped.setdefault(repo, []).append(task_id)
    return tuple(
        task_id
        for row in zip_longest(*(grouped[repo] for repo in sorted(grouped)))
        for task_id in row
        if task_id is not None
    )
