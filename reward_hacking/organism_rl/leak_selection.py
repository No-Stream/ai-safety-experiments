"""Explicit channel arms and configurable named channel transfer splits."""

from __future__ import annotations

import argparse
import json
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
        default=Path("/var/tmp/leak-env/organism-assets.json"),  # noqa: S108 - trusted local asset index
    )
    parser.add_argument("--leak-channel-split", help="named split in --leak-channel-splits")
    parser.add_argument(
        "--leak-channel-splits",
        type=Path,
        help="JSON object mapping split names to train/held-out channel lists",
    )


def resolve_leak_options(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, partition: str
) -> None:
    """Validate the named split and prevent training arms from opening held-out channels."""
    args.leak_channels_open = args.leak_channels is None or args.leak_channels_open
    if (args.leak_channel_split is None) != (args.leak_channel_splits is None):
        parser.error("leak-channel-split and leak-channel-splits must be provided together")
    if args.leak_channel_split is None:
        args.leak_channels = (
            parse_channels("open") if args.leak_channels is None else args.leak_channels
        )
        return
    channels_by_partition = _load_channel_split(
        args.leak_channel_splits, args.leak_channel_split, parser
    )
    if set(channels_by_partition["train"]) & set(channels_by_partition["held-out"]):
        parser.error("train and held-out leak channel sets must be disjoint")
    allowed = channels_by_partition[partition]
    if args.leak_channels_open:
        selected = tuple(channel for channel in allowed if channel in ALL_COMPATIBLE_CHANNELS)
    else:
        if args.leak_channels is None:
            parser.error("explicit leak channel selection is missing")
        selected = args.leak_channels
    if set(selected) - set(allowed):
        parser.error(f"leak channel arm includes channels outside the {partition} split")
    try:
        args.leak_channels = parse_channels(",".join(selected)) if selected else ()
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))


def _load_channel_split(
    config_path: Path, split_name: str, parser: argparse.ArgumentParser
) -> dict[str, tuple[str, ...]]:
    raw: object = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or split_name not in raw:
        parser.error("leak-channel-splits must contain the requested split name")
    split: object = cast("dict[str, object]", raw)[split_name]
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
    return channels_by_partition


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
