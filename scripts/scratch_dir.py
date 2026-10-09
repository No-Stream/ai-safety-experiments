#!/usr/bin/python3
"""Create scratch directories with an explicit expiry and purpose."""

from __future__ import annotations

import argparse
import datetime
import getpass
import json
import logging
import re
import sys
from pathlib import Path

if __package__:
    from scripts.ttl_marker import read_ttl_marker
else:
    from ttl_marker import read_ttl_marker

LOGGER = logging.getLogger(__name__)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRATCH_ROOT = Path("/var/tmp/scratch")  # noqa: S108 - intentional scratch root
ARTIFACTS_ROOT = REPOSITORY_ROOT / "artifacts"


def parse_ttl_duration(value: str) -> datetime.timedelta:
    """Parse a positive whole-number duration such as 3d."""
    match = re.fullmatch(r"([1-9][0-9]*)d", value)
    if match is None:
        raise ValueError("TTL must be a positive whole number followed by d")
    amount = int(match.group(1))
    return datetime.timedelta(days=amount)


def create_scratch_dir(  # noqa: PLR0913 - explicit scratch creation inputs
    name: str,
    ttl: str,
    purpose: str,
    root: Path,
    *,
    created_by: str | None = None,
    now: datetime.datetime | None = None,
) -> Path:
    """Create one new scratch directory under root and write its expiry marker."""
    if not name or name in (".", "..") or Path(name).name != name or "\\" in name:
        raise ValueError("scratch directory name must be one path component")
    if not purpose.strip():
        raise ValueError("scratch purpose must be nonempty text")

    duration = parse_ttl_duration(ttl)
    current_time = now or datetime.datetime.now(datetime.timezone.utc)  # noqa: UP017 - Python 3.8
    creator = getpass.getuser() if created_by is None else created_by
    if not creator.strip():
        raise ValueError("created_by must be nonempty text")

    root.mkdir(parents=True, exist_ok=True)
    scratch_path = root / name
    scratch_path.mkdir()
    marker_document = {
        "expires": (current_time + duration).date().isoformat(),
        "created_by": creator,
        "purpose": purpose,
    }
    marker_path = scratch_path / ".ttl"
    with marker_path.open("x", encoding="utf-8") as marker_file:
        json.dump(marker_document, marker_file, sort_keys=True)
        marker_file.write("\n")
    read_ttl_marker(marker_path)
    return scratch_path


def main(argv: list[str] | None = None) -> int:
    """Create a TTL-marked directory under /var/tmp/scratch or artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="create a TTL-marked scratch directory")
    create.add_argument("name")
    create.add_argument("--ttl", required=True, help="positive duration such as 3d")
    create.add_argument("--purpose", required=True)
    create.add_argument("--location", choices=("scratch", "artifacts"), default="scratch")
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    roots = {"scratch": SCRATCH_ROOT, "artifacts": ARTIFACTS_ROOT}
    scratch_path = create_scratch_dir(
        arguments.name,
        arguments.ttl,
        arguments.purpose,
        roots[arguments.location],
    )
    LOGGER.info("%s", scratch_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
