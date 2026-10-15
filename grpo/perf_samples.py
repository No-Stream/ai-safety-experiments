"""Incremental gzip telemetry: each durable observation is a complete gzip member."""

from __future__ import annotations

import gzip
import json
import os
import zlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from pathlib import Path

SAMPLES_FILE = "throughput-samples.jsonl.gz"


def read_samples(path: Path, *, repair_tail: bool = False) -> Iterator[dict[str, Any]]:
    """Read complete members; discard only an incomplete final write on recovery.

    CRC failures raise. A killed writer may leave a partial final member, whose
    observations were never durable. Earlier members remain independently readable.
    """
    with path.open("r+b" if repair_tail else "rb") as stream:
        member_start = 0
        pending = b""
        while pending or (pending := stream.read(65536)):
            decoder = zlib.decompressobj(wbits=31)
            decoded = bytearray()
            while True:
                decoded.extend(decoder.decompress(pending))
                if decoder.eof:
                    pending = decoder.unused_data
                    member_start = stream.tell() - len(pending)
                    for line in decoded.splitlines():
                        yield json.loads(line)
                    break
                pending = stream.read(65536)
                if not pending:
                    if repair_tail:
                        stream.truncate(member_start)
                        stream.flush()
                        os.fsync(stream.fileno())
                    return


def append_sample(path: Path, sample: Mapping[str, Any]) -> None:
    """Close and fsync each member before acknowledging an observation."""
    member = gzip.compress((json.dumps(sample, separators=(",", ":")) + "\n").encode(), mtime=0)
    with path.open("ab") as stream:
        stream.write(member)
        stream.flush()
        os.fsync(stream.fileno())
