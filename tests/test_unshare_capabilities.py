"""The unshare payload must lose privileges after the jail mounts are complete."""

from __future__ import annotations

import subprocess
from pathlib import Path

JAIL = Path(__file__).resolve().parents[1] / "scripts" / "episode_jail.sh"


def test_usr_alternative_commands_resolve_inside_the_jail(tmp_path: Path) -> None:
    result = subprocess.run(  # noqa: S603 - trusted repo script and synthetic payload
        [
            str(JAIL),
            "--backend",
            "unshare",
            "--episode-dir",
            str(tmp_path),
            "--",
            "bash",
            "-c",
            "printf '2\\n3\\n' | awk '{ total += $1 } END { print total }'",
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "5"


def test_unshare_payload_has_no_capabilities_and_cannot_mount(tmp_path: Path) -> None:
    command = (
        "sed -n '/^Cap/p; /^NoNewPrivs/p' /proc/self/status; "
        "if mount -t tmpfs tmpfs /usr/bin; then echo MOUNT_SUCCEEDED; else echo MOUNT_REFUSED; fi"
    )
    result = subprocess.run(  # noqa: S603 - trusted repo script and synthetic payload
        [
            str(JAIL),
            "--backend",
            "unshare",
            "--episode-dir",
            str(tmp_path),
            "--",
            "bash",
            "-c",
            command,
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    status = dict(line.split(":", 1) for line in result.stdout.splitlines() if ":" in line)
    assert {
        name: int(status[name], 16) for name in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")
    } == dict.fromkeys(("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb"), 0)
    assert int(status["NoNewPrivs"]) == 1
    assert "MOUNT_REFUSED" in result.stdout
    assert "MOUNT_SUCCEEDED" not in result.stdout
