"""Pin what the host tripwire watches, and what it must stop watching.

Two bugs, both of which produced a green-looking gate that measured the wrong thing.

The collector accepted any whitespace token that started with ``/`` and existed, so a crontab
line's shell *redirect target* was collected as if it were a script. One of this box's cron entries
appends to a log, that log entered the baseline, and its hash therefore changed every time cron ran
-- turning ``make canary-check`` red on a schedule for a reason that has nothing to do with a jail
escape. A required gate that fails daily for a benign reason is a gate people stop reading, which
costs more than the gate was worth.

The watched set also missed live execution surfaces. ``.claude/hooks/*`` does not descend, so the
helper library three of those hooks source on every tool call was unwatched: a write there is
arbitrary code execution on the next Claude Code tool call, which is precisely the property the
watched set is defined by. Hooks registered *outside* that directory were unwatched for the same
reason, so the tripwire now reads the settings file for the commands it registers, the way it
already reads the crontab -- which is what makes it stay correct when a hook is added rather than
requiring somebody to remember this file.

Nothing here writes to a watched path or to the baseline manifest: the fake homes are ``tmp_path``
trees, and the crontab is a canned string rather than the real one.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts import canary_manifest


def _fake_crontab(monkeypatch: pytest.MonkeyPatch, body: str) -> None:
    """Answer ``crontab -l`` with a canned crontab, leaving the real one untouched."""

    def fake_run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        assert argv == ["crontab", "-l"], f"unexpected command {argv}"
        return subprocess.CompletedProcess(argv, 0, stdout=body, stderr="")

    monkeypatch.setattr(canary_manifest.subprocess, "run", fake_run)


def _executable(path: Path) -> Path:
    """Create an executable file, as a cron-invoked script is."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\ntrue\n")
    path.chmod(0o755)
    return path


def _data_file(path: Path) -> Path:
    """Create a plain non-executable file, as an append-only log is."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("log line\n")
    return path


class TestCrontabTargetsExcludeRedirects:
    """The live bug: an append-only log entered the baseline and drifted every time cron ran."""

    def test_a_spaced_redirect_target_is_not_collected(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        script = _executable(tmp_path / "update.sh")
        log = _data_file(tmp_path / "update.log")
        _fake_crontab(monkeypatch, f"0 12 * * * {script} >> {log} 2>&1\n")

        targets = canary_manifest.crontab_script_targets()

        assert str(script) in targets
        assert str(log) not in targets

    def test_an_attached_redirect_target_is_not_collected(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``>>/var/log/x`` with no space is the same redirect and the same non-surface."""
        script = _executable(tmp_path / "backup.sh")
        log = _data_file(tmp_path / "backup.log")
        _fake_crontab(monkeypatch, f"7 * * * * {script} 2>{log}\n")

        targets = canary_manifest.crontab_script_targets()

        assert str(script) in targets
        assert str(log) not in targets

    def test_a_brace_group_still_yields_the_command_it_runs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """This box's real shape: ``{ date -u; /path/to/thing update; } >> /path/to/log``."""
        script = _executable(tmp_path / "claude")
        log = _data_file(tmp_path / "cc-update.log")
        _fake_crontab(monkeypatch, f"0 12 * * * {{ date -u; {script} update; }} >> {log} 2>&1\n")

        targets = canary_manifest.crontab_script_targets()

        assert str(script) in targets
        assert str(log) not in targets

    def test_an_interpreted_script_is_collected_even_though_it_is_not_executable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Why redirects are excluded by *position* rather than by an executable-bit test.

        ``python3 /path/to/job.py`` is arbitrary code execution on cron's schedule whether or not
        the file carries the executable bit, so filtering on the bit would drop a real surface while
        fixing the log. The redirect operator is what actually distinguishes the two.
        """
        job = _data_file(tmp_path / "job.py")
        _fake_crontab(monkeypatch, f"30 3 * * * python3 {job}\n")

        assert str(job) in canary_manifest.crontab_script_targets()

    def test_a_missing_crontab_is_a_legitimate_empty_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def no_crontab(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr="no crontab for someone")

        monkeypatch.setattr(canary_manifest.subprocess, "run", no_crontab)

        assert canary_manifest.crontab_script_targets() == []

    def test_an_unreadable_crontab_raises_rather_than_shrinking_the_watched_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(argv, 126, stdout="", stderr="permission denied")

        monkeypatch.setattr(canary_manifest.subprocess, "run", broken)

        with pytest.raises(RuntimeError, match="could not read crontab"):
            canary_manifest.crontab_script_targets()


@pytest.fixture
def fake_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the tripwire at a throwaway home with no crontab: only globs and settings run."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(canary_manifest, "HOME", str(home))
    monkeypatch.setattr(canary_manifest, "crontab_script_targets", list)
    return home


class TestWatchedSurfaces:
    """Every path where a write buys code execution the next time something routine happens."""

    def test_the_helper_library_the_hooks_source_is_watched(self, fake_home: Path) -> None:
        """The sharpest gap: ``.claude/hooks/*`` does not descend into ``lib/``."""
        library = _executable(fake_home / ".claude" / "hooks" / "lib" / "skill-inject.sh")
        sibling = _executable(fake_home / ".claude" / "hooks" / "auto-skill-tools.sh")

        paths = canary_manifest.watched_paths()

        assert str(library) in paths
        assert str(sibling) in paths

    def test_a_hook_registered_outside_the_hooks_directory_is_watched(
        self, fake_home: Path
    ) -> None:
        """Registration is by path, so a hook can live anywhere; several on this box do."""
        hook = _executable(fake_home / ".local" / "bin" / "multi-linter.sh")
        settings = fake_home / ".claude" / "settings.json"
        settings.parent.mkdir(parents=True, exist_ok=True)
        settings.write_text(
            json.dumps(
                {
                    "hooks": {
                        "PostToolUse": [
                            {
                                "hooks": [
                                    {"type": "command", "command": "~/.local/bin/multi-linter.sh"}
                                ]
                            }
                        ]
                    }
                }
            )
        )

        assert str(hook) in canary_manifest.watched_paths()

    def test_the_status_line_and_credential_helper_are_watched(self, fake_home: Path) -> None:
        """Both run on this box's real settings, and neither is a ``hooks`` entry."""
        statusline = _executable(fake_home / ".claude" / "statusline.js")
        helper = _executable(fake_home / ".devtools" / "bin" / "credential-helper")
        settings = fake_home / ".claude" / "settings.json"
        settings.parent.mkdir(parents=True, exist_ok=True)
        settings.write_text(
            json.dumps(
                {
                    "statusLine": {"type": "command", "command": "~/.claude/statusline.js"},
                    "awsCredentialExport": f'"{helper}" default-credential-export',
                }
            )
        )

        paths = canary_manifest.watched_paths()

        assert str(statusline) in paths
        assert str(helper) in paths

    def test_a_settings_command_redirect_target_is_not_watched(self, fake_home: Path) -> None:
        """The same exclusion as the crontab, because it is the same tokenizer and the same bug."""
        hook = _executable(fake_home / ".claude" / "hooks" / "noisy.sh")
        log = _data_file(fake_home / ".claude" / "logs" / "noisy.log")
        settings = fake_home / ".claude" / "settings.json"
        settings.parent.mkdir(parents=True, exist_ok=True)
        settings.write_text(
            json.dumps(
                {
                    "hooks": {
                        "Stop": [
                            {"hooks": [{"type": "command", "command": f"{hook} >> {log} 2>&1"}]}
                        ]
                    }
                }
            )
        )

        paths = canary_manifest.watched_paths()

        assert str(hook) in paths
        assert str(log) not in paths

    def test_a_home_without_a_settings_file_is_a_legitimate_empty_set(
        self, fake_home: Path
    ) -> None:
        _executable(fake_home / ".claude" / "hooks" / "only-hook.sh")

        assert canary_manifest.settings_command_targets() == []


class TestThisBoxIsFullyCovered:
    """The check that keeps the watched set honest as this machine's configuration changes.

    Read off the live settings file rather than a fixture, because the failure mode is a hook
    registered somewhere the watched set does not reach -- which no fixture can notice. Skipped
    where there is no settings file, since the tripwire's baseline is machine-local anyway.
    """

    def test_every_registered_command_on_this_box_is_watched(self) -> None:
        settings = Path(canary_manifest.HOME) / ".claude" / "settings.json"
        if not settings.is_file():
            pytest.skip(f"no Claude Code settings at {settings}")

        registered = canary_manifest.settings_command_targets()
        watched = set(canary_manifest.watched_paths())

        assert registered, (
            f"{settings} registered no runnable command, which would mean the collector matched "
            "nothing rather than that this box runs no hooks."
        )
        assert set(registered) <= watched
