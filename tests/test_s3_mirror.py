"""Drive scripts/s3_mirror.sh against a stubbed ``aws``, because nothing else checks it.

The script keeps one shared local mirror of a finished run's S3 artifacts per box, so that the prep,
readout and audit agents reading the same run read one copy instead of pulling the same objects down
once each. Three of its guarantees are the reason it exists rather than being a one-line
``aws s3 sync`` in each recipe, and all three are invisible in a green run: the default filters drop
the class no readout opens (checkpoint directories, measured at 95% of a run's bytes), an in-use
marker makes a sibling's clean-up refuse instead of deleting the copy another agent is reading (which
is what happened to one GRPO audit mid-analysis), and nothing it creates in or deletes may sit
anywhere but under the mirror root.

Nothing here reaches the network or S3. Every test runs the real script with ``aws`` shimmed onto
PATH, records the argument vector the script constructed, and asserts on that; none of them read the
script's source. The mirror root is a throwaway directory under ``/var/tmp`` (never ``/tmp``, whose
inode cap is shared box-wide) and the fixture removes it.

Twenty-three mutations were run against a pristine copy of the script on 2026-09-04, one per guard,
and every one of them turned this suite red; the two the suite could not see are named at the end.

Which filters it asks for:

- the ``*.log`` exclude dropped from the defaults;
- the ``*/checkpoint-*`` half of the checkpoint filter dropped, keeping only ``checkpoint-*``. An aws
  s3 filter is matched against the whole key with the source prefix prepended, so that one pattern
  reaches a checkpoint directory at the run's root and nothing below it -- which is where every
  checkpoint of every run in this repository actually lives, so the exclude looked present and
  excluded nothing;
- ``--include-checkpoint-state``'s two re-includes moved ahead of the checkpoint excludes. The filter
  list is applied in sequence and the last match wins, so from there the flag reads as present in the
  argument vector and brings back nothing at all;
- ``--include-logs`` reduced to dropping the ``*.log`` exclude, without the second sync of the sibling
  ``<base>/logs/<run>/`` prefix. That prefix is where a run's logs are -- the run's own prefix holds
  none for any recent run -- so the flag named something it never fetched.

Where it may write and delete:

- the run-name shape check removed, which is what keeps the mirror path one component under the root.
  It took an extra assertion to see: with only that check gone a hostile name was still refused, by
  the root resolution behind it, so a test asserting "it refused" stayed green. What fires is
  ``assert not mirror.root.exists()``, because that later refusal comes after the root is created;
- the mirror root's resolution removed. It is what refuses a root reached through a ``/var/tmp``
  symlink whose target is elsewhere, or spelled ``/var/tmp/../../tmp/x``; without it the script syncs
  a run into the home tree and exits 0;
- the "a real directory at its own name" check removed for the mirror, then for ``<root>/.holders``,
  then for ``<root>/.holders/<run>``, three mutations and three failures. The holder ones carry the
  worst of the four defects this file's guards came from: ``rm -rf`` and ``mkdir -p`` both follow a
  symlinked intermediate component, so with ``.holders`` pointed elsewhere a ``--delete`` deleted a
  subtree outside the mirror root, printed "nothing to delete" and exited 0, and a sync wrote its
  marker out there too. Which of the two checks fires depends on whether the plant's target already
  holds a directory for this run, so both shapes are here;
- the holder tag's shape check removed; the tag is a path component under the holder directory too.

The in-use marker, which is the only reason a sibling's clean-up refuses:

- ``--delete`` made to ignore the markers;
- the holder query made to report "no holders" when it cannot read the holder directory, which is how
  ``--delete`` came to destroy a mirror a reader was holding while exiting non-zero. The injection is
  mode 0o111 and not 0o000 for a reason recorded on the test itself;
- that query's refusal left unpropagated out of its command substitution. A ``die`` inside ``$(...)``
  exits the subshell only, so the message printed while ``--delete`` carried on with an empty list;
- the holder tag given a default of the parent shell's pid. Two readers started from one shell then
  share one marker, so the first one's release clears the second's claim; the same reader syncing
  inside ``$(...)`` and releasing from a later shell gets two different values, so the release finds
  nothing and the claim outlives the reader;
- ``--release`` under a tag nobody took made a no-op that exits 0, which is what leaves a real claim
  standing until someone reaches for ``--delete --force``;
- ``--release`` left carrying the exit status of the ``rmdir`` that tidies an empty holder directory
  away, so a release while another holder was still there -- the overlap the design exists for --
  reported failure.

What reaches stdout, since a caller writes ``stage=$(s3_mirror.sh ...)``:

- a sync that brought nothing down made to print its path and exit 0, which hands a readout an empty
  directory that reads exactly like a staged one;
- the ``>&2`` dropped from the sync call, so the CLI's own chatter joins the mirror path on stdout;
- ``--status`` made to print a path for a run nobody has mirrored;
- the destination wiped before each sync, which turns a delta sync into a full re-download and throws
  away whatever a reader put beside the mirror.

What it refuses to be configured with:

- the ``SHIP_S3_PREFIX`` refusal replaced by a baked-in default bucket, which is both a wrong-source
  bug and the privacy rule this repository is built around (a bucket name may never be tracked here);
- that same refusal hoisted ahead of the mode dispatch, so releasing a claim or reaping a mirror,
  neither of which names a bucket, would then demand one and the marker convention would go unused by
  any agent that had not exported the launch variables;
- the refusal of a flag that cannot apply to the chosen mode (``--force`` on a sync, ``--include-*``
  on a delete) removed, so each went back to being a silent no-op.

Two checks that were in the script and are not any more, because nothing could make them fail: a
re-resolution of the mirror root after ``mkdir -p`` created it, and a check that the mirror's parent
resolves to the root. Both are tautologies once the root is resolved before creation and the run name
carries no slash -- ``mkdir -p`` only ever creates real directories and refuses a dangling symlink
component -- so they could differ only if something rearranged /var/tmp mid-run, which asking twice
narrows rather than closes. They read as guards in a review and pinned nothing.
"""

from __future__ import annotations

import itertools
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from box_stubs import write_stub

if TYPE_CHECKING:
    from collections.abc import Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
MIRROR = REPO_ROOT / "scripts" / "s3_mirror.sh"

# Never /tmp, whose inode cap is shared box-wide; the script refuses a root outside here for that.
SCRATCH_ROOT = "/var/tmp"  # noqa: S108

RUN = "track-record-v2"
# A stand-in: the real bucket is account-specific and never tracked here, hence SHIP_S3_PREFIX.
PREFIX = "s3://stub-bucket/base-prefix"
REGION = "us-west-2"
# The tag the fixture exports, standing in for the reader an agent would name itself after.
HOLDER = "test-reader"

LOG_EXCLUDE = ["--exclude", "*.log"]
CHECKPOINT_EXCLUDES = ["--exclude", "checkpoint-*", "--exclude", "*/checkpoint-*"]
CHECKPOINT_STATE_INCLUDES = [
    "--include",
    "checkpoint-*/trainer_state.json",
    "--include",
    "*/checkpoint-*/trainer_state.json",
]


def sync_argv(destination: Path, *filters: str) -> list[str]:
    """The whole argument vector a sync of RUN should hand the CLI."""
    return [
        "s3",
        "sync",
        f"{PREFIX}/{RUN}/",
        str(destination),
        "--region",
        REGION,
        "--no-progress",
        *filters,
    ]


def logs_argv(destination: Path) -> list[str]:
    """The second sync ``--include-logs`` asks for: the sibling prefix, into <mirror>/logs/."""
    return [
        "s3",
        "sync",
        f"{PREFIX}/logs/{RUN}/",
        f"{destination}/logs",
        "--region",
        REGION,
        "--no-progress",
    ]


# One file per call, one argument per line; the witness file and the stdout chatter each carry a test.
STUB_AWS = """\
calls=$MIRROR_TEST_CALLS
n=1
while [ -e "$calls/call-$n" ]; do n=$((n + 1)); done
for arg in "$@"; do printf '%s\\n' "$arg" >> "$calls/call-$n"; done
printf 'stub aws chatter on stdout\\n'
if [ -f "$calls/fail" ]; then printf 'stub aws: refusing to sync\\n' >&2; exit 1; fi
dest=$4
if [ -d "$dest" ] && [ ! -f "$calls/transfer-nothing" ]; then
  printf 'downloaded\\n' > "$dest/witness.txt"
fi
exit 0
"""


@dataclass(frozen=True)
class MirrorHarness:
    scratch: Path
    root: Path
    calls_dir: Path
    env: dict[str, str]

    def run(
        self, *args: str, env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603 - repo script, literal arguments
            [str(MIRROR), *args],
            env=self.env if env is None else env,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )

    def calls(self) -> list[list[str]]:
        recorded: list[list[str]] = []
        for index in itertools.count(1):
            call = self.calls_dir / f"call-{index}"
            if not call.exists():
                return recorded
            recorded.append(call.read_text().splitlines())
        raise AssertionError("unreachable")

    def env_without(self, *names: str) -> dict[str, str]:
        return {key: value for key, value in self.env.items() if key not in names}

    def mirror_of(self, run_name: str = RUN) -> Path:
        return self.root / run_name

    def markers_of(self, run_name: str = RUN) -> Path:
        return self.root / ".holders" / run_name


@pytest.fixture
def mirror() -> Iterator[MirrorHarness]:
    """A throwaway mirror root under /var/tmp, which is the only place the script will work.

    Not ``tmp_path``: that lives under /tmp, a RAM-backed tmpfs whose inode cap every concurrent
    session on this box shares, and the script refuses a root outside /var/tmp for exactly that
    reason (``test_a_mirror_root_outside_var_tmp_is_refused`` pins the refusal).
    """
    scratch = Path(tempfile.mkdtemp(prefix="s3-mirror-test-", dir=SCRATCH_ROOT))
    try:
        bin_dir = scratch / "bin"
        bin_dir.mkdir()
        calls_dir = scratch / "calls"
        calls_dir.mkdir()
        write_stub(bin_dir, "aws", STUB_AWS)
        yield MirrorHarness(
            scratch=scratch,
            root=scratch / "mirror",
            calls_dir=calls_dir,
            env={
                "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
                "HOME": str(scratch),
                "SHIP_S3_PREFIX": PREFIX,
                "SHIP_S3_REGION": REGION,
                "S3_MIRROR_ROOT": str(scratch / "mirror"),
                "S3_MIRROR_HOLDER": HOLDER,
                "MIRROR_TEST_CALLS": str(calls_dir),
            },
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


class TestTheSyncItAsksFor:
    def test_the_default_sync_takes_the_run_prefix_and_excludes_logs_and_checkpoints(
        self, mirror: MirrorHarness
    ) -> None:
        done = mirror.run(RUN)

        assert done.returncode == 0, done.stderr
        assert done.stdout == f"{mirror.mirror_of()}\n"
        assert mirror.calls() == [sync_argv(mirror.mirror_of(), *LOG_EXCLUDE, *CHECKPOINT_EXCLUDES)]

    def test_include_logs_drops_the_log_filter_and_mirrors_the_sibling_logs_prefix(
        self, mirror: MirrorHarness
    ) -> None:
        """A run's logs are not under its own prefix, so the flag is a second sync as well."""
        done = mirror.run(RUN, "--include-logs")

        assert done.returncode == 0, done.stderr
        assert mirror.calls() == [
            sync_argv(mirror.mirror_of(), *CHECKPOINT_EXCLUDES),
            logs_argv(mirror.mirror_of()),
        ]
        assert (mirror.mirror_of() / "logs").is_dir()

    def test_include_checkpoints_drops_both_checkpoint_filters(self, mirror: MirrorHarness) -> None:
        done = mirror.run(RUN, "--include-checkpoints")

        assert done.returncode == 0, done.stderr
        assert mirror.calls() == [sync_argv(mirror.mirror_of(), *LOG_EXCLUDE)]

    def test_asking_for_everything_passes_no_filters_at_all(self, mirror: MirrorHarness) -> None:
        """Also the empty-array expansion under ``set -u``, which is where this would die."""
        done = mirror.run(RUN, "--include-logs", "--include-checkpoints")

        assert done.returncode == 0, done.stderr
        assert mirror.calls() == [sync_argv(mirror.mirror_of()), logs_argv(mirror.mirror_of())]

    def test_the_checkpoint_state_flag_re_includes_that_file_after_the_excludes(
        self, mirror: MirrorHarness
    ) -> None:
        """Order is the whole flag: an s3 filter list is applied in sequence and the last match wins."""
        done = mirror.run(RUN, "--include-checkpoint-state")

        assert done.returncode == 0, done.stderr
        assert mirror.calls() == [
            sync_argv(
                mirror.mirror_of(), *LOG_EXCLUDE, *CHECKPOINT_EXCLUDES, *CHECKPOINT_STATE_INCLUDES
            )
        ]

    def test_asking_for_whole_checkpoints_makes_the_state_flag_redundant(
        self, mirror: MirrorHarness
    ) -> None:
        done = mirror.run(RUN, "--include-checkpoints", "--include-checkpoint-state")

        assert done.returncode == 0, done.stderr
        assert mirror.calls() == [sync_argv(mirror.mirror_of(), *LOG_EXCLUDE)]

    def test_only_the_mirror_path_reaches_stdout(self, mirror: MirrorHarness) -> None:
        done = mirror.run(RUN)

        assert done.stdout == f"{mirror.mirror_of()}\n"
        assert "stub aws chatter on stdout" in done.stderr

    def test_no_mode_ever_hands_the_cli_an_s3_destination_or_a_delete(
        self, mirror: MirrorHarness
    ) -> None:
        for arguments in ([RUN], [RUN, "--status"], [RUN, "--release"], [RUN, "--delete"]):
            done = mirror.run(*arguments)
            assert done.returncode == 0, f"{arguments}: {done.stderr}"

        calls = mirror.calls()
        assert len(calls) == 1, "only the sync mode talks to the CLI at all"
        for call in calls:
            assert [argument for argument in call if argument.startswith("s3://")] == [
                f"{PREFIX}/{RUN}/"
            ]
            assert call[3].startswith(str(mirror.root))
            assert "--delete" not in call

    def test_only_a_sync_prints_a_stage_path(self, mirror: MirrorHarness) -> None:
        """A release or a reap leaves nothing to read, so neither may look like a staged mirror."""
        assert mirror.run(RUN).returncode == 0

        for arguments in ([RUN, "--release"], [RUN, "--delete"]):
            done = mirror.run(*arguments)
            assert done.returncode == 0, done.stderr
            assert done.stdout == "", arguments

    def test_a_failed_sync_is_a_refusal_that_prints_no_path_but_keeps_the_claim(
        self, mirror: MirrorHarness
    ) -> None:
        """``stage=$(s3_mirror.sh run)`` must not yield a path to a mirror that is missing objects."""
        (mirror.calls_dir / "fail").write_text("")

        done = mirror.run(RUN, "--holder", "readout-a")

        assert done.returncode != 0
        assert done.stdout == ""
        assert (mirror.markers_of() / "readout-a").exists(), (
            "a partial mirror still must not be reaped"
        )

    def test_a_sync_that_brought_nothing_down_is_a_refusal_unless_asked_for(
        self, mirror: MirrorHarness
    ) -> None:
        """A mistyped run name, a wrong base prefix and a never-uploaded run all look like this."""
        (mirror.calls_dir / "transfer-nothing").write_text("")

        done = mirror.run(RUN)

        assert done.returncode != 0
        assert done.stdout == "", "an empty directory must not be handed over as a stage"
        assert "--allow-empty" in done.stderr

        allowed = mirror.run(RUN, "--allow-empty")

        assert allowed.returncode == 0, allowed.stderr
        assert allowed.stdout == f"{mirror.mirror_of()}\n"


class TestTheConfigurationItRefuses:
    def test_an_unset_source_prefix_is_refused_by_name(self, mirror: MirrorHarness) -> None:
        done = mirror.run(RUN, env=mirror.env_without("SHIP_S3_PREFIX"))

        assert done.returncode != 0
        assert "SHIP_S3_PREFIX" in done.stderr
        assert done.stdout == ""
        assert mirror.calls() == [], "no bucket may be guessed at when the variable is unset"

    def test_the_local_modes_need_no_aws_configuration_at_all(self, mirror: MirrorHarness) -> None:
        """A reader releasing its claim, or a sibling reaping a mirror, has no bucket to name."""
        assert mirror.run(RUN, "--holder", "readout-a").returncode == 0
        bare = mirror.env_without("SHIP_S3_PREFIX", "SHIP_S3_REGION")

        for arguments in (
            [RUN, "--status"],
            [RUN, "--release", "--holder", "readout-a"],
            [RUN, "--delete"],
        ):
            done = mirror.run(*arguments, env=bare)
            assert done.returncode == 0, f"{arguments} refused without a bucket: {done.stderr}"

        assert not mirror.mirror_of().exists()

    def test_an_unset_region_is_refused_by_name(self, mirror: MirrorHarness) -> None:
        done = mirror.run(RUN, env=mirror.env_without("SHIP_S3_REGION"))

        assert done.returncode != 0
        assert "SHIP_S3_REGION" in done.stderr
        assert mirror.calls() == []

    def test_aws_region_stands_in_for_the_ship_variable(self, mirror: MirrorHarness) -> None:
        env = mirror.env_without("SHIP_S3_REGION")
        env["AWS_REGION"] = "us-east-2"

        done = mirror.run(RUN, env=env)

        assert done.returncode == 0, done.stderr
        (call,) = mirror.calls()
        assert call[call.index("--region") + 1] == "us-east-2"

    def test_a_mirror_root_outside_var_tmp_is_refused(
        self, mirror: MirrorHarness, tmp_path: Path
    ) -> None:
        env = mirror.env | {"S3_MIRROR_ROOT": str(tmp_path / "mirror")}

        done = mirror.run(RUN, env=env)

        assert done.returncode != 0
        assert SCRATCH_ROOT in done.stderr
        assert mirror.calls() == []

    def test_a_mirror_root_reached_through_a_symlink_out_of_var_tmp_is_refused(
        self, mirror: MirrorHarness, tmp_path: Path
    ) -> None:
        """The string is under /var/tmp and the directory is not, which is the interesting case."""
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        link = mirror.scratch / "looks-like-var-tmp"
        link.symlink_to(outside)
        env = mirror.env | {"S3_MIRROR_ROOT": str(link / "mirror")}

        done = mirror.run(RUN, env=env)

        assert done.returncode != 0
        assert str(outside) in done.stderr, "the refusal must name where the root really resolves"
        assert mirror.calls() == []
        assert list(outside.iterdir()) == [], "and must refuse before creating anything there"

    def test_a_mirror_root_that_climbs_out_of_var_tmp_is_refused(
        self, mirror: MirrorHarness
    ) -> None:
        """``/var/tmp/../../tmp/x`` starts with /var/tmp/ as a string and is /tmp/x as a directory."""
        # The leaf is unique per invocation, and swept afterwards, because the whole point of the
        # assertion is that this path does not exist: shared and fixed, one sabotaged run leaving it
        # behind would fail every later run of the pristine script for a reason of its own making.
        target = Path(SCRATCH_ROOT) / f"..{os.sep}..{os.sep}tmp" / mirror.scratch.name
        env = mirror.env | {"S3_MIRROR_ROOT": str(target)}

        try:
            done = mirror.run(RUN, env=env)

            assert done.returncode != 0
            assert mirror.calls() == []
            assert not target.exists(), "a refusal must leave no directory outside /var/tmp behind"
        finally:
            shutil.rmtree(target, ignore_errors=True)

    def test_two_modes_at_once_are_refused(self, mirror: MirrorHarness) -> None:
        done = mirror.run(RUN, "--release", "--delete")

        assert done.returncode != 0
        assert mirror.calls() == []

    @pytest.mark.parametrize(
        "arguments",
        [
            [RUN, "--force"],
            [RUN, "--delete", "--include-checkpoints"],
            [RUN, "--status", "--allow-empty"],
            [RUN, "--release", "--force"],
        ],
    )
    def test_a_flag_that_cannot_apply_to_the_mode_is_refused(
        self, mirror: MirrorHarness, arguments: list[str]
    ) -> None:
        """Silently ignoring one leaves the caller believing something happened that did not."""
        done = mirror.run(*arguments)

        assert done.returncode != 0
        assert mirror.calls() == []

    def test_the_usage_text_prints_when_the_script_is_reached_through_path(
        self, mirror: MirrorHarness
    ) -> None:
        """usage() reads the header out of the file, and $0 is a bare name for a PATH invocation."""
        env = mirror.env | {"PATH": f"{MIRROR.parent}{os.pathsep}{mirror.env['PATH']}"}

        done = subprocess.run(  # noqa: S603 - repo script by name, resolved from PATH
            [MIRROR.name, "--help"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )

        assert done.returncode == 2
        assert "SHIP_S3_PREFIX" in done.stderr


class TestTheTraversalGuard:
    @pytest.mark.parametrize(
        "hostile_name",
        ["../escape", "..", ".", "a/b", ".hidden", f"{SCRATCH_ROOT}/escape", "sub/../../escape"],
    )
    def test_a_run_name_that_leaves_the_root_is_refused_and_creates_nothing(
        self, mirror: MirrorHarness, hostile_name: str
    ) -> None:
        done = mirror.run(hostile_name)

        assert done.returncode != 0
        assert done.stdout == ""
        assert mirror.calls() == []
        assert not (mirror.scratch / "escape").exists(), "the sync landed outside the mirror root"
        assert not mirror.root.exists(), "a refused run name must not even create the root"

    def test_a_symlink_planted_at_the_mirrors_own_name_is_refused(
        self, mirror: MirrorHarness
    ) -> None:
        outside = mirror.scratch / "outside"
        outside.mkdir()
        mirror.root.mkdir(parents=True)
        mirror.mirror_of().symlink_to(outside)

        done = mirror.run(RUN)

        assert done.returncode != 0
        assert str(outside) in done.stderr
        assert mirror.calls() == []
        assert list(outside.iterdir()) == []

    def test_a_symlink_planted_at_the_holder_directory_is_refused_by_every_mode(
        self, mirror: MirrorHarness
    ) -> None:
        """``rm -rf`` follows a symlinked intermediate component, so --delete reached outside here."""
        victim = mirror.scratch / "victim" / RUN / "inner"
        victim.mkdir(parents=True)
        keep = victim / "keep.txt"
        keep.write_text("a sibling agent's data, nothing to do with any mirror\n")
        mirror.root.mkdir(parents=True)
        (mirror.root / ".holders").symlink_to(mirror.scratch / "victim")

        for arguments in ([RUN], [RUN, "--delete"], [RUN, "--status"]):
            done = mirror.run(*arguments)
            assert done.returncode != 0, arguments
            assert ".holders" in done.stderr
            assert keep.exists(), f"{arguments} deleted a subtree outside the mirror root"

        assert mirror.calls() == []

    def test_a_symlink_at_the_holder_directory_is_refused_before_a_marker_is_written_through_it(
        self, mirror: MirrorHarness
    ) -> None:
        """The victim holds no directory for this run, which is what makes ``mkdir -p`` the hazard.

        With one there, the per-run check refuses the same plant, so only this shape reaches the check
        on the holder directory itself -- and only this shape reproduces the marker that landed outside
        the mirror root.
        """
        victim = mirror.scratch / "victim"
        victim.mkdir()
        mirror.root.mkdir(parents=True)
        (mirror.root / ".holders").symlink_to(victim)

        done = mirror.run(RUN, "--holder", "readout-a")

        assert done.returncode != 0
        assert done.stdout == ""
        assert list(victim.iterdir()) == [], "a marker was written outside the mirror root"
        assert mirror.calls() == []

    def test_a_symlink_planted_at_one_runs_holder_directory_is_refused(
        self, mirror: MirrorHarness
    ) -> None:
        victim = mirror.scratch / "victim"
        victim.mkdir()
        (victim / "keep.txt").write_text("not a holder tag\n")
        (mirror.root / ".holders").mkdir(parents=True)
        mirror.markers_of().symlink_to(victim)

        done = mirror.run(RUN, "--delete")

        assert done.returncode != 0
        assert (victim / "keep.txt").exists()
        assert mirror.calls() == []


class TestTheInUseMarker:
    def test_a_held_mirror_refuses_to_be_deleted_and_release_lifts_it(
        self, mirror: MirrorHarness
    ) -> None:
        assert mirror.run(RUN, "--holder", "readout-a").returncode == 0
        witness = mirror.mirror_of() / "witness.txt"
        assert witness.exists()

        refused = mirror.run(RUN, "--delete")

        assert refused.returncode != 0
        assert "readout-a" in refused.stderr
        assert witness.exists(), "the mirror a reader is holding must survive a sibling's clean-up"

        assert mirror.run(RUN, "--release", "--holder", "readout-a").returncode == 0
        deleted = mirror.run(RUN, "--delete")

        assert deleted.returncode == 0, deleted.stderr
        assert not mirror.mirror_of().exists()

    def test_one_holder_releasing_does_not_lift_another_holders_claim(
        self, mirror: MirrorHarness
    ) -> None:
        """The overlap this design exists for: two agents on one mirror, one of them finishing."""
        assert mirror.run(RUN, "--holder", "readout-a").returncode == 0
        assert mirror.run(RUN, "--holder", "audit-b").returncode == 0
        released = mirror.run(RUN, "--release", "--holder", "readout-a")

        assert released.returncode == 0, released.stderr

        refused = mirror.run(RUN, "--delete")

        assert refused.returncode != 0
        assert "audit-b" in refused.stderr
        assert "readout-a" not in refused.stderr
        assert (mirror.mirror_of() / "witness.txt").exists()

    def test_two_holders_of_one_mirror_get_a_marker_each(self, mirror: MirrorHarness) -> None:
        """One file per holder is the whole mechanism: a single flag could not survive the overlap."""
        assert mirror.run(RUN, "--holder", "readout-a").returncode == 0
        assert mirror.run(RUN, "--holder", "audit-b").returncode == 0

        assert sorted(path.name for path in mirror.markers_of().iterdir()) == [
            "audit-b",
            "readout-a",
        ]

    def test_a_release_under_a_tag_nobody_took_is_refused_and_names_the_real_holder(
        self, mirror: MirrorHarness
    ) -> None:
        """A no-op here leaves the real claim standing until someone reaches for --force."""
        assert mirror.run(RUN, "--holder", "readout-a").returncode == 0

        done = mirror.run(RUN, "--release", "--holder", "audit-b")

        assert done.returncode != 0
        assert "readout-a" in done.stderr
        assert (mirror.markers_of() / "readout-a").exists()

    def test_force_is_what_clears_a_marker_whose_holder_is_gone(
        self, mirror: MirrorHarness
    ) -> None:
        assert mirror.run(RUN, "--holder", "dead-agent").returncode == 0

        done = mirror.run(RUN, "--delete", "--force")

        assert done.returncode == 0, done.stderr
        assert not mirror.mirror_of().exists()
        assert not mirror.markers_of().exists()

    @pytest.mark.skipif(os.geteuid() == 0, reason="root lists an unreadable directory regardless")
    def test_an_unreadable_holder_directory_refuses_the_delete_rather_than_assuming_it_is_free(
        self, mirror: MirrorHarness
    ) -> None:
        """A query that cannot read the holder directory must refuse, not report the mirror free.

        Mode 0o111 rather than 0o000 on purpose: the directory has to stay traversable, or the earlier
        check that it is a real directory at its own name refuses first and this test passes against a
        holder query that fails open, which is exactly what it did until 2026-09-04.
        """
        assert mirror.run(RUN, "--holder", "readout-a").returncode == 0
        markers = mirror.markers_of()
        markers.chmod(0o111)
        try:
            done = mirror.run(RUN, "--delete")
        finally:
            markers.chmod(0o755)

        assert done.returncode != 0
        assert (mirror.mirror_of() / "witness.txt").exists(), "a held mirror was deleted anyway"
        assert (markers / "readout-a").exists()

    @pytest.mark.parametrize("hostile_tag", ["../escape", ".hidden", "a/b", "sub/../../escape"])
    def test_a_holder_tag_that_leaves_the_holder_directory_is_refused(
        self, mirror: MirrorHarness, hostile_tag: str
    ) -> None:
        """The tag is one path component under <root>/.holders/<run>/, so it is held to that shape."""
        done = mirror.run(RUN, "--holder", hostile_tag)

        assert done.returncode != 0
        assert done.stdout == ""
        assert mirror.calls() == []
        assert not (mirror.scratch / "escape").exists()
        assert not mirror.root.exists(), "a refused tag must not even create the root"

    def test_the_environment_variable_supplies_the_holder_tag(self, mirror: MirrorHarness) -> None:
        """An agent reading several runs exports it once instead of repeating --holder."""
        assert mirror.run(RUN).returncode == 0

        assert (mirror.markers_of() / HOLDER).exists()

    def test_a_sync_without_a_holder_tag_is_refused_by_name(self, mirror: MirrorHarness) -> None:
        done = mirror.run(RUN, env=mirror.env_without("S3_MIRROR_HOLDER"))

        assert done.returncode != 0
        assert "--holder" in done.stderr
        assert "S3_MIRROR_HOLDER" in done.stderr
        assert mirror.calls() == []
        assert not mirror.root.exists(), "a refused sync must not even create the root"

    def test_a_release_without_a_holder_tag_is_refused_by_name(self, mirror: MirrorHarness) -> None:
        assert mirror.run(RUN).returncode == 0

        done = mirror.run(RUN, "--release", env=mirror.env_without("S3_MIRROR_HOLDER"))

        assert done.returncode != 0
        assert "--holder" in done.stderr
        assert (mirror.markers_of() / HOLDER).exists()

    def test_status_and_delete_need_no_holder_tag(self, mirror: MirrorHarness) -> None:
        """Neither reads one, so a missing or malformed tag must not refuse them."""
        assert mirror.run(RUN).returncode == 0
        bare = mirror.env_without("S3_MIRROR_HOLDER")

        malformed = mirror.env | {"S3_MIRROR_HOLDER": "../escape"}
        for env in (bare, malformed):
            assert mirror.run(RUN, "--status", env=env).returncode == 0

        assert mirror.run(RUN, "--delete", "--force", env=malformed).returncode == 0

    def test_status_names_the_path_and_who_holds_it(self, mirror: MirrorHarness) -> None:
        assert mirror.run(RUN, "--holder", "readout-a").returncode == 0

        done = mirror.run(RUN, "--status")

        assert done.returncode == 0, done.stderr
        assert done.stdout == f"{mirror.mirror_of()}\n"
        assert "readout-a" in done.stderr

    def test_status_on_a_run_nobody_mirrored_prints_no_path(self, mirror: MirrorHarness) -> None:
        done = mirror.run(RUN, "--status")

        assert done.returncode != 0
        assert done.stdout == ""
        assert "not mirrored" in done.stderr


class TestIdempotence:
    def test_a_second_sync_reuses_the_directory_and_keeps_what_is_already_there(
        self, mirror: MirrorHarness
    ) -> None:
        first = mirror.run(RUN)
        assert first.returncode == 0, first.stderr
        kept = mirror.mirror_of() / "kept-by-the-first-sync.txt"
        kept.write_text("an object the first sync brought down\n")

        second = mirror.run(RUN)

        assert second.returncode == 0, second.stderr
        assert kept.exists(), "a delta sync must not wipe the mirror and re-download it"
        assert first.stdout == second.stdout
        calls = mirror.calls()
        assert len(calls) == 2
        assert calls[0] == calls[1], (
            "the same source, destination and filters, so sync sees no delta"
        )
