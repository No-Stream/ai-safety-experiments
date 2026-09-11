"""Run the reference kit's runner end to end against stubs, and drive each of its gates to fail.

`scripts/kit_reference/runner_skeleton.sh` is the agenda every future launch kit is copied from
(tests/test_kit_reference.py pins what it carries; scripts/kit_reference/README.md says why). Greps
and shellcheck cannot tell whether its gates work, so here the whole file executes inside a scratch
tree against one stub program that stands in for every binary it calls (aws, uv, curl, tmux, sudo,
systemd-run, ...) and records each call:

- a fresh box sweeps alone with no corpus pinned, resolves and exports the swept corpus, trains with
  the hub offline, evaluates each step with its own sync destination, closes every cell, and touches
  the agenda-complete marker last, after the final upload;
- a relaunch restores the bank first, walks the banked checkpoints newest-first past a torn one, and
  skips the sweep; one relaunch hands `--print-corpus` and `--print-plan` to the real
  `games.arm_sequence`, which is what keeps the stub's answers honest, and the real module is also
  asked directly why the sweep has to be its own invocation;
- each gate is then driven to fail: a plan printing `--save-steps 10` or `--max-steps 700`, a failed
  dependency sync, a failed checkpoint listing or restore, an unratified on-demand priced shape, a
  stop shutdown behaviour, a failed tmux hand-off, an eval cell that never closed, too little disk.

The two root-owned paths the runner writes (BOX_LOG_DIR, BOX_BIN_DIR) are the parameters that let it
run here; a box never sets them.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from test_kit_reference import KIT_SETS, LAUNCHER_SETS, RUNNER_PATH
from test_ship_tree import BASH, REPO_ROOT, RUN_NAME, RUN_PREFIX, S3_PREFIX, S3_REGION, SCRATCH_ROOT

from games.plans import artifact_model_slug

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

# Every binary the runner calls that is not plain coreutils. One program answers for all of them,
# dispatched on the name it was invoked as; the test installs it under each name on the box's PATH.
STUBBED_TOOLS = (
    "aws",
    "uv",
    "curl",
    "tmux",
    "sudo",
    "getent",
    "df",
    "nvidia-smi",
    "systemctl",
    "systemd-run",
    "chown",
    "sleep",
    "shutdown",
)
INSTANCE_ID = "i-0123456789abcdef0"
ARM_TAG = f"{KIT_SETS['ARM']}-{artifact_model_slug(KIT_SETS['MODEL'])}"
RUN_DIR_PREFIX = f"{RUN_PREFIX}/runs/{ARM_TAG}"
BUCKET, _, RUN_DIR_KEY = RUN_DIR_PREFIX[len("s3://") :].partition("/")
# What games.train.missing_checkpoint_files requires of a checkpoint a resume may use.
COMPLETE_CHECKPOINT_FILES = (
    "trainer_state.json",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "adapter_model.safetensors",
)

STUB_PROGRAM = r'''#!<python>
"""One stub for every binary the reference runner calls; dispatched on the name it was invoked as.

Records every call as a JSON line in $KIT_TEST_CONTROL/calls.log and answers from files under that
directory, so a test can set the instance type, the shutdown behaviour, a failing sync or a plan that
prints the wrong save cadence, and then read back what the runner did about it. The `uv` half mirrors
the three answers of games.arm_sequence the runner depends on (--print-corpus resolves exactly one
non-empty corpus-*.jsonl in the sweep directory; --print-plan refuses 'sweep' beside a consuming
stage and renders argv space-separated; the sweep writes a timestamped corpus) and, with the
`real-plan` control set, hands those two flags to the real module instead.
"""
import fnmatch
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys

CONTROL = pathlib.Path(os.environ["KIT_TEST_CONTROL"])
TOOL = pathlib.Path(sys.argv[0]).name
ARGV = sys.argv[1:]
RECORDED_ENV = ("GAMES_ARM_SEQ_STAGES", "GAMES_ARM_SEQ_CORPUS", "HF_HUB_OFFLINE", "UV_NO_SYNC", "HF_HOME")


def record():
    line = {
        "tool": TOOL,
        "argv": ARGV,
        "cwd": os.getcwd(),
        "env": {name: os.environ[name] for name in RECORDED_ENV if name in os.environ},
    }
    with (CONTROL / "calls.log").open("a") as log:
        log.write(json.dumps(line) + "\n")


def control(name, default=None):
    path = CONTROL / name
    return path.read_text().strip() if path.exists() else default


def s3_path(uri):
    return CONTROL / "s3" / uri[len("s3://"):]


def option(name):
    return ARGV[ARGV.index(name) + 1]


def die(message, code=1):
    sys.stderr.write(f"stub {TOOL}: {message}\n")
    sys.exit(code)


def copy_tree(source, target, excludes):
    if not source.is_dir():
        return
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source).as_posix()
        if any(fnmatch.fnmatch(relative, pattern) for pattern in excludes):
            continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, destination)


def aws():
    if ARGV[:2] == ["ec2", "describe-instance-attribute"]:
        print(control("shutdown-behaviour", "terminate"))
        return
    if ARGV[:2] == ["s3api", "list-objects-v2"]:
        if control("fail-s3api-list") is not None:
            die("An error occurred (AccessDenied) when calling the ListObjectsV2 operation", 254)
        prefix = option("--prefix")
        base = CONTROL / "s3" / option("--bucket") / prefix
        children = sorted(f"{prefix}{p.name}/" for p in base.iterdir() if p.is_dir()) if base.is_dir() else []
        print("\t".join(children) if children else "None")
        return
    if ARGV[:2] == ["s3", "ls"]:
        sys.exit(0 if s3_path(ARGV[2]).exists() else 1)
    if ARGV[:2] == ["s3", "cp"]:
        source, target = ARGV[2], ARGV[3]
        if target.startswith("s3://"):
            s3_path(target).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source, s3_path(target))
        else:
            shutil.copy(s3_path(source), target)
        return
    if ARGV[:2] == ["s3", "sync"]:
        source, target = ARGV[2], ARGV[3]
        failing = control("fail-sync-from")
        if failing and source.startswith(failing):
            die(f"fatal error: An error occurred (503) when calling the ListObjectsV2 operation on {source}")
        excludes = [ARGV[i + 1] for i, argument in enumerate(ARGV) if argument == "--exclude"]
        if source.startswith("s3://"):
            copy_tree(s3_path(source), pathlib.Path(target), excludes)
        else:
            copy_tree(pathlib.Path(source), s3_path(target), excludes)
        return
    die(f"unexpected call {ARGV}")


def artifact_paths():
    arm, model = os.environ["GAMES_ARM_SEQ_ARM"], os.environ["GAMES_ARM_SEQ_MODEL"]
    tag = model.rsplit("/", 1)[-1].replace(".", "").lower()  # games.plans.artifact_model_slug
    repo = pathlib.Path.cwd()
    return {
        "arm_output_dir": repo / "artifacts" / "games" / "runs" / f"{arm}-{tag}",
        "sweep_dir": repo / "artifacts" / "games" / "select" / f"{arm}-{tag}",
        "arm_s3_dest": os.environ["GAMES_ARM_SEQ_S3_DEST"].rstrip("/") + f"/{arm}-{tag}",
        "arm_log": repo / "artifacts" / "games" / "logs" / f"arm-{arm}-{tag}.log",
    }


def stages():
    return os.environ.get("GAMES_ARM_SEQ_STAGES", "sweep").split(",")


def uv():
    if ARGV[:1] == ["sync"]:
        print("Resolved 400 packages (stub)")
        sys.exit(int(control("uv-sync-rc", "0")))
    if ARGV[:3] != ["run", "--frozen", "python"]:
        die(f"unexpected call {ARGV}")
    rest = ARGV[3:]
    paths = artifact_paths()
    if rest[:1] == ["-c"]:
        code = rest[1]
        for name in ("arm_output_dir", "arm_s3_dest", "sweep_dir"):
            if name in code:
                print(paths[name])
                return
        if "snapshot_download" in code:
            print("cached at", pathlib.Path(os.environ["HF_HOME"]) / "hub" / "stub")
            return
        die(f"unexpected -c {code!r}")
    module = rest[rest.index("-m") + 1]
    flags = rest[rest.index("-m") + 2:]
    if module == "games.arm_sequence":
        arm_sequence(flags, paths)
    elif module == "games.stage_runner":
        stage_runner(paths)
    elif module == "games.run_evals":
        run_evals(flags)
    else:
        die(f"unexpected module {module}")


def arm_sequence(flags, paths):
    if control("real-plan") is not None:
        # The real module answers, from the real repository, told to write where the stub world does.
        os.environ["GAMES_ARM_SEQ_OUTPUT_DIR"] = str(paths["arm_output_dir"])
        os.environ["GAMES_ARM_SEQ_SWEEP_DIR"] = str(paths["sweep_dir"])
        os.chdir(os.environ["KIT_TEST_REAL_REPO"])
        real_uv = os.environ["KIT_TEST_REAL_UV"]
        os.execv(real_uv, [real_uv, *ARGV])
    if "--print-corpus" in flags:
        sweep_dir = paths["sweep_dir"]
        candidates = (
            sorted(p for p in sweep_dir.glob("corpus-*.jsonl") if "-regraded-" not in p.name)
            if sweep_dir.is_dir()
            else []
        )
        if len(candidates) == 1 and candidates[0].stat().st_size > 0:
            print(candidates[0])
            return
        die(f"no single non-empty corpus-*.jsonl in {sweep_dir}: {[c.name for c in candidates]}")
    if "--print-plan" in flags:
        chosen = stages()
        consuming = [stage for stage in chosen if stage in ("regrade", "arm")]
        if "sweep" in chosen and consuming:
            die(f"ValueError: this plan selects 'sweep' together with {consuming}, which takes two invocations")
        if consuming and "GAMES_ARM_SEQ_CORPUS" not in os.environ:
            die("FileNotFoundError: no corpus resolves and GAMES_ARM_SEQ_CORPUS is unset")
        save_steps = control("plan-save-steps", os.environ["GAMES_ARM_SEQ_SAVE_STEPS"])
        max_steps = control("plan-max-steps", os.environ["GAMES_ARM_SEQ_MAX_STEPS"])
        print(f"stages      {', '.join(chosen)}")
        for stage in chosen:
            print(f"\nSTAGE {stage}")
            if stage == "arm":
                print(
                    f"  argv      timeout 36h uv run --frozen python -m games.train "
                    f"--arm {os.environ['GAMES_ARM_SEQ_ARM']} --corpus {os.environ['GAMES_ARM_SEQ_CORPUS']} "
                    f"--output-dir {paths['arm_output_dir']} --resume-from-checkpoint latest "
                    f"--max-steps {max_steps} --save-steps {save_steps} --num-generations 8"
                )
            else:
                print(
                    f"  argv      timeout 12h uv run --frozen python -m games.select_prompts "
                    f"--out-dir {paths['sweep_dir']}"
                )
        return
    die(f"unexpected flags {flags}")


def stage_runner(paths):
    chosen = stages()
    if "sweep" in chosen:
        paths["sweep_dir"].mkdir(parents=True, exist_ok=True)
        (paths["sweep_dir"] / "corpus-20260903T000000-stub.jsonl").write_text('{"prompt": "stub"}\n')
    if "arm" in chosen:
        out = paths["arm_output_dir"]
        checkpoint = out / f"checkpoint-{os.environ['GAMES_ARM_SEQ_MAX_STEPS']}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        complete = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_model.safetensors")
        for name in complete:
            (checkpoint / name).write_text(name)
        (out / "train_summary.json").write_text("{}")
        paths["arm_log"].parent.mkdir(parents=True, exist_ok=True)
        paths["arm_log"].write_text(
            "{'reward': 0.41, 'frac_groups_pure': 0.12, 'step': 70}\n"
            "padding trim: step tokens padded=1000 trimmed=400 kept_fraction=0.6\n"
        )
        copy_tree(out, s3_path(paths["arm_s3_dest"]), [])  # the trainer's own per-save sync
    sys.exit(int(control("stage-rc-" + "+".join(chosen), "0")))


def run_evals(flags):
    out_dir = pathlib.Path(option_of(flags, "--out-dir"))
    step = option_of(flags, "--steps")
    trace, summary = out_dir / f"step-{step}.jsonl", out_dir / f"step-{step}.summary.json"
    if "--summarise-only" in flags:
        if not trace.exists():
            die(f"--summarise-only needs a trace at {trace}, and there is none.")
        if summary.exists():
            return
        die(f"{trace} is unfinished: it died before its last generate call")
    rc = int(control(f"eval-rc-step-{step}", "0"))
    out_dir.mkdir(parents=True, exist_ok=True)
    trace.write_text('{"record": 1}\n')
    if rc == 0:
        summary.write_text("{}")
    sys.exit(rc)


def option_of(flags, name):
    return flags[flags.index(name) + 1]


def curl():
    url = next(argument for argument in ARGV if argument.startswith("http"))
    if url.endswith("/api/token"):
        print("stub-imds-token")
    elif url.endswith("/instance-id"):
        print("i-0123456789abcdef0")
    elif url.endswith("/placement/region"):
        print("us-west-2")
    elif url.endswith("/instance-type"):
        print(control("instance-type", "g7e.2xlarge"))
    elif url.endswith("/instance-life-cycle"):
        print(control("lifecycle", "on-demand"))
    elif url.endswith("/uv/install.sh"):
        print('mkdir -p "$HOME/.local/bin"\n: >"$HOME/.local/bin/env"')
    else:
        die(f"unexpected url {url}")


def tmux():
    if ARGV[:1] == ["new-session"]:
        rc = int(control("tmux-rc", "0"))
        if rc:
            die("no server running on /tmp/tmux-1000/default", rc)
        # The session's command runs here and now rather than detached, so the test sees the agenda end.
        subprocess.run(["/bin/bash", "-c", ARGV[-1]], check=False)
        return
    if ARGV[:1] == ["ls"]:
        rc = int(control("tmux-ls-rc", "0"))
        if rc == 0:
            print("wave-reskin: 1 windows")
        sys.exit(rc)
    die(f"unexpected call {ARGV}")


def sudo():
    rest = list(ARGV)
    while rest[:1] == ["-u"]:
        rest = rest[2:]
    os.execvp(rest[0], rest)


def getent():
    print(f"{ARGV[1]}:x:1000:1000:Run user:{os.environ['KIT_TEST_HOME']}:/bin/bash")


def df():
    if "--output=avail" in ARGV:
        print("Avail")
        print(control("disk-avail-gb", "200") + "G")
    else:
        print("Filesystem      Size  Used Avail Use% Mounted on")
        print("/dev/root       500G  100G  400G  20% /")


def nvidia_smi():
    print("12 %, 3000 MiB" if any(a.startswith("--query-gpu") for a in ARGV) else "NVIDIA L4 (stub)")


def sleep():
    # A heartbeat loop under test ends at its first sleep, with one beat written.
    if control("sleep-stops-the-loop") is not None:
        os.kill(os.getppid(), signal.SIGTERM)


def systemctl():
    print("active")


def noop():
    return


DISPATCH = {
    "aws": aws,
    "uv": uv,
    "curl": curl,
    "tmux": tmux,
    "sudo": sudo,
    "getent": getent,
    "df": df,
    "nvidia-smi": nvidia_smi,
    "systemctl": systemctl,
    "systemd-run": noop,
    "chown": noop,
    "sleep": sleep,
    "shutdown": noop,
}
record()
DISPATCH[TOOL]()
'''


@dataclass
class Box:
    """A scratch tree standing in for a rented box: its home, repo, control directory and stub PATH."""

    root: Path
    home: Path
    repo: Path
    control: Path
    log_dir: Path
    bin_dir: Path
    env: dict[str, str]

    def set_control(self, name: str, value: str = "") -> None:
        (self.control / name).write_text(value)

    def s3(self, uri: str) -> Path:
        return self.control / "s3" / uri[len("s3://") :]

    def bank(self, uri: str, text: str) -> None:
        """Put one object under the stubbed S3."""
        target = self.s3(uri)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    def run(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603 - the tracked runner, under the stub PATH
            [BASH, str(RUNNER_PATH)],
            env=self.env,
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
        )

    def calls(self, tool: str | None = None, *, module: str | None = None) -> list[dict[str, Any]]:
        """Every recorded stub call, optionally one tool's, optionally one `python -m` module's."""
        log = self.control / "calls.log"
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        if tool is not None:
            calls = [call for call in calls if call["tool"] == tool]
        if module is not None:
            calls = [call for call in calls if module in call["argv"]]
        return calls

    def chain_state(self) -> str:
        return (self.home / "CHAIN_STATE").read_text().strip()

    def bootstrap_log(self) -> str:
        return (self.log_dir / "box-bootstrap.log").read_text()

    def run_log(self) -> str:
        path = self.home / f"{RUN_NAME}.log"
        return path.read_text() if path.exists() else ""

    @property
    def done_marker(self) -> Path:
        return self.home / "AGENDA_DONE"

    @property
    def out_dir(self) -> Path:
        return self.repo / "artifacts" / "games" / "runs" / ARM_TAG

    @property
    def sweep_dir(self) -> Path:
        return self.repo / "artifacts" / "games" / "select" / ARM_TAG


@pytest.fixture
def box() -> Iterator[Box]:
    """The scratch box, seeded the way the reference user-data leaves it before the hand-off."""
    root = Path(tempfile.mkdtemp(prefix="kit-reference-box-", dir=SCRATCH_ROOT))
    home, repo, control, stubs, log_dir, bin_dir, hf_cache = (
        root / name for name in ("home", "repo", "control", "stubs", "log", "bin", "hf")
    )
    for directory in (home, repo, control / "s3", stubs, log_dir, bin_dir, hf_cache):
        directory.mkdir(parents=True)
    program = stubs / "stub.py"
    program.write_text(STUB_PROGRAM.replace("#!<python>", f"#!{sys.executable}", 1))
    program.chmod(0o755)
    for tool in STUBBED_TOOLS:
        (stubs / tool).symlink_to(program)
    (home / "IID").write_text(f"{INSTANCE_ID}\n")
    (home / "CHAIN_STATE").write_text("bootstrapping started=2026-09-03T00:00:00Z\n")
    (repo / "GIT_SHA").write_text(f"{LAUNCHER_SETS['SHIP_HEAD_SHA']}\n")
    (repo / "SHIP_TREE").write_text("abcdef0123456789\n")
    real_uv = shutil.which("uv")
    assert real_uv is not None, "the real-plan scenario needs uv on PATH"

    environment = {
        name: value for name, value in os.environ.items() if not name.startswith("GAMES_")
    }
    environment.update(
        {
            "PATH": f"{stubs}:{environment['PATH']}",
            "KIT_TEST_CONTROL": str(control),
            "KIT_TEST_HOME": str(home),
            "KIT_TEST_REAL_UV": real_uv,
            "KIT_TEST_REAL_REPO": str(REPO_ROOT),
            "BOX_LOG_DIR": str(log_dir),
            "BOX_BIN_DIR": str(bin_dir),
            "HF_CACHE_DIR": str(hf_cache),
            "HOLD_MINUTES": "10",
            "RUN_NAME": RUN_NAME,
            "RUN_PREFIX": RUN_PREFIX,
            "S3_BASE": S3_PREFIX,
            "S3_REGION": S3_REGION,
            "ARM": KIT_SETS["ARM"],
            "MODEL": KIT_SETS["MODEL"],
            "MAX_STEPS": KIT_SETS["MAX_STEPS"],
            "EVAL_STEPS": KIT_SETS["EVAL_STEPS"],
            "PLAN_STEP_MINUTES": KIT_SETS["PLAN_STEP_MINUTES"],
            "DONE_MARKER": str(home / "AGENDA_DONE"),
            "REPO": str(repo),
        }
    )
    yield Box(root, home, repo, control, log_dir, bin_dir, environment)
    shutil.rmtree(root, ignore_errors=True)


def bank_complete_checkpoint(box: Box, step: int) -> None:
    for name in COMPLETE_CHECKPOINT_FILES:
        box.bank(f"{RUN_DIR_PREFIX}/checkpoint-{step}/{name}", name)


def bank_corpus(box: Box, name: str = "corpus-20260901T000000-banked.jsonl") -> None:
    box.bank(f"{RUN_PREFIX}/select/{name}", '{"prompt": "banked"}\n')


def stage_invocations(box: Box) -> list[dict[str, str]]:
    """The environment of each `games.stage_runner` invocation, in order."""
    return [call["env"] for call in box.calls("uv", module="games.stage_runner")]


def eval_invocations(box: Box) -> list[list[str]]:
    return [call["argv"] for call in box.calls("uv", module="games.run_evals")]


def index_of(calls: list[dict[str, Any]], tool: str, matches: Callable[[list[str]], bool]) -> int:
    """Position of the first call of `tool` whose argv satisfies `matches`."""
    for index, call in enumerate(calls):
        if call["tool"] == tool and matches(call["argv"]):
            return index
    raise AssertionError(f"no {tool} call matched; calls were {calls}")


def sync_between(source: str, target: str) -> Callable[[list[str]], bool]:
    """An `aws s3 sync` from `source` to `target`, both as the runner spells them (trailing slash)."""
    return lambda argv: argv[:4] == ["s3", "sync", source, target]


def python_running(*fragments: str) -> Callable[[list[str]], bool]:
    """A `uv run` whose argv carries every fragment, in order."""

    def matches(argv: list[str]) -> bool:
        joined = " ".join(argv)
        position = 0
        for fragment in fragments:
            position = joined.find(fragment, position)
            if position < 0:
                return False
            position += len(fragment)
        return True

    return matches


class TestTheRunnerOnAFreshBox:
    def test_it_sweeps_alone_resolves_the_corpus_trains_evaluates_and_closes(
        self, box: Box
    ) -> None:
        completed = box.run()
        assert completed.returncode == 0, box.bootstrap_log()
        assert box.chain_state().startswith("exited rc=0 reason=done logs=synced "), box.run_log()
        assert box.done_marker.exists()
        assert (box.home / "BOOTSTRAP_DONE").exists()

        # The two-invocation recipe: the sweep alone with no corpus pinned, then the arm with the
        # swept corpus exported; every stage runs with the hub already offline.
        invocations = stage_invocations(box)
        assert [env["GAMES_ARM_SEQ_STAGES"] for env in invocations] == ["sweep", "arm"]
        assert "GAMES_ARM_SEQ_CORPUS" not in invocations[0]
        assert invocations[1]["GAMES_ARM_SEQ_CORPUS"] == str(
            box.sweep_dir / "corpus-20260903T000000-stub.jsonl"
        )
        assert all(env["HF_HUB_OFFLINE"] == "1" for env in invocations)
        assert all(env["UV_NO_SYNC"] == "1" for env in invocations)

        # Each invocation sits behind its own plan gate, and the gates' reports are banked.
        plans = [
            c["env"]["GAMES_ARM_SEQ_STAGES"] for c in box.calls("uv") if "--print-plan" in c["argv"]
        ]
        assert plans == ["sweep", "arm"]
        for label in ("sweep", "arm"):
            assert box.s3(f"{RUN_PREFIX}/logs/plan-{label}-{INSTANCE_ID}.txt").exists()

        # The order of the whole day, read from what was actually called.
        calls = box.calls()
        restore = index_of(calls, "aws", sync_between(f"{RUN_PREFIX}/select/", f"{box.sweep_dir}/"))
        listing = index_of(calls, "aws", lambda argv: argv[:2] == ["s3api", "list-objects-v2"])
        prefetch = index_of(calls, "uv", python_running("snapshot_download"))
        sweep_plan = index_of(calls, "uv", python_running("games.arm_sequence", "--print-plan"))
        sweep = index_of(calls, "uv", python_running("games.stage_runner"))
        banked = index_of(calls, "aws", sync_between(f"{box.sweep_dir}/", f"{RUN_PREFIX}/select/"))
        arm = sweep + 1 + index_of(calls[sweep + 1 :], "uv", python_running("games.stage_runner"))
        first_eval = index_of(calls, "uv", python_running("games.run_evals"))
        final_sync = index_of(
            calls, "aws", sync_between(f"{box.repo}/artifacts/games/logs/", f"{RUN_PREFIX}/logs/")
        )
        assert (
            restore
            < listing
            < prefetch
            < sweep_plan
            < sweep
            < banked
            < arm
            < first_eval
            < final_sync
        )
        listing_argv = calls[listing]["argv"]
        assert listing_argv[listing_argv.index("--bucket") + 1] == BUCKET
        assert listing_argv[listing_argv.index("--prefix") + 1] == f"{RUN_DIR_KEY}/"
        assert "--delimiter" in listing_argv

        # The corpus is banked under the run prefix the moment the sweep ends.
        assert list(box.s3(f"{RUN_PREFIX}/select").glob("corpus-*.jsonl"))

        # Eval cells: the trained step then the base, each with its own sync destination and the
        # battery's own default sections (no --sections), then one closing pass per step.
        evals = eval_invocations(box)
        steps = [argv[argv.index("--steps") + 1] for argv in evals]
        assert steps == ["70", "0", "70", "0"]
        assert [("--summarise-only" in argv) for argv in evals] == [False, False, True, True]
        cell_prefix = f"{RUN_PREFIX}/evals/{KIT_SETS['ARM']}"
        assert all(argv[argv.index("--sync-dest") + 1] == cell_prefix for argv in evals)
        assert all("--sections" not in argv for argv in evals)

        # The exit: the state file, then the logs, then the marker.
        assert box.s3(f"{RUN_PREFIX}/logs/{INSTANCE_ID}.log").exists()
        assert box.s3(f"{RUN_PREFIX}/logs/uv-sync.log").exists()

    def test_the_heartbeat_is_armed_outside_tmux_and_writes_both_keys_from_the_live_log(
        self, box: Box
    ) -> None:
        assert box.run().returncode == 0, box.bootstrap_log()
        [unit] = box.calls("systemd-run")
        assert "--unit=box-heartbeat" in unit["argv"]
        heartbeat = box.bin_dir / "box-heartbeat.sh"
        assert str(heartbeat) in unit["argv"][-1]
        assert str(box.home / "run.env") in unit["argv"][-1]
        # One beat of the unit's own script; the stubbed sleep then ends the loop.
        box.set_control("sleep-stops-the-loop")
        completed = subprocess.run(  # noqa: S603 - the runner's own heartbeat script, under the stub PATH
            [BASH, str(heartbeat), str(box.home / "run.env")],
            env=box.env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == -signal.SIGTERM, completed.stderr
        shared = box.s3(f"{S3_PREFIX}/status/{INSTANCE_ID}.txt").read_text()
        own = box.s3(f"{RUN_PREFIX}/status/{INSTANCE_ID}.txt").read_text()
        assert shared == own
        assert "work=RUNNING (tmux present)" in shared
        assert "frac_groups_pure': 0.12" in shared
        assert "'reward': 0.41" in shared
        assert "padding trim: step tokens" in shared
        assert f"TRAINING COMPLETE: {box.out_dir / 'train_summary.json'}" in shared
        assert "step-70.summary.json" in shared

    def test_a_failed_final_log_upload_is_recorded_in_the_exit_state(self, box: Box) -> None:
        box.set_control("fail-sync-from", str(box.repo / "artifacts" / "games" / "logs"))
        assert box.run().returncode == 0
        assert box.chain_state().startswith("exited rc=0 reason=done logs=SYNC-FAILED rc=1 ")
        assert box.done_marker.exists()


class TestTheRunnerOnARelaunch:
    def test_it_restores_the_bank_past_a_torn_checkpoint_and_skips_the_sweep(
        self, box: Box
    ) -> None:
        bank_corpus(box)
        bank_complete_checkpoint(box, 69)
        box.bank(f"{RUN_DIR_PREFIX}/checkpoint-70/trainer_state.json", "torn")
        box.bank(f"{RUN_DIR_PREFIX}/checkpoint-70/adapter_model.safetensors", "torn")
        box.bank(
            f"{RUN_DIR_PREFIX}/incomplete-checkpoints/2026/checkpoint-1/optimizer.pt", "set aside"
        )
        box.bank(f"{RUN_DIR_PREFIX}/mem_log.csv", "step,rss\n")
        assert box.run().returncode == 0, box.bootstrap_log()
        assert box.chain_state().startswith("exited rc=0 reason=done "), box.run_log()

        invocations = stage_invocations(box)
        assert [env["GAMES_ARM_SEQ_STAGES"] for env in invocations] == ["arm"]
        assert invocations[0]["GAMES_ARM_SEQ_CORPUS"] == str(
            box.sweep_dir / "corpus-20260901T000000-banked.jsonl"
        )
        assert "the sweep is not needed" in box.run_log()

        # Newest first: the torn 70 came down and was named torn, the complete 69 stopped the walk,
        # and nothing under incomplete-checkpoints/ was fetched. (The stubbed arm stage then writes
        # its own complete checkpoint-70, as the trainer would, so the walk is read from the calls.)
        log = box.run_log()
        assert "checkpoint-70 is torn" in log
        assert "restored complete checkpoint-69" in log
        assert log.index("checkpoint-70 is torn") < log.index("restored complete checkpoint-69")
        pulled = [
            call["argv"][2]
            for call in box.calls("aws")
            if call["argv"][:2] == ["s3", "sync"] and "/checkpoint-" in call["argv"][2]
        ]
        assert pulled == [f"{RUN_DIR_PREFIX}/checkpoint-70/", f"{RUN_DIR_PREFIX}/checkpoint-69/"]
        run_dir_restore = next(
            call["argv"]
            for call in box.calls("aws")
            if call["argv"][:3] == ["s3", "sync", f"{RUN_DIR_PREFIX}/"]
        )
        assert "incomplete-checkpoints/*" in run_dir_restore
        assert (box.out_dir / "mem_log.csv").exists()
        assert not (box.out_dir / "incomplete-checkpoints").exists()

    def test_the_real_plan_answers_the_gate_the_stub_mirrors(self, box: Box) -> None:
        """--print-corpus and --print-plan go to the real games.arm_sequence; the gate must pass on its output."""
        box.set_control("real-plan")
        bank_corpus(box)
        completed = box.run()
        assert completed.returncode == 0, box.bootstrap_log()
        assert box.chain_state().startswith("exited rc=0 reason=done "), box.run_log()
        plan = (box.repo / "artifacts" / "games" / "logs" / f"plan-arm-{RUN_NAME}.txt").read_text()
        assert f"STAGE train {KIT_SETS['ARM']} @ {KIT_SETS['MODEL']}" in plan
        assert "--max-steps 70 --save-steps 1 " in plan
        assert f"--corpus {box.sweep_dir / 'corpus-20260901T000000-banked.jsonl'}" in plan
        assert [env["GAMES_ARM_SEQ_STAGES"] for env in stage_invocations(box)] == ["arm"]

    def test_the_split_exists_because_the_real_plan_refuses_sweep_beside_arm(self) -> None:
        """What the runner would hit with one invocation of `sweep,arm`, from the real module."""
        environment = {
            name: value for name, value in os.environ.items() if not name.startswith("GAMES_")
        }
        environment.update(
            {
                "GAMES_ARM_SEQ_ARM": KIT_SETS["ARM"],
                "GAMES_ARM_SEQ_MODEL": KIT_SETS["MODEL"],
                "GAMES_ARM_SEQ_STAGES": "sweep,arm",
                "UV_NO_SYNC": "1",
            }
        )
        real_uv = shutil.which("uv")
        assert real_uv is not None
        completed = subprocess.run(  # noqa: S603 - the repo's own plan module, read-only flags
            [real_uv, "run", "--frozen", "python", "-m", "games.arm_sequence", "--print-plan"],
            env=environment,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode != 0
        assert "which takes two invocations rather than one" in completed.stderr

    def test_a_corpus_the_resolver_refuses_is_not_swept_over(self, box: Box) -> None:
        bank_corpus(box, "corpus-20260901T000000-first.jsonl")
        bank_corpus(box, "corpus-20260902T000000-second.jsonl")
        assert box.run().returncode == 0
        assert box.chain_state().startswith(
            f"exited rc=1 reason=corpus-unresolvable in {box.sweep_dir} "
        ), box.run_log()
        assert stage_invocations(box) == []
        assert box.done_marker.exists()


class TestEachRunnerGateFailsWhenItShould:
    @pytest.mark.parametrize(
        ("control", "printed", "want"),
        [
            ("plan-save-steps", "10", "--save-steps 1"),
            ("plan-max-steps", "700", "--max-steps 70"),
        ],
    )
    def test_the_plan_gate_refuses_a_registered_line_that_differs(
        self, box: Box, control: str, printed: str, want: str
    ) -> None:
        """A substring match would pass `--save-steps 10` under `--save-steps 1`; whole tokens do not."""
        box.set_control(control, printed)
        assert box.run().returncode == 0
        assert box.chain_state().startswith(f"exited rc=1 reason=plan-gate-arm-missing: {want} "), (
            box.run_log()
        )
        # Caught after the sweep on a fresh box, whose corpus is banked, so the fixed kit skips it.
        assert [env["GAMES_ARM_SEQ_STAGES"] for env in stage_invocations(box)] == ["sweep"]
        assert list(box.s3(f"{RUN_PREFIX}/select").glob("corpus-*.jsonl"))
        assert box.done_marker.exists()

    def test_the_plan_gate_passes_the_registered_lines_it_wants(self, box: Box) -> None:
        """The same gate, with the values it asks for: the control's teeth are not a sticky failure."""
        box.set_control("plan-save-steps", "1")
        box.set_control("plan-max-steps", "70")
        assert box.run().returncode == 0
        assert box.chain_state().startswith("exited rc=0 reason=done ")

    def test_a_failed_dependency_sync_stops_the_agenda_by_name(self, box: Box) -> None:
        box.set_control("uv-sync-rc", "1")
        assert box.run().returncode == 0
        assert box.chain_state().startswith("exited rc=1 reason=uv-sync "), box.run_log()
        assert "Resolved 400 packages (stub)" in box.run_log(), "the sync's log tail was not shown"
        assert stage_invocations(box) == []
        assert box.done_marker.exists()

    def test_a_failed_checkpoint_listing_stops_the_run_rather_than_starting_fresh(
        self, box: Box
    ) -> None:
        bank_corpus(box)
        bank_complete_checkpoint(box, 70)
        box.set_control("fail-s3api-list")
        assert box.run().returncode == 0
        assert box.chain_state().startswith(
            "exited rc=1 reason=restore-list-checkpoints rc=254 "
        ), box.run_log()
        assert stage_invocations(box) == []
        assert box.done_marker.exists()

    def test_a_failed_restore_stops_the_run(self, box: Box) -> None:
        box.set_control("fail-sync-from", f"{RUN_DIR_PREFIX}/")
        assert box.run().returncode == 0
        assert box.chain_state().startswith("exited rc=1 reason=restore-run-dir rc=1 "), (
            box.run_log()
        )
        assert stage_invocations(box) == []

    def test_a_failed_checkpoint_restore_stops_the_run(self, box: Box) -> None:
        bank_complete_checkpoint(box, 70)
        box.set_control("fail-sync-from", f"{RUN_DIR_PREFIX}/checkpoint-70/")
        assert box.run().returncode == 0
        assert box.chain_state().startswith("exited rc=1 reason=restore-checkpoint-70 rc=1 "), (
            box.run_log()
        )
        assert stage_invocations(box) == []

    def test_an_on_demand_priced_shape_is_held_and_stands_down_unratified(self, box: Box) -> None:
        box.set_control("instance-type", "p5.48xlarge")
        assert box.run().returncode == 0
        assert box.chain_state().startswith("exited rc=0 reason=market-ratification-timeout "), (
            box.run_log()
        )
        assert stage_invocations(box) == []
        assert not (box.home / "KEEPALIVE").exists()
        polls = [c for c in box.calls("aws") if c["argv"][:2] == ["s3", "ls"]]
        assert len(polls) == 1 + 10 // 5, "one look, then one per five held minutes"
        assert all(c["argv"][2] == f"{RUN_PREFIX}/control/ONDEMAND_RATIFIED" for c in polls)
        assert [c["argv"] for c in box.calls("sleep")] == [["300"], ["300"]]
        assert box.done_marker.exists()

    def test_a_ratified_priced_shape_and_a_spot_one_both_proceed(self, box: Box) -> None:
        box.set_control("instance-type", "p5.48xlarge")
        box.bank(f"{RUN_PREFIX}/control/ONDEMAND_RATIFIED", "yes\n")
        assert box.run().returncode == 0
        assert box.chain_state().startswith("exited rc=0 reason=done "), box.run_log()
        assert "on-demand p5.48xlarge ratified; proceeding" in box.run_log()

        (box.control / "calls.log").unlink()
        shutil.rmtree(box.s3(RUN_PREFIX))
        (box.home / "CHAIN_STATE").write_text("bootstrapping started\n")
        box.set_control("lifecycle", "spot")
        assert box.run().returncode == 0
        assert not [c for c in box.calls("aws") if c["argv"][:2] == ["s3", "ls"]]
        assert box.chain_state().startswith("exited rc=0 reason=done "), box.run_log()

    def test_a_stop_shutdown_behaviour_halts_before_the_agenda(self, box: Box) -> None:
        box.set_control("shutdown-behaviour", "stop")
        assert box.run().returncode == 1
        assert box.chain_state().startswith("exited rc=1 reason=shutdown-behaviour-stop ")
        [halt] = box.calls("shutdown")
        assert halt["argv"][:2] == ["-h", "+5"]
        assert box.calls("tmux") == []
        assert box.calls("systemd-run") == []

    def test_a_failed_tmux_hand_off_records_itself_and_touches_the_marker(self, box: Box) -> None:
        box.set_control("tmux-rc", "1")
        assert box.run().returncode == 1
        assert box.chain_state().startswith("exited rc=1 reason=tmux-launch-failed ")
        assert box.done_marker.exists()
        assert not (box.home / "BOOTSTRAP_DONE").exists()
        assert box.calls("uv") == []

    def test_an_eval_cell_that_never_closed_fails_the_agenda_after_every_cell_ran(
        self, box: Box
    ) -> None:
        box.set_control("eval-rc-step-70", "1")
        assert box.run().returncode == 0
        assert box.chain_state().startswith("exited rc=1 reason=eval-cell-unclosed "), box.run_log()
        steps = [argv[argv.index("--steps") + 1] for argv in eval_invocations(box)]
        assert steps == ["70", "0", "70", "0"], "the base cell and the closing pass still ran"
        assert "BATTERY CELL FAILED: step 70 rc=1" in box.run_log()
        assert "CELL NOT CLOSED: step 70" in box.run_log()
        assert box.done_marker.exists()

    def test_too_little_disk_stops_before_anything_is_derived(self, box: Box) -> None:
        box.set_control("disk-avail-gb", "20")
        assert box.run().returncode == 0
        assert box.chain_state().startswith("exited rc=1 reason=disk: only 20G free "), (
            box.run_log()
        )
        assert not [c for c in box.calls("uv") if c["argv"][:1] == ["run"]]
