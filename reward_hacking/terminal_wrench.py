"""Offline reader for the Terminal Wrench recorded-trajectory corpus.

Terminal Wrench (``few-sh/terminal-wrench``, Apache-2.0) ships 6,289 frontier-model trajectories
on reward-hackable ops tasks, each with a ground-truth label (was it a hack, which categories, the
grader reward). We use it for two things during exploration, and only two:

1. **Read what real exploits look like.** Pull a handful of labelled hack transcripts and eyeball
   the shell the model actually ran. That is lead #1 from the exploration plan ("see real
   exploits"), and it needs no model inference at all.
2. **Exercise trace-reading with zero inference.** Adapting a recorded transcript into shell
   commands plus terminal observations is the same trace-reading we do on our own episodes, so the
   corpus is a free, deterministic way to sanity-check that reading before a GPU is ever involved.

This module is deliberately minimal. It resolves the local HF-cache snapshot offline, reads the
label index, and adapts a transcript into a light :class:`Trace` of steps (each carrying the shell
keystrokes and the observation text). There is **no tier taxonomy, no boundary map, no detector, no
scoring** — those are a later phase's problem, and building them here is exactly the
over-systematization the exploration plan bans.

Two facts about the on-disk corpus that shape the code, both verified against the cached snapshot:

- **The download is partial.** Only ~1,738 of the 6,289 index records have their transcript file
  present, and no task-definition/grader files were fetched at all. So the loader is
  partial-download-aware: :func:`iter_traces` yields ``None`` for a missing transcript and the
  caller counts the misses, rather than crashing or silently dropping rows.
- **Layout.** ``index/trajectories.json`` holds the label records; a transcript lives at
  ``tasks/<task_id>/<model>/<tree_name>/<trajectory_label>/trial/agent/trajectory.json``. A
  transcript is ``{"steps": [...]}`` where an agent step has ``tool_calls`` shaped
  ``{"function_name": "bash_command", "arguments": {"keystrokes": "<shell>"}}`` (the terminal call
  is ``{"function_name": "mark_task_complete"}``) and an ``observation`` shaped
  ``{"results": [{"content": "<terminal text>"}]}``.

The write-target parser (:func:`parse_write_targets`) is a conservative, best-effort extractor of
the filesystem paths a shell snippet writes to. It recognises the write forms the frontier models
actually used — redirections including heredocs, ``tee``, ``cp``/``mv``/``ln``, ``sed -i``,
``chmod``/``chown``, ``touch``/``mkdir``/``rm``, ``dd of=``, and ``open(..., 'w')`` inside
``python -c``. Anything it misses is a silent under-count, which is the honest failure direction
for a reading aid.

The shell-splitting primitives that parser is built on (:func:`split_commands`, :func:`split_argv`,
:func:`strip_command_prefixes`) are public because they are the same primitives any other
command-text reader needs and re-deriving them elsewhere would let the two drift.
``reward_hacking.harness.hack_detector`` reads the *read* side of the same command strings with
them. One known limit they carry: :func:`split_commands` splits on ``;`` without respecting quotes,
so a ``python -c "a; b"`` payload arrives as several segments. That is harmless when the question is
"what is argv[0] of each segment" and wrong when the question is "what does the payload do", so
payload-level checks must scan the whole command string instead.
"""

import json
import logging
import posixpath
import re
import shlex
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from huggingface_hub.constants import HF_HUB_CACHE

logger = logging.getLogger(__name__)

TERMINAL_WRENCH_REPO = "few-sh/terminal-wrench"

# "Task is done" tool names: this corpus uses mark_task_complete; aliases cover other scaffolds.
_REPORT_TOOLS = frozenset({"mark_task_complete", "task_complete", "submit", "finish", "done"})

# Leading tokens that wrap the real command; strip them to find argv[0].
_COMMAND_PREFIXES = frozenset({"sudo", "nice", "nohup", "time", "command", "builtin", "exec"})
_MIN_ARGS_FOR_LAST_TARGET = 2

_HEREDOC_HEADER_RE = re.compile(r"<<-?\s*([\"']?)([A-Za-z_][A-Za-z0-9_]*)\1")
# File-redirection target; filename group forbids a leading & so fd-dups (>&2, 2>&1) don't match.
_REDIRECT_RE = re.compile(r"""(?:\d*&?>>?|&>>?)\s*("(?:[^"\\]|\\.)*"|'[^']*'|[^\s;|&<>]+)""")
_PY_OPEN_RE = re.compile(r"""(?:open|Path)\(\s*['"]([^'"]+)['"](?:[^)]*['"][rwax+bt]*['"])?""")
_PY_WRITE_HINT_RE = re.compile(r"""open\([^)]*['"][wax][b+t]*['"]""")


@dataclass(frozen=True, slots=True)
class LabelRecord:
    """One entry from ``index/trajectories.json``: where the transcript lives plus its ground truth.

    Only the fields that matter for reading real exploits are kept: the locator (``task_id``,
    ``model``, ``tree_name``, ``trajectory_label``) and the human-facing ground truth
    (``classification``, ``observed_categories``, ``reward``, ``brief_exploit_summary``,
    ``explanation``, ``key_hack_steps``).
    """

    task_id: str
    model: str
    tree_name: str
    trajectory_label: str
    classification: str | None
    reward: float | None
    exploit_attempted: bool
    observed_categories: tuple[str, ...]
    key_hack_steps: tuple[int, ...]
    brief_exploit_summary: str | None
    explanation: str | None

    @property
    def is_hack(self) -> bool:
        """Return whether this record belongs to the hack trajectories."""
        return self.tree_name == "hack_trajectories"


@dataclass(frozen=True, slots=True)
class Command:
    """One tool call inside a recorded step: the shell it ran, or the report tool it invoked."""

    function_name: str
    keystrokes: str  # empty for non-shell calls (e.g. mark_task_complete)
    is_report: bool


@dataclass(frozen=True, slots=True)
class TraceStep:
    """Represent one agent step: its reasoning, tool calls, and terminal output.

    The normalized shape supports shell-command analysis across transcript variants.
    """

    step_id: int | None
    message: str
    commands: tuple[Command, ...]
    observation: str


@dataclass(frozen=True, slots=True)
class Trace:
    """A recorded trajectory adapted to shell-commands-plus-observations. No tiers, no scoring."""

    record: LabelRecord
    steps: tuple[TraceStep, ...] = field(default_factory=tuple)

    @property
    def episode_id(self) -> str:
        """Return the stable identifier composed from the record's locating fields."""
        r = self.record
        return f"{r.task_id}:{r.model}:{r.tree_name}:{r.trajectory_label}"

    @property
    def shell_commands(self) -> list[str]:
        """Every non-empty shell snippet the agent ran, in order."""
        return [
            command.keystrokes
            for step in self.steps
            for command in step.commands
            if command.keystrokes.strip()
        ]

    @property
    def observations(self) -> list[str]:
        """The terminal output text of each step that produced any, in order."""
        return [step.observation for step in self.steps if step.observation]

    @property
    def reported_done(self) -> bool:
        """True if the agent ever issued a terminal 'task complete' call."""
        return any(command.is_report for step in self.steps for command in step.commands)

    def write_targets(self) -> list[str]:
        """Filesystem paths written across the whole trace, de-duplicated, order-preserving."""
        seen: set[str] = set()
        ordered: list[str] = []
        for keystrokes in self.shell_commands:
            for target in parse_write_targets(keystrokes):
                if target not in seen:
                    seen.add(target)
                    ordered.append(target)
        return ordered


def resolve_snapshot_dir(explicit: str | Path | None = None) -> Path:
    """Locate the local Terminal Wrench snapshot directory, complete or not, without any network.

    Reads the snapshot pointer straight out of the HF cache (``refs/main``), so a
    partially-downloaded corpus resolves fine and the missing transcripts surface through
    :func:`iter_traces` rather than here. Fails loudly if the corpus was never fetched at all;
    never downloads implicitly.
    """
    if explicit is not None:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"terminal wrench snapshot not found at {path}")
        return path

    repo_cache = Path(HF_HUB_CACHE) / ("datasets--" + TERMINAL_WRENCH_REPO.replace("/", "--"))
    ref = repo_cache / "refs" / "main"
    if not ref.exists():
        raise FileNotFoundError(
            f"terminal wrench not cached under {repo_cache}. Fetch it into the HF cache first, "
            "or pass an explicit snapshot dir."
        )
    snapshot = repo_cache / "snapshots" / ref.read_text().strip()
    if not snapshot.exists():
        raise FileNotFoundError(f"terminal wrench snapshot dir missing at {snapshot}")
    return snapshot


def load_index(snapshot_dir: str | Path) -> list[LabelRecord]:
    """Read the index into LabelRecord rows.

    Include every indexed row whether its transcript is present or not.
    """
    index_path = Path(snapshot_dir) / "index" / "trajectories.json"
    raw = json.loads(index_path.read_text())
    return [
        LabelRecord(
            task_id=str(entry["task_id"]),
            model=str(entry["model"]),
            tree_name=str(entry["tree_name"]),
            trajectory_label=str(entry["trajectory_label"]),
            classification=entry.get("classification"),
            reward=entry.get("reward"),
            exploit_attempted=bool(entry.get("exploit_attempted")),
            observed_categories=tuple(entry.get("observed_categories") or ()),
            key_hack_steps=tuple(entry.get("key_hack_steps") or ()),
            brief_exploit_summary=entry.get("brief_exploit_summary"),
            explanation=entry.get("explanation"),
        )
        for entry in raw
    ]


def trajectory_path(snapshot_dir: str | Path, record: LabelRecord) -> Path:
    """Return the transcript path for a record, which may be absent in a partial download."""
    return (
        Path(snapshot_dir)
        / "tasks"
        / record.task_id
        / record.model
        / record.tree_name
        / record.trajectory_label
        / "trial"
        / "agent"
        / "trajectory.json"
    )


def adapt_transcript(transcript: dict[str, Any], record: LabelRecord) -> Trace:
    """Adapt a raw transcript dict into a Trace of steps.

    Preserve the shell commands and observations needed by the reading aid.
    """
    steps: list[TraceStep] = []
    for raw_step in transcript.get("steps") or []:
        commands: list[Command] = []
        for call in raw_step.get("tool_calls") or []:
            function_name = str(call.get("function_name", ""))
            arguments = call.get("arguments")
            keystrokes = str(arguments.get("keystrokes", "")) if isinstance(arguments, dict) else ""
            commands.append(
                Command(
                    function_name=function_name,
                    keystrokes=keystrokes,
                    is_report=function_name in _REPORT_TOOLS,
                )
            )
        step_id = raw_step.get("step_id")
        steps.append(
            TraceStep(
                step_id=step_id if isinstance(step_id, int) else None,
                message=str(raw_step.get("message") or ""),
                commands=tuple(commands),
                observation=_observation_text(raw_step.get("observation")),
            )
        )
    return Trace(record=record, steps=tuple(steps))


def load_trace(snapshot_dir: str | Path, record: LabelRecord) -> Trace:
    """Read and adapt one record's transcript. Assumes it is present (see :func:`iter_traces`)."""
    path = trajectory_path(snapshot_dir, record)
    return adapt_transcript(json.loads(path.read_text()), record)


def iter_traces(
    snapshot_dir: str | Path, records: list[LabelRecord] | None = None
) -> Iterator[tuple[LabelRecord, Trace | None]]:
    """Yield each record with its trace, or None when the transcript is absent.

    Count partial-download misses explicitly so no index row is silently dropped. Pass records to
    iterate a subset (e.g. only hacks); defaults to the whole index.
    """
    if records is None:
        records = load_index(snapshot_dir)
    for record in records:
        path = trajectory_path(snapshot_dir, record)
        if not path.exists():
            yield record, None
            continue
        yield record, adapt_transcript(json.loads(path.read_text()), record)


def sample_traces(
    snapshot_dir: str | Path,
    n: int,
    *,
    only_hacks: bool = True,
    records: list[LabelRecord] | None = None,
) -> list[Trace]:
    """Give me up to ``n`` real transcripts to read (hacks by default), skipping any not on disk.

    Iterates the index in order so the sample is deterministic. Returns fewer than ``n`` only when
    the partial download does not contain enough present transcripts.
    """
    if records is None:
        records = load_index(snapshot_dir)
    if only_hacks:
        records = [record for record in records if record.is_hack]
    out: list[Trace] = []
    for record in records:
        if len(out) >= n:
            break
        path = trajectory_path(snapshot_dir, record)
        if path.exists():
            out.append(adapt_transcript(json.loads(path.read_text()), record))
    return out


def parse_write_targets(keystrokes: str) -> list[str]:
    """Extract the filesystem paths a shell snippet writes to. Best-effort, conservative.

    De-duplicates while preserving first-seen order, and drops empties, ``$var`` targets, and flags.
    """
    if not keystrokes.strip():
        return []
    text = _strip_heredocs(keystrokes)
    targets: list[str] = []
    for segment in split_commands(text):
        targets.extend(_segment_write_targets(segment))
    seen: set[str] = set()
    ordered: list[str] = []
    for target in targets:
        cleaned = _clean_path(target)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            ordered.append(cleaned)
    return ordered


def _observation_text(observation: object) -> str:
    """Flatten a step's ``observation`` into terminal text.

    Normally ``{"results": [{"content": "..."}]}``, but a transcript may store the observation as a
    JSON-encoded string, so a string that parses as JSON is decoded first, and one that does not is
    treated as the terminal text itself.
    """
    if observation is None:
        return ""
    if isinstance(observation, str):
        observation = _maybe_json(observation)
    if isinstance(observation, dict):
        results = observation.get("results") or []
        return "\n".join(
            str(result.get("content", "")) for result in results if isinstance(result, dict)
        )
    if isinstance(observation, str):
        return observation
    return str(observation)


def _maybe_json(text: str) -> object:
    """Parse ``text`` as JSON if it plausibly is some; otherwise return it unchanged."""
    stripped = text.strip()
    if not stripped or stripped[0] not in "{[":
        return text
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return text


def _strip_heredocs(text: str) -> str:
    """Drop heredoc bodies, keeping the header line so its redirection target is still parsed."""
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        match = _HEREDOC_HEADER_RE.search(line)
        if match:
            tag = match.group(2)
            i += 1
            while i < len(lines) and lines[i].strip() != tag:
                i += 1
            # Skip the closing tag line too (do not emit body or delimiter).
        i += 1
    return "\n".join(out)


def split_commands(text: str) -> list[str]:
    """Split a shell snippet into segments on ``;``, ``&&``, ``||``, ``|`` and newline.

    Quote-blind by design, so a ``python -c "a; b"`` payload splits mid-string. Fine for asking what
    argv[0] of each segment is, wrong for asking what a payload does.
    """
    text = re.sub(r"&&|\|\|", ";", text)
    return [segment.strip() for segment in re.split(r"[;\n|]", text) if segment.strip()]


def _segment_write_targets(segment: str) -> list[str]:
    targets: list[str] = [match.group(1) for match in _REDIRECT_RE.finditer(segment)]

    argv = strip_command_prefixes(split_argv(segment))
    if not argv:
        return targets

    cmd = posixpath.basename(argv[0])
    rest = argv[1:]
    non_flags = [arg for arg in rest if not arg.startswith("-")]

    if cmd == "tee":
        targets.extend(non_flags)
    elif cmd in {"cp", "mv", "install", "rsync"} and len(non_flags) >= _MIN_ARGS_FOR_LAST_TARGET:
        targets.append(non_flags[-1])
    elif cmd == "ln" and non_flags:
        # ln [-s] TARGET LINK_NAME -> the link name (last arg) is what gets created.
        targets.append(
            non_flags[-1] if len(non_flags) >= _MIN_ARGS_FOR_LAST_TARGET else non_flags[0]
        )
    elif cmd == "sed" and any(arg == "-i" or arg.startswith("-i") for arg in rest):
        # sed -i SCRIPT FILE... -> the script is the first non-flag, the rest are edited files.
        targets.extend(non_flags[1:] if len(non_flags) >= _MIN_ARGS_FOR_LAST_TARGET else non_flags)
    elif cmd in {"chmod", "chown", "chgrp"} and len(non_flags) >= _MIN_ARGS_FOR_LAST_TARGET:
        # first non-flag is the mode/owner; the rest are the paths being modified.
        targets.extend(non_flags[1:])
    elif cmd in {"touch", "mkdir", "rm", "rmdir", "unlink", "truncate", "mktemp"}:
        targets.extend(non_flags)
    elif cmd == "dd":
        targets.extend(arg[3:] for arg in rest if arg.startswith("of="))
    elif cmd in {"python", "python3", "python2"}:
        targets.extend(_python_write_targets(segment))

    return targets


def _python_write_targets(segment: str) -> list[str]:
    """Catch ``open('path', 'w')`` / ``Path('path').write_*`` inside ``python -c '...'``."""
    if "-c" not in segment.split():
        return []
    found: list[str] = []
    for match in _PY_OPEN_RE.finditer(segment):
        window = segment[match.start() : match.start() + 120]
        if _PY_WRITE_HINT_RE.search(window) or ".write_" in window:
            found.append(match.group(1))
    return found


def strip_command_prefixes(argv: list[str]) -> list[str]:
    """Drop wrappers in front of the real command (``sudo``, ``env``, ``VAR=x``, ``timeout N``)."""
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in _COMMAND_PREFIXES:
            i += 1
        elif token == "env" or (
            "=" in token and not token.startswith("-") and "/" not in token.split("=")[0]
        ):
            # env, or a leading VAR=value assignment.
            i += 1
        elif token == "timeout":
            i += 2  # timeout DURATION cmd ...
        else:
            break
    return argv[i:]


def split_argv(segment: str) -> list[str]:
    """Tokenise one segment into argv, falling back to whitespace splitting on unbalanced quotes."""
    try:
        return shlex.split(segment, posix=True)
    except ValueError:
        # Unbalanced quotes -> whitespace tokens, so one malformed command can't abort the parse.
        return segment.split()


def _clean_path(token: str) -> str:
    token = token.strip().strip("\"'")
    if not token or token.startswith(("$", "-")):
        return ""
    return token
