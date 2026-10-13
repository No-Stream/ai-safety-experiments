"""The ``codex`` CLI text-generation backend, split out of ``model_backend`` (which re-exports it)."""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger("reward_hacking.model_backend")


DEFAULT_CODEX_MODEL = "openai.gpt-5.6-luna"
DEFAULT_CODEX_TIMEOUT = 300.0


class CodexBackend:
    """Text-generation backend that shells the ``codex`` CLI for one next action per prompt.

    ``codex`` runs GPT-5.x on Bedrock; we ask it only for text and use only the text. Our own
    bubblewrap jail runs any shell command the policy emits, so codex must execute nothing itself --
    which is why every call passes ``-s read-only`` (codex's own sandbox denies it writes and
    command execution). Three invocation details are load-bearing and each fails silently or hangs
    if dropped, verified against this box's ``codex`` build (see global CLAUDE.md "Codex / GPT-5.x
    subagents"): capture the reply with ``-o <file>`` (``--output-last-message``) rather than
    parsing stdout, which interleaves session chrome; close stdin with ``DEVNULL`` or ``codex exec``
    blocks forever reading it; and pass ``--skip-git-repo-check`` because the scratch ``-C`` workdir
    is not a git repo. A non-zero exit or a missing output file raises rather than returning a
    wrong-but-quiet empty completion.

    **What we ask for is not what the model receives, and that is a confound.** The prompt is handed
    to ``codex exec`` as its task argument, so the completion comes out of the codex agent harness:
    its own system prompt, its own tool loop, and read commands against the scratch workdir, since
    ``-s read-only`` constrains writes and command execution rather than the scaffolding. That is
    structurally unlike the lone bare user turn :func:`converse_request` renders for the Converse
    and batch transports, and there is no bare-completion mode here. So a ``codex`` record and a
    ``bedrock-converse`` record for the same item are not the same elicitation, and pooling or
    comparing them would put the scaffolding difference inside what reads as a model difference --
    the failure the ``transport`` field exists to make detectable, which is a label, not a fix. The
    frontier model is reachable both ways, so where a bare completion of it is what is wanted, call
    it through :class:`BedrockBackend`; use this backend when the scaffolding is part of the claim.

    Sequential by design: the codex app-server serialises brokered calls, so a thread pool buys
    nothing. For parallel fan-out, run separate ``codex exec`` processes instead.
    """

    transport = "codex"

    def __init__(
        self,
        model_id: str = DEFAULT_CODEX_MODEL,
        *,
        codex_bin: str | None = None,
        reasoning_effort: str | None = None,
        timeout: float = DEFAULT_CODEX_TIMEOUT,
    ) -> None:
        """Resolve the codex binary to an absolute path and freeze the per-call configuration."""
        resolved = codex_bin or shutil.which("codex")
        if resolved is None:
            raise RuntimeError("codex CLI not found on PATH; install it or pass codex_bin=<path>")
        self.model_id = model_id
        self._codex_bin = resolved
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout
        logger.info(
            "CodexBackend ready: bin=%s model=%s effort=%s timeout=%.0fs",
            resolved,
            model_id,
            reasoning_effort,
            timeout,
        )

    def _build_argv(self, prompt: str, *, workdir: str, out_path: str) -> list[str]:
        """Build the ``codex exec`` argv for one prompt, capturing the reply via ``-o``.

        The prompt goes after an end-of-options ``--`` because ``codex exec`` takes it as a trailing
        positional and its usage is ``codex exec [OPTIONS] [PROMPT]`` *or*
        ``codex exec [OPTIONS] <COMMAND> [ARGS]``. Without the separator two ordinary corpus items
        never reach the model: one opening with a hyphen is parsed as a flag, and one whose first
        word is ``review``, ``resume`` or ``help`` is swallowed as an ``exec`` subcommand -- both as
        a non-zero exit that reads like a CLI problem rather than a prompt problem. Verified against
        this box's build: ``codex exec -- review -x`` reports the top-level usage (so ``review``
        became the prompt), while ``codex exec review -x`` reports the review subcommand's.
        """
        argv = [
            self._codex_bin,
            "exec",
            "-s",
            "read-only",
            "-C",
            workdir,
            "--skip-git-repo-check",
            "-o",
            out_path,
            "-m",
            self.model_id,
        ]
        if self.reasoning_effort is not None:
            argv += ["-c", f"model_reasoning_effort={self.reasoning_effort}"]
        argv += ["--", prompt]
        return argv

    def _run_bounded(self, argv: list[str]) -> subprocess.CompletedProcess[str]:
        """Run codex under a wall-clock bound that kills the whole process tree on expiry.

        ``subprocess.run(timeout=...)`` would stop *waiting* but signals only the direct child,
        and the ``codex`` entry point is a shim that spawns the real binary in a grandchild:
        measured on this box, a timed-out call leaves two live codex processes, each still
        holding its Bedrock connection open and its tokens on the meter. ``start_new_session``
        makes the child a process-group leader so the expiry path can signal the whole group.

        The bound matters because codex itself has none we can reach. Its documented
        ``stream_idle_timeout_ms`` / ``stream_max_retries`` knobs are refused for the built-in
        ``amazon-bedrock`` provider (it accepts only ``base_url``, ``auth``, ``http_headers``,
        ``aws.profile`` and ``aws.region``), and an endpoint that completes the TCP handshake and
        then never answers leaves codex asleep on the socket rather than retrying.
        """
        with subprocess.Popen(  # noqa: S603 - trusted CLI, argv built from vetted parts
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        ) as proc:
            try:
                stdout, stderr = proc.communicate(timeout=self.timeout)
            except subprocess.TimeoutExpired as expired:
                # ProcessLookupError: the tree may exit between the expiry and the signal.
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait()
                raise RuntimeError(
                    f"codex exec exceeded {self.timeout:.0f}s and its process group was killed; "
                    "codex was still waiting on Bedrock, not computing"
                ) from expired
        return subprocess.CompletedProcess(argv, proc.returncode, stdout, stderr)

    def _generate_one(self, prompt: str) -> str:
        """Run codex once as a text generator and return its final assistant message."""
        with tempfile.TemporaryDirectory(prefix="codex-backend-") as workdir:
            out_path = Path(workdir) / "last_message.txt"
            argv = self._build_argv(prompt, workdir=workdir, out_path=str(out_path))
            completed = self._run_bounded(argv)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"codex exec failed (exit {completed.returncode}): "
                    f"{completed.stderr.strip()[:2000]}"
                )
            if not out_path.exists():
                raise RuntimeError(
                    "codex exec wrote no --output-last-message file; "
                    f"stderr: {completed.stderr.strip()[:2000]}"
                )
            return out_path.read_text()

    def generate(self, prompts: list[str]) -> list[str]:
        """Generate one completion per prompt, sequentially (the codex broker serialises)."""
        return [self._generate_one(prompt) for prompt in prompts]
