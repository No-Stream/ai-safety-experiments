"""Lifecycle management for an external vLLM OpenAI-compatible server.

The agent harness keeps one ``vllm.LLM`` in process today.  This module owns the other serving
shape: a ``vllm serve`` child with a stable HTTP endpoint.  The child is a process-group leader so
that its engine workers are reaped together when the context exits or the parent receives SIGTERM.
"""

from __future__ import annotations

import errno
import logging
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import FrameType

logger = logging.getLogger(__name__)
_HTTP_OK = 200
_MAX_TCP_PORT = 65535


class VLLMServerStartupError(RuntimeError):
    """The external server failed to become healthy."""


@dataclass(frozen=True, slots=True)
class VLLMServerConfig:
    """All settings that determine one externally served vLLM engine.

    ``max_model_len`` and ``gpu_memory_utilization`` are required rather than defaulted because
    both allocate the shared GPU's finite memory.  The caller resolves them from the run's actual
    device and sampling budget, just as the in-process backend does.  ``revision`` is optional only
    for a local snapshot, which has no Hub revision; Hub models must pass the revision explicitly.
    """

    model: str
    revision: str | None
    max_model_len: int
    gpu_memory_utilization: float
    host: str = "127.0.0.1"
    port: int = 8000
    language_model_only: bool = False
    max_num_seqs: int | None = None
    dtype: str | None = None
    attention_backend: str | None = None
    enable_prefix_caching: bool = False
    kv_cache_dtype: str | None = None
    mamba_cache_mode: str | None = None
    max_num_batched_tokens: int | None = None
    quantization: str | None = None
    executable: str = "vllm"
    startup_timeout_seconds: float = 600.0
    health_poll_interval_seconds: float = 0.25
    health_request_timeout_seconds: float = 1.0
    shutdown_timeout_seconds: float = 10.0
    health_path: str = "/health"

    def __post_init__(self) -> None:
        """Reject malformed launch settings before starting a process."""
        self._validate_model_settings()
        self._validate_network_settings()
        self._validate_optional_engine_settings()
        self._validate_time_settings()

    def _validate_model_settings(self) -> None:
        """Validate model identity and required memory settings."""
        if not self.model:
            raise ValueError("model must be non-empty")
        if self.revision == "":
            raise ValueError("revision must be non-empty when supplied")
        if self.revision is None and not Path(self.model).is_dir():
            raise ValueError("revision must be explicit for Hub models")
        if self.max_model_len <= 0:
            raise ValueError("max_model_len must be positive")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")

    def _validate_network_settings(self) -> None:
        """Validate the local HTTP bind address."""
        if not self.host:
            raise ValueError("host must be non-empty")
        if not 1 <= self.port <= _MAX_TCP_PORT:
            raise ValueError(f"port must be between 1 and {_MAX_TCP_PORT}")

    def _validate_optional_engine_settings(self) -> None:
        """Validate optional engine and health endpoint settings."""
        if self.max_num_seqs is not None and self.max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be positive when supplied")
        if self.attention_backend == "":
            raise ValueError("attention_backend must be non-empty when supplied")
        if self.mamba_cache_mode == "":
            raise ValueError("mamba_cache_mode must be non-empty when supplied")
        if self.max_num_batched_tokens is not None and self.max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens must be positive when supplied")
        if not self.health_path.startswith("/"):
            raise ValueError("health_path must start with '/'")

    def _validate_time_settings(self) -> None:
        """Validate startup, probe, and shutdown timeouts."""
        if self.startup_timeout_seconds <= 0:
            raise ValueError("startup_timeout_seconds must be positive")
        if self.health_poll_interval_seconds <= 0:
            raise ValueError("health_poll_interval_seconds must be positive")
        if self.health_request_timeout_seconds <= 0:
            raise ValueError("health_request_timeout_seconds must be positive")
        if self.shutdown_timeout_seconds <= 0:
            raise ValueError("shutdown_timeout_seconds must be positive")

    @property
    def base_url(self) -> str:
        """The HTTP origin used by the OpenAI-compatible client."""
        host = self.host if ":" not in self.host or self.host.startswith("[") else f"[{self.host}]"
        return f"http://{host}:{self.port}"

    @property
    def health_url(self) -> str:
        """The vLLM readiness URL."""
        return f"{self.base_url}{self.health_path}"

    def command(self) -> list[str]:
        """Build the verified ``vllm serve`` command line."""
        command = [self.executable, "serve", self.model]
        if self.revision is not None:
            command.extend(["--revision", self.revision])
        command.extend(
            [
                "--host",
                self.host,
                "--port",
                str(self.port),
                "--max-model-len",
                str(self.max_model_len),
                "--gpu-memory-utilization",
                str(self.gpu_memory_utilization),
            ]
        )
        if self.language_model_only:
            command.append("--language-model-only")
        for option, value in (
            ("--max-num-seqs", self.max_num_seqs),
            ("--dtype", self.dtype),
            ("--attention-backend", self.attention_backend),
        ):
            if value is not None:
                command.extend([option, str(value)])
        if self.enable_prefix_caching:
            command.append("--enable-prefix-caching")
        for option, value in (
            ("--kv-cache-dtype", self.kv_cache_dtype),
            ("--mamba-cache-mode", self.mamba_cache_mode),
            ("--max-num-batched-tokens", self.max_num_batched_tokens),
            ("--quantization", self.quantization),
        ):
            if value is not None:
                command.extend([option, str(value)])
        return command


class VLLMServer:
    """Start and reap one external vLLM server process group."""

    def __init__(self, config: VLLMServerConfig) -> None:
        """Create a stopped server wrapper; :meth:`start` launches the child."""
        self.config = config
        self._process: subprocess.Popen[bytes] | None = None
        self._previous_sigterm_handler: int | Callable[[int, FrameType | None], object] | None = (
            None
        )

    def __enter__(self) -> Self:
        """Start the server and wait until its HTTP health endpoint is ready."""
        return self.start()

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception_value: BaseException | None,
        traceback: object,
    ) -> None:
        """Restore signal handling and terminate the entire process group."""
        del exception_type, exception_value, traceback
        self.stop()

    @property
    def process(self) -> subprocess.Popen[bytes]:
        """The launched child, including after it has been reaped."""
        if self._process is None:
            raise RuntimeError("vLLM server has not been started")
        return self._process

    @property
    def base_url(self) -> str:
        """The HTTP origin to pass to an OpenAI-compatible client."""
        return self.config.base_url

    def start(self) -> Self:
        """Launch the process group and wait for readiness, cleaning up on every failure."""
        if self._process is not None:
            raise RuntimeError("vLLM server has already been started")
        self._process = subprocess.Popen(  # noqa: S603 - command is built from typed config fields
            self.config.command(),
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            self._install_signal_handler()
            self._wait_until_healthy()
        except BaseException:
            self.stop()
            raise
        logger.info(
            "vLLM server ready: model=%s revision=%s base_url=%s max_model_len=%s "
            "gpu_memory_utilization=%s language_model_only=%s",
            self.config.model,
            self.config.revision,
            self.base_url,
            self.config.max_model_len,
            self.config.gpu_memory_utilization,
            self.config.language_model_only,
        )
        return self

    def stop(self) -> None:
        """Terminate and reap the process group, escalating only after the grace period."""
        self._restore_signal_handler()
        if self._process is None:
            return
        process = self._process
        if process.poll() is not None:
            process.wait()
            return
        self._signal_process_group(signal.SIGTERM)
        try:
            process.wait(timeout=self.config.shutdown_timeout_seconds)
        except subprocess.TimeoutExpired:
            logger.warning("vLLM server did not exit after SIGTERM; killing process group")
            self._signal_process_group(signal.SIGKILL)
            process.wait()

    def _install_signal_handler(self) -> None:
        """Install a parent-side SIGTERM handler while this server owns a process group."""
        if self._previous_sigterm_handler is not None:
            raise RuntimeError("vLLM server signal handler is already installed")

        def handle_sigterm(signum: int, frame: FrameType | None) -> None:
            del frame
            self.stop()
            raise SystemExit(128 + signum)

        self._previous_sigterm_handler = signal.signal(signal.SIGTERM, handle_sigterm)

    def _restore_signal_handler(self) -> None:
        """Restore the caller's SIGTERM handler exactly once."""
        if self._previous_sigterm_handler is None:
            return
        signal.signal(signal.SIGTERM, self._previous_sigterm_handler)
        self._previous_sigterm_handler = None

    def _signal_process_group(self, signum: signal.Signals) -> None:
        """Send a signal to the child process group, tolerating a raced child exit."""
        process = self.process
        try:
            os.killpg(os.getpgid(process.pid), signum)
        except ProcessLookupError:
            logger.debug("vLLM process group exited before signal %s", signum)

    def _wait_until_healthy(self) -> None:
        """Probe readiness; only connection-refused errors are startup-transient."""
        deadline = time.monotonic() + self.config.startup_timeout_seconds
        while True:
            if self.process.poll() is not None:
                raise VLLMServerStartupError(
                    f"vLLM serve exited during startup with code {self.process.returncode}"
                )
            try:
                with urllib.request.urlopen(  # noqa: S310 - URL is built from local config
                    self.config.health_url, timeout=self.config.health_request_timeout_seconds
                ) as response:
                    if response.status != _HTTP_OK:
                        raise VLLMServerStartupError(
                            f"vLLM health endpoint returned HTTP {response.status}"
                        )
                    return
            except urllib.error.HTTPError as exc:
                raise VLLMServerStartupError(
                    f"vLLM health endpoint returned HTTP {exc.code}"
                ) from exc
            except urllib.error.URLError as exc:
                if not _is_connection_refused(exc):
                    raise VLLMServerStartupError(f"vLLM health probe failed: {exc.reason}") from exc
            except ConnectionRefusedError as exc:
                del exc
            if time.monotonic() >= deadline:
                raise VLLMServerStartupError(
                    f"vLLM server did not become healthy within "
                    f"{self.config.startup_timeout_seconds:.1f}s"
                )
            time.sleep(self.config.health_poll_interval_seconds)


def _is_connection_refused(error: urllib.error.URLError) -> bool:
    """Return whether urllib wrapped the one startup error that is safe to retry."""
    reason = error.reason
    return isinstance(reason, OSError) and reason.errno == errno.ECONNREFUSED


__all__ = ["VLLMServer", "VLLMServerConfig", "VLLMServerStartupError"]
