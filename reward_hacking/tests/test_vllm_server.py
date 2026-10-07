"""CPU tests for the subprocess lifecycle around the vLLM OpenAI server."""

from __future__ import annotations

import errno
import json
import os
import signal
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Self
from urllib.error import URLError

import pytest

from reward_hacking import vllm_server
from reward_hacking.ladder.thinking import QWEN_THINKING_REASONING_CONFIG
from reward_hacking.vllm_server import VLLMServer, VLLMServerConfig, VLLMServerStartupError

if TYPE_CHECKING:
    import subprocess
    from pathlib import Path


class _HealthResponse:
    """Minimal response object matching the status field used by urllib probing."""

    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        del args


def _fake_server_executable(tmp_path: Path) -> Path:
    """Create a quiet executable that stays alive until its process group is reaped."""
    executable = tmp_path / "fake-vllm"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import signal\n"
        "import time\n"
        "signal.signal(signal.SIGTERM, lambda signum, frame: raise_system_exit())\n"
        "def raise_system_exit():\n"
        "    raise SystemExit(0)\n"
        "while True:\n"
        "    time.sleep(1)\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


@pytest.fixture
def fake_vllm(tmp_path: Path) -> Path:
    """Return a fake executable without touching a real GPU or vLLM engine."""
    return _fake_server_executable(tmp_path)


def _config(fake_vllm: Path) -> VLLMServerConfig:
    """Build a deliberately explicit, small test configuration."""
    return VLLMServerConfig(
        model="Qwen/test-model",
        revision="test-revision",
        max_model_len=4096,
        gpu_memory_utilization=0.55,
        language_model_only=True,
        host="127.0.0.1",
        port=8000,
        executable=str(fake_vllm),
        startup_timeout_seconds=1.0,
        health_poll_interval_seconds=0.01,
        health_request_timeout_seconds=0.2,
    )


def _ready_response(*args: object, **kwargs: object) -> _HealthResponse:
    """Return a successful fake HTTP response without opening a socket."""
    del args, kwargs
    return _HealthResponse(200)


def _connection_refused() -> URLError:
    """Build the exact urllib error produced when startup probes a closed port."""
    return URLError(ConnectionRefusedError(errno.ECONNREFUSED, "connection refused"))


def test_command_contains_verified_serve_flags(fake_vllm: Path) -> None:
    config = _config(fake_vllm)

    assert config.command() == [
        str(fake_vllm),
        "serve",
        "Qwen/test-model",
        "--revision",
        "test-revision",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--max-model-len",
        "4096",
        "--gpu-memory-utilization",
        "0.55",
        "--language-model-only",
    ]


def test_command_renders_optional_attention_and_cache_settings(fake_vllm: Path) -> None:
    config = replace(
        _config(fake_vllm),
        attention_backend="FLASHINFER",
        enable_prefix_caching=True,
        mamba_cache_mode="align",
        max_num_batched_tokens=8192,
        kv_cache_dtype="fp8",
        quantization="fp8_per_tensor",
    )

    assert config.command() == [
        str(fake_vllm),
        "serve",
        "Qwen/test-model",
        "--revision",
        "test-revision",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--max-model-len",
        "4096",
        "--gpu-memory-utilization",
        "0.55",
        "--language-model-only",
        "--attention-backend",
        "FLASHINFER",
        "--enable-prefix-caching",
        "--kv-cache-dtype",
        "fp8",
        "--mamba-cache-mode",
        "align",
        "--max-num-batched-tokens",
        "8192",
        "--quantization",
        "fp8_per_tensor",
    ]


def test_command_renders_lora_modules_and_multimodal_limit(tmp_path: Path, fake_vllm: Path) -> None:
    adapters = (tmp_path / "s1", tmp_path / "s2")
    for adapter in adapters:
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "allenai/OLMo-3-7B"}), encoding="utf-8"
        )
    config = replace(
        _config(fake_vllm),
        enable_lora=True,
        max_lora_rank=64,
        lora_modules=(
            ("somo-olmo-s1", str(adapters[0])),
            ("somo-olmo-s2", str(adapters[1])),
        ),
        limit_mm_per_prompt='{"image": 0}',
    )

    assert config.command()[-8:] == [
        "--enable-lora",
        "--max-lora-rank",
        "64",
        "--lora-modules",
        f"somo-olmo-s1={adapters[0]}",
        f"somo-olmo-s2={adapters[1]}",
        "--limit-mm-per-prompt",
        '{"image": 0}',
    ]


def test_command_renders_reasoning_config(fake_vllm: Path) -> None:
    config = replace(_config(fake_vllm), reasoning_config=QWEN_THINKING_REASONING_CONFIG)

    assert config.command()[-2:] == [
        "--reasoning-config",
        json.dumps(dict(QWEN_THINKING_REASONING_CONFIG), separators=(",", ":")),
    ]


def test_command_enables_sleep_mode_for_runtime_lora_training(fake_vllm: Path) -> None:
    config = replace(
        _config(fake_vllm),
        enable_lora=True,
        max_lora_rank=16,
        enable_sleep_mode=True,
    )

    assert "--enable-lora" in config.command()
    assert "--max-lora-rank" in config.command()
    assert "--enable-sleep-mode" in config.command()


def test_owned_server_logs_child_output_and_sets_runtime_lora_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    executable = tmp_path / "fake-vllm-log"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import os\n"
        "import signal\n"
        "import time\n"
        "print(os.environ.get('VLLM_SERVER_DEV_MODE'), flush=True)\n"
        "print(os.environ.get('VLLM_ALLOW_RUNTIME_LORA_UPDATING'), flush=True)\n"
        "signal.signal(signal.SIGTERM, lambda signum, frame: raise_system_exit())\n"
        "def raise_system_exit():\n"
        "    raise SystemExit(0)\n"
        "while True:\n"
        "    time.sleep(1)\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    log_path = tmp_path / "logs" / "server.log"
    config = replace(
        _config(executable),
        enable_lora=True,
        max_lora_rank=16,
        enable_sleep_mode=True,
        log_path=log_path,
    )

    def wait_for_child_output(server: VLLMServer) -> None:
        del server
        deadline = time.monotonic() + 2
        while not log_path.exists() or log_path.read_text(encoding="utf-8").splitlines() != [
            "1",
            "1",
        ]:
            if time.monotonic() >= deadline:
                raise AssertionError("fake server did not write its environment to the log")
            time.sleep(0.01)

    monkeypatch.setattr(VLLMServer, "_wait_until_healthy", wait_for_child_output)

    with VLLMServer(config):
        pass

    assert log_path.read_text(encoding="utf-8").splitlines() == ["1", "1"]


def test_local_snapshot_may_omit_revision(tmp_path: Path, fake_vllm: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    config = VLLMServerConfig(
        model=str(snapshot),
        revision=None,
        max_model_len=4096,
        gpu_memory_utilization=0.55,
        executable=str(fake_vllm),
    )

    assert "--revision" not in config.command()


def test_remote_model_requires_revision(fake_vllm: Path) -> None:
    with pytest.raises(ValueError, match="revision must be explicit for Hub models"):
        VLLMServerConfig(
            model="Qwen/test-model",
            revision=None,
            max_model_len=4096,
            gpu_memory_utilization=0.55,
            executable=str(fake_vllm),
        )


def test_ipv6_host_is_bracketed_in_urls(fake_vllm: Path) -> None:
    config = replace(_config(fake_vllm), host="::1")

    assert config.base_url == "http://[::1]:8000"
    assert config.health_url == "http://[::1]:8000/health"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"model": ""}, "model must be non-empty"),
        ({"revision": ""}, "revision must be non-empty when supplied"),
        ({"max_model_len": 0}, "max_model_len must be positive"),
        ({"gpu_memory_utilization": 0.0}, "gpu_memory_utilization must be in"),
        ({"host": ""}, "host must be non-empty"),
        ({"port": 0}, "port must be between"),
        ({"port": 65536}, "port must be between"),
        ({"max_num_seqs": 0}, "max_num_seqs must be positive"),
        ({"attention_backend": ""}, "attention_backend must be non-empty"),
        ({"mamba_cache_mode": ""}, "mamba_cache_mode must be non-empty"),
        ({"max_num_batched_tokens": 0}, "max_num_batched_tokens must be positive"),
        ({"max_lora_rank": 0}, "max_lora_rank must be positive"),
        (
            {"lora_modules": (("adapter", "/models/adapter"),)},
            "lora_modules require enable_lora=True",
        ),
        (
            {
                "enable_lora": True,
                "max_lora_rank": 64,
                "lora_modules": (("adapter", "/models/a"), ("adapter", "/models/b")),
            },
            "lora module names must be unique",
        ),
        (
            {
                "enable_lora": True,
                "max_lora_rank": 64,
                "lora_modules": (("", "/models/adapter"),),
            },
            "lora module names and paths must be non-empty",
        ),
        (
            {"enable_lora": True, "lora_modules": (("adapter", "/models/adapter"),)},
            "lora_modules require max_lora_rank",
        ),
        ({"health_path": "health"}, "health_path must start with"),
        ({"startup_timeout_seconds": 0.0}, "startup_timeout_seconds must be positive"),
        ({"health_poll_interval_seconds": 0.0}, "health_poll_interval_seconds must be positive"),
        (
            {"health_request_timeout_seconds": 0.0},
            "health_request_timeout_seconds must be positive",
        ),
        ({"shutdown_timeout_seconds": 0.0}, "shutdown_timeout_seconds must be positive"),
    ],
)
def test_invalid_config_settings_fail_fast(
    fake_vllm: Path, changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_config(fake_vllm), **changes)


def test_server_retries_connection_refused_then_starts(
    fake_vllm: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probes = 0

    def probe(*args: object, **kwargs: object) -> _HealthResponse:
        nonlocal probes
        del args, kwargs
        probes += 1
        if probes < 3:
            raise _connection_refused()
        return _HealthResponse(200)

    monkeypatch.setattr(vllm_server.urllib.request, "urlopen", probe)
    with VLLMServer(_config(fake_vllm)) as server:
        assert server.base_url == "http://127.0.0.1:8000"
        assert server.process.poll() is None
    assert probes == 3


def test_server_reaps_process_group_on_context_exit(
    fake_vllm: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(vllm_server.urllib.request, "urlopen", _ready_response)
    with VLLMServer(_config(fake_vllm)) as server:
        process = server.process
        process_group = os.getpgid(process.pid)

    assert process.poll() is not None
    with pytest.raises(ProcessLookupError):
        os.getpgid(process_group)


def test_server_reaps_process_group_when_training_raises(
    fake_vllm: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(vllm_server.urllib.request, "urlopen", _ready_response)
    processes: list[subprocess.Popen[bytes]] = []
    process_groups: list[int] = []

    def run_failing_training_step() -> None:
        with VLLMServer(_config(fake_vllm)) as server:
            processes.append(server.process)
            process_groups.append(os.getpgid(server.process.pid))
            raise RuntimeError("synthetic training failure")

    with pytest.raises(RuntimeError, match="synthetic training failure"):
        run_failing_training_step()

    process = processes[0]
    assert process.poll() is not None
    with pytest.raises(ProcessLookupError):
        os.getpgid(process_groups[0])


def test_non_200_health_response_fails_without_retry(
    fake_vllm: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probes = 0

    def probe(*args: object, **kwargs: object) -> _HealthResponse:
        nonlocal probes
        del args, kwargs
        probes += 1
        return _HealthResponse(500)

    monkeypatch.setattr(vllm_server.urllib.request, "urlopen", probe)

    with pytest.raises(VLLMServerStartupError, match="health endpoint returned HTTP 500"):
        VLLMServer(_config(fake_vllm)).start()

    assert probes == 1


def test_startup_failure_reaps_process_and_restores_signal_handler(
    fake_vllm: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(fake_vllm)
    config = replace(config, startup_timeout_seconds=0.05)
    previous_handler = signal.getsignal(signal.SIGTERM)

    def refuse(*args: object, **kwargs: object) -> _HealthResponse:
        del args, kwargs
        raise _connection_refused()

    monkeypatch.setattr(vllm_server.urllib.request, "urlopen", refuse)
    server = VLLMServer(config)
    with pytest.raises(VLLMServerStartupError, match="did not become healthy"):
        server.start()

    assert server.process.poll() is not None
    assert signal.getsignal(signal.SIGTERM) is previous_handler


def test_sigterm_reaps_process_group_and_restores_handler(
    fake_vllm: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(vllm_server.urllib.request, "urlopen", _ready_response)
    previous_handler = signal.getsignal(signal.SIGTERM)

    with pytest.raises(SystemExit, match=str(128 + signal.SIGTERM)), VLLMServer(_config(fake_vllm)):
        os.kill(os.getpid(), signal.SIGTERM)

    assert signal.getsignal(signal.SIGTERM) is previous_handler
