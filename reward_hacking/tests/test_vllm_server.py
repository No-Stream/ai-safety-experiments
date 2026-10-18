"""CPU tests for the subprocess lifecycle around the vLLM OpenAI server."""

from __future__ import annotations

import errno
import json
import os
import re
import signal
import sys
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Self
from urllib.error import URLError

import pytest

from grpo import gpu_memory
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


@pytest.fixture(autouse=True)
def avoid_real_cuda_sizing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep lifecycle tests on CPU; the focused test below checks the sizing boundary."""
    monkeypatch.setattr(gpu_memory, "_nvidia_smi_rows", lambda: [(0, "GPU-test", 1024.0, 32768.0)])


def _config(fake_vllm: Path) -> VLLMServerConfig:
    """Build a deliberately explicit, small test configuration."""
    return VLLMServerConfig(
        model="Qwen/test-model",
        revision="test-revision",
        max_model_len=4096,
        gpu_memory_utilization=0.55,
        hf_overrides={"head_dtype": "float32"},
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
        "--hf-overrides",
        '{"head_dtype":"float32"}',
    ]


def test_auto_fit_context_length_is_forwarded_to_vllm(fake_vllm: Path) -> None:
    config = replace(_config(fake_vllm), max_model_len=-1)

    command = config.command()

    assert command[command.index("--max-model-len") + 1] == "-1"


def test_command_renders_optional_attention_and_cache_settings(fake_vllm: Path) -> None:
    config = replace(
        _config(fake_vllm),
        attention_backend="FLASHINFER",
        enable_prefix_caching=True,
        mamba_cache_mode="align",
        mamba_ssm_cache_dtype="bfloat16",
        max_num_batched_tokens=8192,
        kv_cache_dtype="fp8",
        quantization="fp8_per_tensor",
        quantization_config={"ignore": ["*linear_attn.*", "lm_head"]},
        model="models/sampler-checkpoint",
        served_model_name="Qwen/test-model",
    )

    assert config.command() == [
        str(fake_vllm),
        "serve",
        "models/sampler-checkpoint",
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
        "--mamba-ssm-cache-dtype",
        "bfloat16",
        "--max-num-batched-tokens",
        "8192",
        "--quantization",
        "fp8_per_tensor",
        "--quantization-config",
        '{"ignore":["*linear_attn.*","lm_head"]}',
        "--served-model-name",
        "Qwen/test-model",
        "--hf-overrides",
        '{"head_dtype":"float32"}',
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

    command = config.command()
    assert command[command.index("--enable-lora") :] == [
        "--enable-lora",
        "--max-lora-rank",
        "64",
        "--lora-modules",
        f"somo-olmo-s1={adapters[0]}",
        f"somo-olmo-s2={adapters[1]}",
        "--limit-mm-per-prompt",
        '{"image": 0}',
        "--hf-overrides",
        '{"head_dtype":"float32"}',
    ]


def test_command_renders_reasoning_config(fake_vllm: Path) -> None:
    config = replace(_config(fake_vllm), reasoning_config=QWEN_THINKING_REASONING_CONFIG)

    command = config.command()
    assert command[command.index("--reasoning-config") + 1] == json.dumps(
        dict(QWEN_THINKING_REASONING_CONFIG), separators=(",", ":")
    )
    assert command[-2:] == ["--hf-overrides", '{"head_dtype":"float32"}']


def test_command_renders_explicit_hf_overrides(fake_vllm: Path) -> None:
    config = replace(
        _config(fake_vllm), hf_overrides={"head_dtype": "float32", "trust_remote_code": True}
    )

    command = config.command()

    assert json.loads(command[command.index("--hf-overrides") + 1]) == {
        "head_dtype": "float32",
        "trust_remote_code": True,
    }


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


def test_server_preserves_explicit_utilization(
    fake_vllm: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(vllm_server.urllib.request, "urlopen", _ready_response)
    with VLLMServer(_config(fake_vllm)) as server:
        assert server.config.gpu_memory_utilization == 0.55


def test_server_derives_inference_default(fake_vllm: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vllm_server.urllib.request, "urlopen", _ready_response)
    with VLLMServer(replace(_config(fake_vllm), gpu_memory_utilization=None)) as server:
        assert server.config.gpu_memory_utilization == pytest.approx((31 - 0.75 - 2.4) / 32)
        assert server.memory_provenance["launch_free_bytes"] == 31 * 1024**3


@pytest.mark.parametrize("requested_utilization", [None, 0.9, 0.01])
@pytest.mark.parametrize("free_gib", [1.2, 2.5, 3.0])
def test_server_checks_trainer_headroom_and_reaps_failure(
    fake_vllm: Path,
    monkeypatch: pytest.MonkeyPatch,
    free_gib: float,
    tmp_path: Path,
    requested_utilization: float | None,
) -> None:
    measurements = iter([1024.0, (32 - free_gib) * 1024, (32 - free_gib) * 1024])
    monkeypatch.setattr(
        gpu_memory,
        "_nvidia_smi_rows",
        lambda: [(0, "GPU-test", next(measurements), 32768.0)],
    )
    monkeypatch.setattr(vllm_server.urllib.request, "urlopen", _ready_response)
    server = VLLMServer(
        replace(
            _config(fake_vllm),
            gpu_memory_utilization=requested_utilization,
            trainer_vram_headroom_gib=2.5,
            log_path=tmp_path / "server.log",
        )
    )
    try:
        if free_gib < 2.5:
            with pytest.raises(
                VLLMServerStartupError, match=r"1.20 GiB.*2.50 GiB.*--gpu-memory-utilization"
            ) as error:
                server.start()
            suggestion = re.search(r"--gpu-memory-utilization ([0-9.]+)", str(error.value))
            assert suggestion is not None
            assert 0 < float(suggestion[1]) < (requested_utilization or (31 - 2.5 - 0.75) / 32)
            assert server.process.poll() is not None
            assert server.memory_provenance["headroom_satisfied"] is False
        else:
            with server:
                assert server.memory_provenance["ready_free_bytes"] == round(free_gib * 1024**3)
                assert server.memory_provenance["headroom_satisfied"] is True
        recorded = json.loads((tmp_path / "server.memory.json").read_text())
        assert recorded == server.memory_provenance
        expected_utilization = requested_utilization or (31 - 2.5 - 0.75) / 32
        if requested_utilization is None and free_gib < 2.5:
            expected_utilization -= (2.5 - free_gib) / 32 + 0.01
        assert recorded["gpu_memory_utilization"] == pytest.approx(expected_utilization)
        assert recorded["trainer_vram_headroom_gib"] == 2.5
    finally:
        server.stop()


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


def test_command_passes_speculative_json_without_changing_values(fake_vllm: Path) -> None:
    config = replace(
        _config(fake_vllm), speculative_config={"method": "mtp", "num_speculative_tokens": 2}
    )
    command = config.command()
    assert (
        json.loads(command[command.index("--speculative-config") + 1]) == config.speculative_config
    )


def _batched_tokens_flag(config: VLLMServerConfig) -> str | None:
    command = config.command()
    if "--max-num-batched-tokens" not in command:
        return None
    return command[command.index("--max-num-batched-tokens") + 1]


def test_speculative_drafting_raises_the_default_scheduled_token_budget(fake_vllm: Path) -> None:
    speculative = replace(
        _config(fake_vllm), speculative_config={"method": "mtp", "num_speculative_tokens": 2}
    )
    assert _batched_tokens_flag(speculative) == str(vllm_server.SPECULATIVE_MAX_NUM_BATCHED_TOKENS)
    assert _batched_tokens_flag(_config(fake_vllm)) is None
    explicit = replace(speculative, max_num_batched_tokens=4096)
    assert _batched_tokens_flag(explicit) == "4096"


@pytest.mark.parametrize("under_wsl", [True, False])
def test_server_environment_enables_pin_memory_only_on_wsl_and_preserves_runner_choice(
    under_wsl: bool, fake_vllm: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.delenv("VLLM_WSL2_ENABLE_PIN_MEMORY", raising=False)
    monkeypatch.setattr(vllm_server, "_running_under_wsl", lambda: under_wsl)
    environment = VLLMServer(
        replace(_config(fake_vllm), reasoning_config=QWEN_THINKING_REASONING_CONFIG)
    )._environment()
    assert environment["VLLM_USE_V2_MODEL_RUNNER"] == "1"
    assert environment.get("VLLM_WSL2_ENABLE_PIN_MEMORY") == ("1" if under_wsl else None)


@pytest.mark.parametrize(
    ("under_wsl", "speculative", "expected"),
    [
        (True, True, "max_split_size_mb:512,expandable_segments:False"),
        (True, False, "max_split_size_mb:512,expandable_segments:True"),
        (False, True, "max_split_size_mb:512,expandable_segments:True"),
    ],
)
def test_server_environment_disables_expandable_segments_for_speculative_drafts_on_wsl(
    under_wsl: bool,
    speculative: bool,
    expected: str,
    fake_vllm: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512,expandable_segments:True")
    monkeypatch.setattr(vllm_server, "_running_under_wsl", lambda: under_wsl)
    config = replace(
        _config(fake_vllm),
        speculative_config={"method": "mtp", "num_speculative_tokens": 2} if speculative else None,
    )
    assert VLLMServer(config)._environment()["PYTORCH_CUDA_ALLOC_CONF"] == expected


def test_owned_server_restores_info_logging_in_redirected_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_CONFIGURE_LOGGING", "0")
    monkeypatch.setenv("VLLM_LOGGING_LEVEL", "ERROR")
    monkeypatch.setenv("VLLM_LOGGING_CONFIG_PATH", "unavailable-logging.json")
    executable = tmp_path / "logging-vllm"
    executable.write_text(
        f"#!{sys.executable}\n"
        "from vllm.logger import _configure_vllm_root_logger, init_logger\n"
        "_configure_vllm_root_logger()\n"
        "logger = init_logger('vllm.synthetic_worker')\n"
        "logger.info('synthetic non-default args')\n"
        "logger.info('synthetic KV cache size')\n"
        "logger.info('synthetic speculative decode stats')\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    log_path = tmp_path / "server.log"
    config = replace(_config(executable), log_path=log_path)
    monkeypatch.setattr(vllm_server.urllib.request, "urlopen", _ready_response)
    with VLLMServer(config) as server:
        assert server.process.wait(timeout=30) == 0
    output = log_path.read_text(encoding="utf-8")
    assert "synthetic non-default args" in output
    assert "synthetic KV cache size" in output
    assert "synthetic speculative decode stats" in output
    assert os.environ["VLLM_CONFIGURE_LOGGING"] == "0"


@pytest.mark.parametrize("trainer_headroom_gib", [0.0, 1.0, 4.5])
def test_derived_share_reserves_runtime_floor_or_trainer_headroom(
    fake_vllm: Path, trainer_headroom_gib: float
) -> None:
    config = replace(
        _config(fake_vllm),
        gpu_memory_utilization=None,
        trainer_vram_headroom_gib=trainer_headroom_gib,
    )
    memory = gpu_memory.NvidiaSMIMemory(
        device_index=0, device_uuid="GPU-test", free_bytes=31 * 1024**3, total_bytes=32 * 1024**3
    )
    resolved = config.resolve_memory(memory)
    assert resolved.gpu_memory_utilization == pytest.approx(
        (31 - 0.75 - max(trainer_headroom_gib, 2.4)) / 32
    )


@pytest.mark.parametrize("requested_utilization", [None, 0.85])
@pytest.mark.parametrize("second_free_gib", [1.0, 2.4, 2.6])
@pytest.mark.parametrize("trainer_headroom_gib", [0.0, 4.5])
def test_readiness_memory_retry_is_once_and_only_for_derived_shares(
    fake_vllm: Path,
    monkeypatch: pytest.MonkeyPatch,
    requested_utilization: float | None,
    second_free_gib: float,
    trainer_headroom_gib: float,
) -> None:
    tmp_path = fake_vllm.parent
    required_free_gib = max(trainer_headroom_gib, 2.4)
    first_free_gib = required_free_gib - 0.4
    final_free_gib = required_free_gib + second_free_gib - 2.4
    free_readings = iter([31.0, first_free_gib, final_free_gib])
    monkeypatch.setattr(
        vllm_server,
        "nvidia_smi_memory_info",
        lambda: gpu_memory.NvidiaSMIMemory(
            device_index=0,
            device_uuid="GPU-test",
            free_bytes=round(next(free_readings) * 1024**3),
            total_bytes=32 * 1024**3,
        ),
    )
    processes: list[subprocess.Popen[bytes]] = []
    shares: list[float | None] = []
    previous_handler = signal.getsignal(signal.SIGTERM)
    server = VLLMServer(
        replace(
            _config(fake_vllm),
            gpu_memory_utilization=requested_utilization,
            log_path=tmp_path / "retry.log",
            trainer_vram_headroom_gib=trainer_headroom_gib,
        )
    )

    def ready(*args: object, **kwargs: object) -> _HealthResponse:
        del args, kwargs
        processes.append(server.process)
        shares.append(server.config.gpu_memory_utilization)
        return _HealthResponse(200)

    monkeypatch.setattr(vllm_server.urllib.request, "urlopen", ready)
    try:
        if requested_utilization is not None or second_free_gib < 2.4:
            with pytest.raises(VLLMServerStartupError, match="--gpu-memory-utilization"):
                server.start()
            assert all(process.poll() is not None for process in processes)
            assert signal.getsignal(signal.SIGTERM) is previous_handler
        else:
            server.start()
            assert server.process.poll() is None
        expected_attempts = 2 if requested_utilization is None else 1
        assert len(processes) == expected_attempts
        if expected_attempts == 2:
            assert processes[0].poll() is not None
            assert shares[0] is not None
            assert shares[1] is not None
            assert shares[0] - shares[1] == pytest.approx(0.4 / 32 + 0.01)
        else:
            assert shares == [requested_utilization]
        recorded = json.loads((tmp_path / "retry.memory.json").read_text())
        assert recorded == server.memory_provenance
        assert recorded["required_free_bytes"] == int(required_free_gib * 1024**3)
        assert [attempt["gpu_memory_utilization"] for attempt in recorded["attempts"]] == shares
        assert recorded["attempts"][0]["ready_free_bytes"] == round(first_free_gib * 1024**3)
        assert recorded["attempts"][0]["headroom_satisfied"] is False
        if expected_attempts == 2:
            assert recorded["attempts"][1]["ready_free_bytes"] == round(final_free_gib * 1024**3)
            assert recorded["attempts"][1]["headroom_satisfied"] is (second_free_gib >= 2.4)
    finally:
        server.stop()
    assert signal.getsignal(signal.SIGTERM) is previous_handler
