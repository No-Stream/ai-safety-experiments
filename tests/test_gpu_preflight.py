"""Pin the GPU preflight gate against the two ways a busy card can look free.

This gate had no test at all, and it was broken in the way that matters: it read VRAM nvidia-smi
would not attribute to a process as an empty card, so it passed on a nearly full GPU while logging
"preflight OK, used=22000 of total=23034 MiB in use". Both halves of that hole are pinned here, and
both were watched to fail against the pre-fix logic before being trusted:

- a peer whose ``used_memory`` reads ``[N/A]`` (nvidia-smi can see it but not inspect it, e.g.
  one in another container) used to become ``0 MiB`` and fall below the busy threshold;
- the aggregate ``memory.used`` was fetched only to interpolate into the error message, never
  compared, so a full card with no ``compute-apps`` rows at all read as free.

The passing cases matter just as much, because the cheap overcorrection -- refuse whenever any VRAM
is in use -- would wedge every GPU job on this box. An idle card, a small attributed CUDA context
below the threshold, and our own allocation all have to stay allowed, on one card and across two.
``nvidia-smi`` is never invoked: ``_nvidia_smi`` is the seam, monkeypatched with the exact CSV rows
the real one emits (``--format=csv,noheader,nounits``) -- one ``(used, total)`` pair per device,
including the observed idle case of zero rows. One test drops a layer lower and pins the command
line ``_nvidia_smi`` assembles, because the seam hides it: a device-scoping flag added there would
restrict the gate to GPU 0 with everything else still green.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.gpu_preflight import (
    GpuBusyError,
    GpuProcess,
    _nvidia_smi,
    gpu_processes,
    host_is_wsl,
    require_free_gpu,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PYTHON = Path("/usr/bin/python3")

SMI_COMPUTE_APPS = "compute-apps"
SMI_COMPUTE_APPS_FIELDS = "pid,used_memory,process_name"
SMI_GPU = "gpu"
SMI_GPU_FIELDS = "memory.used,memory.total"

# This box's single L4, as nvidia-smi reports its capacity.
TOTAL_MIB = 23034


def _fake_smi(
    monkeypatch: pytest.MonkeyPatch, *, rows: list[str], gpus: list[tuple[int, int]]
) -> None:
    """Stand in for nvidia-smi, returning canned CSV exactly as the real query shape does.

    ``rows`` are ``compute-apps`` rows; ``gpus`` carries one ``(used, total)`` MiB pair per device,
    so a single-card host is the one-element case rather than a second double with its own copy of
    the dispatch below.

    Strict about the question, not only the answer. This double used to hand back the memory row
    for any section it did not recognise and ignore ``fields`` entirely, so the gate could ask for
    a section nvidia-smi rejects, or read ``memory.total`` as the used figure, with every test in
    this file green -- the query is half of what makes the parsing below correct.
    """

    def fake_nvidia_smi(section: str, fields: str) -> list[str]:
        if section == SMI_COMPUTE_APPS:
            assert fields == SMI_COMPUTE_APPS_FIELDS, f"wrong compute-apps fields: {fields!r}"
            return rows
        assert section == SMI_GPU, f"unknown nvidia-smi section: {section!r}"
        assert fields == SMI_GPU_FIELDS, f"wrong gpu fields: {fields!r}"
        return [f"{used}, {total}" for used, total in gpus]

    monkeypatch.setattr("scripts.gpu_preflight._nvidia_smi", fake_nvidia_smi)


def _add_wsl_process(proc_root: Path, pid: int, comm: str, cmdline: str) -> Path:
    """Add a fake /proc process whose fd 3 points at WSL's Linux GPU device."""
    process_dir = proc_root / str(pid)
    fd_dir = process_dir / "fd"
    fd_dir.mkdir(parents=True)
    (process_dir / "comm").write_text(f"{comm}\n")
    (process_dir / "cmdline").write_text(cmdline.replace(" ", "\0"))
    (fd_dir / "3").symlink_to("/dev/dxg")
    return fd_dir


@pytest.fixture(autouse=True)
def native_linux_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin every test to a native-Linux host unless it says otherwise.

    The gate reads unattributed VRAM differently under WSL2, and the suite runs on a WSL2 box, so
    without this the unattributed-VRAM tests would silently exercise the WSL branch instead.
    """
    monkeypatch.setattr("scripts.gpu_preflight.host_is_wsl", lambda: False)


class TestTheQueryItselfIsPinned:
    """Every other test fakes ``_nvidia_smi``, so nothing above it can see the argv it builds."""

    def test_the_query_covers_every_device_and_asks_for_bare_numbers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A device-scoping flag here is invisible from above the seam.

        Adding ``-i 0`` to this command line reinstates exactly the whole-host bug
        ``TestTheAggregateSpansTheSameDevicesAsTheAttribution`` exists to catch -- watched, and
        every test in this file stayed green. ``noheader`` and ``nounits`` are pinned alongside it
        because the parsing above assumes bare integers with no header row, and ``check=True``
        because a section nvidia-smi rejects has to raise rather than read as an empty card.
        """
        argv: list[str] = []

        def fake_which(name: str) -> str:
            assert name == "nvidia-smi"
            return "/opt/bin/nvidia-smi"

        def fake_run(
            command: list[str], *, capture_output: bool, text: bool, check: bool
        ) -> subprocess.CompletedProcess[str]:
            assert capture_output
            assert text
            assert check
            argv.extend(command)
            return subprocess.CompletedProcess(command, 0, stdout="0, 23034\n")

        monkeypatch.setattr(shutil, "which", fake_which)
        monkeypatch.setattr(subprocess, "run", fake_run)

        assert _nvidia_smi("gpu", "memory.used,memory.total") == ["0, 23034"]
        assert argv == [
            "/opt/bin/nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]


class TestUnknownUsageIsNotZeroUsage:
    """A peer nvidia-smi will not quantify has to read as busy, not as holding nothing."""

    def test_an_uninspectable_peer_parses_as_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake_smi(monkeypatch, rows=["4242, [N/A], python"], gpus=[(22000, TOTAL_MIB)])
        assert gpu_processes() == [GpuProcess(pid=4242, used_mib=None, name="python")]

    def test_an_uninspectable_peer_makes_the_card_busy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _fake_smi(monkeypatch, rows=["4242, [N/A], python"], gpus=[(22000, TOTAL_MIB)])
        with pytest.raises(GpuBusyError) as caught:
            require_free_gpu()
        # The diagnostic must not claim a number nvidia-smi refused to give.
        assert "unknown amount of VRAM" in str(caught.value)
        assert "pid 4242" in str(caught.value)

    def test_an_uninspectable_peer_is_busy_even_below_the_aggregate_threshold(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unknown usage is decided on the row, so it does not need the aggregate to agree."""
        _fake_smi(monkeypatch, rows=["4242, [N/A], python"], gpus=[(0, TOTAL_MIB)])
        with pytest.raises(GpuBusyError):
            require_free_gpu()


class TestVramNoProcessAccountsForIsBusy:
    """The aggregate is compared, not merely printed: a peer can hold VRAM with no row at all."""

    def test_a_full_card_with_no_rows_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake_smi(monkeypatch, rows=[], gpus=[(22000, TOTAL_MIB)])
        with pytest.raises(GpuBusyError) as caught:
            require_free_gpu()
        message = str(caught.value)
        assert "22000" in message
        assert "no process it will name" in message

    def test_a_partly_attributed_card_is_refused_on_the_remainder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """8 GiB named, 22 GiB used: the 14 GiB nobody claims is what decides it."""
        _fake_smi(monkeypatch, rows=[f"{os.getpid()}, 8000, python"], gpus=[(22000, TOTAL_MIB)])
        with pytest.raises(GpuBusyError) as caught:
            require_free_gpu()
        assert "14000 MiB" in str(caught.value)


class TestTheGateStaysUsableOnAFreeCard:
    """The overcorrection -- refuse on any VRAM at all -- would block every job on this box."""

    def test_an_idle_card_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The observed idle state of this box: zero compute-apps rows, zero MiB used."""
        _fake_smi(monkeypatch, rows=[], gpus=[(0, TOTAL_MIB)])
        require_free_gpu()

    def test_a_small_attributed_context_below_the_threshold_passes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """What the 512 MiB threshold exists to tolerate: someone's idle CUDA context."""
        _fake_smi(monkeypatch, rows=["777, 300, python"], gpus=[(300, TOTAL_MIB)])
        require_free_gpu()

    def test_our_own_allocation_is_not_another_process(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _fake_smi(monkeypatch, rows=[f"{os.getpid()}, 8000, python"], gpus=[(8000, TOTAL_MIB)])
        require_free_gpu()

    def test_a_real_peer_over_the_threshold_is_still_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The case the gate already caught, kept pinned so the fix did not trade it away."""
        _fake_smi(monkeypatch, rows=["4242, 8000, python"], gpus=[(8000, TOTAL_MIB)])
        with pytest.raises(GpuBusyError) as caught:
            require_free_gpu()
        assert "pid 4242 holding 8000 MiB" in str(caught.value)


class TestTheAggregateSpansTheSameDevicesAsTheAttribution:
    """Both sides of the unattributed-VRAM subtraction have to cover every card.

    ``gpu_processes`` returns compute-apps rows for every device, while the aggregate used to be
    read from row ``[0]`` of ``--query-gpu`` alone. Subtracting an all-device attributed total from
    a GPU-0 used total is arithmetic across two different populations: on a multi-GPU host it can
    compute a negative or zero remainder for a card that is entirely full, and log ``GPU preflight
    OK``. Not reachable on this single-L4 box, which is exactly why it needs a test rather than a
    run.
    """

    def test_a_full_second_card_with_no_rows_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GPU 0 idle, GPU 1 full and unattributed: reading GPU 0 alone reports an empty host."""
        _fake_smi(monkeypatch, rows=[], gpus=[(0, TOTAL_MIB), (20000, TOTAL_MIB)])
        with pytest.raises(GpuBusyError) as caught:
            require_free_gpu()
        assert "20000 MiB held by no process it will name" in str(caught.value)

    def test_attribution_on_one_card_does_not_excuse_another(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Our own small context on GPU 0 must not cancel out a peer holding GPU 1."""
        _fake_smi(
            monkeypatch,
            rows=[f"{os.getpid()}, 300, python"],
            gpus=[(300, TOTAL_MIB), (20000, TOTAL_MIB)],
        )
        with pytest.raises(GpuBusyError) as caught:
            require_free_gpu()
        assert "20000 MiB held by no process it will name" in str(caught.value)

    def test_an_idle_multi_gpu_host_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake_smi(monkeypatch, rows=[], gpus=[(0, TOTAL_MIB), (0, TOTAL_MIB)])
        require_free_gpu()

    def test_our_own_allocation_across_two_cards_passes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _fake_smi(
            monkeypatch,
            rows=[f"{os.getpid()}, 8000, python", f"{os.getpid()}, 4000, python"],
            gpus=[(8000, TOTAL_MIB), (4000, TOTAL_MIB)],
        )
        require_free_gpu()


class TestUnderWslTheWindowsHostIsNotAPeer:
    """The Windows desktop's VRAM is expected; Linux GPU peers are found through /dev/dxg fds.

    Measured on the 5090 box (2026-09-26): with no Linux GPU process, ~3.5 GiB is in use and
    compute-apps lists nothing. Linux CUDA processes also do not appear in that list, so the preflight
    scans /proc for open /dev/dxg descriptors while treating the remaining VRAM as Windows-owned.
    """

    @pytest.fixture(autouse=True)
    def _wsl_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("scripts.gpu_preflight.host_is_wsl", lambda: True)

    def test_the_windows_desktop_share_passes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO)
        _fake_smi(monkeypatch, rows=[], gpus=[(3519, 32607)])
        require_free_gpu(proc_root=tmp_path)
        assert "GPU preflight OK under WSL2, 3519 MiB held by the Windows host" in caplog.text

    def test_a_linux_dxg_holder_is_refused_with_proc_details(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cmdline = "python -m vllm " + "x" * 200
        _add_wsl_process(tmp_path, pid=4242, comm="vllm", cmdline=cmdline)
        _fake_smi(monkeypatch, rows=[], gpus=[(31470, 32607)])

        with pytest.raises(GpuBusyError) as caught:
            require_free_gpu(proc_root=tmp_path)

        message = str(caught.value)
        assert "GPU is busy: 31470/32607 MiB in use by another process" in message
        assert "pid 4242" in message
        assert "/proc/4242/comm: vllm" in message
        assert f"cmdline: {cmdline[:160]}..." in message
        assert f"cmdline: {cmdline}" not in message

    def test_our_own_dxg_descriptor_is_ignored(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _add_wsl_process(tmp_path, pid=os.getpid(), comm="pytest", cmdline="pytest")
        _fake_smi(monkeypatch, rows=[], gpus=[(8000, 32607)])

        require_free_gpu(proc_root=tmp_path)

    def test_an_unreadable_fd_directory_is_skipped_and_logged(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        fd_dir = _add_wsl_process(tmp_path, pid=4242, comm="vllm", cmdline="python -m vllm")
        fd_dir.chmod(0)
        caplog.set_level(logging.INFO)
        _fake_smi(monkeypatch, rows=[], gpus=[(31470, 32607)])

        try:
            require_free_gpu(proc_root=tmp_path)
        finally:
            fd_dir.chmod(0o700)

        assert "WSL /dev/dxg unreadable fd process count: 1" in caplog.text
        assert "GPU preflight OK under WSL2, 31470 MiB held by the Windows host" in caplog.text

    @pytest.mark.skipif(
        not SYSTEM_PYTHON.exists(), reason=f"no system interpreter at {SYSTEM_PYTHON}"
    )
    def test_the_scan_runs_under_the_system_interpreter(self, tmp_path: Path) -> None:
        """The scan executes under the interpreter that really runs the script, not only the venv's.

        Compiling under it is not enough: ``Path.readlink`` compiles everywhere and raised
        AttributeError at runtime under this box's 3.8 system Python.
        """
        _add_wsl_process(tmp_path, pid=4242, comm="vllm", cmdline="python -m vllm")
        scan = (
            "import sys; from pathlib import Path; sys.path.insert(0, 'scripts'); "
            "import gpu_preflight; "
            f"holders, unreadable = gpu_preflight._wsl_dxg_processes(Path({str(tmp_path)!r}), 1); "
            "print([holder.pid for holder in holders], unreadable)"
        )
        completed = subprocess.run(  # noqa: S603 - fixed interpreter, script from this repo
            [str(SYSTEM_PYTHON), "-c", scan],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        )
        assert completed.stdout.strip() == "[4242] 0"


class TestWslDetection:
    """The WSL2 kernel names itself in its release string; a native kernel does not."""

    def test_a_wsl2_release_string_is_wsl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.undo()  # the autouse stub replaced host_is_wsl; test the real function
        monkeypatch.setattr(
            "scripts.gpu_preflight._kernel_osrelease", lambda: "6.18.33.2-microsoft-standard-WSL2"
        )
        assert host_is_wsl()

    def test_a_native_release_string_is_not_wsl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.undo()
        monkeypatch.setattr("scripts.gpu_preflight._kernel_osrelease", lambda: "5.15.0-1066-aws")
        assert not host_is_wsl()
