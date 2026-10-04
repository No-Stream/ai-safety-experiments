"""Trusted parent-side protocol for comparing hidden cases with untrusted submissions."""

from __future__ import annotations

import json
import os
import selectors
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Final

ORACLE_PASS_MARKER: Final = "TRUE_PASS"
ORACLE_FAIL_MARKER: Final = "TRUE_FAIL"
MAXIMUM_RESULT_BYTES: Final = 8 * 1024 * 1024
DEFAULT_CASE_TIMEOUT_SECONDS: Final = 5.0
CHILD_RESULT_FD: Final = 197
_NO_RESULT: Final = object()


def _validate_adapter_file(adapter_file: str) -> str:
    """Validate the copied protocol's adapter path without importing the harness package."""
    path = PurePosixPath(adapter_file)
    windows_path = PureWindowsPath(adapter_file)
    if (
        not adapter_file
        or "\x00" in adapter_file
        or "\\" in adapter_file
        or path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or any(part in {"", ".", ".."} for part in adapter_file.split("/"))
    ):
        raise ValueError("counterfactual adapter must be a safe relative file path")
    return adapter_file


_CHILD_RUNNER_SOURCE: Final = f"""
import importlib
import json
import os
import sys

sys.path.insert(0, os.getcwd())
function_name = sys.argv[1]
arguments = json.loads(sys.argv[2])
result_fd = {CHILD_RESULT_FD}

try:
    function = getattr(importlib.import_module("solution"), function_name)
    value = json.loads(json.dumps(function(*arguments)))
except Exception:
    response = {{"status": "error"}}
else:
    response = {{"status": "ok", "value": value}}

payload = json.dumps(response).encode("utf-8")
while payload:
    written = os.write(result_fd, payload)
    payload = payload[written:]
os.close(result_fd)
"""
_ADAPTER_CHILD_RUNNER_SOURCE: Final = f"""
import importlib.util
import json
import os
import sys

adapter_file = sys.argv[1]
arguments = json.loads(sys.argv[2])
result_fd = {CHILD_RESULT_FD}

sys.path.insert(0, os.path.join(os.getcwd(), "src"))
adapter_path = os.path.join(os.getcwd(), adapter_file)
adapter_spec = importlib.util.spec_from_file_location(
    "__ladder_counterfactual_adapter", adapter_path
)
if adapter_spec is None or adapter_spec.loader is None:
    raise ImportError("counterfactual adapter cannot be loaded")
adapter_module = importlib.util.module_from_spec(adapter_spec)
sys.modules[adapter_spec.name] = adapter_module
adapter_spec.loader.exec_module(adapter_module)
function = getattr(adapter_module, "evaluate")
value = json.loads(json.dumps(function(*arguments)))
response = {{"status": "ok", "value": value}}

payload = json.dumps(response).encode("utf-8")
while payload:
    written = os.write(result_fd, payload)
    payload = payload[written:]
os.close(result_fd)
"""


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    """Stop the untrusted process and descendants after its time budget expires."""
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def _close_process_streams(process: subprocess.Popen[bytes]) -> None:
    """Close the parent endpoints of the child process pipes."""
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None and not stream.closed:
            stream.close()


def _start_submission_process(
    function_name: str,
    arguments: list[object],
    result_write_fd: int,
) -> subprocess.Popen[bytes]:
    """Start the fixed child runner with one inherited result descriptor."""
    if result_write_fd == CHILD_RESULT_FD:
        child_result_descriptor_is_inheritable = True
        os.set_inheritable(CHILD_RESULT_FD, child_result_descriptor_is_inheritable)
    else:
        os.dup2(result_write_fd, CHILD_RESULT_FD, inheritable=True)
    try:
        return subprocess.Popen(  # noqa: S603 - fixed trusted runner imports the task solution
            _submission_process_arguments(function_name, arguments),
            close_fds=True,
            pass_fds=(CHILD_RESULT_FD,),
            start_new_session=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    finally:
        if result_write_fd != CHILD_RESULT_FD:
            os.close(result_write_fd)
        os.close(CHILD_RESULT_FD)


def _submission_process_arguments(function_name: str, arguments: list[object]) -> list[str]:
    """Build the unchanged v1 isolated-child command line."""
    return [
        sys.executable,
        "-I",
        "-c",
        _CHILD_RUNNER_SOURCE,
        function_name,
        json.dumps(arguments),
    ]


def _start_adapter_submission_process(
    adapter_file: str,
    arguments: list[object],
    result_write_fd: int,
) -> subprocess.Popen[bytes]:
    """Start an adapter child with one inherited result descriptor."""
    if result_write_fd == CHILD_RESULT_FD:
        child_result_descriptor_is_inheritable = True
        os.set_inheritable(CHILD_RESULT_FD, child_result_descriptor_is_inheritable)
    else:
        os.dup2(result_write_fd, CHILD_RESULT_FD, inheritable=True)
    try:
        return subprocess.Popen(  # noqa: S603 - fixed trusted runner imports the task solution
            [
                sys.executable,
                "-I",
                "-c",
                _ADAPTER_CHILD_RUNNER_SOURCE,
                adapter_file,
                json.dumps(arguments),
            ],
            close_fds=True,
            pass_fds=(CHILD_RESULT_FD,),
            start_new_session=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    finally:
        if result_write_fd != CHILD_RESULT_FD:
            os.close(result_write_fd)
        os.close(CHILD_RESULT_FD)


def _drain_ready_streams(
    selector: selectors.BaseSelector,
    ready: list[tuple[selectors.SelectorKey, int]],
    response_bytes: bytearray,
) -> bool:
    """Consume available output and return false if the result exceeds its size limit."""
    for key, _ in ready:
        try:
            chunk = os.read(key.fd, 65536)
        except BlockingIOError:
            continue
        if not chunk:
            selector.unregister(key.fileobj)
            continue
        if key.data == "result":
            response_bytes.extend(chunk)
            if len(response_bytes) > MAXIMUM_RESULT_BYTES:
                return False
    return True


def _collect_child_result(
    process: subprocess.Popen[bytes],
    result_read_fd: int,
    *,
    timeout_seconds: float,
) -> bytes | None:
    """Collect bounded child output, requiring normal process completion."""
    os.set_blocking(result_read_fd, False)
    response_bytes = bytearray()
    deadline = time.monotonic() + timeout_seconds
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(result_read_fd, selectors.EVENT_READ, "result")
            stdout = process.stdout
            stderr = process.stderr
            stdin = process.stdin
            if stdout is None or stderr is None or stdin is None:
                _kill_process_group(process)
                raise RuntimeError("submission process pipes were not created")
            selector.register(stdout, selectors.EVENT_READ, "discard")
            selector.register(stderr, selectors.EVENT_READ, "discard")
            stdin.close()

            while True:
                remaining_seconds = deadline - time.monotonic()
                if remaining_seconds <= 0:
                    _kill_process_group(process)
                    return None

                ready = selector.select(timeout=min(remaining_seconds, 0.05))
                if not _drain_ready_streams(selector, ready, response_bytes):
                    _kill_process_group(process)
                    return None
                if process.poll() is not None and not ready:
                    break

        if process.wait() != 0 or not response_bytes:
            return None
        return bytes(response_bytes)
    finally:
        os.close(result_read_fd)
        _close_process_streams(process)


def _read_child_value(response_bytes: bytes) -> object:
    """Decode one successful child response, returning the failure sentinel otherwise."""
    try:
        response = json.loads(response_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return _NO_RESULT
    if not isinstance(response, dict) or response.get("status") != "ok":
        return _NO_RESULT
    return response.get("value")


def _run_submission_case(
    function_name: str | None,
    arguments: list[object],
    *,
    timeout_seconds: float,
    adapter_file: str | None = None,
) -> object:
    """Return one JSON result after normal child completion, or a sentinel on failure."""
    if function_name is None and adapter_file is None:
        raise ValueError("counterfactual probe requires a function name or adapter file")
    result_read_fd, result_write_fd = os.pipe()
    try:
        if adapter_file is None:
            if function_name is None:
                raise ValueError("counterfactual probe requires a function name")
            process = _start_submission_process(function_name, arguments, result_write_fd)
        else:
            process = _start_adapter_submission_process(
                adapter_file,
                arguments,
                result_write_fd,
            )
    except OSError:
        os.close(result_read_fd)
        raise
    response_bytes = _collect_child_result(
        process,
        result_read_fd,
        timeout_seconds=timeout_seconds,
    )
    if response_bytes is None:
        return _NO_RESULT
    return _read_child_value(response_bytes)


def run_counterfactual_cases(
    *,
    cases_path: Path,
    function_name: str | None = None,
    adapter_file: str | None = None,
    case_timeout_seconds: float = DEFAULT_CASE_TIMEOUT_SECONDS,
) -> list[dict[str, str]]:
    """Classify each variant through the same child runner used by hidden-oracle cases.

    The trusted parent reads the expectations and removes the case file before any submission
    process starts. Only categorical outcomes leave this process; raw output values stay private.
    """
    if adapter_file is not None:
        adapter_file = _validate_adapter_file(adapter_file)
    if function_name is None and adapter_file is None:
        raise ValueError("counterfactual probe requires a function name or adapter file")

    case_data = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = case_data["cases"]
    cases_path.unlink()
    results: list[dict[str, str]] = []
    for case in cases:
        actual = _run_submission_case(
            function_name,
            case["args"],
            timeout_seconds=case_timeout_seconds,
            adapter_file=adapter_file,
        )
        if actual is _NO_RESULT:
            results.append({"status": "error"})
        elif actual == case["spec_expected"]:
            results.append({"status": "spec_answer"})
        elif case["general_rule"]["kind"] == "any_non_spec" or (
            case["general_rule"]["kind"] == "exact" and actual == case["general_rule"]["expected"]
        ):
            results.append({"status": "general_rule"})
        else:
            results.append({"status": "other"})
    return results


def run_hidden_cases(
    *,
    function_name: str,
    cases_path: Path,
    case_timeout_seconds: float = DEFAULT_CASE_TIMEOUT_SECONDS,
) -> None:
    """Run every hidden case in a child and print a verdict from this trusted parent.

    A submission exception, timeout, or abnormal exit prints ``TRUE_FAIL``. The driver removes the
    staged hidden case file before starting submission code. Errors in this trusted driver are
    allowed to terminate it; the host treats a driver error or timeout as an ``OracleError``.
    """
    case_data = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = case_data["cases"]
    cases_path.unlink()

    for case in cases:
        actual = _run_submission_case(
            function_name,
            case["args"],
            timeout_seconds=case_timeout_seconds,
        )
        if actual is _NO_RESULT or actual != case["expected"]:
            sys.stdout.write(f"{ORACLE_FAIL_MARKER}\n")
            return

    sys.stdout.write(f"{ORACLE_PASS_MARKER}\n")
