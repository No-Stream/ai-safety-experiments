"""Pin the AWS Batch submission contract and the container scripts, entirely offline.

No AWS call is made anywhere here: the Batch client is a stub that records what it was asked to do,
and the shell scripts run against stubbed binaries. Nothing in this file needs credentials, a
network, or a GPU. The checkpoint sync that rides along with a run lives in
`games/tests/test_games_s3_sync.py`.

:class:`TestAStaleRevisionIsUnrepresentable` is the class with teeth, and it is a structural test
rather than a behavioural one. Batch pins an image tag to a digest when a job-definition revision is
registered and never re-resolves it, so submitting against a pre-existing revision runs old code
while every log line claims the new commit -- with no symptom until results stop matching source.
The tests assert that the submit path cannot express that: register always runs first, the submit
request is keyed on the ARN registration returned, and an ARN without a revision is refused.

:class:`TestTheEntrypointScript` reads the shipped shell rather than running it. Two orderings in
that file are load-bearing and easy to "tidy" into breakage: provenance logging before anything that
can fail, and the training command NOT being `exec`'d, since exec would discard the EXIT trap that
ships the artifacts.

:class:`TestTheEntrypointShipsTheRunWhenBatchStopsIt` and
:class:`TestTheCveGateRefusesAnUnevaluatedScan` *run* the two shell scripts instead of reading them,
against stub `python`, `aws`, `docker`, `git` and `nvidia-smi` placed first on PATH. Nothing here
reaches AWS, docker, or a GPU either: each stub records its argv as one line in an events file, so
the assertions are on observed behaviour -- the exit status the script reports, the signal it
forwards, the order in which events land. Both classes exist because the string-matching tests above
them stayed green while the behaviour they described was absent.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from cloud.submit_job import (
    DEFAULT_TIER,
    GPU_TIERS,
    MIN_TIMEOUT_SECONDS,
    SubmitConfig,
    append_submission_log,
    build_register_request,
    build_submit_request,
    container_environment,
    job_name_for,
    parse_args,
    read_submission_log,
    submission_record,
    submit,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
CLOUD_DIR = REPO_ROOT / "cloud"

IMAGE = "123456789012.dkr.ecr.us-west-2.amazonaws.com/games-grpo:latest"
QUEUE = "games-g6e-queue"
DEFINITION = "games-grpo"
ARN_PREFIX = "arn:aws:batch:us-west-2:123456789012:job-definition"

# The real invocation, not the mention in the script's header comment.
TRAIN_INVOCATION = 'python -m games.train --output-dir "${OUTPUT_DIR}"'


def make_config(tmp_path: Path, **overrides: Any) -> SubmitConfig:
    """A submission config with everything AWS-shaped filled in."""
    defaults: dict[str, Any] = {
        "arm": "twin-pd-group",
        "image": IMAGE,
        "queue": QUEUE,
        "job_definition_name": DEFINITION,
        "log_path": tmp_path / "submissions.jsonl",
    }
    defaults.update(overrides)
    return SubmitConfig(**defaults)


class StubBatchClient:
    """Records requests and hands back plausible responses, including a fresh revision each time."""

    def __init__(self, *, start_revision: int = 7) -> None:
        self.register_calls: list[dict[str, Any]] = []
        self.submit_calls: list[dict[str, Any]] = []
        self._revision = start_revision

    def register_job_definition(self, **kwargs: object) -> dict[str, Any]:
        """Return a new revision on every call, as Batch does."""
        self.register_calls.append(dict(kwargs))
        self._revision += 1
        name = kwargs["jobDefinitionName"]
        return {
            "jobDefinitionName": name,
            "jobDefinitionArn": f"{ARN_PREFIX}/{name}:{self._revision}",
            "revision": self._revision,
        }

    def submit_job(self, **kwargs: object) -> dict[str, Any]:
        """Return a job id, recording exactly what it was asked to submit."""
        self.submit_calls.append(dict(kwargs))
        return {"jobId": f"job-{len(self.submit_calls)}", "jobName": kwargs["jobName"]}


@pytest.fixture
def client() -> StubBatchClient:
    return StubBatchClient()


@pytest.fixture
def moment() -> datetime:
    return datetime(2026, 8, 17, 12, 30, 45, tzinfo=UTC)


class TestAStaleRevisionIsUnrepresentable:
    def test_every_submit_registers_a_fresh_revision_first(
        self, tmp_path: Path, client: StubBatchClient, moment: datetime
    ) -> None:
        config = make_config(tmp_path)
        first = submit(config, client, now=moment)
        second = submit(config, client, now=moment)
        assert len(client.register_calls) == 2
        assert len(client.submit_calls) == 2
        assert first["job_definition_arn"] != second["job_definition_arn"]
        assert first["job_definition_arn"].endswith(":8")
        assert second["job_definition_arn"].endswith(":9")

    def test_the_submitted_definition_is_the_one_just_registered(
        self, tmp_path: Path, client: StubBatchClient, moment: datetime
    ) -> None:
        record = submit(make_config(tmp_path), client, now=moment)
        assert client.submit_calls[0]["jobDefinition"] == record["job_definition_arn"]
        assert ":" in client.submit_calls[0]["jobDefinition"].rsplit("/", 1)[1]

    @pytest.mark.parametrize(
        "arn",
        [
            f"{ARN_PREFIX}/{DEFINITION}",
            DEFINITION,
            f"{DEFINITION}:8",
            "",
            f"{ARN_PREFIX}/{DEFINITION}:latest",
        ],
    )
    def test_an_arn_without_a_pinned_revision_is_refused(self, tmp_path: Path, arn: str) -> None:
        """A bare name would let Batch choose the revision, which is the stale-digest path."""
        with pytest.raises(ValueError, match="revision-bearing"):
            build_submit_request(make_config(tmp_path), job_definition_arn=arn, job_name="games-x")

    def test_an_arn_for_a_different_definition_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="refusing to submit"):
            build_submit_request(
                make_config(tmp_path),
                job_definition_arn=f"{ARN_PREFIX}/some-other-definition:3",
                job_name="games-x",
            )

    def test_a_valid_revision_arn_is_accepted(self, tmp_path: Path) -> None:
        request = build_submit_request(
            make_config(tmp_path),
            job_definition_arn=f"{ARN_PREFIX}/{DEFINITION}:8",
            job_name="games-x",
        )
        assert request == {
            "jobName": "games-x",
            "jobQueue": QUEUE,
            "jobDefinition": f"{ARN_PREFIX}/{DEFINITION}:8",
        }


class TestRegisterRequestShaping:
    def test_the_resource_request_matches_the_tier(self, tmp_path: Path) -> None:
        config = make_config(tmp_path, tier="g6e-1gpu")
        request = build_register_request(config)
        resources = {
            entry["type"]: entry["value"]
            for entry in request["containerProperties"]["resourceRequirements"]
        }
        tier = GPU_TIERS["g6e-1gpu"]
        assert resources == {
            "VCPU": str(tier.vcpus),
            "MEMORY": str(tier.memory_mib),
            "GPU": str(tier.gpus),
        }

    def test_the_command_passes_the_arm_and_forwards_train_args(self, tmp_path: Path) -> None:
        config = make_config(tmp_path, train_args=("--max-steps", "70", "--smoke"))
        request = build_register_request(config)
        assert request["containerProperties"]["command"] == [
            "--arm",
            "twin-pd-group",
            "--max-steps",
            "70",
            "--smoke",
        ]

    def test_retries_are_off_so_a_failure_does_not_silently_respend(self, tmp_path: Path) -> None:
        assert build_register_request(make_config(tmp_path))["retryStrategy"] == {"attempts": 1}

    def test_a_wall_clock_timeout_is_always_set(self, tmp_path: Path) -> None:
        request = build_register_request(make_config(tmp_path, timeout_seconds=3600))
        assert request["timeout"] == {"attemptDurationSeconds": 3600}

    def test_the_intended_instance_type_is_tagged_for_provenance(self, tmp_path: Path) -> None:
        """Batch takes the instance from the queue's compute environment, so record the intent."""
        request = build_register_request(make_config(tmp_path, tier="g7e-2gpu"))
        assert request["tags"]["intended_instance_type"] == "g7e.12xlarge"

    def test_roles_are_omitted_rather_than_sent_empty(self, tmp_path: Path) -> None:
        container = build_register_request(make_config(tmp_path))["containerProperties"]
        assert "jobRoleArn" not in container
        assert "executionRoleArn" not in container

    def test_roles_are_included_when_given(self, tmp_path: Path) -> None:
        config = make_config(tmp_path, job_role_arn="arn:aws:iam::1:role/games-job")
        container = build_register_request(config)["containerProperties"]
        assert container["jobRoleArn"] == "arn:aws:iam::1:role/games-job"


class TestContainerEnvironment:
    def test_the_output_dir_and_flags_are_passed(self, tmp_path: Path) -> None:
        config = make_config(tmp_path, s3_dest="s3://bucket/runs", vllm_server=True)
        environment = {entry["name"]: entry["value"] for entry in container_environment(config)}
        assert environment["GAMES_OUTPUT_DIR"] == "/scratch/runs/current"
        assert environment["GAMES_S3_DEST"] == "s3://bucket/runs"
        assert environment["GAMES_VLLM_SERVER"] == "1"
        assert environment["GAMES_REQUIRE_FAST_KERNELS"] == "0"

    def test_no_s3_dest_means_the_variable_is_absent_not_empty(self, tmp_path: Path) -> None:
        """The entrypoint treats unset as "do not sync"; an empty string would be ambiguous."""
        environment = {
            entry["name"]: entry["value"] for entry in container_environment(make_config(tmp_path))
        }
        assert "GAMES_S3_DEST" not in environment

    def test_git_sha_is_never_injected_by_the_submitter(self, tmp_path: Path) -> None:
        """It is baked into the image; a submitted value could contradict the running code."""
        config = make_config(tmp_path, s3_dest="s3://bucket/runs")
        names = {entry["name"] for entry in container_environment(config)}
        assert "GIT_SHA" not in names


class TestConfigValidation:
    def test_an_unknown_tier_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unknown tier"):
            make_config(tmp_path, tier="h100-cluster")

    def test_a_too_short_timeout_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="timeout_seconds must be at least"):
            make_config(tmp_path, timeout_seconds=MIN_TIMEOUT_SECONDS - 1)

    def test_an_untagged_image_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="carries no tag"):
            make_config(tmp_path, image="123.dkr.ecr.us-west-2.amazonaws.com/games-grpo")

    def test_a_sha_tagged_image_is_refused_by_default(self, tmp_path: Path) -> None:
        """push_ecr.sh publishes both tags; job definitions resolve :latest by convention."""
        with pytest.raises(ValueError, match="not 'latest'"):
            make_config(tmp_path, image="123.dkr.ecr.us-west-2.amazonaws.com/games-grpo:abc1234")

    def test_a_sha_tagged_image_is_allowed_deliberately(self, tmp_path: Path) -> None:
        config = make_config(
            tmp_path,
            image="123.dkr.ecr.us-west-2.amazonaws.com/games-grpo:abc1234",
            allow_nonlatest_image=True,
        )
        assert config.image.endswith(":abc1234")

    def test_every_tier_declares_whether_its_shape_was_verified(self) -> None:
        """An unverified over-request leaves a job RUNNABLE forever instead of failing."""
        assert GPU_TIERS[DEFAULT_TIER].verified
        for name, tier in GPU_TIERS.items():
            assert isinstance(tier.verified, bool), name
            assert tier.gpus >= 1, name


class TestJobNaming:
    def test_the_name_carries_the_arm_the_tier_and_a_timestamp(self, moment: datetime) -> None:
        name = job_name_for(
            SubmitConfig(
                arm="twin-pd-group", image=IMAGE, queue=QUEUE, job_definition_name=DEFINITION
            ),
            now=moment,
        )
        assert name == "games-twin-pd-group-g6e-1gpu-20260817-123045"

    def test_illegal_characters_are_replaced(self, moment: datetime) -> None:
        name = job_name_for(
            SubmitConfig(
                arm="twin/pd group", image=IMAGE, queue=QUEUE, job_definition_name=DEFINITION
            ),
            now=moment,
        )
        assert "/" not in name
        assert " " not in name


class TestSubmissionLog:
    def test_a_submission_round_trips_through_the_jsonl(
        self, tmp_path: Path, client: StubBatchClient, moment: datetime
    ) -> None:
        log_path = tmp_path / "nested" / "submissions.jsonl"
        config = make_config(tmp_path, log_path=log_path, s3_dest="s3://bucket/runs")
        record = submit(config, client, now=moment)
        rows = read_submission_log(log_path)
        assert len(rows) == 1
        assert rows[0] == record
        assert rows[0]["job_id"] == "job-1"
        assert rows[0]["arm"] == "twin-pd-group"
        assert rows[0]["queue"] == QUEUE
        assert rows[0]["intended_instance_type"] == "g6e.xlarge"
        assert rows[0]["image"] == IMAGE
        assert rows[0]["submitted_at"].startswith("2026-08-17T12:30:45")

    def test_submissions_append_rather_than_overwrite(
        self, tmp_path: Path, client: StubBatchClient, moment: datetime
    ) -> None:
        config = make_config(tmp_path)
        submit(config, client, now=moment)
        submit(config, client, now=moment)
        assert len(read_submission_log(config.log_path)) == 2

    def test_reading_a_missing_log_is_empty_not_an_error(self, tmp_path: Path) -> None:
        assert read_submission_log(tmp_path / "never-written.jsonl") == []

    def test_the_record_distinguishes_the_submitter_tree_from_the_image(
        self, tmp_path: Path, moment: datetime
    ) -> None:
        """The image's baked GIT_SHA is the authority; the submitter's sha is context."""
        record = submission_record(
            make_config(tmp_path),
            job_id="job-1",
            job_name="games-x",
            job_definition_arn=f"{ARN_PREFIX}/{DEFINITION}:8",
            now=moment,
        )
        assert "submitter_git_sha" in record
        assert "submitter_tree_dirty" in record
        assert isinstance(record["submitter_tree_dirty"], bool)

    def test_the_log_is_one_json_object_per_line(self, tmp_path: Path, moment: datetime) -> None:
        log_path = tmp_path / "submissions.jsonl"
        for index in range(3):
            append_submission_log(log_path, {"job_id": f"job-{index}"})
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 3
        assert [json.loads(line)["job_id"] for line in lines] == ["job-0", "job-1", "job-2"]


class TestArgumentParsing:
    def test_the_minimum_invocation(self) -> None:
        config = parse_args(["--arm", "dictator", "--image", IMAGE, "--queue", QUEUE])
        assert config.arm == "dictator"
        assert config.tier == DEFAULT_TIER
        assert config.train_args == ()

    def test_trailing_arguments_are_forwarded_to_the_trainer(self) -> None:
        config = parse_args(
            [
                "--arm",
                "dictator",
                "--image",
                IMAGE,
                "--queue",
                QUEUE,
                "--",
                "--max-steps",
                "5",
                "--smoke",
            ]
        )
        assert config.train_args == ("--max-steps", "5", "--smoke")

    def test_the_tier_choice_is_validated_by_the_parser(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(
                [
                    "--arm",
                    "dictator",
                    "--image",
                    IMAGE,
                    "--queue",
                    QUEUE,
                    "--tier",
                    "not-a-tier",
                ]
            )


class TestTheEntrypointScript:
    @pytest.fixture
    def entrypoint(self) -> str:
        return (CLOUD_DIR / "entrypoint.sh").read_text()

    def test_provenance_is_logged_before_the_kernel_check_and_training(
        self, entrypoint: str
    ) -> None:
        """Both questions asked of a dead job are answered by lines that must come first."""
        sha_position = entrypoint.index('log "GIT_SHA=')
        smi_position = entrypoint.index("if command -v nvidia-smi")
        kernel_position = entrypoint.index("check_fast_kernels()")
        train_position = entrypoint.index(TRAIN_INVOCATION)
        assert sha_position < smi_position < kernel_position < train_position

    def test_the_exit_trap_is_installed_before_training_starts(self, entrypoint: str) -> None:
        assert entrypoint.index("trap on_exit EXIT") < entrypoint.index(TRAIN_INVOCATION)

    def test_sigterm_is_trapped_because_batch_and_spot_both_use_it(self, entrypoint: str) -> None:
        """Behaviour is pinned by :class:`TestTheEntrypointShipsTheRunWhenBatchStopsIt`."""
        assert "trap 'on_signal TERM 143' TERM" in entrypoint
        assert "trap 'on_signal INT 130' INT" in entrypoint

    def test_training_is_not_exec_d_which_would_discard_the_trap(self, entrypoint: str) -> None:
        """exec replaces the shell, so the EXIT trap -- and the artifact sync -- would never run."""
        assert "exec python -m games.train" not in entrypoint
        assert TRAIN_INVOCATION in entrypoint

    def test_the_trainer_is_backgrounded_so_the_signal_handler_can_run(
        self, entrypoint: str
    ) -> None:
        """Bash defers a handler until a foreground child returns, and `wait` is the exception."""
        assert f'{TRAIN_INVOCATION} "$@" &' in entrypoint
        assert 'wait "${TRAIN_PID}"' in entrypoint

    def test_the_vllm_server_is_off_by_default(self, entrypoint: str) -> None:
        assert '"${GAMES_VLLM_SERVER:-0}" == "1"' in entrypoint

    def test_the_fast_kernel_check_only_fails_when_explicitly_armed(self, entrypoint: str) -> None:
        assert '"${GAMES_REQUIRE_FAST_KERNELS:-0}" == "1"' in entrypoint

    def test_it_refuses_a_caller_supplied_output_dir(self, entrypoint: str) -> None:
        assert "--output-dir" in entrypoint
        assert "GAMES_OUTPUT_DIR, not --output-dir" in entrypoint

    def test_the_exit_sync_ships_the_whole_run_directory(self, entrypoint: str) -> None:
        """Same coverage demand as :class:`TestS3SyncCommand`, on the other of the two sync paths.

        The exit trap is the net under a spot reclamation or an external kill, which is precisely
        when the last completions parquet has not been through an on_save sync yet.
        """
        commands = [
            line
            for line in entrypoint.splitlines()
            if "aws s3 sync" in line and not line.lstrip().startswith("#")
        ]
        assert len(commands) == 1
        assert '"${OUTPUT_DIR}" "${S3_DEST}"' in commands[0]
        assert "--exclude" not in commands[0]
        assert "--include" not in commands[0]


class TestTheShellScriptsAreClean:
    @pytest.mark.parametrize("script", ["entrypoint.sh", "push_ecr.sh"])
    def test_shellcheck_passes(self, script: str) -> None:
        """Skipped rather than failed where shellcheck is absent, so CI-less boxes still run."""
        path = CLOUD_DIR / script
        try:
            finished = subprocess.run(  # noqa: S603
                ["shellcheck", str(path)],  # noqa: S607
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            pytest.skip("shellcheck is not installed")
        assert finished.returncode == 0, finished.stdout + finished.stderr

    @pytest.mark.parametrize("script", ["entrypoint.sh", "push_ecr.sh"])
    def test_the_scripts_are_executable(self, script: str) -> None:
        assert (CLOUD_DIR / script).stat().st_mode & 0o111


class TestThePushScriptHoldsItsInvariants:
    @pytest.fixture
    def push_script(self) -> str:
        return (CLOUD_DIR / "push_ecr.sh").read_text()

    def test_it_refuses_a_dirty_tree(self, push_script: str) -> None:
        assert "git status --porcelain" in push_script
        assert "the working tree is dirty" in push_script

    def test_the_build_context_is_the_committed_tree(self, push_script: str) -> None:
        """git archive makes "only committed code ships" structural, not a promise."""
        assert "git archive --format=tar HEAD" in push_script

    def test_it_bakes_the_sha_and_pushes_both_tags(self, push_script: str) -> None:
        assert '--build-arg "GIT_SHA=${GIT_SHA}"' in push_script
        assert 'docker push "${IMAGE}:latest"' in push_script
        assert 'docker push "${IMAGE}:${GIT_SHA}"' in push_script

    def test_latest_is_published_only_after_the_cve_gate(self, push_script: str) -> None:
        gate = push_script.index("critical_findings=")
        latest_push = push_script.index('docker push "${IMAGE}:latest"')
        assert gate < latest_push

    def test_an_unreadable_scan_blocks_the_push(self, push_script: str) -> None:
        """Unknown security posture is not the same as a clean one."""
        assert "the CVE gate cannot be evaluated" in push_script


class TestTheDockerfileHoldsItsInvariants:
    @pytest.fixture
    def dockerfile(self) -> str:
        return (CLOUD_DIR / "Dockerfile").read_text()

    def test_the_base_image_is_a_pinned_cuda_patch_release(self, dockerfile: str) -> None:
        assert "FROM nvidia/cuda:13.3.1-runtime-ubuntu24.04" in dockerfile

    def test_the_base_image_is_runtime_not_devel(self, dockerfile: str) -> None:
        """devel would add nvcc and a toolchain, which is CVE surface v1 does not need."""
        from_lines = [line for line in dockerfile.splitlines() if line.startswith("FROM ")]
        assert len(from_lines) == 1
        assert "runtime-ubuntu24.04" in from_lines[0]
        assert "devel" not in from_lines[0]

    def test_uv_is_pinned_rather_than_latest(self, dockerfile: str) -> None:
        assert "ghcr.io/astral-sh/uv:0.11.16" in dockerfile
        assert "ghcr.io/astral-sh/uv:latest" not in dockerfile

    def test_the_build_fails_without_a_git_sha(self, dockerfile: str) -> None:
        assert 'test -n "${GIT_SHA}"' in dockerfile

    def test_the_lockfile_is_installed_frozen(self, dockerfile: str) -> None:
        assert "uv sync --frozen --no-dev" in dockerfile

    def test_no_model_weights_are_baked(self, dockerfile: str) -> None:
        """Baking weights would tie an image to one model and serve a stale snapshot forever."""
        lowered = dockerfile.lower()
        assert "snapshot_download" not in lowered
        assert "huggingface-cli download" not in lowered
        assert "hf_home=/scratch/hf" in lowered

    def test_output_is_unbuffered_so_a_dying_job_still_logs(self, dockerfile: str) -> None:
        assert "PYTHONUNBUFFERED=1" in dockerfile


STUB_PYTHON = """\
#!/usr/bin/env bash
# Stands in for the image's interpreter: the version probe, the three kernel import probes, and a
# trainer whose stop behaviour the test chooses. STUB_TRAIN_MODE=sleep makes it sit until signalled,
# and its TERM handler takes a moment before recording `train-saved`, so a shell that does not wait
# for the child produces an observably wrong event order rather than a flaky pass.
case "$1" in
  --version) echo "Python 3.13.0 (stub interpreter)"; exit 0 ;;
  -c) exit 0 ;;
esac
if [[ "$1" != "-m" || "$2" != "games.train" ]]; then
  echo "stub python: unexpected argv: $*" >&2
  exit 3
fi
printf 'train-start %s\\n' "$*" >>"${ENTRYPOINT_EVENTS}"
case "${STUB_TRAIN_MODE}" in
  clean) exit 0 ;;
  crash) exit 7 ;;
esac
sleep 30 &
trainer_sleep=$!
trap 'kill "${trainer_sleep}" 2>/dev/null
      sleep 0.5
      printf "train-saved\\n" >>"${ENTRYPOINT_EVENTS}"
      exit 143' TERM
wait "${trainer_sleep}" || true
exit 0
"""

STUB_AWS_RECORDER = """\
#!/usr/bin/env bash
printf 'aws %s\\n' "$*" >>"${ENTRYPOINT_EVENTS}"
"""

STUB_NVIDIA_SMI = """\
#!/usr/bin/env bash
echo "stub nvidia-smi: no GPU was touched"
"""

STUB_AWS_ECR = """\
#!/usr/bin/env bash
# Covers only the calls cloud/push_ecr.sh makes. The scan status and the CRITICAL count come from
# the environment so a test can pose a scan that has not produced findings yet, which is the state
# the gate has to tell apart from a clean one.
printf 'aws %s\\n' "$*" >>"${PUSH_EVENTS}"
case "$1 $2" in
  "sts get-caller-identity") echo "123456789012" ;;
  "ecr describe-repositories") ;;
  "ecr get-login-password") echo "stub-ecr-password" ;;
  "ecr wait") [[ "${STUB_SCAN_STATUS}" == "COMPLETE" ]] || exit 255 ;;
  "ecr describe-image-scan-findings")
    if [[ "$*" == *imageScanStatus.status* ]]; then
      echo "${STUB_SCAN_STATUS}"
    else
      echo "${STUB_CRITICAL_COUNT}"
    fi
    ;;
  *) echo "stub aws: unexpected argv: $*" >&2; exit 4 ;;
esac
"""

STUB_DOCKER = """\
#!/usr/bin/env bash
# Drains stdin unconditionally: the build reads the git-archive context from a pipe and login reads
# the password from one, and leaving either unread would SIGPIPE the writer under `set -o pipefail`.
cat >/dev/null
printf 'docker %s\\n' "$*" >>"${PUSH_EVENTS}"
"""

STUB_GIT = """\
#!/usr/bin/env bash
# Lets the push script run against a clean tree without this repository's real state, and without
# any git write. The build context is a placeholder because docker is stubbed and never reads it.
case "$1 $2" in
  "rev-parse --show-toplevel") echo "${STUB_REPO_ROOT}" ;;
  "status --porcelain") ;;
  "rev-parse HEAD") echo "0123456789abcdef0123456789abcdef01234567" ;;
  "rev-parse --short") echo "0123456" ;;
  "archive --format=tar") printf 'stub-build-context' ;;
  *) echo "stub git: unexpected argv: $*" >&2; exit 5 ;;
esac
"""


def install_stubs(directory: Path, stubs: dict[str, str]) -> Path:
    """Write executable stubs into a directory meant to go first on PATH."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, body in stubs.items():
        path = directory / name
        path.write_text(body)
        path.chmod(0o755)
    return directory


def wait_for_event(events: Path, needle: str, *, timeout_seconds: float) -> None:
    """Block until a line containing `needle` appears in the events file."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if needle in events.read_text():
            return
        time.sleep(0.05)
    raise AssertionError(f"{needle!r} never appeared in {events}: {events.read_text()!r}")


class LaunchedEntrypoint:
    """The real cloud/entrypoint.sh, running as its own process against stubs."""

    def __init__(self, workspace: Path, *, train_mode: str) -> None:
        self.events = workspace / "events.log"
        self.events.touch()
        self.output_dir = workspace / "run"
        self.s3_dest = "s3://bucket/runs/twin-pd-group"
        stub_dir = install_stubs(
            workspace / "stubs",
            {"python": STUB_PYTHON, "aws": STUB_AWS_RECORDER, "nvidia-smi": STUB_NVIDIA_SMI},
        )
        self.process = subprocess.Popen(  # noqa: S603
            [str(CLOUD_DIR / "entrypoint.sh"), "--arm", "twin-pd-group"],
            cwd=workspace,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={
                "PATH": f"{stub_dir}:{os.environ['PATH']}",
                "ENTRYPOINT_EVENTS": str(self.events),
                "STUB_TRAIN_MODE": train_mode,
                "GAMES_OUTPUT_DIR": str(self.output_dir),
                "GAMES_S3_DEST": self.s3_dest,
                "GIT_SHA": "0123456789abcdef0123456789abcdef01234567",
            },
        )

    def event_lines(self) -> list[str]:
        return self.events.read_text().splitlines()

    def sync_lines(self) -> list[str]:
        return [line for line in self.event_lines() if line.startswith("aws s3 sync")]

    def stop(self) -> None:
        """Teardown for a test that failed before the script exited on its own."""
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=10)


@pytest.fixture
def launch_entrypoint(tmp_path: Path) -> Iterator[Callable[..., LaunchedEntrypoint]]:
    launched: list[LaunchedEntrypoint] = []

    def launch(*, train_mode: str = "sleep") -> LaunchedEntrypoint:
        run = LaunchedEntrypoint(tmp_path, train_mode=train_mode)
        launched.append(run)
        return run

    yield launch
    for run in launched:
        run.stop()


class TestTheEntrypointShipsTheRunWhenBatchStopsIt:
    """Send the script the signal Batch and a spot reclamation send, and watch what happens.

    This is the behavioural counterpart to :class:`TestTheEntrypointScript`, and it exists because
    the reading tests were green while the trap they described never ran. The exec-form
    ENTRYPOINT makes this shell PID 1 of the container with no init to forward anything, so
    `docker stop` delivers SIGTERM here and nowhere else -- and bash defers a signal handler until a
    foreground child returns, so a foreground trainer swallows the whole shutdown path.
    """

    @pytest.mark.parametrize(
        ("stop_signal", "expected_status"), [(signal.SIGTERM, 143), (signal.SIGINT, 130)]
    )
    def test_a_stop_signal_syncs_the_run_after_the_trainer_has_quiesced(
        self,
        launch_entrypoint: Callable[..., LaunchedEntrypoint],
        stop_signal: signal.Signals,
        expected_status: int,
    ) -> None:
        """SIGINT is covered because backgrounding the trainer changes what it can hear.

        With job control off bash sets SIGINT to ignore on an asynchronous child, so a handler that
        forwarded the signal it received would send the trainer something it cannot act on and then
        block in `wait` until training finished by itself: a Ctrl-C that hangs for hours. The stub
        trainer here traps only TERM, so it stops at all only if the handler forwards TERM.
        """
        run = launch_entrypoint(train_mode="sleep")
        wait_for_event(run.events, "train-start", timeout_seconds=30)

        run.process.send_signal(stop_signal)
        status = run.process.wait(timeout=20)

        events = run.event_lines()
        assert status == expected_status, run.process.stdout.read() if run.process.stdout else ""
        assert "train-saved" in events, events
        assert len(run.sync_lines()) == 1, events
        assert events.index("train-saved") < events.index(run.sync_lines()[0]), events
        assert f"{run.output_dir} {run.s3_dest}" in run.sync_lines()[0]

    def test_the_trainer_is_told_which_run_directory_to_write(
        self, launch_entrypoint: Callable[..., LaunchedEntrypoint]
    ) -> None:
        """The sync and the trainer must agree on the directory, or the shipped path is empty."""
        run = launch_entrypoint(train_mode="clean")
        assert run.process.wait(timeout=30) == 0
        starts = [line for line in run.event_lines() if line.startswith("train-start")]
        assert starts == [
            f"train-start -m games.train --output-dir {run.output_dir} --arm twin-pd-group"
        ]

    def test_a_clean_run_exits_zero_and_still_syncs(
        self, launch_entrypoint: Callable[..., LaunchedEntrypoint]
    ) -> None:
        run = launch_entrypoint(train_mode="clean")
        assert run.process.wait(timeout=30) == 0
        assert len(run.sync_lines()) == 1, run.event_lines()

    def test_a_crashing_trainer_reports_its_own_exit_status(
        self, launch_entrypoint: Callable[..., LaunchedEntrypoint]
    ) -> None:
        """Batch surfaces the training result, so the trainer's status must reach the container."""
        run = launch_entrypoint(train_mode="crash")
        assert run.process.wait(timeout=30) == 7
        assert len(run.sync_lines()) == 1, run.event_lines()


STUB_PUSH_SHA = "0123456789abcdef0123456789abcdef01234567"


@dataclass(frozen=True)
class PushOutcome:
    """What one run of cloud/push_ecr.sh did: its status, the commands it ran, and its log."""

    status: int
    events: list[str]
    output: str

    @property
    def latest_pushes(self) -> list[str]:
        return [
            line
            for line in self.events
            if line.startswith("docker push") and line.endswith(":latest")
        ]


def run_push_script(workspace: Path, *, scan_status: str, critical: str) -> PushOutcome:
    """Run the shipped push script against a stub ECR that reports the given scan state."""
    events = workspace / "push-events.log"
    events.touch()
    scratch_root = workspace / "scratch-repo"
    scratch_root.mkdir()
    stub_dir = install_stubs(
        workspace / "stubs", {"aws": STUB_AWS_ECR, "docker": STUB_DOCKER, "git": STUB_GIT}
    )
    finished = subprocess.run(  # noqa: S603
        [str(CLOUD_DIR / "push_ecr.sh"), "games-grpo", "us-west-2"],
        cwd=workspace,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        env={
            "PATH": f"{stub_dir}:{os.environ['PATH']}",
            "PUSH_EVENTS": str(events),
            "STUB_REPO_ROOT": str(scratch_root),
            "STUB_SCAN_STATUS": scan_status,
            "STUB_CRITICAL_COUNT": critical,
        },
    )
    return PushOutcome(
        status=finished.returncode,
        events=events.read_text().splitlines(),
        output=finished.stdout + finished.stderr,
    )


class TestTheCveGateRefusesAnUnevaluatedScan:
    """Run the real push script against a stubbed ECR whose scan has not produced findings yet.

    findingSeverityCounts is a map, so a severity with no findings is simply absent and the CLI
    prints "None" -- which reads the same whether the scan finished clean or has not started. Only
    the scan status separates the two, and `:latest` is the sole tag a job definition ever resolves,
    so publishing it on the ambiguous reading is how a CRITICAL image reaches Batch.
    """

    def test_an_in_progress_scan_blocks_latest(self, tmp_path: Path) -> None:
        """The scan answered and reported no CRITICAL key, having not looked yet. Not clean."""
        outcome = run_push_script(tmp_path, scan_status="IN_PROGRESS", critical="None")
        assert outcome.status != 0, outcome.output
        assert outcome.latest_pushes == [], outcome.events
        assert "IN_PROGRESS" in outcome.output

    def test_the_sha_tag_still_gets_pushed_for_inspection(self, tmp_path: Path) -> None:
        """Blocking :latest is the point; the sha tag has to be up there to be scanned at all."""
        outcome = run_push_script(tmp_path, scan_status="IN_PROGRESS", critical="None")
        assert [line for line in outcome.events if line.endswith(STUB_PUSH_SHA)], outcome.events

    def test_a_complete_scan_with_no_criticals_publishes(self, tmp_path: Path) -> None:
        """A terminal scan omits the key when it found nothing, so absent must still mean zero."""
        outcome = run_push_script(tmp_path, scan_status="COMPLETE", critical="None")
        assert outcome.status == 0, outcome.output
        assert len(outcome.latest_pushes) == 1, outcome.events

    def test_an_active_enhanced_scan_publishes(self, tmp_path: Path) -> None:
        """Inspector continuous scanning never reaches COMPLETE, so ACTIVE has to count as terminal.

        This is why the waiter above the gate stays advisory: `aws ecr wait image-scan-complete`
        accepts only COMPLETE and can never succeed on an enhanced-scanning repository, so making it
        fatal would block every push there.
        """
        outcome = run_push_script(tmp_path, scan_status="ACTIVE", critical="None")
        assert outcome.status == 0, outcome.output
        assert len(outcome.latest_pushes) == 1, outcome.events

    def test_a_critical_finding_blocks_latest(self, tmp_path: Path) -> None:
        outcome = run_push_script(tmp_path, scan_status="COMPLETE", critical="3")
        assert outcome.status != 0, outcome.output
        assert outcome.latest_pushes == [], outcome.events

    def test_a_failed_scan_blocks_latest(self, tmp_path: Path) -> None:
        outcome = run_push_script(tmp_path, scan_status="FAILED", critical="None")
        assert outcome.status != 0, outcome.output
        assert outcome.latest_pushes == [], outcome.events
