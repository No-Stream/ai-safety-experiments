"""Submit a games training arm to AWS Batch, registering a fresh job definition every time.

**Why this file exists at all.** Batch resolves an image tag to a digest once, when a job
definition revision is registered, and then pins it forever. Reusing an existing revision after a
`docker push` therefore runs the *old* image while every log line, tag, and console page says the
new one -- a silent stale-code failure that has no symptom until results disagree with the source.
So every submit registers a new revision, and the submit request is keyed on the ARN that
registration just returned. There is deliberately no code path that accepts a job-definition name
or an externally supplied revision, because that is the path that would let a stale digest back in.

The AWS-touching surface is one thin function. Everything that decides *what* to send --
`parse_args`, `build_register_request`, `build_submit_request`, `submission_record` -- is pure and
tested offline against a stub client, so the request shape is pinned without an account.

**The second CVE gate.** No image reaches Batch from here without a terminal ECR scan reporting zero
CRITICAL findings; `main` refuses before anything is registered. `cloud/push_ecr.sh` enforces the
same rule when it publishes `:latest`, and this is not redundant with it, because what matters is the
image Batch *actually runs*, decided here rather than at push time. Three ways a CRITICAL image gets
past the push gate and would otherwise be submitted: enhanced (Inspector) scanning keeps rescanning
what is already in ECR, so a tag clean when published acquires findings from CVEs disclosed
afterwards and nobody re-reads them; when the push gate does fire it has already pushed the sha tag
and deliberately leaves it in ECR for inspection, which `--allow-nonlatest-image` will then happily
submit; and an image pushed by hand never passed the push gate at all.

One Batch subtlety worth stating: **the instance type comes from the compute environment behind the
queue, not from the job definition.** `--tier` sizes the resource request to fit a given instance
and records which instance was intended; picking the hardware is what `--queue` does. Requesting
more vCPU, memory, or GPUs than the queue's instances provide does not fail -- the job sits in
RUNNABLE indefinitely, which reads as a capacity problem and is not one.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)


class BatchClient(Protocol):
    """The two AWS Batch calls this module makes, so a test stub can stand in for boto3."""

    def register_job_definition(self, **kwargs: object) -> dict[str, Any]:
        """Register a new job-definition revision and return it, ARN included."""
        ...

    def submit_job(self, **kwargs: object) -> dict[str, Any]:
        """Submit a job against a pinned job-definition revision."""
        ...


class EcrClient(Protocol):
    """The one ECR call the CVE gate makes, so a test stub can stand in for boto3."""

    def describe_image_scan_findings(self, **kwargs: object) -> dict[str, Any]:
        """Return the scan status and severity counts for one image."""
        ...


DEFAULT_REGION = "us-west-2"
DEFAULT_LOG_PATH = Path("artifacts/games/submissions/submissions.jsonl")
DEFAULT_CONTAINER_OUTPUT_DIR = "/scratch/runs/current"

# A registered job-definition ARN always ends in ":<revision>"; see build_submit_request.
JOB_DEFINITION_ARN_PATTERN = re.compile(
    r"^arn:aws[a-z-]*:batch:[a-z0-9-]+:\d+:job-definition/(?P<name>[^:]+):(?P<revision>\d+)$"
)
REQUIRED_IMAGE_TAG = "latest"

# Batch rejects a longer jobName outright.
MAX_JOB_NAME_CHARS = 128

# Below this a job cannot realistically start, let alone train.
MIN_TIMEOUT_SECONDS = 60

# An ECR image reference, the only kind this script will submit; see parse_ecr_image.
ECR_IMAGE_PATTERN = re.compile(
    r"^(?P<account_id>\d{12})\.dkr\.ecr\.(?P<region>[a-z0-9-]+)\.amazonaws\.com/"
    r"(?P<repository>[a-z0-9][a-z0-9._/-]*):(?P<tag>[A-Za-z0-9_][A-Za-z0-9._-]*)$"
)

# The scan statuses whose findings are real; see critical_finding_count.
TERMINAL_SCAN_STATUSES = frozenset({"COMPLETE", "ACTIVE"})


@dataclass(frozen=True)
class GpuTier:
    """A resource request sized for one instance shape.

    `verified` records whether the shape was confirmed against AWS's published table for this
    instance family. Unverified tiers still submit, but log a warning: an over-request leaves the
    job RUNNABLE forever rather than failing, so a wrong number here costs an operator an hour of
    confused queue-watching.
    """

    instance_type: str
    vcpus: int
    memory_mib: int
    gpus: int
    verified: bool


GPU_TIERS: dict[str, GpuTier] = {
    "g6e-1gpu": GpuTier(
        instance_type="g6e.xlarge", vcpus=4, memory_mib=30000, gpus=1, verified=True
    ),
    # Memory sits below each instance's own total on purpose: the ECS agent and the kernel need theirs.
    "g7e-1gpu": GpuTier(
        instance_type="g7e.2xlarge", vcpus=8, memory_mib=61440, gpus=1, verified=True
    ),
    "g7e-2gpu": GpuTier(
        instance_type="g7e.12xlarge", vcpus=48, memory_mib=491520, gpus=2, verified=True
    ),
}
DEFAULT_TIER = "g6e-1gpu"


@dataclass(frozen=True)
class EcrImageRef:
    """The parts of an ECR image reference the scan lookup needs."""

    account_id: str
    region: str
    repository: str
    tag: str


@dataclass(frozen=True)
class CveScanResult:
    """A terminal, zero-CRITICAL scan of one image, recorded so a run can be audited later.

    `image_digest` is what makes the record worth keeping: the gate reads findings by tag, and a tag
    is mutable, so the digest is the only durable statement of *what* was found clean.
    """

    repository: str
    tag: str
    scan_status: str
    critical_findings: int
    image_digest: str | None


def parse_ecr_image(image: str) -> EcrImageRef:
    """Split an ECR image reference into registry account, region, repository and tag.

    Refusing anything that is not an ECR reference is deliberate rather than a limitation. The CVE
    gate below can only read findings for an image in ECR, so a reference it cannot parse is a
    reference whose vulnerability posture is simply unknown -- and an image of unknown posture is
    exactly what the gate exists to keep off Batch. Repository names may contain slashes, so
    the tag is anchored to the end of the string and the repository is everything between the
    registry host and the final colon.
    """
    match = ECR_IMAGE_PATTERN.match(image)
    if match is None:
        raise ValueError(
            f"image {image!r} is not an ECR reference of the form "
            f"<account>.dkr.ecr.<region>.amazonaws.com/<repository>:<tag>. Only ECR images can be "
            f"CVE-scanned before submission, and an image whose CRITICAL count cannot be read is "
            f"an image that must not reach Batch."
        )
    return EcrImageRef(
        account_id=match.group("account_id"),
        region=match.group("region"),
        repository=match.group("repository"),
        tag=match.group("tag"),
    )


def critical_finding_count(response: Mapping[str, Any]) -> int:
    """Read the CRITICAL count out of a describe-image-scan-findings response.

    Two fields are required, and the status is read first. `findingSeverityCounts` is a map, so a
    severity with no findings is simply absent from it -- which reads identically whether the scan
    finished clean or has not looked yet. Only the status separates those, so an absent CRITICAL key
    counts as zero under a terminal status (COMPLETE for a finished basic scan, ACTIVE for live
    enhanced scanning, which never reaches COMPLETE) and raises under any other. This mirrors
    cloud/push_ecr.sh exactly: two gates that disagree about what "clean" means are one gate plus a
    false reassurance.
    """
    status = str(response.get("imageScanStatus", {}).get("status", "UNKNOWN"))
    if status not in TERMINAL_SCAN_STATUSES:
        raise ValueError(
            f"ECR reports scan status {status!r}, so the CVE gate cannot be evaluated. A COMPLETE "
            f"basic scan or an ACTIVE enhanced one is required; under any other status an absent "
            f"CRITICAL count means the findings are not in yet, not that there are none. Enable "
            f"scanning on the repository or wait for the scan, then resubmit."
        )
    counts = response.get("imageScanFindings", {}).get("findingSeverityCounts", {})
    return int(counts.get("CRITICAL", 0))


def assert_image_carries_no_critical_cves(ref: EcrImageRef, client: EcrClient) -> CveScanResult:
    """Refuse the submission unless ECR reports a terminal scan with zero CRITICAL findings.

    A hard gate, not a tradeoff: inheriting a CRITICAL from a base image is still shipping one, and
    an image carrying one must not be the image Batch actually runs. HIGH and below are advisory and
    deliberately not read here.
    """
    response = client.describe_image_scan_findings(
        registryId=ref.account_id,
        repositoryName=ref.repository,
        imageId={"imageTag": ref.tag},
    )
    critical = critical_finding_count(response)
    digest = response.get("imageId", {}).get("imageDigest")
    if critical > 0:
        raise ValueError(
            f"{critical} CRITICAL CVE finding(s) on {ref.repository}:{ref.tag} "
            f"(digest {digest}). Refusing to submit: no image carrying a CRITICAL finding may be "
            f"the one Batch runs. Bump the base image in "
            f"cloud/Dockerfile (or the offending package), rebuild via cloud/push_ecr.sh, and "
            f"resubmit. Inspect with: aws ecr describe-image-scan-findings --repository-name "
            f"{ref.repository} --image-id imageTag={ref.tag} --region {ref.region}"
        )
    return CveScanResult(
        repository=ref.repository,
        tag=ref.tag,
        scan_status=str(response["imageScanStatus"]["status"]),
        critical_findings=critical,
        image_digest=None if digest is None else str(digest),
    )


@dataclass(frozen=True)
class SubmitConfig:
    """Everything one submission needs, with nothing AWS-specific resolved yet."""

    arm: str
    image: str
    queue: str
    job_definition_name: str
    tier: str = DEFAULT_TIER
    region: str = DEFAULT_REGION
    s3_dest: str | None = None
    output_dir: str = DEFAULT_CONTAINER_OUTPUT_DIR
    timeout_seconds: int = 24 * 60 * 60
    job_role_arn: str | None = None
    execution_role_arn: str | None = None
    vllm_server: bool = False
    require_fast_kernels: bool = False
    allow_nonlatest_image: bool = False
    train_args: tuple[str, ...] = ()
    log_path: Path = DEFAULT_LOG_PATH

    def __post_init__(self) -> None:
        """Reject a configuration that would submit against the wrong hardware or the wrong tag."""
        if self.tier not in GPU_TIERS:
            raise ValueError(f"Unknown tier {self.tier!r}; known tiers: {sorted(GPU_TIERS)}.")
        if self.timeout_seconds < MIN_TIMEOUT_SECONDS:
            raise ValueError(
                f"timeout_seconds must be at least {MIN_TIMEOUT_SECONDS}, got "
                f"{self.timeout_seconds}; a job with no "
                f"real wall-clock cap is how a wedged run bills overnight."
            )
        if ":" not in self.image:
            raise ValueError(
                f"image {self.image!r} carries no tag. Pass an explicit tag so the log records "
                f"what was submitted."
            )
        tag = self.image.rsplit(":", 1)[1]
        if tag != REQUIRED_IMAGE_TAG and not self.allow_nonlatest_image:
            raise ValueError(
                f"image tag is {tag!r}, not {REQUIRED_IMAGE_TAG!r}. Job definitions resolve "
                f"{REQUIRED_IMAGE_TAG!r} by convention (see cloud/push_ecr.sh): both tags point at "
                f"the same digest at push time, and re-registering every submit is what keeps that "
                f"safe. Pass --allow-nonlatest-image to submit a pinned sha deliberately."
            )

    @property
    def gpu_tier(self) -> GpuTier:
        """The resource request this submission is sized for."""
        return GPU_TIERS[self.tier]


def container_environment(config: SubmitConfig) -> list[dict[str, str]]:
    """Build the container environment, in Batch's name/value form.

    GIT_SHA is deliberately absent: it is baked into the image at build time, and passing it here
    would let a submitter's value silently contradict the code actually running.
    """
    environment = {
        "GAMES_OUTPUT_DIR": config.output_dir,
        "GAMES_REQUIRE_FAST_KERNELS": "1" if config.require_fast_kernels else "0",
        "GAMES_VLLM_SERVER": "1" if config.vllm_server else "0",
    }
    if config.s3_dest:
        environment["GAMES_S3_DEST"] = config.s3_dest
    return [{"name": name, "value": value} for name, value in sorted(environment.items())]


def build_register_request(config: SubmitConfig) -> dict[str, Any]:
    """Shape the RegisterJobDefinition payload for a fresh revision.

    `retryStrategy.attempts` is 1 on purpose. A retried training run spends the money again and
    writes a second set of artifacts under the same name, and a run that died for a real reason
    dies again -- so a failure should surface, not silently repeat.
    """
    tier = config.gpu_tier
    command = ["--arm", config.arm, *config.train_args]
    container: dict[str, Any] = {
        "image": config.image,
        "command": command,
        "environment": container_environment(config),
        "resourceRequirements": [
            {"type": "VCPU", "value": str(tier.vcpus)},
            {"type": "MEMORY", "value": str(tier.memory_mib)},
            {"type": "GPU", "value": str(tier.gpus)},
        ],
    }
    if config.job_role_arn:
        container["jobRoleArn"] = config.job_role_arn
    if config.execution_role_arn:
        container["executionRoleArn"] = config.execution_role_arn
    return {
        "jobDefinitionName": config.job_definition_name,
        "type": "container",
        "platformCapabilities": ["EC2"],
        "containerProperties": container,
        "retryStrategy": {"attempts": 1},
        "timeout": {"attemptDurationSeconds": config.timeout_seconds},
        "tags": {
            "arm": config.arm,
            "tier": config.tier,
            "intended_instance_type": tier.instance_type,
        },
        "propagateTags": True,
    }


def build_submit_request(
    config: SubmitConfig, *, job_definition_arn: str, job_name: str
) -> dict[str, Any]:
    """Shape the SubmitJob payload, keyed on the revision-bearing ARN just registered.

    Takes an ARN rather than a job-definition name, and requires the revision suffix, so no caller
    can express "submit against whatever revision Batch thinks is current". That is the one thing
    this module exists to make unrepresentable.
    """
    match = JOB_DEFINITION_ARN_PATTERN.match(job_definition_arn)
    if match is None:
        raise ValueError(
            f"job_definition_arn {job_definition_arn!r} is not a revision-bearing job-definition "
            f"ARN. Submitting without a pinned revision lets Batch resolve one itself, which is "
            f"how a stale image digest gets run while the logs claim the new code."
        )
    if match.group("name") != config.job_definition_name:
        raise ValueError(
            f"ARN names job definition {match.group('name')!r} but this submission is for "
            f"{config.job_definition_name!r}; refusing to submit against a different definition."
        )
    return {
        "jobName": job_name,
        "jobQueue": config.queue,
        "jobDefinition": job_definition_arn,
    }


def job_name_for(config: SubmitConfig, *, now: datetime) -> str:
    """Build a Batch-legal job name: alphanumerics, hyphens and underscores only, <= 128 chars."""
    stamp = now.strftime("%Y%m%d-%H%M%S")
    raw = f"games-{config.arm}-{config.tier}-{stamp}"
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "-", raw)
    return cleaned[:MAX_JOB_NAME_CHARS]


def _git_output(*args: str) -> str:
    """Run a read-only git command, returning empty string if git cannot answer."""
    finished = subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    return finished.stdout.strip() if finished.returncode == 0 else ""


def submission_record(  # noqa: PLR0913
    config: SubmitConfig,
    *,
    job_id: str,
    job_name: str,
    job_definition_arn: str,
    now: datetime,
    image_scan: CveScanResult | None = None,
) -> dict[str, Any]:
    """Build the JSONL row describing one submission.

    `submitter_git_sha` is the tree that ran this script, which is *not* proof of what the image
    contains -- the image's own baked GIT_SHA is, and the entrypoint logs it as its first line.
    Both are recorded, plus whether the submitter's tree was dirty, so the two can be compared
    rather than assumed equal.

    `image_scan` records the CVE posture the gate actually observed, digest included, so "was this
    run's image clean when it was submitted?" is answerable months later from the log rather than
    from a rescan that will by then report a different answer.
    """
    tier = config.gpu_tier
    scan = None if image_scan is None else asdict(image_scan)
    return {
        "submitted_at": now.isoformat(),
        "job_id": job_id,
        "job_name": job_name,
        "job_definition_arn": job_definition_arn,
        "arm": config.arm,
        "queue": config.queue,
        "tier": config.tier,
        "intended_instance_type": tier.instance_type,
        "gpus": tier.gpus,
        "vcpus": tier.vcpus,
        "memory_mib": tier.memory_mib,
        "image": config.image,
        "region": config.region,
        "s3_dest": config.s3_dest,
        "train_args": list(config.train_args),
        "submitter_git_sha": _git_output("rev-parse", "HEAD"),
        "submitter_tree_dirty": bool(_git_output("status", "--porcelain")),
        "image_scan": scan,
    }


def append_submission_log(log_path: Path, record: dict[str, Any]) -> None:
    """Append one submission row, creating the directory if this is the first submit."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def read_submission_log(log_path: Path) -> list[dict[str, Any]]:
    """Read the submission log back, for anyone reconciling a job id with an arm months later."""
    if not log_path.is_file():
        return []
    with log_path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def submit(
    config: SubmitConfig,
    client: BatchClient,
    *,
    now: datetime | None = None,
    image_scan: CveScanResult | None = None,
) -> dict[str, Any]:
    """Register a fresh job definition, submit against it, and log the submission.

    `client` is injected rather than constructed here so the whole flow can be exercised against a
    stub. Register-then-submit is the required order: the ARN the submit is keyed on does not exist
    until registration returns it.
    """
    moment = now if now is not None else datetime.now(UTC)
    tier = config.gpu_tier
    if not tier.verified:
        logger.warning(
            f"tier {config.tier!r} has an unverified instance shape "
            f"({tier.instance_type}, {tier.vcpus} vCPU, {tier.memory_mib} MiB, {tier.gpus} GPU). "
            f"Confirm it against the queue's compute environment: an over-request does not fail, "
            f"it leaves the job RUNNABLE indefinitely."
        )

    registered = client.register_job_definition(**build_register_request(config))
    job_definition_arn = registered["jobDefinitionArn"]
    revision = registered.get("revision")
    logger.info(
        f"registered a fresh job definition, {job_definition_arn=} {revision=} image={config.image}"
    )

    job_name = job_name_for(config, now=moment)
    submitted = client.submit_job(
        **build_submit_request(config, job_definition_arn=job_definition_arn, job_name=job_name)
    )
    job_id = submitted["jobId"]

    record = submission_record(
        config,
        job_id=job_id,
        job_name=job_name,
        job_definition_arn=job_definition_arn,
        now=moment,
        image_scan=image_scan,
    )
    append_submission_log(config.log_path, record)
    logger.info(f"submitted, {job_id=} {job_name=} queue={config.queue}")
    return record


def parse_args(argv: Sequence[str] | None = None) -> SubmitConfig:
    """Parse the CLI into a `SubmitConfig`; everything after `--` goes to games.train."""
    parser = argparse.ArgumentParser(
        description="Submit a games training arm to AWS Batch (fresh job definition every time).",
        epilog="Arguments after -- are forwarded to python -m games.train.",
    )
    parser.add_argument("--arm", required=True)
    parser.add_argument("--image", required=True, help="ECR image reference, normally :latest")
    parser.add_argument("--queue", required=True, help="Batch job queue; this picks the hardware")
    parser.add_argument("--job-definition-name", default="games-grpo")
    parser.add_argument("--tier", default=DEFAULT_TIER, choices=sorted(GPU_TIERS))
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--s3-dest", default=None, help="s3://bucket/prefix for run artifacts")
    parser.add_argument("--output-dir", default=DEFAULT_CONTAINER_OUTPUT_DIR)
    parser.add_argument("--timeout-seconds", type=int, default=24 * 60 * 60)
    parser.add_argument("--job-role-arn", default=None)
    parser.add_argument("--execution-role-arn", default=None)
    parser.add_argument("--vllm-server", action="store_true")
    parser.add_argument("--require-fast-kernels", action="store_true")
    parser.add_argument("--allow-nonlatest-image", action="store_true")
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("train_args", nargs="*", help="forwarded to games.train after --")
    args = parser.parse_args(argv)
    return SubmitConfig(
        arm=args.arm,
        image=args.image,
        queue=args.queue,
        job_definition_name=args.job_definition_name,
        tier=args.tier,
        region=args.region,
        s3_dest=args.s3_dest,
        output_dir=args.output_dir,
        timeout_seconds=args.timeout_seconds,
        job_role_arn=args.job_role_arn,
        execution_role_arn=args.execution_role_arn,
        vllm_server=args.vllm_server,
        require_fast_kernels=args.require_fast_kernels,
        allow_nonlatest_image=args.allow_nonlatest_image,
        train_args=tuple(args.train_args),
        log_path=args.log_path,
    )


def build_batch_client(region: str) -> BatchClient:
    """Create the real Batch client.

    boto3 is imported here rather than at module top because it is an optional dependency of this
    repo (the `bedrock` extra) and is deliberately absent from the training image: nothing inside a
    job submits jobs. Importing it at module scope would make the offline tests, which never touch
    AWS, depend on an extra they do not need.
    """
    import boto3  # noqa: PLC0415

    return cast("BatchClient", boto3.client("batch", region_name=region))


def build_ecr_client(region: str) -> EcrClient:
    """Create the real ECR client, for the region the image lives in.

    Takes its own region rather than the submission's: the image and the Batch queue need not sit in
    the same region, and reading findings from the wrong registry would gate on another image.
    """
    import boto3  # noqa: PLC0415

    return cast("EcrClient", boto3.client("ecr", region_name=region))


def main(
    argv: Sequence[str] | None = None,
    *,
    batch_client: BatchClient | None = None,
    ecr_client: EcrClient | None = None,
) -> None:
    """Scan the image, refuse it if it carries a CRITICAL CVE, then submit one arm.

    The gate runs before the Batch client is even built, so a refused image leaves no job-definition
    revision behind. Both clients are injectable so the whole entry point, gate included, can be
    exercised without an account; absent an injection each is the real boto3 client.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = parse_args(argv)

    ref = parse_ecr_image(config.image)
    ecr = build_ecr_client(ref.region) if ecr_client is None else ecr_client
    scan = assert_image_carries_no_critical_cves(ref, ecr)
    logger.info(
        f"CVE gate passed, {scan.repository}:{scan.tag} status={scan.scan_status} "
        f"critical={scan.critical_findings} digest={scan.image_digest}"
    )

    batch = build_batch_client(config.region) if batch_client is None else batch_client
    record = submit(config, batch, image_scan=scan)
    sys.stdout.write(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
