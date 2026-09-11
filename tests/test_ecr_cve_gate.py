"""The submit-time CVE gate: no image with a CRITICAL finding may reach AWS Batch.

`cloud/push_ecr.sh` gates the publish of `:latest` on the same zero-CRITICAL rule, and these tests
cover the second gate, in `cloud/submit_job.py`, which is the one that decides which image Batch
actually runs. The three holes it closes are in that module's docstring; the shape of them all is
that an image can be clean at push time and CRITICAL at submit time, or never have passed the push
gate at all.

These tests live under tests/ rather than games/tests/ because the gate is operational tooling on
the ECR/Batch path, alongside the idle watchdog.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from cloud.submit_job import (
    CveScanResult,
    EcrImageRef,
    assert_image_carries_no_critical_cves,
    critical_finding_count,
    main,
    parse_ecr_image,
    read_submission_log,
)

if TYPE_CHECKING:
    from pathlib import Path

ACCOUNT = "123456789012"
IMAGE = f"{ACCOUNT}.dkr.ecr.us-west-2.amazonaws.com/games-grpo:latest"
DIGEST = "sha256:0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c4b5a69788796a5b4c3d2e1f0"


def scan_response(
    status: str, *, critical: int | None = None, digest: str | None = DIGEST
) -> dict[str, Any]:
    """Shape a describe-image-scan-findings response.

    `critical=None` omits the CRITICAL key entirely, which is what ECR really does for a severity
    with no findings -- and is the ambiguity the gate has to resolve against the status, because it
    reads identically to a scan that has not looked yet.
    """
    counts: dict[str, int] = {"HIGH": 4, "MEDIUM": 11}
    if critical is not None:
        counts["CRITICAL"] = critical
    return {
        "imageScanStatus": {"status": status},
        "imageScanFindings": {"findingSeverityCounts": counts},
        "imageId": {"imageTag": "latest", "imageDigest": digest},
    }


class StubEcrClient:
    """Answers one describe-image-scan-findings call, recording what it was asked about."""

    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def describe_image_scan_findings(self, **kwargs: object) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        return self.response


class StubBatchClient:
    """Records anything Batch is asked to do, so a blocked submission can be shown to do nothing."""

    def __init__(self) -> None:
        self.register_calls: list[dict[str, Any]] = []
        self.submit_calls: list[dict[str, Any]] = []

    def register_job_definition(self, **kwargs: object) -> dict[str, Any]:
        self.register_calls.append(dict(kwargs))
        name = kwargs["jobDefinitionName"]
        return {
            "jobDefinitionName": name,
            "jobDefinitionArn": f"arn:aws:batch:us-west-2:{ACCOUNT}:job-definition/{name}:8",
            "revision": 8,
        }

    def submit_job(self, **kwargs: object) -> dict[str, Any]:
        self.submit_calls.append(dict(kwargs))
        return {"jobId": "job-1", "jobName": kwargs["jobName"]}


class TestParsingAnEcrReference:
    def test_the_parts_come_out(self) -> None:
        assert parse_ecr_image(IMAGE) == EcrImageRef(
            account_id=ACCOUNT, region="us-west-2", repository="games-grpo", tag="latest"
        )

    def test_a_namespaced_repository_keeps_its_slash(self) -> None:
        """Repository names may contain slashes, so the tag anchors to the end, not to a colon."""
        ref = parse_ecr_image(f"{ACCOUNT}.dkr.ecr.us-east-2.amazonaws.com/team/games-grpo:abc123")
        assert ref.repository == "team/games-grpo"
        assert ref.tag == "abc123"
        assert ref.region == "us-east-2"

    @pytest.mark.parametrize(
        "image",
        [
            "games-grpo:latest",
            "docker.io/library/python:3.13",
            "nvidia/cuda:13.3.1-runtime-ubuntu24.04",
            f"{ACCOUNT}.dkr.ecr.us-west-2.amazonaws.com/games-grpo",
            "notanaccount.dkr.ecr.us-west-2.amazonaws.com/games-grpo:latest",
        ],
    )
    def test_anything_unscannable_is_refused(self, image: str) -> None:
        """An image whose findings cannot be read has an unknown posture, which is not a clean one.

        A public base image is the interesting case: it is perfectly runnable and completely
        unscannable by this gate, so letting it through would be a hole shaped exactly like the one
        the gate exists to close.
        """
        with pytest.raises(ValueError, match="not an ECR reference"):
            parse_ecr_image(image)


class TestReadingTheCriticalCount:
    """The status is load-bearing: an absent CRITICAL key means zero only under a terminal one."""

    def test_a_complete_scan_reports_its_criticals(self) -> None:
        assert critical_finding_count(scan_response("COMPLETE", critical=3)) == 3

    def test_a_complete_scan_with_the_key_absent_is_zero(self) -> None:
        """A finished basic scan that found nothing omits the key rather than reporting zero."""
        assert critical_finding_count(scan_response("COMPLETE")) == 0

    def test_an_active_enhanced_scan_counts_as_terminal(self) -> None:
        """Inspector continuous scanning sits at ACTIVE forever and never reaches COMPLETE."""
        assert critical_finding_count(scan_response("ACTIVE")) == 0

    @pytest.mark.parametrize(
        "status", ["IN_PROGRESS", "FAILED", "SCAN_ELIGIBILITY_EXPIRED", "UNKNOWN"]
    )
    def test_a_non_terminal_status_refuses_to_answer(self, status: str) -> None:
        """Reading "no CRITICAL key" as "no CRITICALs" is how a CRITICAL image reaches Batch."""
        with pytest.raises(ValueError, match="cannot be evaluated"):
            critical_finding_count(scan_response(status))

    def test_a_missing_status_field_refuses_too(self) -> None:
        with pytest.raises(ValueError, match="cannot be evaluated"):
            critical_finding_count({"imageScanFindings": {"findingSeverityCounts": {}}})

    def test_an_in_progress_scan_that_already_found_criticals_still_refuses(self) -> None:
        """Refusing on the status is the conservative read whichever way the counts lean."""
        with pytest.raises(ValueError, match="cannot be evaluated"):
            critical_finding_count(scan_response("IN_PROGRESS", critical=2))


class TestTheGate:
    def test_a_clean_image_passes_and_reports_its_digest(self) -> None:
        client = StubEcrClient(scan_response("COMPLETE"))
        result = assert_image_carries_no_critical_cves(parse_ecr_image(IMAGE), client)
        assert result == CveScanResult(
            repository="games-grpo",
            tag="latest",
            scan_status="COMPLETE",
            critical_findings=0,
            image_digest=DIGEST,
        )

    def test_it_asks_about_the_image_being_submitted(self) -> None:
        """Scanning the wrong repository, registry or tag would be a gate that always passes."""
        client = StubEcrClient(scan_response("COMPLETE"))
        assert_image_carries_no_critical_cves(parse_ecr_image(IMAGE), client)
        assert client.calls == [
            {
                "registryId": ACCOUNT,
                "repositoryName": "games-grpo",
                "imageId": {"imageTag": "latest"},
            }
        ]

    def test_a_critical_finding_is_refused_with_somewhere_to_go(self) -> None:
        client = StubEcrClient(scan_response("COMPLETE", critical=2))
        with pytest.raises(ValueError, match="2 CRITICAL CVE finding") as raised:
            assert_image_carries_no_critical_cves(parse_ecr_image(IMAGE), client)
        message = str(raised.value)
        assert "cloud/Dockerfile" in message
        assert "describe-image-scan-findings" in message

    def test_high_and_medium_findings_are_advisory(self) -> None:
        """Only CRITICAL gates; every stub response here carries HIGH and MEDIUM counts too."""
        client = StubEcrClient(scan_response("ACTIVE"))
        result = assert_image_carries_no_critical_cves(parse_ecr_image(IMAGE), client)
        assert result.critical_findings == 0


class TestTheEntryPointRefusesBeforeItRegistersAnything:
    """The end-to-end assertion: a CRITICAL image must leave no trace in Batch at all.

    Batch pins an image tag to a digest when a job-definition revision is registered, so a revision
    registered against a CRITICAL image is a loaded gun even if the submit itself fails. The gate
    therefore has to run before the Batch client is built, and this is where that ordering is tested
    rather than assumed.
    """

    def argv(self, log_path: Path) -> list[str]:
        return [
            "--arm",
            "twin-pd-group",
            "--image",
            IMAGE,
            "--queue",
            "games-g6e-queue",
            "--log-path",
            str(log_path),
        ]

    def test_a_critical_image_never_reaches_batch(self, tmp_path: Path) -> None:
        log_path = tmp_path / "submissions.jsonl"
        batch = StubBatchClient()
        ecr = StubEcrClient(scan_response("COMPLETE", critical=1))
        with pytest.raises(ValueError, match="CRITICAL CVE finding"):
            main(self.argv(log_path), batch_client=batch, ecr_client=ecr)
        assert batch.register_calls == []
        assert batch.submit_calls == []
        assert not log_path.exists(), "a refused submission must not be logged as one"

    def test_an_unevaluated_scan_never_reaches_batch(self, tmp_path: Path) -> None:
        log_path = tmp_path / "submissions.jsonl"
        batch = StubBatchClient()
        ecr = StubEcrClient(scan_response("IN_PROGRESS"))
        with pytest.raises(ValueError, match="cannot be evaluated"):
            main(self.argv(log_path), batch_client=batch, ecr_client=ecr)
        assert batch.register_calls == []
        assert batch.submit_calls == []

    def test_a_clean_image_submits_and_the_log_records_what_was_scanned(
        self, tmp_path: Path
    ) -> None:
        log_path = tmp_path / "submissions.jsonl"
        batch = StubBatchClient()
        ecr = StubEcrClient(scan_response("ACTIVE"))
        main(self.argv(log_path), batch_client=batch, ecr_client=ecr)
        assert len(batch.submit_calls) == 1
        rows = read_submission_log(log_path)
        assert len(rows) == 1
        assert rows[0]["image_scan"] == {
            "repository": "games-grpo",
            "tag": "latest",
            "scan_status": "ACTIVE",
            "critical_findings": 0,
            "image_digest": DIGEST,
        }

    def test_the_recorded_scan_survives_a_json_round_trip(self, tmp_path: Path) -> None:
        """The row goes through json.dumps, so an unserialisable scan field would fail at submit."""
        log_path = tmp_path / "submissions.jsonl"
        main(
            self.argv(log_path),
            batch_client=StubBatchClient(),
            ecr_client=StubEcrClient(scan_response("COMPLETE", digest=None)),
        )
        row = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
        assert row["image_scan"]["image_digest"] is None
