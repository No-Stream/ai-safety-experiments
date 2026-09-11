"""The transfer CLI end to end offline: plan writing, live resume, batch submit/collect, the smoke.

Nothing here touches AWS. The batch path runs against an in-memory S3 and a scripted control plane, which
is what makes the five-way handle check testable at all: each of its five mismatches is planted into a
saved handle deliberately and required to refuse, because every one of them would otherwise file real
responses under the wrong cells with every count looking plausible.

Three of the classes here are refusals rather than behaviours, and each covers a failure that leaves every
artifact looking healthy: a resume that continued a file written under a different stimulus, a judge that
ran before anything checked its rubrics, and a writer pointed at a git-tracked path on a public remote.
The writer sweep is parametrized over every writer this pass has, because the one leak of that class this
repository has actually had was a second call site nobody had guarded.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from games.prompts import MATCHED_DECISION_TRANSFER_GAME_ID, ONE_WAY_TRANSFER_GAME_ID
from reward_hacking.bedrock_batch import (
    BATCH_BUCKET_ENV,
    BATCH_BUCKET_OWNER_ENV,
    BATCH_PROFILE_ENV,
    BATCH_ROLE_ARN_ENV,
    MIN_BATCH_RECORDS,
)
from reward_hacking.trace import _repo_root
from sociology import transfer_cli as cli
from sociology import transfer_judge as judge_module
from sociology import transfer_scans as scans_module
from sociology.model_stub import ScriptedDetailedBackend
from sociology.records import append_replies, write_summary
from sociology.tests.conftest import (
    synthetic_transfer_payload,
    synthetic_transfer_payload_without_boards,
    write_synthetic_transfer_stimulus,
)
from sociology.transfer_plan import (
    CORRELATION_DOSE_PLAN,
    DESIGN_LABELS,
    FINGERPRINT_PLAN,
    KNOCKOUT_PLAN,
    ONE_WAY_TRANSFER_PLAN,
    leg_for,
    plan_table,
    planned_calls_for_leg,
)
from sociology.transfer_scans import load_run_replies
from sociology.transfer_stimulus import (
    BOARD_IDS,
    BOARD_MESSAGE_COUNT,
    BOARD_REDRAW_CAP,
    BOARD_SOURCE_BOARD,
    BOARD_SOURCE_DIGEST_FIELD,
    JUDGE_ARM_GAME_IDS,
    MATCHED_ROUNDS_LADDER,
    N_SCENARIOS,
    RUNG_SAME_CHECKPOINT_RECORD_0,
    load_stimulus,
    message_digest,
    record_rung_id,
)

ONE_WAY = ONE_WAY_TRANSFER_PLAN
"""The pass these tests exercise. Spelled once, because every subcommand now takes a plan table."""


PASS_ARGV: tuple[str, str] = ("--pass", ONE_WAY.pass_id)
"""The pass flag every invocation carries, because ``--pass`` is required and never defaulted."""


def narrowed(table: PlanTable, *leg_ids: str) -> PlanTable:
    """The same pass carrying only the named legs, so a build or dry-run test stays cheap.

    A narrowed TABLE rather than a patched module function: the subcommands read their legs off the table
    they are handed, which is what makes the pass a parameter rather than a global.
    """
    return replace(table, legs=tuple(leg_for(leg_id) for leg_id in leg_ids))


if TYPE_CHECKING:
    from collections.abc import Callable

    from reward_hacking.model_backend import DetailedBackend
    from sociology.transfer_plan import PlannedCall, PlanTable
    from sociology.transfer_stimulus import TransferStimulus

FLOOR_LEG_ID = "gpt-oss-20b--default--B--floor"
FLOOR_RECORDS = 256
LIVE_LEG_ID = "luna--default--A--identity-ow"
LIVE_REPLY_FILE = (
    "replies--global-openai-gpt-5-6-luna--effort-default--sitting-A--identity-ow.jsonl"
)

JOB_ARN = "SYNTHETIC-JOB-ARN/testjob"


class FakeS3:
    """An in-memory S3, keyed by ``s3://bucket/key``."""

    def __init__(self) -> None:
        self.objects: dict[str, str] = {}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> dict[str, Any]:  # noqa: N803
        self.objects[f"s3://{Bucket}/{Key}"] = Body.decode("utf-8")
        return {}

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        uri = f"s3://{Bucket}/{Key}"
        if uri not in self.objects:
            raise KeyError(uri)
        return {"Body": SimpleNamespace(read=lambda: self.objects[uri].encode("utf-8"))}


class FakeBedrock:
    """A fake ``bedrock`` control plane that always reports an ACTIVE model and a Completed job."""

    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []

    def get_foundation_model(self, *, modelIdentifier: str) -> dict[str, Any]:  # noqa: N803
        return {
            "modelDetails": {"modelId": modelIdentifier, "modelLifecycle": {"status": "ACTIVE"}}
        }

    def get_inference_profile(self, *, inferenceProfileIdentifier: str) -> dict[str, Any]:  # noqa: N803
        return {"inferenceProfileId": inferenceProfileIdentifier, "status": "ACTIVE"}

    def create_model_invocation_job(self, **kwargs: Any) -> dict[str, Any]:
        self.created.append(kwargs)
        return {"jobArn": JOB_ARN}

    def get_model_invocation_job(self, *, jobIdentifier: str) -> dict[str, Any]:  # noqa: N803
        return {"jobArn": jobIdentifier, "status": "Completed"}


@pytest.fixture
def aws(monkeypatch: pytest.MonkeyPatch) -> tuple[FakeBedrock, FakeS3]:
    """Export synthetic values for the four account settings and stand in for boto3.

    The real account, role, bucket and profile are deliberately absent from this repository, so a test that
    wants a constructed backend supplies its own -- and these are obviously artificial rather than
    realistic, because a realistic-looking ARN in a tracked test is the same leak.
    """
    monkeypatch.setenv(BATCH_ROLE_ARN_ENV, "SYNTHETIC-SERVICE-ROLE")
    monkeypatch.setenv(BATCH_BUCKET_ENV, "synthetic-bucket")
    monkeypatch.setenv(BATCH_PROFILE_ENV, "synthetic-profile")
    monkeypatch.setenv(BATCH_BUCKET_OWNER_ENV, "SYNTHETIC-BUCKET-OWNER")
    bedrock = FakeBedrock()
    s3 = FakeS3()

    def make_session(**_: Any) -> SimpleNamespace:
        return SimpleNamespace(
            client=lambda service_name: bedrock if service_name == "bedrock" else s3
        )

    import sys  # noqa: PLC0415 - the fake has to land in sys.modules before the lazy import runs

    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(Session=make_session))
    return bedrock, s3


@pytest.fixture
def pinned_stimulus(
    monkeypatch: pytest.MonkeyPatch, transfer_stimulus: TransferStimulus
) -> TransferStimulus:
    """Point the CLI's loader at the synthetic stimulus rather than the gitignored real file.

    The stand-in takes the loader's keyword arguments and ignores them: the synthetic file always carries
    its boards, so ``allow_empty_boards`` has nothing to relax, and a stand-in that refused the keyword
    would make every build test fail for a reason that has nothing to do with what it checks.
    """

    def loader(path: Path | None = None, *, allow_empty_boards: bool = False) -> TransferStimulus:
        del path, allow_empty_boards
        return transfer_stimulus

    monkeypatch.setattr(cli, "load_stimulus", loader)
    return transfer_stimulus


def scripted_factory(*replies: str) -> Callable[[str, str | None, int], DetailedBackend]:
    def factory(model_id: str, effort: str | None, concurrency: int) -> DetailedBackend:
        del model_id, effort, concurrency
        return ScriptedDetailedBackend(list(replies))

    return factory


def fulfil_job(s3: FakeS3, handle: dict[str, Any]) -> None:
    """Write the batch job's result and manifest objects, in a shuffled record order."""
    body = s3.objects[str(handle["input_uri"])]
    records = [json.loads(line) for line in body.splitlines() if line.strip()]
    job_id = str(handle["job_arn"]).rsplit("/", 1)[-1]
    base = f"{str(handle['output_uri']).rstrip('/')}/{job_id}"
    lines: list[str] = []
    for record in reversed(records):
        prompt = record["modelInput"]["messages"][0]["content"][0]["text"]
        lines.append(
            json.dumps(
                {
                    "recordId": record["recordId"],
                    "modelInput": record["modelInput"],
                    "modelOutput": {
                        "output": {"message": {"content": [{"text": cli._scripted_reply(prompt)}]}},
                        "stopReason": "end_turn",
                        "usage": {"inputTokens": 10, "outputTokens": 5},
                    },
                }
            )
        )
    s3.objects[f"{base}/input.jsonl.out"] = "\n".join(lines) + "\n"
    s3.objects[f"{base}/manifest.json.out"] = json.dumps(
        {
            "totalRecordCount": len(records),
            "successRecordCount": len(records),
            "errorRecordCount": 0,
            "inputTokenCount": 10 * len(records),
            "outputTokenCount": 5 * len(records),
        }
    )


class TestBuildAndDryRun:
    TWO_LEGS: tuple[str, str] = (FLOOR_LEG_ID, "glm-4-7--default--B--floor")

    def test_build_writes_the_plan_and_its_summary_marker_last(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus
    ) -> None:
        plan = cli.build(tmp_path, narrowed(ONE_WAY, *self.TWO_LEGS))
        assert plan["total_records"] == 2 * FLOOR_RECORDS
        assert plan["audited_prompts"] == 2 * FLOOR_RECORDS
        # Eight scenarios, both polarities, and all three unordered pairs of the three games.
        assert plan["game_pairs_compared"] == 48
        assert plan["stimulus_digest"] == pinned_stimulus.digest
        on_disk = json.loads((tmp_path / "plan.json").read_text(encoding="utf-8"))
        assert [entry["leg_id"] for entry in on_disk["legs"]] == [
            FLOOR_LEG_ID,
            "glm-4-7--default--B--floor",
        ]
        assert all(entry["prompt_digest"] and entry["cell_digest"] for entry in on_disk["legs"])
        # The plan names every batch job before a submit, so a refused name is readable off the plan.
        by_id = {entry["leg_id"]: entry for entry in on_disk["legs"]}
        glm_floor = by_id["glm-4-7--default--B--floor"]
        assert glm_floor["batch_run_id"] == "owt-glm-4-7-default-b-floor"
        assert glm_floor["batch_job_name"] == "jagged-zai-glm-4-7-owt-glm-4-7-default-b-floor"
        for entry in on_disk["legs"]:
            if entry["transport"] != "batch":
                assert entry["batch_run_id"] is None
                assert entry["batch_job_name"] is None
        summary = json.loads((tmp_path / "summary-build.json").read_text(encoding="utf-8"))
        assert summary["finished_at"]
        assert "git_sha" in summary

    def test_the_plan_never_stores_prompt_or_clause_text(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus
    ) -> None:
        del pinned_stimulus
        cli.build(tmp_path, narrowed(ONE_WAY, *self.TWO_LEGS))
        text = (tmp_path / "plan.json").read_text(encoding="utf-8")
        assert "About the other side" not in text
        assert "SYNTHETIC-IDENTITY" not in text
        assert "SYNTHETIC-FRAME" not in text

    def test_dry_run_prices_batch_off_the_verified_roster(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus
    ) -> None:
        del pinned_stimulus
        report = cli.dry_run(tmp_path, narrowed(ONE_WAY, *self.TWO_LEGS))
        legs = report["legs"]
        assert isinstance(legs, list)
        assert all(entry["price_note"] == "verified batch price" for entry in legs)
        assert report["legs_priced_unverified"] == []
        assert report["estimated_total_usd_production_legs"] > 0
        assumptions = report["assumptions"]
        assert isinstance(assumptions, dict)
        assert assumptions["output_tokens_per_call"] == cli.ASSUMED_OUTPUT_TOKENS_PER_CALL

    def test_dry_run_names_every_leg_priced_off_an_unverified_figure(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus
    ) -> None:
        """Two of the three live rows are priced off list prices nobody has read back."""
        del pinned_stimulus
        report = cli.dry_run(
            tmp_path,
            narrowed(
                ONE_WAY,
                "luna--default--B--floor",
                "sonnet-5--default--B--floor",
                "opus-5--default--A--identity-ow",
            ),
        )
        by_id = {entry["leg_id"]: entry for entry in report["legs"]}
        assert by_id["luna--default--B--floor"]["price_verified"] is True
        assert by_id["sonnet-5--default--B--floor"]["price_verified"] is False
        assert "UNVERIFIED" in by_id["opus-5--default--A--identity-ow"]["price_note"]
        assert report["legs_priced_unverified"] == [
            "sonnet-5--default--B--floor",
            "opus-5--default--A--identity-ow",
        ]
        assert (
            report["estimated_total_usd_all_legs"] > (report["estimated_total_usd_production_legs"])
        )


class TestLiveResume:
    def calls(self, stimulus: TransferStimulus, count: int) -> list[PlannedCall]:
        return cli.planned_calls_for_leg(leg_for(LIVE_LEG_ID), stimulus)[:count]

    def reply_file(self, tmp_path: Path) -> Path:
        return tmp_path / LIVE_REPLY_FILE

    def test_a_second_pass_resumes_what_the_first_finished(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus
    ) -> None:
        """Gate 5 at smoke scale: kill after two calls, relaunch, and the finished work is skipped."""
        out = self.reply_file(tmp_path)
        first = cli._run_live_calls(
            self.calls(pinned_stimulus, 2),
            out,
            pinned_stimulus,
            concurrency=1,
            chunk_size=1,
            backend_factory=scripted_factory("<set>4</set>"),
        )
        assert first == {"planned": 2, "resumed": 0, "ran": 2, "incomplete": 0}
        second = cli._run_live_calls(
            self.calls(pinned_stimulus, 5),
            out,
            pinned_stimulus,
            concurrency=1,
            chunk_size=2,
            backend_factory=scripted_factory("<set>4</set>"),
        )
        assert second == {"planned": 5, "resumed": 2, "ran": 3, "incomplete": 0}
        assert len(load_run_replies(tmp_path)) == 5

    def test_an_incomplete_record_is_flagged_kept_and_counted(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus
    ) -> None:
        def factory(model_id: str, effort: str | None, concurrency: int) -> DetailedBackend:
            del model_id, effort, concurrency
            return ScriptedDetailedBackend(
                ["partial text"], stop_reason="call_failed:ReadTimeoutError"
            )

        counts = cli._run_live_calls(
            self.calls(pinned_stimulus, 1),
            self.reply_file(tmp_path),
            pinned_stimulus,
            concurrency=1,
            chunk_size=1,
            backend_factory=factory,
        )
        assert counts["incomplete"] == 1
        row = next(iter(load_run_replies(tmp_path).values()))
        assert row["incomplete"] is True
        assert row["reply"] == "partial text"

    def test_a_duplicated_key_on_disk_refuses_at_load(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus
    ) -> None:
        out = self.reply_file(tmp_path)
        cli._run_live_calls(
            self.calls(pinned_stimulus, 1),
            out,
            pinned_stimulus,
            concurrency=1,
            chunk_size=1,
            backend_factory=scripted_factory("<set>4</set>"),
        )
        line = out.read_text(encoding="utf-8")
        out.write_text(line + line, encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate reply key"):
            load_run_replies(tmp_path)

    def test_the_reply_record_carries_every_planned_label_and_the_sampler_stamp(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus
    ) -> None:
        cli._run_live_calls(
            self.calls(pinned_stimulus, 1),
            self.reply_file(tmp_path),
            pinned_stimulus,
            concurrency=1,
            chunk_size=1,
            backend_factory=scripted_factory("<set>4</set>"),
        )
        row = next(iter(load_run_replies(tmp_path).values()))
        for field in (
            "block",
            "game_id",
            "cell",
            "variant",
            "scenario_id",
            "polarity",
            "prompt_id",
            "endowment",
            "credit_numerator",
            "credit_denominator",
            "beneficiary_count",
            "own_stake_scale",
            "model_id",
            "transport",
            "sitting",
            "draw",
            "max_tokens",
            "stimulus_digest",
            "prompt_digest",
            "prompt_chars",
            "stop_reason",
            "input_tokens",
            "output_tokens",
            "recorded_at",
        ):
            assert field in row, field
        assert row["max_tokens"] == cli.MAX_TOKENS
        assert row["stimulus_digest"] == pinned_stimulus.prompt_digest
        assert row["stimulus_digest"] != pinned_stimulus.digest
        assert "prompt" not in row, "the prompt text is re-derivable and must not be stored"

    def test_a_second_pass_under_an_edited_stimulus_refuses(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus
    ) -> None:
        """Nothing about the file's shape changes when a resume stops being a continuation."""
        out = self.reply_file(tmp_path)
        cli._run_live_calls(
            self.calls(pinned_stimulus, 2),
            out,
            pinned_stimulus,
            concurrency=1,
            chunk_size=1,
            backend_factory=scripted_factory("<set>4</set>"),
        )
        edited = replace(pinned_stimulus, prompt_digest="0f0f0f0f0f0f0f0f")
        with pytest.raises(ValueError, match="stimulus_digest") as raised:
            cli._run_live_calls(
                self.calls(edited, 4),
                out,
                edited,
                concurrency=1,
                chunk_size=1,
                backend_factory=scripted_factory("<set>4</set>"),
            )
        assert pinned_stimulus.prompt_digest in str(raised.value)
        assert "0f0f0f0f0f0f0f0f" in str(raised.value)

    def test_a_rubric_only_edit_between_passes_lets_the_resume_proceed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A rubric edit changes what the judge reads and nothing a reply row answered.

        Through the real loader on two files rather than a ``replace``, because the split between the two
        digests is what is under test: the whole-payload digest moves and the prompt digest does not.
        """
        first = load_stimulus(write_synthetic_transfer_stimulus(tmp_path / "first.json"))
        second = load_stimulus(
            write_synthetic_transfer_stimulus(
                tmp_path / "second.json",
                judge_instructions_twin="SYNTHETIC-RUBRIC-MD: reply once, differently.",
            )
        )
        assert second.digest != first.digest
        assert second.prompt_digest == first.prompt_digest
        out = self.reply_file(tmp_path)
        monkeypatch.setattr(cli, "load_stimulus", lambda: first)
        cli._run_live_calls(
            self.calls(first, 2),
            out,
            first,
            concurrency=1,
            chunk_size=1,
            backend_factory=scripted_factory("<set>4</set>"),
        )
        resumed = cli._run_live_calls(
            self.calls(second, 3),
            out,
            second,
            concurrency=1,
            chunk_size=1,
            backend_factory=scripted_factory("<set>4</set>"),
        )
        assert resumed == {"planned": 3, "resumed": 2, "ran": 1, "incomplete": 0}

    def test_a_frame_edit_between_passes_refuses_the_resume_by_the_stimulus_digest(
        self, tmp_path: Path
    ) -> None:
        """Edit a frame none of the resumed calls render, so only the stimulus digest can catch it."""
        first = load_stimulus(write_synthetic_transfer_stimulus(tmp_path / "first.json"))
        calls = self.calls(first, 2)
        rendered = {call.scenario_id for call in calls}
        payload = synthetic_transfer_payload()
        scenarios = payload["scenarios"]
        assert isinstance(scenarios, list)
        untouched = next(entry for entry in scenarios if entry["scenario_id"] not in rendered)
        edited_scenarios = [
            {**entry, "frame": entry["frame"] + " SYNTHETIC-EDIT."} if entry is untouched else entry
            for entry in scenarios
        ]
        second = load_stimulus(
            write_synthetic_transfer_stimulus(tmp_path / "second.json", scenarios=edited_scenarios)
        )
        assert second.prompt_digest != first.prompt_digest
        out = self.reply_file(tmp_path)
        cli._run_live_calls(
            calls,
            out,
            first,
            concurrency=1,
            chunk_size=1,
            backend_factory=scripted_factory("<set>4</set>"),
        )
        with pytest.raises(ValueError, match="stimulus_digest") as raised:
            cli._run_live_calls(
                self.calls(second, 2),
                out,
                second,
                concurrency=1,
                chunk_size=1,
                backend_factory=scripted_factory("<set>4</set>"),
            )
        assert "prompt_digest=" not in str(raised.value), (
            "the resumed calls render the same prompts; only the stimulus digest moved"
        )

    def test_a_record_whose_prompt_digest_moved_refuses_and_names_both(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus
    ) -> None:
        out = self.reply_file(tmp_path)
        cli._run_live_calls(
            self.calls(pinned_stimulus, 1),
            out,
            pinned_stimulus,
            concurrency=1,
            chunk_size=1,
            backend_factory=scripted_factory("<set>4</set>"),
        )
        row = json.loads(out.read_text(encoding="utf-8").strip())
        row["prompt_digest"] = "ffffffffffffffff"
        out.write_text(json.dumps(row) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="prompt_digest") as raised:
            cli._run_live_calls(
                self.calls(pinned_stimulus, 1),
                out,
                pinned_stimulus,
                concurrency=1,
                chunk_size=1,
                backend_factory=scripted_factory("<set>4</set>"),
            )
        assert row["key"] in str(raised.value)

    def test_run_live_refuses_a_batch_leg(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not live"):
            cli.run_live(FLOOR_LEG_ID, tmp_path, ONE_WAY, concurrency=1, chunk_size=1)


class TestBatchSubmit:
    def test_a_submit_writes_the_handle_and_creates_exactly_one_job(
        self, tmp_path: Path, aws: tuple[FakeBedrock, FakeS3], pinned_stimulus: TransferStimulus
    ) -> None:
        del pinned_stimulus
        bedrock, _ = aws
        summary = cli.submit_batch(FLOOR_LEG_ID, tmp_path, ONE_WAY, check_only=False)
        assert summary["records"] == FLOOR_RECORDS
        assert len(bedrock.created) == 1
        assert cli.handle_path(tmp_path, leg_for(FLOOR_LEG_ID)).exists()

    def test_check_only_spends_nothing_and_creates_no_job(
        self, tmp_path: Path, aws: tuple[FakeBedrock, FakeS3], pinned_stimulus: TransferStimulus
    ) -> None:
        del pinned_stimulus
        bedrock, s3 = aws
        summary = cli.submit_batch(FLOOR_LEG_ID, tmp_path, ONE_WAY, check_only=True)
        assert summary["check_only"] is True
        assert bedrock.created == []
        assert s3.objects == {}
        assert not cli.handle_path(tmp_path, leg_for(FLOOR_LEG_ID)).exists()

    def test_a_second_submit_reuses_the_saved_handle_and_bills_nothing_new(
        self, tmp_path: Path, aws: tuple[FakeBedrock, FakeS3], pinned_stimulus: TransferStimulus
    ) -> None:
        del pinned_stimulus
        bedrock, _ = aws
        first = cli.submit_batch(FLOOR_LEG_ID, tmp_path, ONE_WAY, check_only=False)
        second = cli.submit_batch(FLOOR_LEG_ID, tmp_path, ONE_WAY, check_only=False)
        assert second["job_arn"] == first["job_arn"]
        assert len(bedrock.created) == 1

    @pytest.mark.parametrize(
        ("field", "value", "expected"),
        [
            ("model_id", "deepseek.v3.2", "model"),
            ("record_count", 191, "records on the handle"),
            ("prompt_digest", "0" * 64, "prompt digest"),
            ("cell_digest", "0" * 64, "cell digest"),
            ("max_tokens", 4096, "on the handle vs"),
            ("reasoning_effort", "high", "on the handle vs"),
        ],
    )
    def test_each_of_the_handle_mismatches_refuses(  # noqa: PLR0913, PLR0917 - parametrized triple
        self,
        tmp_path: Path,
        aws: tuple[FakeBedrock, FakeS3],
        pinned_stimulus: TransferStimulus,
        field: str,
        value: object,
        expected: str,
    ) -> None:
        del pinned_stimulus
        cli.submit_batch(FLOOR_LEG_ID, tmp_path, ONE_WAY, check_only=False)
        path = cli.handle_path(tmp_path, leg_for(FLOOR_LEG_ID))
        handle = json.loads(path.read_text(encoding="utf-8"))
        handle[field] = value
        path.write_text(json.dumps(handle), encoding="utf-8")
        with pytest.raises(RuntimeError, match=expected):
            cli.submit_batch(FLOOR_LEG_ID, tmp_path, ONE_WAY, check_only=False)

    def test_a_live_leg_refuses_the_batch_subcommand(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus
    ) -> None:
        del pinned_stimulus
        with pytest.raises(ValueError, match="not batch"):
            cli.submit_batch(LIVE_LEG_ID, tmp_path, ONE_WAY, check_only=True)

    def test_the_opus_legs_wait_on_add_opus_and_the_rest_do_not(self, tmp_path: Path) -> None:
        """Per-sample classifier refusal is stimulus-dependent, so spending on this row is a decision."""
        for leg_id in ("opus-5--default--A--identity-ow", "opus-5--default--A--identity-md"):
            with pytest.raises(RuntimeError, match="add-opus"):
                cli._refuse_unauthorized_leg(leg_for(leg_id), tmp_path)
        cli._refuse_unauthorized_leg(leg_for(LIVE_LEG_ID), tmp_path)
        summary = cli.add_opus(tmp_path, ONE_WAY, "opus")
        assert summary["legs"] == [
            "opus-5--default--A--identity-ow",
            "opus-5--default--A--identity-md",
        ]
        cli._refuse_unauthorized_leg(leg_for("opus-5--default--A--identity-ow"), tmp_path)


class TestBatchCollect:
    def handle(self, tmp_path: Path) -> dict[str, Any]:
        return json.loads(
            cli.handle_path(tmp_path, leg_for(FLOOR_LEG_ID)).read_text(encoding="utf-8")
        )

    def test_a_collect_writes_every_record_once_and_resumes_on_a_second_pass(
        self, tmp_path: Path, aws: tuple[FakeBedrock, FakeS3], pinned_stimulus: TransferStimulus
    ) -> None:
        del pinned_stimulus
        _, s3 = aws
        cli.submit_batch(FLOOR_LEG_ID, tmp_path, ONE_WAY, check_only=False)
        fulfil_job(s3, self.handle(tmp_path))
        first = cli.collect_batch(FLOOR_LEG_ID, tmp_path, ONE_WAY, timeout_seconds=1.0)
        assert first["collected"] == FLOOR_RECORDS
        assert first["resumed"] == 0
        second = cli.collect_batch(FLOOR_LEG_ID, tmp_path, ONE_WAY, timeout_seconds=1.0)
        assert second["collected"] == 0
        assert second["resumed"] == FLOOR_RECORDS
        assert len(load_run_replies(tmp_path)) == FLOOR_RECORDS

    def test_a_collected_run_scans_to_the_records_it_holds(
        self, tmp_path: Path, aws: tuple[FakeBedrock, FakeS3], pinned_stimulus: TransferStimulus
    ) -> None:
        del pinned_stimulus
        _, s3 = aws
        cli.submit_batch(FLOOR_LEG_ID, tmp_path, ONE_WAY, check_only=False)
        fulfil_job(s3, self.handle(tmp_path))
        cli.collect_batch(FLOOR_LEG_ID, tmp_path, ONE_WAY, timeout_seconds=1.0)
        totals = cli.scan(tmp_path)
        assert totals["examined"] == FLOOR_RECORDS
        assert totals["scanned"] == FLOOR_RECORDS
        assert totals["parsed"] == FLOOR_RECORDS
        assert totals["wrong_tag"] == 0
        assert totals["errored"] == 0

    def test_the_submit_summary_records_the_stimulus_prompt_digest(
        self, tmp_path: Path, aws: tuple[FakeBedrock, FakeS3], pinned_stimulus: TransferStimulus
    ) -> None:
        cli.submit_batch(FLOOR_LEG_ID, tmp_path, ONE_WAY, check_only=False)
        summary = json.loads(
            cli.submit_summary_path(tmp_path, leg_for(FLOOR_LEG_ID)).read_text(encoding="utf-8")
        )
        assert summary["stimulus_prompt_digest"] == pinned_stimulus.prompt_digest
        assert summary["stimulus_digest"] == pinned_stimulus.digest

    def test_a_collect_under_a_stimulus_whose_prompt_digest_moved_refuses(
        self, tmp_path: Path, aws: tuple[FakeBedrock, FakeS3], pinned_stimulus: TransferStimulus
    ) -> None:
        """The handle cannot carry the stimulus digest, so the submit summary carries it instead."""
        del pinned_stimulus
        _, s3 = aws
        cli.submit_batch(FLOOR_LEG_ID, tmp_path, ONE_WAY, check_only=False)
        fulfil_job(s3, self.handle(tmp_path))
        path = cli.submit_summary_path(tmp_path, leg_for(FLOOR_LEG_ID))
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["stimulus_prompt_digest"] = "0f0f0f0f0f0f0f0f"
        path.write_text(json.dumps(summary), encoding="utf-8")
        with pytest.raises(RuntimeError, match="stimulus prompt digest") as raised:
            cli.collect_batch(FLOOR_LEG_ID, tmp_path, ONE_WAY, timeout_seconds=1.0)
        assert "0f0f0f0f0f0f0f0f" in str(raised.value)
        assert load_run_replies(tmp_path) == {}

    def test_a_resubmit_under_a_different_stimulus_refuses_rather_than_rewriting_the_digest(
        self, tmp_path: Path, aws: tuple[FakeBedrock, FakeS3], pinned_stimulus: TransferStimulus
    ) -> None:
        """Rewriting it would make the collect-time check compare the new digest with itself."""
        del pinned_stimulus, aws
        cli.submit_batch(FLOOR_LEG_ID, tmp_path, ONE_WAY, check_only=False)
        path = cli.submit_summary_path(tmp_path, leg_for(FLOOR_LEG_ID))
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["stimulus_prompt_digest"] = "0f0f0f0f0f0f0f0f"
        path.write_text(json.dumps(summary), encoding="utf-8")
        with pytest.raises(RuntimeError, match="compare the new digest with itself"):
            cli.submit_batch(FLOOR_LEG_ID, tmp_path, ONE_WAY, check_only=False)

    def test_a_resubmit_that_reconstructs_a_lost_summary_says_the_digest_was_not_recorded(
        self, tmp_path: Path, aws: tuple[FakeBedrock, FakeS3], pinned_stimulus: TransferStimulus
    ) -> None:
        """The documented repair still runs; what it cannot claim is that the digest is the job's own."""
        del pinned_stimulus, aws
        first = cli.submit_batch(FLOOR_LEG_ID, tmp_path, ONE_WAY, check_only=False)
        assert first["stimulus_prompt_digest_recorded_at_submit"] is True
        cli.submit_summary_path(tmp_path, leg_for(FLOOR_LEG_ID)).unlink()
        again = cli.submit_batch(FLOOR_LEG_ID, tmp_path, ONE_WAY, check_only=False)
        assert again["resumed_handle"] is True
        assert again["stimulus_prompt_digest_recorded_at_submit"] is False

    def test_a_collect_with_no_submit_summary_refuses_and_names_the_fix(
        self, tmp_path: Path, aws: tuple[FakeBedrock, FakeS3], pinned_stimulus: TransferStimulus
    ) -> None:
        del pinned_stimulus
        _, s3 = aws
        cli.submit_batch(FLOOR_LEG_ID, tmp_path, ONE_WAY, check_only=False)
        fulfil_job(s3, self.handle(tmp_path))
        cli.submit_summary_path(tmp_path, leg_for(FLOOR_LEG_ID)).unlink()
        with pytest.raises(FileNotFoundError, match="submit-batch"):
            cli.collect_batch(FLOOR_LEG_ID, tmp_path, ONE_WAY, timeout_seconds=1.0)
        assert load_run_replies(tmp_path) == {}

    def test_a_rubric_edit_between_submit_and_collect_still_collects(
        self,
        tmp_path: Path,
        aws: tuple[FakeBedrock, FakeS3],
        pinned_stimulus: TransferStimulus,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _, s3 = aws
        cli.submit_batch(FLOOR_LEG_ID, tmp_path, ONE_WAY, check_only=False)
        fulfil_job(s3, self.handle(tmp_path))
        edited = replace(
            pinned_stimulus,
            digest="0f0f0f0f0f0f0f0f",
            judge_instructions={
                **pinned_stimulus.judge_instructions,
                next(iter(pinned_stimulus.judge_instructions)): "SYNTHETIC-RUBRIC: reply once.",
            },
        )
        monkeypatch.setattr(cli, "load_stimulus", lambda: edited)
        summary = cli.collect_batch(FLOOR_LEG_ID, tmp_path, ONE_WAY, timeout_seconds=1.0)
        assert summary["collected"] == FLOOR_RECORDS
        row = next(iter(load_run_replies(tmp_path).values()))
        assert row["stimulus_digest"] == pinned_stimulus.prompt_digest

    def test_a_re_collect_of_rows_sampled_under_another_stimulus_refuses_by_key(
        self,
        tmp_path: Path,
        aws: tuple[FakeBedrock, FakeS3],
        pinned_stimulus: TransferStimulus,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A key already on disk is skipped only if its record answers THIS prompt under THIS material.

        The live loop has always compared the two stored digests before skipping a key; the batch collect
        skipped on the key alone. The gap needs the whole-file evidence gone -- an operator repair that
        rebuilt the summary -- and then one edit to a prompt-affecting field this leg does not render, so
        the handle's own prompt digest still matches and the rewritten summary agrees with itself. The
        rows on disk are then the only thing that remembers which stimulus they answered.
        """
        _, s3 = aws
        cli.submit_batch(FLOOR_LEG_ID, tmp_path, ONE_WAY, check_only=False)
        fulfil_job(s3, self.handle(tmp_path))
        cli.collect_batch(FLOOR_LEG_ID, tmp_path, ONE_WAY, timeout_seconds=1.0)
        assert len(load_run_replies(tmp_path)) == FLOOR_RECORDS
        elsewhere = replace(pinned_stimulus, prompt_digest="0f0f0f0f0f0f0f0f")
        monkeypatch.setattr(cli, "load_stimulus", lambda: elsewhere)
        path = cli.submit_summary_path(tmp_path, leg_for(FLOOR_LEG_ID))
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["stimulus_prompt_digest"] = elsewhere.prompt_digest
        path.write_text(json.dumps(summary), encoding="utf-8")
        with pytest.raises(ValueError, match="refusing to resume") as raised:
            cli.collect_batch(FLOOR_LEG_ID, tmp_path, ONE_WAY, timeout_seconds=1.0)
        assert "stimulus_digest" in str(raised.value)
        assert len(load_run_replies(tmp_path)) == FLOOR_RECORDS

    def test_a_collect_on_a_rebuilt_summary_refuses_until_the_operator_says_so(
        self, tmp_path: Path, aws: tuple[FakeBedrock, FakeS3], pinned_stimulus: TransferStimulus
    ) -> None:
        """The repair's own flag gets a consumer: a reconstructed digest proves nothing on its own.

        Warning rather than refusing would leave the collect stamping rows with a digest read after the
        fact, and refusing outright would leave a paid job uncollectable, so the escape is an explicit
        statement about the file's history -- which nothing on disk can make in the operator's place.
        """
        del pinned_stimulus
        _, s3 = aws
        cli.submit_batch(FLOOR_LEG_ID, tmp_path, ONE_WAY, check_only=False)
        fulfil_job(s3, self.handle(tmp_path))
        cli.submit_summary_path(tmp_path, leg_for(FLOOR_LEG_ID)).unlink()
        rebuilt = cli.submit_batch(FLOOR_LEG_ID, tmp_path, ONE_WAY, check_only=False)
        assert rebuilt["stimulus_prompt_digest_recorded_at_submit"] is False
        with pytest.raises(RuntimeError, match="stimulus-unchanged-since-submit"):
            cli.collect_batch(FLOOR_LEG_ID, tmp_path, ONE_WAY, timeout_seconds=1.0)
        assert load_run_replies(tmp_path) == {}
        collected = cli.collect_batch(
            FLOOR_LEG_ID,
            tmp_path,
            ONE_WAY,
            timeout_seconds=1.0,
            stimulus_unchanged_since_submit=True,
        )
        assert collected["collected"] == FLOOR_RECORDS
        assert collected["stimulus_unchanged_since_submit_asserted"] is True

    def test_a_summary_written_before_the_flag_existed_warns_and_collects(
        self,
        tmp_path: Path,
        aws: tuple[FakeBedrock, FakeS3],
        pinned_stimulus: TransferStimulus,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """An older run directory carries no flag, which is unknown rather than false.

        Refusing there would be a gate that fires only on history nobody can change: those runs were
        collected under code that had no flag to write. It says so in the log and compares the digest.
        """
        del pinned_stimulus
        _, s3 = aws
        cli.submit_batch(FLOOR_LEG_ID, tmp_path, ONE_WAY, check_only=False)
        fulfil_job(s3, self.handle(tmp_path))
        path = cli.submit_summary_path(tmp_path, leg_for(FLOOR_LEG_ID))
        summary = json.loads(path.read_text(encoding="utf-8"))
        del summary["stimulus_prompt_digest_recorded_at_submit"]
        path.write_text(json.dumps(summary), encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger=cli.__name__):
            collected = cli.collect_batch(FLOOR_LEG_ID, tmp_path, ONE_WAY, timeout_seconds=1.0)
        assert collected["collected"] == FLOOR_RECORDS
        assert "recorded_at_submit" in caplog.text

    @pytest.mark.parametrize(
        ("field", "expected"), [("prompt_digest", "prompt digest"), ("cell_digest", "cell digest")]
    )
    def test_a_digest_mismatch_refuses_the_whole_collect(
        self,
        tmp_path: Path,
        aws: tuple[FakeBedrock, FakeS3],
        pinned_stimulus: TransferStimulus,
        field: str,
        expected: str,
    ) -> None:
        del pinned_stimulus
        _, s3 = aws
        cli.submit_batch(FLOOR_LEG_ID, tmp_path, ONE_WAY, check_only=False)
        path = cli.handle_path(tmp_path, leg_for(FLOOR_LEG_ID))
        fulfil_job(s3, self.handle(tmp_path))
        handle = self.handle(tmp_path)
        handle[field] = "0" * 64
        path.write_text(json.dumps(handle), encoding="utf-8")
        with pytest.raises(RuntimeError, match=expected):
            cli.collect_batch(FLOOR_LEG_ID, tmp_path, ONE_WAY, timeout_seconds=1.0)
        assert load_run_replies(tmp_path) == {}


class TestScriptedSmoke:
    def test_the_offline_smoke_runs_render_sample_scan_validate_and_judge(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus
    ) -> None:
        summary = cli.smoke(tmp_path, ONE_WAY, backend="scripted")
        assert summary["backend"] == "scripted"
        assert summary["calls"] == cli.SMOKE_SCRIPTED_CALLS
        assert summary["ran"] == cli.SMOKE_SCRIPTED_CALLS
        scans = summary["scans"]
        assert isinstance(scans, dict)
        assert scans["scanned"] == cli.SMOKE_SCRIPTED_CALLS
        assert scans["parsed"] == cli.SMOKE_SCRIPTED_CALLS
        assert scans["wrong_tag"] == 0
        judge_counts = summary["judge_counts"]
        assert isinstance(judge_counts, dict)
        assert judge_counts["judged"] == cli.SMOKE_SCRIPTED_CALLS
        validation = summary["judge_validation"]
        assert isinstance(validation, dict)
        assert validation["validated"] == sum(
            len(pinned_stimulus.validation_replies[game_id]) for game_id in JUDGE_ARM_GAME_IDS
        )
        assert summary["measured_chars_per_token"]
        smoke_dir = tmp_path / "smoke" / "scripted"
        assert (smoke_dir / "summary-smoke.json").exists()
        assert (smoke_dir / "scans.jsonl").exists()
        assert (smoke_dir / "judged.jsonl").exists()
        assert not list(tmp_path.glob("replies--*.jsonl")), "smoke records stay under smoke/"

    def test_the_scripted_reply_answers_in_whichever_tag_the_row_asked_for(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus
    ) -> None:
        """A fixed tag would never exercise the inversion, which is where a polarity bug would live."""
        del pinned_stimulus
        cli.smoke(tmp_path, ONE_WAY, backend="scripted")
        rows = load_run_replies(tmp_path / "smoke" / "scripted")
        for row in rows.values():
            assert f"<{row['polarity']}>" in str(row["reply"])

    def test_the_smoke_is_resumable_the_same_way_a_production_leg_is(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus
    ) -> None:
        del pinned_stimulus
        cli.smoke(tmp_path, ONE_WAY, backend="scripted")
        again = cli.smoke(tmp_path, ONE_WAY, backend="scripted")
        assert again["resumed"] == cli.SMOKE_SCRIPTED_CALLS
        assert again["ran"] == 0

    def test_a_reply_that_thinks_in_the_answer_channel_is_counted_and_judged_consistently(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pinned_stimulus: TransferStimulus
    ) -> None:
        """The counting has teeth only if a reply that does carry a block moves the number."""
        del pinned_stimulus
        plain = cli._scripted_reply

        def thinking_reply(prompt: str) -> str:
            return f"<think>weighing it up</think>{plain(prompt)}"

        monkeypatch.setattr(cli, "_scripted_reply", thinking_reply)
        summary = cli.smoke(tmp_path, ONE_WAY, backend="scripted")
        counts = summary["inline_think_by_model"]
        assert isinstance(counts, dict)
        assert counts[ONE_WAY.smoke_live_model_id]["with_inline_think"] == cli.SMOKE_SCRIPTED_CALLS
        assert summary["judged_rows_with_inline_think"] == cli.SMOKE_SCRIPTED_CALLS
        scans = summary["scans"]
        assert isinstance(scans, dict)
        assert scans["parsed"] == cli.SMOKE_SCRIPTED_CALLS

    def test_a_judged_row_that_disagrees_with_its_reply_about_thinking_refuses(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus
    ) -> None:
        """The sabotage of the consistency check: flip one stored flag and the smoke must go red."""
        del pinned_stimulus
        cli.smoke(tmp_path, ONE_WAY, backend="scripted")
        smoke_dir = tmp_path / "smoke" / "scripted"
        judged = smoke_dir / "judged.jsonl"
        rows = [
            json.loads(line) for line in judged.read_text(encoding="utf-8").splitlines() if line
        ]
        rows[0]["had_inline_think"] = True
        judged.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        with pytest.raises(RuntimeError, match="thinking block"):
            cli._refuse_judge_that_read_a_different_text(load_run_replies(smoke_dir), judged)


class TestThePassIsRequiredAndSelectsTheTable:
    """Two passes share this CLI and the first one is finished, so nothing may default to a pass.

    The failure a default would allow is the quiet one: a command run without the flag re-renders,
    re-prices or re-submits a finished pass into its own run directory, and every artifact it writes is
    complete and plausible.
    """

    def test_a_command_with_no_pass_flag_refuses_at_the_parser(self) -> None:
        with pytest.raises(SystemExit):
            cli._parse_args(["build"])
        with pytest.raises(SystemExit):
            cli._parse_args(["submit-batch", "--leg", FLOOR_LEG_ID])

    def test_an_unknown_pass_is_rejected_by_the_parser(self) -> None:
        with pytest.raises(SystemExit):
            cli._parse_args(["--pass", "one-way", "build"])

    @pytest.mark.parametrize("pass_id", [ONE_WAY.pass_id, KNOCKOUT_PLAN.pass_id])
    def test_the_pass_carries_its_own_run_directory_when_none_is_given(self, pass_id: str) -> None:
        """So the two passes' replies cannot land in one directory by an operator's omission."""
        args = cli._parse_args(["--pass", pass_id, "scan"])
        assert args.run_dir is None
        assert args.pass_id == pass_id
        table = plan_table(pass_id)
        assert (
            table.default_run_dir
            != plan_table(
                ONE_WAY.pass_id if pass_id == KNOCKOUT_PLAN.pass_id else KNOCKOUT_PLAN.pass_id
            ).default_run_dir
        )

    def test_a_leg_of_the_other_pass_refuses_after_the_parser_accepts_it(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus
    ) -> None:
        """Argparse offers every pass's leg ids, because it builds its choices before --pass is parsed.

        The table is what refuses: a mistyped pass with a real leg id of the other one would otherwise
        submit that pass's job into this pass's run directory.
        """
        del pinned_stimulus
        args = cli._parse_args(
            ["--pass", KNOCKOUT_PLAN.pass_id, "submit-batch", "--leg", FLOOR_LEG_ID]
        )
        assert args.leg == FLOOR_LEG_ID
        with pytest.raises(ValueError, match="is not a leg of the 'knockout' pass"):
            cli.submit_batch(FLOOR_LEG_ID, tmp_path, KNOCKOUT_PLAN, check_only=True)

    def test_the_knockout_smoke_reads_its_own_model_and_a_selectable_block(self) -> None:
        leg = cli.smoke_leg(cli.TRANSPORT_LIVE, KNOCKOUT_PLAN)
        assert leg.model_id == KNOCKOUT_PLAN.smoke_live_model_id
        assert leg.block == KNOCKOUT_PLAN.smoke_block
        assert leg.batch_prefix == KNOCKOUT_PLAN.batch_prefix
        other = KNOCKOUT_PLAN.reading_blocks[1]
        assert cli.smoke_leg(cli.TRANSPORT_LIVE, KNOCKOUT_PLAN, other).block == other
        assert (
            cli._parse_args(["--pass", KNOCKOUT_PLAN.pass_id, "smoke", "--block", other]).block
            == other
        )

    def test_the_build_and_plan_record_which_pass_they_are(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus
    ) -> None:
        """A plan.json read months later has to say which pass wrote it."""
        del pinned_stimulus
        plan = cli.build(tmp_path, narrowed(KNOCKOUT_PLAN, "luna--default--A--knockout-dd"))
        assert plan["pass_id"] == KNOCKOUT_PLAN.pass_id
        on_disk = json.loads((tmp_path / "plan.json").read_text(encoding="utf-8"))
        assert on_disk["pass_id"] == KNOCKOUT_PLAN.pass_id
        assert [entry["leg_id"] for entry in on_disk["legs"]] == ["luna--default--A--knockout-dd"]
        assert on_disk["legs"][0]["records"] == 256


class TestCliParsing:
    def test_every_leg_id_is_an_accepted_choice(self) -> None:
        for leg_id in cli.LEGS_BY_ID:
            args = cli._parse_args([*PASS_ARGV, "submit-batch", "--leg", leg_id])
            assert args.leg == leg_id

    def test_an_unknown_leg_is_rejected_by_the_parser(self) -> None:
        with pytest.raises(SystemExit):
            cli._parse_args([*PASS_ARGV, "submit-batch", "--leg", "not-a-leg"])

    def test_add_opus_offers_only_the_model_with_an_on_demand_leg(self) -> None:
        assert cli._parse_args([*PASS_ARGV, "add-opus", "--model", "opus"]).model == "opus"
        with pytest.raises(SystemExit):
            cli._parse_args([*PASS_ARGV, "add-opus", "--model", "haiku"])

    def test_the_four_passes_are_the_only_accepted_choices(self) -> None:
        for pass_id in ("one-way-transfer", "knockout", "fingerprint", "correlation-dose"):
            assert cli._parse_args(["--pass", pass_id, "scan"]).pass_id == pass_id
        with pytest.raises(SystemExit):
            cli._parse_args(["--pass", "not-a-pass", "scan"])

    def test_generate_boards_belongs_to_the_pass_that_owns_the_boards(self) -> None:
        """The dose pass reads the same stimulus file and generates nothing in it."""
        args = cli._parse_args(["--pass", "correlation-dose", "generate-boards"])
        with pytest.raises(SystemExit, match="belongs to the"):
            cli._dispatch(args, Path("artifacts/unused"), plan_table("correlation-dose"))

    def test_build_takes_the_empty_boards_escape_and_defaults_to_refusing_one(self) -> None:
        assert cli._parse_args([*PASS_ARGV, "build"]).allow_empty_boards is False
        assert (
            cli._parse_args([*PASS_ARGV, "build", "--allow-empty-boards"]).allow_empty_boards
            is True
        )

    def test_the_smoke_backend_choices_are_scripted_and_live(self) -> None:
        assert cli._parse_args([*PASS_ARGV, "smoke"]).backend == "scripted"
        assert cli._parse_args([*PASS_ARGV, "smoke", "--backend", "live"]).backend == "live"

    @pytest.mark.parametrize("command", ["judge", "cross-judge"])
    def test_both_judge_subcommands_take_a_concurrency_of_at_least_one(self, command: str) -> None:
        assert cli._parse_args([*PASS_ARGV, command]).concurrency == cli.DEFAULT_JUDGE_CONCURRENCY
        assert cli._parse_args([*PASS_ARGV, command, "--concurrency", "48"]).concurrency == 48
        for rejected in ("0", "-4", "many"):
            with pytest.raises(SystemExit):
                cli._parse_args([*PASS_ARGV, command, "--concurrency", rejected])

    def test_the_cross_judge_default_is_forty_rows(self) -> None:
        assert (
            cli._parse_args([*PASS_ARGV, "cross-judge"]).n == judge_module.CROSS_JUDGE_RECORDS == 40
        )

    def test_the_cross_judge_game_filter_is_unset_by_default_and_offers_only_the_two_arms(
        self,
    ) -> None:
        assert cli._parse_args([*PASS_ARGV, "cross-judge"]).game is None
        for game_id in (ONE_WAY_TRANSFER_GAME_ID, MATCHED_DECISION_TRANSFER_GAME_ID):
            assert cli._parse_args([*PASS_ARGV, "cross-judge", "--game", game_id]).game == game_id
        with pytest.raises(SystemExit):
            cli._parse_args([*PASS_ARGV, "cross-judge", "--game", "not-an-arm"])

    def test_the_intent_check_scope_defaults_to_the_endpoints_and_offers_only_the_two(self) -> None:
        assert cli._parse_args([*PASS_ARGV, "intent-check"]).scope == cli.SCOPE_ENDPOINT
        assert (
            cli._parse_args([*PASS_ARGV, "intent-check", "--scope", cli.SCOPE_ALL]).scope
            == cli.SCOPE_ALL
        )
        with pytest.raises(SystemExit):
            cli._parse_args([*PASS_ARGV, "intent-check", "--scope", "everything"])

    def test_the_intent_check_model_filter_is_unset_by_default(self) -> None:
        assert cli._parse_args([*PASS_ARGV, "intent-check"]).model is None
        assert cli._parse_args(
            [*PASS_ARGV, "intent-check", "--model", cli.SONNET_MODEL_ID]
        ).model == (cli.SONNET_MODEL_ID)

    def test_the_intent_check_takes_the_shared_judge_knobs(self) -> None:
        args = cli._parse_args(
            [*PASS_ARGV, "intent-check", "--concurrency", "24", "--skip-validation-gate"]
        )
        assert args.concurrency == 24
        assert args.skip_validation_gate is True
        for rejected in ("0", "-4", "many"):
            with pytest.raises(SystemExit):
                cli._parse_args([*PASS_ARGV, "intent-check", "--concurrency", rejected])

    def test_the_judge_limit_is_unset_by_default_and_at_least_one_when_given(self) -> None:
        assert cli._parse_args([*PASS_ARGV, "judge"]).limit is None
        assert cli._parse_args([*PASS_ARGV, "judge", "--limit", "5"]).limit == 5
        for rejected in ("0", "-1", "some"):
            with pytest.raises(SystemExit):
                cli._parse_args([*PASS_ARGV, "judge", "--limit", rejected])


def assert_no_design_label(prompt: str) -> None:
    """Fail if any string that names the design appears in one prompt, case-insensitively."""
    lowered = prompt.lower()
    for label in DESIGN_LABELS:
        assert label.lower() not in lowered, label


def scripted_judge(stimulus: TransferStimulus) -> ScriptedDetailedBackend:
    """One offline judge for every path: it agrees with both validation sets and abstains elsewhere."""
    by_name = {
        case.name: case
        for game_id in judge_module.VERDICT_SCHEMA_BY_GAME
        for case in judge_module.validation_cases(stimulus, game_id)
    }
    names = sorted(by_name, key=lambda name: -len(name))

    def answer(prompt: str) -> str:
        name = next((name for name in names if name in prompt), None)
        if name is None:
            return cli._scripted_judge_verdict(prompt)
        return json.dumps({**by_name[name].expected, "evidence": "SYNTHETIC-EVIDENCE"})

    return ScriptedDetailedBackend(answer)


class RecordingJudgeFactory:
    """Stands in for the CLI's judge-backend seam and keeps every request it was asked to build."""

    def __init__(self, backend: ScriptedDetailedBackend) -> None:
        self.backend = backend
        self.calls: list[tuple[str, str | None, int]] = []

    def __call__(self, model_id: str, effort: str | None, *, concurrency: int) -> DetailedBackend:
        self.calls.append((model_id, effort, concurrency))
        return self.backend


@pytest.fixture
def judge_factory(
    monkeypatch: pytest.MonkeyPatch, transfer_stimulus: TransferStimulus
) -> RecordingJudgeFactory:
    """Route every judge-backend construction through a recording factory over one scripted judge."""
    factory = RecordingJudgeFactory(scripted_judge(transfer_stimulus))
    monkeypatch.setattr(cli, "_judge_backend", factory)
    return factory


def seed_replies(
    run_dir: Path,
    stimulus: TransferStimulus,
    count: int = 4,
    leg_ids: tuple[str, ...] = (LIVE_LEG_ID,),
) -> None:
    """Write a few real reply records per leg into a run dir, through the production live path."""
    for leg_id in leg_ids:
        leg = leg_for(leg_id)
        calls = cli.planned_calls_for_leg(leg, stimulus)[:count]
        cli._run_live_calls(
            calls,
            cli.reply_path(run_dir, leg),
            stimulus,
            concurrency=1,
            chunk_size=count,
            backend_factory=scripted_factory("<set>4</set>", "<keep>16</keep>"),
        )


BOTH_ARMS_THREE_MODELS_LEG_IDS = (
    "luna--default--A--identity-ow",
    "luna--default--A--identity-md",
    "haiku-4-5--default--A--identity-ow",
    "haiku-4-5--default--A--identity-md",
    "sonnet-5--default--A--identity-md",
    "sonnet-5--default--A--identity-ow",
)
"""Three models, each with one leg per arm: the smallest roster on which a per-model, per-arm draw shows."""


class TestTheJudgeWaitsForItsValidation:
    """An uncalibrated judge produces a headline rate and looks perfectly healthy doing it.

    So the gate is on the validation SUMMARY for the CURRENT rubric digests of BOTH arms with nothing
    outstanding, rather than on the operator remembering the order of the subcommands.
    """

    def test_judging_before_any_validation_refuses_and_names_the_fix(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus, judge_factory: object
    ) -> None:
        del judge_factory
        seed_replies(tmp_path, pinned_stimulus)
        with pytest.raises(RuntimeError, match="judge-validate"):
            cli.judge(tmp_path)
        assert not (tmp_path / "judged.jsonl").exists()

    def test_cross_judging_before_any_validation_refuses(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus, judge_factory: object
    ) -> None:
        del judge_factory
        seed_replies(tmp_path, pinned_stimulus)
        cli.scan(tmp_path)
        with pytest.raises(RuntimeError, match="judge-validate"):
            cli.cross_judge(tmp_path, n=2)
        assert not (tmp_path / "cross-judged.jsonl").exists()

    def test_a_clean_validation_opens_the_gate_and_the_summary_says_it_was_not_skipped(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus, judge_factory: object
    ) -> None:
        del judge_factory
        seed_replies(tmp_path, pinned_stimulus)
        report = cli.judge_validate(tmp_path)
        assert report["misses"] == []
        summary = json.loads((tmp_path / cli.JUDGE_VALIDATION_SUMMARY).read_text(encoding="utf-8"))
        assert summary["judge_prompt_digests"] == cli.rubric_digests(pinned_stimulus)
        counts = cli.judge(tmp_path)
        assert counts["judged"] == 4
        judge_summary = json.loads((tmp_path / "summary-judge.json").read_text(encoding="utf-8"))
        assert judge_summary["skipped_validation_gate"] is False

    def test_a_validation_with_misses_keeps_the_gate_shut(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus, judge_factory: object
    ) -> None:
        del judge_factory
        seed_replies(tmp_path, pinned_stimulus)
        cli.judge_validate(tmp_path)
        path = tmp_path / cli.JUDGE_VALIDATION_SUMMARY
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["misses"] = [{"name": "v-solo-0", "field": "they_are_me"}]
        path.write_text(json.dumps(summary), encoding="utf-8")
        with pytest.raises(RuntimeError, match="field misses"):
            cli.judge(tmp_path)

    def test_a_validation_of_one_arms_rubric_does_not_clear_the_other(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus, judge_factory: object
    ) -> None:
        """The twin's rubric carries the pass's headline field, so its validation is not optional."""
        del judge_factory
        seed_replies(tmp_path, pinned_stimulus)
        cli.judge_validate(tmp_path)
        path = tmp_path / cli.JUDGE_VALIDATION_SUMMARY
        summary = json.loads(path.read_text(encoding="utf-8"))
        digests = dict(summary["judge_prompt_digests"])
        digests["matched-decision-transfer"] = "0f0f0f0f0f0f0f0f"
        summary["judge_prompt_digests"] = digests
        path.write_text(json.dumps(summary), encoding="utf-8")
        with pytest.raises(RuntimeError, match="not this run's"):
            cli.judge(tmp_path)

    def test_a_validation_made_before_a_code_side_prompt_edit_does_not_clear_the_run(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus, judge_factory: object
    ) -> None:
        """The rubric digests cover the authored text alone; the scaffold lives here in code."""
        del judge_factory
        seed_replies(tmp_path, pinned_stimulus)
        cli.judge_validate(tmp_path)
        path = tmp_path / cli.JUDGE_VALIDATION_SUMMARY
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["judge_prompt_shape_digest"] = "0f0f0f0f0f0f0f0f"
        path.write_text(json.dumps(summary), encoding="utf-8")
        with pytest.raises(RuntimeError, match="prompt-shape digest"):
            cli.judge(tmp_path)

    def test_a_validation_under_an_earlier_prompt_version_does_not_clear_the_run(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus, judge_factory: object
    ) -> None:
        """The per-row resume compares the version, so every row would be re-judged unvalidated."""
        del judge_factory
        seed_replies(tmp_path, pinned_stimulus)
        cli.judge_validate(tmp_path)
        path = tmp_path / cli.JUDGE_VALIDATION_SUMMARY
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["judge_prompt_version"] = "one-way-transfer-judge-v1"
        path.write_text(json.dumps(summary), encoding="utf-8")
        with pytest.raises(RuntimeError, match="prompt version"):
            cli.judge(tmp_path)

    def test_pooling_replies_from_two_stimuli_refuses_before_any_judge_call(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus, judge_factory: object
    ) -> None:
        """A mid-run frame edit lands two stimuli in one directory with no shared key to refuse on."""
        del judge_factory
        seed_replies(tmp_path, pinned_stimulus)
        cli.judge_validate(tmp_path)
        path = next(tmp_path.glob("replies--*.jsonl"))
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[0]["stimulus_digest"] = "0f0f0f0f0f0f0f0f"
        path.write_text("".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf-8")
        with pytest.raises(ValueError, match="different stimulus prompt digests"):
            cli.judge(tmp_path)

    def test_the_skip_flag_runs_anyway_and_stamps_the_summary(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus, judge_factory: object
    ) -> None:
        del judge_factory
        seed_replies(tmp_path, pinned_stimulus)
        cli.judge(tmp_path, skip_validation_gate=True)
        judge_summary = json.loads((tmp_path / "summary-judge.json").read_text(encoding="utf-8"))
        assert judge_summary["skipped_validation_gate"] is True

    def test_no_design_label_reaches_the_backend_on_the_cross_judge_path(
        self,
        tmp_path: Path,
        pinned_stimulus: TransferStimulus,
        judge_factory: RecordingJudgeFactory,
    ) -> None:
        seed_replies(tmp_path, pinned_stimulus)
        cli.scan(tmp_path)
        cli.judge_validate(tmp_path)
        counts = cli.cross_judge(tmp_path, n=2)
        assert counts["judged"] == 2
        assert judge_factory.backend.prompts_seen
        for prompt in judge_factory.backend.prompts_seen:
            assert_no_design_label(prompt)

    def test_the_judge_concurrency_reaches_the_factory_and_the_summary(
        self,
        tmp_path: Path,
        pinned_stimulus: TransferStimulus,
        judge_factory: RecordingJudgeFactory,
    ) -> None:
        seed_replies(tmp_path, pinned_stimulus)
        cli.scan(tmp_path)
        cli.judge_validate(tmp_path)
        assert (
            cli.main([*PASS_ARGV, "--run-dir", str(tmp_path), "judge", "--concurrency", "48"]) == 0
        )
        assert judge_factory.calls[-1] == (
            judge_module.JUDGE_MODEL_ID,
            judge_module.JUDGE_REASONING_EFFORT,
            48,
        )
        summary = json.loads((tmp_path / "summary-judge.json").read_text(encoding="utf-8"))
        assert summary["judge_concurrency"] == 48


def cross_judged_rows(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "cross-judged.jsonl"
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def cross_judge_summary(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "summary-cross-judge.json").read_text(encoding="utf-8"))


class TestTheCrossJudgeGameFilter:
    """The twin arm is a quarter of the run and carries the field the two judges agree on least.

    A draw over both arms therefore hands each model a handful of twin rows, too few to read the
    twin-only field per model. ``--game`` narrows the candidates to one arm BEFORE the draw, so the same
    request size lands on that arm alone while the per-model round robin still holds.
    """

    def seed_both_arms(self, run_dir: Path, stimulus: TransferStimulus) -> None:
        seed_replies(run_dir, stimulus, count=4, leg_ids=BOTH_ARMS_THREE_MODELS_LEG_IDS)
        cli.scan(run_dir)
        cli.judge_validate(run_dir)

    def test_a_twin_only_draw_selects_only_twin_rows_and_still_covers_every_model_evenly(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus, judge_factory: object
    ) -> None:
        del judge_factory
        self.seed_both_arms(tmp_path, pinned_stimulus)
        argv = [
            *PASS_ARGV,
            "--run-dir",
            str(tmp_path),
            "cross-judge",
            "--n",
            "6",
            "--game",
            MATCHED_DECISION_TRANSFER_GAME_ID,
        ]
        assert cli.main(argv) == 0
        rows = cross_judged_rows(tmp_path)
        assert len(rows) == 6
        assert {row["game_id"] for row in rows} == {MATCHED_DECISION_TRANSFER_GAME_ID}
        assert Counter(row["model_id"] for row in rows) == {
            cli.LUNA_MODEL_ID: 2,
            cli.SONNET_MODEL_ID: 2,
            leg_for("haiku-4-5--default--A--identity-md").model_id: 2,
        }
        summary = cross_judge_summary(tmp_path)
        assert summary["game_filter"] == MATCHED_DECISION_TRANSFER_GAME_ID
        assert summary["requested"] == 6
        assert summary["selected"] == 6
        assert summary["models_covered"] == 3

    def test_without_the_flag_both_arms_are_drawn_and_the_summary_records_no_filter(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus, judge_factory: object
    ) -> None:
        del judge_factory
        self.seed_both_arms(tmp_path, pinned_stimulus)
        counts = cli.cross_judge(tmp_path, n=12)
        assert counts["judged"] == 12
        rows = cross_judged_rows(tmp_path)
        assert {row["game_id"] for row in rows} == {
            ONE_WAY_TRANSFER_GAME_ID,
            MATCHED_DECISION_TRANSFER_GAME_ID,
        }
        assert len({row["model_id"] for row in rows}) == 3
        assert cross_judge_summary(tmp_path)["game_filter"] is None

    def test_a_filtered_draw_resumes_by_key_on_top_of_an_unfiltered_one(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus, judge_factory: object
    ) -> None:
        """The two draws share one file: the twin rows already judged are skipped, the rest appended."""
        del judge_factory
        self.seed_both_arms(tmp_path, pinned_stimulus)
        cli.cross_judge(tmp_path, n=6)
        before = {row["key"] for row in cross_judged_rows(tmp_path)}
        twin_before = {
            row["key"]
            for row in cross_judged_rows(tmp_path)
            if row["game_id"] == MATCHED_DECISION_TRANSFER_GAME_ID
        }
        counts = cli.cross_judge(tmp_path, n=6, game=MATCHED_DECISION_TRANSFER_GAME_ID)
        twin_counts = counts["by_game"][MATCHED_DECISION_TRANSFER_GAME_ID]
        assert twin_counts["records"] == 6
        assert twin_counts["already_judged"] == len(twin_before)
        assert counts["by_game"][ONE_WAY_TRANSFER_GAME_ID]["records"] == 0
        assert counts["judged"] == 6 - len(twin_before)
        after = {row["key"] for row in cross_judged_rows(tmp_path)}
        assert before <= after
        assert len(after) == len(before) + counts["judged"]

    def test_a_filter_that_matches_no_reply_refuses_rather_than_judging_nothing(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus, judge_factory: object
    ) -> None:
        del judge_factory
        seed_replies(tmp_path, pinned_stimulus)
        cli.scan(tmp_path)
        cli.judge_validate(tmp_path)
        with pytest.raises(ValueError, match=MATCHED_DECISION_TRANSFER_GAME_ID):
            cli.cross_judge(tmp_path, n=2, game=MATCHED_DECISION_TRANSFER_GAME_ID)
        assert not (tmp_path / "cross-judged.jsonl").exists()
        assert not (tmp_path / "summary-cross-judge.json").exists()

    def test_an_unknown_game_is_refused_before_anything_is_read(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus, judge_factory: object
    ) -> None:
        del pinned_stimulus, judge_factory
        with pytest.raises(ValueError, match="--game must be one of"):
            cli.cross_judge(tmp_path, n=2, game="not-an-arm")


def tracked_probe_dir() -> Path:
    """A directory inside the repository that git tracks; nothing may ever be created under it.

    Anchored on the real repository root rather than the caller's cwd: a relative path would resolve
    against wherever pytest was launched from, and the guard under test would be answering a different
    question than the one these tests name.
    """
    root = _repo_root()
    assert root is not None, "these tests describe behaviour inside a git repository"
    return root / "sociology" / "transfer-writer-guard-probe"


WRITER_CASES: tuple[tuple[str, Callable[[Path], object]], ...] = (
    ("append_replies", lambda base: append_replies(base / "replies--probe.jsonl", [{"key": "k"}])),
    ("write_summary", lambda base: write_summary(base, "probe", {"command": "probe"})),
    (
        "append_judged",
        lambda base: judge_module.append_judged(base / "judged.jsonl", [{"key": "k"}]),
    ),
    (
        "append_cross_judged",
        lambda base: judge_module.append_judged(base / "cross-judged.jsonl", [{"key": "k"}]),
    ),
    (
        "append_intent_checked",
        lambda base: judge_module.append_judged(base / "intent-checked.jsonl", [{"key": "k"}]),
    ),
    ("scans", scans_module.scan_run),
    ("plan_json", lambda base: cli.build(base, ONE_WAY)),
    ("add_opus", lambda base: cli.add_opus(base, ONE_WAY, "opus")),
    (
        "submit_handle",
        lambda base: cli.submit_batch(FLOOR_LEG_ID, base, ONE_WAY, check_only=True),
    ),
)


class TestEveryWriterRefusesATrackedDestination:
    """One case per writer, because the leak of this class this repo had was an unguarded second one.

    Each writer carries verbatim replies, the judge's quoted evidence, the authored frames inside a
    digest, or the batch job's account-identifying values, and this remote is public. The refusal has to
    land BEFORE anything is created, which is why every case asserts the empty directory as well: a guard
    that fires after the write, or after a mkdir, has already published the path.
    """

    @pytest.mark.parametrize(("name", "writer"), WRITER_CASES, ids=[n for n, _ in WRITER_CASES])
    def test_it_refuses_before_creating_anything(
        self, name: str, writer: Callable[[Path], object], pinned_stimulus: TransferStimulus
    ) -> None:
        del pinned_stimulus, name
        base = tracked_probe_dir()
        leaked = (
            f"{base} exists before this test ran. Something wrote there, which is what these tests "
            f"exist to prevent -- most likely a deliberate sabotage run of one of these guards. "
            f"Delete the directory and re-run."
        )
        assert not base.exists(), leaked
        with pytest.raises(ValueError, match="not under a gitignored root"):
            writer(base)
        assert not base.exists(), f"{base} was created despite the refusal"

    def test_a_gitignored_destination_is_accepted(self, tmp_path: Path) -> None:
        """The other half of the check: the guard must not refuse a legitimate run dir."""
        write_summary(tmp_path, "probe", {"command": "probe"})
        assert (tmp_path / "summary-probe.json").exists()


INTENT_ENDPOINT_REPLIES = ("<set>20</set>", "<keep>20</keep>", "<set>7</set>")
"""What the scripted sampler answers when a test needs both endpoints AND an interior row.

The two ends are what the endpoint scope selects and the only figures a slip can move across the whole
range; the interior row is what the two scopes differ by, so a test that seeded only endpoints could not
tell one scope from the other.
"""


def intent_checked_rows(run_dir: Path) -> list[dict[str, Any]]:
    path = cli.intent_checked_path(run_dir)
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def intent_summary(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "summary-intent-check.json").read_text(encoding="utf-8"))


def scripted_intent_reader(stimulus: TransferStimulus) -> ScriptedDetailedBackend:
    """One offline reader for every intent path: it agrees with the authored cases and slips elsewhere.

    Away from the validation set it answers the way the roster row that motivated this pass answers: a
    reply whose tag reads as the whole stock is reported as having concluded to hold everything back,
    which is a slip, while a reply whose tag reads as nothing is reported as agreeing with its tag. That
    asymmetry is the real defect's shape, and it means the per-cell counts a test reads are not constants.
    """
    by_name = {case.name: case for case in judge_module.intent_validation_cases(stimulus)}
    names = sorted(by_name, key=lambda name: -len(name))

    def answer(prompt: str) -> str:
        name = next((name for name in names if name in prompt), None)
        if name is not None:
            return json.dumps({**by_name[name].expected, "evidence": "SYNTHETIC-EVIDENCE"})
        slipped = "<set>20</set>" in prompt
        return json.dumps(
            {
                "concluded_action": judge_module.CONCLUDED_GIVE_NONE
                if slipped
                else judge_module.CONCLUDED_GIVE_SOME,
                "concluded_units": 0 if slipped else 7,
                "assumes_return": not slipped,
                "evidence": "SYNTHETIC-EVIDENCE",
            }
        )

    return ScriptedDetailedBackend(answer)


class CombinedReaderFactory:
    """One recording factory over two scripted readers, dispatched on which rubric built the prompt.

    Both instruments are constructed through the same seam, so a test that exercises the judge and the
    intent check in one run dir needs one factory that can answer both. The intent prompt is the only one
    that states the rules the reply was written under, so that header is the discriminator -- read off the
    prompt rather than off the call, because the CLI builds its backend before it knows a single record.
    """

    def __init__(self, judge: ScriptedDetailedBackend, intent: ScriptedDetailedBackend) -> None:
        self.judge = judge
        self.intent = intent
        self.calls: list[tuple[str, str | None, int]] = []
        self.backend = ScriptedDetailedBackend(self._answer)

    def _answer(self, prompt: str) -> str:
        target = self.intent if judge_module.INTENT_RULES_HEADER in prompt else self.judge
        return target.generate([prompt])[0]

    def __call__(self, model_id: str, effort: str | None, *, concurrency: int) -> DetailedBackend:
        self.calls.append((model_id, effort, concurrency))
        return self.backend


@pytest.fixture
def intent_factory(
    monkeypatch: pytest.MonkeyPatch, transfer_stimulus: TransferStimulus
) -> CombinedReaderFactory:
    """Route every backend construction through one recording factory that answers both rubrics."""
    factory = CombinedReaderFactory(
        scripted_judge(transfer_stimulus), scripted_intent_reader(transfer_stimulus)
    )
    monkeypatch.setattr(cli, "_judge_backend", factory)
    return factory


def seed_for_intent(
    run_dir: Path,
    stimulus: TransferStimulus,
    count: int = 6,
    leg_ids: tuple[str, ...] = (LIVE_LEG_ID,),
) -> None:
    """Sample a few replies at both ends of the range plus an interior one, scan them, and validate."""
    for leg_id in leg_ids:
        leg = leg_for(leg_id)
        calls = cli.planned_calls_for_leg(leg, stimulus)[:count]
        cli._run_live_calls(
            calls,
            cli.reply_path(run_dir, leg),
            stimulus,
            concurrency=1,
            chunk_size=count,
            backend_factory=scripted_factory(*INTENT_ENDPOINT_REPLIES),
        )
    cli.scan(run_dir)
    cli.intent_check_validate(run_dir)


class TestTheIntentCheckWaitsForItsValidation:
    """This instrument's verdicts CORRECT figures, so an unvalidated reader is worse here than elsewhere.

    A reader with ``concluded_action`` backwards on the authored slip would move real answers to the other
    end of the range, and every table built on it would still look healthy.
    """

    def test_reading_before_any_validation_refuses_and_names_the_fix(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus, intent_factory: object
    ) -> None:
        del intent_factory
        seed_replies(tmp_path, pinned_stimulus)
        cli.scan(tmp_path)
        with pytest.raises(RuntimeError, match="intent-check-validate"):
            cli.intent_check(tmp_path)
        assert not cli.intent_checked_path(tmp_path).exists()

    def test_a_clean_validation_opens_the_gate_and_the_summary_says_it_was_not_skipped(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus, intent_factory: object
    ) -> None:
        del intent_factory
        seed_for_intent(tmp_path, pinned_stimulus)
        report = json.loads((tmp_path / cli.INTENT_VALIDATION_SUMMARY).read_text(encoding="utf-8"))
        assert report["misses"] == []
        assert report["intent_prompt_digest"] == judge_module.intent_digest(pinned_stimulus)
        cli.intent_check(tmp_path)
        assert intent_summary(tmp_path)["skipped_validation_gate"] is False

    def test_a_validation_with_misses_keeps_the_gate_shut(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus, intent_factory: object
    ) -> None:
        del intent_factory
        seed_for_intent(tmp_path, pinned_stimulus)
        path = tmp_path / cli.INTENT_VALIDATION_SUMMARY
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["misses"] = [{"name": "v-intent-slipped-tag", "field": "concluded_action"}]
        path.write_text(json.dumps(summary), encoding="utf-8")
        with pytest.raises(RuntimeError, match="field misses"):
            cli.intent_check(tmp_path)

    def test_a_validation_of_an_earlier_wording_does_not_clear_this_one(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus, intent_factory: object
    ) -> None:
        del intent_factory
        seed_for_intent(tmp_path, pinned_stimulus)
        path = tmp_path / cli.INTENT_VALIDATION_SUMMARY
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["intent_prompt_digest"] = "0f0f0f0f0f0f0f0f"
        path.write_text(json.dumps(summary), encoding="utf-8")
        with pytest.raises(RuntimeError, match="not this run's"):
            cli.intent_check(tmp_path)

    def test_a_validation_under_an_earlier_prompt_version_does_not_clear_this_one(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus, intent_factory: object
    ) -> None:
        """The version is what the per-row resume compares, and the gate ignored it entirely.

        So a bumped version used to re-read every production row under the new prompt while the gate
        still accepted a validation made under the old one -- the two halves of one decision disagreeing.
        """
        del intent_factory
        seed_for_intent(tmp_path, pinned_stimulus)
        path = tmp_path / cli.INTENT_VALIDATION_SUMMARY
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["intent_prompt_version"] = "one-way-transfer-intent-check-v0"
        path.write_text(json.dumps(summary), encoding="utf-8")
        with pytest.raises(RuntimeError, match="validated prompt version"):
            cli.intent_check(tmp_path)

    def test_a_validation_under_an_earlier_code_side_prompt_does_not_clear_this_one(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus, intent_factory: object
    ) -> None:
        """The rubric digest covers the authored text only; the rules paragraphs live in code.

        Before the shape digest, editing a rules paragraph (or a rung clause, or a channel heading) left
        the gate happy with a validation of the previous prompt -- and this instrument's verdicts rewrite
        published figures.
        """
        del intent_factory
        seed_for_intent(tmp_path, pinned_stimulus)
        path = tmp_path / cli.INTENT_VALIDATION_SUMMARY
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["intent_prompt_shape_digest"] = "0f0f0f0f0f0f0f0f"
        path.write_text(json.dumps(summary), encoding="utf-8")
        with pytest.raises(RuntimeError, match="prompt-shape digest"):
            cli.intent_check(tmp_path)

    def test_the_validation_summary_records_all_three_keys_the_gate_compares(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus, intent_factory: object
    ) -> None:
        del intent_factory
        seed_for_intent(tmp_path, pinned_stimulus)
        summary = json.loads((tmp_path / cli.INTENT_VALIDATION_SUMMARY).read_text(encoding="utf-8"))
        assert summary["intent_prompt_digest"] == judge_module.intent_digest(pinned_stimulus)
        assert summary["intent_prompt_version"] == judge_module.INTENT_CHECK_PROMPT_VERSION
        assert summary["intent_prompt_shape_digest"] == judge_module.intent_prompt_shape_digest()

    def test_the_skip_flag_reads_anyway_and_stamps_the_summary(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus, intent_factory: object
    ) -> None:
        del intent_factory
        seed_replies(tmp_path, pinned_stimulus, count=6)
        cli.scan(tmp_path)
        cli.intent_check(tmp_path, scope=cli.SCOPE_ALL, skip_validation_gate=True)
        assert intent_summary(tmp_path)["skipped_validation_gate"] is True

    def test_it_refuses_before_any_call_when_the_run_has_no_scans(
        self,
        tmp_path: Path,
        pinned_stimulus: TransferStimulus,
        intent_factory: CombinedReaderFactory,
    ) -> None:
        """A slip is defined against the parsed figure, so with no scans every count would read zero."""
        seed_replies(tmp_path, pinned_stimulus)
        cli.intent_check_validate(tmp_path)
        prompts_before = len(intent_factory.backend.prompts_seen)
        with pytest.raises(FileNotFoundError, match="scan"):
            cli.intent_check(tmp_path)
        assert len(intent_factory.backend.prompts_seen) == prompts_before


class TestTheIntentCheckScopeAndFilter:
    def test_the_endpoint_scope_reads_the_two_ends_and_leaves_the_interior_alone(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus, intent_factory: object
    ) -> None:
        del intent_factory
        seed_for_intent(tmp_path, pinned_stimulus)
        summary = cli.intent_check(tmp_path)
        figures = cli.scan_figures_by_key(tmp_path)
        read = {row["key"] for row in intent_checked_rows(tmp_path)}
        by_position = {
            position: {
                key for key, figure in figures.items() if cli.figure_position(figure) == position
            }
            for position in cli.FIGURE_POSITIONS
        }
        endpoints = by_position[cli.POSITION_ENDPOINT]
        assert read == endpoints
        assert by_position[cli.POSITION_INTERIOR], (
            "the fixture must sample an interior row for the two scopes to differ"
        )
        assert by_position[cli.POSITION_UNPARSED], (
            "the fixture must sample a reply that parsed to nothing, which is the second way the "
            "endpoint filter used to say `not an endpoint` for a different reason"
        )
        invocation = summary["invocation"]
        assert invocation["scope"] == cli.SCOPE_ENDPOINT
        assert invocation["selected"] == len(endpoints)
        assert invocation["replies_on_disk"] == len(figures)
        positions = invocation["figure_positions"]
        assert positions == {position: len(keys) for position, keys in by_position.items()}
        assert positions[cli.POSITION_UNSCANNED] == 0
        assert sum(positions.values()) == invocation["after_model_filter"]

    def test_the_all_scope_reads_the_interior_too_and_appends_to_the_same_file(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus, intent_factory: object
    ) -> None:
        del intent_factory
        seed_for_intent(tmp_path, pinned_stimulus)
        cli.intent_check(tmp_path)
        endpoint_rows = {row["key"] for row in intent_checked_rows(tmp_path)}
        summary = cli.intent_check(tmp_path, scope=cli.SCOPE_ALL)
        after = {row["key"] for row in intent_checked_rows(tmp_path)}
        assert endpoint_rows < after
        assert after == set(cli.scan_figures_by_key(tmp_path))
        assert summary["invocation"]["already_judged"] == len(endpoint_rows)
        assert summary["invocation"]["judged"] == len(after) - len(endpoint_rows)

    def test_a_second_pass_resumes_by_key_and_reads_nothing_again(
        self,
        tmp_path: Path,
        pinned_stimulus: TransferStimulus,
        intent_factory: CombinedReaderFactory,
    ) -> None:
        del intent_factory
        seed_for_intent(tmp_path, pinned_stimulus)
        first = cli.intent_check(tmp_path)
        rows_before = len(intent_checked_rows(tmp_path))
        second = cli.intent_check(tmp_path)
        assert second["invocation"]["already_judged"] == first["invocation"]["judged"]
        assert second["invocation"]["judged"] == 0
        assert len(intent_checked_rows(tmp_path)) == rows_before

    def test_the_model_filter_reads_one_roster_row_and_records_it(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus, intent_factory: object
    ) -> None:
        del intent_factory
        seed_for_intent(
            tmp_path,
            pinned_stimulus,
            leg_ids=("luna--default--A--identity-ow", "sonnet-5--default--A--identity-ow"),
        )
        summary = cli.intent_check(tmp_path, model=cli.SONNET_MODEL_ID)
        assert {row["model_id"] for row in intent_checked_rows(tmp_path)} == {cli.SONNET_MODEL_ID}
        assert summary["invocation"]["model_filter"] == cli.SONNET_MODEL_ID
        assert (
            summary["invocation"]["after_model_filter"] < summary["invocation"]["replies_on_disk"]
        )

    def test_a_model_with_no_replies_refuses_rather_than_reading_nothing(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus, intent_factory: object
    ) -> None:
        del intent_factory
        seed_for_intent(tmp_path, pinned_stimulus)
        with pytest.raises(ValueError, match="no reply"):
            cli.intent_check(tmp_path, model="a.model-that-never-ran")
        assert not cli.intent_checked_path(tmp_path).exists()
        assert not (tmp_path / "summary-intent-check.json").exists()

    def test_a_reply_the_scans_file_has_no_row_for_is_counted_rather_than_read_as_interior(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus, intent_factory: object
    ) -> None:
        """The third way the endpoint filter used to answer "no": no scan row at all.

        A scans file written before the last leg landed narrowed the pass with no refusal and no count,
        so a summary could report a complete endpoint pass over a fraction of the replies.
        """
        del intent_factory
        seed_for_intent(tmp_path, pinned_stimulus)
        scans_path = tmp_path / scans_module.SCANS_FILENAME
        kept = [
            line for line in scans_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        dropped = json.loads(kept[-1])["key"]
        scans_path.write_text("\n".join(kept[:-1]) + "\n", encoding="utf-8")
        summary = cli.intent_check(tmp_path)
        positions = summary["invocation"]["figure_positions"]
        assert positions[cli.POSITION_UNSCANNED] == 1
        assert dropped not in {row["key"] for row in intent_checked_rows(tmp_path)}

    def test_a_scans_file_covering_none_of_the_replies_refuses_rather_than_reading_zero_slips(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus, intent_factory: object
    ) -> None:
        """With no figure to compare against, every cell would report zero slips over a full denominator."""
        del intent_factory
        seed_for_intent(tmp_path, pinned_stimulus)
        scans_path = tmp_path / scans_module.SCANS_FILENAME
        rows = [
            {**json.loads(line), "key": f"{json.loads(line)['key']}|not-a-reply"}
            for line in scans_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        scans_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        with pytest.raises(ValueError, match="has a row in"):
            cli.intent_check(tmp_path, scope=cli.SCOPE_ALL)

    def test_an_unknown_scope_refuses_before_anything_is_read(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus, intent_factory: object
    ) -> None:
        del intent_factory
        seed_for_intent(tmp_path, pinned_stimulus)
        with pytest.raises(ValueError, match="--scope must be one of"):
            cli.intent_check(tmp_path, scope="everything")


class TestTheIntentCheckSummary:
    """The per (model, game, polarity) counts, which are what a correction is applied from."""

    def test_the_slip_count_finds_the_planted_slips_and_nothing_else(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus, intent_factory: object
    ) -> None:
        del intent_factory
        seed_for_intent(tmp_path, pinned_stimulus)
        summary = cli.intent_check(tmp_path)
        figures = cli.scan_figures_by_key(tmp_path)
        rows = {row["key"]: row for row in intent_checked_rows(tmp_path)}
        planted = {
            key
            for key, row in rows.items()
            if row["verdict"]["concluded_action"] == judge_module.CONCLUDED_GIVE_NONE
            and figures[key][0] == figures[key][1]
        }
        assert planted, (
            "the scripted reader must slip on at least one row for this to read anything"
        )
        assert summary["over_the_whole_file"]["totals"]["slip_count"] == len(planted)
        by_cell = summary["over_the_whole_file"]["by_cell"]
        for key in planted:
            cell = "|".join(
                (str(rows[key]["model_id"]), str(rows[key]["game_id"]), str(rows[key]["polarity"]))
            )
            assert by_cell[cell]["slip_count"] >= 1
        assert (
            sum(cell["checked"] for cell in by_cell.values()) == summary["invocation"]["selected"]
        )

    def test_every_count_is_recomputed_over_the_whole_file_rather_than_this_pass(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus, intent_factory: object
    ) -> None:
        """A narrowed pass must not publish a summary that reads as though it were the run."""
        del intent_factory
        seed_for_intent(
            tmp_path,
            pinned_stimulus,
            leg_ids=("luna--default--A--identity-ow", "sonnet-5--default--A--identity-ow"),
        )
        cli.intent_check(tmp_path)
        whole = intent_summary(tmp_path)["over_the_whole_file"]["totals"]["checked"]
        narrowed = cli.intent_check(tmp_path, model=cli.SONNET_MODEL_ID)
        assert narrowed["invocation"]["judged"] == 0
        assert narrowed["over_the_whole_file"]["totals"]["checked"] == whole
        assert narrowed["invocation"]["selected"] < whole

    def test_the_assumes_return_count_and_the_quiet_buckets_are_reported_per_cell(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus, intent_factory: object
    ) -> None:
        del intent_factory
        seed_for_intent(tmp_path, pinned_stimulus)
        summary = cli.intent_check(tmp_path, scope=cli.SCOPE_ALL)
        rows = intent_checked_rows(tmp_path)
        expected = sum(1 for row in rows if row["verdict"]["assumes_return"])
        whole_file = summary["over_the_whole_file"]
        assert whole_file["totals"]["assumes_return_count"] == expected
        assert whole_file["rows"] == len(rows)
        assert set(next(iter(whole_file["by_cell"].values()))) == set(
            judge_module.INTENT_COUNT_NAMES
        )
        assert set(whole_file["totals"]) == set(judge_module.INTENT_COUNT_NAMES)

    def test_the_reader_and_its_concurrency_reach_the_factory_and_the_summary(
        self,
        tmp_path: Path,
        pinned_stimulus: TransferStimulus,
        intent_factory: CombinedReaderFactory,
    ) -> None:
        seed_for_intent(tmp_path, pinned_stimulus)
        assert (
            cli.main(
                [*PASS_ARGV, "--run-dir", str(tmp_path), "intent-check", "--concurrency", "24"]
            )
            == 0
        )
        assert intent_factory.calls[-1] == (
            judge_module.INTENT_CHECK_MODEL_ID,
            judge_module.INTENT_CHECK_REASONING_EFFORT,
            24,
        )
        assert intent_summary(tmp_path)["judge_concurrency"] == 24


GOOD_BOARD_MESSAGE = (
    "SYNTHETIC-DRAWN-MESSAGE: rinse it under the warm tap until the water runs clear, then stand it "
    "on end and let it drip dry before it goes back."
)
"""What a drawn board message looks like when it clears every board gate."""

MESSAGE_NAMING_A_LAB = (
    "SYNTHETIC-DRAWN-MESSAGE: as Claude, I would rinse it under the warm tap until the water runs "
    "clear, and then let it drip dry before refitting."
)
"""A draw the board gate refuses: the reader must INFER sameness, never be told it."""


def board_stimulus_file(tmp_path: Path, *, with_boards: bool = False) -> Path:
    """Write a synthetic v3 stimulus file, with the board slots empty unless asked otherwise."""
    payload = (
        synthetic_transfer_payload() if with_boards else synthetic_transfer_payload_without_boards()
    )
    path = tmp_path / "one_way_transfer_stimulus.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class ScriptedDraws:
    """A live backend factory whose replies are decided by how many calls have been made.

    A counter rather than a fixed script, because the generator draws in ROUNDS over whichever slots are
    still pending: the number of calls in a round is the number of messages the board gate refused, which
    is the thing the re-draw tests are about.
    """

    def __init__(self, *, bad_calls: int = 0, always_bad: bool = False) -> None:
        self.bad_calls = bad_calls
        self.always_bad = always_bad
        self.calls = 0

    def reply(self, prompt: str) -> str:
        del prompt
        self.calls += 1
        if self.always_bad or self.calls <= self.bad_calls:
            return MESSAGE_NAMING_A_LAB
        return GOOD_BOARD_MESSAGE

    def factory(self, model_id: str, effort: str | None, concurrency: int) -> DetailedBackend:
        del model_id, effort, concurrency
        return ScriptedDetailedBackend(self.reply)


class TestBuildWithoutBoards:
    """Nothing may be sampled before the boards are generated, and the escape hatch does not change it."""

    def test_the_dose_pass_builds_before_the_boards_exist(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pass E reads no board, so it must not wait on pass D's generation to be planned or priced."""
        path = board_stimulus_file(tmp_path)
        monkeypatch.setattr(
            cli,
            "load_stimulus",
            lambda _path=None, *, allow_empty_boards=False: load_stimulus(
                path, allow_empty_boards=allow_empty_boards
            ),
        )
        plan = cli.build(
            tmp_path / "run",
            narrowed(CORRELATION_DOSE_PLAN, "luna--default--B--dose-floor"),
            allow_empty_boards=True,
        )
        assert plan["allow_empty_boards"] is True
        assert plan["total_records"] == 256

    def test_the_fingerprint_pass_refuses_to_build_and_names_the_generator(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The escape hatch loads the file; it does not render a board of nothing."""
        path = board_stimulus_file(tmp_path)
        monkeypatch.setattr(
            cli,
            "load_stimulus",
            lambda _path=None, *, allow_empty_boards=False: load_stimulus(
                path, allow_empty_boards=allow_empty_boards
            ),
        )
        with pytest.raises(ValueError, match="generate-boards"):
            cli.build(
                tmp_path / "run",
                narrowed(FINGERPRINT_PLAN, "luna--default--A--fingerprint-md"),
                allow_empty_boards=True,
            )

    def test_the_dose_plan_records_the_two_payoff_thresholds_and_the_others_do_not(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus
    ) -> None:
        """The readout marks numbers this code computed from the dose, never numbers somebody typed."""
        del pinned_stimulus
        dose = cli.build(
            tmp_path / "dose", narrowed(CORRELATION_DOSE_PLAN, "luna--default--B--dose-floor")
        )
        thresholds = dose["matched_rounds_thresholds"]["thresholds"]
        assert thresholds["match_else_independent"]["between_rungs"] == [1, 2]
        assert thresholds["binary_mismatch"]["between_rungs"] == [5, 7]
        assert dose["record_rungs_planned"] == sorted(
            {record_rung_id(matched) for matched in MATCHED_ROUNDS_LADDER}
            | {RUNG_SAME_CHECKPOINT_RECORD_0}
        )
        one_way = cli.build(tmp_path / "ow", narrowed(ONE_WAY, FLOOR_LEG_ID))
        assert one_way["matched_rounds_thresholds"] is None
        assert one_way["record_rungs_planned"] == []


class TestTheBoardCondition:
    def test_every_board_reply_row_carries_what_the_board_is_to_its_reader(
        self, tmp_path: Path, pinned_stimulus: TransferStimulus
    ) -> None:
        """The 2x2 decomposition groups by this label, so it rides on the row rather than being re-derived."""
        del tmp_path
        leg = FINGERPRINT_PLAN.leg_for("luna--default--A--fingerprint-md")
        calls = planned_calls_for_leg(leg, pinned_stimulus)
        rows = [
            cli.reply_record(
                call, ScriptedDetailedBackend(["x"]).generate_detailed(["p"])[0], pinned_stimulus
            )
            for call in calls[::128]
        ]
        assert rows
        for row, call in zip(rows, calls[::128], strict=True):
            assert row["board_condition"] == call.board_condition
        assert {row["board_condition"] for row in rows} >= {"own-raw", "other-raw"}


class TestGenerateBoards:
    """The boards are drawn by the roster models themselves, gated draw by draw, and then frozen.

    Every test here runs the whole path offline: the live probe, both raw boards, both crossed boards, the
    re-draw loop, the write, and the re-read of the written file through the FULL loader -- which is the
    check that a generation cannot leave behind a file nobody can sample.
    """

    def generate(
        self, tmp_path: Path, draws: ScriptedDraws, *, regenerate: bool = False, **kwargs: Any
    ) -> dict[str, Any]:
        path = kwargs.pop("stimulus_path", None) or board_stimulus_file(tmp_path)
        return cli.generate_boards(
            tmp_path / "run",
            regenerate=regenerate,
            stimulus_path=path,
            run_roots=(tmp_path / "no-runs",),
            live_backend_factory=draws.factory,
            **kwargs,
        )

    def test_it_draws_every_board_gates_them_and_freezes_a_file_that_loads(
        self, tmp_path: Path
    ) -> None:
        draws = ScriptedDraws()
        path = board_stimulus_file(tmp_path)
        summary = self.generate(tmp_path, draws, stimulus_path=path)
        assert summary["live_probe"]["live"] is True
        assert summary["messages"] == len(BOARD_IDS) * N_SCENARIOS * BOARD_MESSAGE_COUNT
        assert summary["redraws"] == 0
        frozen = load_stimulus(path)
        assert summary["stimulus_prompt_digest"] == frozen.prompt_digest
        for scenario in frozen.scenarios:
            for board_id in BOARD_IDS:
                board = frozen.boards[scenario.scenario_id][board_id]
                assert len(board.messages) == BOARD_MESSAGE_COUNT
                assert all(entry["transport"] == "live" for entry in board.provenance)
                assert all(entry["redraws"] == 0 for entry in board.provenance)

    def test_a_crossed_board_records_the_digest_of_the_message_it_rewrote(
        self, tmp_path: Path
    ) -> None:
        """Which is what makes the crossed corner one side's content rather than three fresh draws."""
        path = board_stimulus_file(tmp_path)
        self.generate(tmp_path, ScriptedDraws(), stimulus_path=path)
        frozen = load_stimulus(path)
        scenario_id = frozen.scenarios[0].scenario_id
        for crossed, source_id in BOARD_SOURCE_BOARD.items():
            source = frozen.boards[scenario_id][source_id]
            digests = {message_digest(message) for message in source.messages}
            for entry in frozen.boards[scenario_id][crossed].provenance:
                assert entry[BOARD_SOURCE_DIGEST_FIELD] in digests

    def test_a_refused_draw_is_re_drawn_and_counted(self, tmp_path: Path) -> None:
        """The sabotage: a backend that names a lab. The generator re-draws and records how many times."""
        path = board_stimulus_file(tmp_path)
        # The probe's three calls come first, then the first raw board's whole round of 24.
        draws = ScriptedDraws(bad_calls=3 + N_SCENARIOS * BOARD_MESSAGE_COUNT)
        summary = self.generate(tmp_path, draws, stimulus_path=path)
        assert summary["redraws"] == N_SCENARIOS * BOARD_MESSAGE_COUNT
        frozen = load_stimulus(path)
        first_board = frozen.boards[frozen.scenarios[0].scenario_id][BOARD_IDS[0]]
        assert all(entry["redraws"] == 1 for entry in first_board.provenance)

    def test_a_backend_that_never_clears_the_gate_refuses_by_name_at_the_cap(
        self, tmp_path: Path
    ) -> None:
        """Past the cap it stops rather than spending forever, and says which messages are stuck."""
        with pytest.raises(RuntimeError, match="still fail the board gate after"):
            self.generate(tmp_path, ScriptedDraws(always_bad=True))

    def test_it_refuses_to_overwrite_boards_that_exist_without_regenerate(
        self, tmp_path: Path
    ) -> None:
        path = board_stimulus_file(tmp_path, with_boards=True)
        with pytest.raises(RuntimeError, match="already carry messages"):
            self.generate(tmp_path, ScriptedDraws(), stimulus_path=path)

    def test_regenerate_is_refused_once_either_pass_has_sampled_a_reply(
        self, tmp_path: Path
    ) -> None:
        """A second generation is a new stimulus version, so it may not land in a sampled run directory."""
        path = board_stimulus_file(tmp_path, with_boards=True)
        run_root = tmp_path / "runs"
        (run_root / "prod-20260904").mkdir(parents=True)
        (run_root / "prod-20260904" / "replies--luna.jsonl").write_text("{}\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="already sampled replies"):
            cli.generate_boards(
                tmp_path / "run",
                regenerate=True,
                stimulus_path=path,
                run_roots=(run_root,),
                live_backend_factory=ScriptedDraws().factory,
            )

    def test_regenerate_is_refused_when_the_generations_own_run_dir_holds_replies(
        self, tmp_path: Path
    ) -> None:
        """The sweep covers the directory this invocation was pointed at, not only the design's roots.

        Spec 1.8 puts both passes' run directories under two roots, and the refusal used to look only
        there: a leg sampled into a directory of an operator's own was invisible, so the regeneration went
        through and left replies to two different boards pooled under one cell id.
        """
        path = board_stimulus_file(tmp_path, with_boards=True)
        run_dir = tmp_path / "elsewhere" / "prod-20260904"
        run_dir.mkdir(parents=True)
        (run_dir / "replies--luna.jsonl").write_text("{}\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="already sampled replies"):
            cli.generate_boards(
                run_dir,
                regenerate=True,
                stimulus_path=path,
                run_roots=(tmp_path / "no-runs",),
                live_backend_factory=ScriptedDraws().factory,
            )

    def test_regenerate_replaces_the_boards_when_nothing_has_been_sampled(
        self, tmp_path: Path
    ) -> None:
        path = board_stimulus_file(tmp_path, with_boards=True)
        before = load_stimulus(path).boards
        summary = self.generate(tmp_path, ScriptedDraws(), regenerate=True, stimulus_path=path)
        after = load_stimulus(path).boards
        assert summary["regenerated"] is True
        scenario_id = next(iter(before))
        assert (
            before[scenario_id][BOARD_IDS[0]].messages != after[scenario_id][BOARD_IDS[0]].messages
        )

    def test_it_refuses_to_write_a_tracked_stimulus_file(self, tmp_path: Path) -> None:
        """The boards ARE authored stimulus; a tracked destination is a privacy failure, not a typo."""
        del tmp_path
        with pytest.raises(ValueError, match="refusing to write a trace"):
            cli.generate_boards(
                Path("artifacts/unused"),
                regenerate=False,
                stimulus_path=Path("README.md"),
                run_roots=(),
                live_backend_factory=ScriptedDraws().factory,
            )


class FakeBoardBatchHandle:
    """The little a board-generation batch handle has to be: a name, a count, and a saved copy."""

    def __init__(self, records: int) -> None:
        self.records = records
        self.job_name = "SYNTHETIC-BOARD-JOB"
        self.job_arn = f"{JOB_ARN}-boards"
        self.submitted_at = "2026-09-04T00:00:00+00:00"

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"job_name": self.job_name, "records": self.records}) + "\n")


class FakeBoardBatchBackend:
    """A batch backend that answers every padded record with one scripted message."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.submitted: list[list[str]] = []
        self.usage = SimpleNamespace(input_tokens=0, output_tokens=0)

    def submit(self, prompts: list[str], metadata: Any = None) -> FakeBoardBatchHandle:
        del metadata
        self.submitted.append(list(prompts))
        return FakeBoardBatchHandle(len(prompts))

    def collect(self, handle: FakeBoardBatchHandle, *, timeout_seconds: float) -> list[Any]:
        del timeout_seconds
        return ScriptedDetailedBackend([self.reply]).generate_detailed(["p"] * handle.records)


class TestGenerateBoardsOnTheBatchFallback:
    """If the batch row does not answer live, the same material comes back through padded batch jobs.

    Which transport a board was drawn on lands in its provenance, because the two are not
    interchangeable: a batch draw and a live draw are the same sampler on the same prompt, and the record
    is what lets a later reader say so rather than assume it.
    """

    def test_a_refusing_probe_sends_that_model_s_boards_to_batch_and_records_it(
        self, tmp_path: Path
    ) -> None:
        path = board_stimulus_file(tmp_path)
        live = ScriptedDraws()

        def refusing_probe(model_id: str, effort: str | None, concurrency: int) -> DetailedBackend:
            del effort, concurrency
            # The probe asks the BATCH row; a row that answers nothing is the case this test is about.
            if model_id not in cli.LIVE_MODEL_IDS:
                return ScriptedDetailedBackend([""])
            return ScriptedDetailedBackend(live.reply)

        backends: list[FakeBoardBatchBackend] = []

        def batch_factory(model_id: str, run_id: str) -> Any:
            del model_id, run_id
            backend = FakeBoardBatchBackend(GOOD_BOARD_MESSAGE)
            backends.append(backend)
            return backend

        summary = cli.generate_boards(
            tmp_path / "run",
            regenerate=False,
            stimulus_path=path,
            run_roots=(tmp_path / "no-runs",),
            live_backend_factory=refusing_probe,
            batch_backend_factory=batch_factory,
        )
        assert summary["live_probe"]["live"] is False
        assert summary["live_probe"]["answered"] == 0
        frozen = load_stimulus(path)
        transports = {
            board_id: {
                entry["transport"]
                for scenario in frozen.scenarios
                for entry in frozen.boards[scenario.scenario_id][board_id].provenance
            }
            for board_id in BOARD_IDS
        }
        assert transports["board-qwen-raw"] == {"batch"}
        assert transports["board-luna-in-qwen-words"] == {"batch"}
        assert transports["board-luna-raw"] == {"live"}
        assert transports["board-qwen-in-luna-words"] == {"live"}
        assert len(backends) == 2
        for backend in backends:
            padded = backend.submitted[0]
            assert len(padded) >= MIN_BATCH_RECORDS
            assert len(padded) >= (BOARD_REDRAW_CAP + 1) * N_SCENARIOS * BOARD_MESSAGE_COUNT
