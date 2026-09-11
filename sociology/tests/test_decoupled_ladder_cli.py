"""The ladder CLI end to end offline: plan writing, live resume, batch submit/collect, the smoke.

Nothing here touches AWS. The batch path runs against an in-memory S3 and a scripted control plane,
which is what makes the five-way handle check testable at all: each of its five mismatches is planted
into a saved handle deliberately and required to refuse, because every one of them would otherwise
file real responses under the wrong cells with every count looking plausible.

Three of the classes here are refusals rather than behaviours, and each covers a failure that leaves
every artifact looking healthy: a resume that continued a file written under a different stimulus, a
judge that ran before anything checked its rubric, and a writer pointed at a git-tracked path on a
public remote. The writer sweep is parametrized over every writer this pass has, because the one leak
of that class this repository has actually had was a second call site nobody had guarded.
"""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from reward_hacking.bedrock_batch import (
    BATCH_BUCKET_ENV,
    BATCH_BUCKET_OWNER_ENV,
    BATCH_PROFILE_ENV,
    BATCH_ROLE_ARN_ENV,
)
from reward_hacking.trace import _repo_root
from sociology import decoupled_judge as judge_module
from sociology import decoupled_ladder as cli
from sociology import decoupled_scans as scans_module
from sociology.decoupled_plan import DESIGN_LABELS, leg_for
from sociology.decoupled_scans import load_run_replies
from sociology.model_stub import ScriptedDetailedBackend
from sociology.records import append_replies, write_summary

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from reward_hacking.model_backend import DetailedBackend
    from sociology.decoupled_plan import PlannedCall
    from sociology.decoupled_stimulus import DecoupledStimulus

FLOOR_LEG_ID = "gpt-oss-20b--default--B--floor"
FLOOR_RECORDS = 768

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

    The real account, role, bucket and profile are deliberately absent from this repository, so a
    test that wants a constructed backend supplies its own -- and these are obviously artificial
    rather than realistic, because a realistic-looking ARN in a tracked test is the same leak.
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
    monkeypatch: pytest.MonkeyPatch, decoupled_stimulus: DecoupledStimulus
) -> DecoupledStimulus:
    """Point the CLI's loader at the synthetic stimulus rather than the gitignored real file."""
    monkeypatch.setattr(cli, "load_stimulus", lambda: decoupled_stimulus)
    return decoupled_stimulus


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
    def two_legs(self) -> tuple[object, ...]:
        return (leg_for(FLOOR_LEG_ID), leg_for("glm-4-7--default--B--floor"))

    def test_build_writes_the_plan_and_its_summary_marker_last(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pinned_stimulus: DecoupledStimulus
    ) -> None:
        monkeypatch.setattr(cli, "all_legs", self.two_legs)
        plan = cli.build(tmp_path)
        assert plan["total_records"] == 2 * FLOOR_RECORDS
        assert plan["audited_prompts"] == 2 * FLOOR_RECORDS
        assert plan["stimulus_digest"] == pinned_stimulus.digest
        on_disk = json.loads((tmp_path / "plan.json").read_text(encoding="utf-8"))
        assert [entry["leg_id"] for entry in on_disk["legs"]] == [
            FLOOR_LEG_ID,
            "glm-4-7--default--B--floor",
        ]
        assert all(entry["prompt_digest"] and entry["cell_digest"] for entry in on_disk["legs"])
        summary = json.loads((tmp_path / "summary-build.json").read_text(encoding="utf-8"))
        assert summary["finished_at"]
        assert "git_sha" in summary

    def test_the_plan_never_stores_prompt_text(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pinned_stimulus: DecoupledStimulus
    ) -> None:
        del pinned_stimulus
        monkeypatch.setattr(cli, "all_legs", self.two_legs)
        cli.build(tmp_path)
        text = (tmp_path / "plan.json").read_text(encoding="utf-8")
        assert "About the other side" not in text
        assert "SYNTHETIC-COUNTERPART" not in text

    def test_dry_run_prices_batch_off_the_verified_roster(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pinned_stimulus: DecoupledStimulus
    ) -> None:
        del pinned_stimulus
        monkeypatch.setattr(cli, "all_legs", self.two_legs)
        report = cli.dry_run(tmp_path)
        legs = report["legs"]
        assert isinstance(legs, list)
        assert all(entry["price_note"] == "verified batch price" for entry in legs)
        assert all(entry["price_verified"] is True for entry in legs)
        assert report["legs_priced_unverified"] == []
        assert report["estimated_total_usd_production_legs"] > 0
        assert (
            report["estimated_total_usd_all_legs"] == report["estimated_total_usd_production_legs"]
        )
        assumptions = report["assumptions"]
        assert isinstance(assumptions, dict)
        assert assumptions["output_tokens_per_call"] == cli.ASSUMED_OUTPUT_TOKENS_PER_CALL
        live_prices = assumptions["live_prices_per_mtok"]
        assert isinstance(live_prices, dict)
        assert set(live_prices) == set(cli.LIVE_PRICES) == {cli.LUNA_MODEL_ID, cli.SONNET_MODEL_ID}
        assert live_prices[cli.LUNA_MODEL_ID]["verified"] is True
        assert live_prices[cli.SONNET_MODEL_ID]["verified"] is False

    def test_dry_run_names_every_leg_priced_off_an_unverified_figure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pinned_stimulus: DecoupledStimulus
    ) -> None:
        """Sonnet 5's live price is a list price nobody has read back, and the report must say so per leg."""
        del pinned_stimulus
        monkeypatch.setattr(
            cli,
            "all_legs",
            lambda: (
                leg_for("luna--default--B--floor"),
                leg_for("sonnet-5--default--B--floor"),
                leg_for("sonnet-5--default--A--ladder"),
            ),
        )
        report = cli.dry_run(tmp_path)
        by_id = {entry["leg_id"]: entry for entry in report["legs"]}
        assert by_id["luna--default--B--floor"]["price_verified"] is True
        assert by_id["luna--default--B--floor"]["price_note"].startswith("live price, verified")
        assert by_id["sonnet-5--default--B--floor"]["price_verified"] is False
        assert "UNVERIFIED" in by_id["sonnet-5--default--B--floor"]["price_note"]
        assert by_id["sonnet-5--default--B--floor"]["price_per_mtok"] == [3.0, 15.0]
        assert report["legs_priced_unverified"] == [
            "sonnet-5--default--B--floor",
            "sonnet-5--default--A--ladder",
        ]
        production = report["estimated_total_usd_production_legs"]
        assert production == round(
            by_id["luna--default--B--floor"]["estimated_cost_usd"]
            + by_id["sonnet-5--default--B--floor"]["estimated_cost_usd"],
            2,
        )
        assert report["estimated_total_usd_all_legs"] > production


class TestLiveResume:
    def calls(self, stimulus: DecoupledStimulus, count: int) -> list[PlannedCall]:
        planned = cli.planned_calls_for_leg(leg_for("luna--default--A--anchor"), stimulus)
        return planned[:count]

    def test_a_second_sitting_of_the_loop_resumes_what_the_first_finished(
        self, tmp_path: Path, pinned_stimulus: DecoupledStimulus
    ) -> None:
        out = tmp_path / "replies--luna--effort-default--sitting-A--anchor.jsonl"
        first = cli._run_live_calls(
            self.calls(pinned_stimulus, 2),
            out,
            pinned_stimulus,
            concurrency=1,
            chunk_size=1,
            backend_factory=scripted_factory("<action>SHORT</action>"),
        )
        assert first == {"planned": 2, "resumed": 0, "ran": 2, "incomplete": 0}
        second = cli._run_live_calls(
            self.calls(pinned_stimulus, 5),
            out,
            pinned_stimulus,
            concurrency=1,
            chunk_size=2,
            backend_factory=scripted_factory("<action>SHORT</action>"),
        )
        assert second == {"planned": 5, "resumed": 2, "ran": 3, "incomplete": 0}
        assert len(load_run_replies(tmp_path)) == 5

    def test_an_incomplete_record_is_flagged_kept_and_counted(
        self, tmp_path: Path, pinned_stimulus: DecoupledStimulus
    ) -> None:
        out = tmp_path / "replies--luna--effort-default--sitting-A--anchor.jsonl"

        def factory(model_id: str, effort: str | None, concurrency: int) -> DetailedBackend:
            del model_id, effort, concurrency
            return ScriptedDetailedBackend(
                ["partial text"], stop_reason="call_failed:ReadTimeoutError"
            )

        counts = cli._run_live_calls(
            self.calls(pinned_stimulus, 1),
            out,
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
        self, tmp_path: Path, pinned_stimulus: DecoupledStimulus
    ) -> None:
        out = tmp_path / "replies--luna--effort-default--sitting-A--anchor.jsonl"
        cli._run_live_calls(
            self.calls(pinned_stimulus, 1),
            out,
            pinned_stimulus,
            concurrency=1,
            chunk_size=1,
            backend_factory=scripted_factory("<action>SHORT</action>"),
        )
        line = out.read_text(encoding="utf-8")
        out.write_text(line + line, encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate reply key"):
            load_run_replies(tmp_path)

    def test_the_reply_record_carries_every_planned_label_and_the_sampler_stamp(
        self, tmp_path: Path, pinned_stimulus: DecoupledStimulus
    ) -> None:
        out = tmp_path / "replies--luna--effort-default--sitting-A--anchor.jsonl"
        cli._run_live_calls(
            self.calls(pinned_stimulus, 1),
            out,
            pinned_stimulus,
            concurrency=1,
            chunk_size=1,
            backend_factory=scripted_factory("<action>SHORT</action>"),
        )
        row = next(iter(load_run_replies(tmp_path).values()))
        for field in (
            "block",
            "cell",
            "game_id",
            "prompt_id",
            "reskin_id",
            "payoff_variant",
            "label_print_order",
            "coop_label",
            "label_a",
            "label_b",
            "coop_label_index",
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
        assert row["stimulus_digest"] == pinned_stimulus.digest
        assert "prompt" not in row, "the prompt text is re-derivable and must not be stored"


class TestBatchSubmit:
    def test_a_submit_writes_the_handle_and_creates_exactly_one_job(
        self, tmp_path: Path, aws: tuple[FakeBedrock, FakeS3], pinned_stimulus: DecoupledStimulus
    ) -> None:
        del pinned_stimulus
        bedrock, _ = aws
        summary = cli.submit_batch(FLOOR_LEG_ID, tmp_path, check_only=False)
        assert summary["records"] == FLOOR_RECORDS
        assert len(bedrock.created) == 1
        assert cli.handle_path(tmp_path, leg_for(FLOOR_LEG_ID)).exists()

    def test_check_only_spends_nothing_and_creates_no_job(
        self, tmp_path: Path, aws: tuple[FakeBedrock, FakeS3], pinned_stimulus: DecoupledStimulus
    ) -> None:
        del pinned_stimulus
        bedrock, s3 = aws
        summary = cli.submit_batch(FLOOR_LEG_ID, tmp_path, check_only=True)
        assert summary["check_only"] is True
        assert bedrock.created == []
        assert s3.objects == {}
        assert not cli.handle_path(tmp_path, leg_for(FLOOR_LEG_ID)).exists()

    def test_a_second_submit_reuses_the_saved_handle_and_bills_nothing_new(
        self, tmp_path: Path, aws: tuple[FakeBedrock, FakeS3], pinned_stimulus: DecoupledStimulus
    ) -> None:
        del pinned_stimulus
        bedrock, _ = aws
        first = cli.submit_batch(FLOOR_LEG_ID, tmp_path, check_only=False)
        second = cli.submit_batch(FLOOR_LEG_ID, tmp_path, check_only=False)
        assert second["job_arn"] == first["job_arn"]
        assert len(bedrock.created) == 1

    @pytest.mark.parametrize(
        ("field", "value", "expected"),
        [
            ("model_id", "deepseek.v3.2", "model"),
            ("record_count", 383, "records on the handle"),
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
        pinned_stimulus: DecoupledStimulus,
        field: str,
        value: object,
        expected: str,
    ) -> None:
        del pinned_stimulus
        cli.submit_batch(FLOOR_LEG_ID, tmp_path, check_only=False)
        path = cli.handle_path(tmp_path, leg_for(FLOOR_LEG_ID))
        handle = json.loads(path.read_text(encoding="utf-8"))
        handle[field] = value
        path.write_text(json.dumps(handle), encoding="utf-8")
        with pytest.raises(RuntimeError, match=expected):
            cli.submit_batch(FLOOR_LEG_ID, tmp_path, check_only=False)

    def test_a_live_leg_refuses_the_batch_subcommand(
        self, tmp_path: Path, pinned_stimulus: DecoupledStimulus
    ) -> None:
        del pinned_stimulus
        with pytest.raises(ValueError, match="not batch"):
            cli.submit_batch("luna--default--A--anchor", tmp_path, check_only=True)

    def test_an_unauthorized_on_demand_ladder_refuses_until_add_ladder_records_it(
        self, tmp_path: Path, aws: tuple[FakeBedrock, FakeS3], pinned_stimulus: DecoupledStimulus
    ) -> None:
        del pinned_stimulus, aws
        with pytest.raises(RuntimeError, match="add-ladder"):
            cli.submit_batch("opus-5--default--A--ladder", tmp_path, check_only=True)
        summary = cli.add_ladder(tmp_path, "opus")
        assert summary["legs"] == [
            "opus-5--default--A--anchor",
            "opus-5--default--B--floor",
            "opus-5--default--A--ladder",
        ]
        cli.submit_batch("opus-5--default--A--ladder", tmp_path, check_only=True)

    def test_opus_anchor_and_floor_are_on_demand_too_since_the_row_is_not_read(
        self, tmp_path: Path, aws: tuple[FakeBedrock, FakeS3], pinned_stimulus: DecoupledStimulus
    ) -> None:
        """Re-submitting the refused row is a recorded decision, not something a default run does."""
        del pinned_stimulus, aws
        for leg_id in ("opus-5--default--A--anchor", "opus-5--default--B--floor"):
            with pytest.raises(RuntimeError, match="add-ladder"):
                cli.submit_batch(leg_id, tmp_path, check_only=True)
        cli.submit_batch("haiku-4-5--default--A--anchor", tmp_path, check_only=True)

    def test_the_sonnet_ladder_waits_on_add_ladder_but_its_anchor_and_floor_do_not(
        self, tmp_path: Path
    ) -> None:
        """The live path's authorization gate, checked without a backend: it runs before any call."""
        cli._refuse_unauthorized_leg(leg_for("sonnet-5--default--A--anchor"), tmp_path)
        cli._refuse_unauthorized_leg(leg_for("sonnet-5--default--B--floor"), tmp_path)
        with pytest.raises(RuntimeError, match="add-ladder --model sonnet"):
            cli._refuse_unauthorized_leg(leg_for("sonnet-5--default--A--ladder"), tmp_path)
        summary = cli.add_ladder(tmp_path, "sonnet")
        assert summary["legs"] == ["sonnet-5--default--A--ladder"]
        cli._refuse_unauthorized_leg(leg_for("sonnet-5--default--A--ladder"), tmp_path)

    def test_run_live_refuses_a_batch_leg(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not live"):
            cli.run_live("haiku-4-5--default--A--anchor", tmp_path, concurrency=1, chunk_size=1)


class TestBatchCollect:
    def test_a_collect_writes_every_record_once_and_resumes_on_a_second_pass(
        self, tmp_path: Path, aws: tuple[FakeBedrock, FakeS3], pinned_stimulus: DecoupledStimulus
    ) -> None:
        del pinned_stimulus
        _, s3 = aws
        cli.submit_batch(FLOOR_LEG_ID, tmp_path, check_only=False)
        handle = json.loads(
            cli.handle_path(tmp_path, leg_for(FLOOR_LEG_ID)).read_text(encoding="utf-8")
        )
        fulfil_job(s3, handle)
        first = cli.collect_batch(FLOOR_LEG_ID, tmp_path, timeout_seconds=1.0)
        assert first["collected"] == FLOOR_RECORDS
        assert first["resumed"] == 0
        second = cli.collect_batch(FLOOR_LEG_ID, tmp_path, timeout_seconds=1.0)
        assert second["collected"] == 0
        assert second["resumed"] == FLOOR_RECORDS
        assert len(load_run_replies(tmp_path)) == FLOOR_RECORDS

    def test_a_collected_run_scans_to_the_records_it_holds(
        self, tmp_path: Path, aws: tuple[FakeBedrock, FakeS3], pinned_stimulus: DecoupledStimulus
    ) -> None:
        del pinned_stimulus
        _, s3 = aws
        cli.submit_batch(FLOOR_LEG_ID, tmp_path, check_only=False)
        handle = json.loads(
            cli.handle_path(tmp_path, leg_for(FLOOR_LEG_ID)).read_text(encoding="utf-8")
        )
        fulfil_job(s3, handle)
        cli.collect_batch(FLOOR_LEG_ID, tmp_path, timeout_seconds=1.0)
        totals = cli.scan(tmp_path)
        assert totals["examined"] == FLOOR_RECORDS
        assert totals["scanned"] == FLOOR_RECORDS
        assert totals["parsed"] == FLOOR_RECORDS
        assert totals["errored"] == 0

    @pytest.mark.parametrize(
        ("field", "expected"), [("prompt_digest", "prompt digest"), ("cell_digest", "cell digest")]
    )
    def test_a_digest_mismatch_refuses_the_whole_collect(
        self,
        tmp_path: Path,
        aws: tuple[FakeBedrock, FakeS3],
        pinned_stimulus: DecoupledStimulus,
        field: str,
        expected: str,
    ) -> None:
        del pinned_stimulus
        _, s3 = aws
        cli.submit_batch(FLOOR_LEG_ID, tmp_path, check_only=False)
        path = cli.handle_path(tmp_path, leg_for(FLOOR_LEG_ID))
        handle = json.loads(path.read_text(encoding="utf-8"))
        fulfil_job(s3, handle)
        handle[field] = "0" * 64
        path.write_text(json.dumps(handle), encoding="utf-8")
        with pytest.raises(RuntimeError, match=expected):
            cli.collect_batch(FLOOR_LEG_ID, tmp_path, timeout_seconds=1.0)
        assert load_run_replies(tmp_path) == {}


class TestScriptedSmoke:
    def test_the_offline_smoke_runs_render_sample_scan_validate_and_judge(
        self, tmp_path: Path, pinned_stimulus: DecoupledStimulus
    ) -> None:
        summary = cli.smoke(tmp_path, backend="scripted")
        assert summary["backend"] == "scripted"
        assert summary["calls"] == cli.SMOKE_SCRIPTED_CALLS
        assert summary["ran"] == cli.SMOKE_SCRIPTED_CALLS
        scans = summary["scans"]
        assert isinstance(scans, dict)
        assert scans["scanned"] == cli.SMOKE_SCRIPTED_CALLS
        assert scans["parsed"] == cli.SMOKE_SCRIPTED_CALLS
        judge_counts = summary["judge_counts"]
        assert isinstance(judge_counts, dict)
        assert judge_counts["judged"] == cli.SMOKE_SCRIPTED_CALLS
        validation = summary["judge_validation"]
        assert isinstance(validation, dict)
        assert validation["validated"] == len(pinned_stimulus.validation_replies)
        assert summary["measured_chars_per_token"]
        smoke_dir = tmp_path / "smoke" / "scripted"
        assert (smoke_dir / "summary-smoke.json").exists()
        assert (smoke_dir / "scans.jsonl").exists()
        assert (smoke_dir / "judged.jsonl").exists()
        assert not list(tmp_path.glob("replies--*.jsonl")), "smoke records stay under smoke/"

    def test_the_smoke_is_resumable_the_same_way_a_production_leg_is(
        self, tmp_path: Path, pinned_stimulus: DecoupledStimulus
    ) -> None:
        del pinned_stimulus
        cli.smoke(tmp_path, backend="scripted")
        again = cli.smoke(tmp_path, backend="scripted")
        assert again["resumed"] == cli.SMOKE_SCRIPTED_CALLS
        assert again["ran"] == 0

    def test_the_smoke_reports_inline_thinking_per_model_with_its_denominator(
        self, tmp_path: Path, pinned_stimulus: DecoupledStimulus
    ) -> None:
        """Which roster rows reason inside the answer channel decides what both instruments read."""
        del pinned_stimulus
        summary = cli.smoke(tmp_path, backend="scripted")
        counts = summary["inline_think_by_model"]
        assert isinstance(counts, dict)
        assert counts == {
            cli.SMOKE_LIVE_MODEL_ID: {
                "replies": cli.SMOKE_SCRIPTED_CALLS,
                "with_inline_think": 0,
            }
        }
        assert summary["judged_rows_with_inline_think"] == 0

    def test_a_reply_that_thinks_in_the_answer_channel_is_counted_and_judged_consistently(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pinned_stimulus: DecoupledStimulus
    ) -> None:
        """The counting has teeth only if a reply that does carry a block moves the number."""
        del pinned_stimulus
        plain = cli._scripted_reply

        def thinking_reply(prompt: str) -> str:
            return f"<think>weighing it up</think>{plain(prompt)}"

        monkeypatch.setattr(cli, "_scripted_reply", thinking_reply)
        summary = cli.smoke(tmp_path, backend="scripted")
        counts = summary["inline_think_by_model"]
        assert isinstance(counts, dict)
        assert counts[cli.SMOKE_LIVE_MODEL_ID]["with_inline_think"] == cli.SMOKE_SCRIPTED_CALLS
        assert summary["judged_rows_with_inline_think"] == cli.SMOKE_SCRIPTED_CALLS
        scans = summary["scans"]
        assert isinstance(scans, dict)
        assert scans["parsed"] == cli.SMOKE_SCRIPTED_CALLS

    def test_the_smoke_lists_every_validation_miss_by_name_and_key(
        self, tmp_path: Path, pinned_stimulus: DecoupledStimulus
    ) -> None:
        """The offline smoke's fixed verdict disagrees on purpose; the misses must be readable here."""
        del pinned_stimulus
        summary = cli.smoke(tmp_path, backend="scripted")
        misses = summary["judge_validation_misses"]
        assert isinstance(misses, list)
        assert misses
        for miss in misses:
            assert miss["key"] == f"validation|{miss['name']}"
            assert miss["field"]
            assert miss["expected"] != miss["got"]

    def test_a_judged_row_that_disagrees_with_its_reply_about_thinking_refuses(
        self, tmp_path: Path, pinned_stimulus: DecoupledStimulus
    ) -> None:
        """The sabotage of the consistency check: flip one stored flag and the smoke must go red."""
        del pinned_stimulus
        cli.smoke(tmp_path, backend="scripted")
        smoke_dir = tmp_path / "smoke" / "scripted"
        judged = smoke_dir / "judged.jsonl"
        rows = [
            json.loads(line) for line in judged.read_text(encoding="utf-8").splitlines() if line
        ]
        rows[0]["had_inline_think"] = True
        judged.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        with pytest.raises(RuntimeError, match="thinking block"):
            cli._refuse_judge_that_read_a_different_text(load_run_replies(smoke_dir), judged)


class TestCliParsing:
    def test_every_leg_id_is_an_accepted_choice(self) -> None:
        for leg_id in cli.LEGS_BY_ID:
            args = cli._parse_args(["submit-batch", "--leg", leg_id])
            assert args.leg == leg_id

    def test_an_unknown_leg_is_rejected_by_the_parser(self) -> None:
        with pytest.raises(SystemExit):
            cli._parse_args(["submit-batch", "--leg", "not-a-leg"])

    @pytest.mark.parametrize(
        "leg_id",
        [
            "sonnet-5--default--A--anchor",
            "sonnet-5--default--B--floor",
            "sonnet-5--default--A--ladder",
        ],
    )
    def test_the_sonnet_legs_are_run_live_choices(self, leg_id: str) -> None:
        args = cli._parse_args(["run-live", "--leg", leg_id])
        assert args.leg == leg_id
        assert leg_for(leg_id).transport == "live"

    def test_add_ladder_offers_every_model_with_an_on_demand_leg(self) -> None:
        for handle in ("opus", "luna", "sonnet"):
            assert cli._parse_args(["add-ladder", "--model", handle]).model == handle
        with pytest.raises(SystemExit):
            cli._parse_args(["add-ladder", "--model", "haiku"])

    def test_the_smoke_backend_choices_are_scripted_and_live(self) -> None:
        assert cli._parse_args(["smoke"]).backend == "scripted"
        assert cli._parse_args(["smoke", "--backend", "live"]).backend == "live"

    @pytest.mark.parametrize("command", ["judge", "cross-judge"])
    def test_both_judge_subcommands_take_a_concurrency_of_at_least_one(self, command: str) -> None:
        assert cli._parse_args([command]).concurrency == cli.DEFAULT_JUDGE_CONCURRENCY
        assert cli._parse_args([command, "--concurrency", "48"]).concurrency == 48
        assert cli._parse_args([command, "--concurrency", "1"]).concurrency == 1
        for rejected in ("0", "-4", "many"):
            with pytest.raises(SystemExit):
                cli._parse_args([command, "--concurrency", rejected])


def assert_no_design_label(prompt: str) -> None:
    """Fail if any string that names the design appears in one prompt, case-insensitively."""
    lowered = prompt.lower()
    for label in DESIGN_LABELS:
        assert label.lower() not in lowered, label


def seed_replies(run_dir: Path, stimulus: DecoupledStimulus, count: int = 3) -> None:
    """Write a few real reply records into a run dir, through the production live path."""
    calls = cli.planned_calls_for_leg(leg_for("luna--default--A--anchor"), stimulus)[:count]
    cli._run_live_calls(
        calls,
        run_dir / "replies--luna--effort-default--sitting-A--anchor.jsonl",
        stimulus,
        concurrency=1,
        chunk_size=count,
        backend_factory=scripted_factory("<action>SHORT</action>"),
    )


class TestLiveResumeRefusesADifferentExperiment:
    """A resumed key must answer the same prompt under the same stimulus, or it is two experiments.

    Nothing about the file's shape changes when it stops being true: the keys still match, the counts
    still add up, and the second pass reports itself finished having called nothing.
    """

    def calls(self, stimulus: DecoupledStimulus, count: int) -> list[PlannedCall]:
        return cli.planned_calls_for_leg(leg_for("luna--default--A--anchor"), stimulus)[:count]

    def reply_file(self, tmp_path: Path) -> Path:
        return tmp_path / "replies--luna--effort-default--sitting-A--anchor.jsonl"

    def test_a_second_sitting_under_an_edited_stimulus_refuses(
        self, tmp_path: Path, pinned_stimulus: DecoupledStimulus
    ) -> None:
        out = self.reply_file(tmp_path)
        cli._run_live_calls(
            self.calls(pinned_stimulus, 2),
            out,
            pinned_stimulus,
            concurrency=1,
            chunk_size=1,
            backend_factory=scripted_factory("<action>SHORT</action>"),
        )
        edited = replace(
            pinned_stimulus, digest="0f0f0f0f0f0f0f0f", judge_instructions="SYNTHETIC-RUBRIC: two."
        )
        with pytest.raises(ValueError, match="stimulus_digest") as raised:
            cli._run_live_calls(
                self.calls(edited, 4),
                out,
                edited,
                concurrency=1,
                chunk_size=1,
                backend_factory=scripted_factory("<action>SHORT</action>"),
            )
        assert pinned_stimulus.digest in str(raised.value)
        assert "0f0f0f0f0f0f0f0f" in str(raised.value)

    def test_a_record_whose_prompt_digest_moved_refuses_and_names_both(
        self, tmp_path: Path, pinned_stimulus: DecoupledStimulus
    ) -> None:
        """The renderer changing under a stable key is the same failure as an edited stimulus."""
        out = self.reply_file(tmp_path)
        cli._run_live_calls(
            self.calls(pinned_stimulus, 1),
            out,
            pinned_stimulus,
            concurrency=1,
            chunk_size=1,
            backend_factory=scripted_factory("<action>SHORT</action>"),
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
                backend_factory=scripted_factory("<action>SHORT</action>"),
            )
        assert row["key"] in str(raised.value)
        assert "ffffffffffffffff" in str(raised.value)

    def test_an_unchanged_second_pass_still_resumes(
        self, tmp_path: Path, pinned_stimulus: DecoupledStimulus
    ) -> None:
        out = self.reply_file(tmp_path)

        def pass_over_two_calls() -> dict[str, int]:
            return cli._run_live_calls(
                self.calls(pinned_stimulus, 2),
                out,
                pinned_stimulus,
                concurrency=1,
                chunk_size=2,
                backend_factory=scripted_factory("<action>SHORT</action>"),
            )

        assert pass_over_two_calls() == {"planned": 2, "resumed": 0, "ran": 2, "incomplete": 0}
        assert pass_over_two_calls() == {"planned": 2, "resumed": 2, "ran": 0, "incomplete": 0}


def scripted_judge(stimulus: DecoupledStimulus) -> ScriptedDetailedBackend:
    """One offline judge for every path: it agrees with the validation set and abstains elsewhere.

    Keyed on the prompt rather than served round-robin, because the validation pass and the production
    pass run through the same backend here and only the validation prompts have a registered right
    answer. Longest name first: several validation names are prefixes of others.
    """
    by_name = {reply.name: reply for reply in stimulus.validation_replies}
    names = sorted(by_name, key=lambda name: -len(name))

    def answer(prompt: str) -> str:
        name = next((name for name in names if name in prompt), None)
        if name is None:
            return cli._SCRIPTED_JUDGE_VERDICT
        return json.dumps({**by_name[name].expected, "evidence": "SYNTHETIC-EVIDENCE"})

    return ScriptedDetailedBackend(answer)


class RecordingJudgeFactory:
    """Stands in for the CLI's judge-backend seam and keeps every request it was asked to build.

    The judge summary claims a concurrency, and the only way to know the claim is true is to watch the
    same number arrive at the constructor the production path would call.
    """

    def __init__(self, backend: ScriptedDetailedBackend) -> None:
        self.backend = backend
        self.calls: list[tuple[str, str | None, int]] = []

    def __call__(self, model_id: str, effort: str | None, *, concurrency: int) -> DetailedBackend:
        self.calls.append((model_id, effort, concurrency))
        return self.backend


@pytest.fixture
def judge_factory(
    monkeypatch: pytest.MonkeyPatch, decoupled_stimulus: DecoupledStimulus
) -> RecordingJudgeFactory:
    """Route every judge-backend construction through a recording factory over one scripted judge."""
    factory = RecordingJudgeFactory(scripted_judge(decoupled_stimulus))
    monkeypatch.setattr(cli, "_judge_backend", factory)
    return factory


@pytest.fixture
def offline_judge(judge_factory: RecordingJudgeFactory) -> ScriptedDetailedBackend:
    """Serve every judge call from one scripted backend, and keep the prompts it was handed."""
    return judge_factory.backend


class TestTheJudgeWaitsForItsValidation:
    """An uncalibrated judge produces the ladder's headline rate and looks perfectly healthy doing it.

    So the gate is on the validation SUMMARY for the current rubric digest with nothing outstanding,
    rather than on the operator remembering the order of the subcommands.
    """

    def test_judging_before_any_validation_refuses_and_names_the_fix(
        self, tmp_path: Path, pinned_stimulus: DecoupledStimulus, offline_judge: object
    ) -> None:
        del offline_judge
        seed_replies(tmp_path, pinned_stimulus)
        with pytest.raises(RuntimeError, match="judge-validate"):
            cli.judge(tmp_path)
        assert not (tmp_path / "judged.jsonl").exists()

    def test_cross_judging_before_any_validation_refuses(
        self, tmp_path: Path, pinned_stimulus: DecoupledStimulus, offline_judge: object
    ) -> None:
        del offline_judge
        seed_replies(tmp_path, pinned_stimulus)
        cli.scan(tmp_path)
        with pytest.raises(RuntimeError, match="judge-validate"):
            cli.cross_judge(tmp_path, n=2)
        assert not (tmp_path / "cross-judged.jsonl").exists()

    def test_a_clean_validation_opens_the_gate_and_the_summary_says_it_was_not_skipped(
        self, tmp_path: Path, pinned_stimulus: DecoupledStimulus, offline_judge: object
    ) -> None:
        del offline_judge
        seed_replies(tmp_path, pinned_stimulus)
        report = cli.judge_validate(tmp_path)
        assert report["misses"] == []
        summary = json.loads((tmp_path / cli.JUDGE_VALIDATION_SUMMARY).read_text(encoding="utf-8"))
        assert summary["judge_prompt_digest"] == judge_module.rubric_digest(pinned_stimulus)
        counts = cli.judge(tmp_path)
        assert counts["judged"] == 3
        judge_summary = json.loads((tmp_path / "summary-judge.json").read_text(encoding="utf-8"))
        assert judge_summary["skipped_validation_gate"] is False

    def test_a_validation_with_misses_keeps_the_gate_shut(
        self, tmp_path: Path, pinned_stimulus: DecoupledStimulus, offline_judge: object
    ) -> None:
        del offline_judge
        seed_replies(tmp_path, pinned_stimulus)
        cli.judge_validate(tmp_path)
        path = tmp_path / cli.JUDGE_VALIDATION_SUMMARY
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["misses"] = [
            {"name": "v-mirror", "key": "validation|v-mirror", "field": "they_are_me"}
        ]
        path.write_text(json.dumps(summary), encoding="utf-8")
        with pytest.raises(RuntimeError, match="field misses"):
            cli.judge(tmp_path)

    def test_a_validation_of_a_different_rubric_does_not_clear_this_one(
        self, tmp_path: Path, pinned_stimulus: DecoupledStimulus, offline_judge: object
    ) -> None:
        del offline_judge
        seed_replies(tmp_path, pinned_stimulus)
        cli.judge_validate(tmp_path)
        path = tmp_path / cli.JUDGE_VALIDATION_SUMMARY
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["judge_prompt_digest"] = "0f0f0f0f0f0f0f0f"
        path.write_text(json.dumps(summary), encoding="utf-8")
        with pytest.raises(RuntimeError, match="not this run's"):
            cli.judge(tmp_path)

    def test_the_skip_flag_runs_anyway_and_stamps_the_summary(
        self, tmp_path: Path, pinned_stimulus: DecoupledStimulus, offline_judge: object
    ) -> None:
        del offline_judge
        seed_replies(tmp_path, pinned_stimulus)
        cli.judge(tmp_path, skip_validation_gate=True)
        judge_summary = json.loads((tmp_path / "summary-judge.json").read_text(encoding="utf-8"))
        assert judge_summary["skipped_validation_gate"] is True

    def test_no_design_label_reaches_the_backend_on_the_cross_judge_path(
        self,
        tmp_path: Path,
        pinned_stimulus: DecoupledStimulus,
        offline_judge: ScriptedDetailedBackend,
    ) -> None:
        seed_replies(tmp_path, pinned_stimulus)
        cli.scan(tmp_path)
        cli.judge_validate(tmp_path)
        counts = cli.cross_judge(tmp_path, n=2)
        assert counts["judged"] == 2
        assert offline_judge.prompts_seen
        for prompt in offline_judge.prompts_seen:
            assert_no_design_label(prompt)

    def test_the_cross_judge_subset_is_allocated_per_model_before_per_cell(
        self,
        tmp_path: Path,
        pinned_stimulus: DecoupledStimulus,
        offline_judge: ScriptedDetailedBackend,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The agreement table is read per model, so every model has to reach the second judge.

        The shared loop grew an outer stratum for exactly this after a 30-row subset over
        ``model|cell|action`` strata reached two models of nine; this checks the ladder's call site
        actually passes it, with the model id as the outer key, and that a two-model run at ``n=2``
        covers both.
        """
        del offline_judge
        seed_replies(tmp_path, pinned_stimulus)
        second_leg = leg_for("sonnet-5--default--A--anchor")
        cli._run_live_calls(
            cli.planned_calls_for_leg(second_leg, pinned_stimulus)[:3],
            cli.reply_path(tmp_path, second_leg),
            pinned_stimulus,
            concurrency=1,
            chunk_size=3,
            backend_factory=scripted_factory("<action>SHORT</action>"),
        )
        cli.scan(tmp_path)
        cli.judge_validate(tmp_path)
        seen: dict[str, Any] = {}
        real_subset = judge_module.stratified_subset

        def recording_subset(records: Any, **kwargs: Any) -> Any:
            seen["records"] = list(records)
            seen["outer_stratum"] = kwargs.get("outer_stratum")
            return real_subset(records, **kwargs)

        monkeypatch.setattr(judge_module, "stratified_subset", recording_subset)
        cli.cross_judge(tmp_path, n=2)
        outer = seen["outer_stratum"]
        assert outer is not None, "the ladder's cross-judge must stratify per model first"
        assert {outer(record) for record in seen["records"]} == {
            str(record["model_id"]) for record in seen["records"]
        }
        judged = json.loads((tmp_path / "summary-cross-judge.json").read_text(encoding="utf-8"))
        rows = [
            json.loads(line)
            for line in (tmp_path / "cross-judged.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert judged["selected"] == 2
        assert {str(row["model_id"]) for row in rows} == {
            str(record["model_id"]) for record in seen["records"]
        }


class TestJudgeConcurrencyReachesTheBackend:
    """``--concurrency`` is a throughput knob, and the summary's claim about it has to be the real value.

    A flag parsed but never threaded through would leave the pass at the transport default while the
    summary said otherwise, so each test reads the number off the backend factory the production path
    calls, and off the summary written last.
    """

    def validated_run(self, tmp_path: Path, stimulus: DecoupledStimulus) -> None:
        seed_replies(tmp_path, stimulus)
        cli.scan(tmp_path)
        cli.judge_validate(tmp_path)

    def test_the_judge_default_is_the_transport_default_and_is_recorded(
        self,
        tmp_path: Path,
        pinned_stimulus: DecoupledStimulus,
        judge_factory: RecordingJudgeFactory,
    ) -> None:
        self.validated_run(tmp_path, pinned_stimulus)
        cli.judge(tmp_path)
        assert judge_factory.calls[-1] == (
            judge_module.JUDGE_MODEL_ID,
            judge_module.JUDGE_REASONING_EFFORT,
            cli.DEFAULT_JUDGE_CONCURRENCY,
        )
        summary = json.loads((tmp_path / "summary-judge.json").read_text(encoding="utf-8"))
        assert summary["judge_concurrency"] == cli.DEFAULT_JUDGE_CONCURRENCY

    def test_the_judge_flag_reaches_the_factory_through_main_and_the_summary(
        self,
        tmp_path: Path,
        pinned_stimulus: DecoupledStimulus,
        judge_factory: RecordingJudgeFactory,
    ) -> None:
        self.validated_run(tmp_path, pinned_stimulus)
        assert cli.main(["--run-dir", str(tmp_path), "judge", "--concurrency", "48"]) == 0
        assert judge_factory.calls[-1] == (
            judge_module.JUDGE_MODEL_ID,
            judge_module.JUDGE_REASONING_EFFORT,
            48,
        )
        summary = json.loads((tmp_path / "summary-judge.json").read_text(encoding="utf-8"))
        assert summary["judge_concurrency"] == 48
        assert summary["judged"] == 3

    def test_the_cross_judge_flag_reaches_the_factory_through_main_and_the_summary(
        self,
        tmp_path: Path,
        pinned_stimulus: DecoupledStimulus,
        judge_factory: RecordingJudgeFactory,
    ) -> None:
        self.validated_run(tmp_path, pinned_stimulus)
        argv = ["--run-dir", str(tmp_path), "cross-judge", "--n", "2", "--concurrency", "48"]
        assert cli.main(argv) == 0
        assert judge_factory.calls[-1] == (
            judge_module.CROSS_JUDGE_MODEL_ID,
            judge_module.CROSS_JUDGE_REASONING_EFFORT,
            48,
        )
        summary = json.loads((tmp_path / "summary-cross-judge.json").read_text(encoding="utf-8"))
        assert summary["judge_concurrency"] == 48
        assert summary["judged"] == 2

    def test_the_cross_judge_default_is_recorded_too(
        self,
        tmp_path: Path,
        pinned_stimulus: DecoupledStimulus,
        judge_factory: RecordingJudgeFactory,
    ) -> None:
        self.validated_run(tmp_path, pinned_stimulus)
        cli.cross_judge(tmp_path, n=2)
        assert judge_factory.calls[-1][2] == cli.DEFAULT_JUDGE_CONCURRENCY
        summary = json.loads((tmp_path / "summary-cross-judge.json").read_text(encoding="utf-8"))
        assert summary["judge_concurrency"] == cli.DEFAULT_JUDGE_CONCURRENCY


def tracked_probe_dir() -> Path:
    """A directory inside the repository that git tracks; nothing may ever be created under it.

    Anchored on the real repository root rather than the caller's cwd: a relative path would resolve
    against wherever pytest was launched from, and the guard under test would be answering a different
    question than the one these tests name.
    """
    root = _repo_root()
    assert root is not None, "these tests describe behaviour inside a git repository"
    return root / "sociology" / "writer-guard-probe"


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
        "append_judge_validation",
        lambda base: judge_module.append_judged(base / "judge-validation.jsonl", [{"key": "k"}]),
    ),
    ("scans", scans_module.scan_run),
    ("plan_json", cli.build),
    ("add_ladder", lambda base: cli.add_ladder(base, "opus")),
    ("submit_handle", lambda base: cli.submit_batch(FLOOR_LEG_ID, base, check_only=True)),
)


class TestEveryWriterRefusesATrackedDestination:
    """One case per writer, because the leak of this class this repo had was an unguarded second one.

    Each writer carries verbatim replies, the judge's quoted evidence, the authored clause prose in a
    digest, or the batch job's account-identifying values, and this remote is public. The refusal has
    to land BEFORE anything is created, which is why every case asserts the empty directory as well:
    a guard that fires after the write, or after a mkdir, has already published the path.
    """

    @pytest.mark.parametrize(("name", "writer"), WRITER_CASES, ids=[n for n, _ in WRITER_CASES])
    def test_it_refuses_before_creating_anything(
        self,
        name: str,
        writer: Callable[[Path], object],
        pinned_stimulus: DecoupledStimulus,
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
