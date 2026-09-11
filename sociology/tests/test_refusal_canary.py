"""The live refusal canary in front of a Claude batch submit, offline against scripted live backends.

Nothing here touches AWS: the batch control plane and S3 are the same in-memory fakes the ladder CLI
tests use, and the canary's "live" backend is scripted per prompt. The property under test is the
decision, not the transport: a Claude leg whose canary comes back refused wholesale creates no job and
saves no handle, a leg whose canary answers submits exactly as before with the canary's counts beside
the job, and a leg on any other vendor never builds a canary backend at all. All three Claude batch
submit paths are covered -- the decoupled ladder, the analysis-model study's stage 1, and the one-way
transfer study -- because the canary's contract is "before every Claude batch submit".
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from reward_hacking.bedrock_batch import (
    BATCH_BUCKET_ENV,
    BATCH_BUCKET_OWNER_ENV,
    BATCH_PROFILE_ENV,
    BATCH_ROLE_ARN_ENV,
)
from reward_hacking.model_backend import BedrockCompletion, TokenUsage
from sociology import decoupled_ladder, runner, transfer_cli
from sociology.bundles import CELLS_BY_NAME, build_manifest, write_manifest
from sociology.decoupled_plan import LEGS_BY_ID as LADDER_LEGS_BY_ID
from sociology.decoupled_plan import TRANSPORT_BATCH, leg_for, planned_calls_for_leg
from sociology.decoupled_scans import STOP_REASON_CONTENT_FILTERED
from sociology.refusal_canary import (
    REFUSAL_CANARY_CALLS,
    RefusalCanaryError,
    canary_calls,
    canary_plan,
    needs_refusal_canary,
    recorded_canary,
    run_refusal_canary,
)
from sociology.tests.test_bundles import synthetic_pools
from sociology.tests.test_decoupled_ladder_cli import FakeBedrock, FakeS3
from sociology.tests.test_runner_plan_resume import synthetic_units
from sociology.transfer_plan import LEGS_BY_ID as TRANSFER_LEGS_BY_ID
from sociology.transfer_plan import ONE_WAY_TRANSFER_PLAN
from sociology.transfer_plan import leg_for as transfer_leg_for
from sociology.transfer_plan import planned_calls_for_leg as transfer_calls_for_leg

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from reward_hacking.model_backend import DetailedBackend
    from sociology.decoupled_stimulus import DecoupledStimulus
    from sociology.stimulus import Stimulus
    from sociology.transfer_stimulus import TransferStimulus

HAIKU_ANCHOR_LEG_ID = "haiku-4-5--default--A--anchor"
FLOOR_LEG_ID = "gpt-oss-20b--default--B--floor"
TRANSFER_HAIKU_FLOOR_LEG_ID = "haiku-4-5--default--B--floor"

REFUSED = STOP_REASON_CONTENT_FILTERED
ANSWERED = "end_turn"
FAILED = "call_failed:ThrottlingException"

CANARY_REPLY_SENTINEL = "CANARY-REPLY-SENTINEL-NEVER-ON-DISK"
"""What every answering canary call says; a run dir or an S3 object containing it is a leak."""


class CanaryBackend:
    """A scripted live backend: one stop reason per prompt, every prompt it saw recorded."""

    def __init__(self, stop_reason_for: Callable[[str], str], *, model_id: str) -> None:
        self.model_id = model_id
        self._stop_reason_for = stop_reason_for
        self.prompts_seen: list[str] = []

    def generate_detailed(self, prompts: list[str]) -> list[BedrockCompletion]:
        self.prompts_seen.extend(prompts)
        return [
            BedrockCompletion(
                text="" if self._stop_reason_for(prompt) == REFUSED else CANARY_REPLY_SENTINEL,
                reasoning="",
                usage=TokenUsage(input_tokens=len(prompt) // 4, output_tokens=3),
                stop_reason=self._stop_reason_for(prompt),
                elapsed_seconds=1.5,
                first_event_seconds=0.5,
                attempts=1,
            )
            for prompt in prompts
        ]


class CanaryFactory:
    """A backend factory that remembers what it built, so a test can prove it was never asked."""

    def __init__(self, stop_reason_for: Callable[[str], str]) -> None:
        self._stop_reason_for = stop_reason_for
        self.built: list[CanaryBackend] = []
        self.requested: list[tuple[str, str | None, int]] = []

    def __call__(self, model_id: str, effort: str | None, concurrency: int) -> DetailedBackend:
        self.requested.append((model_id, effort, concurrency))
        backend = CanaryBackend(self._stop_reason_for, model_id=model_id)
        self.built.append(backend)
        return backend

    @property
    def prompts_seen(self) -> list[str]:
        return [prompt for backend in self.built for prompt in backend.prompts_seen]


def always(stop_reason: str) -> Callable[[str], str]:
    return lambda _prompt: stop_reason


def submitted_prompts(s3: FakeS3) -> list[str]:
    """Read the prompt sequence the batch input object carries, in record order."""
    (body,) = [body for uri, body in s3.objects.items() if uri.endswith("input.jsonl")]
    return [
        json.loads(line)["modelInput"]["messages"][0]["content"][0]["text"]
        for line in body.splitlines()
        if line.strip()
    ]


def assert_sentinel_nowhere(run_dir: Path, s3: FakeS3) -> None:
    """Fail if the canary's reply text landed in any file under the run dir or any S3 object."""
    for path in run_dir.rglob("*"):
        if path.is_file():
            assert CANARY_REPLY_SENTINEL not in path.read_text(encoding="utf-8"), path
    for uri, body in s3.objects.items():
        assert CANARY_REPLY_SENTINEL not in body, uri


@pytest.fixture
def aws(monkeypatch: pytest.MonkeyPatch) -> tuple[FakeBedrock, FakeS3]:
    """The ladder CLI tests' in-memory batch control plane and S3, under synthetic account values."""
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
def pinned_decoupled_stimulus(
    monkeypatch: pytest.MonkeyPatch, decoupled_stimulus: DecoupledStimulus
) -> DecoupledStimulus:
    monkeypatch.setattr(decoupled_ladder, "load_stimulus", lambda: decoupled_stimulus)
    return decoupled_stimulus


@pytest.fixture
def pinned_transfer_stimulus(
    monkeypatch: pytest.MonkeyPatch, transfer_stimulus: TransferStimulus
) -> TransferStimulus:
    monkeypatch.setattr(transfer_cli, "load_stimulus", lambda: transfer_stimulus)
    return transfer_stimulus


@pytest.fixture
def stage_one_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stimulus: Stimulus) -> Path:
    """A stage-1 run dir with a synthetic manifest, the loaders pointed at synthetic material."""
    write_manifest(build_manifest(synthetic_pools(), stimulus), tmp_path)
    monkeypatch.setattr(runner, "load_stimulus", lambda: stimulus)
    monkeypatch.setattr(runner, "load_units", synthetic_units)
    return tmp_path


def stage_one_cells(model_id: str) -> set[str]:
    """The cells the stage-1 plan pools into one model's job, read off the production leg table."""
    return {leg.cell for leg in runner.PRODUCTION_LEGS if leg.model_id == model_id}


class TestWhichModelsGetACanary:
    @pytest.mark.parametrize(
        "model_id",
        [
            "global.anthropic.claude-opus-5",
            "global.anthropic.claude-haiku-4-5-20251001-v1:0",
            "us.anthropic.claude-sonnet-5",
            "anthropic.claude-sonnet-5",
        ],
    )
    def test_every_claude_id_shape_gets_one(self, model_id: str) -> None:
        assert needs_refusal_canary(model_id)

    @pytest.mark.parametrize(
        "model_id",
        ["openai.gpt-oss-20b-1:0", "global.openai.gpt-5.6-luna", "deepseek.v3.2", "zai.glm-4.7"],
    )
    def test_no_other_vendor_does(self, model_id: str) -> None:
        assert not needs_refusal_canary(model_id)


class TestWhichPromptsTheCanarySends:
    def test_the_first_distinct_prompts_of_every_cell_in_plan_order(
        self, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        """Draws repeat a prompt contiguously; five draws of one text would test the classifier once."""
        calls = planned_calls_for_leg(leg_for(HAIKU_ANCHOR_LEG_ID), decoupled_stimulus)
        picked = canary_calls(calls, group_of=lambda call: call.cell, calls_per_group=3)
        assert set(picked) == {call.cell for call in calls}
        for cell, chosen in picked.items():
            assert len(chosen) == 3
            assert len({call.prompt for call in chosen}) == 3, "distinct prompts, not draws"
            first_three_distinct: list[str] = []
            for call in calls:
                if call.cell == cell and call.prompt not in first_three_distinct:
                    first_three_distinct.append(call.prompt)
                if len(first_three_distinct) == 3:
                    break
            assert [call.prompt for call in chosen] == first_three_distinct

    def test_a_group_with_fewer_distinct_prompts_than_asked_refuses_at_plan_time(
        self, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        """A one-prompt group canaried on one call runs the one-call rule the module rejects.

        SABOTAGE target: let a thin group "contribute what it has" and this passes silently.
        """
        calls = planned_calls_for_leg(leg_for(HAIKU_ANCHOR_LEG_ID), decoupled_stimulus)
        every_draw_of_one_prompt = [call for call in calls if call.prompt == calls[0].prompt]
        assert len(every_draw_of_one_prompt) > 1, "the plan lists several draws of one prompt"
        with pytest.raises(
            ValueError, match=r"have fewer: \{'[^']+': 1\}.*--canary-calls 1 or fewer"
        ):
            canary_calls(
                every_draw_of_one_prompt, group_of=lambda call: call.cell, calls_per_group=5
            )
        with pytest.raises(ValueError, match="have fewer"):
            canary_plan(
                every_draw_of_one_prompt, group_of=lambda call: call.cell, calls_per_group=2
            )
        picked = canary_calls(
            every_draw_of_one_prompt, group_of=lambda call: call.cell, calls_per_group=1
        )
        assert [len(chosen) for chosen in picked.values()] == [1]

    def test_every_production_claude_batch_group_supplies_the_default_call_count(
        self,
        decoupled_stimulus: DecoupledStimulus,
        transfer_stimulus: TransferStimulus,
        stage_one_run: Path,
        stimulus: Stimulus,
    ) -> None:
        """The thin-group refusal must never bite a plan as it stands; a new thin cell fails here first."""
        plans: list[dict[str, Any]] = [
            canary_plan(
                planned_calls_for_leg(leg, decoupled_stimulus),
                group_of=lambda call: call.cell,
                calls_per_group=REFUSAL_CANARY_CALLS,
            )
            for leg in LADDER_LEGS_BY_ID.values()
            if leg.transport == TRANSPORT_BATCH and needs_refusal_canary(leg.model_id)
        ]
        plans.extend(
            canary_plan(
                transfer_calls_for_leg(leg, transfer_stimulus),
                group_of=lambda call: call.cell,
                calls_per_group=REFUSAL_CANARY_CALLS,
            )
            for leg in TRANSFER_LEGS_BY_ID.values()
            if leg.transport == TRANSPORT_BATCH and needs_refusal_canary(leg.model_id)
        )
        manifest = runner.load_manifest(stage_one_run)
        plans.extend(
            canary_plan(
                runner.planned_calls_for_model(model_id, manifest, synthetic_units(), stimulus),
                group_of=lambda call: call.cell,
                calls_per_group=REFUSAL_CANARY_CALLS,
            )
            for model_id in (runner.OPUS_MODEL_ID, runner.HAIKU_MODEL_ID)
        )
        assert len(plans) >= 3, "every study contributed at least one Claude batch job"
        for plan in plans:
            assert set(plan["groups"].values()) == {REFUSAL_CANARY_CALLS}

    def test_the_plan_reports_counts_without_a_backend(
        self, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        calls = planned_calls_for_leg(leg_for(HAIKU_ANCHOR_LEG_ID), decoupled_stimulus)
        plan = canary_plan(calls, group_of=lambda call: call.cell, calls_per_group=2)
        assert plan["calls_per_group"] == 2
        assert plan["calls"] == 2 * len(plan["groups"])
        assert set(plan["groups"]) == {call.cell for call in calls}

    def test_zero_calls_per_group_refuses(self, decoupled_stimulus: DecoupledStimulus) -> None:
        calls = planned_calls_for_leg(leg_for(HAIKU_ANCHOR_LEG_ID), decoupled_stimulus)
        with pytest.raises(ValueError, match="at least 1"):
            canary_calls(calls, group_of=lambda call: call.cell, calls_per_group=0)


class TestTheCanaryDecision:
    def calls(self, stimulus: DecoupledStimulus) -> list[Any]:
        return planned_calls_for_leg(leg_for(HAIKU_ANCHOR_LEG_ID), stimulus)

    def test_every_call_of_one_cell_refused_trips_and_names_the_cell(
        self, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        calls = self.calls(decoupled_stimulus)
        refused_cell = calls[0].cell
        refused_prompts = {call.prompt for call in calls if call.cell == refused_cell}
        factory = CanaryFactory(lambda prompt: REFUSED if prompt in refused_prompts else ANSWERED)
        result = run_refusal_canary(
            calls, group_of=lambda call: call.cell, backend_factory=factory, calls_per_group=5
        )
        assert result.tripped
        assert result.unanswered_groups == (refused_cell,)
        assert result.summary["groups"][refused_cell] == {
            "calls": 5,
            "refused": 5,
            "failed": 0,
            "answered": 0,
            "stop_reasons": {REFUSED: 5},
        }
        with pytest.raises(RefusalCanaryError, match=refused_cell):
            result.refuse_if_tripped(job="leg x", records=len(calls))

    def test_four_of_five_refused_does_not_trip_but_is_counted(
        self, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        """Per-record refusal is real on merely disliked stimulus; only a whole group refusing is wholesale."""
        calls = self.calls(decoupled_stimulus)
        picked = canary_calls(calls, group_of=lambda call: call.cell, calls_per_group=5)
        spared = {chosen[0].prompt for chosen in picked.values()}
        factory = CanaryFactory(lambda prompt: ANSWERED if prompt in spared else REFUSED)
        result = run_refusal_canary(
            calls, group_of=lambda call: call.cell, backend_factory=factory, calls_per_group=5
        )
        assert not result.tripped
        assert result.summary["refused"] == 4 * len(picked)
        assert result.summary["answered"] == len(picked)
        result.refuse_if_tripped(job="leg x", records=len(calls))

    def test_a_cell_that_failed_on_every_call_trips_and_the_message_says_failed(
        self, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        calls = self.calls(decoupled_stimulus)
        factory = CanaryFactory(always(FAILED))
        result = run_refusal_canary(
            calls, group_of=lambda call: call.cell, backend_factory=factory, calls_per_group=2
        )
        assert result.tripped
        assert result.summary["failed"] == result.summary["calls"]
        with pytest.raises(RefusalCanaryError, match="0 refused, 2 failed of 2"):
            result.refuse_if_tripped(job="leg x", records=len(calls))

    def test_a_truncated_reply_is_an_answer_because_the_classifier_let_it_through(
        self, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        calls = self.calls(decoupled_stimulus)
        factory = CanaryFactory(always("max_tokens"))
        result = run_refusal_canary(
            calls, group_of=lambda call: call.cell, backend_factory=factory, calls_per_group=2
        )
        assert not result.tripped
        assert result.summary["answered"] == result.summary["calls"]
        assert result.summary["failed"] == 0

    def test_the_backend_is_built_once_for_the_jobs_model_at_the_jobs_effort(
        self, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        calls = self.calls(decoupled_stimulus)
        factory = CanaryFactory(always(ANSWERED))
        result = run_refusal_canary(
            calls, group_of=lambda call: call.cell, backend_factory=factory, calls_per_group=5
        )
        (requested,) = factory.requested
        assert requested[:2] == (calls[0].model_id, calls[0].reasoning_effort)
        assert requested[2] <= 8
        assert len(factory.prompts_seen) == result.summary["calls"]
        assert result.summary["input_tokens"] == sum(len(p) // 4 for p in factory.prompts_seen)
        assert result.summary["max_elapsed_seconds"] == 1.5

    def test_calls_spanning_two_models_refuse_before_any_call(
        self, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        calls = self.calls(decoupled_stimulus)
        mixed = [*calls, *planned_calls_for_leg(leg_for(FLOOR_LEG_ID), decoupled_stimulus)]
        factory = CanaryFactory(always(ANSWERED))
        with pytest.raises(ValueError, match="one model at one effort"):
            run_refusal_canary(
                mixed,
                group_of=lambda call: call.model_id,
                backend_factory=factory,
                calls_per_group=1,
            )
        assert factory.requested == []


class TestRecordedCanary:
    def test_a_missing_summary_and_a_pre_canary_summary_both_read_as_none(
        self, tmp_path: Path
    ) -> None:
        assert recorded_canary(tmp_path / "summary-submit-x.json") is None
        (tmp_path / "summary-submit-x.json").write_text(
            json.dumps({"records": 3}), encoding="utf-8"
        )
        assert recorded_canary(tmp_path / "summary-submit-x.json") is None

    def test_a_recorded_block_is_returned_whole(self, tmp_path: Path) -> None:
        block = {"calls": 10, "refused": 1, "groups": {"twin": {"calls": 5}}}
        (tmp_path / "summary-submit-x.json").write_text(
            json.dumps({"records": 3, "refusal_canary": block}), encoding="utf-8"
        )
        assert recorded_canary(tmp_path / "summary-submit-x.json") == block


class TestTheLadderSubmitRunsTheCanary:
    def test_a_refused_canary_creates_no_job_saves_no_handle_and_leaves_its_counts(
        self,
        tmp_path: Path,
        aws: tuple[FakeBedrock, FakeS3],
        pinned_decoupled_stimulus: DecoupledStimulus,
    ) -> None:
        del pinned_decoupled_stimulus
        bedrock, s3 = aws
        factory = CanaryFactory(always(REFUSED))
        with pytest.raises(RefusalCanaryError, match="refusing to submit leg"):
            decoupled_ladder.submit_batch(
                HAIKU_ANCHOR_LEG_ID, tmp_path, check_only=False, canary_backend_factory=factory
            )
        assert bedrock.created == []
        assert s3.objects == {}, (
            "nothing reached S3: the canary runs before the input file is written"
        )
        assert not decoupled_ladder.handle_path(tmp_path, leg_for(HAIKU_ANCHOR_LEG_ID)).exists()
        leg = leg_for(HAIKU_ANCHOR_LEG_ID)
        evidence = tmp_path / f"summary-refusal-canary-{leg.file_stem}.json"
        assert evidence.exists(), "a refused submit leaves the canary's counts in the run dir"
        assert not decoupled_ladder.submit_summary_path(tmp_path, leg).exists()
        assert factory.prompts_seen, "the canary did call"

    def test_an_answering_canary_submits_the_untouched_batch_with_its_counts_beside_it(
        self,
        tmp_path: Path,
        aws: tuple[FakeBedrock, FakeS3],
        pinned_decoupled_stimulus: DecoupledStimulus,
    ) -> None:
        bedrock, s3 = aws
        factory = CanaryFactory(always(ANSWERED))
        summary = decoupled_ladder.submit_batch(
            HAIKU_ANCHOR_LEG_ID, tmp_path, check_only=False, canary_backend_factory=factory
        )
        assert len(bedrock.created) == 1
        calls = planned_calls_for_leg(leg_for(HAIKU_ANCHOR_LEG_ID), pinned_decoupled_stimulus)
        cells = {call.cell for call in calls}
        canary = summary["refusal_canary"]
        assert canary["calls_per_group"] == REFUSAL_CANARY_CALLS
        assert set(canary["groups"]) == cells
        assert canary["calls"] == REFUSAL_CANARY_CALLS * len(cells)
        assert canary["refused"] == 0
        assert len(factory.prompts_seen) == canary["calls"]
        assert submitted_prompts(s3) == [call.prompt for call in calls], (
            "the batch content is untouched: same prompts, same order"
        )
        assert summary["resumed_handle"] is False

    def test_the_canary_reply_lands_in_no_file_and_no_s3_object(
        self,
        tmp_path: Path,
        aws: tuple[FakeBedrock, FakeS3],
        pinned_decoupled_stimulus: DecoupledStimulus,
    ) -> None:
        """SABOTAGE target: write the canary's completions anywhere and the sentinel turns up."""
        del pinned_decoupled_stimulus
        _, s3 = aws
        factory = CanaryFactory(always(ANSWERED))
        decoupled_ladder.submit_batch(
            HAIKU_ANCHOR_LEG_ID, tmp_path, check_only=False, canary_backend_factory=factory
        )
        assert factory.prompts_seen, "the canary did call, so its replies existed to leak"
        assert_sentinel_nowhere(tmp_path, s3)

    def test_a_leg_on_another_vendor_builds_no_canary_backend(
        self,
        tmp_path: Path,
        aws: tuple[FakeBedrock, FakeS3],
        pinned_decoupled_stimulus: DecoupledStimulus,
    ) -> None:
        del pinned_decoupled_stimulus
        bedrock, _ = aws
        factory = CanaryFactory(always(REFUSED))
        summary = decoupled_ladder.submit_batch(
            FLOOR_LEG_ID, tmp_path, check_only=False, canary_backend_factory=factory
        )
        assert factory.requested == []
        assert summary["refusal_canary"] is None
        assert len(bedrock.created) == 1

    def test_a_resumed_handle_runs_no_canary_and_carries_the_recorded_block_forward(
        self,
        tmp_path: Path,
        aws: tuple[FakeBedrock, FakeS3],
        pinned_decoupled_stimulus: DecoupledStimulus,
    ) -> None:
        """SABOTAGE target: rewrite the resumed submit's summary with ``refusal_canary: null``."""
        del aws, pinned_decoupled_stimulus
        first = decoupled_ladder.submit_batch(
            HAIKU_ANCHOR_LEG_ID,
            tmp_path,
            check_only=False,
            canary_backend_factory=CanaryFactory(always(ANSWERED)),
        )
        second = CanaryFactory(always(REFUSED))
        summary = decoupled_ladder.submit_batch(
            HAIKU_ANCHOR_LEG_ID, tmp_path, check_only=False, canary_backend_factory=second
        )
        assert second.requested == []
        assert summary["resumed_handle"] is True
        assert summary["refusal_canary"] == first["refusal_canary"]
        on_disk = json.loads(
            decoupled_ladder.submit_summary_path(tmp_path, leg_for(HAIKU_ANCHOR_LEG_ID)).read_text(
                encoding="utf-8"
            )
        )
        assert on_disk["refusal_canary"] == first["refusal_canary"]

    def test_check_only_spends_nothing_and_reports_the_canary_plan(
        self,
        tmp_path: Path,
        aws: tuple[FakeBedrock, FakeS3],
        pinned_decoupled_stimulus: DecoupledStimulus,
    ) -> None:
        del aws, pinned_decoupled_stimulus
        factory = CanaryFactory(always(REFUSED))
        summary = decoupled_ladder.submit_batch(
            HAIKU_ANCHOR_LEG_ID, tmp_path, check_only=True, canary_backend_factory=factory
        )
        assert factory.requested == []
        plan = summary["refusal_canary_plan"]
        assert plan["calls_per_group"] == REFUSAL_CANARY_CALLS
        assert plan["calls"] == REFUSAL_CANARY_CALLS * len(plan["groups"])

    def test_the_call_count_is_an_operator_knob_of_at_least_one(
        self,
        tmp_path: Path,
        aws: tuple[FakeBedrock, FakeS3],
        pinned_decoupled_stimulus: DecoupledStimulus,
    ) -> None:
        del aws, pinned_decoupled_stimulus
        factory = CanaryFactory(always(ANSWERED))
        summary = decoupled_ladder.submit_batch(
            HAIKU_ANCHOR_LEG_ID,
            tmp_path,
            check_only=False,
            canary_calls=1,
            canary_backend_factory=factory,
        )
        assert summary["refusal_canary"]["calls"] == len(summary["refusal_canary"]["groups"])
        with pytest.raises(SystemExit):
            decoupled_ladder._parse_args(
                ["submit-batch", "--leg", HAIKU_ANCHOR_LEG_ID, "--canary-calls", "0"]
            )
        args = decoupled_ladder._parse_args(["submit-batch", "--leg", HAIKU_ANCHOR_LEG_ID])
        assert args.canary_calls == REFUSAL_CANARY_CALLS


class TestTheStageOneSubmitRunsTheCanary:
    def test_a_refused_opus_canary_creates_no_job_after_probing_every_cell(
        self, stage_one_run: Path, aws: tuple[FakeBedrock, FakeS3]
    ) -> None:
        """SABOTAGE target: group by framing, and five of the seven cells are never sent."""
        bedrock, _ = aws
        factory = CanaryFactory(always(REFUSED))
        with pytest.raises(RefusalCanaryError, match="refusing to submit the opus batch job"):
            runner.submit_batch(
                runner.OPUS_MODEL_ID,
                stage_one_run,
                dry_run=False,
                canary_backend_factory=factory,
            )
        assert bedrock.created == []
        assert not runner._handle_path(stage_one_run, runner.OPUS_MODEL_ID).exists()
        evidence = json.loads(
            (stage_one_run / "summary-refusal-canary-opus.json").read_text(encoding="utf-8")
        )
        cells = stage_one_cells(runner.OPUS_MODEL_ID)
        assert len(cells) == 7, "Opus reads every stage-1 cell"
        assert set(evidence["groups"]) == cells
        assert set(factory.requested[0][:1]) == {runner.OPUS_MODEL_ID}
        assert len(factory.prompts_seen) == REFUSAL_CANARY_CALLS * len(cells)

    def test_an_answering_haiku_canary_submits_with_one_group_per_cell_labelled_by_framing(
        self, stage_one_run: Path, aws: tuple[FakeBedrock, FakeS3]
    ) -> None:
        bedrock, s3 = aws
        factory = CanaryFactory(always(ANSWERED))
        summary = runner.submit_batch(
            runner.HAIKU_MODEL_ID, stage_one_run, dry_run=False, canary_backend_factory=factory
        )
        assert len(bedrock.created) == 1
        cells = stage_one_cells(runner.HAIKU_MODEL_ID)
        canary = summary["refusal_canary"]
        assert set(canary["groups"]) == cells
        assert canary["cell_framings"] == {cell: CELLS_BY_NAME[cell].framing for cell in cells}
        assert canary["refused"] == 0
        assert_sentinel_nowhere(stage_one_run, s3)

    def test_the_dry_run_reports_the_plan_and_calls_nothing(
        self, stage_one_run: Path, aws: tuple[FakeBedrock, FakeS3]
    ) -> None:
        del aws
        factory = CanaryFactory(always(REFUSED))
        summary = runner.submit_batch(
            runner.OPUS_MODEL_ID, stage_one_run, dry_run=True, canary_backend_factory=factory
        )
        assert factory.requested == []
        assert set(summary["refusal_canary_plan"]["groups"]) == stage_one_cells(
            runner.OPUS_MODEL_ID
        )


class TestTheTransferSubmitRunsTheCanary:
    def test_a_refused_canary_creates_no_job_saves_no_handle_and_leaves_its_counts(
        self,
        tmp_path: Path,
        aws: tuple[FakeBedrock, FakeS3],
        pinned_transfer_stimulus: TransferStimulus,
    ) -> None:
        del pinned_transfer_stimulus
        bedrock, s3 = aws
        factory = CanaryFactory(always(REFUSED))
        with pytest.raises(RefusalCanaryError, match="refusing to submit leg"):
            transfer_cli.submit_batch(
                TRANSFER_HAIKU_FLOOR_LEG_ID,
                tmp_path,
                ONE_WAY_TRANSFER_PLAN,
                check_only=False,
                canary_backend_factory=factory,
            )
        leg = transfer_leg_for(TRANSFER_HAIKU_FLOOR_LEG_ID)
        assert bedrock.created == []
        assert s3.objects == {}
        assert not transfer_cli.handle_path(tmp_path, leg).exists()
        assert (tmp_path / f"summary-refusal-canary-{leg.file_stem}.json").exists()
        assert not transfer_cli.submit_summary_path(tmp_path, leg).exists()

    def test_an_answering_canary_submits_the_untouched_batch_with_its_counts_beside_it(
        self,
        tmp_path: Path,
        aws: tuple[FakeBedrock, FakeS3],
        pinned_transfer_stimulus: TransferStimulus,
    ) -> None:
        bedrock, s3 = aws
        factory = CanaryFactory(always(ANSWERED))
        summary = transfer_cli.submit_batch(
            TRANSFER_HAIKU_FLOOR_LEG_ID,
            tmp_path,
            ONE_WAY_TRANSFER_PLAN,
            check_only=False,
            canary_backend_factory=factory,
        )
        assert len(bedrock.created) == 1
        calls = transfer_calls_for_leg(
            transfer_leg_for(TRANSFER_HAIKU_FLOOR_LEG_ID), pinned_transfer_stimulus
        )
        cells = {call.cell for call in calls}
        canary = summary["refusal_canary"]
        assert set(canary["groups"]) == cells
        assert canary["calls"] == REFUSAL_CANARY_CALLS * len(cells)
        assert len(factory.prompts_seen) == canary["calls"]
        assert submitted_prompts(s3) == [call.prompt for call in calls]
        assert_sentinel_nowhere(tmp_path, s3)

    def test_a_leg_on_another_vendor_builds_no_canary_backend(
        self,
        tmp_path: Path,
        aws: tuple[FakeBedrock, FakeS3],
        pinned_transfer_stimulus: TransferStimulus,
    ) -> None:
        del pinned_transfer_stimulus
        bedrock, _ = aws
        factory = CanaryFactory(always(REFUSED))
        summary = transfer_cli.submit_batch(
            FLOOR_LEG_ID,
            tmp_path,
            ONE_WAY_TRANSFER_PLAN,
            check_only=False,
            canary_backend_factory=factory,
        )
        assert factory.requested == []
        assert summary["refusal_canary"] is None
        assert len(bedrock.created) == 1

    def test_a_resumed_handle_runs_no_canary_and_carries_the_recorded_block_forward(
        self,
        tmp_path: Path,
        aws: tuple[FakeBedrock, FakeS3],
        pinned_transfer_stimulus: TransferStimulus,
    ) -> None:
        del aws, pinned_transfer_stimulus
        first = transfer_cli.submit_batch(
            TRANSFER_HAIKU_FLOOR_LEG_ID,
            tmp_path,
            ONE_WAY_TRANSFER_PLAN,
            check_only=False,
            canary_backend_factory=CanaryFactory(always(ANSWERED)),
        )
        second = CanaryFactory(always(REFUSED))
        summary = transfer_cli.submit_batch(
            TRANSFER_HAIKU_FLOOR_LEG_ID,
            tmp_path,
            ONE_WAY_TRANSFER_PLAN,
            check_only=False,
            canary_backend_factory=second,
        )
        assert second.requested == []
        assert summary["resumed_handle"] is True
        assert summary["refusal_canary"] == first["refusal_canary"]

    def test_check_only_reports_the_plan_and_the_knob_parses(
        self,
        tmp_path: Path,
        aws: tuple[FakeBedrock, FakeS3],
        pinned_transfer_stimulus: TransferStimulus,
    ) -> None:
        del aws, pinned_transfer_stimulus
        factory = CanaryFactory(always(REFUSED))
        summary = transfer_cli.submit_batch(
            TRANSFER_HAIKU_FLOOR_LEG_ID,
            tmp_path,
            ONE_WAY_TRANSFER_PLAN,
            check_only=True,
            canary_backend_factory=factory,
        )
        assert factory.requested == []
        plan = summary["refusal_canary_plan"]
        assert plan["calls"] == REFUSAL_CANARY_CALLS * len(plan["groups"])
        args = transfer_cli._parse_args(
            [
                "--pass",
                ONE_WAY_TRANSFER_PLAN.pass_id,
                "submit-batch",
                "--leg",
                TRANSFER_HAIKU_FLOOR_LEG_ID,
                "--canary-calls",
                "2",
            ]
        )
        assert args.canary_calls == 2
        with pytest.raises(SystemExit):
            transfer_cli._parse_args(
                ["submit-batch", "--leg", TRANSFER_HAIKU_FLOOR_LEG_ID, "--canary-calls", "0"]
            )
