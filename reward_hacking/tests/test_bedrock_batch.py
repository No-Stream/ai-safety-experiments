"""Offline tests for the Bedrock batch-inference backend (``reward_hacking/bedrock_batch.py``).

Fully offline and credential-free, the same way ``test_bedrock_backend.py`` is: a fake ``boto3``
goes into ``sys.modules`` so the real ``BedrockBatchBackend.__init__`` runs unchanged against fake
``bedrock`` and ``s3`` clients. Nothing here reaches AWS, and the suite passes with ``boto3`` absent
from the venv (it is the optional ``bedrock`` extra).

Every check below is aimed at a failure that has actually happened on this path in this AWS account,
not at the happy path, and each is written so that removing the corresponding guard makes it fail:

* ``Completed`` with every record in error. A real Claude Haiku 4.5 job in this account reported
  status ``Completed``, ``successRecordCount`` 0 and ``errorRecordCount`` 500, every record dead on
  "`temperature` and `top_p` cannot both be specified". The manifest numbers here are that job's.
* Result-file line order is genuinely shuffled. ``SHUFFLED_IDS`` is the first five recordIds of a
  real 500-record output file, in the order they appeared.
* A content block in a batch result carries every member of the Converse union at once, with
  ``null`` for the ones that do not apply. ``REAL_BATCH_BLOCKS`` reproduces that exact key set,
  taken from a downloaded result record. The first parse of a real record crashed on it, so a
  hand-written two-key fake would test the wrong thing.
* Per-record ``error`` replaces ``modelOutput``. Turning that into an empty completion would grade
  as "the model did not make the move", reporting a transport failure as a behavioural finding.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import MISSING, asdict, fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from reward_hacking.bedrock_batch import (
    _FOUNDATION_MODEL_VENDORS,
    _INFERENCE_PROFILE_PREFIXES,
    BATCH_BUCKET_ENV,
    BATCH_BUCKET_OWNER_ENV,
    BATCH_PROFILE_ENV,
    BATCH_ROLE_ARN_ENV,
    BATCH_ROSTER,
    KNOWN_NOT_BATCH_CAPABLE,
    MAX_JOB_NAME_CHARS,
    MIN_BATCH_RECORDS,
    NOVA_MICRO_MAX_TOKENS,
    SAMPLING_LABEL_FIELDS,
    BatchJobHandle,
    BedrockBatchBackend,
    RosterModel,
    assert_roster_ids_well_formed,
    batch_bucket_owner,
    cell_digest,
    prompt_digest,
    roster_model,
    vendor_namespace,
)
from reward_hacking.model_backend import (
    DEFAULT_BEDROCK_MAX_TOKENS,
    Backend,
    BedrockSamplingConfig,
    _join_reasoning_blocks,
    _join_text_blocks,
    _sum_usage,
    converse_request,
)

NOVA_MICRO = "us.amazon.nova-micro-v1:0"
GPT_OSS_20B = "openai.gpt-oss-20b-1:0"
HAIKU = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
OPUS_5 = "global.anthropic.claude-opus-5"

# Dummy throughout: the real values are environment-only (see ``_batch_env``).
TEST_ACCOUNT = "000000000000"
TEST_ROLE_ARN = f"arn:aws:iam::{TEST_ACCOUNT}:role/BedrockBatchInferenceRole"
TEST_BUCKET = "batch-inference-test-bucket"
TEST_PROFILE = "test-batch-profile"

JOB_ARN = f"arn:aws:bedrock:us-west-2:{TEST_ACCOUNT}:model-invocation-job/abc123xyz789"

# Real key set, from a downloaded gpt-oss-120b batch result file.
_UNION_MEMBERS = (
    "audio",
    "cachePoint",
    "citationsContent",
    "document",
    "guardContent",
    "image",
    "reasoningContent",
    "searchResult",
    "text",
    "toolResult",
    "toolUse",
    "video",
)


def _batch_block(**populated: Any) -> dict[str, Any]:
    """Build a batch-shaped content block: every union member present, the named ones set."""
    return {member: populated.get(member) for member in _UNION_MEMBERS}


REAL_BATCH_BLOCKS = [
    _batch_block(reasoningContent={"reasoningText": {"signature": None, "text": "thinking hard"}}),
    _batch_block(text="the answer"),
]

# A real batch result's usage: both spellings of each cache counter, all null.
REAL_BATCH_USAGE: dict[str, Any] = {
    "serverToolUsage": {"webSearchRequests": None},
    "cacheReadInputTokens": None,
    "cacheWriteInputTokens": None,
    "cacheReadInputTokenCount": None,
    "cacheWriteInputTokenCount": None,
    "cacheDetails": None,
    "inputTokens": 419,
    "outputTokens": 377,
    "totalTokens": 796,
}

# The order the first five records came back in from a real 500-record job.
SHUFFLED_IDS = ("rec_000266", "rec_000144", "rec_000402", "rec_000127", "rec_000371")


def _model_output(text: str) -> dict[str, Any]:
    """A batch-shaped ``modelOutput`` whose answer block carries ``text``."""
    return {
        "output": {"message": {"content": [_batch_block(text=text)]}},
        "stopReason": "end_turn",
        "usage": dict(REAL_BATCH_USAGE),
    }


class _FakeS3:
    """An in-memory S3, keyed by ``s3://bucket/key``, recording every upload."""

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


class _FakeBedrock:
    """A fake ``bedrock`` control-plane client scripted with job states and record counts."""

    def __init__(
        self,
        *,
        statuses: list[str] | None = None,
        counts: dict[str, int] | None = None,
        message: str = "",
        model_active: bool = True,
    ) -> None:
        self.created: list[dict[str, Any]] = []
        self.describe_calls = 0
        self._statuses = statuses or ["Completed"]
        self._counts = counts
        self._message = message
        self._model_active = model_active

    def get_foundation_model(self, *, modelIdentifier: str) -> dict[str, Any]:  # noqa: N803
        status = "ACTIVE" if self._model_active else "LEGACY"
        return {"modelDetails": {"modelId": modelIdentifier, "modelLifecycle": {"status": status}}}

    def get_inference_profile(self, *, inferenceProfileIdentifier: str) -> dict[str, Any]:  # noqa: N803
        return {
            "inferenceProfileId": inferenceProfileIdentifier,
            "status": "ACTIVE" if self._model_active else "INACTIVE",
        }

    def create_model_invocation_job(self, **kwargs: Any) -> dict[str, Any]:
        self.created.append(kwargs)
        return {"jobArn": JOB_ARN}

    def get_model_invocation_job(self, *, jobIdentifier: str) -> dict[str, Any]:  # noqa: N803
        index = min(self.describe_calls, len(self._statuses) - 1)
        self.describe_calls += 1
        job: dict[str, Any] = {"jobArn": jobIdentifier, "status": self._statuses[index]}
        if self._message:
            job["message"] = self._message
        if self._counts is not None:
            job.update(self._counts)
        return job


def _batch_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Export dummy values for the four account-specific settings the backend reads from the env.

    The real account id, role ARN, bucket and profile are deliberately absent from this repository,
    so a test that wants a constructed backend has to supply its own. Setting them here rather than
    passing constructor keywords is what keeps the environment-reading path itself under test.
    """
    monkeypatch.setenv(BATCH_ROLE_ARN_ENV, TEST_ROLE_ARN)
    monkeypatch.setenv(BATCH_BUCKET_ENV, TEST_BUCKET)
    monkeypatch.setenv(BATCH_PROFILE_ENV, TEST_PROFILE)
    monkeypatch.delenv(BATCH_BUCKET_OWNER_ENV, raising=False)


def _build(
    monkeypatch: pytest.MonkeyPatch,
    model_id: str = GPT_OSS_20B,
    *,
    bedrock: _FakeBedrock | None = None,
    s3: _FakeS3 | None = None,
    **kwargs: Any,
) -> tuple[BedrockBatchBackend, _FakeBedrock, _FakeS3]:
    """Construct a real ``BedrockBatchBackend`` over fake AWS clients."""
    _batch_env(monkeypatch)
    fake_bedrock = bedrock or _FakeBedrock()
    fake_s3 = s3 or _FakeS3()

    def make_client(service_name: str) -> object:
        return fake_bedrock if service_name == "bedrock" else fake_s3

    def make_session(**_: Any) -> SimpleNamespace:
        return SimpleNamespace(client=make_client)

    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(Session=make_session))
    kwargs.setdefault("run_id", "testrun")
    backend = BedrockBatchBackend(model_id, **kwargs)
    return backend, fake_bedrock, fake_s3


def _prompts(count: int = MIN_BATCH_RECORDS) -> list[str]:
    return [f"prompt {index}" for index in range(count)]


def _within(model: RosterModel) -> BedrockSamplingConfig:
    """A sampling config whose cap fits under this model's ceiling, whatever the shared default is.

    Needed by any test that walks the whole roster: four rows cap output below
    ``DEFAULT_BEDROCK_MAX_TOKENS``, so a bare config raises the ceiling guard on them and the test
    would never reach what it meant to check.
    """
    return BedrockSamplingConfig(max_tokens=min(DEFAULT_BEDROCK_MAX_TOKENS, model.max_tokens_limit))


def _cell_metadata(count: int) -> list[dict[str, Any]]:
    return [
        {"item_id": f"item-{i}", "dimension": "verification", "arm": "spontaneous", "repeat": 0}
        for i in range(count)
    ]


def _handle(prompts: list[str], model_id: str = GPT_OSS_20B) -> BatchJobHandle:
    return BatchJobHandle(
        job_arn=JOB_ARN,
        job_name="jagged-test",
        model_id=model_id,
        record_count=len(prompts),
        prompt_digest=prompt_digest(prompts),
        cell_digest=cell_digest(_cell_metadata(len(prompts))),
        input_uri="s3://bucket/run/input.jsonl",
        output_uri="s3://bucket/run/output/",
        region="us-west-2",
        profile=TEST_PROFILE,
        submitted_at="2026-08-17T00:00:00+00:00",
    )


def _clean_manifest(count: int) -> dict[str, Any]:
    """A manifest claiming every record succeeded, for testing a file that disagrees with it."""
    return {
        "totalRecordCount": count,
        "processedRecordCount": count,
        "successRecordCount": count,
        "errorRecordCount": 0,
        "inputTokenCount": count * REAL_BATCH_USAGE["inputTokens"],
        "outputTokenCount": count * REAL_BATCH_USAGE["outputTokens"],
    }


def _write_results(  # noqa: PLR0913 - one knob per corruption the join is meant to catch
    s3: _FakeS3,
    handle: BatchJobHandle,
    prompts: list[str],
    *,
    order: list[int] | None = None,
    errors: dict[int, dict[str, Any]] | None = None,
    drop: set[int] | None = None,
    duplicate: int | None = None,
    manifest: dict[str, Any] | None = None,
    escape_non_ascii: bool = True,
) -> None:
    """Lay down a result file and manifest for ``handle``, with the requested corruptions.

    ``escape_non_ascii=False`` writes the file the way Bedrock does, with non-ASCII characters raw
    inside JSON strings rather than ``\\uXXXX``-escaped; that is what puts a Unicode line separator
    on a record's line.
    """
    indices = order if order is not None else list(range(len(prompts)))
    lines: list[str] = []
    for index in indices:
        if drop and index in drop:
            continue
        record: dict[str, Any] = {
            "recordId": f"rec_{index:06d}",
            "modelInput": converse_request(
                prompts[index], {"maxTokens": DEFAULT_BEDROCK_MAX_TOKENS}
            ),
        }
        if errors and index in errors:
            record["error"] = errors[index]
        else:
            record["modelOutput"] = _model_output(f"answer {index}")
        lines.append(json.dumps(record, ensure_ascii=escape_non_ascii))
    if duplicate is not None:
        lines.append(lines[duplicate])

    success = len(prompts) - len(errors or {}) - len(drop or set())
    s3.objects["s3://bucket/run/output/abc123xyz789/input.jsonl.out"] = "".join(
        line + "\n" for line in lines
    )
    s3.objects["s3://bucket/run/output/abc123xyz789/manifest.json.out"] = json.dumps(
        manifest
        if manifest is not None
        else {
            "totalRecordCount": len(prompts),
            "processedRecordCount": len(prompts),
            "successRecordCount": success,
            "errorRecordCount": len(errors or {}),
            "inputTokenCount": success * REAL_BATCH_USAGE["inputTokens"],
            "outputTokenCount": success * REAL_BATCH_USAGE["outputTokens"],
        }
    )


class TestSharedParsersOnRealBatchShape:
    """The batch result's fully-populated union crashed the parsers written for the live path."""

    def test_null_members_are_not_read_as_answer_text(self) -> None:
        assert _join_text_blocks(REAL_BATCH_BLOCKS) == "the answer"

    def test_null_reasoning_content_does_not_crash_the_reasoning_join(self) -> None:
        assert _join_reasoning_blocks(REAL_BATCH_BLOCKS) == "thinking hard"

    def test_reasoning_is_never_folded_into_the_answer(self) -> None:
        """The whole reason Converse is used instead of the native schema."""
        assert "thinking hard" not in _join_text_blocks(REAL_BATCH_BLOCKS)

    def test_null_cache_counters_do_not_break_the_input_sum(self) -> None:
        usage = _sum_usage(REAL_BATCH_USAGE)
        assert (usage.input_tokens, usage.output_tokens) == (419, 377)

    def test_redacted_reasoning_yields_no_reasoning_rather_than_crashing(self) -> None:
        """Luna returns ``redactedContent`` with ``reasoningText`` present and null."""
        blocks = [_batch_block(reasoningContent={"redactedContent": "b64blob"})]
        assert _join_reasoning_blocks(blocks) == ""

    def test_populated_cache_counters_are_added_to_the_input_total(self) -> None:
        usage = _sum_usage({"inputTokens": 2, "cacheWriteInputTokens": 1619, "outputTokens": 7})
        assert usage.input_tokens == 1621

    def test_alias_spellings_are_not_double_counted(self) -> None:
        """Both spellings are one quantity; summing them would inflate the input-token cost."""
        usage = _sum_usage(
            {
                "inputTokens": 10,
                "cacheReadInputTokens": 100,
                "cacheReadInputTokenCount": 100,
                "outputTokens": 1,
            }
        )
        assert usage.input_tokens == 110

    def test_disagreeing_alias_spellings_are_reported(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The tripwire on the assumption that the two spellings are aliases."""
        with caplog.at_level(logging.WARNING):
            _sum_usage({"cacheReadInputTokens": 100, "cacheReadInputTokenCount": 250})
        assert "disagree" in caplog.text


class TestRoster:
    """A model id outside the verified table is refused rather than guessed at."""

    def test_every_roster_id_resolves(self) -> None:
        for model in BATCH_ROSTER:
            assert roster_model(model.model_id) is model

    def test_unknown_model_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not in the verified batch roster"):
            roster_model("openai.gpt-5.6-luna")

    @pytest.mark.parametrize("model_id", KNOWN_NOT_BATCH_CAPABLE)
    def test_a_model_the_api_answered_no_about_stays_off_the_roster(self, model_id: str) -> None:
        """Each of these was asked and answered, so re-adding one has to argue with the answer.

        ``CreateModelInvocationJob`` rejected them with "Batch inference is not supported for the
        requested model.", alongside a control that was accepted in the same run. The failure this
        guards is a quiet one: they are the most capable models on the account, so the reason to
        reach for them recurs, and the roster would accept a row for one without complaint. A
        submitted job would then queue and come back rejected, having spent the S3 upload first.
        """
        assert model_id not in {model.model_id for model in BATCH_ROSTER}
        with pytest.raises(ValueError, match="not in the verified batch roster"):
            roster_model(model_id)

    def test_the_probe_confirmed_answers_are_still_listed(self) -> None:
        """Otherwise the check above is neutered by deleting a line rather than by a new probe.

        A parametrized test over a tuple gets quieter, not redder, when the tuple shrinks: drop an
        entry and the case for it simply stops existing. These three are the ones an actual
        ``CreateModelInvocationJob`` call rejected, so they are the ones worth pinning by name.
        """
        for probed in (
            "global.anthropic.claude-sonnet-5",
            "global.anthropic.claude-fable-5",
            "openai.gpt-5.6-sol",
        ):
            assert probed in KNOWN_NOT_BATCH_CAPABLE

    def test_a_bare_id_that_needs_a_prefix_is_not_silently_accepted(self) -> None:
        """The bare Nova Micro id is INFERENCE_PROFILE-only in us-west-2 and must not be usable."""
        with pytest.raises(ValueError, match="not in the verified batch roster"):
            roster_model("amazon.nova-micro-v1:0")

    def test_prefixed_ids_are_flagged_as_inference_profiles(self) -> None:
        assert roster_model(NOVA_MICRO).is_inference_profile
        assert not roster_model(GPT_OSS_20B).is_inference_profile

    def test_every_price_on_the_roster_was_read_rather_than_reasoned_to(self) -> None:
        """Including Anthropic's, once its rates were found under the Marketplace service code.

        Claude Haiku 4.5 carried a halved list price for as long as the search looked only under the
        ``AmazonBedrock`` service code, where no Claude usagetype newer than Claude 3 exists. The
        rates live under ``AmazonBedrockFoundationModels`` instead, because Claude bills through AWS
        Marketplace, and reading them there agreed with the halved figure exactly.
        """
        unverified = [model.model_id for model in BATCH_ROSTER if not model.batch_price_verified]
        assert unverified == []

    def test_an_unverified_price_still_annotates_the_cost_line(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The mechanism outlives the row that needed it, which is why the flag stays.

        With every row verified, nothing in the table exercises the false branch any more, so this
        drives it from a synthetic row. Without this the annotation could rot unnoticed and the next
        model added on a reasoned-to price would report its cost as though it had been measured.
        """
        backend, _, s3 = _build(monkeypatch)
        backend.roster = replace(backend.roster, batch_price_verified=False)
        prompts = _prompts()
        handle = _handle(prompts)
        _write_results(s3, handle, prompts)
        with caplog.at_level(logging.INFO):
            backend.collect(handle)
        assert "PRICE UNVERIFIED" in caplog.text

    def test_every_roster_model_has_an_output_ceiling(self) -> None:
        """A ceiling is required rather than optional, and the reason is Llama 4 Scout.

        While it was optional, seven of eight rows left it unset to mean "no documented ceiling".
        Reading the real ceilings off the API found both Llama 4 models capped at 8,192, below the
        shared Converse default of 30,000, so a roster-wide sweep at the default was submitting jobs
        whose every record could only come back with a 400.
        """
        assert all(model.max_tokens_limit > 0 for model in BATCH_ROSTER)

    def test_the_real_roster_passes_its_own_id_check(self) -> None:
        """The import-time guard, called deliberately rather than relied on as a side effect."""
        assert_roster_ids_well_formed(BATCH_ROSTER)

    def test_every_roster_id_names_a_vendor_bedrock_has(self) -> None:
        """Pins the ids well-formed independently of the guard's own bookkeeping."""
        assert {vendor_namespace(model.model_id) for model in BATCH_ROSTER} <= set(
            _FOUNDATION_MODEL_VENDORS
        )

    def test_a_prefix_missing_from_the_allowlist_is_refused(self) -> None:
        """``ap.`` for ``apac.``: a real typo, and silent without this check.

        The id reads as a foundation model because no allowlisted prefix matches, so the submit
        would ask ``get_foundation_model`` about something that exists only as a profile and die at a
        call unrelated to the typo.
        """
        mistyped = replace(roster_model(HAIKU), model_id="ap.anthropic.claude-opus-5")
        assert not mistyped.is_inference_profile
        with pytest.raises(RuntimeError, match="naming no known vendor"):
            assert_roster_ids_well_formed([mistyped])

    def test_a_prefixed_id_with_the_vendor_dropped_is_refused(self) -> None:
        """This one *does* read as a profile, so only the vendor segment gives it away."""
        vendorless = replace(roster_model(HAIKU), model_id="us.claude-opus-5")
        assert vendorless.is_inference_profile
        with pytest.raises(RuntimeError, match="naming no known vendor"):
            assert_roster_ids_well_formed([vendorless])

    def test_a_duplicated_id_is_refused(self) -> None:
        """``_ROSTER_BY_ID`` would silently keep the last row and orphan the first."""
        model = roster_model(HAIKU)
        with pytest.raises(RuntimeError, match="duplicated ids"):
            assert_roster_ids_well_formed([model, replace(model, batch_price_in_per_mtok=99.0)])

    def test_the_geo_prefixes_the_model_cards_advertise_are_all_recognised(self) -> None:
        """``au.`` and ``jp.`` were missing, and both appear on cards for models we use."""
        for advertised in ("us.", "global.", "eu.", "apac.", "au.", "jp."):
            assert advertised in _INFERENCE_PROFILE_PREFIXES


class TestAccountSettingsComeFromTheEnvironment:
    """The account, role, bucket and profile are environment-only, and unset must be loud.

    This repository is public, so none of those four values can be committed. The failure that
    matters is the quiet one: an unset variable defaulting to an empty string, boto3 accepting it,
    and the submit uploading the input file before the create call dies -- paid for, with nothing to
    show. Each check below removes one variable and requires the error to name it.
    """

    @pytest.mark.parametrize(
        "variable",
        [BATCH_PROFILE_ENV, BATCH_ROLE_ARN_ENV, BATCH_BUCKET_ENV],
    )
    def test_an_unset_variable_is_named_in_the_failure(
        self, monkeypatch: pytest.MonkeyPatch, variable: str
    ) -> None:
        _batch_env(monkeypatch)
        monkeypatch.delenv(variable)
        with pytest.raises(RuntimeError, match=variable):
            BedrockBatchBackend(GPT_OSS_20B)

    def test_a_blank_variable_is_treated_as_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An exported-but-empty variable is the likelier mistake than a missing one."""
        _batch_env(monkeypatch)
        monkeypatch.setenv(BATCH_BUCKET_ENV, "   ")
        with pytest.raises(RuntimeError, match=BATCH_BUCKET_ENV):
            BedrockBatchBackend(GPT_OSS_20B)

    def test_an_override_beats_the_account_in_the_role_arn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(BATCH_BUCKET_OWNER_ENV, "111122223333")
        assert batch_bucket_owner(TEST_ROLE_ARN) == "111122223333"

    def test_an_arn_without_a_readable_account_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Silently deriving a wrong ExpectedBucketOwner would fail at job creation, post-upload."""
        monkeypatch.delenv(BATCH_BUCKET_OWNER_ENV, raising=False)
        with pytest.raises(RuntimeError, match=BATCH_BUCKET_OWNER_ENV):
            batch_bucket_owner("arn:aws:iam::not-an-account:role/BedrockBatchInferenceRole")


class TestSubmitPreflight:
    """Everything cheap enough to check before a job exists is checked before a job exists."""

    def test_below_the_hard_floor_is_refused_with_the_arithmetic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend, bedrock, _ = _build(monkeypatch)
        with pytest.raises(ValueError, match="at least 100 records"):
            backend.submit(_prompts(35))
        assert bedrock.created == [], "no job may be created for a run that cannot run"

    def test_the_refusal_says_which_repeat_count_would_clear_the_floor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend, _, _ = _build(monkeypatch)
        with pytest.raises(ValueError, match=r"multiply --repeats by 3"):
            backend.submit(_prompts(35))

    def test_the_refusal_frames_the_number_as_a_multiplier_not_an_absolute(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """It sees only a record count, so a run already at --repeats 3 needs 3x this, not this."""
        backend, _, _ = _build(monkeypatch)
        with pytest.raises(ValueError, match=r"multiply --repeats by 4 \(taking this run to 120"):
            backend.submit(_prompts(30))

    def test_an_empty_run_says_so_rather_than_dividing_by_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--limit 0`` renders no prompts, and no repeat count multiplies zero up to 100."""
        backend, _, _ = _build(monkeypatch)
        with pytest.raises(ValueError, match="renders no prompts at all"):
            backend.submit([])

    def test_the_real_corpus_size_clears_the_floor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """29 items x 5 arms = 145 records at one repeat, which is why batch is usable at all."""
        backend, bedrock, _ = _build(monkeypatch)
        backend.submit(_prompts(145))
        assert len(bedrock.created) == 1

    def test_anthropic_temperature_and_top_p_together_is_refused_at_construction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The combination that returned Completed with 500 of 500 records failed."""
        with pytest.raises(ValueError, match="rejects temperature and topP together"):
            _build(
                monkeypatch,
                HAIKU,
                sampling=BedrockSamplingConfig(temperature=0.7, top_p=0.9),
            )

    def test_a_max_tokens_above_the_models_ceiling_is_refused_at_construction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same archetype as the temperature conflict, one step ahead of it happening.

        ``converse_inference_config`` writes ``maxTokens`` unconditionally and the sweep feeds one
        flat ``--max-new-tokens`` across a whole roster, so a roster-wide 24,576 submits a job Nova
        rejects per record -- reported ``Completed``, every record errored, queue time burned.
        """
        with pytest.raises(ValueError, match="hard-rejects maxTokens above 10000"):
            _build(
                monkeypatch,
                NOVA_MICRO,
                sampling=BedrockSamplingConfig(max_tokens=24_576),
            )

    def test_a_max_tokens_at_the_ceiling_is_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend, _, _ = _build(
            monkeypatch,
            "us.amazon.nova-micro-v1:0",
            sampling=BedrockSamplingConfig(max_tokens=NOVA_MICRO_MAX_TOKENS),
        )
        assert backend.sampling.max_tokens == NOVA_MICRO_MAX_TOKENS

    def test_a_cap_below_the_models_ceiling_is_left_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Haiku 4.5's ceiling is 64,000, so a cap under it passes and one over it does not."""
        backend, _, _ = _build(
            monkeypatch, HAIKU, sampling=BedrockSamplingConfig(max_tokens=63_999)
        )
        assert backend.roster.max_tokens_limit == 64_000

    def test_a_high_ceiling_is_still_a_ceiling(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The same guard on a generous limit, so "high" is not quietly treated as "absent"."""
        with pytest.raises(ValueError, match="hard-rejects maxTokens above 64000"):
            _build(monkeypatch, HAIKU, sampling=BedrockSamplingConfig(max_tokens=100_000))

    def test_opus_5_refuses_a_temperature_set_on_its_own(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The pair-only rule would wave this through, and the job would come back all-dead.

        Claude Opus 5 answers a request carrying either knob with "`temperature` is deprecated for
        this model", so the refusal it needs is stricter than Haiku 4.5's. Verified live on the
        Converse path, which shares request validation with the batch path.
        """
        with pytest.raises(ValueError, match="deprecates both temperature and topP"):
            _build(monkeypatch, OPUS_5, sampling=BedrockSamplingConfig(temperature=0.5))

    def test_opus_5_refuses_a_top_p_set_on_its_own(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Both knobs, not just the one whose error message got read first."""
        with pytest.raises(ValueError, match="deprecates both temperature and topP"):
            _build(monkeypatch, OPUS_5, sampling=BedrockSamplingConfig(top_p=0.9))

    def test_opus_5_with_neither_knob_set_is_fine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Omission is the default, so the frontier row is usable without opting out of anything."""
        backend, _, _ = _build(
            monkeypatch, OPUS_5, sampling=BedrockSamplingConfig(max_tokens=30_000)
        )
        assert backend.sampling.temperature is None
        assert backend.sampling.top_p is None

    def test_either_one_alone_is_fine_on_anthropic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend, _, _ = _build(monkeypatch, HAIKU, sampling=BedrockSamplingConfig(temperature=0.7))
        assert backend.model_id == HAIKU

    def test_a_model_that_does_not_resolve_active_stops_the_submit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The cap is passed explicitly, and within Nova's ceiling, to reach the check under test.

        The shared Converse default is above Nova's 10,000-token limit on purpose, so a bare
        sampling config raises the ceiling error two tests up and this one would pass without ever
        consulting whether the model resolved ACTIVE.
        """
        backend, bedrock, _ = _build(
            monkeypatch,
            NOVA_MICRO,
            bedrock=_FakeBedrock(model_active=False),
            sampling=BedrockSamplingConfig(max_tokens=NOVA_MICRO_MAX_TOKENS),
        )
        with pytest.raises(RuntimeError, match="not ACTIVE"):
            backend.submit(_prompts())
        assert bedrock.created == []

    def test_metadata_length_mismatch_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Metadata is positional, so a wrong length would label the wrong cells."""
        backend, _, _ = _build(monkeypatch)
        with pytest.raises(ValueError, match="metadata has 2 entries"):
            backend.submit(_prompts(), metadata=[{"item_id": "a"}, {"item_id": "b"}])


class TestSubmitRequestShape:
    """What lands in S3 and in the job parameters, checked field by field."""

    def test_the_job_asks_for_converse_not_the_default_invoke_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The default is InvokeModel, under which every Converse-shaped record errors."""
        backend, bedrock, _ = _build(monkeypatch)
        backend.submit(_prompts())
        assert bedrock.created[0]["modelInvocationType"] == "Converse"

    def test_the_job_name_clears_bedrocks_ten_character_minimum(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend, bedrock, _ = _build(monkeypatch)
        backend.submit(_prompts())
        assert len(bedrock.created[0]["jobName"]) >= 10

    def test_the_longest_roster_id_keeps_its_run_id_inside_the_name_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Truncating from the right drops the run id, so two runs the same day would collide."""
        backend, bedrock, _ = _build(monkeypatch, HAIKU, run_id="20260817T002716Z")
        backend.submit(_prompts())
        name = bedrock.created[0]["jobName"]
        assert len(name) <= MAX_JOB_NAME_CHARS
        assert name.endswith("20260817t002716z")

    def test_no_two_roster_models_share_a_job_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Truncation is what makes this reachable, and the wider roster is what made it a risk.

        Two ids are long enough to lose their tails now, Haiku 4.5 and Llama 4 Maverick, where one
        was before. A third sharing a 46-character prefix with either would have both models ask
        Bedrock for one job name in the same sweep, and since this path exists to submit every model
        at once, the second submit is the one that breaks, having already paid for its S3 upload.
        This walks the real roster rather than a constructed pair, since uniqueness is a claim about
        the twenty-three rows actually there.
        """
        names: dict[str, str] = {}
        for model in BATCH_ROSTER:
            backend, bedrock, _ = _build(
                monkeypatch, model.model_id, run_id="20260831T054500Z", sampling=_within(model)
            )
            backend.submit(_prompts())
            name = bedrock.created[0]["jobName"]
            assert name not in names, f"{model.model_id} collides with {names.get(name)} on {name}"
            names[name] = model.model_id
        assert len(names) == len(BATCH_ROSTER)

    def test_the_bucket_owner_travels_with_the_bucket(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A custom bucket in another account must not upload fine and then fail job creation."""
        backend, bedrock, _ = _build(
            monkeypatch, bucket="other-bucket", bucket_owner="111122223333"
        )
        backend.submit(_prompts())
        created = bedrock.created[0]
        assert created["inputDataConfig"]["s3InputDataConfig"]["s3BucketOwner"] == "111122223333"
        assert created["outputDataConfig"]["s3OutputDataConfig"]["s3BucketOwner"] == "111122223333"

    def test_the_bucket_owner_defaults_to_the_service_roles_account(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no override, the account is read off the role ARN rather than left unset."""
        backend, bedrock, _ = _build(monkeypatch)
        backend.submit(_prompts())
        created = bedrock.created[0]
        assert created["inputDataConfig"]["s3InputDataConfig"]["s3BucketOwner"] == TEST_ACCOUNT

    def test_the_timeout_is_the_api_minimum(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """24 hours is the floor; a shorter deadline cannot be requested."""
        backend, bedrock, _ = _build(monkeypatch)
        backend.submit(_prompts())
        assert bedrock.created[0]["timeoutDurationInHours"] == 24

    def test_records_carry_the_same_body_the_live_backend_would_send(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One renderer for both transports, or a drift reads as a model difference."""
        backend, _, s3 = _build(monkeypatch)
        prompts = _prompts()
        backend.submit(prompts)
        first = json.loads(s3.objects[backend._s3_base() + "/input.jsonl"].splitlines()[0])
        assert first["recordId"] == "rec_000000"
        assert first["modelInput"] == converse_request(
            prompts[0], {"maxTokens": DEFAULT_BEDROCK_MAX_TOKENS}
        )
        assert "modelId" not in first["modelInput"]

    def test_reasoning_effort_reaches_the_record_in_the_family_dialect(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend, _, s3 = _build(
            monkeypatch,
            GPT_OSS_20B,
            sampling=BedrockSamplingConfig(reasoning_effort="low"),
        )
        backend.submit(_prompts())
        first = json.loads(s3.objects[backend._s3_base() + "/input.jsonl"].splitlines()[0])
        assert first["modelInput"]["additionalModelRequestFields"] == {"reasoning_effort": "low"}

    def test_an_effort_level_the_family_rejects_fails_before_any_upload(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with pytest.raises(ValueError, match="not accepted by this model family"):
            _build(
                monkeypatch,
                GPT_OSS_20B,
                sampling=BedrockSamplingConfig(reasoning_effort="bogus"),
            )

    def test_the_sidecar_describes_the_run_without_the_local_trace(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend, _, s3 = _build(monkeypatch)
        prompts = _prompts()
        metadata = _cell_metadata(len(prompts))
        backend.submit(prompts, metadata=metadata)
        sidecar = json.loads(s3.objects[backend._s3_base() + "/sidecar.json"])
        assert sidecar["prompt_digest"] == prompt_digest(prompts)
        assert sidecar["cell_digest"] == cell_digest(metadata)
        assert sidecar["cells"][0] == {
            "record_id": "rec_000000",
            "item_id": "item-0",
            "dimension": "verification",
            "arm": "spontaneous",
            "repeat": 0,
        }


class TestSubmitIdempotency:
    """A re-submit of the same run must be recognised by the service, not billed a second time.

    ``clientRequestToken`` is a real idempotency token in the ``CreateModelInvocationJob`` shape,
    and botocore's auto-injected UUID only covers its own retries -- an operator re-run, or a re-run
    after a connection dropped between the service accepting the job and the handle being written,
    pays for the inference twice. That is the exact outcome ``BatchJobHandle``'s docstring says must
    not happen, and ``jagged/sweep.py`` documents ``--run-id`` reuse as the recovery route for it.
    """

    def test_the_same_run_id_and_prompts_carry_one_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Re-running the sweep's documented recovery path must not mint a second billable job."""
        prompts = _prompts()
        bedrock = _FakeBedrock()
        first, _, _ = _build(monkeypatch, bedrock=bedrock)
        first.submit(prompts)
        second, _, _ = _build(monkeypatch, bedrock=bedrock)
        second.submit(prompts)
        assert bedrock.created[0]["clientRequestToken"] == bedrock.created[1]["clientRequestToken"]

    def test_a_fresh_run_id_is_a_genuinely_new_job(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two deliberate draws of the same corpus must not collapse into one job.

        The token is a digest of the job name AND the prompt content together -- not the job name
        *instead of* the content, which is what this docstring said until 2026-08-24 and which would
        mean a reused --run-id over an edited corpus hands back the old job's completions (the
        sibling test below pins that it does not). The name is in the digest so that a second
        deliberate sample of the SAME corpus at temperature > 0 is a new measurement rather than a
        cache hit, which would be a corrupted result rather than a saved few dollars.
        """
        prompts = _prompts()
        bedrock = _FakeBedrock()
        first, _, _ = _build(monkeypatch, bedrock=bedrock, run_id="firstrun")
        first.submit(prompts)
        second, _, _ = _build(monkeypatch, bedrock=bedrock, run_id="secondrun")
        second.submit(prompts)
        assert bedrock.created[0]["clientRequestToken"] != bedrock.created[1]["clientRequestToken"]

    def test_the_same_run_id_with_other_prompts_is_a_new_job(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Or a reused --run-id over an edited corpus returns a job built from other prompts."""
        bedrock = _FakeBedrock()
        first, _, _ = _build(monkeypatch, bedrock=bedrock)
        first.submit(_prompts())
        second, _, _ = _build(monkeypatch, bedrock=bedrock)
        second.submit([f"a different prompt {index}" for index in range(MIN_BATCH_RECORDS)])
        assert bedrock.created[0]["clientRequestToken"] != bedrock.created[1]["clientRequestToken"]

    def test_the_token_is_in_the_shape_the_api_accepts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The shape allows [a-zA-Z0-9]{1,256}; a rejected token fails after the S3 upload."""
        backend, bedrock, _ = _build(monkeypatch)
        backend.submit(_prompts())
        token = bedrock.created[0]["clientRequestToken"]
        assert token.isalnum()
        assert 1 <= len(token) <= 256


class TestCollectValidation:
    """Everything that a returned result file can be wrong about, and is refused for."""

    def test_the_happy_path_returns_completions_in_submitted_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend, _, s3 = _build(monkeypatch)
        prompts = _prompts()
        handle = _handle(prompts)
        _write_results(s3, handle, prompts)
        completions = backend.collect(handle)
        assert [c.text for c in completions] == [f"answer {i}" for i in range(len(prompts))]

    def test_the_batch_path_inherits_the_stop_reason(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A result file's ``modelOutput`` is a Converse body, so it carries ``stopReason`` too.

        Worth asserting rather than assuming: truncation labelling is what keeps a token cap from
        reading as non-compliance, and it would be silently absent on this transport if the shared
        parser had been widened for the live path alone.
        """
        backend, _, s3 = _build(monkeypatch)
        prompts = _prompts()
        handle = _handle(prompts)
        _write_results(s3, handle, prompts)
        completions = backend.collect(handle)
        assert {c.stop_reason for c in completions} == {"end_turn"}

    def test_a_unicode_line_separator_inside_a_string_does_not_split_a_record(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Bedrock leaves U+0085 and U+2028 raw inside JSON strings; ``str.splitlines`` breaks on them.

        The first TMAX judge job came back with eight U+0085s in 1,200 records (terminal output
        quoted in the prompts) and every record they sat in became two unparseable fragments. The
        join has to split on the newline alone.
        """
        backend, _, s3 = _build(monkeypatch)
        prompts = _prompts()
        prompts[3] = "terminal output with a NEL \x85 and a line separator \u2028 inside"
        handle = _handle(prompts)
        _write_results(s3, handle, prompts, escape_non_ascii=False)
        completions = backend.collect(handle)
        assert [c.text for c in completions] == [f"answer {i}" for i in range(len(prompts))]

    def test_a_shuffled_result_file_still_joins_correctly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Result order is genuinely not input order, so the join must be on recordId."""
        backend, _, s3 = _build(monkeypatch)
        prompts = _prompts()
        handle = _handle(prompts)
        shuffled = [
            int(record_id.removeprefix("rec_")) % len(prompts) for record_id in SHUFFLED_IDS
        ]
        order = shuffled + [i for i in range(len(prompts)) if i not in shuffled]
        _write_results(s3, handle, prompts, order=order)
        completions = backend.collect(handle)
        assert [c.text for c in completions] == [f"answer {i}" for i in range(len(prompts))]

    def test_a_short_manifest_is_caught_before_the_file_is_even_read(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend, _, s3 = _build(monkeypatch)
        prompts = _prompts()
        handle = _handle(prompts)
        _write_results(s3, handle, prompts, drop={7})
        with pytest.raises(RuntimeError, match="status Completed but only 99"):
            backend.collect(handle)

    def test_a_missing_record_is_refused_even_when_the_manifest_claims_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The join is the second line of defence, for a manifest that does not reflect the file.

        The manifest check above catches the ordinary case. This one hands over a manifest claiming
        all 100 records succeeded while the file holds 99, which is what a truncated upload or a
        stale manifest would look like, and the record-set assertion has to catch it on its own.
        """
        backend, _, s3 = _build(monkeypatch)
        prompts = _prompts()
        handle = _handle(prompts)
        _write_results(s3, handle, prompts, drop={7}, manifest=_clean_manifest(len(prompts)))
        with pytest.raises(RuntimeError, match="different record set"):
            backend.collect(handle)

    def test_a_duplicated_record_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A duplicate means some submitted prompt has no result of its own."""
        backend, _, s3 = _build(monkeypatch)
        prompts = _prompts()
        handle = _handle(prompts)
        _write_results(s3, handle, prompts, duplicate=3)
        with pytest.raises(RuntimeError, match="more than once"):
            backend.collect(handle)

    def test_a_per_record_error_never_becomes_an_empty_completion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty completion grades as "did not move": a claim about a transport bug.

        The manifest normally counts the error first, so this hands over a clean manifest to prove
        the per-record check stands on its own -- and asserts the model's own error text reaches the
        exception, because a count alone does not say what went wrong.
        """
        backend, _, s3 = _build(monkeypatch)
        prompts = _prompts()
        handle = _handle(prompts)
        _write_results(
            s3,
            handle,
            prompts,
            errors={
                4: {
                    "errorCode": 400,
                    "errorMessage": "`temperature` and `top_p` cannot both be specified",
                    "expired": False,
                    "retryable": False,
                }
            },
            manifest=_clean_manifest(len(prompts)),
        )
        with pytest.raises(RuntimeError, match="cannot both be specified"):
            backend.collect(handle)

    def test_completed_with_zero_successes_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The real Haiku 4.5 job's manifest: Completed, 0 of 500 succeeded."""
        backend, _, s3 = _build(monkeypatch)
        prompts = _prompts()
        handle = _handle(prompts)
        _write_results(
            s3,
            handle,
            prompts,
            manifest={
                "totalRecordCount": len(prompts),
                "processedRecordCount": len(prompts),
                "successRecordCount": 0,
                "errorRecordCount": len(prompts),
                "inputTokenCount": 0,
                "outputTokenCount": 0,
            },
        )
        with pytest.raises(RuntimeError, match="status Completed but only 0"):
            backend.collect(handle)

    def test_job_counts_disagreeing_with_the_manifest_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two sources for the same number; if they differ, neither is trustworthy."""
        prompts = _prompts()
        backend, _, s3 = _build(
            monkeypatch,
            bedrock=_FakeBedrock(counts={"successRecordCount": 42}),
        )
        handle = _handle(prompts)
        _write_results(s3, handle, prompts)
        with pytest.raises(RuntimeError, match="successRecordCount=42"):
            backend.collect(handle)

    def test_a_failed_job_reports_its_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        prompts = _prompts()
        backend, _, s3 = _build(
            monkeypatch,
            bedrock=_FakeBedrock(statuses=["Failed"], message="access denied on the input bucket"),
        )
        handle = _handle(prompts)
        _write_results(s3, handle, prompts)
        with pytest.raises(RuntimeError, match="access denied on the input bucket"):
            backend.collect(handle)

    def test_results_from_a_different_submission_are_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The digest tripwire: right record ids, wrong prompts."""
        backend, _, s3 = _build(monkeypatch)
        prompts = _prompts()
        handle = _handle(prompts)
        _write_results(s3, handle, [f"a different prompt {i}" for i in range(len(prompts))])
        with pytest.raises(RuntimeError, match="do not belong to this handle"):
            backend.collect(handle)

    def test_waiting_polls_through_the_non_terminal_states(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prompts = _prompts()
        backend, bedrock, s3 = _build(
            monkeypatch,
            bedrock=_FakeBedrock(statuses=["Submitted", "Scheduled", "InProgress", "Completed"]),
        )
        handle = _handle(prompts)
        _write_results(s3, handle, prompts)
        backend.collect(handle, poll_seconds=0.0)
        assert bedrock.describe_calls == 4

    def test_a_handle_for_another_model_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Everything else in collect derives from the handle; the model id is read off the backend.

        So a handle loaded into the wrong backend -- realistic when one interactive session collects
        several saved handles -- prices the run off the wrong roster row, carries the wrong
        PRICE UNVERIFIED annotation, and attributes the responses to a model that never ran them.
        """
        backend, _, s3 = _build(monkeypatch, GPT_OSS_20B)
        prompts = _prompts()
        handle = _handle(prompts, model_id=NOVA_MICRO)
        _write_results(s3, handle, prompts)
        with pytest.raises(ValueError, match="wrong model"):
            backend.collect(handle)

    def test_a_wait_timeout_says_the_job_is_still_collectable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Losing patience must not mean losing the inference."""
        backend, _, _ = _build(monkeypatch, bedrock=_FakeBedrock(statuses=["InProgress"]))
        with pytest.raises(TimeoutError, match="still collectable"):
            backend.wait(_handle(_prompts()), poll_seconds=0.0, timeout_seconds=-1.0)

    def test_token_usage_accumulates_over_the_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend, _, s3 = _build(monkeypatch)
        prompts = _prompts()
        handle = _handle(prompts)
        _write_results(s3, handle, prompts)
        backend.collect(handle)
        assert backend.usage.input_tokens == len(prompts) * 419
        assert backend.usage.output_tokens == len(prompts) * 377


class TestCellDigest:
    """A corpus edit the prompts do not reveal must still stop a collect."""

    def test_relabelling_a_cell_changes_the_digest_though_the_prompt_does_not(self) -> None:
        """Renaming an item, or moving which arm a prompt belongs to, has to be caught."""
        original = _cell_metadata(3)
        renamed = [{**original[0], "item_id": "renamed"}, *original[1:]]
        rearmed = [{**original[0], "arm": "pressured"}, *original[1:]]
        assert cell_digest(original) != cell_digest(renamed)
        assert cell_digest(original) != cell_digest(rearmed)

    def test_reordering_cells_changes_the_digest(self) -> None:
        cells = _cell_metadata(3)
        assert cell_digest(cells) != cell_digest([cells[1], cells[0], cells[2]])

    def test_length_prefixing_stops_two_sequences_hashing_alike(self) -> None:
        """A bare separator byte makes ["a\0b", "c"] and ["a", "b\0c"] collide."""
        assert prompt_digest(["a\x00b", "c"]) != prompt_digest(["a", "b\x00c"])
        assert prompt_digest(["ab", "c"]) != prompt_digest(["a", "bc"])

    def test_renaming_a_key_changes_the_digest_though_the_values_do_not(self) -> None:
        """The way this tripwire could be disarmed while still reporting a match on both sides.

        With a fixed key list and a default of empty string, a caller whose metadata builder used
        other key names hashed every cell to the same constant, so the digest degenerated to a
        function of the record count -- and matched between submit and collect because both sides
        ran the same builder. Full-content hashing has to make key names themselves significant.
        """
        assert cell_digest([{"item_id": "a"}]) != cell_digest([{"iid": "a"}])

    def test_two_wholly_misnamed_cell_lists_cannot_hash_alike(self) -> None:
        """Both hashed to b7dc482c... under the defaulted four-key scheme: equal to each other,
        and to any three cells."""
        assert cell_digest([{"iid": 1}, {"iid": 2}, {"iid": 3}]) != cell_digest(
            [{"zz": 9}, {"zz": 8}, {"zz": 7}]
        )
        assert cell_digest([{"iid": 1}, {"iid": 2}, {"iid": 3}]) != cell_digest(_cell_metadata(3))

    def test_the_hatch_probes_metadata_shape_digests_and_discriminates(self) -> None:
        """The second real caller: the hatch probe labels records cell/group/sample, not
        item/dimension/arm/repeat, and the four-key scheme met it with a KeyError at submit time
        (caught live, 2026-08-25). The digest must accept the shape and stay a tripwire on it."""
        probe_cells = [
            {
                "cell": "misspecified.hatch-present.framing-neutral",
                "group_index": 0,
                "sample_index": 0,
            },
            {
                "cell": "misspecified.hatch-present.framing-neutral",
                "group_index": 0,
                "sample_index": 1,
            },
            {"cell": "control.hatch-absent.framing-fallible", "group_index": 1, "sample_index": 0},
        ]
        relabelled = [
            {**probe_cells[0], "cell": "control.hatch-absent.framing-neutral"},
            *probe_cells[1:],
        ]
        assert cell_digest(probe_cells) != cell_digest(relabelled)
        assert cell_digest(probe_cells) != cell_digest(
            [probe_cells[1], probe_cells[0], probe_cells[2]]
        )

    def test_a_value_the_serializer_cannot_represent_raises(self) -> None:
        """Coercing an unserializable value would compare equal across the edits it hides."""
        with pytest.raises(TypeError):
            cell_digest([{"item_id": object()}])

    def test_the_digest_of_the_real_caller_shape_has_not_moved(self) -> None:
        """Pinned because handles saved in ``artifacts/`` compare against it in a later session.

        A change to the digest scheme silently invalidates every handle already on disk, turning a
        collect into a mismatch that reads as a corpus edit. This pin is what makes such a change
        deliberate: it moved once, from the indexed four-key scheme to canonical full-content JSON
        (2026-08-25), after checking that every handle directory on disk had already been
        collected. This is the value the sweep's four-key metadata produces under that scheme.
        """
        assert (
            cell_digest(_cell_metadata(3))
            == "afb7093a9b9c1a0e238cd86c3b6c5e39b81b664cc329641d7d969ec2afdc130d"
        )


class TestHandleRoundTrip:
    """A handle has to survive the process that made it, or a lost session means paying twice."""

    def test_a_handle_reloads_field_for_field(self, tmp_path: Path) -> None:
        handle = _handle(_prompts())
        assert BatchJobHandle.load(handle.save(tmp_path / "h.json")) == handle

    def test_the_job_id_is_the_output_subdirectory_name(self) -> None:
        assert _handle(_prompts()).job_id == "abc123xyz789"

    def test_the_sampling_labels_travel_on_the_handle(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The collect leg is a fresh process, so the cap it labels its trace with comes from here.

        Without these two fields, saying what cap a completed sweep ran at means reading the S3
        sidecar, which is an AWS call -- and the handle directories already on disk carry neither.
        """
        sampling = BedrockSamplingConfig(max_tokens=24576, reasoning_effort="high")
        backend, _, _ = _build(monkeypatch, GPT_OSS_20B, sampling=sampling)
        handle = backend.submit(_prompts())
        assert (handle.max_tokens, handle.reasoning_effort) == (24576, "high")

    def test_a_handle_written_before_the_fields_existed_reads_them_as_absent(
        self, tmp_path: Path
    ) -> None:
        """Three handle directories predate the fields; None reads as "written before these"."""
        path = tmp_path / "old.json"
        without_sampling = {
            key: value
            for key, value in asdict(_handle(_prompts())).items()
            if key not in {"max_tokens", "reasoning_effort"}
        }
        path.write_text(json.dumps(without_sampling), encoding="utf-8")

        loaded = BatchJobHandle.load(path)
        assert (loaded.max_tokens, loaded.reasoning_effort) == (None, None)

    def test_a_handle_missing_a_required_field_says_which_one_and_why_it_cannot_be_collected(
        self, tmp_path: Path
    ) -> None:
        """Not every added field was given a default, and the oldest handle on disk lacks one.

        ``cell_digest`` arrived after the first sweeps and is required, so the handle directory from
        2026-08-17T00:27 does not load at all -- it raises a bare ``TypeError`` naming a dataclass
        argument, which reads like a code defect rather than "this handle predates the corpus-drift
        tripwire and cannot be safely collected". That refusal is correct and stays; what it says is
        what changes, because an operator holding a paid job's only ARN deserves to know which.
        """
        path = tmp_path / "pre-cell-digest.json"
        without_digest = {
            key: value for key, value in asdict(_handle(_prompts())).items() if key != "cell_digest"
        }
        path.write_text(json.dumps(without_digest), encoding="utf-8")

        with pytest.raises(ValueError, match="cell_digest") as caught:
            BatchJobHandle.load(path)
        assert str(path) in str(caught.value)

    def test_a_handle_carrying_an_unknown_field_is_refused_rather_than_silently_dropping_it(
        self, tmp_path: Path
    ) -> None:
        """The mirror case: a renamed field would otherwise raise the same opaque ``TypeError``."""
        path = tmp_path / "from-the-future.json"
        with_extra = asdict(_handle(_prompts())) | {"sampling_profile": "balanced"}
        path.write_text(json.dumps(with_extra), encoding="utf-8")

        with pytest.raises(ValueError, match="sampling_profile"):
            BatchJobHandle.load(path)

    def test_a_save_onto_a_tracked_path_is_refused_at_the_writer(self) -> None:
        """The guard lives in the writer, not only in callers: a handle carries the AWS account id
        (in ``job_arn``), the bucket (in the S3 URIs) and the profile name, and the last leak of
        this class came through a second call site nothing guarded. The message must name that
        cargo rather than the item prompt text a trace carries -- an operator told to look for
        benchmark items in a file that has none will not find the real reason."""
        repo_root = Path(__file__).resolve().parents[2]
        tracked = repo_root / "reward_hacking" / "leaked-batch-handle.json"
        try:
            with pytest.raises(ValueError, match="not under a gitignored root") as caught:
                _handle(_prompts()).save(tracked)
            assert not tracked.exists()
            message = str(caught.value)
            assert "account id" in message, "the refusal must name what a handle actually carries"
            assert "prompt text" not in message
        finally:
            tracked.unlink(missing_ok=True)


class TestSamplingLabels:
    """One comparison for both CLIs' resume/skip checks, mechanically coupled to the handle.

    Before this lived on the handle, the field list was hand-written in three places that had to
    agree (submit's stamping plus two comparison sites in different CLI modules) with nothing
    keeping them agreeing -- the parallel-dict copies had already diverged once.
    """

    def test_the_label_fields_are_exactly_the_defaulted_handle_fields(self) -> None:
        """What goes red the day someone adds temperature to the handle without teaching the
        comparison about it. Optional-ness is read off the dataclass the same way ``load`` reads
        it: the sampling labels are precisely the trailing defaulted fields."""
        defaulted = tuple(
            field.name for field in fields(BatchJobHandle) if field.default is not MISSING
        )
        assert defaulted == SAMPLING_LABEL_FIELDS

    def test_matching_labels_report_no_mismatch(self) -> None:
        handle = replace(_handle(_prompts()), max_tokens=2048, reasoning_effort="high")
        assert handle.sampling_label_mismatches(max_tokens=2048, reasoning_effort="high") == []

    def test_each_differing_label_is_named_with_both_values(self) -> None:
        handle = replace(_handle(_prompts()), max_tokens=2048, reasoning_effort="high")
        mismatches = handle.sampling_label_mismatches(max_tokens=24576, reasoning_effort=None)
        assert len(mismatches) == 2
        assert any(
            "max_tokens" in line and "2048" in line and "24576" in line for line in mismatches
        )
        assert any("reasoning_effort" in line and "high" in line for line in mismatches)

    def test_a_handle_predating_the_labels_reads_as_unrecorded_not_matching(self) -> None:
        """All-None means "written before the fields existed" -- unknown, not a cap of zero. The
        caller decides what to do with unknown; the comparison itself would report it as differing,
        so a caller that tolerates old handles must check this first."""
        old = _handle(_prompts())
        assert old.sampling_labels_unrecorded is True
        assert replace(old, max_tokens=2048).sampling_labels_unrecorded is False
        assert replace(old, reasoning_effort="low").sampling_labels_unrecorded is False


class TestProtocolCompliance:
    """The runner, graders and analysis see this as an ordinary backend."""

    def test_it_satisfies_the_backend_protocol(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend, _, _ = _build(monkeypatch)
        assert isinstance(backend, Backend)

    def test_it_labels_its_transport_distinctly_from_the_live_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend, _, _ = _build(monkeypatch)
        assert backend.transport == "bedrock-batch"
