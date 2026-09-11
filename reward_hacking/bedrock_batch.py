"""Bedrock batch-inference backend: the wide-roster sweep path, submit and collect split apart.

The live Converse backend in ``model_backend.py`` fans one model's prompts across a thread pool.
That is the right shape for one model and the wrong shape for a roster: it means one sequential run
per model, a rate-limit budget to respect for each, and full price. Bedrock's batch inference
service takes a JSONL file of records in S3, runs them asynchronously, and writes the results back
to S3 at half the on-demand token price. The quota allows 100 in-progress jobs per base model, so a
whole-roster sweep is one job per model submitted at once and roughly 25 minutes of total wall
clock however wide the roster is. That fan-out, not per-model speed, is the reason to use it -- a
single 145-record Converse run finishes in a couple of minutes, faster than batch's ~5-minute
queueing floor. **Anything batch-capable belongs on this path**, since it is half the price and one
round of wall clock rather than one per model.

**Which models are batch-capable is a per-model fact, and the authority is
``CreateModelInvocationJob`` itself.** Point it at a nonexistent S3 input prefix and it answers the
support question before anything can run or be billed: an unsupported model comes back with
``Batch inference is not supported for the requested model.`` The user guide's
``batch-inference-supported`` page agrees with the API everywhere the two have been compared and is
the right place to *look first*, but it is documentation, not the thing being asked. Two other
plausible-looking substitutes are wrong outright:

* **``list-foundation-models`` cannot answer it.** It reports ``inferenceTypesSupported:
  [ON_DEMAND]`` for every roster model, batch-proven ones included.
* A model card's **Service Tiers** table has a ``Batch`` column for some models and none at all for
  others, but that column is about the per-request ``"service_tier": "batch"`` field, a different
  mechanism from ``CreateModelInvocationJob``. Claude Haiku 4.5 is the disproof: its card carries no
  ``Batch`` column and it has three completed 500-record batch jobs in this account.
* Price List records run ahead of, and behind, the doc page in both directions. ``xai.grok-4.3``
  carries batch rates in us-west-2 and appears nowhere on the support page; several Claude 3 models
  are on the support page with no batch rate at all.

**Two things make such a probe read as a clean answer when it is really a failed probe, and both
cost several attempts to find.** First, the API validates in a fixed order -- cross-account
pass-role, then caller permissions, then model support -- so a probe run under the wrong profile
returns an identical ``AccessDenied`` for every model asked about, which looks exactly like a
uniform result. **Include a known-batch-capable control in any such probe**; if the control does not
come back supported, nothing else in that run means anything. Second, id form and model support are
separate failure modes: the ``modelId`` regex rejects the profile-prefixed
``us.openai.gpt-5.6-sol`` and ``global.openai.gpt-5.6-sol`` before the support check runs at all,
while accepting ``global.anthropic.*``. A rejected prefixed id says nothing about the model, so
retest the bare id before concluding anything.

**Claude 5 is partly batch-capable, and an earlier note here saying otherwise was wrong.** Claude
Opus 5 supports batch inference, confirmed by a probe job that the API created, and is on the
roster. Confirmed *not* batch-capable by the same probe, each rejected with that exact
not-supported message and each corroborated by the absence of any batch price row: **Claude Sonnet
5, Claude Fable 5, and GPT-5.6 Sol** (the rest of the GPT-5.6 family is absent from the support
page too). Anthropic's own split is per-model rather than per-generation -- Opus 4.5, Opus 4.6,
Sonnet 4, Sonnet 4.5, Sonnet 4.6 and Haiku 4.5 all price batch, while Opus 4.7 and Opus 4.8 do not
-- so a new Claude release is a fresh check every time, never an inference from its family.
:data:`KNOWN_NOT_BATCH_CAPABLE` keeps that answer where a test can hold it.

**The format is Converse, not the native schema.** ``CreateModelInvocationJob`` takes an optional
``modelInvocationType`` whose values are ``InvokeModel`` (the default) and ``Converse``. Under
``Converse`` a record's ``modelInput`` is exactly the Converse request body minus ``modelId``, and
its ``modelOutput`` is exactly the Converse response body. So this module renders prompts with
``converse_request`` and parses results with ``parse_converse_output``, the same two functions the
live backend uses, and there is deliberately no second prompt renderer or response parser here. It
also means none of the native-path traps apply, in particular GPT-OSS concatenating its reasoning
into the answer string.

Two tooling facts that are silent when you get them wrong:

* ``aws-cli`` 2.33.15 on this box does not expose ``--model-invocation-type`` at all, so a CLI
  submit silently defaults to ``InvokeModel`` and every Converse-shaped record fails. ``boto3``
  1.43.72 does expose it. Submit from Python.
* A batch result file serialises every member of the Converse content-block union with an explicit
  ``null``, where botocore strips unset members on the live path. The shared parsers are
  null-tolerant for that reason; the first parse of a real batch record crashed on it.

Nothing in this module creates AWS infrastructure. The service role and the bucket it writes to
already exist in a development account with seventy jobs of history, and this writes under a new
prefix in that bucket rather than a new bucket, because the role's S3 policy cannot be read by our
principal and is almost certainly scoped to the buckets it already uses.

**Which account, role, bucket and profile those are is read from the environment, never committed.**
This repository is public, so the four settings below name environment variables instead of values,
and every one of them fails loudly at construction time with the variable's name when it is unset.
See ``docs/scratch/bedrock-batch-env.md`` for the local values; that file is gitignored.

**The AWS profile differs from the live Converse path's, and that is the one real surprise here.**
``CreateModelInvocationJob`` hands a service role to Bedrock, so the caller needs ``iam:PassRole``
on that role -- a requirement the AWS batch-permissions doc omits. The profile the live path uses
resolves to a PowerUser role with no IAM permissions at all (verified: PassRole, CreatePolicy and
GetRole are each denied for it), so a submit under it uploads the input file and then dies at the
create call, having already paid for the S3 write. ``JAGGED_BATCH_PROFILE`` therefore has to name a
principal that can pass the role -- an Admin role in the same account does. If least privilege is
wanted later, an account admin can grant PowerUser ``iam:PassRole`` on that single role ARN
conditioned on ``iam:PassedToService = bedrock.amazonaws.com``, after which the live path's profile
works here too.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import time
from dataclasses import MISSING, asdict, dataclass, fields
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from reward_hacking.model_backend import (
    DEFAULT_BEDROCK_REGION,
    BedrockCompletion,
    BedrockSamplingConfig,
    TokenUsage,
    converse_extra_request_fields,
    converse_inference_config,
    converse_request,
    parse_converse_output,
)
from reward_hacking.trace import refuse_tracked_trace_path

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

logger = logging.getLogger(__name__)

BATCH_ROLE_ARN_ENV = "JAGGED_BATCH_ROLE_ARN"
BATCH_BUCKET_ENV = "JAGGED_BATCH_BUCKET"
BATCH_PROFILE_ENV = "JAGGED_BATCH_PROFILE"
BATCH_BUCKET_OWNER_ENV = "JAGGED_BATCH_BUCKET_OWNER"

BATCH_PREFIX = "batch_jobs/jaggedbench"

_ACCOUNT_ID_DIGITS = 12

# arn:aws:iam::<account>:role/<name> -- the account is the fifth colon-separated field.
_ARN_ACCOUNT_FIELD = 4

# Per job, and Service Quotas marks it not adjustable. A smaller run has to go on Converse.
MIN_BATCH_RECORDS = 100

# The API minimum, not a choice; a shorter deadline cannot be requested.
BATCH_TIMEOUT_HOURS = 24

# The service's own job deadline in the unit the wait knobs take: the longest a wait can ever be
# worth, since past it the job itself has expired.
BATCH_TIMEOUT_SECONDS = BATCH_TIMEOUT_HOURS * 3600.0

# From the CreateModelInvocationJob shape metadata (the AWS CLI help claims a minimum of 10).
MAX_JOB_NAME_CHARS = 63

# Well past the observed 8.7-25 minute band; a timeout never stops the job (see ``wait``).
DEFAULT_POLL_SECONDS = 30.0
# Interactive patience, deliberately NOT the service deadline above: a caller that omits the
# timeout gets a session-sized wait whose TimeoutError names the resume route, not a silent
# day-long block. A caller prepared to wait out the job passes BATCH_TIMEOUT_SECONDS explicitly,
# as the hatch probe CLI does. This fixed hour is what once orphaned paid jobs -- survivable now
# only because every submit persists its handle first.
DEFAULT_WAIT_SECONDS = 3600.0

TRANSPORT = "bedrock-batch"

# What a handle file would publish if committed, for the tracked-path refusal: not item text, but
# the account-identifying values (job_arn, S3 URIs, profile) this public repository never tracks.
HANDLE_CARRIES = "the batch job's AWS account id, bucket and profile"

_TERMINAL_STATUSES = frozenset({"Completed", "Failed", "Stopped", "PartiallyCompleted", "Expired"})


def _required_env(variable: str, holds: str) -> str:
    """Read an account-specific setting, failing with the variable's name when it is unset.

    Unset must raise rather than fall back to anything. A default here would either be a committed
    account identifier, which this public repository must not carry, or an empty string, which boto3
    would carry all the way to a submit that uploads the input file before dying.
    """
    value = os.environ.get(variable, "").strip()
    if not value:
        raise RuntimeError(
            f"{variable} is not set; it must hold {holds}. The batch path's account, role, bucket "
            "and profile are read from the environment because this repository is public. Export "
            "them in your shell or a local gitignored env file (see docs/scratch/)."
        )
    return value


def batch_role_arn() -> str:
    """Return the Bedrock service-role ARN the job creation passes to the service."""
    return _required_env(BATCH_ROLE_ARN_ENV, "the ARN of the Bedrock batch-inference service role")


def batch_bucket() -> str:
    """Return the S3 bucket the input JSONL and the results live in."""
    return _required_env(BATCH_BUCKET_ENV, "the S3 bucket name for batch inputs and outputs")


def batch_profile() -> str:
    """Return the named AWS profile that can pass the service role to Bedrock."""
    return _required_env(
        BATCH_PROFILE_ENV,
        "a named AWS profile whose principal holds iam:PassRole on the service role",
    )


def batch_bucket_owner(role_arn: str) -> str:
    """Return the account id that must own the bucket, defaulting to the service role's account.

    ``ExpectedBucketOwner`` exists so that an upload cannot succeed against someone else's bucket
    and then be rejected at job creation, leaving the objects behind and paid for. In every
    configuration used here the bucket and the service role sit in the same account, so the role
    ARN already carries the answer; a cross-account bucket has to say so explicitly.
    """
    override = os.environ.get(BATCH_BUCKET_OWNER_ENV, "").strip()
    if override:
        return override
    fields = role_arn.split(":")
    account = fields[_ARN_ACCOUNT_FIELD] if len(fields) > _ARN_ACCOUNT_FIELD else ""
    if not (len(account) == _ACCOUNT_ID_DIGITS and account.isdigit()):
        raise RuntimeError(
            f"cannot read an account id out of {BATCH_ROLE_ARN_ENV}={role_arn!r}: expected an ARN "
            f"of the form arn:aws:iam::<{_ACCOUNT_ID_DIGITS}-digit account>:role/<name>. Set "
            f"{BATCH_BUCKET_OWNER_ENV} to the account that owns the bucket instead."
        )
    return account


_INFERENCE_PROFILE_PREFIXES = ("us.", "global.", "eu.", "apac.", "au.", "jp.")
"""Every geography prefix a cross-region inference profile id can carry.

Only ``us.`` and ``global.`` profiles exist in us-west-2 (51 and 18 of them), so the rest are here
for the id spellings the model cards advertise rather than for anything reachable from this region.
``au.`` and ``jp.`` were missing and are real: the Claude Opus 5 card lists
``au.anthropic.claude-opus-5`` among its geo ids and the Nova 2 Lite card lists a ``jp.`` one. A
prefix missing from this tuple fails in a way worth spelling out, because it is silent at the point
of the mistake: :attr:`RosterModel.is_inference_profile` returns false, so
``_assert_model_available`` asks ``get_foundation_model`` about an id that only exists as a profile
and the submit dies there. :func:`assert_roster_ids_well_formed` is the guard against that.
"""

# The vendor namespaces a Bedrock *foundation-model* id starts with. Read off the model-card index,
# and the counterpart to the prefixes above: an id whose first segment is in neither set is
# malformed, which is what assert_roster_ids_well_formed refuses.
_FOUNDATION_MODEL_VENDORS = (
    "ai21",
    "amazon",
    "anthropic",
    "cohere",
    "deepseek",
    "google",
    "meta",
    "minimax",
    "mistral",
    "moonshot",
    "moonshotai",
    "nvidia",
    "openai",
    "qwen",
    "stability",
    "twelvelabs",
    "writer",
    "xai",
    "zai",
)


@dataclass(frozen=True, slots=True)
class RosterModel:
    """One verified batch-capable model, its batch token prices, and its per-model refusals.

    ``batch_price_verified`` records whether the figures beside it were read out of the Price List
    API or reasoned to. Every row is currently true, and keeping the flag is not decoration: it is
    what stops the *next* addition presenting a guess as a measurement, and ``collect`` annotates its
    cost line with ``PRICE UNVERIFIED`` when it is false. Reading an Anthropic batch price takes
    knowing that Claude is billed through AWS Marketplace and so lives under the
    ``AmazonBedrockFoundationModels`` service code, not ``AmazonBedrock`` -- looking only in the
    latter is what once made Claude Haiku 4.5's price a halved list price rather than a reading, and
    the reading turned out to agree with it exactly.

    ``rejects_temperature_with_top_p`` is not a nicety. A Claude Haiku 4.5 job in this account came
    back with status ``Completed`` and 500 of 500 records dead on "`temperature` and `top_p` cannot
    both be specified for this model", having burned the queue time to find out. Refusing that
    combination before submission is the cheapest possible version of that check.

    ``deprecates_temperature_and_top_p`` is the stricter form of the same refusal and strictly
    implies it, so the two are true together on the row that needs it. Claude Opus 5 answers a
    request carrying either knob with "`temperature` is deprecated for this model" (likewise
    ``top_p``), so the pair-only check would wave through a run that set just one and hand back the
    identical all-records-dead job. Only the no-op values pass -- ``temperature`` 1.0 is accepted
    where 0.0 and 0.5 are not -- which is why the guard refuses the field being set at all rather
    than trying to police values: a run that asked for a temperature and silently got none measured
    something other than what it recorded.

    ``max_tokens_limit`` is the same class of refusal for the same reason: a model hard-rejects a
    ``maxTokens`` above its ceiling per record, so a job asking for more reports ``Completed`` with
    every record errored. It is required rather than optional, and that is a fix rather than
    tidiness. While it was optional, seven of the eight rows left it unset meaning "no documented
    ceiling", and one of those seven -- Llama 4 Scout, whose real ceiling is 8,192 -- sat below
    ``DEFAULT_BEDROCK_MAX_TOKENS`` of 30,000, so any sweep including it at the default cap was
    submitting a job that could only come back dead. Requiring the field means a new model cannot
    inherit that silence.

    **The ceilings come from the API, not from the model card, and the two disagree.** A card's "max
    output tokens" is a documented figure; the enforced number is whatever
    ``The maximum tokens you requested exceeds the model limit of N`` says, and one call at an
    impossible cap reads it off for any model. Nova Micro is the counterexample that settles which to
    trust: its card says 5K and the API enforces 10,000.
    """

    model_id: str
    name: str
    batch_price_in_per_mtok: float
    batch_price_out_per_mtok: float
    batch_price_verified: bool
    rejects_temperature_with_top_p: bool
    deprecates_temperature_and_top_p: bool
    max_tokens_limit: int
    notes: str

    @property
    def is_inference_profile(self) -> bool:
        """Whether this id names a cross-region inference profile rather than a foundation model."""
        return self.model_id.startswith(_INFERENCE_PROFILE_PREFIXES)

    def estimated_cost_usd(self, usage: TokenUsage) -> float:
        """Batch-price this usage in dollars, for the log line that puts a run's cost on record."""
        return (
            usage.input_tokens * self.batch_price_in_per_mtok
            + usage.output_tokens * self.batch_price_out_per_mtok
        ) / 1_000_000


# Amazon Nova Micro hard-rejects any maxTokens above this. A property of the model on every
# transport, not a research budget, which is why it lives beside the roster rather than beside
# RecoveryBench's measured caps -- and the one ceiling on the roster with a name, because
# RecoveryBench imports it. The rest sit inline in the rows: twenty-odd named constants carrying one
# integer each would bury the table they exist to annotate.
NOVA_MICRO_MAX_TOKENS = 10_000

BATCH_ROSTER: tuple[RosterModel, ...] = (
    RosterModel(
        model_id="us.amazon.nova-micro-v1:0",
        name="Amazon Nova Micro",
        batch_price_in_per_mtok=0.0175,
        batch_price_out_per_mtok=0.07,
        batch_price_verified=True,
        rejects_temperature_with_top_p=False,
        deprecates_temperature_and_top_p=False,
        max_tokens_limit=NOVA_MICRO_MAX_TOKENS,
        notes=(
            "Cheapest text model on Bedrock and the bottom rung of the capability ladder, which is "
            "what makes it the arm that says whether an item fails for capability reasons at all. "
            "The us. prefix is required: the bare amazon.nova-micro-v1:0 is INFERENCE_PROFILE-only "
            "in us-west-2, and there is no global. variant."
        ),
    ),
    RosterModel(
        model_id="openai.gpt-oss-20b-1:0",
        name="OpenAI GPT-OSS 20B",
        batch_price_in_per_mtok=0.035,
        batch_price_out_per_mtok=0.15,
        batch_price_verified=True,
        rejects_temperature_with_top_p=False,
        deprecates_temperature_and_top_p=False,
        max_tokens_limit=128_000,
        notes=(
            "Small sibling of the model already trusted on the Converse path, so a 20B/120B pair "
            "isolates scale within one family and tokenizer. Returns readable chain of thought "
            "in reasoningContent.reasoningText.text, unlike Luna's encrypted blob, so it is one "
            "of the few arms where reasoning could later be scored rather than just answers."
        ),
    ),
    RosterModel(
        model_id="openai.gpt-oss-120b-1:0",
        name="OpenAI GPT-OSS 120B",
        batch_price_in_per_mtok=0.075,
        batch_price_out_per_mtok=0.3,
        batch_price_verified=True,
        rejects_temperature_with_top_p=False,
        deprecates_temperature_and_top_p=False,
        max_tokens_limit=128_000,
        notes=(
            "In the roster for a methodological reason rather than a capability one: it is "
            "verified on the live Converse path AND has a completed 500-record batch job in this "
            "account, so it is the one model where the same prompts can run both ways and the two "
            "traces compared. That agreement check is the cheapest guard against a silent join or "
            "parsing bug on this path. Emits ~1,780 output tokens per call at default effort, so "
            "its real cost runs well above the nominal figure."
        ),
    ),
    RosterModel(
        model_id="mistral.ministral-3-8b-instruct",
        name="Mistral Ministral 3 8B Instruct",
        batch_price_in_per_mtok=0.07,
        batch_price_out_per_mtok=0.07,
        batch_price_verified=True,
        rejects_temperature_with_top_p=False,
        deprecates_temperature_and_top_p=False,
        max_tokens_limit=262_144,
        notes=(
            "Small dense non-reasoning slot plus vendor diversity. Flat input-equals-output "
            "pricing means the verbose hedging the pressured arm provokes cannot blow up the bill."
        ),
    ),
    RosterModel(
        model_id="google.gemma-3-12b-it",
        name="Google Gemma 3 12B IT",
        batch_price_in_per_mtok=0.05,
        batch_price_out_per_mtok=0.15,
        batch_price_verified=True,
        rejects_temperature_with_top_p=False,
        deprecates_temperature_and_top_p=False,
        max_tokens_limit=131_072,
        notes=(
            "A post-training lineage that is not RLVR-heavy in the way this repo's central "
            "hypothesis is about, which makes it a contrast arm rather than another point on the "
            "size axis. Doc-and-pricing verified only, with no batch job in this account yet."
        ),
    ),
    RosterModel(
        model_id="us.meta.llama4-scout-17b-instruct-v1:0",
        name="Meta Llama 4 Scout 17B Instruct",
        batch_price_in_per_mtok=0.085,
        batch_price_out_per_mtok=0.33,
        batch_price_verified=True,
        rejects_temperature_with_top_p=False,
        deprecates_temperature_and_top_p=False,
        max_tokens_limit=8_192,
        notes=(
            "Mixture-of-experts, another vendor. The us. prefix is required; Llama 4 Scout has "
            "no single-region batch support anywhere. Its Maverick sibling has a completed batch "
            "job in this account, so the prefixed Llama 4 path is known to work end to end. "
            "The 8,192-token ceiling is the tightest on the roster and sits BELOW the shared "
            "Converse default of 30,000, so this row is the reason max_tokens_limit is required: "
            "while it read None, every sweep that included this model at the default cap was "
            "submitting a job whose every record could only come back with a 400. Its 10M-token "
            "context window is no guide at all to what it will emit."
        ),
    ),
    RosterModel(
        model_id="qwen.qwen3-235b-a22b-2507-v1:0",
        name="Qwen3 235B A22B 2507",
        batch_price_in_per_mtok=0.11,
        batch_price_out_per_mtok=0.44,
        batch_price_verified=True,
        rejects_temperature_with_top_p=False,
        deprecates_temperature_and_top_p=False,
        max_tokens_limit=262_144,
        notes=(
            "The strong-open-weights ceiling, and unusually relevant here: the local training "
            "substrate is the Qwen family, so this gives a ceiling in the same lineage as anything "
            "later fine-tuned, keeping the hosted-probe and RL-phase numbers comparable. "
            "Batch-proven in this account."
        ),
    ),
    RosterModel(
        model_id="nvidia.nemotron-nano-3-30b",
        name="NVIDIA Nemotron Nano 3 30B",
        batch_price_in_per_mtok=0.03,
        batch_price_out_per_mtok=0.12,
        batch_price_verified=True,
        rejects_temperature_with_top_p=False,
        deprecates_temperature_and_top_p=False,
        max_tokens_limit=262_144,
        notes=(
            "Cheapest reasoning-capable row on the roster and the twelfth vendor on it, which is "
            "the point: NVIDIA's post-training recipe is neither the RLVR-heavy lineage this "
            "repo's hypothesis is about nor a frontier lab's RLHF, so it sits with Gemma as a "
            "contrast arm rather than a rung on the size ladder. Doc-and-pricing verified, no "
            "batch job in this account yet."
        ),
    ),
    RosterModel(
        model_id="zai.glm-4.7-flash",
        name="Z.AI GLM 4.7 Flash",
        batch_price_in_per_mtok=0.035,
        batch_price_out_per_mtok=0.2,
        batch_price_verified=True,
        rejects_temperature_with_top_p=False,
        deprecates_temperature_and_top_p=False,
        max_tokens_limit=202_752,
        notes=(
            "The cheap half of a within-family pair with GLM 4.7, the same design as the "
            "20B/120B GPT-OSS pair: one tokenizer and one post-training recipe, two capability "
            "points, so a difference between them is scale rather than lineage."
        ),
    ),
    RosterModel(
        model_id="qwen.qwen3-next-80b-a3b",
        name="Qwen3 Next 80B A3B",
        batch_price_in_per_mtok=0.07,
        batch_price_out_per_mtok=0.6,
        batch_price_verified=True,
        rejects_temperature_with_top_p=False,
        deprecates_temperature_and_top_p=False,
        max_tokens_limit=262_144,
        notes=(
            "The hybrid-attention arm, and the closest hosted analogue to the local training "
            "substrate: like the Qwen3.5 checkpoints this repo fine-tunes, it interleaves linear "
            "attention with full attention rather than being uniformly quadratic, so anything "
            "that turns out to be an artefact of that architecture shows up here too. Its price "
            "is the one on the roster read off the -mantle- usagetype spelling, because the plain "
            "spelling carries standard, flex and priority rows for this model but no batch row; "
            "the mantle batch rate is exactly half the mantle standard rate, as everywhere else."
        ),
    ),
    RosterModel(
        model_id="qwen.qwen3-32b-v1:0",
        name="Qwen3 32B",
        batch_price_in_per_mtok=0.075,
        batch_price_out_per_mtok=0.3,
        batch_price_verified=True,
        rejects_temperature_with_top_p=False,
        deprecates_temperature_and_top_p=False,
        max_tokens_limit=32_768,
        notes=(
            "Dense rather than mixture-of-experts, and the smallest Qwen here, so the Qwen rows "
            "span dense-32B, hybrid-80B and MoE-235B inside one lineage. The 32,768 ceiling is "
            "the second tightest on the roster and clears the shared 30,000-token Converse "
            "default by only 2,768, so a run that raises the cap at all has to skip this row."
        ),
    ),
    RosterModel(
        model_id="mistral.ministral-3-14b-instruct",
        name="Mistral Ministral 14B 3.0",
        batch_price_in_per_mtok=0.1,
        batch_price_out_per_mtok=0.1,
        batch_price_verified=True,
        rejects_temperature_with_top_p=False,
        deprecates_temperature_and_top_p=False,
        max_tokens_limit=262_144,
        notes=(
            "The larger half of a within-family pair with Ministral 3 8B, and flat "
            "input-equals-output pricing like its sibling, so the verbose hedging the pressured "
            "arm provokes cannot blow up the bill on either."
        ),
    ),
    RosterModel(
        model_id="google.gemma-3-27b-it",
        name="Google Gemma 3 27B",
        batch_price_in_per_mtok=0.12,
        batch_price_out_per_mtok=0.19,
        batch_price_verified=True,
        rejects_temperature_with_top_p=False,
        deprecates_temperature_and_top_p=False,
        max_tokens_limit=131_072,
        notes=(
            "Scale partner to Gemma 3 12B, holding the not-RLVR-heavy post-training lineage "
            "fixed. Note the doc-page name says 'Gemma 3 27B PT', for pretrained, while the id "
            "and the price rows both say -it, for instruction-tuned; the id is what gets sent, "
            "and it resolves ACTIVE, but a run that expects instruction-following should confirm "
            "it is getting it rather than trusting either label."
        ),
    ),
    RosterModel(
        model_id="us.meta.llama4-maverick-17b-instruct-v1:0",
        name="Meta Llama 4 Maverick 17B Instruct",
        batch_price_in_per_mtok=0.12,
        batch_price_out_per_mtok=0.485,
        batch_price_verified=True,
        rejects_temperature_with_top_p=False,
        deprecates_temperature_and_top_p=False,
        max_tokens_limit=8_192,
        notes=(
            "Scale partner to Llama 4 Scout, and the row whose completed batch job in this "
            "account is what made the prefixed Llama 4 path trustworthy in the first place. The "
            "us. prefix is not optional and there is no global. profile for it at all. Shares "
            "Scout's 8,192-token ceiling, well below the shared Converse default."
        ),
    ),
    RosterModel(
        model_id="minimax.minimax-m2.1",
        name="MiniMax M2.1",
        batch_price_in_per_mtok=0.15,
        batch_price_out_per_mtok=0.6,
        batch_price_verified=True,
        rejects_temperature_with_top_p=False,
        deprecates_temperature_and_top_p=False,
        max_tokens_limit=196_608,
        notes=(
            "Already load-bearing in this repo's published results, which is the reason it is "
            "here rather than its identically-priced M2 predecessor. Its traces are known to be "
            "convention-sensitive -- whether it asserts and then retracts swings a control's "
            "false-positive rate by more than an order of magnitude -- so any scorer run over "
            "this row has to name the convention it scored under."
        ),
    ),
    RosterModel(
        model_id="global.amazon.nova-2-lite-v1:0",
        name="Amazon Nova 2 Lite",
        batch_price_in_per_mtok=0.15,
        batch_price_out_per_mtok=1.25,
        batch_price_verified=True,
        rejects_temperature_with_top_p=False,
        deprecates_temperature_and_top_p=False,
        max_tokens_limit=65_535,
        notes=(
            "Current-generation Amazon, against Nova Micro's floor. The global. prefix is a "
            "price decision as well as a routing one and this is the row that shows it: Amazon "
            "prices the global profile and the us. geo profile separately, at 0.15/1.25 and "
            "0.1595/1.3255, so the prices here belong to this prefix and swapping it silently "
            "makes them wrong by ~6%. Its ceiling is 65,535 where the card says 64K, one below "
            "the round number somebody would guess."
        ),
    ),
    RosterModel(
        model_id="mistral.mistral-large-3-675b-instruct",
        name="Mistral Large 3",
        batch_price_in_per_mtok=0.25,
        batch_price_out_per_mtok=0.75,
        batch_price_verified=True,
        rejects_temperature_with_top_p=False,
        deprecates_temperature_and_top_p=False,
        max_tokens_limit=262_144,
        notes=(
            "Mistral's frontier, completing a three-point ladder in one lineage with the two "
            "Ministrals: 8B, 14B and 675B under one post-training recipe is the widest "
            "within-vendor capability span the roster has."
        ),
    ),
    RosterModel(
        model_id="zai.glm-4.7",
        name="Z.AI GLM 4.7",
        batch_price_in_per_mtok=0.3,
        batch_price_out_per_mtok=1.1,
        batch_price_verified=True,
        rejects_temperature_with_top_p=False,
        deprecates_temperature_and_top_p=False,
        max_tokens_limit=202_752,
        notes=(
            "Strong open-weights reasoning model from a lineage with no US frontier lab behind "
            "it, and the capable half of the pair with GLM 4.7 Flash."
        ),
    ),
    RosterModel(
        model_id="moonshot.kimi-k2-thinking",
        name="Moonshot Kimi K2 Thinking",
        batch_price_in_per_mtok=0.3,
        batch_price_out_per_mtok=1.25,
        batch_price_verified=True,
        rejects_temperature_with_top_p=False,
        deprecates_temperature_and_top_p=False,
        max_tokens_limit=262_144,
        notes=(
            "An explicitly thinking-first open-weights model, so it belongs with GPT-OSS among "
            "the arms whose reasoning could be scored rather than only their answers. Note the "
            "vendor namespace: this one is moonshot. while its K2.5 sibling is moonshotai., and "
            "the price rows use moonshotai. for both. The doc page's id is what gets sent and "
            "both spellings resolve ACTIVE only for their own model."
        ),
    ),
    RosterModel(
        model_id="moonshotai.kimi-k2.5",
        name="Moonshot Kimi K2.5",
        batch_price_in_per_mtok=0.3,
        batch_price_out_per_mtok=1.5,
        batch_price_verified=True,
        rejects_temperature_with_top_p=False,
        deprecates_temperature_and_top_p=False,
        max_tokens_limit=262_144,
        notes=(
            "The newer Kimi, on the moonshotai. namespace where K2 Thinking is on moonshot. -- "
            "a one-character-class difference between two rows of the same table, which is "
            "exactly the kind of id mistake the roster exists to have already made once."
        ),
    ),
    RosterModel(
        model_id="deepseek.v3.2",
        name="DeepSeek V3.2",
        batch_price_in_per_mtok=0.31,
        batch_price_out_per_mtok=0.925,
        batch_price_verified=True,
        rejects_temperature_with_top_p=False,
        deprecates_temperature_and_top_p=False,
        max_tokens_limit=163_840,
        notes=(
            "The open-weights lineage whose RL-heavy post-training is closest in spirit to what "
            "this repo's central hypothesis is about, which makes it the most directly relevant "
            "non-Qwen arm here rather than just another vendor. Bare id with no prefix: batch "
            "support is single-region and us-west-2 is one of those regions."
        ),
    ),
    RosterModel(
        model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",
        name="Anthropic Claude Haiku 4.5",
        batch_price_in_per_mtok=0.5,
        batch_price_out_per_mtok=2.5,
        batch_price_verified=True,
        rejects_temperature_with_top_p=True,
        deprecates_temperature_and_top_p=False,
        max_tokens_limit=64_000,
        notes=(
            "The cheap half of the frontier-lab pair with Opus 5, in an otherwise open-weights "
            "roster. Batch support is as solid as it gets here -- three completed 500-record "
            "jobs in this account -- and the price is now read rather than reasoned to: it lives "
            "under the AmazonBedrockFoundationModels service code, spelled "
            "USW2_InputTokenCount_Global_Batch, and comes out at exactly the halved list price "
            "this row used to carry as a guess. Rejects temperature and topP together, which is "
            "how one of those three jobs came back Completed with every record dead."
        ),
    ),
    RosterModel(
        model_id="global.anthropic.claude-opus-5",
        name="Anthropic Claude Opus 5",
        batch_price_in_per_mtok=2.5,
        batch_price_out_per_mtok=12.5,
        batch_price_verified=True,
        rejects_temperature_with_top_p=True,
        deprecates_temperature_and_top_p=True,
        max_tokens_limit=128_000,
        notes=(
            "The frontier arm, and the roster's ceiling by a factor of eight on output price. "
            "Batch support is confirmed twice over: the support page lists it under cross-region "
            "with us-west-2, and its price rows include Input/Output Tokens - Batch, Global, "
            "which is also where the prices here come from. Two things make it the row most "
            "likely to waste a job. It refuses temperature and topP outright rather than only "
            "together, so it is the only row with deprecates_temperature_and_top_p set. And "
            "adaptive thinking is on by default, billed against output at 12.5/Mtok, so its real "
            "cost per record runs well above what a prompt-length estimate suggests -- budget it "
            "from a measured run, not from the nominal figure. Do not budget it for games-shaped "
            "stimulus at all: on 2026-09-02 Anthropic's safety classifier refused ~97% of short "
            "matrix-game prompts (Converse stopReason content_filtered, 2,613 of 2,688 batch "
            "records, reproduced 18/20 on the live path), so its parsed rows on such prompts are a "
            "classifier-selected subset rather than a rate."
        ),
    ),
)


KNOWN_NOT_BATCH_CAPABLE = (
    "global.anthropic.claude-sonnet-5",
    "anthropic.claude-sonnet-5",
    "global.anthropic.claude-fable-5",
    "anthropic.claude-fable-5",
    "openai.gpt-5.6-sol",
    "openai.gpt-5.6-luna",
    "openai.gpt-5.6-terra",
)
"""Models asked about and answered NO, kept so the question is not reopened by guessing.

The first three pairs were each probed against ``CreateModelInvocationJob`` and rejected with
``Batch inference is not supported for the requested model.``, alongside a known-capable control
that was accepted in the same run. Luna and Terra are here from the support page rather than a probe;
they are listed because the reason to reach for them is the same reason Sol was reached for, and a
future session should read "answered no" rather than "not tried yet". Both spellings of each Claude
id appear because the bare and ``global.`` forms are what a caller would plausibly write.

This is not a denylist the code consults -- :func:`roster_model` already refuses everything outside
:data:`BATCH_ROSTER`, so nothing here is reachable. It is a record with a test on it, so adding one
of these rows back is a deliberate act that has to argue with the probe rather than a quiet edit.
The right way to overturn any of it is a fresh probe with a control, not a doc page.
"""


def vendor_namespace(model_id: str) -> str:
    """Return the vendor segment of a model id, skipping a cross-region geography prefix.

    ``us.meta.llama4-maverick-17b-instruct-v1:0`` and ``mistral.devstral-2-123b`` both answer with
    their vendor. Counting segments instead would not work: ``deepseek.v3.2``, ``zai.glm-4.7`` and
    ``minimax.minimax-m2.1`` all split into three parts on the dots inside their version numbers,
    the same shape as a genuinely prefixed id, so the prefix has to be recognised by name.
    """
    for prefix in _INFERENCE_PROFILE_PREFIXES:
        if model_id.startswith(prefix):
            return model_id[len(prefix) :].split(".", 1)[0]
    return model_id.split(".", 1)[0]


def assert_roster_ids_well_formed(roster: Sequence[RosterModel]) -> None:
    """Refuse a roster whose ids are duplicated or name no vendor Bedrock actually has.

    Both failures are silent where they happen and expensive later, which is why this runs at import
    rather than living only in a test. A duplicate id makes ``_ROSTER_BY_ID`` shorter than the tuple
    it was built from, so one row becomes unreachable while ``--models`` still offers it and the row
    that answers is whichever came last -- possibly at another price.

    The vendor check catches the two ways a prefix goes wrong, which look nothing alike from here
    and fail identically:

    * **A geography prefix missing from** :data:`_INFERENCE_PROFILE_PREFIXES`. ``ap.`` typed for
      ``apac.``, or ``au.`` before it was added there, leaves
      :attr:`RosterModel.is_inference_profile` false, so ``_assert_model_available`` asks
      ``get_foundation_model`` about an id that exists only as a profile and the submit dies at a
      call with nothing to do with the mistake. Here the leading segment is not a vendor either, so
      it is caught.
    * **A prefix with the vendor dropped.** ``us.claude-opus-5`` reads as a profile and gets as far
      as ``get_inference_profile``, so nothing about its shape is suspicious -- but the segment after
      the prefix has to be a vendor namespace and is not.
    """
    seen: dict[str, int] = {}
    for model in roster:
        seen[model.model_id] = seen.get(model.model_id, 0) + 1
    duplicated = sorted(name for name, count in seen.items() if count > 1)

    unknown_vendor = sorted(
        model.model_id
        for model in roster
        if vendor_namespace(model.model_id) not in _FOUNDATION_MODEL_VENDORS
    )
    if duplicated or unknown_vendor:
        raise RuntimeError(
            f"BATCH_ROSTER is malformed: duplicated ids {duplicated or 'none'}; ids naming no known "
            f"vendor {unknown_vendor or 'none'}. A vendor segment that reads as unknown is usually "
            f"a geography prefix missing from {_INFERENCE_PROFILE_PREFIXES}, or a prefixed id with "
            "the vendor left out. Either way the id resolves against the wrong Bedrock call and the "
            "submit fails somewhere unrelated to the typo."
        )


assert_roster_ids_well_formed(BATCH_ROSTER)

_ROSTER_BY_ID = {model.model_id: model for model in BATCH_ROSTER}


def roster_model(model_id: str) -> RosterModel:
    """Look up a model in the verified roster, refusing to guess for anything outside it.

    Guessing is what the roster exists to prevent. A model id can be wrong in ways the API answers
    cheerfully: the wrong prefix on a cross-region-only model, a model absent from the batch-support
    list, or a family whose per-model refusals (Anthropic's temperature/topP conflict) this module
    would then not know about. Each of those surfaces as a job that queues for ten minutes and comes
    back either failed or, worse, ``Completed`` with every record in error.

    Widening the sweep means adding a verified row here, which is a deliberate act with a citation
    attached, rather than passing a new string on a command line.
    """
    model = _ROSTER_BY_ID.get(model_id)
    if model is None:
        raise ValueError(
            f"{model_id!r} is not in the verified batch roster. Known ids: "
            f"{', '.join(sorted(_ROSTER_BY_ID))}. Batch support is not discoverable from "
            "list-foundation-models (its inferenceTypesSupported enum has no BATCH value), so a "
            "new model needs adding to BATCH_ROSTER with its id, prefix and price checked by hand."
        )
    return model


# The sampling labels a handle carries across to a resume or skip decision: exactly the trailing
# defaulted fields of BatchJobHandle. test_bedrock_batch pins that equality, so a new label cannot
# join the handle without joining the comparison both CLIs run before reusing paid inference.
SAMPLING_LABEL_FIELDS = ("max_tokens", "reasoning_effort")


@dataclass(frozen=True, slots=True)
class BatchJobHandle:
    """Everything needed to collect a submitted job from a fresh process.

    Written to disk at submit time so a job outlives the session that started it. Batch turnaround
    in this account runs 8.7 to 24.7 minutes with a fixed ~5-minute queueing floor, and a
    whole-roster sweep is submitted all at once, so "the process that submitted it is still alive"
    is not an assumption worth building on -- and re-submitting because the handle was lost means
    paying for the inference twice.

    ``prompt_digest`` is the tripwire on the join. It hashes the submitted prompts in order, and
    ``collect`` recomputes it from the prompts Bedrock echoes back in each record's ``modelInput``,
    sorted into ``recordId`` order. A digest match means the records that came back are the records
    that went out, in the order this handle claims. Nothing else in the round trip proves that: the
    output file's line order is genuinely shuffled (the first five recordIds of a real 500-record
    result were rec_000266, rec_000144, rec_000402, rec_000127, rec_000371).

    ``max_tokens`` and ``reasoning_effort`` carry the sampling labels across to the collect leg,
    which runs in a different process and has no other way to learn them: a cap below a model's
    thinking length truncates a reply before any marker can appear, so an unlabelled trace reports a
    capability absence that was a config choice. They are trailing and default to ``None``, so a
    handle written before they existed loads with ``None`` reading as "written before the field
    existed" rather than as a cap of zero.

    That is not a general backward-compatibility promise, and the oldest handles on disk are the
    counterexample: ``cell_digest`` was added without a default, so the 2026-08-17T00:27 directory
    does not load at all. Refusing it is right -- a handle with no cell digest cannot pass the
    corpus-drift tripwire ``collect`` runs, so there is nothing safe to do with it -- and
    :meth:`load` says which field is missing rather than raising a bare ``TypeError`` about a
    dataclass argument.
    """

    job_arn: str
    job_name: str
    model_id: str
    record_count: int
    prompt_digest: str
    cell_digest: str
    input_uri: str
    output_uri: str
    region: str
    profile: str | None
    submitted_at: str
    max_tokens: int | None = None
    reasoning_effort: str | None = None

    @property
    def job_id(self) -> str:
        """The job id Bedrock names the output subdirectory after."""
        return self.job_arn.rsplit("/", 1)[-1]

    @property
    def sampling_labels_unrecorded(self) -> bool:
        """Whether this handle predates the sampling labels: all ``None`` is unknown, not zero.

        The caller decides what unknown means -- the jagged sweep tolerates it (three handle
        directories on disk predate the fields) where the hatch probe, whose handles all carry
        them, treats it as any other mismatch. :meth:`sampling_label_mismatches` itself reports
        ``None`` as differing, so a tolerant caller checks this first.
        """
        return all(getattr(self, name) is None for name in SAMPLING_LABEL_FIELDS)

    def sampling_label_mismatches(
        self, *, max_tokens: int | None, reasoning_effort: str | None
    ) -> list[str]:
        """Name every sampling label recorded here that differs from what a re-run would sample at.

        The one comparison behind both CLIs' reuse-paid-inference checks. It matters because the
        records a collect writes are stamped from the *invocation's* config while the completions
        come from the *handle's* job, so reusing a job sampled at another cap or effort produces a
        trace whose every row is correctly labelled and whose comparison across rows is
        meaningless. Driven off :data:`SAMPLING_LABEL_FIELDS` so a label added to the handle joins
        this comparison mechanically.
        """
        requested = {"max_tokens": max_tokens, "reasoning_effort": reasoning_effort}
        return [
            f"{name}={getattr(self, name)!r} on the handle vs {requested[name]!r} requested"
            for name in SAMPLING_LABEL_FIELDS
            if getattr(self, name) != requested[name]
        ]

    def save(self, path: Path) -> Path:
        """Write this handle as JSON, creating the parent directory.

        The tracked-path refusal lives here in the writer rather than only in callers, because the
        handle carries the AWS account id, bucket and profile and a second unguarded call site is
        exactly how the last leak of this class happened (one writer, one guard, both callers
        covered -- :mod:`reward_hacking.trace`'s own lesson). Submit-first callers must still ask
        separately BEFORE the spend: by the time save runs the job exists and is paid for, so a
        refusal here alone would orphan it.
        """
        refuse_tracked_trace_path(path, carries=HANDLE_CARRIES)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        logger.info("wrote batch handle %s", path)
        return path

    @classmethod
    def load(cls, path: Path) -> BatchJobHandle:
        """Read a handle back from JSON, naming any field that does not line up with this class.

        A handle is often the only surviving reference to a paid job, so what a refusal says
        matters: ``cls(**payload)`` alone raises ``TypeError: __init__() missing 1 required
        positional argument: 'cell_digest'``, which reads as a code defect rather than as "this
        handle predates the corpus-drift tripwire". Both directions are named, since a renamed field
        looks the same from here as a missing one, and the ARN in the file is what an operator needs
        next.
        """
        payload = json.loads(path.read_text(encoding="utf-8"))
        declared = {field.name: field for field in fields(cls)}
        unknown = sorted(set(payload) - set(declared))
        # Which fields are optional is read off the dataclass rather than listed here, so adding a
        # trailing defaulted field cannot leave a second list of them to fall out of date.
        missing = sorted(
            name
            for name, field in declared.items()
            if name not in payload and field.default is MISSING and field.default_factory is MISSING
        )
        if unknown or missing:
            raise ValueError(
                f"{path} is not a handle this code can read: "
                f"missing {missing or 'nothing'}, unrecognised {unknown or 'nothing'}. "
                "A handle written before a required field existed cannot be collected (with no "
                "cell_digest it cannot pass the corpus-drift check), but the job it names may "
                "still be alive -- its ARN is in the file."
            )
        return cls(**payload)


def _digest(parts: Iterable[str]) -> str:
    r"""Hash an ordered sequence of strings, length-prefixed so no two sequences collide.

    A separator byte is not enough: with a NUL delimiter, ``["a\0b", "c"]`` and ``["a", "b\0c"]``
    hash identically. Prompts rendered from this corpus cannot contain a NUL today, which is exactly
    the kind of "cannot happen" that turns into a silent mis-join later, and the length prefix costs
    nothing.
    """
    digest = hashlib.sha256()
    for part in parts:
        encoded = part.encode("utf-8")
        digest.update(str(len(encoded)).encode("ascii"))
        digest.update(b":")
        digest.update(encoded)
    return digest.hexdigest()


def prompt_digest(prompts: Sequence[str]) -> str:
    """Hash prompts in order, so a submitted list and a collected one can be compared cheaply."""
    return _digest(prompts)


def cell_digest(cells: Sequence[Mapping[str, Any]]) -> str:
    """Hash the cell labels in order, so a corpus edit the prompts do not show is still caught.

    The prompt digest alone is not enough, and the gap is subtle. Two different cells can render
    byte-identical prompts -- renaming an item, reordering the corpus, or swapping which arm a
    prompt belongs to all change what a response *means* without changing a character of what was
    sent. A collector re-derives the labels locally and attaches them positionally, so a change like
    that would attribute real responses to the wrong items or arms and grade them accordingly,
    with the prompt digest still matching. Hashing the labels closes that.

    **Comparing this digest is the caller's obligation, not this backend's.** ``collect`` compares
    only the prompt digest, because the backend never sees the corpus and so has nothing to
    re-derive labels from. ``reward_hacking/jagged/sweep.py::collect`` is the reference
    implementation: it re-renders the cells from the current corpus and raises on either digest.

    Each cell is hashed as the canonical serialization of its **whole content** -- every key and
    every value, keys sorted -- rather than a fixed set of indexed keys. The backend has two
    callers with legitimately different metadata shapes (the jagged sweep's item/dimension/arm/
    repeat, the hatch probe's cell/group/sample), so the schema cannot live here; hashing the full
    content keeps the digest schema-free while staying strictly more sensitive than any key list.
    The degeneration a defaulted key list invites -- other key names hashing every cell to one
    constant, matching on both sides because both ran the same builder -- cannot arise, because
    whatever keys the builder wrote are what gets hashed. A value ``json`` cannot serialize raises
    rather than being coerced into something that would compare equal across edits.
    """
    return _digest(json.dumps(dict(cell), sort_keys=True, separators=(",", ":")) for cell in cells)


def _record_id(index: int) -> str:
    """Spell a record id from its position in the submitted list.

    Positional rather than composed from item, arm and repeat, and that is the safety property, not
    a shortcut. A composed key mis-joins *silently* when the composition has a bug, producing a
    plausible trace with arms attached to the wrong responses -- the exact class of failure this
    repo keeps getting bitten by. A position is checked by one set-equality assertion that no
    wrong-but-plausible mapping can satisfy. The spelling matches the completed jobs already in this
    account.
    """
    return f"rec_{index:06d}"


def _split_s3_uri(uri: str) -> tuple[str, str]:
    """Split an ``s3://bucket/key`` URI into bucket and key."""
    if not uri.startswith("s3://"):
        raise ValueError(f"not an S3 URI: {uri!r}")
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    return bucket, key


def _now() -> str:
    return datetime.now(UTC).isoformat()


class BedrockBatchBackend:
    """Submit JaggedBench prompts as a Bedrock batch job and collect them back as completions.

    Satisfies the ``Backend`` and ``DetailedBackend`` protocols, so the JaggedBench runner, graders,
    analysis and trace schema need no changes at all -- ``generate_detailed`` returns
    ``BedrockCompletion`` objects in prompt order exactly as the live backend does. But the useful
    interface is the split one:

    * ``submit(prompts)`` writes the input JSONL and a self-describing sidecar to S3, creates the
      job, and returns a ``BatchJobHandle`` to persist.
    * ``collect(handle)`` polls to completion, downloads, validates, joins on ``recordId``, and
      returns completions in submitted order.
    * ``generate_detailed(prompts)`` is ``collect(submit(prompts))``, which satisfies the protocol
      and blocks for the whole 10-25 minutes.

    Use the split path for a sweep. N submits followed by N collects runs all N jobs concurrently,
    which is the entire reason batch beats the Converse path across a roster; the blocking form
    would serialise them and be slower than Converse for every model.

    Every validation in here corresponds to a failure observed in this account's existing job
    history rather than an imagined one, and each is exercised against real recorded artefacts in
    ``tests/test_bedrock_batch.py``.
    """

    transport = TRANSPORT

    def __init__(  # noqa: PLR0913 - all keyword-only with defaults; no call site passes them all
        self,
        model_id: str,
        *,
        region: str = DEFAULT_BEDROCK_REGION,
        profile: str | None = None,
        sampling: BedrockSamplingConfig | None = None,
        role_arn: str | None = None,
        bucket: str | None = None,
        bucket_owner: str | None = None,
        prefix: str = BATCH_PREFIX,
        run_id: str | None = None,
    ) -> None:
        """Resolve the model against the roster and freeze the request shape before any spend.

        ``profile``, ``role_arn``, ``bucket`` and ``bucket_owner`` left as ``None`` are read from
        the environment (see the module docstring), which raises with the variable's name rather
        than reaching AWS with a placeholder. An empty ``profile`` is not the same as ``None``: it
        means the ambient credential chain, which is what a Batch container with an instance role
        has and what the CLI's ``--profile ''`` asks for.
        """
        self.model_id = model_id
        self.roster = roster_model(model_id)
        self.sampling = sampling or BedrockSamplingConfig()
        self.region = region
        self.profile = profile if profile is not None else batch_profile()
        self.role_arn = role_arn if role_arn is not None else batch_role_arn()
        self.bucket = bucket if bucket is not None else batch_bucket()
        # Travels with the bucket rather than pinned to the role's account, so a bucket in another
        # account cannot upload fine and then be rejected at job creation, objects left behind.
        self.bucket_owner = (
            bucket_owner if bucket_owner is not None else batch_bucket_owner(self.role_arn)
        )
        self.prefix = prefix
        self.run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.usage = TokenUsage()

        self._check_sampling_against_model()
        self._inference_config = converse_inference_config(self.sampling)
        self._extra_request_fields = converse_extra_request_fields(model_id, self.sampling)
        if self._extra_request_fields is not None:
            logger.warning(
                "reasoning effort %r is being sent as additionalModelRequestFields on the batch "
                "path, where it is documented but unverified, and GPT-OSS ignores unknown fields "
                "without error. Confirm it took effect by comparing reasoning-token distributions, "
                "not by the job succeeding",
                self.sampling.reasoning_effort,
            )

        boto3 = importlib.import_module("boto3")
        session = boto3.Session(profile_name=self.profile or None, region_name=region)
        self._bedrock = session.client("bedrock")
        self._s3 = session.client("s3")
        logger.info(
            "BedrockBatchBackend ready for %s (%s) region=%s profile=%s run_id=%s",
            model_id,
            self.roster.name,
            region,
            self.profile,
            self.run_id,
        )

    def _check_sampling_against_model(self) -> None:
        """Refuse a sampling config this model rejects, before a job exists to reject it.

        The Anthropic case is the one that has already happened here: the job reports ``Completed``
        and every record carries a 400. Catching it at construction costs nothing; catching it from
        the results costs the queue time and reads, at a glance, like a successful run.

        The ``maxTokens`` ceiling is the same archetype, and no longer a hypothetical one:
        ``converse_inference_config`` writes ``maxTokens`` unconditionally and ``jagged/sweep.py``
        feeds it one flat ``--max-new-tokens`` for a whole roster, so ``--models
        us.amazon.nova-micro-v1:0 --max-new-tokens 24576`` submitted a job Nova rejects per record.
        Once the ceilings were read off the API for every row, three turned out to sit below the
        shared 30,000-token default -- both Llama 4 models at 8,192 and Nova Micro at 10,000 -- with
        Qwen3 32B the tightest row that clears it, by 2,768. So a roster-wide sweep at one flat cap
        now refuses here instead of paying for jobs that can only come back dead. That refusal is
        the check working; the fix is a per-model cap, not a lower flat one.

        Only the batch transport is checked here: the live Converse backend cannot consult the
        roster without inverting this module's import direction, it legitimately serves ids the
        roster does not hold, and an over-limit live request fails loudly in seconds with a botocore
        error.
        """
        limit = self.roster.max_tokens_limit
        if self.sampling.max_tokens > limit:
            raise ValueError(
                f"{self.model_id} hard-rejects maxTokens above {limit}; this run asked for "
                f"{self.sampling.max_tokens}. The job would report Completed with every record "
                "failed, after burning the queue time to find out."
            )
        # Checked before the pair-only rule below, which it implies: the stricter message names the
        # knob that is actually unusable rather than suggesting the caller drop one of two.
        if self.roster.deprecates_temperature_and_top_p and (
            self.sampling.temperature is not None or self.sampling.top_p is not None
        ):
            raise ValueError(
                f"{self.model_id} deprecates both temperature and topP and refuses a request "
                "carrying either, per record: the job would report Completed with every record "
                "failed. Set neither --temperature nor --top-p for this model. Only the no-op "
                "values pass, so there is no setting of them that would sample differently."
            )
        both_set = self.sampling.temperature is not None and self.sampling.top_p is not None
        if both_set and self.roster.rejects_temperature_with_top_p:
            raise ValueError(
                f"{self.model_id} rejects temperature and topP together, and does so per record: "
                "the job would report Completed with every record failed. Set exactly one of "
                "--temperature and --top-p, or neither."
            )

    def _assert_model_available(self) -> None:
        """Confirm the id resolves and is ACTIVE, so a bad prefix fails before the job exists."""
        if self.roster.is_inference_profile:
            profile = self._bedrock.get_inference_profile(inferenceProfileIdentifier=self.model_id)
            status = profile["status"]
        else:
            details = self._bedrock.get_foundation_model(modelIdentifier=self.model_id)
            status = details["modelDetails"]["modelLifecycle"]["status"]
        if status != "ACTIVE":
            raise RuntimeError(f"{self.model_id} resolves but its status is {status!r}, not ACTIVE")
        logger.info("model %s resolves ACTIVE in %s", self.model_id, self.region)

    def _model_slug(self) -> str:
        """Build a filesystem- and S3-safe stem for this model's artefacts."""
        return self.model_id.replace(":", "-").replace(".", "-").replace("/", "-")

    def _job_name(self) -> str:
        """Build a job name inside Bedrock's 63-character limit, truncating the model, not the run.

        Truncating from the right drops the run id, which is the only unique part: the longest
        roster id makes a 71-character name and cutting it to 63 leaves ``...-20260817``, so two
        Haiku runs on the same day would ask Bedrock for the same job name.

        Two ids now truncate rather than one -- Haiku 4.5 and Llama 4 Maverick -- and their truncated
        stems are still distinct, checked across the whole roster. A third long id sharing a 46-
        character prefix with either would collide silently, both models asking for one job name, so
        that is the thing to check when adding one rather than the 63-character limit itself.
        """
        suffix = f"-{self.run_id}".lower()
        stem = f"jagged-{self._model_slug()}".lower()
        return stem[: MAX_JOB_NAME_CHARS - len(suffix)] + suffix

    def _s3_base(self) -> str:
        return f"s3://{self.bucket}/{self.prefix}/{self.run_id}/{self._model_slug()}"

    def _put(self, uri: str, body: str) -> None:
        bucket, key = _split_s3_uri(uri)
        self._s3.put_object(Bucket=bucket, Key=key, Body=body.encode("utf-8"))
        logger.info("uploaded %d bytes to %s", len(body), uri)

    def _get(self, uri: str) -> str:
        bucket, key = _split_s3_uri(uri)
        response = self._s3.get_object(Bucket=bucket, Key=key)
        return response["Body"].read().decode("utf-8")

    def _check_record_count(self, count: int) -> None:
        """Refuse a job below the hard 100-record floor, saying what would clear it.

        The floor is per job and Service Quotas marks it not adjustable, so this is arithmetic, not
        policy. The message carries the arithmetic because the alternative is a ValidationException
        from the API that says only that 100 is the minimum.
        """
        if count >= MIN_BATCH_RECORDS:
            return
        if count == 0:
            # Reachable as --limit 0; no repeat count multiplies zero up to the floor.
            raise ValueError(
                "this run renders no prompts at all, so there is nothing to submit; check --limit"
            )
        # A multiplier on whatever --repeats already is, not an absolute value: this sees only the
        # record count, so a run already at --repeats 3 needs 3 x this, not this.
        multiplier = -(-MIN_BATCH_RECORDS // count)
        raise ValueError(
            f"batch inference needs at least {MIN_BATCH_RECORDS} records per job and this run has "
            f"{count}. The minimum is per job and not adjustable. Either multiply --repeats by "
            f"{multiplier} (taking this run to {count * multiplier} records) or run it on the live "
            "Converse backend, which has no floor and finishes a run this size in under a minute. "
            "Calibration runs belong on Converse."
        )

    def submit(
        self, prompts: list[str], *, metadata: Sequence[Mapping[str, Any]] | None = None
    ) -> BatchJobHandle:
        """Write the input file and sidecar to S3, create the job, and return its handle.

        ``metadata`` is one mapping per prompt describing the cell it came from (item, dimension,
        arm, repeat). It never travels to Bedrock -- ``recordId`` carries position only -- and is
        written beside the input file purely so the S3 artefacts stay self-describing after the
        local trace is gone. That follows the repo's habit of retaining raw material so that adding
        rigor later is a re-analysis rather than a re-run.

        **The idempotency token is scoped to (run_id, model, prompts), and the scope is the whole
        point.** ``clientRequestToken`` is a real idempotency token in this API shape, but
        botocore's auto-injected UUID is generated per call, so it protects only botocore's own
        retries. An operator re-run -- or a re-run after the connection dropped between the service
        accepting the job and the handle being written -- pays twice for the inference, the outcome
        ``BatchJobHandle``'s docstring exists to prevent and which ``jagged/sweep.py`` documents
        ``--run-id`` reuse as the recovery route for. Deriving the token from the job name (which
        carries the run id and the model) plus the prompt digest makes that recovery route free.
        Deriving it from the content alone would be worse than the problem: two deliberate draws of
        the same corpus at the same decoding config, which is what a second sample at temperature
        above zero is, would collapse into one job and hand back the first draw's completions. That
        is a corrupted measurement, where the double charge is only money. Including the prompt
        digest means a reused ``--run-id`` over an edited corpus is still a new job rather than a
        silent return of a job built from other prompts.
        """
        self._check_record_count(len(prompts))
        if metadata is not None and len(metadata) != len(prompts):
            raise ValueError(
                f"metadata has {len(metadata)} entries for {len(prompts)} prompts; it is "
                "positional, so a length mismatch means the labels would be attached to the wrong "
                "cells"
            )
        self._assert_model_available()

        base = self._s3_base()
        input_uri = f"{base}/input.jsonl"
        output_uri = f"{base}/output/"
        submitted_prompts = prompt_digest(prompts)
        submitted_cells = cell_digest(metadata or [])
        records = [
            {
                "recordId": _record_id(index),
                "modelInput": converse_request(
                    prompt, self._inference_config, self._extra_request_fields
                ),
            }
            for index, prompt in enumerate(prompts)
        ]
        self._put(input_uri, "".join(json.dumps(record) + "\n" for record in records))
        self._put(
            f"{base}/sidecar.json",
            json.dumps(
                {
                    "model_id": self.model_id,
                    "model_name": self.roster.name,
                    "transport": TRANSPORT,
                    "run_id": self.run_id,
                    "record_count": len(prompts),
                    "prompt_digest": submitted_prompts,
                    "cell_digest": submitted_cells,
                    "inference_config": self._inference_config,
                    "additional_model_request_fields": self._extra_request_fields,
                    "submitted_at": _now(),
                    "cells": [
                        {"record_id": _record_id(index), **dict(entry)}
                        for index, entry in enumerate(metadata or [])
                    ],
                },
                indent=2,
            ),
        )

        job_name = self._job_name()
        response = self._bedrock.create_model_invocation_job(
            jobName=job_name,
            roleArn=self.role_arn,
            modelId=self.model_id,
            # (run_id, model, prompts) via the job name, not content alone -- see the docstring.
            clientRequestToken=_digest((job_name, submitted_prompts)),
            # Not the default. The default is InvokeModel, under which every record here fails.
            modelInvocationType="Converse",
            inputDataConfig={
                "s3InputDataConfig": {
                    "s3InputFormat": "JSONL",
                    "s3Uri": input_uri,
                    "s3BucketOwner": self.bucket_owner,
                }
            },
            outputDataConfig={
                "s3OutputDataConfig": {
                    "s3Uri": output_uri,
                    "s3BucketOwner": self.bucket_owner,
                }
            },
            timeoutDurationInHours=BATCH_TIMEOUT_HOURS,
        )
        handle = BatchJobHandle(
            job_arn=response["jobArn"],
            job_name=job_name,
            model_id=self.model_id,
            record_count=len(prompts),
            prompt_digest=submitted_prompts,
            cell_digest=submitted_cells,
            input_uri=input_uri,
            output_uri=output_uri,
            region=self.region,
            profile=self.profile,
            submitted_at=_now(),
            max_tokens=self.sampling.max_tokens,
            reasoning_effort=self.sampling.reasoning_effort,
        )
        logger.info(
            "submitted %d records to %s as %s (%s)",
            handle.record_count,
            self.model_id,
            handle.job_name,
            handle.job_arn,
        )
        return handle

    def describe(self, handle: BatchJobHandle) -> dict[str, Any]:
        """Fetch the job's current state, unmodified."""
        return self._bedrock.get_model_invocation_job(jobIdentifier=handle.job_arn)

    def wait(
        self,
        handle: BatchJobHandle,
        *,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        timeout_seconds: float = DEFAULT_WAIT_SECONDS,
    ) -> dict[str, Any]:
        """Poll until the job reaches a terminal state, then return its final description.

        A timeout here raises without stopping the job, because the job is the expensive part and it
        stays collectable from the same handle afterwards. Losing patience must not mean losing the
        inference.
        """
        deadline = time.monotonic() + timeout_seconds
        while True:
            job = self.describe(handle)
            status = job["status"]
            if status in _TERMINAL_STATUSES:
                logger.info("job %s reached %s", handle.job_name, status)
                return job
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"job {handle.job_name} was still {status} after {timeout_seconds:.0f}s. The "
                    "job has NOT been stopped and is still collectable from this handle "
                    f"({handle.job_arn}); re-run the collect step later."
                )
            logger.info(
                "job %s is %s; polling again in %.0fs", handle.job_name, status, poll_seconds
            )
            time.sleep(poll_seconds)

    def _output_uris(self, handle: BatchJobHandle) -> tuple[str, str]:
        """Locate the result and manifest objects Bedrock writes under the job id."""
        base = f"{handle.output_uri.rstrip('/')}/{handle.job_id}"
        input_name = handle.input_uri.rsplit("/", 1)[-1]
        return f"{base}/{input_name}.out", f"{base}/manifest.json.out"

    def _check_completion(self, handle: BatchJobHandle, job: Mapping[str, Any]) -> TokenUsage:
        """Refuse anything short of every record succeeding, and return the job's token totals.

        ``Completed`` does not mean "worked". A job in this account reported ``Completed`` with
        ``successRecordCount`` 0 and ``errorRecordCount`` 500. The per-record counts are the only
        thing that says whether the inference happened, and they live in two places -- on the job
        for newer jobs and in ``manifest.json.out`` always -- so both are read and cross-checked
        rather than trusting whichever is more convenient.
        """
        status = job["status"]
        if status != "Completed":
            raise RuntimeError(
                f"job {handle.job_name} ended {status}: {job.get('message', '<no message>')}"
            )

        manifest = json.loads(self._get(self._output_uris(handle)[1]))
        counts = {
            key: manifest[key]
            for key in ("totalRecordCount", "successRecordCount", "errorRecordCount")
        }
        for key, manifest_value in counts.items():
            job_value = job.get(key)
            if job_value is not None and job_value != manifest_value:
                raise RuntimeError(
                    f"job {handle.job_name} reports {key}={job_value} but its manifest reports "
                    f"{manifest_value}; the two disagree, so neither can be trusted"
                )
        if counts["successRecordCount"] != handle.record_count or counts["errorRecordCount"]:
            raise RuntimeError(
                f"job {handle.job_name} reports status Completed but only "
                f"{counts['successRecordCount']} of {handle.record_count} records succeeded "
                f"({counts['errorRecordCount']} errored). A Completed job with failed records is "
                "how a request-shape mistake looks from the outside; read the per-record error "
                f"messages in {self._output_uris(handle)[0]}"
            )
        # Summed the same way ``_sum_usage`` sums a record, so the two totals are comparable: the
        # job-level manifest splits cached input out into its own counters just as a record does.
        return TokenUsage(
            input_tokens=sum(
                manifest.get(field) or 0
                for field in (
                    "inputTokenCount",
                    "cacheReadInputTokenCount",
                    "cacheWriteInputTokenCount",
                )
            ),
            output_tokens=manifest["outputTokenCount"],
        )

    def _join_records(
        self, handle: BatchJobHandle, body: str
    ) -> tuple[list[BedrockCompletion], list[str]]:
        """Join result lines onto submitted positions by ``recordId``, and return them in order.

        Never by line order: the output file's order is genuinely shuffled. The set of returned ids
        must equal the set submitted, exactly -- no missing, no extra, no duplicates -- and a record
        carrying an ``error`` instead of a ``modelOutput`` raises rather than becoming an empty
        completion. An empty completion would grade as "the model did not make the move", which is a
        behavioural claim; reporting a transport failure that way is the worst outcome available.
        """
        by_id: dict[str, dict[str, Any]] = {}
        # Split on the newline only. ``str.splitlines`` also breaks on U+0085, U+2028 and the other
        # Unicode line separators, which JSON leaves unescaped inside strings and which a prompt
        # quoting terminal output does carry: the first TMAX judge job came back with eight U+0085s
        # in 1,200 records, and splitlines cut those records mid-string into unparseable fragments.
        for line in body.split("\n"):
            if not line.strip():
                continue
            record = json.loads(line)
            record_id = record["recordId"]
            if record_id in by_id:
                raise RuntimeError(
                    f"job {handle.job_name} returned {record_id} more than once, so the join is "
                    "ambiguous and some submitted prompt has no result of its own"
                )
            by_id[record_id] = record

        expected = {_record_id(index) for index in range(handle.record_count)}
        if by_id.keys() != expected:
            missing = sorted(expected - by_id.keys())
            extra = sorted(by_id.keys() - expected)
            raise RuntimeError(
                f"job {handle.job_name} returned a different record set than was submitted: "
                f"{len(missing)} missing ({missing}), {len(extra)} unexpected ({extra})"
            )

        failures = {
            record_id: record["error"]
            for record_id, record in by_id.items()
            if record.get("error") is not None
        }
        if failures:
            first_id, first_error = next(iter(sorted(failures.items())))
            raise RuntimeError(
                f"job {handle.job_name} returned {len(failures)} error records; {first_id} failed "
                f"with code {first_error.get('errorCode')}: {first_error.get('errorMessage')}"
            )

        completions: list[BedrockCompletion] = []
        prompts: list[str] = []
        for index in range(handle.record_count):
            record = by_id[_record_id(index)]
            completions.append(parse_converse_output(record["modelOutput"]))
            prompts.append(record["modelInput"]["messages"][0]["content"][0]["text"])
        return completions, prompts

    def collect(
        self,
        handle: BatchJobHandle,
        *,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        timeout_seconds: float = DEFAULT_WAIT_SECONDS,
    ) -> list[BedrockCompletion]:
        """Wait for the job, download its results, validate them, and return them in order.

        The model id is the one join key everything else here does not derive from the handle: the
        prices and the ``PRICE UNVERIFIED`` annotation on the cost line come off ``self.roster``. So
        a handle collected through the wrong backend -- realistic in an interactive session holding
        one backend and several saved handles -- would attribute the responses to a model that never
        ran them and price them off the wrong roster row.
        """
        if handle.model_id != self.model_id:
            raise ValueError(
                f"handle {handle.job_name} was submitted for {handle.model_id} but this backend is "
                f"the wrong model ({self.model_id}); collecting it here would price the run off "
                "the wrong roster row and credit the responses to a model that never ran them"
            )
        job = self.wait(handle, poll_seconds=poll_seconds, timeout_seconds=timeout_seconds)
        job_usage = self._check_completion(handle, job)
        completions, prompts = self._join_records(handle, self._get(self._output_uris(handle)[0]))

        returned_digest = prompt_digest(prompts)
        if returned_digest != handle.prompt_digest:
            raise RuntimeError(
                f"job {handle.job_name} echoed back prompts whose digest is {returned_digest} but "
                f"this handle was submitted with {handle.prompt_digest}. The results do not belong "
                "to this handle, or the record ordering is not what the handle claims; either way "
                "the join would attach responses to the wrong cells"
            )

        record_usage = sum((completion.usage for completion in completions), TokenUsage())
        for label, from_records, from_manifest in (
            ("input", record_usage.input_tokens, job_usage.input_tokens),
            ("output", record_usage.output_tokens, job_usage.output_tokens),
        ):
            if from_records != from_manifest:
                logger.warning(
                    "%s tokens summed from the records give %d but the job manifest reports %d. "
                    "The per-record figure is the one used below, since it is the one that can be "
                    "attributed to a cell, but the disagreement is unexplained so treat this run's "
                    "cost as approximate. A per-record total of 0 would mean the usage counters "
                    "came back null and nothing was accounted at all",
                    label,
                    from_records,
                    from_manifest,
                )
        self.usage += record_usage
        logger.info(
            "collected %d completions from %s | in=%d out=%d tokens, about $%.4f at batch prices%s",
            len(completions),
            handle.job_name,
            record_usage.input_tokens,
            record_usage.output_tokens,
            self.roster.estimated_cost_usd(record_usage),
            "" if self.roster.batch_price_verified else " (PRICE UNVERIFIED for this model)",
        )
        return completions

    def generate_detailed(self, prompts: list[str]) -> list[BedrockCompletion]:
        """Submit and then block until collected, satisfying the detailed-backend protocol.

        Correct but rarely what you want: it blocks for the full 10-25 minutes, and running a roster
        this way serialises jobs that could all have been in flight at once. Use ``submit`` and
        ``collect`` for a sweep.
        """
        return self.collect(self.submit(prompts))

    def generate(self, prompts: list[str]) -> list[str]:
        """Generate one completion per prompt, in submitted order."""
        return [completion.text for completion in self.generate_detailed(prompts)]
