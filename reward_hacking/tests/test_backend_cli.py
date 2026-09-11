"""Offline tests for the shared ``--backend`` plumbing (``reward_hacking/backend_cli.py``).

Everything here runs on CPU with no weights, no ``boto3``, and no credentials. The Bedrock branch
is checked by recording the ``build_backend`` call instead of making it, which is the only way to
assert *which config type* a backend was handed without a live client: ``BedrockBackend`` builds a
boto3 session in ``__init__``. What that recording proves is the whole point of the module -- the
Converse path gets a ``BedrockSamplingConfig`` and the local path a ``SamplingConfig``, never the
other way round, which would drop ``top_k`` and ``do_sample`` in silence.

Half of these tests are the sabotage the repo rule asks for: each one commits the exact mistake a
refusal exists to catch (``--top-k`` at Converse, ``--reasoning-effort`` at local weights, an
activation probe pointed at a hosted endpoint) and requires the error. Delete a knob's entry from
``_KNOBS`` and the matching test goes red.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest

from reward_hacking import backend_cli, model_backend
from reward_hacking.channel import model_policy
from reward_hacking.episodes import runner
from reward_hacking.harness import loop
from reward_hacking.interp import directions
from reward_hacking.model_backend import (
    DEFAULT_CODEX_MODEL,
    BedrockSamplingConfig,
    CodexBackend,
    MockBackend,
    SamplingConfig,
)
from reward_hacking.recoverybench.budgets import MAX_TOKENS_BY_MODEL

if TYPE_CHECKING:
    from reward_hacking.model_backend import Backend

HOSTED_MODEL = "openai.gpt-oss-120b-1:0"
LOCAL_MODEL = "Qwen/Qwen3.5-0.8B"

HOSTED_CAP = backend_cli.output_floor_for(HOSTED_MODEL) + 1000
"""An output cap to hand these CLIs: above the model's measured floor, below nothing in particular.

Derived from the floor rather than written out, because a literal here would silently fall under it
the day that model's measured budget is raised -- and then this test would be asserting that the
harness accepts a truncating cap, which is the opposite of what it pins. The offset keeps it
distinguishable from the floor itself and from the unset default.
"""


def _sampling_base(**overrides: Any) -> SamplingConfig:
    """A local sampling config for a test that is about some field other than the output cap.

    ``SamplingConfig`` has no default cap on purpose -- a cap nobody chose is the failure its class
    docstring describes -- so even a test uninterested in the cap has to name one. It borrows the
    non-thinking preset's rather than inventing a number, so nothing here becomes a second opinion
    about how long a completion may be.
    """
    return SamplingConfig(
        max_new_tokens=SamplingConfig.for_thinking(thinking=False).max_new_tokens, **overrides
    )


@dataclass
class _RecordedBuild:
    """One intercepted ``build_backend`` call: which kind, which model, which config objects."""

    kind: str
    model_id: str
    kwargs: dict[str, Any] = field(default_factory=dict)


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> list[_RecordedBuild]:
    """Replace ``build_backend`` with a recorder: no client, no weights, no credentials."""
    calls: list[_RecordedBuild] = []

    def fake_build_backend(kind: str, model_id: str, **kwargs: Any) -> Backend:
        calls.append(_RecordedBuild(kind=kind, model_id=model_id, kwargs=kwargs))
        return MockBackend(["recorded"], model_id=model_id)

    monkeypatch.setattr(backend_cli, "build_backend", fake_build_backend)
    return calls


ParseArgs = Callable[..., "argparse.Namespace"]
BuildBackend = Callable[["argparse.Namespace"], "Backend"]

# One table so a new CLI joining the pattern is one line and inherits every test below.
SAMPLING_CLIS = [
    pytest.param(runner._parse_args, runner._build_cli_backend, id="episodes"),
    pytest.param(model_policy._parse_args, model_policy._build_cli_backend, id="channel"),
    pytest.param(loop._parse_args, loop._build_cli_backend, id="agent-harness"),
]

# Every CLI on this plumbing can also run fully offline against canned completions.
MOCKABLE_CLIS = SAMPLING_CLIS


@pytest.mark.parametrize(("parse", "build"), SAMPLING_CLIS)
class TestEveryCliReachesEveryBackend:
    """Each sampling CLI accepts ``--backend`` and builds the config that backend actually takes."""

    def test_bedrock_gets_a_bedrock_sampling_config(
        self, parse: ParseArgs, build: BuildBackend, recorded: list[_RecordedBuild]
    ) -> None:
        """Every knob reaches the Converse config, and reaches it as the type that honours it.

        The cap used to be 64 here, which the agent harness now refuses outright as truncation (see
        ``backend_cli.refuse_short_output_cap``). ``HOSTED_CAP`` clears that model's measured floor
        while differing from both the floor and the unset default, so it still pins pass-through
        rather than passing on a coincidence.
        """
        build(
            parse(
                [
                    "--backend",
                    "bedrock",
                    "--model-id",
                    HOSTED_MODEL,
                    "--temperature",
                    "0.5",
                    "--max-new-tokens",
                    str(HOSTED_CAP),
                    "--concurrency",
                    "4",
                    "--reasoning-effort",
                    "high",
                ]
            )
        )
        call = recorded[0]
        assert call.kind == "bedrock"
        assert call.model_id == HOSTED_MODEL
        sampling = call.kwargs["sampling"]
        assert isinstance(sampling, BedrockSamplingConfig)
        assert sampling.max_tokens == HOSTED_CAP
        assert sampling.temperature == 0.5
        assert sampling.top_p is None  # unset stays out of the Converse request
        assert sampling.reasoning_effort == "high"
        assert call.kwargs["concurrency"] == 4
        # thinking is a local chat-template switch and must never reach the Converse backend.
        assert "thinking" not in call.kwargs

    def test_bedrock_defaults_omit_the_optional_inference_fields(
        self, parse: ParseArgs, build: BuildBackend, recorded: list[_RecordedBuild]
    ) -> None:
        build(parse(["--backend", "bedrock", "--model-id", HOSTED_MODEL]))
        sampling = recorded[0].kwargs["sampling"]
        assert isinstance(sampling, BedrockSamplingConfig)
        assert sampling.temperature is None
        assert sampling.top_p is None
        assert sampling.reasoning_effort is None
        assert sampling.max_tokens == BedrockSamplingConfig().max_tokens
        # No region/profile given, so the backend keeps its own defaults instead of being told None.
        assert "region" not in recorded[0].kwargs
        assert "profile" not in recorded[0].kwargs

    def test_local_backend_gets_a_huggingface_sampling_config(
        self, parse: ParseArgs, build: BuildBackend, recorded: list[_RecordedBuild]
    ) -> None:
        build(
            parse(
                [
                    "--backend",
                    "hf",
                    "--model-id",
                    LOCAL_MODEL,
                    "--top-p",
                    "1.0",
                    "--top-k",
                    "0",
                    "--temperature",
                    "1.5",
                ]
            )
        )
        call = recorded[0]
        assert call.kind == "hf"
        sampling = call.kwargs["sampling"]
        assert isinstance(sampling, SamplingConfig)
        assert sampling.temperature == 1.5
        assert sampling.top_p == 1.0
        assert sampling.top_k == 0  # a 0 the user typed, not "unset"
        assert call.kwargs["thinking"] is False

    def test_codex_gets_the_effort_ladder_and_nothing_else(
        self, parse: ParseArgs, build: BuildBackend, recorded: list[_RecordedBuild]
    ) -> None:
        build(parse(["--backend", "codex", "--reasoning-effort", "medium"]))
        call = recorded[0]
        assert call.kind == "codex"
        assert call.kwargs == {"reasoning_effort": "medium"}

    def test_empty_profile_means_the_ambient_credential_chain(
        self, parse: ParseArgs, build: BuildBackend, recorded: list[_RecordedBuild]
    ) -> None:
        build(parse(["--backend", "bedrock", "--model-id", HOSTED_MODEL, "--profile", ""]))
        assert recorded[0].kwargs["profile"] is None


@pytest.mark.parametrize(("parse", "build"), SAMPLING_CLIS)
class TestWrongCombinationsFailLoudly:
    """The sabotage tests: commit the mistake each refusal exists for, and require the error.

    None of these reach ``build_backend`` at all, which is why they need no recorder fixture: the
    refusal happens before anything is constructed, so a mis-specified run cannot spend a token.
    """

    def test_top_k_is_refused_at_bedrock(self, parse: ParseArgs, build: BuildBackend) -> None:
        with pytest.raises(ValueError, match="topK"):
            build(parse(["--backend", "bedrock", "--model-id", HOSTED_MODEL, "--top-k", "20"]))

    def test_thinking_is_refused_at_bedrock(self, parse: ParseArgs, build: BuildBackend) -> None:
        with pytest.raises(ValueError, match="--thinking"):
            build(parse(["--backend", "bedrock", "--model-id", HOSTED_MODEL, "--thinking"]))

    def test_explicit_no_thinking_is_refused_at_bedrock(
        self, parse: ParseArgs, build: BuildBackend
    ) -> None:
        # Refused even though it names the local default: silence would imply Converse honoured it.
        with pytest.raises(ValueError, match="--thinking"):
            build(parse(["--backend", "bedrock", "--model-id", HOSTED_MODEL, "--no-thinking"]))

    def test_decoding_knobs_are_refused_at_codex(
        self, parse: ParseArgs, build: BuildBackend
    ) -> None:
        with pytest.raises(ValueError, match="--temperature"):
            build(parse(["--backend", "codex", "--temperature", "0.5"]))

    def test_reasoning_effort_is_refused_at_local_weights(
        self, parse: ParseArgs, build: BuildBackend
    ) -> None:
        with pytest.raises(ValueError, match="--reasoning-effort"):
            build(
                parse(["--backend", "hf", "--model-id", LOCAL_MODEL, "--reasoning-effort", "high"])
            )

    @pytest.mark.parametrize(("flag", "value"), [("--concurrency", "8"), ("--region", "us-east-1")])
    def test_hosted_client_knobs_are_refused_at_local_weights(
        self, parse: ParseArgs, build: BuildBackend, flag: str, value: str
    ) -> None:
        with pytest.raises(ValueError, match=flag):
            build(parse(["--backend", "hf", "--model-id", LOCAL_MODEL, flag, value]))


class TestMockBackendPath:
    """``--backend mock`` drives a whole CLI with no weights, or says why it cannot."""

    @pytest.mark.parametrize(("parse", "build"), MOCKABLE_CLIS)
    def test_mock_needs_no_model(self, parse: ParseArgs, build: BuildBackend) -> None:
        backend = build(parse(["--backend", "mock"]))
        assert isinstance(backend, MockBackend)
        assert backend.generate(["anything"])  # canned completions, not an empty list

    def test_a_cli_supplying_no_canned_responses_refuses_mock(self) -> None:
        """A real-execution CLI has no free smoke run, and must say so rather than run fiction."""
        parser = argparse.ArgumentParser()
        backend_cli.add_backend_args(parser)
        with pytest.raises(ValueError, match="canned responses"):
            backend_cli.backend_from_args(parser.parse_args(["--backend", "mock"]), LOCAL_MODEL)


class TestReasoningEffortPreCheckIsBedrockOnly:
    """``--reasoning-effort`` is admitted by both hosted kinds and pre-checked by only one.

    The two ladders are genuinely different objects -- codex accepts ``minimal``, which the Bedrock
    gpt-5.6 dialect does not -- so they cannot be validated against each other. These pin which side
    of the split each path is on, so a later silent normalisation on the codex path goes red instead
    of leaving the flag's help text quietly overclaiming.
    """

    def _hosted_args(self, *argv: str) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        backend_cli.add_backend_args(parser)
        return parser.parse_args(list(argv))

    def test_bedrock_rejects_a_level_outside_the_family_ladder(self) -> None:
        args = self._hosted_args("--backend", "bedrock", "--reasoning-effort", "not-a-level")
        with pytest.raises(ValueError, match="not accepted by this model family"):
            backend_cli.backend_from_args(args, HOSTED_MODEL)

    def test_codex_forwards_an_unknown_level_to_the_cli_unchecked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(model_backend.shutil, "which", lambda _name: "/fake/bin/codex")
        args = self._hosted_args("--backend", "codex", "--reasoning-effort", "not-a-level")
        backend = backend_cli.backend_from_args(args, DEFAULT_CODEX_MODEL)
        assert isinstance(backend, CodexBackend)
        assert backend.reasoning_effort == "not-a-level"


class TestInterpProbeRefusesBackendsWithoutLocalWeights:
    """The activation probe takes the flag so it can explain itself, then accepts only ``hf``."""

    def test_flag_parses(self) -> None:
        assert directions._parse_args(["--backend", "bedrock"]).backend == "bedrock"

    def test_local_weights_are_allowed(self) -> None:
        backend_cli.require_in_process_weights("hf", purpose="capture pooled activations")

    @pytest.mark.parametrize("kind", ["bedrock", "codex", "vllm", "mock"])
    def test_everything_else_is_refused(self, kind: str) -> None:
        with pytest.raises(ValueError, match="residual stream"):
            backend_cli.require_in_process_weights(kind, purpose="capture pooled activations")

    def test_the_entry_point_refuses_before_loading_anything(self) -> None:
        # main() must fail on the flag, not somewhere inside a model download.
        with pytest.raises(ValueError, match="residual stream"):
            directions.main(["--backend", "bedrock"])

    def test_a_choice_only_parser_cannot_build_a_backend(self) -> None:
        # The probe's parser has no decoding knobs, so asking it for a backend says why in words
        # rather than dying on an AttributeError inside a resolver.
        args = directions._parse_args(["--backend", "hf"])
        with pytest.raises(ValueError, match="add_backend_args"):
            backend_cli.backend_from_args(args, LOCAL_MODEL)


class TestSamplingResolution:
    """Unset flags fall back to the calling CLI's own defaults, and 0 is a value, not "unset"."""

    def test_cli_defaults_apply_when_no_flag_is_given(self) -> None:
        args = runner._parse_args([])
        sampling = backend_cli.local_sampling_from_args(args, _sampling_base(temperature=1.0))
        assert sampling.temperature == 1.0
        assert sampling.top_p == _sampling_base().top_p
        assert sampling.do_sample is True

    def test_flags_override_the_cli_defaults(self) -> None:
        args = runner._parse_args(["--temperature", "0.2", "--top-k", "0"])
        sampling = backend_cli.local_sampling_from_args(args, _sampling_base(temperature=1.0))
        assert sampling.temperature == 0.2
        assert sampling.top_k == 0

    def test_episode_runner_keeps_its_base_like_temperature(
        self, recorded: list[_RecordedBuild]
    ) -> None:
        # Lead #6 reads a high-temperature policy, so that default lives in the CLI, not the flag.
        runner._build_cli_backend(runner._parse_args(["--backend", "hf"]))
        assert recorded[0].kwargs["sampling"].temperature == 1.0

    def test_channel_probe_keeps_the_model_card_defaults(
        self, recorded: list[_RecordedBuild]
    ) -> None:
        model_policy._build_cli_backend(model_policy._parse_args(["--backend", "hf"]))
        assert recorded[0].kwargs["sampling"].temperature == _sampling_base().temperature

    def test_local_sampling_carries_every_field_from_the_base(self) -> None:
        # M1: reconstructing the config field-by-field dropped min_p/repetition_penalty/
        # presence_penalty back to SamplingConfig()'s defaults, so the thinking preset's
        # presence_penalty=1.5 (vLLM's anti-loop lever) never reached the backend and the recorded
        # metadata under-reported it. The base's every field must survive when no flag overrides it.
        base = SamplingConfig.for_thinking(thinking=True)
        sampling = backend_cli.local_sampling_from_args(runner._parse_args([]), base)
        assert sampling.presence_penalty == 1.5
        assert sampling.min_p == base.min_p
        assert sampling.repetition_penalty == base.repetition_penalty
        assert sampling.top_p == 0.95
        assert sampling.max_new_tokens == 32768

    def test_the_two_penalty_free_knobs_are_reachable_from_the_cli(self) -> None:
        # Both were unreachable: HFBackend forwards min_p and repetition_penalty to generate, and its
        # docstring names repetition_penalty as the anti-loop fallback for a run that loops inside
        # <think>, but no flag existed to set either -- so the documented fallback could not be used.
        args = runner._parse_args(["--min-p", "0.05", "--repetition-penalty", "1.1"])
        sampling = backend_cli.local_sampling_from_args(args, _sampling_base())
        assert sampling.min_p == 0.05
        assert sampling.repetition_penalty == 1.1

    def test_they_default_to_their_identity_values(self) -> None:
        """Unset means off: a penalty is a behavioural intervention, never a default."""
        sampling = backend_cli.local_sampling_from_args(runner._parse_args([]), _sampling_base())
        assert sampling.min_p == 0.0
        assert sampling.repetition_penalty == 1.0

    @pytest.mark.parametrize(
        ("flag", "value"), [("--min-p", "0.05"), ("--repetition-penalty", "1.1")]
    )
    def test_bedrock_refuses_them_rather_than_dropping_them(self, flag: str, value: str) -> None:
        # Converse's inferenceConfig has neither field, so a value passed there would vanish with no
        # error from the API -- the same reason --top-k is refused on bedrock.
        args = runner._parse_args(["--backend", "bedrock", flag, value])
        with pytest.raises(ValueError, match="does not apply to --backend bedrock"):
            backend_cli.backend_from_args(args, HOSTED_MODEL)

    def test_unset_base_falls_back_to_the_mode_correct_preset(self) -> None:
        # With no base given, the fallback is the thinking-aware preset, not a bare SamplingConfig()
        # whose top_p 0.8 / 1024-token cap would loop and truncate Qwen3.5 under --thinking.
        sampling = backend_cli.local_sampling_from_args(runner._parse_args(["--thinking"]), None)
        assert sampling.top_p == 0.95
        assert sampling.max_new_tokens == 32768
        assert sampling.presence_penalty == 1.5

    def test_episode_runner_thinking_uses_the_thinking_preset(
        self, recorded: list[_RecordedBuild]
    ) -> None:
        # D1: `runner --thinking` must run the thinking preset (top_p 0.95, a 32768-token budget,
        # presence_penalty 1.5), not the non-thinking one that loops Qwen3.5 inside <think>. The
        # base-like temperature 1.0 (lead #6) rides along and equals the thinking preset's own.
        runner._build_cli_backend(runner._parse_args(["--backend", "hf", "--thinking"]))
        sampling = recorded[0].kwargs["sampling"]
        assert sampling.top_p == 0.95
        assert sampling.max_new_tokens == 32768
        assert sampling.presence_penalty == 1.5
        assert sampling.temperature == 1.0
        assert recorded[0].kwargs["thinking"] is True

    def test_channel_probe_thinking_uses_the_thinking_preset(
        self, recorded: list[_RecordedBuild]
    ) -> None:
        # D1 (last sibling): the channel CLI (lead #3) registers --thinking, so it must run the
        # thinking preset (top_p 0.95, a 32768-token budget, presence_penalty 1.5), not the
        # non-thinking one that loops Qwen3.5 inside <think> at top_p 0.8 with a 1024-token cap.
        # Unlike the episode runner it does NOT force a base-like temperature: each mode keeps its
        # model-card default (thinking 1.0, non-thinking 0.7).
        model_policy._build_cli_backend(model_policy._parse_args(["--backend", "hf", "--thinking"]))
        thinking = recorded[0].kwargs["sampling"]
        assert thinking.top_p == 0.95
        assert thinking.max_new_tokens == 32768
        assert thinking.presence_penalty == 1.5
        assert thinking.temperature == 1.0
        assert recorded[0].kwargs["thinking"] is True

        model_policy._build_cli_backend(
            model_policy._parse_args(["--backend", "hf", "--no-thinking"])
        )
        non_thinking = recorded[1].kwargs["sampling"]
        assert non_thinking.top_p == 0.8
        assert non_thinking.temperature == 0.7
        assert recorded[1].kwargs["thinking"] is False


class TestTheOutputFloorIsReadFromTheMeasurementsRatherThanGuessed:
    """Where the floor for one model comes from, and what happens when nothing measured it.

    The floor decides whether a cap is refused, so where its number comes from is the whole
    question. A measured model's row is read from ``recoverybench.budgets``, the module that exists
    to hold measurements, so probing a model raises its floor with no second place to edit.
    """

    def test_a_measured_model_gets_its_own_measured_budget(self) -> None:
        assert backend_cli.output_floor_for(HOSTED_MODEL) == MAX_TOKENS_BY_MODEL[HOSTED_MODEL]

    def test_an_unmeasured_model_falls_back_instead_of_refusing_to_answer(self) -> None:
        """The local plumbing tiers are in no roster and still have to be runnable.

        ``budgets.max_tokens_for`` raises for an unlisted model, which is right when picking the cap
        a paid sweep will run at. It would be wrong here: this is asked about every model these CLIs
        can reach, and refusing would make an unscreened checkpoint unrunnable rather than merely
        unscreened.
        """
        assert LOCAL_MODEL not in MAX_TOKENS_BY_MODEL
        assert backend_cli.output_floor_for(LOCAL_MODEL) == backend_cli.UNSCREENED_OUTPUT_FLOOR

    def test_the_fallback_is_not_below_a_reasoning_models_measured_need(self) -> None:
        """A fallback under the thinking lengths already measured would be a floor in name only."""
        assert backend_cli.UNSCREENED_OUTPUT_FLOOR >= 16384

    def test_a_cap_at_the_floor_is_allowed_and_one_token_under_it_is_not(self) -> None:
        """The boundary itself, since an off-by-one here refuses a model's own measured budget."""
        floor = backend_cli.output_floor_for(HOSTED_MODEL)

        backend_cli.refuse_short_output_cap(HOSTED_MODEL, floor, allow_short=False)

        with pytest.raises(ValueError, match="output tokens"):
            backend_cli.refuse_short_output_cap(HOSTED_MODEL, floor - 1, allow_short=False)

    def test_the_escape_hatch_warns_with_both_numbers_in_the_line(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        floor = backend_cli.output_floor_for(HOSTED_MODEL)

        with caplog.at_level(logging.WARNING, logger=backend_cli.logger.name):
            backend_cli.refuse_short_output_cap(HOSTED_MODEL, 128, allow_short=True)

        assert "128" in caplog.text
        assert str(floor) in caplog.text


class TestVllmQuantizationKnob:
    """``--vllm-quantization`` reaches the engine on vllm and is refused everywhere else.

    The refusal side matters more than the forwarding side: every other backend would ignore the
    flag in silence, and an "fp8" sweep that actually ran bf16 weights is exactly the
    mis-labelled-artifact failure the knob registry exists to prevent.
    """

    def _args(self, *argv: str) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        backend_cli.add_backend_args(parser)
        return parser.parse_args(list(argv))

    def test_fp8_reaches_the_engine_kwargs(self, recorded: list[_RecordedBuild]) -> None:
        args = self._args("--backend", "vllm", "--vllm-quantization", "fp8")
        backend_cli.backend_from_args(args, LOCAL_MODEL, local_sampling=_sampling_base())
        assert recorded[-1].kind == "vllm"
        assert recorded[-1].kwargs["quantization"] == "fp8"

    def test_unset_means_no_engine_kwarg_at_all(self, recorded: list[_RecordedBuild]) -> None:
        """The bf16 default engine construction is byte-identical to what it was before the flag."""
        args = self._args("--backend", "vllm")
        backend_cli.backend_from_args(args, LOCAL_MODEL, local_sampling=_sampling_base())
        assert "quantization" not in recorded[-1].kwargs

    @pytest.mark.parametrize("kind", ["hf", "bedrock", "codex"])
    def test_it_is_refused_on_backends_that_would_ignore_it(self, kind: str) -> None:
        args = self._args("--backend", kind, "--vllm-quantization", "fp8")
        with pytest.raises(ValueError, match="--vllm-quantization"):
            backend_cli.backend_from_args(args, LOCAL_MODEL)

    def test_an_unregistered_mode_is_refused_at_the_parser(self) -> None:
        with pytest.raises(SystemExit):
            self._args("--backend", "vllm", "--vllm-quantization", "int4")


class TestVllmGpuMemoryUtilizationKnob:
    """``--vllm-gpu-memory-utilization`` reaches the engine on vllm and is refused everywhere else.

    The flag exists so a smoke can start an engine on a card other processes already hold: vLLM
    refuses to start when the card's free memory is below its claim, and its default claim is 0.9
    of the whole card. Unset must leave the engine construction exactly as it was, so a rented
    single-tenant run keeps vLLM's own default.
    """

    def _args(self, *argv: str) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        backend_cli.add_backend_args(parser)
        return parser.parse_args(list(argv))

    def test_the_fraction_reaches_the_engine_kwargs(self, recorded: list[_RecordedBuild]) -> None:
        args = self._args("--backend", "vllm", "--vllm-gpu-memory-utilization", "0.45")
        backend_cli.backend_from_args(args, LOCAL_MODEL, local_sampling=_sampling_base())
        assert recorded[-1].kind == "vllm"
        assert recorded[-1].kwargs["gpu_memory_utilization"] == pytest.approx(0.45)

    def test_unset_means_no_engine_kwarg_at_all(self, recorded: list[_RecordedBuild]) -> None:
        args = self._args("--backend", "vllm")
        backend_cli.backend_from_args(args, LOCAL_MODEL, local_sampling=_sampling_base())
        assert "gpu_memory_utilization" not in recorded[-1].kwargs

    @pytest.mark.parametrize("kind", ["hf", "bedrock", "codex"])
    def test_it_is_refused_on_backends_that_make_no_such_claim(self, kind: str) -> None:
        args = self._args("--backend", kind, "--vllm-gpu-memory-utilization", "0.45")
        with pytest.raises(ValueError, match="--vllm-gpu-memory-utilization"):
            backend_cli.backend_from_args(args, LOCAL_MODEL)


class TestVllmMaxModelLenKnob:
    """``--vllm-max-model-len`` reaches the engine on vllm and is refused everywhere else.

    Unset, vLLM sizes its KV cache for the checkpoint's whole window and refuses to start when one
    sequence of that length does not fit -- the legibility probe derives the value from its budgets,
    and the harness, whose transcript grows per turn, takes it explicitly.
    """

    def _args(self, *argv: str) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        backend_cli.add_backend_args(parser)
        return parser.parse_args(list(argv))

    def test_the_length_reaches_the_engine_kwargs(self, recorded: list[_RecordedBuild]) -> None:
        args = self._args("--backend", "vllm", "--vllm-max-model-len", "16384")
        backend_cli.backend_from_args(args, LOCAL_MODEL, local_sampling=_sampling_base())
        assert recorded[-1].kwargs["max_model_len"] == 16384

    def test_unset_means_no_engine_kwarg_at_all(self, recorded: list[_RecordedBuild]) -> None:
        args = self._args("--backend", "vllm")
        backend_cli.backend_from_args(args, LOCAL_MODEL, local_sampling=_sampling_base())
        assert "max_model_len" not in recorded[-1].kwargs

    @pytest.mark.parametrize("kind", ["hf", "bedrock", "codex"])
    def test_it_is_refused_on_backends_with_no_such_cache(self, kind: str) -> None:
        args = self._args("--backend", kind, "--vllm-max-model-len", "16384")
        with pytest.raises(ValueError, match="--vllm-max-model-len"):
            backend_cli.backend_from_args(args, LOCAL_MODEL)


class TestPresencePenaltyKnob:
    """``--presence-penalty`` overrides the base sampler on vllm and is refused everywhere else.

    The zero case is the whole reason the knob exists: the thinking preset carries
    ``presence_penalty=1.5``, every 3e7f227 battery cell sampled under it, and training ran
    penalty-free -- so "force it off" has to be expressible, and 0 has to count as a given value
    rather than as "unset" (the ``_resolved`` contract). The refusal side mirrors
    ``--vllm-quantization``: transformers cannot apply the penalty, so anywhere but vllm the flag
    would change the recorded sampling metadata while changing nothing about generation.
    """

    def _args(self, *argv: str) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        backend_cli.add_backend_args(parser)
        return parser.parse_args(list(argv))

    def test_zero_forces_the_preset_penalty_off(self, recorded: list[_RecordedBuild]) -> None:
        args = self._args("--backend", "vllm", "--thinking", "--presence-penalty", "0")
        backend_cli.backend_from_args(args, LOCAL_MODEL)
        sampling = recorded[-1].kwargs["sampling"]
        assert isinstance(sampling, SamplingConfig)
        assert sampling.presence_penalty == 0.0
        # The override is field-wise: the rest of the thinking preset survives untouched.
        assert sampling.top_p == 0.95
        assert sampling.max_new_tokens == 32768

    def test_unset_keeps_the_base_sampler_value(self, recorded: list[_RecordedBuild]) -> None:
        args = self._args("--backend", "vllm", "--thinking")
        backend_cli.backend_from_args(args, LOCAL_MODEL)
        assert recorded[-1].kwargs["sampling"].presence_penalty == 1.5

    def test_an_explicit_value_overrides_the_base(self, recorded: list[_RecordedBuild]) -> None:
        args = self._args("--backend", "vllm", "--presence-penalty", "0.7")
        backend_cli.backend_from_args(args, LOCAL_MODEL, local_sampling=_sampling_base())
        assert recorded[-1].kwargs["sampling"].presence_penalty == 0.7

    @pytest.mark.parametrize("kind", ["hf", "bedrock", "codex"])
    def test_it_is_refused_on_backends_that_would_ignore_it(self, kind: str) -> None:
        args = self._args("--backend", kind, "--presence-penalty", "0")
        with pytest.raises(ValueError, match="--presence-penalty"):
            backend_cli.backend_from_args(args, LOCAL_MODEL)


class TestTheHarnessServesFullWeights:
    """The agent harness resolves a full checkpoint beside the base id it keys its floors on."""

    def test_the_mock_backend_resolves_the_label_without_touching_the_hub(
        self, monkeypatch: pytest.MonkeyPatch, recorded: list[_RecordedBuild]
    ) -> None:
        from games import eval_model  # noqa: PLC0415 - the hub stub is this test's whole subject

        def no_hub() -> object:
            raise AssertionError("the hub must not be consulted for a mock run")

        monkeypatch.setattr(eval_model, "HfApi", no_hub)
        args = loop._parse_args(
            ["--backend", "mock", "--full-weights", "allenai/tmax-4b", "--revision", "step_300"]
        )
        served = loop.served_model_from_args(args)
        assert served.load_mode == "mock-no-load"
        assert served.model_id == "allenai/tmax-4b@step_300"
        backend = loop._build_cli_backend(args, served)
        assert isinstance(backend, MockBackend)
        assert recorded[-1].model_id == "allenai/tmax-4b@step_300"

    def test_a_revision_without_full_weights_is_refused(self) -> None:
        args = loop._parse_args(["--backend", "mock", "--revision", "step_300"])
        with pytest.raises(ValueError, match="which was not given"):
            loop.served_model_from_args(args)

    def test_without_the_flags_the_base_model_is_served_as_before(self) -> None:
        args = loop._parse_args(["--backend", "mock"])
        served = loop.served_model_from_args(args)
        assert served.load_mode == "base"
        assert served.model_id == loop._DEFAULT_MODEL_BY_BACKEND["mock"]
        assert served.backend_kwargs == {}
