"""Offline tests for generating on vLLM while capturing on HuggingFace.

The optimization these cover: generation is most of an interp stage's wall clock, and vLLM runs it
roughly 5x faster than ``transformers.generate`` on this model family, but activation capture needs
forward hooks on an in-process HuggingFace model and vLLM exposes no hidden states. So generation
moves and capture does not, and what crosses between them is the token ids the engine actually read
and wrote.

That makes the load-bearing claim a bookkeeping one rather than a numerical one, and its failure mode
silent: a record whose ``full_ids`` or ``prompt_len`` are off by a token still captures cleanly, still
pools into a well-shaped vector, and labels the wrong positions as the model's own reasoning. So the
claims here are:

* the vLLM record is built from the engine's OWN ids -- prompt+response concatenated, the boundary at
  the prompt length, the mask true exactly over the response -- never from re-tokenised decoded text,
  which shifts at BPE and special-token boundaries the decode does not round-trip;
* the two engines' partitions of a sampler are each honest about that engine: the HF path records
  ``presence_penalty`` as requested-but-dropped because transformers has no such field, while the
  vLLM path records it as APPLIED, and the record says which engine it came from;
* under greedy, vLLM's own rewriting of the truncation knobs (``SamplingParams.__post_init__`` resets
  top_p/top_k/min_p) shows up as dropped-with-a-reason rather than as three settings that ran;
* a stage that resolved one engine refuses a record generated on the other;
* replies are held against the prompts they answer, so a reordered batch cannot file one twin's
  activations under the other.

No GPU and no vllm import: a stub engine stands in for the parts of ``vllm``'s ``RequestOutput`` /
``CompletionOutput`` this path reads. The prediction of vLLM's own behaviour that
``_vllm_sampling_kwargs`` encodes was read off the installed vllm 0.27.1 source and confirmed against
a live ``SamplingParams``; ``VLLMResponseGenerator`` re-checks it against the engine at startup, so a
release that changes the rule fails there rather than here.

The type-check suppressions below all follow from that substitution: the stubs stand where a real
engine and tokenizer are annotated, they are reached through the generator's private attribute, and
the generator's own ``_backend`` is optional because it can be released.
"""
# pyright: reportArgumentType=false, reportPrivateUsage=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from reward_hacking.interp.generation_capture import (
    GEN_ENGINE_HF,
    GEN_ENGINE_VLLM,
    PENALTY_FREE_THINKING_SAMPLING,
    SAMPLING_FIELD_NAMES,
    ResolvedSampler,
    VLLMResponseGenerator,
    _vllm_sampling_kwargs,
    record_from_token_ids,
    resolved_sampler,
    resolved_sampler_for,
    vllm_resolved_sampler,
)
from reward_hacking.interp.run_harness import require_record_sampler
from reward_hacking.model_backend import (
    STOP_REASON_END_TURN,
    STOP_REASON_MAX_TOKENS,
    VLLM_SAMPLING_PARAM_ATTRS,
    BedrockCompletion,
    HFBackend,
    SamplingConfig,
    TokenizedCompletion,
    TokenUsage,
    VLLMBackend,
)

PROMPT_IDS = (101, 102, 103, 104)
RESPONSE_IDS = (7, 8, 9)

FULLY_SET_SAMPLING = SamplingConfig(
    max_new_tokens=77,
    do_sample=True,
    temperature=0.83,
    top_p=0.91,
    top_k=13,
    min_p=0.07,
    repetition_penalty=1.13,
    presence_penalty=1.5,
)
"""Every knob off its identity value, so a dropped one cannot hide behind a matching default.

``presence_penalty`` is deliberately non-zero: it is the one knob the two engines genuinely differ on,
so it is what distinguishes an honest per-engine record from a copied one.
"""


class _IdTokenizer:
    """Decodes ids to their numbers and names each token, which is all a record needs."""

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(str(int(token)) for token in ids)

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [f"t{token}" for token in ids]


def _record(
    *,
    prompt_token_ids: tuple[int, ...] = PROMPT_IDS,
    response_token_ids: tuple[int, ...] = RESPONSE_IDS,
    max_new_tokens: int = 64,
    sampler: ResolvedSampler | None = None,
):
    return record_from_token_ids(
        _IdTokenizer(),
        "solve this",
        prompt_token_ids=prompt_token_ids,
        response_token_ids=response_token_ids,
        max_new_tokens=max_new_tokens,
        sampler=sampler or vllm_resolved_sampler(PENALTY_FREE_THINKING_SAMPLING),
    )


class TestARecordIsBuiltFromTheEnginesOwnIds:
    """The bookkeeping claim, and every way of getting it wrong is invisible downstream."""

    def test_full_ids_are_the_prompt_ids_then_the_response_ids(self) -> None:
        record = _record()

        assert record.full_ids.tolist() == [*PROMPT_IDS, *RESPONSE_IDS]
        assert record.full_ids.dtype == torch.long

    def test_the_boundary_sits_at_the_prompt_length(self) -> None:
        record = _record()

        assert record.prompt_len == len(PROMPT_IDS)
        assert record.n_generated == len(RESPONSE_IDS)
        assert record.seq_len == len(PROMPT_IDS) + len(RESPONSE_IDS)

    def test_the_mask_is_true_exactly_over_the_response(self) -> None:
        record = _record()

        assert record.is_response.tolist() == [False] * 4 + [True] * 3
        assert record.full_ids[record.is_response].tolist() == list(RESPONSE_IDS)

    def test_the_response_text_decodes_only_the_response_ids(self) -> None:
        """Decoding the whole sequence would fold the prompt into the response."""
        assert _record().response_text == "7 8 9"

    def test_every_position_is_named(self) -> None:
        record = _record()

        assert record.token_strings == ["t101", "t102", "t103", "t104", "t7", "t8", "t9"]
        assert len(record.token_strings) == record.seq_len

    def test_the_cap_is_read_off_the_response_length(self) -> None:
        assert not _record(max_new_tokens=64).hit_token_cap
        assert _record(max_new_tokens=len(RESPONSE_IDS)).hit_token_cap

    def test_an_empty_response_is_a_well_formed_record_with_nothing_generated(self) -> None:
        """The contrast stage drops such a pair by counting it, so it must survive construction."""
        record = _record(response_token_ids=())

        assert record.n_generated == 0
        assert record.prompt_len == record.seq_len


class TestEachEngineRecordsItsOwnSampler:
    """Two engines, two honest partitions -- and the record says which one produced it."""

    def test_the_engine_labels_match_the_backends_own_transport_names(self) -> None:
        """A trace already records ``transport``; the sampler must not invent a second vocabulary."""
        assert HFBackend.transport == GEN_ENGINE_HF
        assert VLLMBackend.transport == GEN_ENGINE_VLLM

    def test_vllm_applies_the_presence_penalty_the_hf_path_cannot(self) -> None:
        hf = resolved_sampler(FULLY_SET_SAMPLING)
        vllm = vllm_resolved_sampler(FULLY_SET_SAMPLING)

        assert "presence_penalty" in hf.dropped
        assert vllm.applied["presence_penalty"] == FULLY_SET_SAMPLING.presence_penalty
        assert "presence_penalty" not in vllm.dropped

    def test_the_vllm_partition_covers_every_config_field(self) -> None:
        sampler = vllm_resolved_sampler(FULLY_SET_SAMPLING)

        assert set(sampler.applied) | set(sampler.dropped) == set(SAMPLING_FIELD_NAMES)
        assert not set(sampler.applied) & set(sampler.dropped)

    def test_under_sampling_every_knob_reaches_the_engine(self) -> None:
        sampler = vllm_resolved_sampler(FULLY_SET_SAMPLING)

        assert sampler.dropped == {}
        assert sampler.applied == {
            name: getattr(FULLY_SET_SAMPLING, name) for name in SAMPLING_FIELD_NAMES
        }

    def test_under_greedy_the_truncation_knobs_are_dropped_with_a_reason(self) -> None:
        """vLLM resets top_p/top_k/min_p at temperature 0, so claiming them would be false."""
        sampler = vllm_resolved_sampler(replace(FULLY_SET_SAMPLING, do_sample=False))

        assert set(sampler.dropped) == {"temperature", "top_p", "top_k", "min_p"}
        assert sampler.dropped["top_p"].requested == FULLY_SET_SAMPLING.top_p
        assert "1.0" in sampler.dropped["top_p"].why
        # Penalties still apply under greedy here: they reshape the logits the argmax reads.
        assert sampler.applied["presence_penalty"] == FULLY_SET_SAMPLING.presence_penalty
        assert sampler.applied["repetition_penalty"] == FULLY_SET_SAMPLING.repetition_penalty

    def test_the_payload_names_the_engine(self) -> None:
        """ "presence_penalty applied" means opposite things per engine, so the label is required."""
        assert vllm_resolved_sampler(FULLY_SET_SAMPLING).as_payload()["engine"] == GEN_ENGINE_VLLM
        assert resolved_sampler(FULLY_SET_SAMPLING).as_payload()["engine"] == GEN_ENGINE_HF

    def test_resolving_for_an_engine_picks_that_engines_partition(self) -> None:
        for engine, expected in (
            (GEN_ENGINE_HF, resolved_sampler(FULLY_SET_SAMPLING)),
            (GEN_ENGINE_VLLM, vllm_resolved_sampler(FULLY_SET_SAMPLING)),
        ):
            assert resolved_sampler_for(FULLY_SET_SAMPLING, engine=engine) == expected

    def test_an_unknown_engine_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown generation engine"):
            resolved_sampler_for(FULLY_SET_SAMPLING, engine="tensorrt")

    def test_the_predicted_kwargs_cover_every_field_vllm_can_hold(self) -> None:
        """A field added upstream must land in the prediction, or it goes unrecorded."""
        predicted = _vllm_sampling_kwargs(FULLY_SET_SAMPLING)

        assert set(predicted) == set(SAMPLING_FIELD_NAMES)
        assert set(predicted) == {*VLLM_SAMPLING_PARAM_ATTRS, "do_sample"}


class TestAStageRefusesARecordFromTheOtherEngine:
    """The threading guard, extended: an engine mismatch is as wrong as a knob mismatch."""

    def test_a_vllm_record_fails_a_stage_that_resolved_hf(self) -> None:
        record = _record(sampler=vllm_resolved_sampler(PENALTY_FREE_THINKING_SAMPLING))

        with pytest.raises(RuntimeError, match="did not resolve"):
            require_record_sampler(
                record, resolved_sampler(PENALTY_FREE_THINKING_SAMPLING), stage="contrast"
            )

    def test_a_vllm_record_passes_a_stage_that_resolved_vllm(self) -> None:
        record = _record(sampler=vllm_resolved_sampler(PENALTY_FREE_THINKING_SAMPLING))

        require_record_sampler(
            record,
            resolved_sampler_for(PENALTY_FREE_THINKING_SAMPLING, engine=GEN_ENGINE_VLLM),
            stage="contrast",
        )


class _StubSamplingParams:
    """The ``SamplingParams`` surface ``applied_sampling`` reads, with vLLM's greedy reset in it.

    Replicates ``__post_init__``'s rule as read off the installed vllm 0.27.1 (lines 300-305: at a
    temperature under the greedy epsilon it sets top_p to 1.0, top_k to 0 and min_p to 0.0). This stub
    is not the check on that rule -- ``VLLMResponseGenerator`` compares the prediction against the
    live engine at startup, which is where a vLLM release that changed it would surface.
    """

    def __init__(self, sampling: SamplingConfig) -> None:
        self.max_tokens = sampling.max_new_tokens
        self.temperature = sampling.temperature if sampling.do_sample else 0.0
        greedy = self.temperature == 0.0
        self.top_p = 1.0 if greedy else sampling.top_p
        self.top_k = 0 if greedy else sampling.top_k
        self.min_p = 0.0 if greedy else sampling.min_p
        self.repetition_penalty = sampling.repetition_penalty
        self.presence_penalty = sampling.presence_penalty
        # The stub must carry every attribute VLLM_SAMPLING_PARAM_ATTRS names, or applied_sampling
        # raises AttributeError rather than reporting a mismatch -- which is what a knob added to the
        # map without a slot here does.
        self.seed = sampling.seed
        # A LIST, matching what the installed vLLM hands back: its __post_init__ normalises stop to a
        # list and never returns None, which is the behaviour applied_sampling coerces against.
        self.stop = list(sampling.stop)
        self.sampling_type = "greedy" if greedy else "random"


class _StubEngineBackend:
    """A stand-in for ``VLLMBackend`` that returns scripted ids without an engine."""

    def __init__(self, sampling: SamplingConfig, replies: list[TokenizedCompletion]) -> None:
        self._sampling_params = _StubSamplingParams(sampling)
        self._greedy_sampling_type = "greedy"
        self.tokenizer = _IdTokenizer()
        self._replies = replies
        self.requested: list[list[str]] = []

    applied_sampling = VLLMBackend.applied_sampling

    def generate_tokenized(self, prompts: list[str]) -> list[TokenizedCompletion]:
        self.requested.append(list(prompts))
        return self._replies


def _reply(
    prompt_token_ids: tuple[int, ...], response_token_ids: tuple[int, ...], stop_reason: str
) -> TokenizedCompletion:
    return TokenizedCompletion(
        completion=BedrockCompletion(
            text="ignored: the record decodes the ids itself",
            reasoning="",
            usage=TokenUsage(
                input_tokens=len(prompt_token_ids), output_tokens=len(response_token_ids)
            ),
            stop_reason=stop_reason,
        ),
        prompt_token_ids=prompt_token_ids,
        response_token_ids=response_token_ids,
    )


def _generator(sampling: SamplingConfig, replies: list[TokenizedCompletion]):
    """A ``VLLMResponseGenerator`` around the stub backend, skipping the real engine startup.

    ``__init__`` is bypassed rather than mocked because the only thing it does that these tests are
    not about is bring up a GPU engine; everything after it (the prediction check, the record
    building) runs exactly as it does in production.
    """
    generator = VLLMResponseGenerator.__new__(VLLMResponseGenerator)
    generator._backend = _StubEngineBackend(sampling, replies)
    generator._sampling = sampling
    generator.sampler = vllm_resolved_sampler(sampling)
    return generator


class TestTheGeneratorReadsBackWhatTheEngineWillApply:
    """The prediction an artifact is written from, held against the engine's own report."""

    def test_a_matching_engine_passes_the_startup_check(self) -> None:
        generator = _generator(FULLY_SET_SAMPLING, [])

        generator._require_engine_matches_prediction()

    def test_a_greedy_engine_passes_too_because_the_reset_is_predicted(self) -> None:
        greedy = replace(FULLY_SET_SAMPLING, do_sample=False)

        _generator(greedy, [])._require_engine_matches_prediction()

    def test_an_engine_built_from_a_different_config_is_refused(self) -> None:
        """Exactly what forgetting ``sampling=`` at the VLLMBackend call produces."""
        generator = _generator(FULLY_SET_SAMPLING, [])
        generator._sampling = replace(FULLY_SET_SAMPLING, top_k=99)

        with pytest.raises(RuntimeError, match="will not sample the way"):
            generator._require_engine_matches_prediction()

    def test_the_readback_walks_the_config_fields_not_the_engine_names(self) -> None:
        applied = _generator(FULLY_SET_SAMPLING, [])._backend.applied_sampling()

        assert set(applied) == set(SAMPLING_FIELD_NAMES)
        assert applied["max_new_tokens"] == FULLY_SET_SAMPLING.max_new_tokens
        assert applied["do_sample"] is True


class TestTheGeneratorBuildsRecordsFromTheReplies:
    """One record per prompt, in prompt order, off the ids each reply carries."""

    def test_each_prompt_gets_a_record_over_its_own_ids(self) -> None:
        sampling = replace(PENALTY_FREE_THINKING_SAMPLING, max_new_tokens=64)
        generator = _generator(
            sampling,
            [
                _reply((1, 2), (3, 4, 5), STOP_REASON_END_TURN),
                _reply((11, 12, 13), (14,), STOP_REASON_END_TURN),
            ],
        )

        records = generator.generate_records(["first", "second"])

        assert [record.prompt_text for record in records] == ["first", "second"]
        assert records[0].full_ids.tolist() == [1, 2, 3, 4, 5]
        assert records[0].prompt_len == 2
        assert records[1].full_ids.tolist() == [11, 12, 13, 14]
        assert records[1].prompt_len == 3

    def test_every_record_carries_the_generators_sampler(self) -> None:
        sampling = replace(PENALTY_FREE_THINKING_SAMPLING, max_new_tokens=64)
        generator = _generator(sampling, [_reply((1,), (2,), STOP_REASON_END_TURN)])

        (record,) = generator.generate_records(["only"])

        assert record.sampler == vllm_resolved_sampler(sampling)
        assert record.sampler.engine == GEN_ENGINE_VLLM

    def test_a_capped_reply_is_recorded_as_capped(self) -> None:
        sampling = replace(PENALTY_FREE_THINKING_SAMPLING, max_new_tokens=3)
        generator = _generator(sampling, [_reply((1,), (2, 3, 4), STOP_REASON_MAX_TOKENS)])

        (record,) = generator.generate_records(["only"])

        assert record.hit_token_cap

    def test_a_reply_count_that_does_not_match_the_prompts_is_refused(self) -> None:
        """A dropped reply would silently shift every subsequent record onto the wrong prompt."""
        sampling = replace(PENALTY_FREE_THINKING_SAMPLING, max_new_tokens=64)
        generator = _generator(sampling, [_reply((1,), (2,), STOP_REASON_END_TURN)])

        with pytest.raises(ValueError, match="argument"):
            generator.generate_records(["first", "second"])

    def test_the_whole_batch_goes_to_the_engine_in_one_call(self) -> None:
        """The batching is the optimization; one call per prompt would forfeit it."""
        sampling = replace(PENALTY_FREE_THINKING_SAMPLING, max_new_tokens=64)
        generator = _generator(
            sampling,
            [
                _reply((1,), (2,), STOP_REASON_END_TURN),
                _reply((3,), (4,), STOP_REASON_END_TURN),
            ],
        )

        generator.generate_records(["first", "second"])

        assert generator._backend.requested == [["first", "second"]]
