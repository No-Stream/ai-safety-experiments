"""The guard on the interp path's sampler: every knob forwarded, and nothing claimed that was not.

Two defects motivate this file, and both were silent. The generation path copied FOUR loose scalars
out of a :class:`~reward_hacking.model_backend.SamplingConfig` (max_new_tokens, temperature, top_p,
top_k), so ``min_p`` and ``repetition_penalty`` could not reach ``generate`` however a caller set
them, and ``presence_penalty`` -- which transformers cannot apply at all -- was recorded in a run's
metadata as if it had. A run therefore decoded under one sampler while its artifact asserted another,
and nothing about the output looked wrong.

So the assertions here are about the seam rather than about behaviour:

* what ``_generation_kwargs`` hands ``model.generate`` contains every field transformers honours,
  ``min_p`` and ``repetition_penalty`` included;
* ``presence_penalty`` is never in it, and is recorded as requested-but-dropped with a reason;
* applied and dropped together account for EVERY ``SamplingConfig`` field, so a field added upstream
  cannot land in neither and go unrecorded;
* no artifact payload asserts a value for a knob that did not reach the call;
* the run-harness stages actually forward their resolved sampler, checked against the record that
  comes back -- the one thing a green run cannot otherwise prove.

Sabotage-verified: reverting the ``min_p`` line in ``_sampling_kwargs``, the ``presence_penalty``
omission, or the ``sampling=`` argument at a harness generation site each turns a test here red.
"""
# The fakes stand in for a real HF model/tokenizer at every call site, so scope off that one rule.
# pyright: reportArgumentType=false

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch

from reward_hacking.interp.generation_capture import (
    DEFAULT_VLLM_GPU_FRACTION,
    GEN_ENGINE_HF,
    PENALTY_FREE_THINKING_SAMPLING,
    SAMPLING_FIELD_NAMES,
    WHY_TRANSFORMERS_CANNOT_APPLY,
    HFResponseGenerator,
    ResolvedSampler,
    _generation_kwargs,  # pyright: ignore[reportPrivateUsage]  # the seam under test
    _sampling_kwargs,  # pyright: ignore[reportPrivateUsage]  # the seam under test
    generate_response,
    resolved_sampler,
)
from reward_hacking.interp.prompt_contrast import StimulusPair
from reward_hacking.interp.run_harness import (
    VARIANT_ALL_RESPONSE,
    ContrastArgs,
    _capture_twin_pooled,  # pyright: ignore[reportPrivateUsage]  # the stage driver under test
    _parse_args,  # pyright: ignore[reportPrivateUsage]  # arg -> config mapping without a model load
    require_record_sampler,
    sampling_from_args,
)
from reward_hacking.model_backend import SamplingConfig

if TYPE_CHECKING:
    import argparse

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

``presence_penalty`` is deliberately non-zero: the HF path cannot apply it, and what is under test is
that the record says so rather than recording 1.5 as a setting that ran.
"""


class _EchoTokenizer:
    """Just enough tokenizer surface for one generation, with a padding id to forward."""

    pad_token_id = 41

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        del kwargs
        return messages[0]["content"]

    def __call__(self, text: str, **kwargs: object) -> dict[str, torch.Tensor]:
        del kwargs
        ids = torch.tensor([[(ord(char) % 40) + 1 for char in text] or [1]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, ids: torch.Tensor, **kwargs: object) -> str:
        del kwargs
        return " ".join(str(int(i)) for i in ids)

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [f"t{i}" for i in ids]


class _KwargRecordingLM(torch.nn.Module):
    """Records the kwargs ``generate`` was called with, then appends two tokens."""

    def __init__(self) -> None:
        super().__init__()
        self.recorded: dict[str, object] = {}

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def generate(self, input_ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
        self.recorded = dict(kwargs)
        return torch.cat([input_ids, torch.tensor([[5, 6]], dtype=torch.long)], dim=1)


class TestEveryApplicableKnobReachesGenerate:
    """The forwarding claim: what the config says is what ``generate`` is handed."""

    def test_min_p_and_repetition_penalty_are_forwarded(self) -> None:
        kwargs = _generation_kwargs(_EchoTokenizer(), FULLY_SET_SAMPLING)

        assert kwargs["min_p"] == FULLY_SET_SAMPLING.min_p
        assert kwargs["repetition_penalty"] == FULLY_SET_SAMPLING.repetition_penalty

    def test_every_transformers_knob_is_forwarded_with_the_configured_value(self) -> None:
        kwargs = _sampling_kwargs(FULLY_SET_SAMPLING)
        expected = {
            name: getattr(FULLY_SET_SAMPLING, name)
            for name in SAMPLING_FIELD_NAMES
            if name not in WHY_TRANSFORMERS_CANNOT_APPLY
        }

        assert kwargs == expected

    def test_a_stop_sequence_is_translated_into_the_name_generate_knows(self) -> None:
        """``stop`` is a field name; ``stop_strings`` is the ``generate`` argument. Both must appear.

        The two dicts speak different vocabularies on purpose: ``_sampling_kwargs`` is keyed by
        SamplingConfig FIELD names, because that is what the applied/dropped partition compares
        against, while ``_generation_kwargs`` must be keyed by real ``generate`` arguments. ``stop`` is
        the first field where those diverge, and passing it through untranslated would make every
        generation call raise on an unexpected keyword -- or, worse on a permissive version, silently
        decode with no stop sequence at all. transformers' StopStringCriteria also needs the tokenizer
        or generate refuses to start, which is why it rides along.

        Found by sabotage: replacing the translation with a straight pass-through left every existing
        test green, because none of them built generation kwargs from a config that set a stop.
        """
        stopping = replace(FULLY_SET_SAMPLING, stop=("</run>",))
        tokenizer = _EchoTokenizer()

        kwargs = _generation_kwargs(tokenizer, stopping)

        assert kwargs["stop_strings"] == ["</run>"]
        assert kwargs["tokenizer"] is tokenizer
        assert "stop" not in kwargs, "the field name is not a generate argument"

    def test_no_stop_sequence_adds_neither_the_criteria_nor_the_tokenizer(self) -> None:
        """An empty stop must not drag a StopStringCriteria (and its tokenizer) into every call."""
        kwargs = _generation_kwargs(_EchoTokenizer(), replace(FULLY_SET_SAMPLING, stop=()))

        assert "stop_strings" not in kwargs
        assert "tokenizer" not in kwargs
        assert "stop" not in kwargs

    def test_presence_penalty_is_never_forwarded(self) -> None:
        """transformers has no such parameter, so passing it would be a request it cannot honour."""
        assert "presence_penalty" not in _generation_kwargs(_EchoTokenizer(), FULLY_SET_SAMPLING)

    def test_the_kwargs_reach_the_real_generate_call(self) -> None:
        model = _KwargRecordingLM()

        generate_response(model, _EchoTokenizer(), "solve this", sampling=FULLY_SET_SAMPLING)

        assert model.recorded["min_p"] == FULLY_SET_SAMPLING.min_p
        assert model.recorded["repetition_penalty"] == FULLY_SET_SAMPLING.repetition_penalty
        assert model.recorded["temperature"] == FULLY_SET_SAMPLING.temperature
        assert model.recorded["max_new_tokens"] == FULLY_SET_SAMPLING.max_new_tokens
        assert "presence_penalty" not in model.recorded


class TestNothingIsRecordedThatDidNotApply:
    """The honesty claim: a knob the path dropped is marked dropped, never recorded as applied."""

    def test_applied_and_dropped_partition_every_config_field(self) -> None:
        """A field in neither would be a knob nobody could tell was requested."""
        sampler = resolved_sampler(FULLY_SET_SAMPLING)

        assert set(sampler.applied) | set(sampler.dropped) == set(SAMPLING_FIELD_NAMES)
        assert not set(sampler.applied) & set(sampler.dropped)
        assert set(SAMPLING_FIELD_NAMES) == {field.name for field in fields(SamplingConfig)}

    def test_presence_penalty_is_dropped_with_its_requested_value_and_a_reason(self) -> None:
        sampler = resolved_sampler(FULLY_SET_SAMPLING)

        assert "presence_penalty" not in sampler.applied
        dropped = sampler.dropped["presence_penalty"]
        assert dropped.requested == FULLY_SET_SAMPLING.presence_penalty
        assert "presence_penalty" in dropped.why

    def test_the_payload_marks_a_dropped_knob_unapplied_rather_than_asserting_it(self) -> None:
        payload = resolved_sampler(FULLY_SET_SAMPLING).as_payload()
        applied = payload["applied"]
        assert isinstance(applied, dict)

        assert payload["presence_penalty_requested"] == FULLY_SET_SAMPLING.presence_penalty
        assert payload["presence_penalty_applied"] is None
        assert "presence_penalty" not in applied
        # The whole point: nowhere in the payload does the dropped knob read as a value that ran.
        assert FULLY_SET_SAMPLING.presence_penalty not in applied.values()

    def test_a_generated_record_carries_the_sampler_that_ran(self) -> None:
        record = generate_response(
            _KwargRecordingLM(), _EchoTokenizer(), "solve this", sampling=FULLY_SET_SAMPLING
        )

        assert record.sampler == resolved_sampler(FULLY_SET_SAMPLING)
        assert record.sampler.applied["min_p"] == FULLY_SET_SAMPLING.min_p
        assert record.sampler.dropped["presence_penalty"].requested == 1.5

    def test_greedy_decoding_records_the_knobs_it_skipped_as_dropped(self) -> None:
        """Under greedy the truncation knobs never reach generate, so claiming them would be false."""
        sampler = resolved_sampler(replace(FULLY_SET_SAMPLING, do_sample=False))

        assert sampler.applied == {"max_new_tokens": 77, "do_sample": False, "stop": ()}
        assert sampler.dropped["temperature"].requested == FULLY_SET_SAMPLING.temperature
        assert set(sampler.dropped) == {
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "repetition_penalty",
            "presence_penalty",
            "seed",
        }

    def test_the_default_interp_sampler_asks_for_no_penalty(self) -> None:
        """Penalties stay off by default: they are behavioural interventions, not decoding hygiene."""
        thinking = SamplingConfig.for_thinking(thinking=True)

        assert PENALTY_FREE_THINKING_SAMPLING.min_p == 0.0
        assert PENALTY_FREE_THINKING_SAMPLING.repetition_penalty == 1.0
        assert PENALTY_FREE_THINKING_SAMPLING.presence_penalty == 0.0
        # Everything else IS the shared thinking preset, read rather than restated.
        assert PENALTY_FREE_THINKING_SAMPLING.max_new_tokens == thinking.max_new_tokens
        assert PENALTY_FREE_THINKING_SAMPLING.temperature == thinking.temperature
        assert PENALTY_FREE_THINKING_SAMPLING.top_p == thinking.top_p
        assert PENALTY_FREE_THINKING_SAMPLING.top_k == thinking.top_k
        assert PENALTY_FREE_THINKING_SAMPLING.do_sample


def _contrast_args(*, max_new_tokens: int) -> ContrastArgs:
    """A ContrastArgs whose only load-bearing field here is the sampler it resolves."""
    return ContrastArgs(
        model_id="fake",
        episode_dir=Path("episode"),
        out_dir=Path("out"),
        raw_dir=Path("raw"),
        validated_eval_dirs={},
        poolings=("mean",),
        variants=(VARIANT_ALL_RESPONSE,),
        window_length=2,
        limit=None,
        n_placebos=1,
        seed=0,
        sampling=replace(PENALTY_FREE_THINKING_SAMPLING, max_new_tokens=max_new_tokens),
        deadline_seconds=None,
        gen_engine=GEN_ENGINE_HF,
        vllm_gpu_fraction=DEFAULT_VLLM_GPU_FRACTION,
        gen_batch_pairs=4,
    )


class TestHarnessStagesForwardTheirSampler:
    """The threading claim, which is the one a green run cannot otherwise establish.

    A stage that forgets ``sampling=`` at a generation site still generates, still returns records,
    and still writes a metrics artifact naming the sampler it resolved -- under a different sampler.
    ``require_record_sampler`` is what turns that into a crash, so it is checked here both ways.
    """

    def test_the_contrast_stage_generates_under_the_sampler_it_resolved(self) -> None:
        args = _contrast_args(max_new_tokens=6)
        model, tokenizer = _CountingTrunkLM(), _EchoTokenizer()

        _pooled, coverage, log = _capture_twin_pooled(
            HFResponseGenerator(model, tokenizer, sampling=args.sampling),
            model,
            [StimulusPair("p0", "conflicting transcript", "original transcript")],
            args,
        )

        assert coverage.pairs_completed == 1
        payload = resolved_sampler(args.sampling).as_payload()
        assert [row["sampler"] for row in log] == [payload, payload]

    def test_a_generator_built_from_another_sampler_is_refused_by_the_stage(self) -> None:
        """The generator is now where the forwarding happens, so that is where it can go wrong."""
        args = _contrast_args(max_new_tokens=6)
        model, tokenizer = _CountingTrunkLM(), _EchoTokenizer()

        with pytest.raises(RuntimeError, match="did not resolve"):
            _capture_twin_pooled(
                HFResponseGenerator(model, tokenizer, sampling=PENALTY_FREE_THINKING_SAMPLING),
                model,
                [StimulusPair("p0", "conflicting transcript", "original transcript")],
                args,
            )

    def test_a_stage_that_drops_the_sampler_is_refused(self) -> None:
        """Exactly what a forgotten ``sampling=`` produces: a record under the module default."""
        stage = resolved_sampler(replace(PENALTY_FREE_THINKING_SAMPLING, max_new_tokens=6))
        record = generate_response(
            _KwargRecordingLM(), _EchoTokenizer(), "solve", sampling=PENALTY_FREE_THINKING_SAMPLING
        )

        with pytest.raises(RuntimeError, match="did not resolve"):
            require_record_sampler(record, stage, stage="contrast")

    def test_a_record_under_the_stage_sampler_passes(self) -> None:
        sampling = replace(PENALTY_FREE_THINKING_SAMPLING, max_new_tokens=6)
        record = generate_response(
            _KwargRecordingLM(), _EchoTokenizer(), "solve", sampling=sampling
        )

        require_record_sampler(record, resolved_sampler(sampling), stage="contrast")


class _CountingTrunkLM(_KwargRecordingLM):
    """A fake with the trunk shape ``capture_positionwise_activations`` walks, plus ``generate``."""

    def __init__(self, n_layers: int = 2, hidden: int = 4, vocab: int = 64) -> None:
        super().__init__()
        self.model = _FakeTrunk(n_layers, hidden, vocab)


class _FakeTrunk(torch.nn.Module):
    """Embeds ids and runs constant-offset decoder layers, yielding ``last_hidden_state``."""

    def __init__(self, n_layers: int, hidden: int, vocab: int) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab, hidden)
        self.layers = torch.nn.ModuleList([_OffsetLayer(float(i + 1)) for i in range(n_layers)])

    def forward(self, input_ids: torch.Tensor, **kwargs: object) -> SimpleNamespace:
        del kwargs
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


class _OffsetLayer(torch.nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = offset

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.offset


def test_resolved_sampler_is_comparable() -> None:
    """Equality is what ``require_record_sampler`` rests on, so it has to hold field-wise."""
    assert resolved_sampler(FULLY_SET_SAMPLING) == resolved_sampler(FULLY_SET_SAMPLING)
    assert resolved_sampler(FULLY_SET_SAMPLING) != resolved_sampler(
        replace(FULLY_SET_SAMPLING, top_k=99)
    )
    assert isinstance(resolved_sampler(FULLY_SET_SAMPLING), ResolvedSampler)


class TestTheInterpCliReachesEveryKnob:
    """A flag registered but never read is the same defect one level up, so the wiring is asserted.

    ``--out-dir`` and the ``contrast`` subcommand are only there to make the parser happy; the claim
    is that each decoding flag lands on the config the stage generates under.
    """

    @staticmethod
    def _args(*sampling_argv: str) -> argparse.Namespace:
        return _parse_args(
            [
                "--out-dir",
                "out",
                *sampling_argv,
                "contrast",
                "--validated-eval-dir",
                "axis-dir",
            ]
        )

    def test_each_flag_lands_on_the_stage_sampler(self) -> None:
        args = self._args(
            "--temperature",
            "0.5",
            "--top-p",
            "0.4",
            "--top-k",
            "7",
            "--min-p",
            "0.02",
            "--repetition-penalty",
            "1.07",
        )

        sampling = sampling_from_args(args, max_new_tokens=123)

        assert sampling.temperature == 0.5
        assert sampling.top_p == 0.4
        assert sampling.top_k == 7
        assert sampling.min_p == 0.02
        assert sampling.repetition_penalty == 1.07
        assert sampling.max_new_tokens == 123

    def test_the_unflagged_sampler_is_the_penalty_free_preset_at_the_stage_cap(self) -> None:
        sampling = sampling_from_args(self._args(), max_new_tokens=456)

        assert sampling == replace(PENALTY_FREE_THINKING_SAMPLING, max_new_tokens=456)
        assert sampling.presence_penalty == 0.0

    def test_the_stage_cap_is_the_only_thing_a_subcommand_decides(self) -> None:
        """Two stages, one sampler: only the token budget legitimately differs between them."""
        args = self._args("--temperature", "0.5")

        contrast = sampling_from_args(args, max_new_tokens=100)
        steer = sampling_from_args(args, max_new_tokens=200)

        assert replace(contrast, max_new_tokens=200) == steer
