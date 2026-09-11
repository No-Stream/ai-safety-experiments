"""Offline tests for generation-phase activation capture.

CPU only, no model download, no GPU: a fake tokenizer and a fake causal LM (trunk + LM head, plus a
deterministic ``generate``) stand in for Qwen3.5-4B, so the per-position bookkeeping, the position
selectors, and the pooling-over-selected-positions all have ground-truth answers to check against.

The load-bearing claims:

* generation records the prompt/response boundary and labels every position;
* the capture runs the trunk only and never the LM head (the memory fix), the same guard the pooled
  capture carries -- watched to go red if capture were routed through the full causal LM;
* pooling over the selected positions equals a hand-computed mean / last over exactly those rows;
* an empty selection raises rather than silently pooling a divide-by-zero.
"""
# Every call here passes a duck-typed fake model/tokenizer where the capture functions are annotated
# for real HuggingFace types, so reportArgumentType fires on each call site; the fakes are the point
# of an offline test. Scope the suppression to that one rule rather than scatter inline ignores.
# pyright: reportArgumentType=false

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from reward_hacking.interp.generation_capture import (
    PENALTY_FREE_THINKING_SAMPLING,
    GenerationRecord,
    capture_generation_activations,
    capture_record_activations,
    capture_record_pooled,
    capture_response_pooled,
    generate_response,
    matched_window_positions,
    pool_positions,
    response_positions,
)
from reward_hacking.model_backend import (
    SamplingConfig,
    _as_single_user_turn,  # pyright: ignore[reportPrivateUsage]  # the harness's own render
)

APPENDED_TOKENS = [7, 8, 9, 11]  # what the fake model "generates" after any prompt
THINKING_PRESET = SamplingConfig.for_thinking(thinking=True)

GREEDY = replace(PENALTY_FREE_THINKING_SAMPLING, do_sample=False)
"""The module default made reproducible, for fakes that ignore the sampler anyway.

Every call here passes a config rather than a ``do_sample=False`` flag, because one config IS the
interface now: the four loose scalars that used to sit beside it are what let three knobs go
unreachable while the recorded metadata still claimed them.
"""


def greedy_capped(max_new_tokens: int) -> SamplingConfig:
    """The greedy config at a chosen token cap, for the truncation cases."""
    return replace(GREEDY, max_new_tokens=max_new_tokens)


class _RecordingLMHead(torch.nn.Module):
    """An LM head that records whether it was ever called -- capture must never reach it."""

    def __init__(self, hidden: int, vocab: int) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(hidden, vocab)
        self.called = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.called = True
        return self.projection(hidden_states)


class _AddOffsetLayer(torch.nn.Module):
    """A decoder block that adds a per-layer constant so each layer's residual is
    distinguishable."""

    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = offset

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.offset


class _FakeTextTrunk(torch.nn.Module):
    """Embeds ids and runs the decoder layers, yielding last_hidden_state."""

    def __init__(self, n_layers: int, hidden: int, vocab: int) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab, hidden)
        self.layers = torch.nn.ModuleList([_AddOffsetLayer(float(i + 1)) for i in range(n_layers)])

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs: object
    ) -> SimpleNamespace:
        del attention_mask, kwargs
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


class _FakeCausalLM(torch.nn.Module):
    """Trunk + LM head with a deterministic ``generate`` that appends APPENDED_TOKENS."""

    def __init__(self, n_layers: int = 3, hidden: int = 8, vocab: int = 64) -> None:
        super().__init__()
        self.model = _FakeTextTrunk(n_layers, hidden, vocab)
        self.lm_head = _RecordingLMHead(hidden, vocab)

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def generate(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs: object
    ) -> torch.Tensor:
        del attention_mask, kwargs
        appended = torch.tensor([APPENDED_TOKENS], dtype=torch.long)
        return torch.cat([input_ids, appended], dim=1)


class _CapRecordingLM(_FakeCausalLM):
    """Records the ``max_new_tokens`` cap ``generate`` was handed, then generates as usual."""

    def __init__(self, n_layers: int = 1, hidden: int = 4, vocab: int = 64) -> None:
        super().__init__(n_layers, hidden, vocab)
        self.recorded_max_new_tokens: int | None = None

    def generate(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs: object
    ) -> torch.Tensor:
        self.recorded_max_new_tokens = int(kwargs["max_new_tokens"])  # pyright: ignore[reportArgumentType]
        return super().generate(input_ids, attention_mask, **kwargs)


class _CapFillingLM(_FakeCausalLM):
    """Generates exactly the cap it was given: a trace truncated mid-``<think>``."""

    def generate(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs: object
    ) -> torch.Tensor:
        del attention_mask
        n_new = int(kwargs["max_new_tokens"])  # pyright: ignore[reportArgumentType]
        appended = torch.arange(1, n_new + 1, dtype=torch.long).unsqueeze(0)
        return torch.cat([input_ids, appended], dim=1)


class _FakeEncoding(dict[str, torch.Tensor]):
    """A tokenizer output that stays put under ``.to(device)`` -- the fakes live on CPU."""

    def to(self, device: object) -> _FakeEncoding:
        del device
        return self


class _FakeTokenizer:
    """Char-based tokenizer with just enough surface for the generation-capture path."""

    pad_token: str | None = "<pad>"
    pad_token_id = 0
    eos_token = "<eos>"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
    ) -> str:
        del tokenize, add_generation_prompt, enable_thinking
        return messages[0]["content"]

    def __call__(self, text: str, *, return_tensors: str = "pt") -> _FakeEncoding:
        del return_tensors
        ids = [(ord(char) % 40) + 1 for char in text] or [1]
        input_ids = torch.tensor([ids], dtype=torch.long)
        return _FakeEncoding(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))

    def decode(self, ids: torch.Tensor, *, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(str(int(i)) for i in ids)

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [f"t{i}" for i in ids]


class _ArgumentEchoTokenizer(_FakeTokenizer):
    """Renders every chat-template argument into the text, and records what it was asked to encode.

    Any divergence in the four arguments -- the message list, ``tokenize``,
    ``add_generation_prompt``, ``enable_thinking`` -- then shows up as different text, which is what
    lets one call site be compared byte for byte against the harness's own.
    """

    def __init__(self) -> None:
        self.encoded_texts: list[str] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
    ) -> str:
        return f"{messages!r}|{tokenize=}|{add_generation_prompt=}|{enable_thinking=}"

    def __call__(self, text: str, *, return_tensors: str = "pt") -> _FakeEncoding:
        self.encoded_texts.append(text)
        return super().__call__(text, return_tensors=return_tensors)


class TestGenerationPromptMatchesTheHarness:
    """The capture must run in the same prompt space the policy runs in, by construction.

    The residual stream is only interpretable relative to the text the model actually saw, so a
    chat-template argument that drifts from ``model_backend``'s own render silently changes the
    whole measured space. Asserting equality against that render, rather than re-stating the four
    arguments here, is what makes the drift detectable.
    """

    def test_generate_response_encodes_the_harness_render(self) -> None:
        tokenizer = _ArgumentEchoTokenizer()

        generate_response(_FakeCausalLM(), tokenizer, "solve this", thinking=True, sampling=GREEDY)

        assert tokenizer.encoded_texts == [
            _as_single_user_turn(tokenizer, "solve this", thinking=True)
        ]

    def test_non_thinking_mode_also_matches(self) -> None:
        tokenizer = _ArgumentEchoTokenizer()

        generate_response(_FakeCausalLM(), tokenizer, "solve this", thinking=False, sampling=GREEDY)

        assert tokenizer.encoded_texts == [
            _as_single_user_turn(tokenizer, "solve this", thinking=False)
        ]


def _record_and_positionwise() -> tuple[GenerationRecord, dict[int, torch.Tensor]]:
    return capture_generation_activations(
        _FakeCausalLM(), _FakeTokenizer(), "solve this", thinking=True, sampling=GREEDY
    )


class TestGenerateResponse:
    def test_records_prompt_response_boundary_and_labels_positions(self) -> None:
        record = generate_response(
            _FakeCausalLM(), _FakeTokenizer(), "abc", thinking=True, sampling=GREEDY
        )

        assert record.n_generated == len(APPENDED_TOKENS)
        assert record.seq_len == record.prompt_len + len(APPENDED_TOKENS)
        # Exactly the trailing positions are response tokens.
        assert record.is_response[: record.prompt_len].sum().item() == 0
        assert bool(record.is_response[record.prompt_len :].all())
        assert record.full_ids[record.prompt_len :].tolist() == APPENDED_TOKENS
        assert len(record.token_strings) == record.seq_len

    def test_response_text_decodes_only_generated_ids(self) -> None:
        record = generate_response(
            _FakeCausalLM(), _FakeTokenizer(), "abc", thinking=True, sampling=GREEDY
        )
        assert record.response_text == " ".join(str(t) for t in APPENDED_TOKENS)


class TestThinkingPresetDefaults:
    """The module's sampling defaults must BE the thinking preset, not a copy of it.

    The bug this guards: the four constants carried the thinking preset's temperature/top_p/top_k
    beside the NON-thinking preset's 4096-token cap, under a comment claiming all four mirrored
    thinking. Every entry point here defaults ``thinking=True`` and mean-pools the model's own
    reasoning tokens, so a cap that cuts the trace mid-``<think>`` measures a truncated deliberation
    phase and reads as behaviour. Asserted against the preset object rather than the literal 32768,
    so a preset change cannot re-open the same drift one level up.
    """

    def test_generation_passes_the_thinking_preset_cap_to_generate(self) -> None:
        model = _CapRecordingLM()

        capture_response_pooled(model, _FakeTokenizer(), ["solve this"], sampling=GREEDY)

        assert model.recorded_max_new_tokens == THINKING_PRESET.max_new_tokens

    def test_the_default_sampler_is_the_thinking_preset_with_the_penalties_off(self) -> None:
        assert PENALTY_FREE_THINKING_SAMPLING.max_new_tokens == THINKING_PRESET.max_new_tokens
        assert PENALTY_FREE_THINKING_SAMPLING.temperature == THINKING_PRESET.temperature
        assert PENALTY_FREE_THINKING_SAMPLING.top_p == THINKING_PRESET.top_p
        assert PENALTY_FREE_THINKING_SAMPLING.top_k == THINKING_PRESET.top_k
        # The one deliberate divergence: the shared preset carries presence_penalty=1.5 for vLLM,
        # which transformers cannot apply and which a behavioural read must not run under anyway.
        assert THINKING_PRESET.presence_penalty > 0
        assert PENALTY_FREE_THINKING_SAMPLING.presence_penalty == 0.0
        assert PENALTY_FREE_THINKING_SAMPLING.min_p == 0.0
        assert PENALTY_FREE_THINKING_SAMPLING.repetition_penalty == 1.0


class TestHitTokenCap:
    """A record must say whether generation stopped on its own or ran into the cap.

    ``capture_response_pooled`` mean-pools every response position, so a truncated trace (all
    deliberation, no post-``</think>`` answer) contributes a differently-composed vector to the same
    group mean. Without this field an asymmetric truncation rate between the conflicting and
    original twins is invisible in the saved records.
    """

    def test_a_capped_trace_is_flagged_and_a_finished_one_is_not(self) -> None:
        capped = generate_response(
            _CapFillingLM(), _FakeTokenizer(), "abc", sampling=greedy_capped(4)
        )
        finished = generate_response(
            _FakeCausalLM(),
            _FakeTokenizer(),
            "abc",
            sampling=greedy_capped(len(APPENDED_TOKENS) + 1),
        )

        assert capped.n_generated == 4
        assert capped.hit_token_cap
        assert not finished.hit_token_cap


class TestCaptureGenerationActivations:
    def test_capture_runs_trunk_not_head_and_covers_every_position(self) -> None:
        model = _FakeCausalLM(n_layers=3, hidden=8, vocab=64)
        record, positionwise = capture_generation_activations(
            model, _FakeTokenizer(), "solve this", thinking=True, sampling=GREEDY
        )

        assert not model.lm_head.called, "capture reached the LM head; it must run only the trunk"
        assert sorted(positionwise) == [0, 1, 2]
        for layer_acts in positionwise.values():
            assert layer_acts.shape == (record.seq_len, 8)

    def test_positionwise_matches_the_layer_offsets(self) -> None:
        """Each layer adds its own offset, so layer L's rows sit exactly L*(L+1)/2 above
        the embed."""
        model = _FakeCausalLM(n_layers=3, hidden=4, vocab=64)
        record, positionwise = capture_generation_activations(
            model, _FakeTokenizer(), "abc", thinking=True, sampling=GREEDY
        )
        embed = model.model.embed_tokens(record.full_ids)
        for layer, acts in positionwise.items():
            expected_offset = sum(range(1, layer + 2))  # offsets 1..(layer+1)
            assert torch.allclose(acts, embed + expected_offset, atol=1e-5)


class TestPositionSelectors:
    def test_response_positions_is_the_response_mask(self) -> None:
        record, _ = _record_and_positionwise()
        mask = response_positions(record)
        assert int(mask.sum().item()) == len(APPENDED_TOKENS)
        assert torch.equal(mask, record.is_response)

    def test_matched_window_start_and_end_pick_the_right_tokens(self) -> None:
        record, _ = _record_and_positionwise()
        start = matched_window_positions(record, length=2, anchor="start")
        end = matched_window_positions(record, length=2, anchor="end")

        assert record.full_ids[start].tolist() == APPENDED_TOKENS[:2]
        assert record.full_ids[end].tolist() == APPENDED_TOKENS[-2:]

    def test_matched_window_clamps_to_response_length(self) -> None:
        record, _ = _record_and_positionwise()
        mask = matched_window_positions(record, length=99, anchor="start")
        assert int(mask.sum().item()) == len(APPENDED_TOKENS)

    def test_matched_window_rejects_bad_args(self) -> None:
        record, _ = _record_and_positionwise()
        with pytest.raises(ValueError, match="anchor"):
            matched_window_positions(record, length=1, anchor="middle")
        with pytest.raises(ValueError, match="positive"):
            matched_window_positions(record, length=0, anchor="start")


class TestPoolPositions:
    @staticmethod
    def _positionwise(seq: int, d: int) -> dict[int, torch.Tensor]:
        torch.manual_seed(0)
        return {0: torch.randn(seq, d), 1: torch.randn(seq, d)}

    def test_mean_pool_averages_exactly_the_selected_rows(self) -> None:
        positionwise = self._positionwise(6, 4)
        mask = torch.tensor([0, 0, 1, 1, 0, 1], dtype=torch.bool)

        pooled = pool_positions(positionwise, mask, pooling="mean")

        for layer, acts in positionwise.items():
            assert torch.allclose(pooled[layer], acts[mask].mean(dim=0), atol=1e-6)

    def test_last_pool_takes_the_final_selected_row(self) -> None:
        positionwise = self._positionwise(6, 4)
        mask = torch.tensor([0, 1, 1, 0, 1, 0], dtype=torch.bool)

        pooled = pool_positions(positionwise, mask, pooling="last")

        last_selected = int(torch.nonzero(mask).max().item())
        for layer, acts in positionwise.items():
            assert torch.allclose(pooled[layer], acts[last_selected], atol=1e-6)

    def test_empty_selection_raises(self) -> None:
        positionwise = self._positionwise(4, 3)
        with pytest.raises(ValueError, match="no positions selected"):
            pool_positions(positionwise, torch.zeros(4, dtype=torch.bool), pooling="mean")

    def test_unknown_pooling_raises(self) -> None:
        positionwise = self._positionwise(4, 3)
        with pytest.raises(ValueError, match="unknown pooling"):
            pool_positions(positionwise, torch.ones(4, dtype=torch.bool), pooling="max")


class TestCaptureResponsePooled:
    def test_stacks_one_pooled_vector_per_prompt(self) -> None:
        model = _FakeCausalLM(n_layers=2, hidden=5, vocab=64)
        prompts = ["first prompt", "second", "third one here"]

        records, pooled = capture_response_pooled(
            model, _FakeTokenizer(), prompts, pooling="mean", thinking=True, sampling=GREEDY
        )

        assert len(records) == len(prompts)
        assert sorted(pooled) == [0, 1]
        for layer_acts in pooled.values():
            assert layer_acts.shape == (len(prompts), 5)

    def test_window_selector_changes_the_pooled_vector(self) -> None:
        model = _FakeCausalLM(n_layers=1, hidden=6, vocab=64)
        prompts = ["alpha beta", "gamma"]

        _, all_response = capture_response_pooled(
            model, _FakeTokenizer(), prompts, sampling=GREEDY, position_selector=response_positions
        )
        _, first_only = capture_response_pooled(
            model,
            _FakeTokenizer(),
            prompts,
            sampling=GREEDY,
            position_selector=lambda r: matched_window_positions(r, length=1, anchor="start"),
        )

        assert not torch.allclose(all_response[0], first_only[0])


class TestCaptureRecordPooled:
    """Pooling inside the hook (hot-path backlog rank 29) reads the same bits as stash-then-pool.

    The bf16 case is the one with teeth: pooling BEFORE the float32 upcast (the sabotage: move
    ``.float()`` after the pooler in ``capture_record_pooled``'s hook) averages on the bf16 grid and
    the ``mean`` rows stop being equal, so ``torch.equal`` goes red there while float32 stays green.
    """

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["float32", "bfloat16"])
    def test_every_mask_and_pooling_equals_the_two_step_read_bitwise(
        self, dtype: torch.dtype
    ) -> None:
        model = _FakeCausalLM(n_layers=3, hidden=8, vocab=64).to(dtype)
        record = generate_response(
            model, _FakeTokenizer(), "solve this longer prompt", thinking=True, sampling=GREEDY
        )
        masks = {
            "all_response": response_positions(record),
            "window_start": matched_window_positions(record, length=2, anchor="start"),
        }
        pooled = capture_record_pooled(model, record, masks, poolings=["mean", "last"])
        positionwise = capture_record_activations(model, record)
        assert set(pooled) == {(name, pooling) for name in masks for pooling in ("mean", "last")}
        for (name, pooling), by_layer in pooled.items():
            expected = pool_positions(positionwise, masks[name], pooling=pooling)
            assert sorted(by_layer) == sorted(expected) == [0, 1, 2]
            for layer in expected:
                assert by_layer[layer].dtype == torch.float32
                assert torch.equal(by_layer[layer], expected[layer]), (name, pooling, layer)

    def test_the_capture_runs_the_trunk_not_the_head(self) -> None:
        model = _FakeCausalLM(n_layers=2, hidden=4, vocab=64)
        record = generate_response(model, _FakeTokenizer(), "abc", thinking=True, sampling=GREEDY)
        model.lm_head.called = False
        capture_record_pooled(model, record, {"all": response_positions(record)}, poolings=["mean"])
        assert not model.lm_head.called

    def test_an_empty_mask_is_refused_before_any_forward(self) -> None:
        model = _FakeCausalLM(n_layers=1, hidden=4, vocab=64)
        record = generate_response(model, _FakeTokenizer(), "abc", thinking=True, sampling=GREEDY)
        empty = torch.zeros(record.seq_len, dtype=torch.bool)
        with pytest.raises(ValueError, match="no positions selected to pool for 'empty'"):
            capture_record_pooled(model, record, {"empty": empty}, poolings=["mean"])

    def test_an_unknown_pooling_is_refused(self) -> None:
        model = _FakeCausalLM(n_layers=1, hidden=4, vocab=64)
        record = generate_response(model, _FakeTokenizer(), "abc", thinking=True, sampling=GREEDY)
        with pytest.raises(ValueError, match="unknown poolings"):
            capture_record_pooled(
                model, record, {"all": response_positions(record)}, poolings=["max"]
            )
