"""Offline tests for the exploration inference seam (``reward_hacking/model_backend.py``).

These run on CPU in the ~8s test budget: no weights load and no GPU is touched. ``MockBackend``
is exercised directly; ``HFBackend`` is exercised at construction time only, with the two
``from_pretrained`` calls monkeypatched to lightweight fakes so the config wiring and the load
recipe are checked without downloading or materialising a model.

Per the repo rule that a check never watched fail is not a check, the guards here plant the exact
violation they exist to catch: the empty-response guard, the greedy-decoding sampling-kwarg guard,
and — the one that matters most — the assertion that the model is loaded with ``dtype=`` and never
``torch_dtype=``, the footgun that silently loads Qwen3.5 in float32.
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
import signal
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar, Self, TypedDict, cast

import pytest
import torch
from conftest import MEASURED_MIN_OUTPUT_BUDGET

from reward_hacking import model_backend
from reward_hacking.model_backend import (
    DEFAULT_CODEX_MODEL,
    STOP_REASON_END_TURN,
    STOP_REASON_MAX_TOKENS,
    STOP_REASON_STOP_SEQUENCE,
    Backend,
    CodexBackend,
    DetailedBackend,
    HFBackend,
    MockBackend,
    SamplingConfig,
    TokenUsage,
    VLLMBackend,
    build_backend,
    generate_raw,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class TestMockBackendRoundRobin:
    """A list of responses is served round-robin, with the cursor persisting across calls."""

    def test_cursor_advances_across_generate_calls(self) -> None:
        mock = MockBackend(["first", "second", "third"])
        assert mock.generate(["p"]) == ["first"]
        assert mock.generate(["p"]) == ["second"]
        assert mock.generate(["p"]) == ["third"]
        assert mock.generate(["p"]) == ["first"]  # wraps back to the start

    def test_cursor_advances_within_a_batch(self) -> None:
        mock = MockBackend(["a", "b"])
        assert mock.generate(["p1", "p2", "p3"]) == ["a", "b", "a"]

    def test_one_completion_per_prompt(self) -> None:
        mock = MockBackend(["only"])
        assert mock.generate(["x", "y", "z", "w"]) == ["only", "only", "only", "only"]

    def test_empty_response_list_is_rejected(self) -> None:
        """SABOTAGE: an empty script divides by zero on first generate; watched to raise instead."""
        with pytest.raises(ValueError, match="non-empty"):
            MockBackend([])


class TestMockBackendCallable:
    """A callable serves per-prompt completions, so tests can script behaviour keyed on the
    prompt."""

    def test_callable_is_applied_per_prompt(self) -> None:
        mock = MockBackend(lambda prompt: f"echo:{prompt}")
        assert mock.generate(["x", "y"]) == ["echo:x", "echo:y"]

    def test_callable_scripts_content_dependent_behaviour(self) -> None:
        def leak_sensitive(prompt: str) -> str:
            return "READ_ANSWER_KEY" if "answer key" in prompt else "clean solve"

        mock = MockBackend(leak_sensitive)
        assert mock.generate(["peek at the answer key", "just solve it"]) == [
            "READ_ANSWER_KEY",
            "clean solve",
        ]


class TestBackendProtocol:
    """The structural contract every track imports: a ``model_id`` and a batched ``generate``."""

    def test_mock_backend_satisfies_the_protocol(self) -> None:
        assert isinstance(MockBackend(["x"]), Backend)

    def test_mock_backend_carries_its_model_id(self) -> None:
        assert MockBackend(["x"], model_id="scripted-policy").model_id == "scripted-policy"


class TestBuildBackend:
    """The ``build_backend`` dispatcher routes ``kind`` and forwards construction kwargs."""

    def test_mock_kind_forwards_model_id_and_responses(self) -> None:
        backend = build_backend("mock", "qwen-mock", responses=["r0", "r1"])
        assert isinstance(backend, MockBackend)
        assert backend.model_id == "qwen-mock"
        assert backend.generate(["p", "p"]) == ["r0", "r1"]

    def test_mock_kind_accepts_a_callable(self) -> None:
        backend = build_backend("mock", "qwen-mock", responses=lambda prompt: prompt.upper())
        assert backend.generate(["hi"]) == ["HI"]

    def test_unknown_kind_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown backend kind"):
            build_backend("gpt5", "whatever")


class TestSamplingConfigPresets:
    """The thinking-aware presets, the whole point of the fix: thinking mode gets thinking params.

    Running the non-thinking preset under enable_thinking=True loops Qwen3.5 inside <think> until
    the token budget is gone, so these pin the exact Qwen3.5 model-card values per mode.
    """

    def test_thinking_preset_is_the_qwen_thinking_general_recipe(self) -> None:
        thinking = SamplingConfig.for_thinking(thinking=True)
        assert thinking.temperature == 1.0
        assert thinking.top_p == 0.95
        assert thinking.top_k == 20
        assert thinking.min_p == 0.0
        assert thinking.repetition_penalty == 1.0
        # 1.5 rides along for the vLLM path; HFBackend never passes it (transformers has no field).
        assert thinking.presence_penalty == 1.5
        assert thinking.max_new_tokens == 32768
        assert thinking.do_sample is True

    def test_non_thinking_preset_is_the_qwen_non_thinking_recipe(self) -> None:
        non_thinking = SamplingConfig.for_thinking(thinking=False)
        assert non_thinking.temperature == 0.7
        assert non_thinking.top_p == 0.8
        assert non_thinking.top_k == 20
        assert non_thinking.min_p == 0.0
        assert non_thinking.repetition_penalty == 1.0
        assert non_thinking.presence_penalty == 0.0
        assert non_thinking.max_new_tokens == 4096
        assert non_thinking.do_sample is True


def _sampling_base(**overrides: Any) -> SamplingConfig:
    """A config for a test about some field other than the output cap, with the cap stated.

    ``SamplingConfig`` requires ``max_new_tokens`` (see ``TestNoConfigCarriesACapNobodyChose``), so
    the non-thinking preset's cap is borrowed here rather than a number being invented in a test.
    """
    return SamplingConfig(
        max_new_tokens=SamplingConfig.for_thinking(thinking=False).max_new_tokens, **overrides
    )


class TestNoConfigCarriesACapNobodyChose:
    """``SamplingConfig`` has no default output cap, because a default cap is one nobody chose.

    The field carried 1,024 as a "conservative baseline" for code that builds a config directly.
    Conservative is the wrong axis: a cap is a ceiling and a token is billed only when it is
    generated, so the cost of an over-wide cap is nothing and the cost of a too-tight one is whole
    runs of clipped reasoning read as models that could not do the task. Every real path already
    states its cap (``for_thinking``, the games training sampler, an explicit CLI flag), so
    requiring it costs the callers nothing and closes the one route by which a future path could
    inherit a number nobody picked.
    """

    def test_the_output_cap_has_no_default_at_all(self) -> None:
        cap = next(
            field for field in dataclasses.fields(SamplingConfig) if field.name == "max_new_tokens"
        )
        assert cap.default is dataclasses.MISSING
        assert cap.default_factory is dataclasses.MISSING

    def test_constructing_a_config_without_a_cap_is_an_error(self) -> None:
        with pytest.raises(TypeError, match="max_new_tokens"):
            SamplingConfig()  # pyright: ignore[reportCallIssue]  # the point of the test

    def test_the_live_call_probe_does_not_cap_below_a_reasoning_trace(self) -> None:
        """The ``python -m reward_hacking.model_backend`` credential probe used to cap at 256.

        Its default model is a reasoning model whose thinking is billed against this same budget, so
        at 256 the probe returns an empty answer -- and it exists precisely to tell a broken
        credential path or request shape from a working one, which is the reading that empty answer
        destroys. Asserted against the floor the test suite keeps by hand rather than against the
        constant the flag now reads, since comparing a value to itself passes at any setting.
        """
        args = model_backend._parse_args([])  # pyright: ignore[reportPrivateUsage]  # module CLI
        assert args.max_tokens >= MEASURED_MIN_OUTPUT_BUDGET


class _FakeModel:
    """Stands in for a loaded model: enough surface for ``__init__`` and ``_generation_kwargs``."""

    def __init__(self) -> None:
        self.device = "cpu"

    def eval(self) -> _FakeModel:
        return self


class _FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0
    eos_token_id = 1


class _RecordedLoaders(TypedDict):
    model_id: str
    model_kwargs: dict[str, object]
    tokenizer_kwargs: dict[str, object]


def _patch_loaders(monkeypatch: pytest.MonkeyPatch) -> _RecordedLoaders:
    """Replace both ``from_pretrained`` entry points with fakes that record the kwargs they see."""
    recorded: _RecordedLoaders = cast("_RecordedLoaders", {})

    def fake_model_loader(model_id: str, **kwargs: object) -> _FakeModel:
        recorded["model_id"] = model_id
        recorded["model_kwargs"] = kwargs
        return _FakeModel()

    def fake_tokenizer_loader(model_id: str, **kwargs: object) -> _FakeTokenizer:
        recorded["tokenizer_kwargs"] = kwargs
        return _FakeTokenizer()

    monkeypatch.setattr(model_backend.AutoModelForCausalLM, "from_pretrained", fake_model_loader)
    monkeypatch.setattr(model_backend.AutoTokenizer, "from_pretrained", fake_tokenizer_loader)
    return recorded


class TestHFBackendConstruction:
    """Config defaults and the load recipe, verified without materialising a model."""

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_loaders(monkeypatch)
        backend = HFBackend("Qwen/Qwen3.5-4B")
        assert backend.model_id == "Qwen/Qwen3.5-4B"
        assert backend.thinking is False
        # An unset config follows the thinking flag; thinking=False is the non-thinking preset.
        assert backend.sampling == SamplingConfig.for_thinking(thinking=False)
        assert backend.sampling.temperature == 0.7
        assert backend.sampling.top_p == 0.8
        assert backend.sampling.top_k == 20
        assert backend.sampling.max_new_tokens == 4096
        assert backend.sampling.do_sample is True
        assert isinstance(backend, Backend)

    def test_thinking_default_uses_the_thinking_preset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The bug this fixes: thinking=True with no explicit config must get thinking-mode params.

        The non-thinking preset (temperature 0.7, top_p 0.8) under enable_thinking=True loops the
        model inside <think> until the budget is spent, so the default has to follow the flag.
        """
        _patch_loaders(monkeypatch)
        backend = HFBackend("Qwen/Qwen3.5-4B", thinking=True)
        assert backend.thinking is True
        assert backend.sampling == SamplingConfig.for_thinking(thinking=True)
        assert backend.sampling.temperature == 1.0
        assert backend.sampling.top_p == 0.95
        assert backend.sampling.max_new_tokens == 32768

    def test_loads_with_dtype_never_torch_dtype(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The footgun guard: loaders silently ignore torch_dtype= and fall back to float32."""
        recorded = _patch_loaders(monkeypatch)
        HFBackend("Qwen/Qwen3.5-4B")
        model_kwargs = recorded["model_kwargs"]
        assert "dtype" in model_kwargs
        assert "torch_dtype" not in model_kwargs
        # The dtype is whatever the hardware-aware picker chose, not a hardcoded budget.
        assert model_kwargs["dtype"] is (
            model_backend._pick_dtype()  # pyright: ignore[reportPrivateUsage]  # checks internals
        )

    def test_tokenizer_is_left_padded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorded = _patch_loaders(monkeypatch)
        HFBackend("Qwen/Qwen3.5-4B")
        assert recorded["tokenizer_kwargs"]["padding_side"] == "left"

    def test_explicit_overrides_are_stored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorded = _patch_loaders(monkeypatch)
        sampling = SamplingConfig(do_sample=False, max_new_tokens=64)
        backend = HFBackend(
            "Qwen/Qwen3.5-4B", dtype=torch.float32, device="cpu", thinking=True, sampling=sampling
        )
        assert backend.thinking is True
        assert backend.sampling is sampling
        assert recorded["model_kwargs"]["dtype"] is torch.float32
        assert recorded["model_kwargs"]["device_map"] == "cpu"

    def test_sampling_kwargs_are_omitted_under_greedy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SABOTAGE target: the sampling knobs must not be passed under do_sample=False."""
        _patch_loaders(monkeypatch)
        greedy = HFBackend("Qwen/Qwen3.5-4B", sampling=_sampling_base(do_sample=False))
        kwargs: dict[str, object] = greedy._generation_kwargs()  # pyright: ignore[reportPrivateUsage]  # checks internals
        assert kwargs["do_sample"] is False
        assert "temperature" not in kwargs
        assert "top_p" not in kwargs
        assert "top_k" not in kwargs
        assert "min_p" not in kwargs
        assert "repetition_penalty" not in kwargs

        sampled = HFBackend("Qwen/Qwen3.5-4B", sampling=_sampling_base(do_sample=True))
        sampled_kwargs: dict[str, object] = sampled._generation_kwargs()  # pyright: ignore[reportPrivateUsage]  # checks internals
        assert sampled_kwargs["temperature"] == 0.7
        assert sampled_kwargs["top_p"] == 0.8
        assert sampled_kwargs["top_k"] == 20
        assert sampled_kwargs["min_p"] == 0.0
        assert sampled_kwargs["repetition_penalty"] == 1.0

    def test_presence_penalty_never_reaches_transformers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """transformers 5.15 has no presence_penalty field, so HF must never pass one, even though
        the thinking preset carries presence_penalty=1.5 for the vLLM path."""
        _patch_loaders(monkeypatch)
        thinking = HFBackend("Qwen/Qwen3.5-4B", thinking=True)
        assert thinking.sampling.presence_penalty == 1.5
        assert "presence_penalty" not in thinking._generation_kwargs()  # pyright: ignore[reportPrivateUsage]  # checks internals

    def test_build_backend_routes_to_hf(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_loaders(monkeypatch)
        backend = build_backend("hf", "Qwen/Qwen3.5-4B", thinking=True)
        assert isinstance(backend, HFBackend)
        assert backend.thinking is True


_PAD_ID = 0
_EOS_ID = 1


class _CannedBatch(dict[str, torch.Tensor]):
    """What a tokenizer call returns: a mapping of tensors that survives ``.to(device)``."""

    def to(self, device: object) -> _CannedBatch:
        """Return itself, since these tensors are already where the fake model wants them."""
        del device
        return self


class _CannedTokenizer:
    """Enough tokenizer for one batched generate: chat wrapping, padding ids, and decode.

    ``decode`` renders ids as ``t<id>`` and drops the specials, so the text a row decodes to is
    checkable without a vocabulary -- and a row whose token boundary was computed wrongly decodes
    visibly differently rather than merely counting differently.
    """

    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = _PAD_ID
    eos_token_id = _EOS_ID

    def __init__(self, prompt_ids: list[list[int]]) -> None:
        self.prompt_ids = prompt_ids

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        del kwargs
        return f"<chat>{messages[0]['content']}</chat>"

    def __call__(self, chats: list[str], **kwargs: object) -> _CannedBatch:
        del kwargs
        assert len(chats) == len(self.prompt_ids), "the canned prompts must match the batch"
        return _CannedBatch(
            input_ids=torch.tensor(self.prompt_ids),
            attention_mask=torch.tensor(
                [[0 if token == _PAD_ID else 1 for token in row] for row in self.prompt_ids]
            ),
        )

    def decode(self, ids: object, *, skip_special_tokens: bool = False) -> str:
        assert skip_special_tokens, "the backend decodes answers, never padding"
        return " ".join(f"t{token}" for token in cast("list[int]", ids) if token > _EOS_ID)


class _CannedModel:
    """A model whose ``generate`` replays a recorded batch of ids and records the kwargs it saw."""

    def __init__(self, outputs: list[list[int]]) -> None:
        self.device = "cpu"
        self.outputs = torch.tensor(outputs)
        self.seen_kwargs: dict[str, object] = {}

    def eval(self) -> _CannedModel:
        return self

    def generate(self, **kwargs: object) -> torch.Tensor:
        self.seen_kwargs = kwargs
        return self.outputs


def _canned_hf_backend(
    monkeypatch: pytest.MonkeyPatch,
    *,
    prompt_ids: list[list[int]],
    outputs: list[list[int]],
    cap: int,
) -> HFBackend:
    """Build an ``HFBackend`` over canned ids: no weights, no vocabulary, no download."""
    _patch_loaders(monkeypatch)
    backend = HFBackend("Qwen/Qwen3.5-4B", sampling=SamplingConfig(max_new_tokens=cap))
    monkeypatch.setattr(backend, "_tokenizer", _CannedTokenizer(prompt_ids))
    monkeypatch.setattr(backend, "_model", _CannedModel(outputs))
    monkeypatch.setattr(backend, "_terminator_ids", frozenset({_PAD_ID, _EOS_ID}))
    return backend


class TestALocalRunSaysWhyEachRowStopped:
    """A local completion that ran out of output budget has to be distinguishable from one that
    finished.

    Nothing local could say so before: ``HFBackend`` satisfied only ``Backend``, so ``generate_raw``
    took its bare-strings branch and every turn a local run wrote to disk carried
    ``stop_reason: null`` and ``output_tokens: null``. Across the 4,258 turns retained in
    ``artifacts/harness`` that is 100% of them, including a 4B smoke whose longest turn ran to
    ~90,000 characters against a 32,768-token cap -- exactly the row where "hit the cap" versus
    "chose to stop" decides whether the episode is a finding or a config bug.

    The labels are read off the token ids rather than the text, because batched generation only
    returns before the cap once every row has finished: a row carrying the tokenizer's EOS (or the
    padding it was filled to the batch length with) terminated, and a row carrying neither ran to
    the cap.

    The canned batch below puts both cases side by side, which is the arrangement that catches a
    per-row bookkeeping mistake: row 0 answers in one token and stops, row 1 talks until the
    4-token cap, and left padding holds row 0's shorter prompt behind a pad so the input-token
    count is worth checking too.
    """

    PROMPT_IDS: ClassVar[list[list[int]]] = [[_PAD_ID, 10, 11], [12, 13, 14]]
    OUTPUT_IDS: ClassVar[list[list[int]]] = [
        [_PAD_ID, 10, 11, 20, _EOS_ID, _PAD_ID, _PAD_ID],
        [12, 13, 14, 21, 22, 23, 24],
    ]
    CAP = 4

    def _backend(self, monkeypatch: pytest.MonkeyPatch) -> HFBackend:
        return _canned_hf_backend(
            monkeypatch, prompt_ids=self.PROMPT_IDS, outputs=self.OUTPUT_IDS, cap=self.CAP
        )

    def test_a_row_that_ran_to_the_cap_is_labelled_as_such(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        completions = self._backend(monkeypatch).generate_detailed(["short", "long"])

        assert completions[1].stop_reason == STOP_REASON_MAX_TOKENS
        assert completions[1].usage.output_tokens == self.CAP

    def test_a_row_that_finished_is_labelled_as_finished_and_counted_to_its_terminator(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The padding a finished row is filled with is not output it generated."""
        completions = self._backend(monkeypatch).generate_detailed(["short", "long"])

        assert completions[0].stop_reason == STOP_REASON_END_TURN
        # One answer token plus the end-of-sequence token; the two trailing pads are the batch's.
        assert completions[0].usage.output_tokens == 2

    def test_the_prompt_count_is_the_real_prompt_not_its_padding(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        completions = self._backend(monkeypatch).generate_detailed(["short", "long"])

        assert [completion.usage.input_tokens for completion in completions] == [2, 3]

    def test_the_text_is_the_generated_span_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        completions = self._backend(monkeypatch).generate_detailed(["short", "long"])

        assert [completion.text for completion in completions] == ["t20", "t21 t22 t23 t24"]

    def test_generate_returns_exactly_the_detailed_texts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One code path, so plain and detailed calls cannot drift into different completions."""
        backend = self._backend(monkeypatch)

        assert backend.generate(["short", "long"]) == [
            completion.text for completion in backend.generate_detailed(["short", "long"])
        ]

    def test_generate_raw_takes_the_detailed_branch_for_a_local_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SABOTAGE target: the seam that decides whether any of this reaches a trace.

        ``generate_raw`` keys on the ``DetailedBackend`` protocol, so an ``HFBackend`` that stopped
        satisfying it would silently go back to recording ``None`` for every field while every
        assertion above still passed.
        """
        backend = self._backend(monkeypatch)
        assert isinstance(backend, DetailedBackend)

        responses = generate_raw(backend, ["short", "long"])

        assert [response.stop_reason for response in responses] == [
            STOP_REASON_END_TURN,
            STOP_REASON_MAX_TOKENS,
        ]
        assert [response.output_tokens for response in responses] == [2, self.CAP]


class TestAConfiguredStopStringReachesGenerateAndLabelsTheHalt:
    """The HF half of the harness's ``</run>`` stop sequence, checked at its consumption points.

    Two seams, both silent if wrong. Construction: ``stop_strings`` only stops anything if it is
    in the kwargs ``generate`` actually receives, together with the ``tokenizer=`` transformers
    requires for it -- a config field nothing forwards samples exactly as before, and the agent
    harness goes back to executing commands the model conditioned on its own invented tool
    output. Labelling: a stop-string halt ends a row on an ordinary token, no terminator, so the
    id-level span reads a sole stopped row as a cap hit -- ``turns_truncated`` would then count
    every stopped turn as clipped, on every single harness turn that used the stop.
    """

    def _stopping_backend(self, monkeypatch: pytest.MonkeyPatch, *, stop: tuple[str, ...]):
        """A canned backend whose one row decodes to ``t20 t21`` with no terminator token.

        Under an applied stop string that is exactly what a halt at ``t21`` leaves behind; without
        one it is what running to the cap leaves behind -- the ambiguity the label resolves.
        """
        backend = _canned_hf_backend(
            monkeypatch, prompt_ids=[[10, 11]], outputs=[[10, 11, 20, 21]], cap=2
        )
        backend.sampling = dataclasses.replace(backend.sampling, stop=stop)
        return backend

    def test_the_stop_strings_and_tokenizer_are_in_the_generate_kwargs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = self._stopping_backend(monkeypatch, stop=("</run>",))
        backend.generate_detailed(["prompt"])

        model = cast("_CannedModel", backend._model)
        assert model.seen_kwargs["stop_strings"] == ["</run>"]
        assert model.seen_kwargs["tokenizer"] is backend._tokenizer

    def test_no_stop_configured_means_no_stop_kwargs_at_all(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The default stays byte-identical: nothing new in the request nobody asked for."""
        backend = self._stopping_backend(monkeypatch, stop=())
        backend.generate_detailed(["prompt"])

        model = cast("_CannedModel", backend._model)
        assert "stop_strings" not in model.seen_kwargs
        assert "tokenizer" not in model.seen_kwargs

    def test_a_reply_carrying_the_stop_text_is_labelled_stop_sequence_not_cap_hit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = self._stopping_backend(monkeypatch, stop=("t21",))
        completion = backend.generate_detailed(["prompt"])[0]

        assert completion.text == "t20 t21"
        assert completion.stop_reason == model_backend.STOP_REASON_STOP_SEQUENCE

    def test_the_same_reply_with_no_stop_configured_is_still_a_cap_hit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SABOTAGE control: the override must key on the config, not fire on any text."""
        backend = self._stopping_backend(monkeypatch, stop=())
        completion = backend.generate_detailed(["prompt"])[0]

        assert completion.stop_reason == STOP_REASON_MAX_TOKENS

    def test_a_finished_reply_without_the_stop_text_keeps_its_end_turn_label(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _canned_hf_backend(
            monkeypatch, prompt_ids=[[10, 11]], outputs=[[10, 11, 20, _EOS_ID]], cap=2
        )
        backend.sampling = dataclasses.replace(backend.sampling, stop=("t21",))
        completion = backend.generate_detailed(["prompt"])[0]

        assert completion.stop_reason == STOP_REASON_END_TURN


class TestNeitherLocalBackendGrantsTrustRemoteCode:
    """SABOTAGE target: granting ``trust_remote_code`` in the inference path reddens exactly here.

    The training paths (``games/train.py``, ``games/lora.py``, ``grpo/rlvr_math.py``) pass the flag
    and these two do not, which reads like an oversight and is not. Measured 2026-08-24 against a
    local checkpoint whose ``auto_map`` names a module: WITH the flag transformers executes that
    module's code in this process (the probe module raised, proving execution), WITHOUT it the load
    raises and the module never runs. So the flag is host-side arbitrary code execution at backend
    construction, outside the bubblewrap jail this package exists to keep model-influenced code
    inside.

    The asymmetry is safe rather than silent because of the second half of that measurement: a
    checkpoint requiring remote code RAISES here instead of quietly resolving some other tokenizer,
    so a future checkpoint that needs it announces itself at load time rather than producing
    different prompt text than training used. Qwen3.5 needs no remote code at any size, so the two
    paths agree today. This test is what makes a later "fix" of the asymmetry a deliberate act.
    """

    def test_the_hf_backend_withholds_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorded = _patch_loaders(monkeypatch)
        HFBackend("Qwen/Qwen3.5-4B")

        assert "trust_remote_code" not in recorded["tokenizer_kwargs"]
        assert "trust_remote_code" not in recorded["model_kwargs"]

    def test_the_vllm_backend_withholds_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorded = _patch_loaders(monkeypatch)
        _patch_fake_vllm(monkeypatch)
        model_backend.VLLMBackend("Qwen/Qwen3.5-4B", sampling=SamplingConfig(max_new_tokens=64))

        # The kwargs the construction really passed. Reading a key nothing wrote would KeyError
        # rather than pass, so a helper that shadowed this recorder cannot make it go vacuous.
        assert recorded["tokenizer_kwargs"] == {}

    def test_the_reason_is_written_down_where_a_future_editor_will_look(self) -> None:
        """A bare absence is indistinguishable from an oversight, which is how it gets "fixed"."""
        reason = model_backend.TRUST_REMOTE_CODE_WITHHELD_REASON
        assert "outside the episode jail" in reason
        assert "RAISES" in reason


FINAL_ONLY_OUTPUT_KIND = "final-only"
"""What the fake ``vllm.sampling_params.RequestOutputKind.FINAL_ONLY`` resolves to under the fakes."""


class _RecordingSamplingParams:
    """Records the kwargs ``VLLMBackend`` builds its ``SamplingParams`` with.

    A recorder rather than the real class because constructing the real backend must not load an
    engine, and the question here is what the backend PASSES -- the installed engine's own handling
    of ``stop`` / ``include_stop_str_in_output`` / ``seed`` was verified live on the L4 rather than
    in this suite (a unit test cannot answer whether vLLM honours a field).
    """

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = dict(kwargs)
        self.sampling_type = "random"
        self.output_kind = "cumulative"

    def clone(self) -> _RecordingSamplingParams:
        """``SamplingParams.clone``: a copy the streaming drain may set its output kind on."""
        clone = _RecordingSamplingParams(**self.kwargs)
        clone.output_kind = self.output_kind
        return clone


class _FakeVllmModule:
    """The three attributes ``VLLMBackend.__init__`` reaches for on ``vllm``."""

    SamplingParams = _RecordingSamplingParams

    class LLM:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = dict(kwargs)


def _patch_fake_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point ``importlib.import_module`` at fake vllm modules: no engine, no weights, no GPU.

    Deliberately does NOT patch the tokenizer and model loaders. A caller that wants to inspect
    what the backend passed those has to install its own recorder, and a helper that quietly
    installed a second one would leave the caller reading a dict nothing wrote to.
    """
    fake_lora = SimpleNamespace(LoRARequest=lambda *args: SimpleNamespace(args=args))
    fake_params = SimpleNamespace(
        SamplingType=SimpleNamespace(GREEDY="greedy"),
        RequestOutputKind=SimpleNamespace(FINAL_ONLY=FINAL_ONLY_OUTPUT_KIND),
    )
    modules: dict[str, object] = {
        "vllm": _FakeVllmModule(),
        "vllm.lora.request": fake_lora,
        "vllm.sampling_params": fake_params,
    }
    monkeypatch.setattr(
        model_backend.importlib, "import_module", lambda name: modules[name], raising=True
    )


def _vllm_sampling_params(
    monkeypatch: pytest.MonkeyPatch, sampling: SamplingConfig
) -> _RecordingSamplingParams:
    """Construct a ``VLLMBackend`` over the fakes and return the ``SamplingParams`` it built.

    Returns the recorded params rather than the backend so the private-attribute read lives in one
    place instead of at every call site.
    """
    _patch_loaders(monkeypatch)
    _patch_fake_vllm(monkeypatch)
    backend = model_backend.VLLMBackend("Qwen/Qwen3.5-4B", sampling=sampling)
    return cast("_RecordingSamplingParams", backend._sampling_params)


class TestTheVllmBackendPassesTheStopAndTheSeedItWasConfiguredWith:
    """The vLLM consumption point for both fields, checked as kwargs the backend actually passes.

    Both are silent when dropped: an unpassed stop lets the policy write the environment's replies,
    and an unpassed seed makes a run that reports a seed generate unseeded -- which is the bug the
    seed field was added for (two runs nearly read as seed-matched off a placebo ``seed: 0``).
    """

    def test_a_configured_stop_arrives_with_the_stop_text_kept(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        params = _vllm_sampling_params(
            monkeypatch, SamplingConfig(max_new_tokens=64, stop=("</run>",))
        )

        assert params.kwargs["stop"] == ["</run>"]
        # False here would strip the run-block close and leave the harness parser nothing to match.
        assert params.kwargs["include_stop_str_in_output"] is True

    def test_no_stop_configured_passes_none_and_leaves_the_text_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        params = _vllm_sampling_params(monkeypatch, SamplingConfig(max_new_tokens=64))

        assert params.kwargs["stop"] is None
        assert params.kwargs["include_stop_str_in_output"] is False

    @pytest.mark.parametrize("seed", [None, 0, 7])
    def test_the_seed_arrives_exactly_as_configured_including_zero(
        self, seed: int | None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Zero must arrive as 0 and unseeded as None: collapsing them is the placebo-seed bug."""
        params = _vllm_sampling_params(monkeypatch, SamplingConfig(max_new_tokens=64, seed=seed))

        assert params.kwargs["seed"] == seed


class TestTheGenerationSeedFieldIsHonestPerBackend:
    """``SamplingConfig.seed``: ``None`` means unseeded, and the two backends treat it differently.

    The field exists because two runs were nearly misread as seed-matched off a placebo ``seed: 0``
    in their metadata while generation was actually unseeded. The model_backend half of that fix is
    pinned here (the field's default, the vLLM mapping, and the HF path never forwarding a knob
    transformers has no argument for); the resolved-sampler partition that *records* the drop lives
    with the interp readout and its own tests.
    """

    def test_unseeded_is_none_not_zero(self) -> None:
        assert SamplingConfig(max_new_tokens=1).seed is None
        assert SamplingConfig(max_new_tokens=1, seed=0).seed == 0

    def test_the_vllm_param_mapping_carries_seed(self) -> None:
        """``applied_sampling`` walks this mapping, so a missing entry silently unrecords the seed."""
        assert model_backend.VLLM_SAMPLING_PARAM_ATTRS["seed"] == "seed"

    def test_the_hf_path_never_forwards_a_seed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """transformers' generate has no seed argument, so forwarding one would crash the call --
        and quietly swallowing it into kwargs the model ignores would be the placebo-seed bug again.
        """
        backend = _canned_hf_backend(
            monkeypatch, prompt_ids=[[10, 11]], outputs=[[10, 11, 20, _EOS_ID]], cap=2
        )
        backend.sampling = dataclasses.replace(backend.sampling, seed=7)
        backend.generate_detailed(["prompt"])

        model = cast("_CannedModel", backend._model)
        assert "seed" not in model.seen_kwargs


@pytest.fixture(scope="module")
def real_vllm() -> ModuleType:
    """The installed engine package itself, not a fake.

    Every other vLLM test in this module builds the backend over ``_FakeVllmModule``, which is the
    only way to reach the construction path without loading an engine. That leaves one whole class of
    error undetectable: an attribute name that vLLM does not have. The one test that actually runs
    ``applied_sampling`` (``test_interp_vllm_generation._StubSamplingParams``) mirrors
    :data:`~reward_hacking.model_backend.VLLM_SAMPLING_PARAM_ATTRS` by hand, so the map is only ever
    compared with itself, and a wrong entry surfaces as an ``AttributeError`` on a rented box, twenty
    minutes into the run, after the weights have loaded.

    Module-scoped because importing vLLM costs a few seconds and no test here mutates it. Nothing is
    instantiated beyond ``SamplingParams``, so no CUDA context and no engine.
    """
    return pytest.importorskip("vllm", reason="vllm is an optional GPU-box dependency")


class TestTheMappedSamplingAttributesExistOnTheRealEngine:
    """Every name this repo asks vLLM for, asked of the installed vLLM rather than of a stub.

    Pinned at 0.27.1, which is what the lockfile installs. These are the three ways the vLLM seam
    can break on an upstream rename, and all three are silent in the rest of the suite: a mapped
    attribute :meth:`VLLMBackend.applied_sampling` reads back, a symbol imported by string, and the
    kwargs the constructor passes.

    SABOTAGE: pointing one entry of ``VLLM_SAMPLING_PARAM_ATTRS`` at a name ``SamplingParams`` does
    not carry turns the first test red. Watched, and watched to report *passed* rather than
    *skipped* -- a check that silently skips is the failure shape this repo is built around.
    """

    def test_every_mapped_attribute_is_one_sampling_params_carries(
        self, real_vllm: ModuleType
    ) -> None:
        params = real_vllm.SamplingParams(max_tokens=8)

        missing = [
            f"{field} -> {attr}"
            for field, attr in model_backend.VLLM_SAMPLING_PARAM_ATTRS.items()
            if not hasattr(params, attr)
        ]
        assert not missing, (
            f"applied_sampling would raise AttributeError on a live engine for {missing}; "
            f"vllm {real_vllm.__version__} has no such field"
        )
        # Every SamplingConfig field except do_sample, which vLLM spells as temperature 0.0 and
        # applied_sampling reads off sampling_type instead.
        mapped = set(model_backend.VLLM_SAMPLING_PARAM_ATTRS)
        fields = {field.name for field in dataclasses.fields(SamplingConfig)}
        assert fields - mapped == {"do_sample"}

    def test_the_greedy_sampling_type_still_resolves_by_string(self, real_vllm: ModuleType) -> None:
        """``SamplingType`` is not exported from vLLM's top level, so the backend imports the
        submodule by name and reaches ``GREEDY`` on it. A rename there is an ImportError or an
        AttributeError at engine construction, on the box, after the model load.
        """
        del real_vllm
        greedy = importlib.import_module("vllm.sampling_params").SamplingType.GREEDY

        # applied_sampling reports do_sample by comparing the engine's own sampling_type to this, so
        # the two must be distinguishable: a default (sampled) request must not read as greedy.
        assert importlib.import_module("vllm").SamplingParams(max_tokens=8).sampling_type != greedy

    def test_the_exact_constructor_kwargs_the_backend_passes_are_accepted(
        self, real_vllm: ModuleType
    ) -> None:
        """The kwarg NAMES, on the real class -- including ``include_stop_str_in_output``, which is
        not a ``SamplingConfig`` field and so appears in no mapping any other test walks. With it
        false vLLM strips the matched stop string and the harness's run-block parser then sees an
        unclosed block, which reads as a policy that never closed its command.
        """
        sampling = SamplingConfig(max_new_tokens=64, stop=("</run>",), seed=7)

        params = real_vllm.SamplingParams(
            max_tokens=sampling.max_new_tokens,
            temperature=sampling.temperature if sampling.do_sample else 0.0,
            top_p=sampling.top_p,
            top_k=sampling.top_k,
            min_p=sampling.min_p,
            repetition_penalty=sampling.repetition_penalty,
            presence_penalty=sampling.presence_penalty,
            seed=sampling.seed,
            stop=list(sampling.stop),
            include_stop_str_in_output=bool(sampling.stop),
        )

        assert params.include_stop_str_in_output is True
        assert params.seed == 7
        # The list-versus-tuple coercion applied_sampling performs, measured on the real class here
        # rather than remembered: without it a configured () compares unequal to the engine's [].
        assert params.stop == ["</run>"]
        assert tuple(params.stop) == sampling.stop


class TestVllmStopReasonTranslation:
    """``_vllm_stop_reason``: only a matched stop STRING is a stop-sequence halt.

    vLLM spells a stop-string halt and an ordinary end-of-turn identically in ``finish_reason``
    (both ``"stop"``); the matched stop string in ``CompletionOutput.stop_reason`` is what tells
    them apart, and a stop TOKEN id there is the model's own terminator by another spelling, not a
    caller-configured stop. Folding these together would make a harness turn that halted at
    ``</run>`` unreadable from one that finished -- the exact distinction the field records.
    """

    @pytest.mark.parametrize(
        ("finish_reason", "matched_stop", "expected"),
        [
            ("stop", "</run>", model_backend.STOP_REASON_STOP_SEQUENCE),
            ("stop", None, STOP_REASON_END_TURN),
            ("stop", 151645, STOP_REASON_END_TURN),  # a stop token id is not a stop string
            ("length", None, STOP_REASON_MAX_TOKENS),
            ("abort", None, "abort"),  # no Converse counterpart: travels under its own name
            (None, None, None),
        ],
    )
    def test_translation(
        self, finish_reason: str | None, matched_stop: object, expected: str | None
    ) -> None:
        assert model_backend._vllm_stop_reason(finish_reason, matched_stop) == expected


def test_module_load_does_not_import_vllm() -> None:
    """Anti-goal guard: vllm is imported lazily inside its backend, never at module load.

    Checked in a fresh interpreter rather than against this process's ``sys.modules``: with the
    vllm extra installed (it is, for the GPU training path), TRL pulls vllm in as a side effect
    of the games tests' import chain, so a process-global assertion measures whichever test
    module happened to import first -- not this module. The subprocess isolates the actual
    claim: importing ``reward_hacking.model_backend`` by itself must leave vllm unimported.
    """
    code = (
        "import sys\n"
        "import reward_hacking.model_backend\n"
        "sys.exit(1 if 'vllm' in sys.modules else 0)\n"
    )
    repo_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(  # noqa: S603 - our own interpreter, fixed argv
        [sys.executable, "-c", code], cwd=repo_root, check=False
    )
    assert completed.returncode == 0, "importing reward_hacking.model_backend imported vllm"


class _FakeCodexProcess:
    """Stand-in for the ``Popen`` handle :meth:`CodexBackend._run_bounded` drives.

    ``pid`` is a real-looking value the group-kill assertion can follow, and ``hang`` makes
    ``communicate`` raise the expiry the timeout path exists to handle.
    """

    def __init__(
        self, *, returncode: int, stderr: str, on_communicate: Callable[[], None], hang: bool
    ) -> None:
        self.pid = 424242
        self.returncode = returncode
        self._stderr = stderr
        self._on_communicate = on_communicate
        self._hang = hang
        self.waited = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        self._on_communicate()
        if self._hang:
            raise subprocess.TimeoutExpired(cmd="codex", timeout=timeout or 0.0)
        return ("", self._stderr)

    def wait(self) -> int:
        self.waited = True
        return self.returncode


def _fake_codex_run(
    monkeypatch: pytest.MonkeyPatch,
    completion: str,
    *,
    returncode: int = 0,
    write_output: bool = True,
    hang: bool = False,
) -> dict[str, Any]:
    """Replace subprocess.Popen so no real codex spawns; write the completion to the ``-o`` path.

    Returns a dict the caller can inspect for the exact argv and kwargs the backend passed, which
    is how the argv shape, the DEVNULL stdin and the new-session flag are asserted without a live
    CLI. ``hang=True`` makes the fake process outlive its wall-clock bound instead of answering,
    and records every ``killpg`` the backend issues under ``captured["killpg"]``.
    """
    captured: dict[str, Any] = {"killpg": []}

    def fake_popen(argv: list[str], **kwargs: object) -> _FakeCodexProcess:
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        out_path = argv[argv.index("-o") + 1]

        def write_completion() -> None:
            if write_output and not hang:
                Path(out_path).write_text(completion)

        process = _FakeCodexProcess(
            returncode=returncode,
            stderr="" if returncode == 0 else "codex boom",
            on_communicate=write_completion,
            hang=hang,
        )
        captured["process"] = process
        return process

    monkeypatch.setattr(model_backend.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(model_backend.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(
        model_backend.os, "killpg", lambda pgid, sig: captured["killpg"].append((pgid, sig))
    )
    return captured


class TestCodexBackend:
    """The codex CLI shelled as a pure text generator: argv shape, output capture, fail-fast."""

    def test_builds_the_expected_argv(self) -> None:
        backend = CodexBackend(codex_bin="/fake/bin/codex")
        argv = backend._build_argv(
            "solve it", workdir="/scratch/wd", out_path="/scratch/wd/out.txt"
        )
        assert argv == [
            "/fake/bin/codex",
            "exec",
            "-s",
            "read-only",
            "-C",
            "/scratch/wd",
            "--skip-git-repo-check",
            "-o",
            "/scratch/wd/out.txt",
            "-m",
            DEFAULT_CODEX_MODEL,
            "--",
            "solve it",
        ]

    def test_the_prompt_is_passed_after_an_end_of_options_separator(self) -> None:
        """``codex exec`` has subcommands, so an unseparated prompt can be parsed as one.

        Verified against this box's build: ``codex exec review <anything>`` enters the review
        subcommand's parser, so a corpus item opening with the word "review" (or "resume", or
        "help") never reaches the model, and a prompt opening with a hyphen is read as a flag.
        Both fail as a RuntimeError that reads like a CLI problem rather than a prompt problem.
        """
        backend = CodexBackend(codex_bin="/fake/bin/codex")
        for prompt in ("review this diff", "resume from here", "--help me out"):
            argv = backend._build_argv(prompt, workdir="/w", out_path="/w/o.txt")
            assert argv[-2:] == ["--", prompt]

    def test_reasoning_effort_is_appended_when_set(self) -> None:
        backend = CodexBackend(codex_bin="/fake/bin/codex", reasoning_effort="high")
        argv = backend._build_argv("p", workdir="/w", out_path="/w/o.txt")
        assert "-c" in argv
        assert "model_reasoning_effort=high" in argv
        # The prompt stays last so codex reads it as the instruction, not a flag value.
        assert argv[-1] == "p"

    def test_no_reasoning_flag_by_default(self) -> None:
        backend = CodexBackend(codex_bin="/fake/bin/codex")
        assert "-c" not in backend._build_argv("p", workdir="/w", out_path="/w/o.txt")

    def test_generate_reads_the_output_file_and_closes_stdin(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _fake_codex_run(monkeypatch, "MODEL ACTION TEXT")
        backend = CodexBackend(codex_bin="/fake/bin/codex")
        assert backend.generate(["hello"]) == ["MODEL ACTION TEXT"]
        assert captured["argv"][0] == "/fake/bin/codex"
        # stdin must be closed, or `codex exec` blocks forever reading it.
        assert captured["kwargs"]["stdin"] == subprocess.DEVNULL
        # A new session makes the child a group leader, so the expiry path can kill the tree.
        assert captured["kwargs"]["start_new_session"] is True

    def test_a_call_that_outlives_its_bound_raises_and_kills_the_process_group(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SABOTAGE target: an unkilled codex keeps a Bedrock stream open after we stop waiting.

        Measured on this box, ``subprocess.run(timeout=...)`` leaves two live codex processes
        because the PATH entry is a shim and the real binary is a grandchild. The bound is the
        only wall-clock limit available: codex refuses ``stream_idle_timeout_ms`` and
        ``stream_max_retries`` overrides on its built-in ``amazon-bedrock`` provider, and an
        endpoint that accepts the connection then answers nothing leaves it asleep on the socket.
        """
        captured = _fake_codex_run(monkeypatch, "never arrives", hang=True)
        backend = CodexBackend(codex_bin="/fake/bin/codex", timeout=1.0)
        with pytest.raises(RuntimeError, match="exceeded 1s and its process group was killed"):
            backend.generate(["hi"])
        process = captured["process"]
        assert captured["killpg"] == [(process.pid, signal.SIGKILL)]
        assert process.waited, "the killed group must be reaped, not left as a zombie"

    def test_generate_is_one_completion_per_prompt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake_codex_run(monkeypatch, "same")
        backend = CodexBackend(codex_bin="/fake/bin/codex")
        assert backend.generate(["a", "b", "c"]) == ["same", "same", "same"]

    def test_nonzero_exit_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SABOTAGE target: a silent failure would return an empty completion the loop trusts."""
        _fake_codex_run(monkeypatch, "", returncode=2, write_output=False)
        backend = CodexBackend(codex_bin="/fake/bin/codex")
        with pytest.raises(RuntimeError, match="codex exec failed"):
            backend.generate(["hi"])

    def test_missing_output_file_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A zero exit but no ``-o`` file also raises rather than returning an empty string."""
        _fake_codex_run(monkeypatch, "", returncode=0, write_output=False)
        backend = CodexBackend(codex_bin="/fake/bin/codex")
        with pytest.raises(RuntimeError, match="no --output-last-message"):
            backend.generate(["hi"])

    def test_missing_binary_raises_at_construction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(model_backend.shutil, "which", lambda _name: None)
        with pytest.raises(RuntimeError, match="codex CLI not found"):
            CodexBackend()

    def test_satisfies_the_backend_protocol(self) -> None:
        assert isinstance(CodexBackend(codex_bin="/fake/bin/codex"), Backend)

    def test_build_backend_routes_to_codex(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake_codex_run(monkeypatch, "x")
        backend = build_backend("codex", "openai.gpt-5.6-luna", codex_bin="/fake/bin/codex")
        assert isinstance(backend, CodexBackend)
        assert backend.model_id == "openai.gpt-5.6-luna"

    def test_unknown_kind_message_lists_codex(self) -> None:
        with pytest.raises(ValueError, match="'codex'"):
            build_backend("gemini", "whatever")


# --- serving a verified snapshot directory under a label (the full-weights rung) --------------------


class TestServingFromASnapshotDirectory:
    """``model_path`` decides where the weights come from; ``model_id`` stays the record's label."""

    def test_hf_loads_model_and_tokenizer_from_the_path_and_keeps_the_label(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        recorded = _patch_loaders(monkeypatch)
        backend = HFBackend("allenai/tmax-4b@step_300", model_path=tmp_path / "snap")
        assert backend.model_id == "allenai/tmax-4b@step_300"
        assert backend.model_path == str(tmp_path / "snap")
        assert recorded["model_id"] == str(tmp_path / "snap")

    def test_hf_without_a_path_loads_the_id_as_before(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorded = _patch_loaders(monkeypatch)
        backend = HFBackend("Qwen/Qwen3.5-4B")
        assert backend.model_path is None
        assert recorded["model_id"] == "Qwen/Qwen3.5-4B"

    def test_hf_asserts_the_directory_its_config_names(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_loaders(monkeypatch)
        backend = HFBackend("label", model_path=tmp_path / "snap")
        monkeypatch.setattr(
            backend,
            "_model",
            SimpleNamespace(config=SimpleNamespace(name_or_path=str(tmp_path / "snap"))),
        )
        backend.assert_serves_weights(tmp_path / "snap")
        with pytest.raises(RuntimeError, match="not from the verified snapshot"):
            backend.assert_serves_weights(tmp_path / "other")

    def test_vllm_asserts_the_directory_its_engine_reports(self, tmp_path: Path) -> None:
        """Read off the engine's own record, not the constructor argument."""
        backend = object.__new__(model_backend.VLLMBackend)
        engine = SimpleNamespace(
            llm_engine=SimpleNamespace(model_config=SimpleNamespace(model=str(tmp_path / "snap")))
        )
        backend._llm = engine  # pyright: ignore[reportPrivateUsage]  # a stub engine
        backend.assert_serves_weights(tmp_path / "snap")
        with pytest.raises(RuntimeError, match="not the verified snapshot"):
            backend.assert_serves_weights(tmp_path / "elsewhere")

    def test_vllm_builds_its_engine_and_tokenizer_from_the_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The engine kwargs a full-weights decision derives must reach ``vllm.LLM`` untouched."""
        built: list[dict[str, object]] = []

        class FakeLLM:
            def __init__(self, **kwargs: object) -> None:
                built.append(kwargs)

        class FakeSamplingParams:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

        fake_vllm = SimpleNamespace(
            LLM=FakeLLM,
            SamplingParams=FakeSamplingParams,
            SamplingType=SimpleNamespace(GREEDY="greedy"),
            LoRARequest=object,
        )
        monkeypatch.setattr(model_backend.importlib, "import_module", lambda _name: fake_vllm)
        tokenizer_loads: list[str] = []

        def fake_tokenizer_loader(model_id: str, **_: object) -> _FakeTokenizer:
            tokenizer_loads.append(model_id)
            return _FakeTokenizer()

        monkeypatch.setattr(model_backend.AutoTokenizer, "from_pretrained", fake_tokenizer_loader)
        backend = model_backend.VLLMBackend(
            "allenai/tmax-4b@step_300",
            thinking=True,
            sampling=_sampling_base(),
            model_path=tmp_path / "snap",
            language_model_only=True,
            max_model_len=32768,
        )
        assert backend.model_id == "allenai/tmax-4b@step_300"
        assert tokenizer_loads == [str(tmp_path / "snap")]
        assert built == [
            {"model": str(tmp_path / "snap"), "language_model_only": True, "max_model_len": 32768}
        ]


# --- the end-of-turn stop set, pinned by name on every served unit ------------------------------------


class _VocabTokenizer:
    """A tokenizer that knows a fixed vocabulary of special tokens, TMAX- or base-shaped alike."""

    def __init__(self, vocab: dict[str, int], *, unk_token_id: int | None = None) -> None:
        self.vocab = vocab
        self.unk_token_id = unk_token_id

    def convert_tokens_to_ids(self, token: str) -> int | None:
        if token in self.vocab:
            return self.vocab[token]
        return self.unk_token_id


def _vocab_tokenizer(vocab: dict[str, int], *, unk_token_id: int | None = None) -> Any:
    """The stub typed as the AutoTokenizer the pin helper is annotated with."""
    return _VocabTokenizer(vocab, unk_token_id=unk_token_id)


QWEN_SPECIALS = {"<|im_end|>": 248046, "<|endoftext|>": 248044, "<|im_start|>": 248045}


class TestEndOfTurnStopSet:
    def test_a_tmax_unit_and_a_base_unit_resolve_the_same_set(self) -> None:
        """The pin reads the TOKENIZER, so a checkpoint's generation_config (TMAX: a single eos
        248044; upstream: none at all) cannot make two units of one screen halt differently."""
        base = _vocab_tokenizer(QWEN_SPECIALS)
        tmax = _vocab_tokenizer(dict(QWEN_SPECIALS))
        assert model_backend.end_of_turn_token_ids(base) == (248044, 248046)
        assert model_backend.end_of_turn_token_ids(tmax) == model_backend.end_of_turn_token_ids(
            base
        )

    def test_a_tokenizer_missing_a_token_is_refused_not_half_pinned(self) -> None:
        with pytest.raises(ValueError, match="does not know '<\\|endoftext\\|>'"):
            model_backend.end_of_turn_token_ids(_vocab_tokenizer({"<|im_end|>": 248046}))

    def test_a_tokenizer_mapping_a_token_to_unk_is_refused(self) -> None:
        vocab = {"<|im_end|>": 248046, "<|endoftext|>": 0}
        with pytest.raises(ValueError, match="does not know"):
            model_backend.end_of_turn_token_ids(_vocab_tokenizer(vocab, unk_token_id=0))

    def test_vllm_passes_the_pin_as_stop_token_ids(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_loaders(monkeypatch)
        _patch_fake_vllm(monkeypatch)
        backend = model_backend.VLLMBackend(
            "Qwen/Qwen3.5-4B", sampling=_sampling_base(), stop_token_ids=(248044, 248046)
        )
        params = cast("_RecordingSamplingParams", backend._sampling_params)  # pyright: ignore[reportPrivateUsage]  # checks internals
        assert params.kwargs["stop_token_ids"] == [248044, 248046]
        assert backend.stop_token_ids == (248044, 248046)

    def test_vllm_without_a_pin_leaves_the_engine_its_own_merge(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_loaders(monkeypatch)
        _patch_fake_vllm(monkeypatch)
        backend = model_backend.VLLMBackend("Qwen/Qwen3.5-4B", sampling=_sampling_base())
        params = cast("_RecordingSamplingParams", backend._sampling_params)  # pyright: ignore[reportPrivateUsage]  # checks internals
        assert params.kwargs["stop_token_ids"] is None
        assert backend.stop_token_ids == ()

    def test_hf_passes_the_pin_as_generate_eos_and_counts_it_as_a_terminator(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_loaders(monkeypatch)
        backend = HFBackend("Qwen/Qwen3.5-4B", stop_token_ids=(248044, 248046))
        kwargs = backend._generation_kwargs()  # pyright: ignore[reportPrivateUsage]  # checks internals
        assert kwargs["eos_token_id"] == [248044, 248046]
        assert {248044, 248046} <= backend._terminator_ids  # pyright: ignore[reportPrivateUsage]  # checks internals
        unpinned = HFBackend("Qwen/Qwen3.5-4B")
        assert "eos_token_id" not in unpinned._generation_kwargs()  # pyright: ignore[reportPrivateUsage]  # checks internals


# --- the incremental engine drain: generate_streaming --------------------------------------------------


class _ChatTemplateTokenizer:
    """The one tokenizer call both vLLM generate paths make: the chat template over one user turn."""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        del tokenize, add_generation_prompt, enable_thinking
        return f"<chat>{messages[0]['content']}"


def _scripted_reply(chat: str) -> str:
    return f"reply to {chat}"


class _DrainableEngine:
    """An ``LLMEngine`` stand-in: requests finish a few per step, LAST enqueued first, under the caller's ids.

    Completion order deliberately differs from submission order so the drain's pairing is load-bearing.
    ``add_request`` returns a randomised internal id the way vLLM 0.27.1 does, so a drain that matched on
    the returned id instead of its own would fail here as it did on the L4.
    """

    def __init__(
        self, *, per_step: int = 3, finish_reason: str = "stop", stop_reason: object = None
    ) -> None:
        self.added: list[tuple[str, dict[str, Any], Any, object]] = []
        self.queue: list[tuple[str, dict[str, Any]]] = []
        self.aborted: list[list[str]] = []
        self.per_step = per_step
        self.finish_reason = finish_reason
        self.stop_reason = stop_reason

    def add_request(
        self,
        request_id: str,
        engine_input: dict[str, Any],
        params: Any,
        *,
        lora_request: object = None,
    ) -> str:
        if engine_input.get("type") != "text" or "prompt_token_ids" not in engine_input:
            raise TypeError(f"add_request was handed an unrendered prompt: {engine_input!r}")
        self.added.append((request_id, engine_input, params, lora_request))
        self.queue.append((request_id, engine_input))
        return f"{request_id}-internal"

    def has_unfinished_requests(self) -> bool:
        return bool(self.queue)

    def abort_request(self, request_ids: list[str]) -> None:
        """``LLMEngine.abort_request`` by external id: drops what is still queued, ignores what is not."""
        self.aborted.append(list(request_ids))
        self.queue = [
            (request_id, engine_input)
            for request_id, engine_input in self.queue
            if request_id not in request_ids
        ]

    def step(self) -> list[Any]:
        batch = [self.queue.pop() for _ in range(min(self.per_step, len(self.queue)))]
        return [
            SimpleNamespace(
                request_id=request_id,
                prompt=engine_input["prompt"],
                prompt_token_ids=list(engine_input["prompt_token_ids"]),
                finished=True,
                outputs=[
                    SimpleNamespace(
                        text=_scripted_reply(engine_input["prompt"]),
                        token_ids=[len(engine_input["prompt"]), 7],
                        finish_reason=self.finish_reason,
                        stop_reason=self.stop_reason,
                    )
                ],
            )
            for request_id, engine_input in batch
        ]


class _RenderingLLM:
    """The ``vllm.LLM`` surface both generate paths touch: the engine, its renderer, and ``generate``.

    ``generate`` is ``LLM.generate`` as the installed 0.27.1 implements it (``offline_utils.py``):
    render every prompt, add each under a counter id, drain the engine to empty, sort by id. Modelled
    rather than stubbed so the batch path and the streaming path can be compared on one engine.
    """

    def __init__(self, engine: _DrainableEngine) -> None:
        self.llm_engine = engine
        self.render_calls: list[list[dict[str, str]]] = []
        self.renderer = SimpleNamespace(render_cmpl=self._render_cmpl)
        self.generate_calls: list[tuple[list[str], Any, object]] = []

    def _render_cmpl(self, prompts: list[dict[str, str]]) -> list[dict[str, Any]]:
        self.render_calls.append(prompts)
        return [
            {
                "type": "text",
                "prompt": prompt["prompt"],
                "prompt_token_ids": [ord(char) for char in prompt["prompt"]],
            }
            for prompt in prompts
        ]

    def generate(self, chats: list[str], params: Any, *, lora_request: object = None) -> list[Any]:
        self.generate_calls.append((list(chats), params, lora_request))
        engine = self.llm_engine
        for index, engine_input in enumerate(self._render_cmpl([{"prompt": c} for c in chats])):
            engine.add_request(str(index), engine_input, params, lora_request=lora_request)
        outputs: list[Any] = []
        while engine.has_unfinished_requests():
            outputs.extend(output for output in engine.step() if output.finished)
        return sorted(outputs, key=lambda output: int(output.request_id))


def _streaming_backend(
    monkeypatch: pytest.MonkeyPatch,
    *,
    engine: _DrainableEngine | None = None,
    lora_request: object = None,
) -> tuple[VLLMBackend, _RenderingLLM]:
    """A ``VLLMBackend`` built through its real ``__init__`` over the fakes, then given a drainable engine."""
    _patch_loaders(monkeypatch)
    _patch_fake_vllm(monkeypatch)
    backend = VLLMBackend("Qwen/Qwen3.5-4B", sampling=SamplingConfig(max_new_tokens=64, seed=3))
    llm = _RenderingLLM(engine or _DrainableEngine())
    backend._llm = llm
    backend._tokenizer = cast("Any", _ChatTemplateTokenizer())
    backend._lora_request = lora_request
    return backend, llm


class TestGenerateStreamingDrainsTheEngineUnderItsOwnIds:
    """The incremental twin of ``generate_tokenized``: same requests, same records, per finished sequence."""

    def test_every_prompt_comes_back_once_under_its_own_index_in_completion_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend, llm = _streaming_backend(monkeypatch)
        prompts = [f"p{index}" for index in range(7)]
        drained = list(backend.generate_streaming(prompts))
        indices = [index for index, _ in drained]
        assert sorted(indices) == list(range(7))
        assert indices != list(range(7)), (
            "the fake finishes last-enqueued first; the order must show"
        )
        for index, tokenized in drained:
            chat = f"<chat>{prompts[index]}"
            assert tokenized.completion.text == _scripted_reply(chat)
            assert tokenized.prompt_token_ids == tuple(ord(char) for char in chat)
            assert tokenized.response_token_ids == (len(chat), 7)
            assert tokenized.completion.usage == TokenUsage(input_tokens=len(chat), output_tokens=2)
            assert tokenized.completion.stop_reason == STOP_REASON_END_TURN
        # The ids the backend chose, in submission order, each naming the prompt at that index.
        assert [request_id for request_id, *_ in llm.llm_engine.added] == [
            f"pooled-{index}" for index in range(7)
        ]
        assert [engine_input["prompt"] for _, engine_input, *_ in llm.llm_engine.added] == [
            f"<chat>p{index}" for index in range(7)
        ]
        # One renderer call over every chat, before anything reached add_request.
        assert llm.render_calls == [[{"prompt": f"<chat>p{index}"} for index in range(7)]]

    def test_it_makes_the_same_requests_as_generate_tokenized_and_yields_the_same_records(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same chats, same rendered ids, same LoRA request, same sampler values; the only difference is
        the output kind, set to final-only on a clone so the backend's own params stay untouched."""
        adapter = object()
        backend, llm = _streaming_backend(monkeypatch, lora_request=adapter)
        prompts = ["a", "bb", "ccc", "dddd"]

        streamed = dict(backend.generate_streaming(prompts))
        streamed_requests = list(llm.llm_engine.added)
        llm.llm_engine.added.clear()
        batched = backend.generate_tokenized(prompts)
        batched_requests = list(llm.llm_engine.added)

        assert [streamed[index] for index in range(len(prompts))] == batched
        assert [engine_input for _, engine_input, *_ in streamed_requests] == [
            engine_input for _, engine_input, *_ in batched_requests
        ]
        assert all(lora is adapter for *_, lora in streamed_requests)
        assert all(lora is adapter for *_, lora in batched_requests)
        assert llm.generate_calls == [
            ([f"<chat>{prompt}" for prompt in prompts], backend._sampling_params, adapter)
        ]
        streamed_params = {id(params): params for *_, params, _ in streamed_requests}
        assert len(streamed_params) == 1, "one clone shared by every request of the submission"
        (clone,) = streamed_params.values()
        assert clone is not backend._sampling_params
        assert clone.kwargs == backend._sampling_params.kwargs
        assert clone.output_kind == FINAL_ONLY_OUTPUT_KIND
        assert backend._sampling_params.output_kind == "cumulative"
        assert all(params is backend._sampling_params for *_, params, _ in batched_requests)

    @pytest.mark.parametrize(
        ("finish_reason", "matched_stop", "expected"),
        [
            ("length", None, STOP_REASON_MAX_TOKENS),
            ("stop", "</run>", STOP_REASON_STOP_SEQUENCE),
            ("stop", 151645, STOP_REASON_END_TURN),
            ("abort", None, "abort"),
        ],
    )
    def test_finish_reasons_translate_as_they_do_on_the_batch_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        finish_reason: str,
        matched_stop: object,
        expected: str,
    ) -> None:
        engine = _DrainableEngine(finish_reason=finish_reason, stop_reason=matched_stop)
        backend, _ = _streaming_backend(monkeypatch, engine=engine)
        ((_, streamed),) = list(backend.generate_streaming(["p"]))
        (batched,) = backend.generate_tokenized(["p"])
        assert streamed.completion.stop_reason == expected
        assert batched.completion.stop_reason == expected


class TestGenerateStreamingLeavesTheEngineAsItFoundIt:
    """One submission owns the engine: refused unless it is idle, and nothing of ours stays in flight on exit."""

    def test_an_engine_with_requests_in_flight_is_refused_before_anything_is_rendered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _DrainableEngine()
        engine.queue.append(
            ("stale-0", {"type": "text", "prompt": "<chat>left behind", "prompt_token_ids": [1]})
        )
        backend, llm = _streaming_backend(monkeypatch, engine=engine)
        with pytest.raises(RuntimeError, match="already has requests in flight"):
            list(backend.generate_streaming(["a"]))
        assert llm.render_calls == []
        assert engine.added == []
        assert engine.aborted == [], "the stale request is not this submission's to abort"

    def test_a_consumer_that_stops_early_leaves_nothing_of_its_own_in_flight(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend, llm = _streaming_backend(monkeypatch)
        engine = llm.llm_engine
        drained = backend.generate_streaming([f"p{index}" for index in range(7)])
        index, _ = next(drained)
        assert index == 6, "per_step=3 finishes the last three enqueued; the caller has seen one"
        drained.close()
        # Every id not yet handed to the caller is aborted, in submission order: 4 and 5 had
        # finished inside the engine but never reached the consumer, and aborting a finished id is
        # a no-op in vLLM, so naming them costs nothing and omitting an unfinished one would leak it.
        assert engine.aborted == [[f"pooled-{index}" for index in range(6)]]
        assert engine.queue == []
        assert not engine.has_unfinished_requests()
        # The engine is as it was found: a fresh submission on the same backend is accepted.
        assert sorted(index for index, _ in backend.generate_streaming(["x", "y"])) == [0, 1]

    def test_a_consumer_that_raises_under_closing_leaves_nothing_in_flight(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The idiom the docstring asks of a consumer that may raise mid-drain and keep using the engine."""
        backend, llm = _streaming_backend(monkeypatch)
        engine = llm.llm_engine

        def consume_and_give_up() -> None:
            with contextlib.closing(backend.generate_streaming(["a", "b", "c", "d"])) as drained:
                for _ in drained:
                    raise ValueError("consumer gave up")

        with pytest.raises(ValueError, match="consumer gave up"):
            consume_and_give_up()
        assert engine.aborted == [["pooled-0", "pooled-1", "pooled-2"]]
        assert engine.queue == []

    def test_an_add_request_that_fails_aborts_what_was_already_enqueued(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """What vLLM's own ``_render_and_add_requests`` does for a failed add, kept on this path too."""
        engine = _DrainableEngine()
        original_add = engine.add_request

        def add_until_the_third(
            request_id: str,
            engine_input: dict[str, Any],
            params: Any,
            *,
            lora_request: object = None,
        ) -> str:
            if len(engine.added) == 2:
                raise RuntimeError("engine refused the request")
            return original_add(request_id, engine_input, params, lora_request=lora_request)

        engine.add_request = add_until_the_third  # type: ignore[method-assign]
        backend, _ = _streaming_backend(monkeypatch, engine=engine)
        with pytest.raises(RuntimeError, match="engine refused the request"):
            list(backend.generate_streaming(["a", "b", "c", "d"]))
        assert engine.aborted == [["pooled-0", "pooled-1"]]
        assert engine.queue == []


class TestGenerateStreamingRefusesWhatItCannotPair:
    """Each sabotage is the exact engine misbehaviour the check exists for, planted and watched to raise."""

    def test_a_reply_under_an_id_the_submission_never_made_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _DrainableEngine()
        original_step = engine.step

        def step_with_a_stranger() -> list[Any]:
            outputs = original_step()
            outputs[0].request_id = "999"
            return outputs

        engine.step = step_with_a_stranger  # type: ignore[method-assign]
        backend, _ = _streaming_backend(monkeypatch, engine=engine)
        with pytest.raises(RuntimeError, match="never made"):
            list(backend.generate_streaming(["a", "b"]))
        # The refusal is also an exit: both of the submission's ids are aborted on the way out
        # (the engine had finished them; vLLM treats an abort of a finished id as a no-op).
        assert engine.aborted == [["pooled-0", "pooled-1"]]

    def test_a_second_finished_reply_under_one_id_is_refused_and_nothing_is_yielded_twice(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The engine emits a finished output for the same request in two consecutive steps.

        The shape a stale submission or a non-final output kind would produce; distinct from a
        stranger id, so it carries its own refusal.
        """
        engine = _DrainableEngine()
        original_step = engine.step
        repeated: list[Any] = []

        def step_repeating_the_first_reply() -> list[Any]:
            outputs = original_step()
            if not repeated:
                repeated.append(outputs[0])
                return outputs
            return [repeated[0], *outputs]

        engine.step = step_repeating_the_first_reply  # type: ignore[method-assign]
        backend, _ = _streaming_backend(monkeypatch, engine=engine)
        drained: list[tuple[int, Any]] = []
        with pytest.raises(RuntimeError, match="already answered"):
            # extend appends as it goes, so what reached the caller before the raise is kept.
            drained.extend(backend.generate_streaming([f"p{index}" for index in range(7)]))
        indices = [index for index, _ in drained]
        assert indices == [6, 5, 4], "the first step's three replies reached the caller once each"
        assert len(set(indices)) == len(indices)
        assert engine.aborted == [[f"pooled-{index}" for index in range(4)]]

    def test_a_reply_echoing_another_prompt_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A broken engine that routed replies to the wrong request: the ids are right, the prompts swapped."""
        engine = _DrainableEngine()
        original_step = engine.step

        def step_with_swapped_prompts() -> list[Any]:
            outputs = original_step()
            outputs[0].prompt, outputs[1].prompt = outputs[1].prompt, outputs[0].prompt
            return outputs

        engine.step = step_with_swapped_prompts  # type: ignore[method-assign]
        backend, _ = _streaming_backend(monkeypatch, engine=engine)
        with pytest.raises(RuntimeError, match="paired request"):
            list(backend.generate_streaming(["a", "b", "c"]))

    def test_an_engine_that_drains_with_replies_missing_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _DrainableEngine()
        original_step = engine.step

        def step_dropping_one() -> list[Any]:
            return original_step()[1:]

        engine.step = step_dropping_one  # type: ignore[method-assign]
        backend, _ = _streaming_backend(monkeypatch, engine=engine)
        with pytest.raises(RuntimeError, match="never came back"):
            list(backend.generate_streaming(["a", "b", "c"]))

    def test_a_renderer_that_drops_an_input_is_refused_before_anything_is_enqueued(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend, llm = _streaming_backend(monkeypatch)
        full_render = llm.renderer.render_cmpl
        llm.renderer.render_cmpl = lambda prompts: full_render(prompts)[:-1]
        with pytest.raises(RuntimeError, match="renderer returned 1 engine inputs for 2 prompts"):
            list(backend.generate_streaming(["a", "b"]))
        assert llm.llm_engine.added == []

    def test_a_reply_carrying_two_sequences_is_refused_on_both_paths(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _DrainableEngine()
        original_step = engine.step

        def step_with_a_twin() -> list[Any]:
            outputs = original_step()
            outputs[0].outputs = outputs[0].outputs * 2
            return outputs

        engine.step = step_with_a_twin  # type: ignore[method-assign]
        backend, _ = _streaming_backend(monkeypatch, engine=engine)
        with pytest.raises(RuntimeError, match="2 sequences"):
            list(backend.generate_streaming(["a"]))
        with pytest.raises(RuntimeError, match="2 sequences"):
            backend.generate_tokenized(["a"])

    def test_a_reply_without_prompt_token_ids_is_refused_on_both_paths(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _DrainableEngine()
        original_step = engine.step

        def step_without_prompt_ids() -> list[Any]:
            outputs = original_step()
            outputs[0].prompt_token_ids = []
            return outputs

        engine.step = step_without_prompt_ids  # type: ignore[method-assign]
        backend, _ = _streaming_backend(monkeypatch, engine=engine)
        with pytest.raises(RuntimeError, match="no prompt_token_ids"):
            list(backend.generate_streaming(["a"]))
        with pytest.raises(RuntimeError, match="no prompt_token_ids"):
            backend.generate_tokenized(["a"])
