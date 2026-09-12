"""Pin the capture driver's guards, and run one whole capture on a tiny model with no GPU.

Offline and CPU-only. The model here is eight dimensions wide with three stand-in decoder blocks,
which is enough because everything this driver can get wrong is about *conventions* rather than about
numbers: how a stimulus is rendered, which tokens the tokenizer adds, which axis the layers land on,
and whether a cell says what it measured.

:class:`TestPrerenderedGuard` is the seam between the corpus writer and the capture. The corpus
`games.interp_stimuli` writes is already chat-templated, with the contrast continuation inside the
model's own `<think>` block, and this driver feeds text verbatim. Both halves of getting that wrong
are silent: raw statements captured as a bare document (the whole measured space moves), or
already-rendered text templated again (one user turn nested inside another). The guard derives the
template's own prefix from the tokenizer rather than matching a hardcoded token, so it keeps applying
when a template changes.

:class:`TestTokenAccounting` covers the three ways a token budget goes wrong quietly -- a tokenizer
that adds a BOS on top of an already-templated string, a prompt over budget that would be truncated
into a different stimulus, and a budget set under the floor this corpus is authored against, where a
matched pair's two sides hold nothing but the stem they share.

:class:`TestCaptureEndToEnd` runs stimuli through `capture_cell`, writes the cell, reads it back and
loads it as a ladder, which is the whole offline path. It also pins the layer axis: the stored tensor
is `[rows, layers, hidden]` in decoder-block order, which is the `post_block` convention the cell
identity records and the thing that must not silently become `output_hidden_states` order.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch import nn

from games import interp_capture
from games.interp_capture import (
    LadderCell,
    assert_contrast_survives_window,
    assert_no_added_special_tokens,
    assert_render_convention,
    build_identity,
    build_natural_prefix_stimuli,
    capture_cell,
    capture_pooled_matrix,
    capture_selected_natural_activations,
    contrast_boundaries,
    natural_prefix_source_for_cell,
    parse_arm_spec,
    parse_natural_prefix_sources,
    parse_steps,
    render_stimuli,
    resolve_arm_cells,
    template_prefix,
    token_counts,
)
from games.interp_cells import (
    BASE_ARM,
    BASE_STEP,
    LAYER_CONVENTION_POST_BLOCK,
    STIMULUS_RENDER_TEMPLATED,
    STIMULUS_RENDER_VERBATIM,
    Stimulus,
    digest_of_strings,
    group_by_set,
    load_ladder,
    read_cell,
    step_dir,
    stimuli_digest,
    write_cell,
)
from games.interp_stimuli import MIN_CAPTURE_SEQ_LEN
from reward_hacking.interp.directions import (
    capture_positionwise_activations,
    last_token_pool,
    mean_pool,
)

N_LAYERS = 3
HIDDEN = 8
VOCAB = 64
STIMULUS_SET = "correlated-vs-independent-counterpart"

TURN_PREFIX = "<|im_start|>user\n"
TURN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>"


class OffsetLayer(nn.Module):
    """A stand-in decoder block: adds its own offset so each layer's output is distinguishable."""

    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = offset

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return the residual plus this block's offset."""
        return hidden + self.offset


class TinyTrunk(nn.Module):
    """The transformer trunk: embed, then run the decoder blocks, counting the passes it makes.

    The count is what makes the pooling change checkable rather than merely plausible: both poolers now
    read one hidden state, so a cell's set costs ONE forward however many poolings it asks for, and a
    regression to a pass per pooling shows up as a number rather than only as a slower run.
    """

    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0
        self.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        # Deterministic weights: two constructions must be identical, or "did the adapter move the
        # activations" cannot be tested at all -- random init makes every cell differ for free.
        with torch.no_grad():
            self.embed_tokens.weight.copy_(
                torch.arange(VOCAB * HIDDEN, dtype=torch.float32).reshape(VOCAB, HIDDEN) / 1000.0
            )
        self.layers = nn.ModuleList([OffsetLayer(float(index + 1)) for index in range(N_LAYERS)])

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs: object
    ) -> SimpleNamespace:
        """Run every block and hand back the last hidden state, as a real trunk does."""
        del attention_mask, kwargs
        self.forward_calls += 1
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


class TinyModel(nn.Module):
    """A composite-shaped causal LM: the trunk at `.model`, and a text sub-config like the real one."""

    def __init__(self) -> None:
        super().__init__()
        self.model = TinyTrunk()
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(num_hidden_layers=N_LAYERS, hidden_size=HIDDEN)
        )

    @property
    def device(self) -> torch.device:
        """Where the capture should place its inputs."""
        return torch.device("cpu")


class Encoding(dict[str, torch.Tensor]):
    """A tokenizer output that stays put under `.to(device)`, since the fakes live on CPU."""

    def to(self, device: object) -> Encoding:
        """Return self: there is nowhere else to go."""
        del device
        return self


class TinyTokenizer:
    """A char-based tokenizer with a chat template, enough for the capture loop and no download."""

    pad_token: str | None = "<pad>"
    eos_token = "<eos>"
    padding_side: str = "right"
    added_special_tokens: int = 0

    def get_vocab(self) -> dict[str, int]:
        """Expose the loaded token mapping used by the capture identity."""
        return {chr(code): code for code in range(VOCAB)}

    def get_added_vocab(self) -> dict[str, int]:
        return {}

    def get_chat_template(self) -> str:
        """A template with no `reasoning_effort` knob, so nothing is pinned for it."""
        return "{% for message in messages %}...{% endfor %}"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        enable_thinking: bool = True,
        **kwargs: object,
    ) -> str:
        """Render one user turn the way this family's template does, opening `<think>`."""
        del tokenize, add_generation_prompt, kwargs
        suffix = TURN_SUFFIX if enable_thinking else TURN_SUFFIX.removesuffix("<think>")
        return f"{TURN_PREFIX}{messages[0]['content']}{suffix}"

    def _ids(self, text: str, *, add_special_tokens: bool) -> list[int]:
        ids = [(ord(char) % (VOCAB - 1)) + 1 for char in text] or [1]
        if add_special_tokens:
            ids = [1] * self.added_special_tokens + ids
        return ids

    def __call__(
        self,
        text: str | list[str],
        *,
        return_tensors: str | None = None,
        padding: bool = False,
        add_special_tokens: bool = True,
    ) -> Encoding | dict[str, Any]:
        """Tokenize one string or a batch, returning tensors only when asked for them."""
        del padding
        if isinstance(text, str):
            return {"input_ids": self._ids(text, add_special_tokens=add_special_tokens)}
        sequences = [self._ids(item, add_special_tokens=add_special_tokens) for item in text]
        if return_tensors is None:
            return {"input_ids": sequences}
        longest = max(len(sequence) for sequence in sequences)
        input_ids = torch.zeros(len(sequences), longest, dtype=torch.long)
        attention_mask = torch.zeros(len(sequences), longest, dtype=torch.long)
        for row, sequence in enumerate(sequences):
            input_ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
            attention_mask[row, : len(sequence)] = 1
        return Encoding(input_ids=input_ids, attention_mask=attention_mask)


CONTINUATION = {"A": "so only the diagonal is reachable.", "B": "so the largest cell is available."}


def rendered(stem: str, side: str, *, enable_thinking: bool = True) -> str:
    """The string the model reads: template, then the teacher-forced reasoning after `<think>`."""
    suffix = TURN_SUFFIX if enable_thinking else TURN_SUFFIX.removesuffix("<think>")
    return f"{TURN_PREFIX}{stem}{suffix}{CONTINUATION[side]}"


def make_stimuli(n_pairs: int = 3) -> list[Stimulus]:
    """Matched pairs as the corpus writer emits them: an untemplated stem plus a reasoning prefix."""
    return [
        Stimulus(
            stimulus_id=f"{STIMULUS_SET}--p{index}--{side}",
            stimulus_set=STIMULUS_SET,
            side=side,
            pair_id=f"{STIMULUS_SET}--p{index}",
            text=f"a change record for pair {index}, shared by both sides",
            assistant_prefix=CONTINUATION[side],
        )
        for index in range(n_pairs)
        for side in ("A", "B")
    ]


def prerendered_stimuli(n_pairs: int = 3) -> list[Stimulus]:
    """The same corpus under the other convention: the whole string already in `text`."""
    return [
        Stimulus(
            stimulus_id=stimulus.stimulus_id,
            stimulus_set=stimulus.stimulus_set,
            side=stimulus.side,
            pair_id=stimulus.pair_id,
            text=rendered(stimulus.text, stimulus.side),
        )
        for stimulus in make_stimuli(n_pairs)
    ]


class TestRenderConvention:
    def test_the_prefix_comes_off_the_template(self) -> None:
        assert template_prefix(cast("Any", TinyTokenizer()), enable_thinking=True) == TURN_PREFIX

    def test_an_untemplated_corpus_renders_to_the_intended_string(self) -> None:
        """The whole point: template the stem, then land the reasoning inside the assistant turn."""
        stimuli = make_stimuli(n_pairs=1)
        result = render_stimuli(
            cast("Any", TinyTokenizer()),
            stimuli,
            convention=STIMULUS_RENDER_TEMPLATED,
            enable_thinking=True,
        )
        assert result[stimuli[0].stimulus_id] == rendered(stimuli[0].text, "A")
        assert result[stimuli[0].stimulus_id].count(TURN_PREFIX) == 1

    def test_a_prerendered_corpus_templated_again_is_refused(self) -> None:
        """The failure measured on this corpus: a second user turn wrapped around the first."""
        with pytest.raises(ValueError, match="already carry this template"):
            render_stimuli(
                cast("Any", TinyTokenizer()),
                prerendered_stimuli(),
                convention=STIMULUS_RENDER_TEMPLATED,
                enable_thinking=True,
            )

    def test_a_verbatim_corpus_passes_through_untouched(self) -> None:
        stimuli = prerendered_stimuli(n_pairs=1)
        result = render_stimuli(
            cast("Any", TinyTokenizer()),
            stimuli,
            convention=STIMULUS_RENDER_VERBATIM,
            enable_thinking=True,
        )
        assert result[stimuli[0].stimulus_id] == stimuli[0].text

    def test_bare_stems_under_verbatim_are_refused(self) -> None:
        """The other direction: a bare document rather than a turn the model is answering."""
        with pytest.raises(ValueError, match="do not begin with this template"):
            render_stimuli(
                cast("Any", TinyTokenizer()),
                make_stimuli(),
                convention=STIMULUS_RENDER_VERBATIM,
                enable_thinking=True,
            )

    def test_a_dropped_assistant_prefix_under_verbatim_is_refused(self) -> None:
        """Verbatim appends nothing, so a carried prefix would be silently discarded."""
        stimuli = [
            Stimulus(
                stimulus_id="one",
                stimulus_set=STIMULUS_SET,
                side="A",
                pair_id="p",
                text=rendered("a stem", "A"),
                assistant_prefix="reasoning that would vanish",
            )
        ]
        with pytest.raises(ValueError, match="carry an assistant_prefix"):
            render_stimuli(
                cast("Any", TinyTokenizer()),
                stimuli,
                convention=STIMULUS_RENDER_VERBATIM,
                enable_thinking=True,
            )

    def test_a_template_that_does_not_open_think_is_refused(self) -> None:
        """An appended reasoning prefix has to land inside `<think>`, or it reads as an answer."""
        with pytest.raises(ValueError, match="does not open <think>"):
            render_stimuli(
                cast("Any", TinyTokenizer()),
                make_stimuli(),
                convention=STIMULUS_RENDER_TEMPLATED,
                enable_thinking=False,
            )

    def test_an_unknown_convention_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown --stimulus-render"):
            assert_render_convention(
                cast("Any", TinyTokenizer()),
                make_stimuli(),
                convention="guess",
                enable_thinking=True,
            )

    def test_the_two_sides_of_a_pair_diverge_only_late(self) -> None:
        """Matched-stem by construction, which is what makes a truncating window dangerous."""
        rendered_texts = render_stimuli(
            cast("Any", TinyTokenizer()),
            make_stimuli(n_pairs=1),
            convention=STIMULUS_RENDER_TEMPLATED,
            enable_thinking=True,
        )
        left, right = (rendered_texts[key] for key in sorted(rendered_texts))
        shared = len(TURN_PREFIX) + len("a change record for pair 0, shared by both sides")
        assert left[:shared] == right[:shared]
        assert left != right


class TestTokenAccounting:
    def test_a_tokenizer_that_adds_specials_is_refused(self) -> None:
        """An extra BOS on already-templated text shifts every recorded position by one."""
        tokenizer = TinyTokenizer()
        tokenizer.added_special_tokens = 1
        with pytest.raises(ValueError, match="adds 1 special tokens"):
            assert_no_added_special_tokens(cast("Any", tokenizer), "some text")

    def test_a_clean_tokenizer_passes(self) -> None:
        assert_no_added_special_tokens(cast("Any", TinyTokenizer()), "some text")

    def test_an_over_budget_prompt_is_refused_rather_than_truncated(self) -> None:
        """Truncating one side of a pair leaves the stem both sides already shared."""
        over_budget = rendered("a stem long enough to overrun the floor. " * 20, "A")
        with pytest.raises(ValueError, match=f"exceed the {MIN_CAPTURE_SEQ_LEN}-token budget"):
            token_counts(
                cast("Any", TinyTokenizer()), [over_budget], max_prompt_tokens=MIN_CAPTURE_SEQ_LEN
            )

    def test_a_budget_under_the_corpus_floor_is_refused(self) -> None:
        """These stimuli fit the window, so only the floor itself can refuse this call."""
        stimuli = prerendered_stimuli()
        with pytest.raises(ValueError, match=f"below the {MIN_CAPTURE_SEQ_LEN}-token floor"):
            token_counts(
                cast("Any", TinyTokenizer()),
                [stimulus.text for stimulus in stimuli],
                max_prompt_tokens=MIN_CAPTURE_SEQ_LEN - 1,
            )

    def test_a_budget_at_the_floor_is_allowed(self) -> None:
        """The floor is the lowest window a capture may run under, not the lowest it may not."""
        stimuli = prerendered_stimuli()
        counts = token_counts(
            cast("Any", TinyTokenizer()),
            [stimulus.text for stimulus in stimuli],
            max_prompt_tokens=MIN_CAPTURE_SEQ_LEN,
        )
        assert counts == [len(stimulus.text) for stimulus in stimuli]

    def test_counts_are_per_stimulus(self) -> None:
        stimuli = prerendered_stimuli()
        counts = token_counts(
            cast("Any", TinyTokenizer()),
            [stimulus.text for stimulus in stimuli],
            max_prompt_tokens=10_000,
        )
        assert counts == [len(stimulus.text) for stimulus in stimuli]


def stash_then_pool_reference(
    model: TinyModel, tokenizer: TinyTokenizer, texts: list[str], *, batch_size: int
) -> dict[str, torch.Tensor]:
    """The arithmetic a pass per pooling produced, computed without the pooling hook at all.

    Capture every position, upcast, pool with the batch's own mask, then stack on the layer axis the
    way `capture_pooled_matrix` stacks -- an independent path, so a hook that changed the arithmetic for
    every pooling at once still shows up as a difference here rather than moving in step with it.
    """
    rows: dict[str, list[torch.Tensor]] = {"mean": [], "last": []}
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(texts[start : start + batch_size], return_tensors="pt", padding=True)
        positionwise = capture_positionwise_activations(
            cast("Any", model), encoded["input_ids"], encoded["attention_mask"]
        )
        for pooling, pool_fn in (("mean", mean_pool), ("last", last_token_pool)):
            per_layer = [
                pool_fn(positionwise[layer], cast("torch.Tensor", encoded["attention_mask"]))
                for layer in sorted(positionwise)
            ]
            rows[pooling].append(torch.stack(per_layer, dim=1))
    return {pooling: torch.cat(blocks, dim=0) for pooling, blocks in rows.items()}


class TestContrastAndNaturalCapture:
    def test_contrast_boundary_and_cropped_window_guard(self) -> None:
        stimuli = make_stimuli(n_pairs=1)
        tokenizer = TinyTokenizer()
        rendered_texts = render_stimuli(
            cast("Any", tokenizer),
            stimuli,
            convention=STIMULUS_RENDER_TEMPLATED,
            enable_thinking=True,
        )
        boundaries = contrast_boundaries(cast("Any", tokenizer), stimuli, rendered_texts)
        assert len(boundaries) == 1
        boundary = boundaries[0]
        assert boundary.first_differing_position < min(boundary.left_length, boundary.right_length)
        assert_contrast_survives_window(
            cast("Any", tokenizer),
            stimuli,
            rendered_texts,
            window_end=boundary.first_differing_position + 1,
        )
        with pytest.raises(ValueError, match="removes the contrast"):
            assert_contrast_survives_window(
                cast("Any", tokenizer),
                stimuli,
                rendered_texts,
                window_end=boundary.first_differing_position,
            )

    def test_natural_capture_selects_positions_in_small_batches(self) -> None:
        base = make_stimuli(n_pairs=2)
        stimuli = [
            Stimulus(
                stimulus_id=stimulus.stimulus_id,
                stimulus_set=stimulus.stimulus_set,
                side=stimulus.side,
                pair_id=stimulus.pair_id,
                text=stimulus.text,
                assistant_prefix=stimulus.assistant_prefix,
                metadata={
                    "selected_positions": [0, 2],
                    "selection_rule": "fixed smoke positions",
                },
            )
            for stimulus in base
        ]
        tokenizer = TinyTokenizer()
        rendered_texts = render_stimuli(
            cast("Any", tokenizer),
            stimuli,
            convention=STIMULUS_RENDER_TEMPLATED,
            enable_thinking=True,
        )
        model = TinyModel()
        selected, selections = capture_selected_natural_activations(
            cast("Any", model),
            cast("Any", tokenizer),
            stimuli,
            rendered_texts,
            layers=[0, 2],
            batch_size=1,
        )
        assert tuple(selection.stimulus_id for selection in selections) == tuple(
            stimulus.stimulus_id for stimulus in stimuli
        )
        assert all(tensor.shape == (2, 2, HIDDEN) for tensor in selected.values())
        assert model.model.forward_calls == len(stimuli)

        with pytest.raises(ValueError, match="over the exact 1-token capture budget"):
            capture_selected_natural_activations(
                cast("Any", TinyModel()),
                cast("Any", tokenizer),
                stimuli[:1],
                rendered_texts,
                layers=[0],
                max_prompt_tokens=1,
            )

    def test_natural_prefix_builder_uses_exact_requests_and_predeclared_positions(self) -> None:
        manifest = {
            "version": 1,
            "rollout_ids_by_state": {
                "base": "base-eval-2026-09-12",
                "final": "final-eval-2026-09-12",
            },
            "request_ids": ["game-behavior::pair-7::sample-0"],
            "scenario_groups": ["natural-scenario-7"],
            "selection_rule": "fixed first, middle, and final pre-action positions",
            "layers": [0, 2],
        }
        records = [
            {
                "request_id": "game-behavior::pair-7::sample-0",
                "rollout_id": "base-eval-2026-09-12",
                "scenario_group": "natural-scenario-7",
                "prompt": "Choose one label after considering the situation.",
                "pre_action_prefix": "The situation has several consequences to weigh.",
            }
        ]
        stimuli = build_natural_prefix_stimuli(
            records,
            cast("Any", TinyTokenizer()),
            manifest,
            natural_state="base",
            convention=STIMULUS_RENDER_TEMPLATED,
            enable_thinking=True,
        )
        assert len(stimuli) == 1
        assert stimuli[0].metadata["request_id"] == records[0]["request_id"]
        assert stimuli[0].metadata["selected_positions"] == [105, 130, 153]
        assert stimuli[0].metadata["action_commitment_present"] is False
        rendered = render_stimuli(
            cast("Any", TinyTokenizer()),
            stimuli,
            convention=STIMULUS_RENDER_TEMPLATED,
            enable_thinking=True,
        )
        _, selections = capture_selected_natural_activations(
            cast("Any", TinyModel()),
            cast("Any", TinyTokenizer()),
            stimuli,
            rendered,
            layers=[0, 2],
        )
        assert selections[0].request_id == records[0]["request_id"]
        assert selections[0].rollout_id == records[0]["rollout_id"]
        assert selections[0].scenario_group == records[0]["scenario_group"]

    def test_natural_prefix_builder_refuses_unlisted_request(self) -> None:
        manifest = {
            "version": 1,
            "rollout_ids_by_state": {
                "base": "base-eval-2026-09-12",
                "final": "final-eval-2026-09-12",
            },
            "request_ids": ["required"],
            "scenario_groups": ["natural-scenario-7"],
            "selection_rule": "fixed positions",
            "layers": [0],
        }
        with pytest.raises(ValueError, match="absent from retained records"):
            build_natural_prefix_stimuli(
                [],
                cast("Any", TinyTokenizer()),
                manifest,
                natural_state="base",
                convention=STIMULUS_RENDER_TEMPLATED,
                enable_thinking=True,
            )

    def test_natural_prefix_ids_join_base_and_final_rollouts_by_request(self) -> None:
        manifest = {
            "version": 1,
            "rollout_ids_by_state": {
                "base": "base/step-0",
                "final": "final/step-20",
            },
            "request_ids": ["request-1"],
            "scenario_groups": ["eval-group-1"],
            "selection_rule": "fixed first, middle, and final pre-action positions",
            "layers": [0, 2],
        }
        tokenizer = cast("Any", TinyTokenizer())
        base = build_natural_prefix_stimuli(
            [
                {
                    "request_id": "request-1",
                    "rollout_id": "base/step-0",
                    "scenario_group": "eval-group-1",
                    "prompt": "Choose one label.",
                    "pre_action_prefix": "Base considered the consequences before acting.",
                }
            ],
            tokenizer,
            manifest,
            natural_state="base",
            convention=STIMULUS_RENDER_TEMPLATED,
            enable_thinking=True,
        )
        final = build_natural_prefix_stimuli(
            [
                {
                    "request_id": "request-1",
                    "rollout_id": "final/step-20",
                    "scenario_group": "eval-group-1",
                    "prompt": "Choose one label.",
                    "pre_action_prefix": "Final considered the consequences before acting.",
                }
            ],
            tokenizer,
            manifest,
            natural_state="final",
            convention=STIMULUS_RENDER_TEMPLATED,
            enable_thinking=True,
        )
        assert base[0].stimulus_id == final[0].stimulus_id == "natural-prefix--request-1"
        assert base[0].metadata["rollout_id"] != final[0].metadata["rollout_id"]
        assert base[0].metadata["request_id"] == final[0].metadata["request_id"]
        with pytest.raises(ValueError, match="state 'base' requires"):
            build_natural_prefix_stimuli(
                [
                    {
                        "request_id": "request-1",
                        "rollout_id": "final/step-20",
                        "scenario_group": "eval-group-1",
                        "prompt": "Choose one label.",
                        "pre_action_prefix": "Final considered the consequences before acting.",
                    }
                ],
                tokenizer,
                manifest,
                natural_state="base",
                convention=STIMULUS_RENDER_TEMPLATED,
                enable_thinking=True,
            )

    def test_capture_source_state_binding_rejects_swapped_state_before_forward(self) -> None:
        manifest = {"rollout_ids_by_state": {"base": "base/step-0", "final": "final/step-20"}}
        swapped = Stimulus(
            stimulus_id="natural-prefix--request-1",
            stimulus_set="natural-prefix",
            side="N",
            pair_id="request-1",
            text="prompt",
            metadata={"natural_state": "final", "rollout_id": "final/step-20"},
        )
        with pytest.raises(ValueError, match="expected state='base'"):
            interp_capture._validate_natural_stimuli_state([swapped], manifest, "base")

    def test_natural_prefix_builder_filters_non_behavior_trace_sections(self) -> None:
        manifest = {
            "version": 1,
            "rollout_ids_by_state": {
                "base": "trace-base-1",
                "final": "trace-final-1",
            },
            "request_ids": ["request-1"],
            "scenario_groups": ["natural-group-1"],
            "selection_rule": "fixed first, middle, and final pre-action positions",
            "layers": [0, 2],
        }
        records = [
            {"record": "forecast", "prompt": "unrelated forecast"},
            {
                "record": "game-behavior",
                "request_id": "request-1",
                "rollout_id": "trace-base-1",
                "scenario_group": "natural-group-1",
                "prompt": "Choose one label.",
                "pre_action_prefix": "The consequences should be compared before acting.",
            },
        ]
        stimuli = build_natural_prefix_stimuli(
            records,
            cast("Any", TinyTokenizer()),
            manifest,
            natural_state="base",
            convention=STIMULUS_RENDER_TEMPLATED,
            enable_thinking=True,
        )
        assert [stimulus.pair_id for stimulus in stimuli] == ["request-1"]
        malformed = [
            {"record": "forecast", "prompt": "unrelated forecast"},
            {key: value for key, value in records[1].items() if key != "pre_action_prefix"},
        ]
        with pytest.raises(ValueError, match="pre_action_prefix"):
            build_natural_prefix_stimuli(
                malformed,
                cast("Any", TinyTokenizer()),
                manifest,
                natural_state="base",
                convention=STIMULUS_RENDER_TEMPLATED,
                enable_thinking=True,
            )

    def test_natural_prefix_sources_require_per_state_bindings_for_a_ladder(self) -> None:
        base_source = Path("base.jsonl")
        final_source = Path("final.jsonl")
        checkpoint = Path("checkpoint-20")
        sources = parse_natural_prefix_sources([f"base={base_source}", f"final={final_source}"])
        assert sources == {"base": base_source, "final": final_source}
        base = LadderCell(arm="base", step=0, adapter_dir=None)
        final = LadderCell(arm="trained", step=20, adapter_dir=checkpoint)
        assert natural_prefix_source_for_cell(base, sources, n_non_base_cells=1) == base_source
        assert natural_prefix_source_for_cell(final, sources, n_non_base_cells=1) == final_source
        with pytest.raises(ValueError, match="multiple natural-prefix sources"):
            parse_natural_prefix_sources([str(base_source), str(final_source)])

    def test_natural_prefix_builder_cli_writes_private_jsonl_and_identity_sidecar(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        records_path = tmp_path / "retained.jsonl"
        records_path.write_text(
            json.dumps(
                {
                    "request_id": "request-1",
                    "rollout_id": "trace-base-1",
                    "scenario_group": "natural-group-1",
                    "prompt": "Choose one label.",
                    "pre_action_prefix": "The consequences should be compared before acting.",
                }
            )
            + "\n"
        )
        selection_path = tmp_path / "selection.json"
        selection_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "rollout_ids_by_state": {
                        "base": "trace-base-1",
                        "final": "trace-final-1",
                    },
                    "request_ids": ["request-1"],
                    "scenario_groups": ["natural-group-1"],
                    "selection_rule": "fixed first, middle, and final pre-action positions",
                    "layers": [0, 2],
                }
            )
        )
        monkeypatch.setattr(interp_capture, "PRIVATE_SELECTION_MANIFEST_ROOTS", (tmp_path,))
        monkeypatch.setattr(
            interp_capture,
            "AutoTokenizer",
            SimpleNamespace(from_pretrained=lambda *args, **kwargs: TinyTokenizer()),
        )
        output_path = tmp_path / "natural.jsonl"
        assert (
            interp_capture.main(
                [
                    "--build-natural-prefix-stimuli",
                    "--records",
                    str(records_path),
                    "--natural-selection-manifest",
                    str(selection_path),
                    "--natural-prefix-out",
                    str(output_path),
                    "--natural-state",
                    "base",
                    "--base-model",
                    "tiny/base",
                ]
            )
            == 0
        )
        assert output_path.is_file()
        sidecar = output_path.with_suffix(".jsonl.manifest.json")
        assert sidecar.is_file()
        assert json.loads(sidecar.read_text())["stimulus_ids"] == ["natural-prefix--request-1"]


class TestPooledMatrixPoolings:
    """One forward per set, every pooling off it, and the same bits a pass per pooling produced.

    Hot-path backlog rank 31, the games half. The driver used to loop poolings OUTSIDE the capture, so
    a two-pooling cell paid two passes over the corpus (10-20 s per cell at 2B, about 15 s per pooling
    at 9B) on the argument that a both-at-once hook would be a second capture path to keep correct.
    It is now the only capture path, and what has to hold is that the numbers did not move.

    Sabotage-verified: returning one pooler's matrix under every pooling key in
    `capture_pooled_matrix` leaves the shapes and the forward count right and turns the `last`
    comparison below red, which is the failure a shape-only assertion would have missed.
    """

    def test_every_pooling_matches_the_stash_then_pool_reference_bitwise(self) -> None:
        model, tokenizer = TinyModel(), TinyTokenizer()
        texts = [stimulus.text for stimulus in prerendered_stimuli(n_pairs=2)]

        together = capture_pooled_matrix(
            cast("Any", model),
            cast("Any", tokenizer),
            texts,
            poolings=["mean", "last"],
            batch_size=2,
        )
        reference = stash_then_pool_reference(model, tokenizer, texts, batch_size=2)

        assert sorted(together) == ["last", "mean"]
        for pooling in ("mean", "last"):
            assert together[pooling].shape == (len(texts), N_LAYERS, HIDDEN)
            assert torch.equal(together[pooling], reference[pooling]), pooling

    def test_two_poolings_cost_one_pass_over_the_corpus(self) -> None:
        model, tokenizer = TinyModel(), TinyTokenizer()
        texts = [stimulus.text for stimulus in prerendered_stimuli(n_pairs=2)]

        capture_pooled_matrix(
            cast("Any", model),
            cast("Any", tokenizer),
            texts,
            poolings=["mean", "last"],
            batch_size=2,
        )

        # Four texts at batch 2 is two batches, whatever the pooling count; it was four before.
        assert model.model.forward_calls == 2

    def test_one_pooling_reads_what_that_pooling_read_before(self) -> None:
        """The single-pooling call is the multi call over one name, not a second code path."""
        model, tokenizer = TinyModel(), TinyTokenizer()
        texts = [stimulus.text for stimulus in prerendered_stimuli(n_pairs=1)]

        alone = capture_pooled_matrix(
            cast("Any", model), cast("Any", tokenizer), texts, poolings=["mean"], batch_size=1
        )
        together = capture_pooled_matrix(
            cast("Any", model),
            cast("Any", tokenizer),
            texts,
            poolings=["mean", "last"],
            batch_size=1,
        )

        assert torch.equal(alone["mean"], together["mean"])

    def test_an_unknown_pooling_is_refused(self) -> None:
        model, tokenizer = TinyModel(), TinyTokenizer()
        with pytest.raises(ValueError, match="unknown poolings"):
            capture_pooled_matrix(
                cast("Any", model),
                cast("Any", tokenizer),
                ["some text"],
                poolings=["median"],
                batch_size=1,
            )


class TestArmResolution:
    def test_an_arm_spec_splits_into_name_and_run_dir(self) -> None:
        name, run_dir = parse_arm_spec("twin-pd-group=/tmp/runs/group")
        assert name == "twin-pd-group"
        assert run_dir.name == "group"

    @pytest.mark.parametrize("spec", ["no-equals", "=/tmp/runs", "name="])
    def test_a_malformed_arm_spec_is_refused(self, spec: str) -> None:
        with pytest.raises(ValueError, match="--arm expects"):
            parse_arm_spec(spec)

    def test_steps_parse_and_default_to_none(self) -> None:
        assert parse_steps("10,20, 30") == [10, 20, 30]
        assert parse_steps(None) is None

    def test_checkpoints_come_back_in_step_order(self, tmp_path: Path) -> None:
        """Numeric order, so checkpoint-100 cannot precede checkpoint-20 and invert the trend."""
        for step in (20, 100, 10):
            (tmp_path / f"checkpoint-{step}").mkdir(parents=True)
        cells = resolve_arm_cells("arm", tmp_path, None)
        assert [cell.step for cell in cells] == [10, 20, 100]

    def test_a_missing_step_is_fatal(self, tmp_path: Path) -> None:
        """A ladder quietly short one checkpoint reads later as a flat stretch of trend."""
        (tmp_path / "checkpoint-10").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match=r"no checkpoints \[70\]"):
            resolve_arm_cells("arm", tmp_path, [10, 70])


class TestCaptureEndToEnd:
    def make_cell(self, tmp_path: Path, *, arm: str, step: int, offset: float) -> Path:
        """Capture one cell through the real capture path, with the model shifted by `offset`."""
        stimuli = make_stimuli()
        model = TinyModel()
        with torch.no_grad():
            model.model.embed_tokens.weight.add_(offset)
        tokenizer = TinyTokenizer()
        rendered_texts = render_stimuli(
            cast("Any", tokenizer),
            stimuli,
            convention=STIMULUS_RENDER_TEMPLATED,
            enable_thinking=True,
        )
        counts = token_counts(
            cast("Any", tokenizer),
            [rendered_texts[stimulus.stimulus_id] for stimulus in stimuli],
            max_prompt_tokens=10_000,
        )
        identity = build_identity(
            cast("Any", model),
            base_model="tiny/base",
            digest=stimuli_digest(stimuli),
            rendered_digest=digest_of_strings(
                rendered_texts[stimulus.stimulus_id] for stimulus in stimuli
            ),
            stimulus_render=STIMULUS_RENDER_TEMPLATED,
            batch_size=1,
            compute_dtype="float32",
            store_dtype="float32",
        )
        rows, activations = capture_cell(
            LadderCell(arm=arm, step=step, adapter_dir=None),
            model=cast("Any", model),
            tokenizer=cast("Any", tokenizer),
            grouped=group_by_set(stimuli),
            rendered=rendered_texts,
            counts_by_id=dict(zip([s.stimulus_id for s in stimuli], counts, strict=True)),
            poolings=["mean", "last"],
            batch_size=1,
        )
        return write_cell(
            step_dir(tmp_path, arm, step),
            arm=arm,
            step=step,
            identity=identity,
            rows=rows,
            activations=activations,
            applied_adapter_weights=None if arm == BASE_ARM else 4,
            adapter_weights_sha256=None if arm == BASE_ARM else "abc123",
            provenance={"git_sha": "testing"},
        ).parent

    def test_a_captured_cell_round_trips_and_loads_as_a_ladder(self, tmp_path: Path) -> None:
        stimuli = make_stimuli()
        self.make_cell(tmp_path, arm=BASE_ARM, step=BASE_STEP, offset=0.0)
        self.make_cell(tmp_path, arm="twin-pd-group", step=10, offset=0.25)
        ladder = load_ladder(tmp_path, stimuli_sha256=stimuli_digest(stimuli))

        assert [cell.label for cell in ladder.cells] == ["base/step-0", "twin-pd-group/step-10"]
        assert ladder.identity.layer_convention == LAYER_CONVENTION_POST_BLOCK
        assert ladder.identity.n_layers == N_LAYERS
        assert ladder.identity.hidden_size == HIDDEN
        assert ladder.poolings == ("last", "mean")
        cell = ladder.cell("twin-pd-group", 10)
        assert cell.matrix(STIMULUS_SET, "mean").shape == (len(stimuli), N_LAYERS, HIDDEN)
        assert cell.rows[STIMULUS_SET].stimulus_ids == tuple(s.stimulus_id for s in stimuli)
        assert cell.rows[STIMULUS_SET].sides == tuple(s.side for s in stimuli)

    def test_the_layer_axis_is_decoder_block_order(self, tmp_path: Path) -> None:
        """Each stand-in block adds its own offset, so a permuted layer axis is visible as numbers."""
        cell_dir = self.make_cell(tmp_path, arm=BASE_ARM, step=BASE_STEP, offset=0.0)
        matrix = read_cell(cell_dir).matrix(STIMULUS_SET, "mean")
        # Blocks add 1, 2 and 3, so layer i carries the embedding plus the running sum 1, 3, 6.
        deltas = [
            float((matrix[0, layer + 1, :] - matrix[0, layer, :]).mean())
            for layer in range(N_LAYERS - 1)
        ]
        assert deltas == pytest.approx([2.0, 3.0])
        stimuli = prerendered_stimuli()
        tokenizer = TinyTokenizer()
        ids = cast("list[list[int]]", tokenizer([stimuli[0].text])["input_ids"])[0]
        embedded = TinyModel().model.embed_tokens(torch.tensor(ids)).mean(dim=0)
        assert torch.allclose(matrix[0, 0, :], embedded + 1.0, atol=1e-4)

    def test_last_pooling_reads_the_final_real_token(self, tmp_path: Path) -> None:
        """The position about to be decoded into an action, not a pad."""
        stimuli = prerendered_stimuli(n_pairs=1)
        model = TinyModel()
        tokenizer = TinyTokenizer()
        texts = [stimulus.text for stimulus in stimuli]
        pooled = capture_pooled_matrix(
            cast("Any", model), cast("Any", tokenizer), texts, poolings=["last"], batch_size=1
        )["last"]
        ids = cast("list[list[int]]", tokenizer(texts)["input_ids"])
        embedded = model.model.embed_tokens(torch.tensor([ids[0][-1]]))
        assert torch.allclose(pooled[0, N_LAYERS - 1, :], embedded[0] + 6.0, atol=1e-5)

    def test_an_identical_capture_is_refused_as_an_inert_adapter(self, tmp_path: Path) -> None:
        """The end-to-end version of the guard: same model twice, one claiming an applied adapter."""
        self.make_cell(tmp_path, arm=BASE_ARM, step=BASE_STEP, offset=0.0)
        self.make_cell(tmp_path, arm="twin-pd-group", step=10, offset=0.0)
        with pytest.raises(ValueError, match="bit-identical to the base"):
            load_ladder(tmp_path)

    def test_the_manifest_records_what_it_measured(self, tmp_path: Path) -> None:
        cell_dir = self.make_cell(tmp_path, arm=BASE_ARM, step=BASE_STEP, offset=0.0)
        manifest = cast("dict[str, Any]", json.loads((cell_dir / "manifest.json").read_text()))
        identity = cast("dict[str, Any]", manifest["identity"])
        assert identity["batch_size"] == 1
        assert identity["stimulus_render"] == STIMULUS_RENDER_TEMPLATED
        assert identity["rendered_sha256"] != identity["stimuli_sha256"]
        assert identity["layer_convention"] == LAYER_CONVENTION_POST_BLOCK
        assert manifest["sets"][STIMULUS_SET]["shapes"]["mean"] == [6, N_LAYERS, HIDDEN]


PREFILL_BINDING = {
    "chunk_gated_delta_rule": "fla.ops.gated_delta_rule.chunk_gated_delta_rule",
    "causal_conv1d_fn": "fla.modules.convolution.causal_conv1d_fn",
}
DECODE_BINDING = {
    "recurrent_gated_delta_rule": "fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule",
    "causal_conv1d_update": "fla.modules.convolution.causal_conv1d_update",
}
ADAPTER_CONFIG = '{"peft_type": "LORA", "r": 16, "lora_alpha": 32, "target_modules": ["q_proj"]}'


def write_stimuli_file(path: Path) -> Path:
    """The pre-rendered corpus as a JSONL, in the row shape `load_stimuli` reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "id": stimulus.stimulus_id,
            "set": stimulus.stimulus_set,
            "side": stimulus.side,
            "pair_id": stimulus.pair_id,
            "text": stimulus.text,
        }
        for stimulus in prerendered_stimuli(n_pairs=2)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def write_checkpoint_dir(run_dir: Path, step: int) -> Path:
    """One arm checkpoint: the two files the ladder digests and reads a config identity from."""
    checkpoint = run_dir / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
    (checkpoint / "adapter_config.json").write_text(ADAPTER_CONFIG)
    return checkpoint


class TestLadderManifestKernelProvenance:
    """The ladder manifest may not carry a per-cell kernel binding, because a resume skips cells.

    `main` writes the manifest for the whole run while skipping any cell already on disk, so a
    manifest-level `deltanet_kernel` would name THIS process's binding over cells another process
    captured -- and a pool keyed on it would then read as one binding when it is two. The bridge report
    and the four-kernel binding stay: those are process provenance and true of this process by
    construction. Every cell's own manifest carries its own binding, which is what a pooling read has
    to take.
    """

    @pytest.fixture
    def offline_capture(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stub the card, the weight load and the adapter attach; the capture itself is the real one."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(
            interp_capture,
            "AutoTokenizer",
            SimpleNamespace(from_pretrained=lambda *a, **k: TinyTokenizer()),
        )
        monkeypatch.setattr(interp_capture, "load_adapter_base", lambda *a, **k: TinyModel())
        monkeypatch.setattr(
            interp_capture,
            "bridge_and_check_decode_kernel",
            lambda: {"bridged": True, "reason": "aliased fla's fused per-token kernel"},
        )
        monkeypatch.setattr(
            interp_capture, "bound_deltanet_kernels", lambda: {**PREFILL_BINDING, **DECODE_BINDING}
        )
        monkeypatch.setattr(interp_capture, "git_sha", lambda: "testingsha")
        monkeypatch.setattr(
            interp_capture,
            "attach_adapter",
            lambda model, adapter_dir, base_model_id, existing=None: SimpleNamespace(
                peft_model=model, applied_adapter_weights=4
            ),
        )

    @staticmethod
    def _argv(tmp_path: Path, *extra: str) -> list[str]:
        return [
            "--stimuli",
            str(write_stimuli_file(tmp_path / "stimuli.jsonl")),
            "--out-dir",
            str(tmp_path / "capture"),
            "--base-model",
            "tiny/base",
            "--stimulus-render",
            STIMULUS_RENDER_VERBATIM,
            "--poolings",
            "mean",
            "--batch-size",
            "1",
            "--store-dtype",
            "float32",
            "--compute-dtype",
            "float32",
            *extra,
        ]

    def _manifest(self, tmp_path: Path) -> dict[str, Any]:
        return cast(
            "dict[str, Any]",
            json.loads((tmp_path / "capture" / "ladder-manifest.json").read_text()),
        )

    def test_the_manifest_carries_process_provenance_and_no_per_cell_binding(
        self, tmp_path: Path, offline_capture: None
    ) -> None:
        del offline_capture
        assert interp_capture.main(self._argv(tmp_path)) == 0
        ladder = self._manifest(tmp_path)

        assert "deltanet_kernel" not in ladder, "a ladder-level binding cannot survive a resume"
        assert ladder["deltanet_kernel_bridge"]["bridged"] is True
        assert ladder["deltanet_kernels_bound"] == {**PREFILL_BINDING, **DECODE_BINDING}
        cell = cast(
            "dict[str, Any]",
            json.loads((tmp_path / "capture" / BASE_ARM / "step-0" / "manifest.json").read_text()),
        )
        assert cell["provenance"]["deltanet_kernel"] == PREFILL_BINDING

    def test_a_resumed_run_leaves_the_skipped_cell_to_speak_for_itself(
        self, tmp_path: Path, offline_capture: None
    ) -> None:
        """The case a manifest-level binding would lie about: one cell from each of two processes."""
        del offline_capture
        run_dir = tmp_path / "arm-run"
        write_checkpoint_dir(run_dir, 10)
        assert interp_capture.main(self._argv(tmp_path)) == 0
        assert interp_capture.main(self._argv(tmp_path, "--arm", f"trained={run_dir}")) == 0
        ladder = self._manifest(tmp_path)

        assert list(ladder["cells"]) == ["base/step-0", "trained/step-10"]
        assert ladder["cells"]["base/step-0"] == {"skipped": "already on disk"}
        assert "deltanet_kernel" not in ladder["cells"]["base/step-0"]
        assert "deltanet_kernel" not in ladder
        for arm, step in ((BASE_ARM, 0), ("trained", 10)):
            manifest = cast(
                "dict[str, Any]",
                json.loads(
                    (tmp_path / "capture" / arm / f"step-{step}" / "manifest.json").read_text()
                ),
            )
            assert manifest["provenance"]["deltanet_kernel"] == PREFILL_BINDING, f"{arm}/{step}"
