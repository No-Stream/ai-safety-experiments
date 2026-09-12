"""CPU contracts for the cooperation lens fit-gate corpus binding."""

from __future__ import annotations

import json
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

import pytest

from games import cooperation_lens
from games.cooperation_lens import PreparedFitPrompts
from games.interp_capture import render_stimuli as capture_render_stimuli
from games.interp_cells import Stimulus, digest_of_strings, stimuli_digest
from reward_hacking.interp import lens_fit_gate
from reward_hacking.interp.jacobian import derive_max_seq_len

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PreTrainedTokenizerBase


class RenderTokenizer:
    """Small tokenizer with the chat-template seam used by the real capture path."""

    def __init__(self) -> None:
        self.render_calls: list[dict[str, object]] = []
        self.padding_side = "right"
        self.truncation_side = "right"
        self.model_max_length = 256
        self.clean_up_tokenization_spaces = False
        self.split_special_tokens = False
        self.chat_template = "{{ messages }}"
        self.special_tokens_map = {"eos_token": "<eos>"}
        self.init_kwargs = {"legacy": False, "_commit_hash": "test"}

    def get_chat_template(self) -> str:
        return self.chat_template

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
        **kwargs: object,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        self.render_calls.append(
            {"enable_thinking": enable_thinking, "kwargs": kwargs, "messages": messages}
        )
        return f"<|user|>{messages[0]['content']}<|assistant|><think>"

    def __call__(
        self, text: str | list[str], **kwargs: object
    ) -> dict[str, list[list[int]] | list[int]]:
        assert kwargs.get("add_special_tokens") is False
        texts = [text] if isinstance(text, str) else text
        encoded = [[ord(character) for character in value] for value in texts]
        return {"input_ids": encoded[0] if isinstance(text, str) else encoded}

    def get_vocab(self) -> dict[str, int]:
        return {"alpha": 0, "beta": 1}

    def get_added_vocab(self) -> dict[str, int]:
        return {}


def fit_rows() -> list[Stimulus]:
    return [
        Stimulus(
            stimulus_id=f"fit-{side}",
            stimulus_set="costly-other-regard",
            side=side,
            pair_id="fit-pair",
            text=f"shared premise {side}",
            assistant_prefix="I will compare the consequences.",
            metadata={"scenario_group": "fit-group"},
        )
        for side in ("A", "B")
    ]


def write_fit_file(root: Path, rows: list[Stimulus]) -> Path:
    path = root / "docs" / "scratch" / "cooperation-generalization" / "fit.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        "".join(
            json.dumps(
                {
                    "id": row.stimulus_id,
                    "set": row.stimulus_set,
                    "side": row.side,
                    "pair_id": row.pair_id,
                    "text": row.text,
                    "assistant_prefix": row.assistant_prefix,
                    "metadata": row.metadata,
                },
                sort_keys=True,
            )
            + "\n"
            for row in rows
        )
    )
    return path


def test_fit_gate_parser_accepts_private_stimuli_and_render_settings(tmp_path: Path) -> None:
    args = lens_fit_gate._parse_args(
        [
            "--fit-stimuli",
            str(tmp_path / "docs/scratch/cooperation-generalization/fit.jsonl"),
            "--stimulus-render",
            "templated_here",
            "--no-thinking",
            "--gates",
            "tokens",
            "--out",
            str(tmp_path / "gate.json"),
        ]
    )

    assert args.fit_stimuli.name == "fit.jsonl"
    assert args.stimulus_render == "templated_here"
    assert args.no_thinking is True


def test_fit_gate_uses_capture_renderer_and_cooperation_seq_len_derivation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = fit_rows()
    fit_path = write_fit_file(tmp_path, rows)
    monkeypatch.chdir(tmp_path)
    tokenizer = RenderTokenizer()
    calls: list[dict[str, object]] = []

    def spy_render(
        received_tokenizer: object,
        stimuli: list[Stimulus],
        *,
        convention: str,
        enable_thinking: bool,
    ) -> dict[str, str]:
        calls.append(
            {
                "tokenizer": received_tokenizer,
                "stimuli": stimuli,
                "convention": convention,
                "enable_thinking": enable_thinking,
            }
        )
        return capture_render_stimuli(
            cast("Any", received_tokenizer),
            stimuli,
            convention=convention,
            enable_thinking=enable_thinking,
        )

    monkeypatch.setattr(cooperation_lens, "render_stimuli", spy_render)
    plan = cooperation_lens.prepare_fit_prompts(
        fit_path,
        cast("PreTrainedTokenizerBase", tokenizer),
        convention="templated_here",
        enable_thinking=False,
        max_seq_len_ceiling=512,
    )

    assert len(calls) == 1
    assert calls[0]["convention"] == "templated_here"
    assert calls[0]["enable_thinking"] is False
    rendered = capture_render_stimuli(
        cast("PreTrainedTokenizerBase", tokenizer),
        rows,
        convention="templated_here",
        enable_thinking=False,
    )
    expected_lengths = [len(text) for text in rendered.values()]
    expected_seq_len = derive_max_seq_len(expected_lengths, ceiling=512).max_seq_len
    assert plan.max_seq_len == expected_seq_len
    assert plan.stimuli_sha256 == stimuli_digest(rows)
    assert plan.rendered_sha256 == digest_of_strings(rendered[row.stimulus_id] for row in rows)


def gate_payload() -> dict[str, Any]:
    return {
        "passed": True,
        "jlens": {"commit": "581d398abc"},
        "model": {
            "weights_identity": "hf:base-commit",
            "resolved_weights_identity": "hf:base-commit",
            "tokenizer_content_sha256": "tokenizer-content",
        },
        "max_seq_len": 128,
        "cooperation_fit_corpus": {
            "fit_stimuli_sha256": "fit-stimuli",
            "fit_rendered_sha256": "fit-prompt",
            "stimulus_render": "templated_here",
            "enable_thinking": True,
            "tokenizer_content_sha256": "tokenizer-content",
        },
        "deltanet_kernels_bound": {"chunk": "fused"},
        "gates_requested": ["autograd", "kernel", "sweep", "resume"],
        "gates": {
            "autograd": {
                "passed": True,
                "verdicts": {
                    "autograd_traverses_recurrence": True,
                    "sabotage_reads_exactly_zero": True,
                },
            },
            "kernel": {"passed": True},
            "resume": {"passed": True},
            "sweep": {"passed": True, "choice": {"chosen": 8}},
            "tokens": {"passed": True, "reference_identity": "hf:base-commit"},
        },
    }


def bind(payload: dict[str, Any]) -> dict[str, Any]:
    return cooperation_lens.validate_gradient_gate_binding(
        payload,
        base_weights_identity="hf:base-commit",
        tokenizer_content_identity="tokenizer-content",
        comparison_reference_identity="hf:base-commit",
        fit_stimuli_sha256="fit-stimuli",
        fit_rendered_sha256="fit-prompt",
        stimulus_render="templated_here",
        enable_thinking=True,
        max_seq_len=128,
        dim_batch=8,
        kernels={"chunk": "fused"},
    )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("fit_stimuli_sha256", "fit_stimuli_sha256"),
        ("fit_rendered_sha256", "fit_rendered_sha256"),
    ],
)
def test_cooperation_binding_refuses_stale_fit_corpus_or_render_digest(
    field: str, message: str
) -> None:
    payload = gate_payload()
    payload["cooperation_fit_corpus"][field] = "stale"

    with pytest.raises(ValueError, match=message):
        bind(payload)


def test_fit_gate_report_records_fit_and_rendered_prompt_digests() -> None:
    plan = PreparedFitPrompts(
        stimuli=(),
        rendered={},
        token_ids={},
        prompts=(),
        stimuli_sha256="fit-stimuli",
        rendered_sha256="fit-prompt",
        max_seq_len=128,
    )
    report = plan.binding_payload(
        convention="templated_here",
        enable_thinking=True,
        tokenizer_content_identity="tokenizer-content",
    )

    assert report == {
        "fit_stimuli_sha256": "fit-stimuli",
        "fit_rendered_sha256": "fit-prompt",
        "stimulus_render": "templated_here",
        "enable_thinking": True,
        "tokenizer_content_sha256": "tokenizer-content",
        "n_fit_prompts": 0,
    }


def test_run_gates_writes_private_cooperation_fit_binding_without_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = fit_rows()
    fit_path = write_fit_file(tmp_path, rows)
    monkeypatch.chdir(tmp_path)
    tokenizer = RenderTokenizer()
    fake_jlens = ModuleType("jlens")
    fake_jlens.__file__ = str(tmp_path / "jlens" / "__init__.py")
    out_path = tmp_path / "gate.json"

    monkeypatch.setattr(lens_fit_gate, "_require_jlens", lambda: fake_jlens)
    monkeypatch.setattr(
        lens_fit_gate.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: tokenizer),
    )

    def record_token_gate(run: Any, texts: list[str]) -> None:
        assert texts
        run.record(lens_fit_gate.GATE_TOKENS, {"passed": True})

    monkeypatch.setattr(lens_fit_gate, "run_token_gate", record_token_gate)
    args = lens_fit_gate._parse_args(
        [
            "--model-id",
            "test-model",
            "--revision",
            "test-revision",
            "--fit-stimuli",
            str(fit_path),
            "--no-thinking",
            "--gates",
            "tokens",
            "--out",
            str(out_path),
        ]
    )

    assert lens_fit_gate.run_gates(args) == 0
    report = cast("dict[str, Any]", json.loads(out_path.read_text()))
    binding = cast("dict[str, Any]", report["cooperation_fit_corpus"])
    rendered = capture_render_stimuli(
        cast("PreTrainedTokenizerBase", tokenizer),
        rows,
        convention="templated_here",
        enable_thinking=False,
    )
    expected_max_seq_len = derive_max_seq_len(
        [len(rendered[row.stimulus_id]) for row in rows], ceiling=512
    ).max_seq_len

    assert binding["fit_stimuli_sha256"] == stimuli_digest(rows)
    assert binding["fit_rendered_sha256"] == digest_of_strings(
        rendered[row.stimulus_id] for row in rows
    )
    assert binding["stimulus_render"] == "templated_here"
    assert binding["enable_thinking"] is False
    assert report["max_seq_len"] == expected_max_seq_len
