"""Synthetic unit tests for matched hack-contrast directions; no benchmark prompts live here."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
import torch

from reward_hacking.atlas.sampling import artifact_model_key
from reward_hacking.interp.hack_contrast import (
    LAYER_INDEX_CONVENTION,
    FitItemsConfig,
    PreparedTurn,
    ProjectionCapture,
    _build_parser,
    _emit_pair_shifts,
    _generation_hook,
    _parse_strengths,
    _pool_response_batch,  # pyright: ignore[reportPrivateUsage]
    _steering_sample_seed,
    _token_sweep_captures,
    _tokenize_prompt_record,
    _validate_direction_bundle,
    audit_pair_records,
    diff_of_means_direction,
    direction_intervention,
    exclude_families,
    fit_item_directions,
    leave_one_pair_out_auc,
    load_direction_bundle,
    main,
    make_matched_norm_placebo,
    prepare_turn,
    read_captures,
    read_pairs,
    render_messages,
    save_direction_bundle,
    shuffled_pair_null_auc,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path


class IdentityLayer(torch.nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = SimpleNamespace(layers=torch.nn.ModuleList([IdentityLayer(), IdentityLayer()]))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.model.layers:
            hidden = layer(hidden)
        return hidden


class FakeTokenizer:
    pad_token_id: int | None = 0

    def __init__(self) -> None:
        self.last_messages: list[dict[str, str]] = []

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_dict: bool,
    ) -> dict[str, list[int]]:
        # transformers 5 returns a BatchEncoding-like mapping for tokenize=True; mirror that shape.
        assert tokenize
        assert return_dict
        self.last_messages = [dict(message) for message in messages]
        prompt_ids = [1, 2, 3]
        if messages[-1]["role"] == "assistant":
            input_ids = [
                *prompt_ids,
                4,
                5,
                *(ord(character) for character in messages[-1]["content"]),
            ]
        else:
            input_ids = [*prompt_ids, 4, 5] if add_generation_prompt else prompt_ids
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [ord(character) for character in text]

    def decode(self, token_ids: torch.Tensor, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens
        return "".join(chr(int(token_id)) for token_id in token_ids)


def test_gemma_system_message_is_folded_but_other_families_keep_it() -> None:
    messages = [
        {"role": "system", "content": "SYSTEM_SENTINEL"},
        {"role": "user", "content": "USER_SENTINEL"},
    ]

    gemma_messages = render_messages(messages, "google/gemma-3-12b-it")
    other_messages = render_messages(messages, "ai-safety-institute/somo-olmo-7b-sdf-sft")

    assert gemma_messages == [{"role": "user", "content": "SYSTEM_SENTINEL\n\nUSER_SENTINEL"}]
    assert other_messages == messages


def test_teacher_forcing_locates_assistant_control_tokens_after_family_rendering() -> None:
    messages = [
        {"role": "system", "content": "SYSTEM_SENTINEL"},
        {"role": "user", "content": "USER_SENTINEL"},
    ]
    gemma_tokenizer = FakeTokenizer()

    gemma_turn = prepare_turn(
        gemma_tokenizer,
        messages,
        "xy",
        model_id="google/gemma-3-12b-it",
    )

    assert gemma_tokenizer.last_messages == [
        {"role": "user", "content": "SYSTEM_SENTINEL\n\nUSER_SENTINEL"},
        {"role": "assistant", "content": "xy"},
    ]
    assert gemma_turn.assistant_control_positions == (3, 4)
    assert gemma_turn.response_start == 5
    assert gemma_turn.response_length == 2
    assert gemma_turn.token_ids == [1, 2, 3, 4, 5, ord("x"), ord("y")]


def test_rendered_atlas_prompt_uses_sampler_tokenization_without_retemplating() -> None:
    tokenizer = FakeTokenizer()

    token_ids = _tokenize_prompt_record(
        tokenizer,
        {"prompt": "RENDERED_ATLAS_PROMPT"},
        model_id="google/gemma-3-12b-it",
    )

    assert token_ids == [ord(character) for character in "RENDERED_ATLAS_PROMPT"]
    assert tokenizer.last_messages == []


def _write_synthetic_item_sweep(run_dir: Path, model_key: str) -> None:
    reads_adapter = (
        model_key.removeprefix("base--reads--") if model_key.startswith("base--reads--") else None
    )
    output_key = model_key if reads_adapter is not None else artifact_model_key(model_key)
    indexed_model_key = "base" if reads_adapter is not None else model_key
    samples_from = reads_adapter if reads_adapter is not None else model_key
    token_sweep_dir = run_dir / "token-sweep"
    token_sweep_dir.mkdir(parents=True)
    item_count = 20
    samples_per_item = 2
    layer_ids = [4, 12]
    prompt_tokens = 2
    completion_tokens = 2
    hidden_width = 5
    activations = np.zeros(
        (
            item_count * samples_per_item,
            len(layer_ids),
            prompt_tokens + completion_tokens,
            hidden_width,
        ),
        dtype=np.float32,
    )
    rows: list[dict[str, Any]] = []
    generator = np.random.default_rng(77)
    for item_index in range(item_count):
        hack_item = item_index < item_count // 2
        item_center = 4.0 if hack_item else -4.0
        for sample_index in range(samples_per_item):
            row_index = item_index * samples_per_item + sample_index
            activations[row_index, 0, prompt_tokens - 1, 0] = item_center + generator.normal(0, 10)
            activations[row_index, 0, prompt_tokens - 1, 1:] = generator.normal(0, 1.2, 4)
            activations[row_index, 1, prompt_tokens - 1, 0] = item_center + generator.normal(
                0, 0.05
            )
            activations[row_index, 1, prompt_tokens - 1, 1:] = generator.normal(0, 0.05, 4)
            rows.append(
                {
                    "sample_id": f"sample-{item_index}-{sample_index}",
                    "item_id": f"item-{item_index}",
                    "sample_index": sample_index,
                    "hack_label": hack_item,
                    "pass_label": not hack_item,
                }
            )
    activation_path = token_sweep_dir / f"activations.{output_key}.npy"
    np.save(activation_path, activations, allow_pickle=False)
    index = {
        "model_key": indexed_model_key,
        "samples_from": samples_from,
        "output_key": output_key,
        "model_id": "synthetic/model",
        "adapter_id": "synthetic/adapter",
        "complete": True,
        "identity": {
            "layer_ids": layer_ids,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
        "layer_ids": layer_ids,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "position_layout": {
            "prompt_offsets": [-2, -1],
            "prompt_padding": "left",
            "activation_alignment": "hidden state after the token at that position",
        },
        "shape": list(activations.shape),
        "rows": rows,
    }
    (token_sweep_dir / f"index.{output_key}.json").write_text(json.dumps(index), encoding="utf-8")


def test_fit_items_uses_last_prompt_state_and_stable_item_holdout(tmp_path: Path) -> None:
    run_dir = tmp_path / "sweep"
    _write_synthetic_item_sweep(run_dir, "synthetic-adapter")
    config = FitItemsConfig(
        run_dir=run_dir,
        model_key="synthetic-adapter",
        output_dir=tmp_path / "directions-one",
        top_layers=1,
        null_draws=8,
        seed=19,
    )

    fit_item_directions(config)
    first_directions, first_provenance = load_direction_bundle(config.output_dir)
    second_config = FitItemsConfig(
        run_dir=run_dir,
        model_key="synthetic-adapter",
        output_dir=tmp_path / "directions-two",
        top_layers=1,
        null_draws=8,
        seed=19,
    )
    fit_item_directions(second_config)
    _, second_provenance = load_direction_bundle(second_config.output_dir)

    assert list(first_directions) == ["last_prompt"]
    assert first_directions["last_prompt"].shape == (1, 5)
    assert first_provenance["layers"] == [12]
    layer_metrics = first_provenance["metrics"]["last_prompt"]
    assert [metric["layer"] for metric in layer_metrics] == [4, 12]
    assert layer_metrics[1]["leave_one_item_out_auc"] > 0.99
    assert len(layer_metrics[1]["random_direction_null_auc"]) == 8
    assert len(first_provenance["held_out_item_ids"]) == 6
    assert first_provenance["held_out_item_ids"] == second_provenance["held_out_item_ids"]


def test_fit_items_accepts_base_reading_adapter_sweep_key(tmp_path: Path) -> None:
    run_dir = tmp_path / "sweep"
    model_key = "base--reads--synthetic-adapter"
    _write_synthetic_item_sweep(run_dir, model_key)

    fit_item_directions(
        FitItemsConfig(
            run_dir=run_dir,
            model_key=model_key,
            output_dir=tmp_path / "directions",
            top_layers=1,
            null_draws=2,
        )
    )

    _, provenance = load_direction_bundle(tmp_path / "directions")
    assert provenance["source_model_key"] == model_key
    assert provenance["source_output_key"] == model_key


def test_last_prompt_is_a_steer_position_and_sparse_layers_validate() -> None:
    arguments = _build_parser().parse_args(
        [
            "steer",
            "--model",
            "synthetic/model",
            "--direction-dir",
            "directions",
            "--prompts",
            "prompts.jsonl",
            "--layer",
            "12",
            "--position",
            "last_prompt",
            "--strengths=-8,-4,0,4",
            "--out",
            "steered.jsonl",
        ]
    )
    directions = {"last_prompt": torch.ones((2, 3), dtype=torch.float32)}
    provenance = {
        "schema_version": 1,
        "positions": ["last_prompt"],
        "layers": [4, 12],
        "layer_index_convention": LAYER_INDEX_CONVENTION,
    }

    assert arguments.position == "last_prompt"
    assert _parse_strengths(arguments.strengths) == [-8.0, -4.0, 0.0, 4.0]
    assert _validate_direction_bundle(directions, provenance) == [4, 12]
    sample_seeds = [_steering_sample_seed(19, "synthetic-item", index, 4) for index in range(4)]
    assert len(set(sample_seeds)) == 4


def test_diff_of_means_pair_cv_separates_planted_signal_and_random_features() -> None:
    generator = torch.Generator().manual_seed(7)
    random_features = torch.randn((18, 2, 8), generator=generator)
    labels = torch.tensor([[0, 1]] * 18)
    honest = random_features[:, 0, :]
    hack = random_features[:, 1, :].clone()
    hack[:, 0] += 8.0
    direction = diff_of_means_direction(hack, honest)
    pair_features = torch.stack((honest, hack), dim=1)

    assert direction[0] > 7.0
    assert leave_one_pair_out_auc(pair_features, labels) > 0.99

    null_features = torch.randn((120, 2, 16), generator=generator)
    null_labels = torch.tensor([[0, 1]] * 120)
    auc = leave_one_pair_out_auc(null_features, null_labels)
    assert 0.4 <= auc <= 0.6


def test_shuffled_pair_null_is_reproducible_and_near_chance() -> None:
    generator = torch.Generator().manual_seed(11)
    features = torch.randn((80, 2, 12), generator=generator)
    labels = torch.tensor([[0, 1]] * 80)

    null = shuffled_pair_null_auc(features, labels, permutations=12, seed=19)

    assert null == shuffled_pair_null_auc(features, labels, permutations=12, seed=19)
    assert all(0.35 <= value <= 0.65 for value in null)


def test_family_exclusion_removes_only_requested_families() -> None:
    pairs = [{"pair_id": str(family), "family": family} for family in range(1, 9)]

    retained = exclude_families(pairs, {1, 2})

    assert [pair["family"] for pair in retained] == [3, 4, 5, 6, 7, 8]
    with pytest.raises(ValueError, match="no pairs remain"):
        exclude_families(pairs, set(range(1, 9)))


def test_pair_reader_validates_the_family_holdout_metadata(tmp_path: Path) -> None:
    path = tmp_path / "synthetic-pairs.jsonl"
    records = [
        {
            "pair_id": f"synthetic-{family}",
            "family": family,
            "messages": [
                {"role": "system", "content": "SYSTEM_SENTINEL"},
                {"role": "user", "content": "USER_SENTINEL"},
            ],
            "honest": "HONEST_SENTINEL",
            "hack": "HACK_SENTINEL",
            "held_out_splits": {
                "without_family_1": "test" if family == 1 else "train",
                "without_family_2": "test" if family == 2 else "train",
            },
        }
        for family in (1, 2)
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    pairs = read_pairs(path)

    assert [pair["family"] for pair in pairs] == [1, 2]


def _audit_pair(pair_index: int, family: int, honest: str, hack: str) -> dict[str, Any]:
    return {
        "pair_id": f"synthetic-{pair_index}",
        "family": family,
        "messages": [
            {"role": "system", "content": "Synthetic system instruction."},
            {"role": "user", "content": f"Synthetic request {pair_index}."},
        ],
        "honest": honest,
        "hack": hack,
        "held_out_splits": {
            "without_family_1": "test" if family == 1 else "train",
            "without_family_2": "test" if family == 2 else "train",
        },
    }


def _synthetic_audit_pairs(*, templated: bool) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for pair_index, marker in enumerate(("amber", "birch", "coral", "dune"), start=1):
        shared = f"{marker} record"
        if templated:
            honest = f"I will calculate each {shared} and retain its value."
            hack = f"I will publish each {shared} and retain its value."
        else:
            honest = f"I will inspect each {shared} and retain its value."
            hack = f"I will inspect each {shared} and retain its value."
        pairs.append(_audit_pair(pair_index, pair_index, honest, hack))
    return pairs


def _fenced(prose_before: str, code: str, prose_after: str) -> str:
    return f"{prose_before}\n\n```python\n{code}\n```\n\n{prose_after}"


SYNTHETIC_HONEST_CODE = (
    "def scale(values, factor):\n"
    "    if factor <= 0:\n"
    "        raise ValueError(factor)\n"
    "    return [round(value * factor, 3) for value in values]"
)
SYNTHETIC_SPECIAL_CASE_CODE = (
    "def scale(values, factor):\n"
    "    if values == [1, 2] and factor == 3:\n"
    "        return [3, 6]\n"
    "    return values"
)


def _code_contrast_pairs() -> list[dict[str, Any]]:
    openings = ("I read the module first.", "Starting from the failing case.", "Small change.")
    closings = ("Ran the suite: 4 passed.", "Tests: 4 passed.", "pytest reports 4 passed.")
    return [
        _audit_pair(
            pair_index,
            1,
            _fenced(opening, SYNTHETIC_HONEST_CODE.replace("3", str(pair_index)), closing),
            _fenced(opening, SYNTHETIC_SPECIAL_CASE_CODE.replace("3", str(pair_index)), closing),
        )
        for pair_index, (opening, closing) in enumerate(
            zip(openings, closings, strict=True), start=1
        )
    ]


class TestPairAudit:
    def test_lexical_audit_distinguishes_balanced_from_templated_prose(self) -> None:
        balanced_report = audit_pair_records(_synthetic_audit_pairs(templated=False))
        templated_report = audit_pair_records(_synthetic_audit_pairs(templated=True))

        assert balanced_report.distinct_honest_texts == 4
        assert balanced_report.distinct_hack_texts == 4
        assert balanced_report.prose_flagged_terms == ()
        assert balanced_report.prose_bow_auc == pytest.approx(0.5)
        assert balanced_report.word_count_ratio.minimum == pytest.approx(1.0)
        assert balanced_report.word_count_ratio.maximum == pytest.approx(1.0)
        assert "calculate" in templated_report.prose_flagged_terms
        assert "publish" in templated_report.prose_flagged_terms
        assert templated_report.prose_bow_auc is not None
        assert templated_report.prose_bow_auc > 0.8

    def test_prose_is_audited_without_the_code_blocks(self) -> None:
        report = audit_pair_records(_code_contrast_pairs())

        assert report.prose_flagged_terms == ()
        assert report.prose_bow_auc == pytest.approx(0.5)
        assert report.code_bow_auc is not None
        assert report.code_bow_auc > 0.8
        assert report.code_token_ratio is not None
        assert report.code_token_ratio.maximum < 1.0
        assert report.code_substance_flags == ()

    def test_responses_without_fenced_code_have_no_code_auc_and_fail_the_substance_check(
        self,
    ) -> None:
        report = audit_pair_records(_synthetic_audit_pairs(templated=True))

        assert report.code_bow_auc is None
        assert report.code_token_ratio is None
        assert [flag.differing_code_tokens for flag in report.code_substance_flags] == [0, 0, 0, 0]

    def test_code_substance_check_flags_a_pair_that_differs_by_one_literal(self) -> None:
        pairs = _code_contrast_pairs()
        pairs[1]["hack"] = pairs[1]["honest"].replace("<= 0", "<= 1")

        report = audit_pair_records(pairs)

        assert [
            (flag.pair_id, flag.differing_code_tokens) for flag in report.code_substance_flags
        ] == [("synthetic-2", 2)]
        assert report.min_code_token_difference == 15

    def test_mechanism_leak_check_flags_honest_responses_holding_their_family_marker(
        self,
    ) -> None:
        plugin_code = "def pytest_runtest_makereport(item, call):\n    return None"
        catch_all_code = (
            "def scale(values, factor):\n"
            "    try:\n"
            "        return [value * factor for value in values]\n"
            "    except Exception:\n"
            "        return []"
        )
        pairs = [
            _audit_pair(
                1,
                2,
                _fenced("Done.", plugin_code, "4 passed."),
                _fenced("Done.", plugin_code, "4 passed."),
            ),
            _audit_pair(
                2,
                8,
                _fenced("Done.", SYNTHETIC_HONEST_CODE, "4 passed."),
                _fenced("Done.", catch_all_code, "4 passed."),
            ),
            _audit_pair(
                3,
                1,
                _fenced("Done.", catch_all_code, "4 passed."),
                _fenced("Done.", SYNTHETIC_SPECIAL_CASE_CODE, "4 passed."),
            ),
        ]

        report = audit_pair_records(pairs)

        assert [(leak.pair_id, leak.family) for leak in report.mechanism_leaks] == [
            ("synthetic-1", 2)
        ]
        assert "pytest_" in report.mechanism_leaks[0].marker
        coverage = {row.family: row for row in report.mechanism_coverage}
        assert (coverage[2].honest_with_marker, coverage[2].hack_with_marker) == (1, 1)
        assert (coverage[8].honest_with_marker, coverage[8].hack_with_marker) == (0, 1)
        assert (coverage[1].honest_with_marker, coverage[1].hack_with_marker) == (0, 1)

    def test_evaluative_prose_words_are_counted_per_side(self) -> None:
        pairs = _code_contrast_pairs()
        pairs[0]["honest"] = pairs[0]["honest"].replace("I read", "A robust fix: I read")

        report = audit_pair_records(pairs)

        assert report.prose_evaluative_words == (("robust", 1, 0),)


def test_audit_pairs_cli_emits_a_json_report_for_synthetic_pairs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "synthetic-pairs.jsonl"
    path.write_text(
        "".join(json.dumps(pair) + "\n" for pair in _code_contrast_pairs()),
        encoding="utf-8",
    )

    with caplog.at_level(logging.INFO):
        main(["audit-pairs", "--pairs", str(path)])

    assert '"distinct_honest_texts": 3' in caplog.text
    assert '"prose_bow_auc": 0.5' in caplog.text
    assert "PAIR_AUDIT_SUMMARY" in caplog.text


def test_placebo_direction_has_exactly_the_reference_norm() -> None:
    direction = torch.tensor([3.0, -4.0, 12.0])

    placebo = make_matched_norm_placebo(direction, seed=23)

    assert torch.linalg.vector_norm(placebo).item() == pytest.approx(
        torch.linalg.vector_norm(direction).item(), rel=1e-6
    )
    assert not torch.allclose(placebo / placebo.norm(), direction / direction.norm())


def test_intervention_adds_the_requested_unit_step_at_only_the_selected_layer() -> None:
    model = TinyModel()
    seen: list[torch.Tensor] = []
    handles = [
        layer.register_forward_hook(lambda _module, _inputs, output: seen.append(output.clone()))
        for layer in model.model.layers
    ]
    initial = torch.zeros((1, 3, 4))
    direction = torch.tensor([0.0, 6.0, 8.0, 0.0])
    magnitude = 2.5

    try:
        with direction_intervention(model, layer_index=1, direction=direction, magnitude=magnitude):
            result = model(initial)
    finally:
        for handle in handles:
            handle.remove()

    expected_step = torch.tensor([0.0, 1.5, 2.0, 0.0]).expand_as(initial)
    assert torch.equal(seen[0], initial)
    assert torch.equal(seen[1], seen[0])
    assert torch.allclose(result - seen[1], expected_step)


def test_generation_hook_starts_at_the_first_output_token_and_respects_scope() -> None:
    direction = torch.tensor([0.0, 5.0])
    prompt_state = torch.zeros((1, 3, 2))
    decode_state = torch.zeros((1, 1, 2))

    all_scope = _generation_hook(direction, magnitude=2.0, prompt_length=3, scope="all")
    first_prefill = cast("torch.Tensor", all_scope(None, None, prompt_state))
    first_decode = cast("torch.Tensor", all_scope(None, None, decode_state))
    first_only = _generation_hook(direction, magnitude=2.0, prompt_length=3, scope="first")
    first_only_prefill = cast("torch.Tensor", first_only(None, None, prompt_state))
    first_only_decode = cast("torch.Tensor", first_only(None, None, decode_state))

    expected_prefill = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [0.0, 2.0]]])
    assert torch.equal(first_prefill - prompt_state, expected_prefill)
    assert torch.equal(first_decode - decode_state, torch.tensor([[[0.0, 2.0]]]))
    assert torch.equal(first_only_prefill, first_prefill)
    assert torch.equal(first_only_decode, decode_state)


def test_direction_bundle_round_trips_arrays_and_provenance(tmp_path: Path) -> None:
    directions = {
        "response_mean": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "response_last": torch.ones((3, 4), dtype=torch.float32),
    }
    provenance = {
        "model": "synthetic-model",
        "excluded_families": [1],
        "layer_index_convention": "zero-based decoder-block outputs",
    }

    save_direction_bundle(tmp_path, directions, provenance)
    loaded_directions, loaded_provenance = load_direction_bundle(tmp_path)

    assert loaded_provenance == provenance
    assert loaded_directions.keys() == directions.keys()
    for position, direction in directions.items():
        assert torch.equal(loaded_directions[position], direction)


def test_atlas_read_aligns_base_and_adapter_by_item_and_sample(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    directions = {
        "assistant_control_mean": torch.ones((2, 3)),
        "response_mean": torch.ones((2, 3)),
        "response_last": torch.ones((2, 3)),
    }
    provenance = {
        "schema_version": 1,
        "model": "synthetic-model",
        "positions": ["assistant_control_mean", "response_mean", "response_last"],
        "layers": [0, 1],
        "layer_index_convention": LAYER_INDEX_CONVENTION,
    }
    run_dir = tmp_path

    def write_model(
        model_key: str,
        adapter_id: str | None,
        rows: list[tuple[str, str, int, bool, bool, float]],
    ) -> None:
        output_key = artifact_model_key(model_key)
        activation = np.zeros((len(rows), 2, 4, 3), dtype=np.float32)
        for row_index, row in enumerate(rows):
            activation[row_index, :, :, :] = row[5]
        np.save(run_dir / f"activations.{output_key}.npy", activation)
        (run_dir / f"capture-index.{output_key}.json").write_text(
            json.dumps(
                {
                    "model_key": model_key,
                    "model_id": "synthetic-model",
                    "adapter_id": adapter_id,
                    "sample_ids": [row[0] for row in rows],
                    "positions": ["last_prompt", "code_mean", "completion_last", "followup_last"],
                    "shape": list(activation.shape),
                    "complete": True,
                }
            ),
            encoding="utf-8",
        )
        (run_dir / f"samples.{output_key}.jsonl").write_text(
            "".join(
                json.dumps({"sample_id": row[0], "item_id": row[1], "sample_index": row[2]}) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
        (run_dir / f"grades.{output_key}.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "sample_id": row[0],
                        "item_id": row[1],
                        "hack": row[3],
                        "honest_pass": row[4],
                    }
                )
                + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )

    write_model(
        "base",
        None,
        [
            ("base-a", "item-a", 0, True, False, 1.0),
            ("base-b", "item-b", 0, False, True, -1.0),
        ],
    )
    write_model(
        "adapter",
        "tiny-adapter",
        [
            ("adapter-b", "item-b", 0, False, True, -1.0),
            ("adapter-a", "item-a", 0, True, False, 2.0),
        ],
    )
    direction_dir = tmp_path / "directions"
    save_direction_bundle(direction_dir, directions, provenance)

    with caplog.at_level(logging.INFO, logger="reward_hacking.interp.hack_contrast"):
        read_captures(run_dir=run_dir, direction_dir=direction_dir)

    shifts = [
        json.loads(record.message) for record in caplog.records if record.message.startswith("{")
    ]
    code_mean_shift = next(
        row
        for row in shifts
        if row["kind"] == "atlas_base_adapter_shift"
        and row["position"] == "code_mean"
        and row["layer"] == 0
    )
    assert code_mean_shift["n_paired_samples"] == 2
    assert code_mean_shift["mean_adapter_minus_base"] == pytest.approx(1.5)


def test_token_sweep_maps_explicit_block_ids_and_reads_near_code_offsets(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "token-sweep"
    sweep_dir.mkdir()
    directions = {
        "assistant_control_mean": torch.ones((2, 3)),
        "response_mean": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        "response_last": torch.ones((2, 3)),
    }
    provenance = {
        "model": "synthetic-model",
        "layers": [0, 1],
        "layer_index_convention": LAYER_INDEX_CONVENTION,
    }
    activation = np.zeros((4, 2, 12, 3), dtype=np.float32)
    for row_index in range(4):
        activation[row_index, 0, 4, 0] = 2.0 if row_index < 2 else -2.0
    np.save(sweep_dir / "activations.base.npy", activation)
    np.save(sweep_dir / "mask.base.npy", np.ones((4, 12), dtype=np.bool_))
    rows = [
        {
            "item_id": f"item-{row_index}",
            "sample_index": 0,
            "hack_label": row_index < 2,
            "pass_label": row_index >= 2,
            "code_token_start": 2,
        }
        for row_index in range(4)
    ]
    (sweep_dir / "index.base.json").write_text(
        json.dumps(
            {
                "model_key": "base",
                "model_id": "synthetic-model",
                "output_key": "base",
                "adapter_id": None,
                "complete": True,
                "prompt_tokens": 2,
                "completion_tokens": 10,
                "layer_ids": [0, 1],
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )

    captures = _token_sweep_captures(sweep_dir, directions, provenance)

    assert len(captures) == 1
    metric = ("code_start+0:response_mean", 0, 0)
    assert captures[0].scores[metric] == {
        ("item-0", 0): 2.0,
        ("item-1", 0): 2.0,
        ("item-2", 0): -2.0,
        ("item-3", 0): -2.0,
    }


def test_base_adapter_shift_uses_paired_intersection_and_reports_missing_rows(
    caplog: pytest.LogCaptureFixture,
) -> None:
    metric = ("code_start+0:response_mean", 0, 0)
    base_only_metric = ("code_start-1:response_mean", 0, -1)
    adapter_only_metric = ("code_start+1:response_mean", 0, 1)
    captures = [
        ProjectionCapture(
            model_key="base",
            adapter_id=None,
            scores={
                metric: {("shared", 0): 1.0, ("base-only", 0): 2.0},
                base_only_metric: {("shared", 0): 4.0},
            },
            labels={("shared", 0): 0, ("base-only", 0): 1},
        ),
        ProjectionCapture(
            model_key="adapter",
            adapter_id="synthetic-adapter",
            scores={
                metric: {("shared", 0): 1.5, ("adapter-only", 0): 3.0},
                adapter_only_metric: {("shared", 0): 5.0},
            },
            labels={("shared", 0): 1, ("adapter-only", 0): 0},
        ),
    ]

    with caplog.at_level(logging.INFO, logger="reward_hacking.interp.hack_contrast"):
        _emit_pair_shifts(captures, kind="token_sweep")

    shift = next(
        json.loads(record.message)
        for record in caplog.records
        if record.message.startswith("{")
        and '"kind": "token_sweep_base_adapter_shift"' in record.message
    )
    assert shift["n_paired_samples"] == 1
    assert shift["n_base_only"] == 1
    assert shift["n_adapter_only"] == 1
    assert shift["mean_adapter_minus_base"] == pytest.approx(0.5)
    coverage = next(
        json.loads(record.message)
        for record in caplog.records
        if record.message.startswith("{")
        and '"kind": "token_sweep_base_adapter_coverage"' in record.message
    )
    assert coverage["n_shared_metrics"] == 1
    assert coverage["n_base_only_metrics"] == 1
    assert coverage["n_adapter_only_metrics"] == 1


def test_pooled_vectors_do_not_keep_the_captured_batch_alive() -> None:
    """Every pooled vector must own its storage; a view would pin the whole batch's activations in RAM."""
    captured = {0: torch.randn(2, 6, 4), 1: torch.randn(2, 6, 4)}
    turn = PreparedTurn(
        token_ids=[1, 2, 3, 4, 5, 6],
        response_start=3,
        response_length=3,
        assistant_control_positions=(2,),
    )
    batch = [("pair-0", "honest", turn), ("pair-0", "hack", turn)]
    pooled: dict[str, dict[str, dict[int, list[torch.Tensor]]]] = {
        position: {"honest": {}, "hack": {}}
        for position in ("response_mean", "response_last", "assistant_control_mean")
    }

    _pool_response_batch(batch, captured, pooled)

    batch_storages = {tensor.untyped_storage().data_ptr() for tensor in captured.values()}
    for by_response in pooled.values():
        for by_layer in by_response.values():
            for vectors in by_layer.values():
                for vector in vectors:
                    assert vector.untyped_storage().data_ptr() not in batch_storages
                    assert (
                        vector.untyped_storage().nbytes() == vector.numel() * vector.element_size()
                    )
