"""Offline tests for the steering module: planted geometry in, known causal answers out.

The causal tier's headline risk is a pipeline that cannot produce a null -- a steering readout
whose placebo arm moves as much as the real arm, or a hook that silently does nothing. So the
tests here plant the violations the module's guards exist to catch and watch each one go red:
a zero-alpha steer must be bit-identical to no intervention, an out-of-range layer must raise
rather than no-op, a matched-norm placebo must NOT move a model that the real direction provably
moves, a placebo-dominated sweep must be flagged as carrying no evidence, and a corpus edited by
one token must be refused before any direction is fitted.

Everything runs CPU-only and offline: fit-direction on planted cells written straight to disk
(the same fixture pattern as the trajectory tests), hook mechanics on a tiny hand-built trunk
that `_decoder_layers` resolves, and the generation driver end-to-end with the backend and the
chunked decoder monkeypatched out.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch

from games import interp_steering
from games.interp_cells import (
    BASE_ARM,
    CellFormatError,
    CellIdentity,
    Stimulus,
    row_index_for,
    step_dir,
    stimuli_digest,
    write_cell,
)
from games.interp_steering import (
    CONDITION_ABLATE_PLACEBO,
    CONDITION_ABLATE_REAL,
    CONDITION_NONE,
    CONDITION_PLACEBO_DOWN,
    CONDITION_PLACEBO_UP,
    CONDITION_STEER_DOWN,
    CONDITION_STEER_UP,
    STEERING_CONDITIONS,
    GenerationCondition,
    condition_key,
    generation_conditions,
    generation_rows,
    hook_for,
    label_first_tokens,
    load_direction_file,
    parse_named_layers,
    parse_named_specs,
    plan_generation_cells,
    run_fit_direction,
    run_generate,
    select_intervention,
    steering_rows,
    summarise_records,
)
from games.prompts import FRAMING_HUMAN, FRAMING_TWIN
from reward_hacking.interp.directions import (
    _decoder_layers,  # pyright: ignore[reportPrivateUsage]
    cosine,
    matched_norm_random_direction,
    unit,
)
from reward_hacking.interp.jsonl_resume import ResumeMismatchError, ledger_path_for
from reward_hacking.interp.steering import residual_intervention, steering_hook

if TYPE_CHECKING:
    from pathlib import Path

STIMULUS_SET = "causal-vs-functional-decision"
N_LAYERS = 2
PLANTED_LAYER = 1
HIDDEN = 128
N_PAIRS = 8
AMPLITUDE = 2.0
NOISE = 0.05


# --------------------------------------------------------------------------------------
# Fixtures: a planted cell on disk, and a tiny trunk the hook machinery can drive
# --------------------------------------------------------------------------------------


def make_stimuli() -> list[Stimulus]:
    return [
        Stimulus(
            stimulus_id=f"{STIMULUS_SET}--p{index}--{side}",
            stimulus_set=STIMULUS_SET,
            side=side,
            pair_id=f"{STIMULUS_SET}--p{index}",
            text=f"rendered pair {index} side {side}",
        )
        for index in range(N_PAIRS)
        for side in ("A", "B")
    ]


def planted_axis() -> torch.Tensor:
    axis = torch.zeros(HIDDEN)
    axis[0] = 1.0
    return axis


def identity_for(stimuli: list[Stimulus]) -> CellIdentity:
    return CellIdentity(
        base_model="tiny/base",
        stimuli_sha256=stimuli_digest(stimuli),
        rendered_sha256=stimuli_digest(stimuli),
        layer_convention="post_block",
        n_layers=N_LAYERS,
        hidden_size=HIDDEN,
        batch_size=1,
        compute_dtype="float32",
        store_dtype="float32",
        stimulus_render="verbatim",
    )


def plant_base_cell(root: Path, stimuli: list[Stimulus]) -> None:
    """One base cell whose planted layer separates the sides along a fixed axis."""
    generator = torch.Generator().manual_seed(7)
    matrix = torch.randn(len(stimuli), N_LAYERS, HIDDEN, generator=generator) * NOISE
    axis = planted_axis()
    for row, stimulus in enumerate(stimuli):
        sign = 1.0 if stimulus.side == "A" else -1.0
        matrix[row, PLANTED_LAYER, :] += sign * AMPLITUDE * axis
    write_cell(
        step_dir(root, BASE_ARM, 0),
        arm=BASE_ARM,
        step=0,
        identity=identity_for(stimuli),
        rows={STIMULUS_SET: row_index_for(stimuli, [42] * len(stimuli))},
        activations={(STIMULUS_SET, "last"): matrix},
        applied_adapter_weights=None,
        adapter_weights_sha256=None,
        provenance={"git_sha": "testing"},
    )


def write_stimuli_jsonl(path: Path, stimuli: list[Stimulus]) -> None:
    lines = [
        json.dumps(
            {
                "id": s.stimulus_id,
                "set": s.stimulus_set,
                "side": s.side,
                "pair_id": s.pair_id,
                "text": s.text,
            }
        )
        for s in stimuli
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class _Block(torch.nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden


class _Trunk(torch.nn.Module):
    def __init__(self, n_layers: int) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(_Block() for _ in range(n_layers))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class TinyLM(torch.nn.Module):
    """A hand-built ``.model.layers`` trunk plus a linear readout, enough for hook mechanics."""

    def __init__(self, n_layers: int, readout: torch.Tensor) -> None:
        super().__init__()
        self.model = _Trunk(n_layers)
        self.readout = readout

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.model(hidden) @ self.readout


def fit_args(tmp_path: Path, **overrides: Any) -> argparse.Namespace:
    root = tmp_path / "cells"
    stimuli = make_stimuli()
    plant_base_cell(root, stimuli)
    corpus = tmp_path / "stimuli.jsonl"
    write_stimuli_jsonl(corpus, stimuli)
    defaults: dict[str, Any] = {
        "capture_root": root,
        "stimuli": corpus,
        "arm": BASE_ARM,
        "step": 0,
        "set": STIMULUS_SET,
        "pooling": "last",
        "positive_side": "A",
        "pairs": "even",
        "out": tmp_path / "directions.pt",
        "compare": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestFitDirection:
    def test_the_fitted_direction_points_where_it_was_planted(self, tmp_path: Path) -> None:
        args = fit_args(tmp_path)
        sidecar = run_fit_direction(args)
        directions = load_direction_file(args.out)
        assert sorted(directions) == list(range(N_LAYERS))
        planted = cosine(directions[PLANTED_LAYER], planted_axis())
        assert planted > 0.9, f"fitted direction lost the planted axis, cosine {planted:.3f}"
        assert sidecar["n_fit_pairs"] == N_PAIRS // 2  # even half only

    def test_fitting_on_all_pairs_uses_them_all(self, tmp_path: Path) -> None:
        sidecar = run_fit_direction(fit_args(tmp_path, pairs="all"))
        assert sidecar["n_fit_pairs"] == N_PAIRS

    def test_an_edited_corpus_is_refused_before_any_fit(self, tmp_path: Path) -> None:
        args = fit_args(tmp_path)
        stimuli = make_stimuli()
        stimuli[0] = Stimulus(
            stimulus_id=stimuli[0].stimulus_id,
            stimulus_set=stimuli[0].stimulus_set,
            side=stimuli[0].side,
            pair_id=stimuli[0].pair_id,
            text=stimuli[0].text + " EDITED",
        )
        write_stimuli_jsonl(args.stimuli, stimuli)
        with pytest.raises(CellFormatError):
            run_fit_direction(args)

    def test_compare_records_agreement_with_a_second_file(self, tmp_path: Path) -> None:
        first = fit_args(tmp_path)
        run_fit_direction(first)
        second = fit_args(tmp_path, out=tmp_path / "again.pt", compare=first.out)
        sidecar = run_fit_direction(second)
        cosines = sidecar["compare_cosine_by_layer"]
        assert cosines is not None
        assert cosines[str(PLANTED_LAYER)] == pytest.approx(1.0)


class TestDirectionFiles:
    def test_a_bare_tensor_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.pt"
        torch.save(torch.zeros(4), path)
        with pytest.raises(ValueError, match="non-empty dict"):
            load_direction_file(path)

    def test_string_keys_are_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.pt"
        torch.save({"7": torch.zeros(4)}, path)
        with pytest.raises(ValueError, match="int layer"):
            load_direction_file(path)

    def test_duplicate_names_are_refused(self) -> None:
        with pytest.raises(ValueError, match="twice"):
            parse_named_specs(["a=1", "a=2"], flag="--direction")

    def test_layers_missing_from_the_direction_file_are_refused(self) -> None:
        directions = {"decision": {0: torch.zeros(4), 1: torch.zeros(4)}}
        with pytest.raises(ValueError, match=r"\[5\]"):
            parse_named_layers(["decision=1,5"], directions)

    def test_a_direction_without_layers_is_refused(self) -> None:
        directions = {"decision": {0: torch.zeros(4)}, "lead": {0: torch.zeros(4)}}
        with pytest.raises(ValueError, match="lead"):
            parse_named_layers(["decision=0"], directions)


class TestLabelTokens:
    def test_distinct_first_tokens_come_back_as_a_pair(self) -> None:
        encode = {"SHORT": [11, 12], "LONG": [21]}.__getitem__
        row = {"label_a": "SHORT", "label_b": "LONG", "coop_label": "LONG"}
        assert label_first_tokens(encode, row) == (21, 11)

    def test_a_shared_first_token_is_a_collision(self) -> None:
        encode = {"NORTH": [5, 1], "NORTHEAST": [5, 2]}.__getitem__
        row = {"label_a": "NORTH", "label_b": "NORTHEAST", "coop_label": "NORTH"}
        assert label_first_tokens(encode, row) is None


class TestHookMechanics:
    def test_alpha_zero_is_bit_identical_to_no_intervention(self) -> None:
        model = TinyLM(N_LAYERS, torch.randn(HIDDEN, generator=torch.Generator().manual_seed(0)))
        hidden = torch.randn(2, 3, HIDDEN, generator=torch.Generator().manual_seed(1))
        plain = model(hidden)
        with residual_intervention(model, PLANTED_LAYER, steering_hook(planted_axis(), 0.0)):  # pyright: ignore[reportArgumentType]
            steered = model(hidden)
        assert torch.equal(plain, steered)

    def test_an_out_of_range_layer_raises_rather_than_no_ops(self) -> None:
        model = TinyLM(N_LAYERS, torch.randn(HIDDEN))
        with (
            pytest.raises(IndexError),
            residual_intervention(model, 99, steering_hook(planted_axis(), 1.0)),  # pyright: ignore[reportArgumentType]
        ):
            pass

    def test_the_placebo_cannot_reproduce_a_real_effect(self) -> None:
        """The null the whole readout depends on: a matched-norm random direction must move a
        model far less than the direction its readout is built from."""
        readout = planted_axis() * 10.0
        model = TinyLM(N_LAYERS, readout)
        hidden = torch.randn(2, 3, HIDDEN, generator=torch.Generator().manual_seed(2))
        plain = model(hidden)
        alpha = 1.0
        real = planted_axis()
        placebo = matched_norm_random_direction(real, torch.Generator().manual_seed(3))
        with residual_intervention(model, PLANTED_LAYER, steering_hook(real, alpha)):  # pyright: ignore[reportArgumentType]
            steered_real = model(hidden)
        with residual_intervention(model, PLANTED_LAYER, steering_hook(placebo, alpha)):  # pyright: ignore[reportArgumentType]
            steered_placebo = model(hidden)
        real_effect = (steered_real - plain).abs().mean()
        placebo_effect = (steered_placebo - plain).abs().mean()
        assert real_effect == pytest.approx(alpha * 10.0, rel=1e-4)
        assert placebo_effect < real_effect / 3

    def test_placebo_conditions_push_the_placebo_not_the_real_axis(self) -> None:
        real = planted_axis()
        placebo = matched_norm_random_direction(real, torch.Generator().manual_seed(4))
        hook = hook_for(CONDITION_PLACEBO_UP, real, placebo, 2.0)
        hidden = torch.zeros(1, 1, HIDDEN)
        moved = cast("torch.Tensor", hook(object(), object(), hidden))
        delta = (moved - hidden)[0, 0]
        assert cosine(delta, unit(placebo)) == pytest.approx(1.0, abs=1e-5)
        assert abs(cosine(delta, real)) < 0.3

    def test_an_unknown_condition_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown condition"):
            hook_for("steer:sideways", planted_axis(), planted_axis(), 1.0)


def sweep_payload(cells: list[dict[str, Any]]) -> dict[str, Any]:
    return {"cells": cells}


def grid_point(
    direction: str, layer: int, multiplier: float, *, steer: float, placebo: float
) -> list[dict[str, Any]]:
    deltas = {
        CONDITION_STEER_UP: steer,
        CONDITION_STEER_DOWN: -steer,
        CONDITION_PLACEBO_UP: placebo,
        CONDITION_PLACEBO_DOWN: -placebo,
    }
    return [
        {
            "direction": direction,
            "layer": layer,
            "alpha_multiplier": multiplier,
            "condition": condition,
            "mean_delta_vs_none": delta,
        }
        for condition, delta in deltas.items()
    ]


class TestSelection:
    def test_the_most_selective_point_wins(self) -> None:
        payload = sweep_payload(
            grid_point("decision", 18, 1.0, steer=1.0, placebo=0.1)
            + grid_point("decision", 17, 1.0, steer=1.5, placebo=1.4)
        )
        choice = select_intervention(payload)
        assert (choice["direction"], choice["layer"]) == ("decision", 18)
        assert choice["fallback"] is False

    def test_a_placebo_dominated_sweep_falls_back_and_says_so(self) -> None:
        payload = sweep_payload(grid_point("decision", 18, 1.0, steer=0.2, placebo=0.5))
        choice = select_intervention(payload)
        assert choice["fallback"] is True
        assert "placebo" in choice["reason"]

    def test_an_incomplete_grid_is_refused(self) -> None:
        cells = grid_point("decision", 18, 1.0, steer=1.0, placebo=0.1)[:-1]
        with pytest.raises(ValueError, match="missing conditions"):
            select_intervention(sweep_payload(cells))

    def test_the_plan_holds_winner_larger_alpha_and_neighbour(self) -> None:
        payload = sweep_payload(
            grid_point("decision", 18, 1.0, steer=1.0, placebo=0.1)
            + grid_point("decision", 18, 2.0, steer=0.9, placebo=0.2)
            + grid_point("decision", 17, 1.0, steer=0.8, placebo=0.2)
            + grid_point("decision", 19, 1.0, steer=0.3, placebo=0.2)
        )
        plan = plan_generation_cells(payload, n_cells=3)
        assert plan[0] == {"direction": "decision", "layer": 18, "alpha_multiplier": 1.0}
        assert plan[1] == {"direction": "decision", "layer": 18, "alpha_multiplier": 2.0}
        assert plan[2] == {"direction": "decision", "layer": 17, "alpha_multiplier": 1.0}


class TestConditions:
    def test_one_cell_yields_baseline_steers_and_ablations(self) -> None:
        cells = [{"direction": "decision", "layer": 18, "alpha_multiplier": 1.0}]
        conditions = generation_conditions(cells)
        names = [c.condition for c in conditions]
        assert names == [
            CONDITION_NONE,
            *STEERING_CONDITIONS,
            CONDITION_ABLATE_REAL,
            CONDITION_ABLATE_PLACEBO,
        ]

    def test_two_alphas_at_one_layer_share_their_ablations(self) -> None:
        cells = [
            {"direction": "decision", "layer": 18, "alpha_multiplier": 1.0},
            {"direction": "decision", "layer": 18, "alpha_multiplier": 2.0},
        ]
        conditions = generation_conditions(cells)
        ablations = [c for c in conditions if c.condition.startswith("ablate")]
        assert len(ablations) == 2

    def test_condition_keys_are_stable_and_distinct(self) -> None:
        steer = GenerationCondition(CONDITION_STEER_UP, "decision", 18, 1.0)
        ablate = GenerationCondition(CONDITION_ABLATE_REAL, "decision", 18, None)
        assert condition_key(steer) == "decision:L18:x1.0:steer:+"
        assert condition_key(ablate) == "decision:L18:ablate:real"
        assert condition_key(GenerationCondition(CONDITION_NONE, None, None, None)) == "none"


class TestSummaries:
    def test_rates_carry_their_denominators(self) -> None:
        records = [
            {
                "condition_key": "none",
                "cooperate": value,
                "truncated_thinking": truncated,
                "label_print_order": order,
            }
            for value, truncated, order in [
                (True, False, "canonical"),
                (False, False, "canonical"),
                (None, True, "swapped"),
                (True, False, "swapped"),
            ]
        ]
        summary = summarise_records(records)["none"]
        assert summary["n_completions"] == 4
        assert summary["n_parsed"] == 3
        assert summary["n_parse_failures"] == 1
        assert summary["n_truncated_thinking"] == 1
        assert summary["cooperate_k"] == 2
        assert summary["cooperate_rate"] == pytest.approx(2 / 3)
        assert summary["by_print_order"]["swapped"]["n_parsed"] == 1


class TestRows:
    def test_both_print_orders_and_both_splits_are_rendered(self) -> None:
        rows = steering_rows(["eval", "train"])
        assert len(rows) == 160
        orders = {str(row["label_print_order"]) for row in rows}
        assert orders == {"canonical", "swapped"}


class _FakeBackend:
    """Stands in for HFBackend: carries a tiny trunk for the hooks, never generates."""

    transport = "hf"

    def __init__(self, model_id: str, *, thinking: bool, sampling: Any) -> None:
        self.model_id = model_id
        self.thinking = thinking
        self.sampling = sampling
        self._model = TinyLM(24, torch.randn(HIDDEN))

    def generate(self, prompts: list[str]) -> list[str]:
        raise AssertionError("the fake decode path should be used instead")


def generate_args(tmp_path: Path, directions_path: Path, **overrides: Any) -> argparse.Namespace:
    defaults: dict[str, Any] = {
        "model": "tiny/base",
        "direction": [f"decision={directions_path}"],
        "cells": ["decision:18:1.0"],
        "from_sweep": None,
        "n_cells": 3,
        "n_samples": 1,
        "batch_size": 4,
        "max_new_tokens": 64,
        "seed": 0,
        "deadline": None,
        "conditions": None,
        "counterpart_framing": None,
        "out_dir": tmp_path / "steering",
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@pytest.fixture
def directions_path(tmp_path: Path) -> Path:
    path = tmp_path / "decision.pt"
    torch.save({layer: torch.randn(HIDDEN) for layer in range(24)}, path)
    return path


TINY_KERNEL: dict[str, str] = {
    "chunk_gated_delta_rule": "fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule",
    "recurrent_gated_delta_rule": (
        "transformers.models.qwen3_5.modeling_qwen3_5.torch_recurrent_gated_delta_rule"
    ),
}
"""The un-bridged binding the offline fakes claim, stated rather than read off the modeling module.

Stated so these tests neither import that module nor depend on whether an earlier test in the process
bridged it, and so a relaunch can be handed the OTHER binding and watched being refused.
"""


@pytest.fixture(autouse=True)
def pinned_weights_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for the two things the real leg reads off its own process after loading the model.

    `tiny/base` is no hub id, so the revision lookup has nothing to resolve; and the DeltaNet kernel
    binding is a property of the process, which on a CPU test box would mean importing the Qwen3.5
    modeling module for a fake that has no kernels at all.
    """
    monkeypatch.setattr(
        interp_steering, "resolve_weights_identity", lambda model_id: f"test:{model_id}"
    )
    monkeypatch.setattr(
        interp_steering, "bridge_and_check_decode_kernel", lambda: {"bridged": False}
    )
    monkeypatch.setattr(interp_steering, "bound_deltanet_kernels", lambda: dict(TINY_KERNEL))


class TestGenerateOffline:
    def _patch(self, monkeypatch: pytest.MonkeyPatch, completions_for: Any) -> None:
        monkeypatch.setattr(interp_steering, "HFBackend", _FakeBackend)
        monkeypatch.setattr(interp_steering, "decode_in_chunks", completions_for)

    def test_records_and_summary_come_out_aligned(
        self, tmp_path: Path, directions_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = steering_rows(["eval"])

        def fake_decode(backend: Any, prompts: Any, *, chunk_size: int) -> list[str]:
            assert chunk_size == 4
            completions = []
            for index in range(len(prompts)):
                row = rows[index]
                label = (
                    row["coop_label"]
                    if index % 2 == 0
                    else (row["label_a"] if row["coop_label"] == row["label_b"] else row["label_b"])
                )
                completions.append(f"deliberation</think>\n<action>{label}</action>")
            return completions

        self._patch(monkeypatch, fake_decode)
        args = generate_args(tmp_path, directions_path)
        summary = run_generate(args)

        assert summary["resolved_sampler"]["applied"]["temperature"] == 1.0
        conditions = summary["conditions"]
        assert set(conditions) == {
            "none",
            "decision:L18:x1.0:steer:+",
            "decision:L18:x1.0:steer:-",
            "decision:L18:x1.0:placebo:+",
            "decision:L18:x1.0:placebo:-",
            "decision:L18:ablate:real",
            "decision:L18:ablate:placebo",
        }
        baseline = conditions["none"]
        assert baseline["n_completions"] == 32
        assert baseline["n_parse_failures"] == 0
        assert baseline["cooperate_rate"] == pytest.approx(0.5)
        records_file = args.out_dir / "steering_records.jsonl"
        lines = records_file.read_text().splitlines()
        assert len(lines) == 32 * 7

    def test_every_record_names_the_kernel_its_decode_ran_under(
        self, tmp_path: Path, directions_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rank 15's record half on the leg that actually decodes token by token.

        The fused decode kernel is 1.14-1.51x per step at 0.8B and its greedy tokens diverge from the
        torch fallback's from step 21 (probe I1), so which kernel a completion came out of is part of
        what the record is. Sabotage-verified: dropping the field from the record dict turns this red,
        and pointing one record at the other binding turns the summary red through the mixing guard.
        """
        self._patch(monkeypatch, _neutral_completions)
        args = generate_args(tmp_path, directions_path)
        summary = run_generate(args)

        records = [
            json.loads(line)
            for line in (args.out_dir / "steering_records.jsonl").read_text().splitlines()
        ]
        assert records
        assert all(record["deltanet_kernel"] == TINY_KERNEL for record in records)
        assert summary["deltanet_kernel"] == TINY_KERNEL
        assert summary["deltanet_kernel_bridge"] == {"bridged": False}

    def test_a_summary_over_two_kernels_is_refused(
        self, tmp_path: Path, directions_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The guard where a records file assembled from two runs would otherwise be averaged."""
        self._patch(monkeypatch, _neutral_completions)
        args = generate_args(tmp_path, directions_path, conditions="none")
        run_generate(args)
        records_path = args.out_dir / "steering_records.jsonl"
        lines = records_path.read_text().splitlines()
        tampered = json.loads(lines[0])
        tampered["deltanet_kernel"] = {
            **TINY_KERNEL,
            "recurrent_gated_delta_rule": "fla...fused_recurrent_gated_delta_rule",
        }
        with pytest.raises(ValueError, match="different Gated DeltaNet kernel bindings"):
            interp_steering.summarise_records([tampered, *(json.loads(line) for line in lines[1:])])

    def test_a_passed_deadline_labels_every_skipped_condition(
        self, tmp_path: Path, directions_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_decode(backend: Any, prompts: Any, *, chunk_size: int) -> list[str]:
            raise AssertionError("nothing should decode after the deadline")

        self._patch(monkeypatch, fake_decode)
        past = (dt.datetime.now(dt.UTC) - dt.timedelta(minutes=1)).isoformat()
        args = generate_args(tmp_path, directions_path, deadline=past)
        summary = run_generate(args)
        assert summary["conditions"] == {}
        assert len(summary["skipped_conditions"]) == 7
        assert all(entry["skipped"] == "deadline" for entry in summary["skipped_conditions"])

    def test_a_cell_naming_an_unknown_direction_is_refused(
        self, tmp_path: Path, directions_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch(monkeypatch, lambda *a, **k: [])
        args = generate_args(tmp_path, directions_path, cells=["lead:18:1.0"])
        with pytest.raises(ValueError, match="no --direction flag"):
            run_generate(args)

    def test_cells_and_from_sweep_are_mutually_exclusive(
        self, tmp_path: Path, directions_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch(monkeypatch, lambda *a, **k: [])
        args = generate_args(tmp_path, directions_path, cells=None, from_sweep=None)
        with pytest.raises(ValueError, match="exactly one"):
            run_generate(args)


class TestConditionFilter:
    """The --conditions gap-fill filter (added 2026-08-25 after three spot reclaims).

    The measurement-safety contract: filtering selects which conditions RUN but never renumbers
    them, so every executed condition draws the exact per-condition RNG seed (`args.seed + index`,
    index in the FULL condition list) an unfiltered run would have used. The r1/r2 bit-identical
    reproduction across boxes is the empirical half of that argument; these tests pin the
    structural half.
    """

    def _patch(self, monkeypatch: pytest.MonkeyPatch, completions_for: Any) -> None:
        monkeypatch.setattr(interp_steering, "HFBackend", _FakeBackend)
        monkeypatch.setattr(interp_steering, "decode_in_chunks", completions_for)

    @staticmethod
    def _neutral_decode(backend: Any, prompts: Any, *, chunk_size: int) -> list[str]:
        del backend, chunk_size
        return ["deliberation</think>\n<action>unparseable</action>" for _ in prompts]

    def test_a_filtered_run_executes_exactly_the_requested_conditions(
        self, tmp_path: Path, directions_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch(monkeypatch, self._neutral_decode)
        args = generate_args(
            tmp_path,
            directions_path,
            conditions="placebo:+,placebo:-,ablate:real,ablate:placebo",
        )
        summary = run_generate(args)
        assert set(summary["conditions"]) == {
            "decision:L18:x1.0:placebo:+",
            "decision:L18:x1.0:placebo:-",
            "decision:L18:ablate:real",
            "decision:L18:ablate:placebo",
        }
        skipped = summary["skipped_conditions"]
        assert {entry["condition_key"] for entry in skipped} == {
            "none",
            "decision:L18:x1.0:steer:+",
            "decision:L18:x1.0:steer:-",
        }
        assert all(entry["skipped"] == "not requested" for entry in skipped)

    def test_filtering_never_renumbers_the_per_condition_seeds(
        self, tmp_path: Path, directions_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch(monkeypatch, self._neutral_decode)
        full = run_generate(generate_args(tmp_path, directions_path, out_dir=tmp_path / "full"))
        filtered = run_generate(
            generate_args(
                tmp_path,
                directions_path,
                out_dir=tmp_path / "filtered",
                conditions="placebo:+,ablate:placebo",
            )
        )
        del full, filtered
        full_seeds = _seeds_by_condition(tmp_path / "full" / "steering_records.jsonl")
        filtered_seeds = _seeds_by_condition(tmp_path / "filtered" / "steering_records.jsonl")
        assert set(filtered_seeds) == {
            "decision:L18:x1.0:placebo:+",
            "decision:L18:ablate:placebo",
        }
        for key, seed in filtered_seeds.items():
            assert seed == full_seeds[key], f"{key} was renumbered by the filter"

    def test_a_full_condition_key_is_also_accepted(
        self, tmp_path: Path, directions_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch(monkeypatch, self._neutral_decode)
        args = generate_args(tmp_path, directions_path, conditions="decision:L18:x1.0:placebo:+")
        summary = run_generate(args)
        assert set(summary["conditions"]) == {"decision:L18:x1.0:placebo:+"}

    def test_a_name_matching_nothing_is_refused_before_any_decode(
        self, tmp_path: Path, directions_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def no_decode(backend: Any, prompts: Any, *, chunk_size: int) -> list[str]:
            raise AssertionError("nothing should decode when the filter refuses")

        self._patch(monkeypatch, no_decode)
        args = generate_args(tmp_path, directions_path, conditions="placebo:sideways")
        with pytest.raises(ValueError, match="matches nothing"):
            run_generate(args)

    def test_an_empty_filter_is_refused(
        self, tmp_path: Path, directions_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch(monkeypatch, self._neutral_decode)
        args = generate_args(tmp_path, directions_path, conditions=" , ")
        with pytest.raises(ValueError, match="names nothing"):
            run_generate(args)


def _seeds_by_condition(records_path: Path) -> dict[str, int]:
    seeds: dict[str, int] = {}
    for line in records_path.read_text().splitlines():
        record = json.loads(line)
        seeds[str(record["condition_key"])] = int(record["seed"])
    return seeds


class TestCounterpartFraming:
    """The --counterpart-framing prompt source (framing x interp cross, 2026-08-26).

    The flag swaps WHICH prompts the steering cell generates on and nothing else: the twin
    framing must render byte-identical prompt text to the default trained rendering, the framing
    must ride every record and the summary (so a cell can never be pooled into the wrong framing
    silently), an unknown framing must refuse before any decode, and the condition list and its
    index-derived seeds must be untouched by the flag -- a framing that renumbered seeds would
    break the seed-matched comparison against the arc's twin cells.
    """

    def _patch(self, monkeypatch: pytest.MonkeyPatch, completions_for: Any) -> None:
        monkeypatch.setattr(interp_steering, "HFBackend", _FakeBackend)
        monkeypatch.setattr(interp_steering, "decode_in_chunks", completions_for)

    @staticmethod
    def _neutral_decode(backend: Any, prompts: Any, *, chunk_size: int) -> list[str]:
        del backend, chunk_size
        return ["deliberation</think>\n<action>unparseable</action>" for _ in prompts]

    def test_the_twin_framing_renders_byte_identical_prompt_text(self) -> None:
        default_rows = generation_rows(None)
        twin_rows = generation_rows(FRAMING_TWIN)
        assert len(default_rows) == 32
        assert [str(row["prompt"]) for row in twin_rows] == [
            str(row["prompt"]) for row in default_rows
        ]

    def test_a_framing_swaps_the_prompts_and_tags_the_prompt_ids(self) -> None:
        human_rows = generation_rows(FRAMING_HUMAN)
        default_rows = generation_rows(None)
        assert len(human_rows) == len(default_rows)
        assert all("--framing-human" in str(row["prompt_id"]) for row in human_rows)
        assert [str(row["prompt"]) for row in human_rows] != [
            str(row["prompt"]) for row in default_rows
        ]

    def test_records_and_summary_carry_the_framing(
        self, tmp_path: Path, directions_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch(monkeypatch, self._neutral_decode)
        args = generate_args(
            tmp_path, directions_path, counterpart_framing=FRAMING_HUMAN, conditions="none"
        )
        summary = run_generate(args)
        assert summary["counterpart_framing"] == FRAMING_HUMAN
        records = [
            json.loads(line)
            for line in (args.out_dir / "steering_records.jsonl").read_text().splitlines()
        ]
        assert records
        assert all(record["counterpart_framing"] == FRAMING_HUMAN for record in records)
        assert all("--framing-human" in str(record["prompt_id"]) for record in records)

    def test_the_default_records_say_their_framing_is_none(
        self, tmp_path: Path, directions_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch(monkeypatch, self._neutral_decode)
        args = generate_args(tmp_path, directions_path, conditions="none")
        summary = run_generate(args)
        assert summary["counterpart_framing"] is None
        records = [
            json.loads(line)
            for line in (args.out_dir / "steering_records.jsonl").read_text().splitlines()
        ]
        assert all(record["counterpart_framing"] is None for record in records)

    def test_an_unknown_framing_is_refused_before_any_decode(
        self, tmp_path: Path, directions_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def no_decode(backend: Any, prompts: Any, *, chunk_size: int) -> list[str]:
            raise AssertionError("nothing should decode when the framing is refused")

        self._patch(monkeypatch, no_decode)
        args = generate_args(tmp_path, directions_path, counterpart_framing="martian")
        with pytest.raises(ValueError, match="Unknown framing_id"):
            run_generate(args)

    def test_the_framing_never_renumbers_condition_seeds(
        self, tmp_path: Path, directions_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch(monkeypatch, self._neutral_decode)
        run_generate(generate_args(tmp_path, directions_path, out_dir=tmp_path / "default"))
        run_generate(
            generate_args(
                tmp_path,
                directions_path,
                out_dir=tmp_path / "framed",
                counterpart_framing=FRAMING_HUMAN,
            )
        )
        default_seeds = _seeds_by_condition(tmp_path / "default" / "steering_records.jsonl")
        framed_seeds = _seeds_by_condition(tmp_path / "framed" / "steering_records.jsonl")
        assert framed_seeds == default_seeds
        assert set(framed_seeds) == {
            "none",
            "decision:L18:x1.0:steer:+",
            "decision:L18:x1.0:steer:-",
            "decision:L18:x1.0:placebo:+",
            "decision:L18:x1.0:placebo:-",
            "decision:L18:ablate:real",
            "decision:L18:ablate:placebo",
        }


class TestDecoderLayerContract:
    def test_the_tiny_trunk_resolves_like_a_real_checkpoint(self) -> None:
        model = TinyLM(N_LAYERS, torch.randn(HIDDEN))
        layers = _decoder_layers(model)  # pyright: ignore[reportArgumentType]
        assert len(layers) == N_LAYERS


def _neutral_completions(backend: Any, prompts: Any, *, chunk_size: int) -> list[str]:
    del backend, chunk_size
    return ["deliberation</think>\n<action>unparseable</action>" for _ in prompts]


class _CrashAfterConditions:
    """A decode that dies partway through the grid, the way a spot reclaim does."""

    def __init__(self, survive: int) -> None:
        self.survive = survive
        self.calls = 0

    def __call__(self, backend: Any, prompts: Any, *, chunk_size: int) -> list[str]:
        self.calls += 1
        if self.calls > self.survive:
            raise RuntimeError("simulated reclaim mid-condition")
        return _neutral_completions(backend, prompts, chunk_size=chunk_size)


ALL_CONDITION_KEYS = {
    "none",
    "decision:L18:x1.0:steer:+",
    "decision:L18:x1.0:steer:-",
    "decision:L18:x1.0:placebo:+",
    "decision:L18:x1.0:placebo:-",
    "decision:L18:ablate:real",
    "decision:L18:ablate:placebo",
}


class TestGenerateResume:
    """Automatic condition-level resume (hot-path backlog rank 16).

    The contract: a relaunch into the same ``--out-dir`` carries every complete condition forward
    (counted as ``resumed_conditions``, never as skipped), drops a partial condition and regenerates
    it, and ends with a records file byte-identical to an uninterrupted run's -- which holds because
    each condition reseeds from its own index. A relaunch under a different configuration is refused,
    as is a records file that has lost its ledger or disagrees with it.

    Sabotage-verified: duplicating one line of a complete condition trips the ledger check (the
    tampering test below IS that sabotage, kept as a test; deleting one is instead recovered as a
    partial condition, pinned in the ledger's own tests); changing the seed trips the identity
    check; writing the ledger BEFORE the flush (moving ``mark_complete`` above ``records_file.flush``
    in ``run_generate``) does not change any assertion here, which is why the ordering is also stated
    in the ledger's own docstring rather than left to a test to enforce.
    """

    def _patch(self, monkeypatch: pytest.MonkeyPatch, completions_for: Any) -> None:
        monkeypatch.setattr(interp_steering, "HFBackend", _FakeBackend)
        monkeypatch.setattr(interp_steering, "decode_in_chunks", completions_for)

    @staticmethod
    def _records(out_dir: Path) -> bytes:
        return (out_dir / "steering_records.jsonl").read_bytes()

    def test_a_relaunch_resumes_every_banked_condition_and_rewrites_nothing(
        self, tmp_path: Path, directions_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch(monkeypatch, _neutral_completions)
        args = generate_args(tmp_path, directions_path)
        first = run_generate(args)
        first_bytes = self._records(args.out_dir)

        def no_decode(backend: Any, prompts: Any, *, chunk_size: int) -> list[str]:
            raise AssertionError("every condition is banked; nothing should decode")

        self._patch(monkeypatch, no_decode)
        second = run_generate(generate_args(tmp_path, directions_path))

        assert {entry["condition_key"] for entry in second["resumed_conditions"]} == (
            ALL_CONDITION_KEYS
        )
        assert all(entry["n_records"] == 32 for entry in second["resumed_conditions"])
        assert second["skipped_conditions"] == []
        assert second["n_records_resumed"] == 32 * 7
        assert second["n_records_dropped_partial"] == 0
        assert second["conditions"] == first["conditions"]
        assert self._records(args.out_dir) == first_bytes

    def test_a_run_killed_mid_condition_resumes_into_the_uninterrupted_file(
        self, tmp_path: Path, directions_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The row's verification at CPU scale: kill, relaunch, diff against an unbroken run."""
        self._patch(monkeypatch, _neutral_completions)
        reference = generate_args(tmp_path, directions_path, out_dir=tmp_path / "reference")
        run_generate(reference)
        reference_lines = self._records(reference.out_dir).splitlines(keepends=True)

        self._patch(monkeypatch, _CrashAfterConditions(survive=3))
        interrupted = generate_args(tmp_path, directions_path, out_dir=tmp_path / "interrupted")
        with pytest.raises(RuntimeError, match="simulated reclaim"):
            run_generate(interrupted)
        records_path = interrupted.out_dir / "steering_records.jsonl"
        assert len(records_path.read_bytes().splitlines()) == 32 * 3
        ledger = json.loads(ledger_path_for(records_path).read_text())
        assert len(ledger["completed_units"]) == 3

        # A reclaim mid-write leaves part of the fourth condition plus a torn final line.
        with records_path.open("ab") as handle:
            handle.write(reference_lines[32 * 3])
            handle.write(reference_lines[32 * 3 + 1])
            handle.write(b'{"condition_key": "decision:L18:x1.0:pla')

        self._patch(monkeypatch, _neutral_completions)
        resumed = run_generate(
            generate_args(tmp_path, directions_path, out_dir=tmp_path / "interrupted")
        )

        assert len(resumed["resumed_conditions"]) == 3
        assert resumed["n_records_resumed"] == 32 * 3
        assert resumed["n_records_dropped_partial"] == 2
        assert set(resumed["conditions"]) == ALL_CONDITION_KEYS
        assert self._records(interrupted.out_dir) == self._records(reference.out_dir)

    def test_a_relaunch_under_another_seed_is_refused_as_a_different_run(
        self, tmp_path: Path, directions_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch(monkeypatch, _neutral_completions)
        run_generate(generate_args(tmp_path, directions_path, conditions="none"))
        with pytest.raises(ResumeMismatchError, match="seed"):
            run_generate(generate_args(tmp_path, directions_path, seed=1))

    def test_a_relaunch_across_the_fla_kernel_bridge_is_refused(
        self, tmp_path: Path, directions_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A resume that would decode its remaining conditions through the other kernel is not one.

        Refused here rather than at the summary, which is where the mixed file would otherwise surface:
        by then the conditions have been decoded and the GPU spent. The fix is a fresh --out-dir, since
        a file half decoded under each kernel is two measurements however its counts read.
        """
        self._patch(monkeypatch, _neutral_completions)
        run_generate(generate_args(tmp_path, directions_path, conditions="none"))
        monkeypatch.setattr(
            interp_steering,
            "bound_deltanet_kernels",
            lambda: {
                **TINY_KERNEL,
                "recurrent_gated_delta_rule": (
                    "fla.ops.gated_delta_rule.fused_recurrent.fused_recurrent_gated_delta_rule"
                ),
            },
        )
        with pytest.raises(ResumeMismatchError, match="deltanet_kernel"):
            run_generate(generate_args(tmp_path, directions_path))

    def test_a_relaunch_with_other_direction_vectors_is_refused(
        self, tmp_path: Path, directions_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same names, same keys, different fitted vectors: the digest is what tells them apart."""
        self._patch(monkeypatch, _neutral_completions)
        run_generate(generate_args(tmp_path, directions_path, conditions="none"))
        other = tmp_path / "other.pt"
        torch.save({layer: torch.randn(HIDDEN) for layer in range(24)}, other)
        with pytest.raises(ResumeMismatchError, match="directions_sha256"):
            run_generate(generate_args(tmp_path, other))

    def test_records_without_a_ledger_are_refused(
        self, tmp_path: Path, directions_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch(monkeypatch, _neutral_completions)
        args = generate_args(tmp_path, directions_path, conditions="none")
        run_generate(args)
        ledger_path_for(args.out_dir / "steering_records.jsonl").unlink()
        with pytest.raises(ResumeMismatchError, match="no ledger"):
            run_generate(generate_args(tmp_path, directions_path))

    def test_a_records_file_that_disagrees_with_its_ledger_is_refused(
        self, tmp_path: Path, directions_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One record of a complete condition duplicated: more on disk than the ledger recorded.

        A deleted record is the other direction and is recovered as a partial condition, since
        regeneration reproduces it; a surplus has no such explanation and is refused.
        """
        self._patch(monkeypatch, _neutral_completions)
        args = generate_args(tmp_path, directions_path, conditions="none")
        run_generate(args)
        records_path = args.out_dir / "steering_records.jsonl"
        lines = records_path.read_bytes().splitlines(keepends=True)
        records_path.write_bytes(b"".join([*lines, lines[0]]))
        with pytest.raises(ResumeMismatchError, match="disagrees with its own ledger"):
            run_generate(generate_args(tmp_path, directions_path))

    def test_a_relaunch_with_the_direction_flags_reordered_is_refused(
        self, tmp_path: Path, directions_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Placebos are seeded from a direction's flag position, so the order is part of the run.

        The name-keyed digests compare equal in any order; only the ordered list tells these apart.
        """
        self._patch(monkeypatch, _neutral_completions)
        other = tmp_path / "other.pt"
        torch.save({layer: torch.randn(HIDDEN) for layer in range(24)}, other)
        ordered = [f"decision={directions_path}", f"other={other}"]
        run_generate(generate_args(tmp_path, directions_path, direction=ordered, conditions="none"))
        with pytest.raises(ResumeMismatchError, match="direction_names"):
            run_generate(
                generate_args(tmp_path, directions_path, direction=list(reversed(ordered)))
            )

    def test_a_relaunch_on_other_base_weights_is_refused(
        self, tmp_path: Path, directions_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same model name, another resolved revision: the name alone would have resumed."""
        self._patch(monkeypatch, _neutral_completions)
        first = run_generate(generate_args(tmp_path, directions_path, conditions="none"))
        assert first["model_weights_identity"] == "test:tiny/base"
        monkeypatch.setattr(interp_steering, "resolve_weights_identity", lambda _model: "hf:moved")
        with pytest.raises(ResumeMismatchError, match="model_weights_identity"):
            run_generate(generate_args(tmp_path, directions_path))


class TestGenerationIdentityPinsThePrompts:
    """The resume identity digests the rendered prompts, not just their count.

    A relaunch at a commit whose prompt renderer moved would otherwise resume the same NUMBER of
    different prompts into one file; the digest is what tells the two apart.
    """

    def test_the_same_number_of_different_prompts_is_refused(
        self, tmp_path: Path, directions_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(interp_steering, "HFBackend", _FakeBackend)
        monkeypatch.setattr(interp_steering, "decode_in_chunks", _neutral_completions)
        run_generate(generate_args(tmp_path, directions_path, conditions="none"))
        real_rows = interp_steering.generation_rows

        def reworded_rows(counterpart_framing: str | None) -> list[dict[str, Any]]:
            return [
                {**row, "prompt": str(row["prompt"]) + " (reworded)"}
                for row in real_rows(counterpart_framing)
            ]

        monkeypatch.setattr(interp_steering, "generation_rows", reworded_rows)
        with pytest.raises(ResumeMismatchError, match="prompts_sha256"):
            run_generate(generate_args(tmp_path, directions_path))
