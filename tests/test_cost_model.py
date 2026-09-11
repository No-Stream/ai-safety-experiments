"""Tests for the measured cost model.

The arithmetic here is what `docs/scratch/measured-throughput.md` publishes as its re-derived
table, so an error would propagate straight into a planning decision. The properties worth
pinning are that episodes per hour divides out the batch size (the measured configuration does
not use the note's batch, so a per-step comparison would be apples to oranges), and that the
fitted size exponent refuses to mix token profiles.

One more, added after the price table was found to cover only the two shapes it started with: the
hardware a run is actually submitted to is decided in `cloud/submit_job.py`, and the two tables
drift silently. A tier added there and unpriced here does not raise -- it simply never appears in
the cost table, so the projection a run is sized against describes hardware nobody is renting.
`TestEveryShapeTheBatchSurfaceOffersIsPriced` is the tripwire for that drift.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from cloud.submit_job import GPU_TIERS
from grpo import cost_model as cm


def anchor(  # noqa: PLR0913, PLR0917
    model_id: str = "Qwen/Qwen3.5-4B",
    episodes_per_step: int = 24,
    seconds_per_step: float = 480.0,
    total_params: int = 4_571_730_432,
    prompt_tokens: int = 2048,
    completion_tokens: int = 2048,
    device_name: str = "NVIDIA L4",
) -> cm.MeasuredAnchor:
    return cm.MeasuredAnchor(
        model_id=model_id,
        device_name=device_name,
        episodes_per_step=episodes_per_step,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        seconds_per_step=seconds_per_step,
        peak_device_used_gib=21.0,
        generation_fraction=0.8,
        total_params=total_params,
    )


def _by_instance(measured: cm.MeasuredAnchor, episodes: int = 12_800) -> dict[str, cm.CostRow]:
    return {row.instance: row for row in cm.run_cost_rows(measured, episodes=episodes)}


class TestAnchorArithmetic:
    def test_seconds_per_episode_divides_out_the_batch(self):
        small = anchor(episodes_per_step=24, seconds_per_step=480.0)
        large = anchor(episodes_per_step=48, seconds_per_step=960.0)
        assert small.seconds_per_episode == pytest.approx(20.0)
        assert small.seconds_per_episode == pytest.approx(large.seconds_per_episode), (
            "episodes/hour must be comparable across batch sizes, or the re-derived table "
            "cannot be compared with a note written at a batch that does not fit"
        )

    def test_episodes_per_hour(self):
        assert anchor(episodes_per_step=24, seconds_per_step=480.0).episodes_per_hour == 180.0

    def test_hours_for_a_fixed_episode_budget(self):
        measured = anchor(episodes_per_step=24, seconds_per_step=480.0)
        assert measured.hours_for(12_800) == pytest.approx(12_800 * 20.0 / 3600.0)

    def test_speedup_divides_the_hours(self):
        measured = anchor()
        assert measured.hours_for(12_800, speedup=2.9) == pytest.approx(
            measured.hours_for(12_800) / 2.9
        )

    def test_params_billions_comes_from_the_loaded_model(self):
        assert anchor(total_params=4_571_730_432).params_billions == pytest.approx(4.5717, abs=1e-3)


class TestRunCost:
    def test_l4_row_is_measured_and_l40s_row_is_projected(self):
        by_instance = _by_instance(anchor())
        assert by_instance["g6.xlarge"].basis == "measured"
        assert "projected" in by_instance["g6e.xlarge"].basis
        assert by_instance["g6e.xlarge"].hours < by_instance["g6.xlarge"].hours

    def test_the_measured_row_follows_the_card_the_anchor_ran_on(self):
        """An L40S anchor labelled as an L4 measurement understates both rows by 2.9x.

        That anchor is producible today: the module advertises reproducing on an L40S unedited,
        and the artifact records whatever card `torch.cuda.get_device_properties` named.
        """
        by_instance = _by_instance(anchor(device_name=cm.L40S))
        assert by_instance["g6e.xlarge"].basis == "measured"
        assert "projected" in by_instance["g6.xlarge"].basis
        assert by_instance["g6e.xlarge"].on_demand_usd == pytest.approx(
            by_instance["g6e.xlarge"].hours * 1.8610
        )
        assert by_instance["g6.xlarge"].hours == pytest.approx(
            by_instance["g6e.xlarge"].hours * cm.L40S_OVER_L4_THROUGHPUT
        )

    def test_an_unrecognised_card_is_refused_rather_than_guessed(self):
        with pytest.raises(ValueError, match="NVIDIA A100"):
            cm.run_cost_rows(anchor(device_name="NVIDIA A100"), episodes=12_800)

    def test_dollars_follow_hours_and_price(self):
        row = _by_instance(anchor())["g6.xlarge"]
        assert row.on_demand_usd == pytest.approx(
            row.hours * cm.INSTANCE_ECONOMICS["g6.xlarge"].on_demand_usd_per_hour
        )
        assert row.spot_usd is not None
        assert row.spot_usd < row.on_demand_usd


class TestEveryShapeTheBatchSurfaceOffersIsPriced:
    """Which hardware a run lands on is decided by `cloud/submit_job.py`, not by this module.

    So the set of instances priced here has to cover the tiers that surface offers. It did not: the
    table held g6.xlarge and g6e.xlarge while submit_job offered g6e.xlarge and two g7e shapes, and
    an unpriced shape is not an error -- `run_cost_rows` iterates the price table, so the shape
    simply never appears and the reader is left comparing the two cards that happen to be listed.
    """

    def test_every_submitted_tier_has_an_entry(self):
        submitted = {tier.instance_type for tier in GPU_TIERS.values()}
        unpriced = sorted(submitted - set(cm.INSTANCE_ECONOMICS))
        assert unpriced == [], (
            f"cloud.submit_job can submit to {unpriced} and this module cannot price them, so a "
            "run sized against this table is sized against different hardware"
        )

    def test_every_priced_shape_names_a_card_with_a_throughput_factor(self):
        """A price with no hardware factor cannot make a row at all; catch it here, not mid-run."""
        unscaled = sorted(
            instance
            for instance, economics in cm.INSTANCE_ECONOMICS.items()
            if economics.card not in cm.CARD_THROUGHPUT_VS_L4
        )
        assert unscaled == []


class TestTheG7eShapes:
    """The 96 GiB card this repo actually rents, and the two-GPU shape beside it.

    Both are shapes `cloud/submit_job.py` offers today, and the g7e is where every arm too large for
    the local L4 has run. Pricing them wrong is a planning error; not pricing them at all was the
    state before this class existed.
    """

    def test_a_g7e_anchor_prices_its_own_shape_as_measured(self):
        by_instance = _by_instance(anchor(device_name=cm.RTX_PRO_6000))
        assert by_instance["g7e.2xlarge"].basis == "measured"
        assert "projected" in by_instance["g6.xlarge"].basis
        # The L4 is the slowest card in the table, so projecting onto it can only add hours.
        assert by_instance["g6.xlarge"].hours > by_instance["g7e.2xlarge"].hours

    def test_an_l4_anchor_projects_the_g7e_row_at_the_recorded_factor(self):
        by_instance = _by_instance(anchor())
        assert by_instance["g7e.2xlarge"].hours == pytest.approx(
            by_instance["g6.xlarge"].hours / cm.RTX_PRO_6000_OVER_L4_THROUGHPUT
        )
        assert by_instance["g7e.2xlarge"].on_demand_usd == pytest.approx(
            by_instance["g7e.2xlarge"].hours * 3.363
        )

    def test_the_two_gpu_shape_takes_the_same_hours_and_more_dollars(self):
        """Our GRPO loop is synchronous and single-process, so the second card buys no step time.

        `docs/scratch/games-throughput-optimization-2026-08-18.md` found exactly that: splitting
        generation onto GPU1 does not overlap the phases, so the step time matches colocate on one
        card. Which makes g7e.12xlarge the same hours at 2.5x the price -- true, and worth showing
        rather than hiding, since the plan once budgeted that shape as the cheaper option.
        """
        by_instance = _by_instance(anchor(device_name=cm.RTX_PRO_6000))
        assert by_instance["g7e.12xlarge"].gpus == 2
        assert by_instance["g7e.12xlarge"].hours == pytest.approx(by_instance["g7e.2xlarge"].hours)
        assert by_instance["g7e.12xlarge"].on_demand_usd > by_instance["g7e.2xlarge"].on_demand_usd

    def test_a_shape_with_no_observed_spot_price_says_so_rather_than_guessing(self):
        """No spot bid was ever recorded for the 2-GPU shape, and a guess would read as data."""
        by_instance = _by_instance(anchor(device_name=cm.RTX_PRO_6000))
        assert cm.INSTANCE_ECONOMICS["g7e.12xlarge"].spot_usd_per_hour_seen is None
        assert by_instance["g7e.12xlarge"].spot_usd is None
        report = cm.format_report(anchor(device_name=cm.RTX_PRO_6000), [], episodes=12_800)
        assert "n/a" in report, f"an unobserved spot price must be visible as absent:\n{report}"


class TestEachRowSaysWhichHardwareItPrices:
    """A row that names only an instance type leaves the card to the reader's memory.

    `format_report` prints the anchor's own card, so the measured row is identifiable; every other
    row was not. And keying the measured row on the card the anchor recorded -- rather than on a
    second card-to-instance table -- is what makes the two impossible to drift apart.
    """

    def test_the_measured_row_carries_the_anchors_own_card(self):
        rows = cm.run_cost_rows(anchor(device_name=cm.L40S), episodes=12_800)
        measured = [row for row in rows if row.basis == "measured"]
        assert [row.card for row in measured] == [cm.L40S]
        assert [row.instance for row in measured] == ["g6e.xlarge"]

    def test_every_row_names_its_card_and_gpu_count(self):
        for row in cm.run_cost_rows(anchor(), episodes=12_800):
            economics = cm.INSTANCE_ECONOMICS[row.instance]
            assert row.card == economics.card
            assert row.gpus == economics.gpus

    def test_the_report_prints_the_card_beside_the_instance(self):
        report = cm.format_report(anchor(), [], episodes=12_800)
        assert cm.L40S in report, f"the projected row's card is not in the report:\n{report}"
        assert cm.RTX_PRO_6000 in report, report


class TestSizeExponent:
    def test_recovers_a_planted_exponent(self):
        """Three synthetic points on a clean power law must fit back to their own exponent."""
        exponent = 0.75
        ladder = []
        for params in (0.864e9, 2.03e9, 4.572e9):
            billions = params / 1e9
            seconds_per_episode = 3.0 * billions**exponent
            ladder.append(
                anchor(
                    total_params=int(params),
                    episodes_per_step=16,
                    seconds_per_step=seconds_per_episode * 16,
                    prompt_tokens=512,
                    completion_tokens=512,
                )
            )
        fit = cm.fit_size_exponent(ladder)
        assert fit["exponent"] == pytest.approx(exponent, abs=1e-6)
        assert fit["r_squared"] == pytest.approx(1.0, abs=1e-9)

    def test_refuses_to_mix_token_profiles(self):
        """Fitting across profiles would fold the token axis into the size axis."""
        ladder = [
            anchor(total_params=864_000_000, prompt_tokens=512, completion_tokens=512),
            anchor(total_params=4_572_000_000, prompt_tokens=2048, completion_tokens=2048),
        ]
        with pytest.raises(ValueError, match="must share one token profile"):
            cm.fit_size_exponent(ladder)

    def test_refuses_a_single_anchor(self):
        with pytest.raises(ValueError, match="at least two anchors"):
            cm.fit_size_exponent([anchor()])


def test_load_anchor_reads_a_throughput_artifact(tmp_path: Path) -> None:
    """The loader must track the JSON schema `grpo.throughput` actually writes."""
    artifact = {
        "config": {
            "model_id": "Qwen/Qwen3.5-4B",
            "prompt_tokens": 2048,
            "completion_tokens": 2048,
        },
        "device": {"device_name": "NVIDIA L4"},
        "episodes_per_step": 24,
        "median_step_seconds": 480.0,
        "peak_device_used_gib": 21.0,
        "generation_fraction_of_step": 0.8,
        "lora": {"total_params": 4_571_730_432},
    }
    path = tmp_path / "point.json"
    path.write_text(json.dumps(artifact))
    loaded = cm.load_anchor(path)
    assert loaded.episodes_per_step == 24
    assert loaded.seconds_per_step == 480.0
    assert loaded.params_billions == pytest.approx(4.5717, abs=1e-3)


def test_format_report_includes_the_assumed_and_measured_bases():
    ladder = [
        anchor(
            total_params=int(p),
            episodes_per_step=16,
            seconds_per_step=s * 16,
            prompt_tokens=512,
            completion_tokens=512,
        )
        for p, s in ((0.864e9, 1.0), (2.03e9, 2.0), (4.572e9, 4.0))
    ]
    report = cm.format_report(anchor(), ladder, episodes=12_800)
    assert "episodes/hour" in report
    assert "g6e.xlarge" in report
    assert "N^1.00" in report, "the note's assumed exponent should be shown for comparison"


class TestTokenAxisFit:
    """Separating decode cost from prefill cost, which the budget note charges identically."""

    def test_recovers_a_planted_line_on_the_completion_axis(self):
        per_token, fixed = 0.004, 1.5
        points = [
            anchor(
                episodes_per_step=16,
                seconds_per_step=(fixed + per_token * completion) * 16,
                prompt_tokens=512,
                completion_tokens=completion,
            )
            for completion in (32, 128, 512)
        ]
        fit = cm.fit_token_axis(points, "completion_tokens")
        assert fit["axis_seconds_per_token"] == pytest.approx(per_token, abs=1e-9)
        assert fit["fixed_seconds_per_episode"] == pytest.approx(fixed, abs=1e-9)
        assert fit["r_squared"] == pytest.approx(1.0, abs=1e-9)

    def test_recovers_a_planted_line_on_the_prompt_axis(self):
        per_token, fixed = 0.001, 2.0
        points = [
            anchor(
                episodes_per_step=4,
                seconds_per_step=(fixed + per_token * prompt) * 4,
                prompt_tokens=prompt,
                completion_tokens=256,
            )
            for prompt in (1024, 4096, 16384)
        ]
        fit = cm.fit_token_axis(points, "prompt_tokens")
        assert fit["axis_seconds_per_token"] == pytest.approx(per_token, abs=1e-9)
        assert fit["fixed_seconds_per_episode"] == pytest.approx(fixed, abs=1e-9)

    def test_refuses_when_the_other_axis_moved(self):
        """A slope fitted across two moving axes silently attributes one to the other."""
        points = [
            anchor(episodes_per_step=16, prompt_tokens=512, completion_tokens=128),
            anchor(episodes_per_step=16, prompt_tokens=2048, completion_tokens=512),
        ]
        with pytest.raises(ValueError, match="must hold model, episodes/step"):
            cm.fit_token_axis(points, "completion_tokens")

    def test_refuses_when_the_episode_count_moved(self):
        points = [
            anchor(episodes_per_step=8, prompt_tokens=512, completion_tokens=128),
            anchor(episodes_per_step=16, prompt_tokens=512, completion_tokens=512),
        ]
        with pytest.raises(ValueError, match="must hold model, episodes/step"):
            cm.fit_token_axis(points, "completion_tokens")

    def test_rejects_an_unknown_axis(self):
        with pytest.raises(ValueError, match="axis must be"):
            cm.fit_token_axis([anchor(), anchor()], "episodes")
