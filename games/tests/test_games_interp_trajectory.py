"""Run the whole trajectory analysis on planted geometry, where the right answer is known in advance.

Offline and CPU-only, over cells written straight to disk: the analysis never touches a model, so a
fixture ladder is the real path rather than a stand-in for it. The plant is deliberate -- a fixed unit
axis, a per-checkpoint amplitude, and a per-checkpoint rotation into a second axis -- because a
trajectory read is a claim about *changes*, and the only way to know it reports them faithfully is to
put a change in and check the number that comes out.

What each class pins:

:class:`TestPlantedTrajectory` -- the axis is found where it was planted and nowhere else, the drift
cosine falls as the planted axis rotates, and the held-out projection gap tracks the planted
amplitude. Also the two numbers that make a drift cosine readable: the matched-norm placebo floor
sits near zero and the split-half ceiling sits below one.

:class:`TestCrossArm` -- two arms moving oppositely on the same stimuli, anchored on one shared base
cell. The gap difference has to carry the sign of the divergence, and the cosine between the arms at
the anchor step has to read exactly 1.0, since at that step the two arms *are* the same cell. That
last one is a construction check on the pairing: if it ever drifts off 1.0, the cross-arm read is
comparing something other than what it claims.

:class:`TestCorrelation` -- Pearson and Spearman on eight points, and the two ways a correlation is
unavailable rather than zero (a constant series, too few points). A returned 0.0 would read as
"measured, and unrelated", which is a different claim.

:class:`TestReportArtifacts` -- the report and the saved directions, because the directions file is
the input to the causal tier and to the lens ladder, so its shape is a contract.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
import torch

from games.interp_cells import (
    BASE_ARM,
    BASE_STEP,
    CellIdentity,
    Stimulus,
    load_ladder,
    row_index_for,
    step_dir,
    stimuli_digest,
    write_cell,
)
from games.interp_trajectory import (
    DIRECTIONS_DIRNAME,
    REPORT_FILENAME,
    GroupKey,
    analysis_arms,
    arm_series,
    build_parser,
    cross_arm_reads,
    read_group,
    run,
    trajectory_correlation,
)
from reward_hacking.interp.linear_probe import ProbeConfig
from reward_hacking.interp.run_steer_validation import _spearman as steering_spearman
from reward_hacking.interp.run_steer_validation import average_ranks, pearson_correlation

if TYPE_CHECKING:
    from pathlib import Path

N_LAYERS = 2
PLANTED_LAYER = 1
STIMULUS_SET = "correlated-vs-independent-counterpart"
N_PLACEBOS = 20

# The width, pair count, amplitude and noise are not free parameters: a fixture has to sit inside a
# window, and a fixture outside it tests nothing. A matched-norm placebo retains 1/sqrt(hidden) of the
# planted signal, so it only lands at chance while `2 * amplitude << noise * sqrt(hidden)`; a
# diff-of-means direction estimated from `n_pairs` pairs only points where it was planted while
# `2 * amplitude >> noise * sqrt(2 * hidden / n_pairs)`. At 16 dimensions with amplitude 0.5 and noise
# 0.05 -- the first thing tried here -- a random direction separated the classes as well as the real
# axis did, and the planted-axis test could not pass at all.
HIDDEN = 128
N_PAIRS = 20
NOISE = 0.3

# The amplitude of the planted separation per checkpoint: one arm climbs, its twin falls through zero.
CLIMBING = {0: 1.2, 10: 1.5, 20: 1.8, 30: 2.1}
FALLING = {0: 1.2, 10: 0.7, 20: 0.2, 30: -0.5}

# How far the planted axis rotates toward a second axis by each checkpoint, so drift is predictable.
ROTATION = {0: 0.0, 10: 0.2, 20: 0.4, 30: 0.8}


def planted_projected_gap(amplitudes: dict[int, float], rotation: float, step: int) -> float:
    """The gap a projection onto the *anchor* axis should read at one checkpoint.

    Not the amplitude: the axis rotates as the amplitude grows, and a projection onto a fixed axis
    reads the component along it, so the planted quantity is `2 * amplitude * cos(rotation angle)`.
    That distinction is the whole reason this helper exists rather than the test comparing against the
    amplitude directly -- with the rotation this fixture plants, the amplitude climbs monotonically
    while its projection onto the anchor does not.
    """
    return 2.0 * amplitudes[step] / (1.0 + rotation**2) ** 0.5


def planted_axes() -> tuple[torch.Tensor, torch.Tensor]:
    """Two fixed orthogonal unit axes: the planted direction and the one it rotates toward."""
    first = torch.zeros(HIDDEN)
    first[0] = 1.0
    second = torch.zeros(HIDDEN)
    second[1] = 1.0
    return first, second


def make_stimuli() -> list[Stimulus]:
    """Matched pairs, side A first, in the order the corpus writer emits them."""
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


def identity_for(stimuli: list[Stimulus]) -> CellIdentity:
    """The one identity every fixture cell shares."""
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


def plant_cell(  # noqa: PLR0913 - a planted cell is a root, an arm, a step, an amplitude and a rotation
    root: Path,
    stimuli: list[Stimulus],
    *,
    arm: str,
    step: int,
    amplitude: float,
    rotation: float,
    seed: int = 0,
) -> Path:
    """Write one cell whose planted layer separates the two sides along a rotated axis.

    The rotation is what makes drift measurable: at rotation 0 the axis is the anchor's, and by
    rotation 0.8 it has turned most of the way toward a second, orthogonal axis.
    """
    first, second = planted_axes()
    axis = first + rotation * second
    axis = axis / axis.norm()
    generator = torch.Generator().manual_seed(seed + step)
    matrix = torch.randn(len(stimuli), N_LAYERS, HIDDEN, generator=generator) * NOISE
    for row, stimulus in enumerate(stimuli):
        sign = 1.0 if stimulus.side == "A" else -1.0
        matrix[row, PLANTED_LAYER, :] += sign * amplitude * axis
    return write_cell(
        step_dir(root, arm, step),
        arm=arm,
        step=step,
        identity=identity_for(stimuli),
        rows={STIMULUS_SET: row_index_for(stimuli, [42] * len(stimuli))},
        activations={(STIMULUS_SET, "mean"): matrix},
        applied_adapter_weights=None if arm == BASE_ARM else 372,
        adapter_weights_sha256=None if arm == BASE_ARM else f"{arm}-{step}",
        provenance={"git_sha": "testing"},
    ).parent


def plant_ladder(root: Path, stimuli: list[Stimulus]) -> None:
    """Write the base anchor plus two arms moving oppositely, on identical stimuli."""
    plant_cell(root, stimuli, arm=BASE_ARM, step=BASE_STEP, amplitude=CLIMBING[0], rotation=0.0)
    for step in (10, 20, 30):
        plant_cell(
            root,
            stimuli,
            arm="climbing-arm",
            step=step,
            amplitude=CLIMBING[step],
            rotation=ROTATION[step],
        )
        plant_cell(
            root,
            stimuli,
            arm="falling-arm",
            step=step,
            amplitude=FALLING[step],
            rotation=ROTATION[step] / 2.0,
        )


def read_planted_group(root: Path, arm: str) -> Any:
    """Read one arm's group off a planted ladder, at both layers."""
    ladder = load_ladder(root)
    return read_group(
        ladder,
        GroupKey(arm=arm, stimulus_set=STIMULUS_SET, pooling="mean"),
        layers=list(range(N_LAYERS)),
        positive_side="A",
        config=ProbeConfig(),
        n_placebos=N_PLACEBOS,
    )


@pytest.fixture(scope="module")
def planted_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One planted ladder for the whole module: reading a group is the expensive part, not writing it."""
    root = tmp_path_factory.mktemp("planted-capture")
    plant_ladder(root, make_stimuli())
    return root


@pytest.fixture(scope="module")
def climbing(planted_root: Path) -> Any:
    """The arm whose planted separation grows, read once and shared."""
    return read_planted_group(planted_root, "climbing-arm")


@pytest.fixture(scope="module")
def falling(planted_root: Path) -> Any:
    """The arm whose planted separation falls through zero, read once and shared."""
    return read_planted_group(planted_root, "falling-arm")


class TestPlantedTrajectory:
    def test_the_axis_is_found_where_it_was_planted(self, climbing: Any) -> None:
        group = climbing
        planted = [
            read
            for read in group.axis_reads
            if read.layer == PLANTED_LAYER and read.step == BASE_STEP
        ]
        assert all(read.beats_placebo for read in planted)
        assert all(read.clears_null for read in planted)
        assert group.peak_layer == PLANTED_LAYER

    def test_a_noise_layer_does_not_clear_its_placebo(self, climbing: Any) -> None:
        """The other half of "found where it was planted": nothing to find at the unplanted layer."""
        group = climbing
        noise = next(
            read
            for read in group.axis_reads
            if read.layer != PLANTED_LAYER and read.step == BASE_STEP
        )
        assert not noise.beats_placebo

    def test_drift_falls_as_the_planted_axis_rotates(self, climbing: Any) -> None:
        group = climbing
        by_step = {read.step: read for read in group.drift_reads if read.layer == PLANTED_LAYER}
        assert by_step[BASE_STEP].cosine_to_anchor == pytest.approx(1.0)
        cosines = [by_step[step].cosine_to_anchor for step in (10, 20, 30)]
        assert cosines == sorted(cosines, reverse=True)
        assert by_step[30].cosine_to_anchor < 0.9

    def test_the_drift_floor_and_ceiling_are_reported(self, climbing: Any) -> None:
        """A cosine of 0.85 means nothing without the placebo floor and the split-half ceiling."""
        group = climbing
        anchor = next(
            read
            for read in group.drift_reads
            if read.layer == PLANTED_LAYER and read.step == BASE_STEP
        )
        assert abs(anchor.cosine_anchor_placebo) < 0.4
        assert 0.5 < anchor.anchor_split_half_cosine <= 1.0
        assert anchor.norm_ratio == pytest.approx(1.0)

    def test_the_held_out_projection_gap_tracks_the_planted_amplitude(self, climbing: Any) -> None:
        """Against the planted *projection*, and attenuated: the anchor axis is itself an estimate.

        The measured gap runs about 85% of the planted value here, because the axis it projects onto
        was fitted on ten pairs and so is not exactly the planted one. The attenuation is one-sided --
        a finite-sample axis can only lose signal, never manufacture it -- so the bound is a band
        rather than a tolerance, and the same attenuation applies to the real captures.
        """
        group = climbing
        by_step = {
            read.step: read.heldout_projection_gap
            for read in group.drift_reads
            if read.layer == PLANTED_LAYER
        }
        steps = [0, 10, 20, 30]
        gaps = [by_step[step] for step in steps]
        planted = [planted_projected_gap(CLIMBING, ROTATION[step], step) for step in steps]
        correlation = pearson_correlation(gaps, planted)
        assert correlation is not None
        assert correlation > 0.9
        for gap, expected in zip(gaps, planted, strict=True):
            assert 0.6 * expected < gap <= expected

    def test_the_placebo_projection_gap_stays_at_the_floor(self, climbing: Any) -> None:
        group = climbing
        for read in group.drift_reads:
            if read.layer == PLANTED_LAYER:
                assert abs(read.heldout_projection_gap_placebo) < abs(read.heldout_projection_gap)

    def test_a_falling_arm_crosses_zero(self, falling: Any) -> None:
        """The behavioural analogue: one arm's separation reverses sign rather than just shrinking."""
        group = falling
        by_step = {
            read.step: read.heldout_projection_gap
            for read in group.drift_reads
            if read.layer == PLANTED_LAYER
        }
        assert by_step[0] > 0
        assert by_step[30] < 0

    def test_the_anchor_leads_the_series(self, planted_root: Path) -> None:
        ladder = load_ladder(planted_root)
        series = arm_series(ladder, "climbing-arm")
        assert [cell.label for cell in series] == [
            "base/step-0",
            "climbing-arm/step-10",
            "climbing-arm/step-20",
            "climbing-arm/step-30",
        ]
        assert analysis_arms(ladder) == ("climbing-arm", "falling-arm")


class TestCrossArm:
    def test_the_gap_difference_carries_the_divergence(self, climbing: Any, falling: Any) -> None:
        groups = {
            GroupKey(arm="climbing-arm", stimulus_set=STIMULUS_SET, pooling="mean"): climbing,
            GroupKey(arm="falling-arm", stimulus_set=STIMULUS_SET, pooling="mean"): falling,
        }
        reads = cross_arm_reads(groups, "climbing-arm", "falling-arm")
        by_step = {read.step: read for read in reads if read.layer == PLANTED_LAYER}
        assert by_step[0].projection_gap_difference == pytest.approx(0.0, abs=1e-6)
        assert by_step[30].projection_gap_difference < by_step[0].projection_gap_difference
        assert by_step[30].projection_gap_a > by_step[30].projection_gap_b

    def test_the_shared_anchor_reads_exactly_one(self, climbing: Any, falling: Any) -> None:
        """At step 0 both arms are the same cell, so this is a check on the pairing, not a finding."""
        groups = {
            GroupKey(arm="climbing-arm", stimulus_set=STIMULUS_SET, pooling="mean"): climbing,
            GroupKey(arm="falling-arm", stimulus_set=STIMULUS_SET, pooling="mean"): falling,
        }
        reads = cross_arm_reads(groups, "climbing-arm", "falling-arm")
        anchor = next(read for read in reads if read.step == 0 and read.layer == PLANTED_LAYER)
        assert anchor.cosine_between_arms == pytest.approx(1.0)


class TestCorrelation:
    def test_a_monotone_pair_correlates(self) -> None:
        result = trajectory_correlation([1.0, 2.0, 3.0, 4.0], [0.5, 0.6, 0.7, 0.8])
        assert result.pearson == pytest.approx(1.0)
        assert result.spearman == pytest.approx(1.0)
        assert result.n_points == 4

    def test_an_inverted_pair_anticorrelates(self) -> None:
        result = trajectory_correlation([1.0, 2.0, 3.0, 4.0], [0.8, 0.7, 0.6, 0.5])
        assert result.pearson == pytest.approx(-1.0)

    def test_a_constant_series_has_no_correlation_rather_than_zero(self) -> None:
        result = trajectory_correlation([1.0, 1.0, 1.0, 1.0], [0.5, 0.6, 0.7, 0.8])
        assert result.pearson is None
        assert result.spearman is None
        assert result.unavailable_reason == "a series is constant"

    def test_too_few_points_says_so(self) -> None:
        result = trajectory_correlation([1.0, 2.0], [0.5, 0.6])
        assert result.pearson is None
        assert result.unavailable_reason is not None
        assert "fewer than" in result.unavailable_reason

    def test_mismatched_lengths_are_refused(self) -> None:
        with pytest.raises(ValueError, match="series lengths differ"):
            trajectory_correlation([1.0, 2.0, 3.0], [1.0, 2.0])

    def test_ties_share_an_average_rank(self) -> None:
        """The midrank convention this module's Spearman rests on, pinned where it is consumed.

        The ranking and the Pearson step used to be a second copy living in this module, spelled
        independently from the one in ``reward_hacking.interp.run_steer_validation`` -- two places for
        a tie convention to drift apart. They were checked to agree exactly on 45 series (ties,
        constants, short ones) and then collapsed into the shared pair, so this assertion now pins the
        shared implementation as seen from this consumer.
        """
        assert average_ranks([10.0, 20.0, 20.0, 30.0]) == [1.0, 2.5, 2.5, 4.0]

    def test_this_module_and_the_steering_readout_agree_on_a_tie_containing_series(self) -> None:
        """Both consumers of the shared pair must report the same Spearman on the same ties.

        ``trajectory_correlation`` guards the point count itself and returns a reason string, while
        the steering readout's ``_spearman`` guards internally and returns ``None`` -- two call paths
        with separate guards over one implementation, so agreement is a real check rather than a
        tautology.
        """
        geometry = [1.0, 2.0, 2.0, 3.0, 5.0]
        behavior = [0.5, 0.9, 0.9, 0.4, 2.0]

        assert trajectory_correlation(geometry, behavior).spearman == pytest.approx(
            steering_spearman(geometry, behavior)
        )

    def test_spearman_ignores_a_monotone_transform_pearson_notices(self) -> None:
        """The reason both are reported: they fail differently on a handful of checkpoints."""
        xs = [1.0, 2.0, 3.0, 4.0]
        ys = [1.0, 2.0, 3.0, 100.0]
        assert trajectory_correlation(xs, ys).spearman == pytest.approx(1.0)
        pearson = pearson_correlation(xs, ys)
        assert pearson is not None
        assert pearson < 1.0


class TestReportArtifacts:
    def write_corpus(self, tmp_path: Path, stimuli: list[Stimulus]) -> Path:
        path = tmp_path / "stimuli.jsonl"
        path.write_text(
            "\n".join(
                json.dumps(
                    {
                        "id": stimulus.stimulus_id,
                        "set": stimulus.stimulus_set,
                        "side": stimulus.side,
                        "pair_id": stimulus.pair_id,
                        "text": stimulus.text,
                    }
                )
                for stimulus in stimuli
            )
            + "\n"
        )
        return path

    def test_the_cli_writes_a_report_and_directions(self, tmp_path: Path) -> None:
        stimuli = make_stimuli()
        capture_root = tmp_path / "capture"
        plant_ladder(capture_root, stimuli)
        corpus = self.write_corpus(tmp_path, stimuli)
        out_dir = tmp_path / "analysis"
        behavior = tmp_path / "behavior.json"
        # A behavioural series that genuinely tracks the planted geometry, so a high correlation is
        # the right answer: scaled from the planted projection rather than from the amplitude, since
        # the projection is what the geometry read measures.
        behavior.write_text(
            json.dumps(
                {
                    arm: {
                        str(step): 0.5
                        + planted_projected_gap(amplitudes, ROTATION[step] / divisor, step) / 20.0
                        for step in (0, 10, 20, 30)
                    }
                    for arm, amplitudes, divisor in (
                        ("climbing-arm", CLIMBING, 1.0),
                        ("falling-arm", FALLING, 2.0),
                    )
                }
            )
        )
        args = build_parser().parse_args(
            [
                "--capture-root",
                str(capture_root),
                "--stimuli",
                str(corpus),
                "--out-dir",
                str(out_dir),
                "--positive-side",
                "A",
                "--n-placebos",
                str(N_PLACEBOS),
                "--behavior-json",
                str(behavior),
            ]
        )
        payload = run(args)

        assert (out_dir / REPORT_FILENAME).is_file()
        on_disk = json.loads((out_dir / REPORT_FILENAME).read_text())
        assert on_disk["context"]["stimuli_sha256"] == stimuli_digest(stimuli)
        assert set(payload["groups"]) == {
            "climbing-arm|correlated-vs-independent-counterpart|mean",
            "falling-arm|correlated-vs-independent-counterpart|mean",
        }
        assert payload["cross_arm"]

        correlation = payload["behavior_correlations"][
            "climbing-arm|correlated-vs-independent-counterpart|mean"
        ]
        assert correlation["steps"] == [0, 10, 20, 30]
        assert correlation["projection_gap"]["pearson"] > 0.95

        saved = out_dir / DIRECTIONS_DIRNAME / "climbing-arm" / "step-30"
        path = saved / f"{STIMULUS_SET}-mean.pt"
        directions = torch.load(path, weights_only=True)
        assert sorted(directions) == list(range(N_LAYERS))
        assert directions[PLANTED_LAYER].shape == (HIDDEN,)

    def test_a_mismatched_corpus_stops_the_analysis(self, tmp_path: Path) -> None:
        """The identity guard, reached through the CLI: edited corpus, same ids, refused."""
        stimuli = make_stimuli()
        capture_root = tmp_path / "capture"
        plant_ladder(capture_root, stimuli)
        edited = [*stimuli[:-1], Stimulus(**{**vars(stimuli[-1]), "text": "an edited rendering"})]
        corpus = self.write_corpus(tmp_path, edited)
        args = build_parser().parse_args(
            [
                "--capture-root",
                str(capture_root),
                "--stimuli",
                str(corpus),
                "--out-dir",
                str(tmp_path / "analysis"),
                "--positive-side",
                "A",
                "--n-placebos",
                "2",
            ]
        )
        with pytest.raises(ValueError, match="different stimulus corpus"):
            run(args)
