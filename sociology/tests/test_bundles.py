"""Bundle construction: determinism, the twin rule, exclusion and redraw counting, manifests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import sociology.bundles as bundles_module
from sociology.bundles import (
    CELLS,
    CELLS_BY_NAME,
    MAX_UNIT_RENDER_CHARS,
    BundleSpec,
    assert_manifest_current,
    build_bundles,
    build_manifest,
    build_pool,
    bundles_of,
    first_bundles_per_family,
    load_manifest,
    render_prompt,
    write_manifest,
)
from sociology.corpus import FAMILY_AGENTIC_20B, FAMILY_AGENTIC_120B, FAMILY_SINGLE_TURN_120B
from sociology.tests.conftest import make_unit

if TYPE_CHECKING:
    from pathlib import Path

    from sociology.stimulus import Stimulus


def synthetic_pools() -> dict[str, bundles_module.FamilyPool]:
    """Pools big enough to draw every production cell, with twin structure in the agentic ones."""
    pools: dict[str, bundles_module.FamilyPool] = {}
    for family in (FAMILY_AGENTIC_120B, FAMILY_AGENTIC_20B):
        units = [
            make_unit(
                f"{family}:task_{index}:{variant}",
                family=family,
                exclusion_key=f"task_{index}",
                rendered=f"synthetic {variant} body for task {index} in {family} " * 20,
            )
            for index in range(60)
            for variant in ("conflicting", "original")
        ]
        pools[family] = build_pool(family, units)
    single = [
        make_unit(
            f"prob_{problem}|s{sample}",
            family=FAMILY_SINGLE_TURN_120B,
            exclusion_key=f"prob_{problem}",
            rendered=f"synthetic single-turn body {problem}-{sample} " * 10,
        )
        for problem in range(40)
        for sample in range(8)
    ]
    pools[FAMILY_SINGLE_TURN_120B] = build_pool(FAMILY_SINGLE_TURN_120B, single)
    return pools


class TestDraws:
    def test_bundles_are_deterministic_across_rebuilds(self) -> None:
        first = build_bundles(synthetic_pools())
        second = build_bundles(synthetic_pools())
        assert first == second

    def test_twin_rule_never_co_bundles_a_conflicting_original_pair(self) -> None:
        by_cell = build_bundles(synthetic_pools())
        pools = synthetic_pools()
        for bundles in by_cell.values():
            for bundle in bundles:
                keys = [pools[bundle.family].units[m].exclusion_key for m in bundle.members]
                assert len(keys) == len(set(keys)), f"bundle {bundle.bundle_id} repeats a problem"

    def test_bundles_within_a_cell_are_disjoint(self) -> None:
        by_cell = build_bundles(synthetic_pools())
        for cell_name, bundles in by_cell.items():
            if CELLS_BY_NAME[cell_name].reuses_bundles_of is not None:
                continue
            seen: set[str] = set()
            for bundle in bundles:
                overlap = seen.intersection(bundle.members)
                assert not overlap, f"{cell_name} reuses units across bundles: {overlap}"
                seen.update(bundle.members)

    def test_matched_cells_reuse_the_center_bundles_verbatim(self) -> None:
        by_cell = build_bundles(synthetic_pools())
        center = [(b.bundle_id, b.members) for b in by_cell["center"]]
        for reuser in ("framing-independent", "framing-unstated", "cues-stripped"):
            assert [(b.bundle_id, b.members) for b in by_cell[reuser]] == center

    def test_size_rungs_draw_their_own_bundles(self) -> None:
        by_cell = build_bundles(synthetic_pools())
        center_ids = {b.bundle_id for b in by_cell["center"]}
        for own_cell in ("size-4", "size-16"):
            assert center_ids.isdisjoint(b.bundle_id for b in by_cell[own_cell])

    def test_cell_shapes_match_the_design_table(self) -> None:
        by_cell = build_bundles(synthetic_pools())
        expected = {cell.name: cell.bundles_per_family * len(cell.families) for cell in CELLS}
        assert {name: len(bundles) for name, bundles in by_cell.items()} == expected
        for cell in CELLS:
            assert all(len(b.members) == cell.size for b in by_cell[cell.name])


class TestExclusionAndCap:
    def test_over_cap_unit_is_excluded_and_counted(self) -> None:
        units = [make_unit("small", rendered="tiny body")]
        units.append(make_unit("huge", rendered="x" * (MAX_UNIT_RENDER_CHARS + 1)))
        pool = build_pool(FAMILY_AGENTIC_120B, units)
        assert pool.eligible_ids == ("small",)
        assert pool.excluded_over_cap == ("huge",)

    def test_bundle_cap_forces_counted_redraws(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Only the two small units fit together under the tiny cap, so any draw touching a big
        # unit must be redrawn -- and the accepted bundle must record how many draws it cost.
        units = [
            make_unit("big-a", rendered="B" * 400),
            make_unit("big-b", rendered="B" * 400),
            make_unit("small-a", rendered="s" * 40),
            make_unit("small-b", rendered="s" * 40),
        ]
        pool = build_pool(FAMILY_AGENTIC_120B, units)
        monkeypatch.setattr(bundles_module, "MAX_BUNDLE_RENDER_CHARS", 200)
        cell = bundles_module.CellSpec(
            name="cap-test",
            design_label="T",
            families=(FAMILY_AGENTIC_120B,),
            size=2,
            bundles_per_family=1,
            framing="population",
            cues="kept",
        )
        bundles = bundles_module._draw_family_bundles(cell, pool)
        assert len(bundles) == 1
        assert set(bundles[0].members) == {"small-a", "small-b"}
        assert bundles[0].redraws > 0, (
            "the seed happened to draw the fitting pair first; pick other ids"
        )

    def test_impossible_cap_raises_after_bounded_redraws(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        units = [make_unit(f"unit-{i}", rendered="B" * 400) for i in range(4)]
        pool = build_pool(FAMILY_AGENTIC_120B, units)
        monkeypatch.setattr(bundles_module, "MAX_BUNDLE_RENDER_CHARS", 100)
        monkeypatch.setattr(bundles_module, "MAX_REDRAWS_PER_BUNDLE", 3)
        cell = bundles_module.CellSpec(
            name="cap-impossible",
            design_label="T",
            families=(FAMILY_AGENTIC_120B,),
            size=2,
            bundles_per_family=1,
            framing="population",
            cues="kept",
        )
        with pytest.raises(ValueError, match="does not fit this pool"):
            bundles_module._draw_family_bundles(cell, pool)

    def test_underfilled_draw_raises_rather_than_shrinking(self) -> None:
        units = [
            make_unit("twin-a", exclusion_key="same-problem"),
            make_unit("twin-b", exclusion_key="same-problem"),
        ]
        pool = build_pool(FAMILY_AGENTIC_120B, units)
        cell = bundles_module.CellSpec(
            name="underfilled",
            design_label="T",
            families=(FAMILY_AGENTIC_120B,),
            size=2,
            bundles_per_family=1,
            framing="population",
            cues="kept",
        )
        with pytest.raises(ValueError, match="without repeating a problem identity"):
            bundles_module._draw_family_bundles(cell, pool)


class TestManifest:
    def test_roundtrip_matches_a_rebuild_through_json_normalisation(
        self, tmp_path: Path, stimulus: Stimulus
    ) -> None:
        manifest = build_manifest(synthetic_pools(), stimulus)
        write_manifest(manifest, tmp_path)
        loaded = load_manifest(tmp_path)
        rebuilt = build_manifest(synthetic_pools(), stimulus)
        # Loaded holds lists where the rebuild holds tuples; the comparison must not care.
        assert_manifest_current(loaded, rebuilt)

    def test_corrupted_member_goes_red(self, tmp_path: Path, stimulus: Stimulus) -> None:
        manifest = build_manifest(synthetic_pools(), stimulus)
        write_manifest(manifest, tmp_path)
        loaded = load_manifest(tmp_path)
        loaded["cells"][0]["bundles"][0]["members"][0] = "tampered-unit-id"
        with pytest.raises(RuntimeError, match="does not match a rebuild"):
            assert_manifest_current(loaded, build_manifest(synthetic_pools(), stimulus))

    def test_corrupted_seed_goes_red(self, tmp_path: Path, stimulus: Stimulus) -> None:
        manifest = build_manifest(synthetic_pools(), stimulus)
        write_manifest(manifest, tmp_path)
        loaded = load_manifest(tmp_path)
        loaded["cells"][0]["bundles"][0]["seed"] = "0" * 64
        with pytest.raises(RuntimeError, match="does not match a rebuild"):
            assert_manifest_current(loaded, build_manifest(synthetic_pools(), stimulus))

    def test_pool_exclusions_are_recorded_in_the_manifest(self, stimulus: Stimulus) -> None:
        pools = synthetic_pools()
        oversized = make_unit(
            "zz-oversized",
            family=FAMILY_AGENTIC_120B,
            exclusion_key="zz-oversized",
            rendered="x" * (MAX_UNIT_RENDER_CHARS + 1),
        )
        units = [*pools[FAMILY_AGENTIC_120B].units.values(), oversized]
        pools[FAMILY_AGENTIC_120B] = build_pool(FAMILY_AGENTIC_120B, units)
        manifest = build_manifest(pools, stimulus)
        pool_entry = manifest["pools"][FAMILY_AGENTIC_120B]
        assert pool_entry["excluded_over_cap"] == 1
        assert pool_entry["excluded_unit_ids"] == ["zz-oversized"]

    def test_bundles_of_and_first_per_family(self, tmp_path: Path, stimulus: Stimulus) -> None:
        write_manifest(build_manifest(synthetic_pools(), stimulus), tmp_path)
        loaded = load_manifest(tmp_path)
        center = bundles_of(loaded, "center")
        assert all(isinstance(spec, BundleSpec) for spec in center)
        subset = first_bundles_per_family(center, 5)
        assert len(subset) == 10
        per_family: dict[str, int] = dict.fromkeys((FAMILY_AGENTIC_120B, FAMILY_AGENTIC_20B), 0)
        for spec in subset:
            per_family[spec.family] += 1
        assert per_family == {FAMILY_AGENTIC_120B: 5, FAMILY_AGENTIC_20B: 5}
        assert [spec.bundle_id for spec in subset] == [
            spec.bundle_id for spec in center if spec in subset
        ]


class TestPromptRender:
    def test_prompt_is_frame_then_request_then_bundle(self, stimulus: Stimulus) -> None:
        cell = CELLS_BY_NAME["center"]
        prompt = render_prompt(stimulus, cell, "BUNDLE-TEXT", n=10)
        frame = stimulus.frame_text("population", n=10)
        assert prompt.index(frame) < prompt.index(stimulus.constant_request)
        assert prompt.index(stimulus.constant_request) < prompt.index("BUNDLE-TEXT")
