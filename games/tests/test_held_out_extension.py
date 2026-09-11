"""Tests for the runtime-loaded held-out scenario extension and the battery flag that applies it.

Every frame the extension carries is authored benchmark material that will be run against future
models, so it lives in a gitignored staging file and none of it appears here. The synthetic fixtures
below build staging files of the same SHAPE out of throwaway prose, which is what lets this suite be
green in a fresh clone with no skips; the two tests that read the real file skip loudly when it is
not on the machine.

The gates under test are the ones a widened roster can break silently. An entry that arrived without
`eval_only=True` would join the next training corpus while every count still added up; a reused
scenario id would collide two frames' `prompt_id`s and pool their draws; a reused label string would
pool two frames in every per-label analysis; a matrix frame carrying counterpart-decision language
would reintroduce the twin framing through the fiction of a game that is supposed to state nothing
about how the other side decides.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from games.evals import (
    EVAL_ONLY_GRADING,
    EVAL_RENDER_GRADING_BY_GAME,
    RECORD_META,
    SECTION_FRAMING_SWEEP,
    SECTION_GAME_BEHAVIOR,
    EvalConfig,
    plan_battery,
    read_eval_records,
    run_eval_battery,
)
from games.held_out_extension import (
    CAPS_FILENAME,
    CAPS_VERSION,
    EXTENSION_VERSION,
    STAGING_PATH,
    ExtensionRosters,
    load_extension,
)
from games.prompts import (
    DICTATOR_GAME_ID,
    MATRIX_SCENARIOS,
    RENDERABLE_GAME_IDS,
    SPLIT_EVAL,
    SPLIT_TRAIN,
    TRUST_STATED_RETURN_GAME_ID,
    TRUSTEE_RETURN_GAME_ID,
    generate_prompt_rows,
)
from games.run_evals import prompt_set_digest
from reward_hacking.model_backend import MockBackend

if TYPE_CHECKING:
    from pathlib import Path

# The real staging file's four families, from its own inventory. Pinned here so a file that lost a
# family or gained an entry says so rather than quietly shrinking every held-out bank.
REAL_FAMILY_COUNTS: dict[str, int] = {"matrix": 22, "dictator": 26, "trust": 26, "trustee": 26}

# Throwaway label pairs for the synthetic frames: none of them is a tracked label, which is what the
# global label-uniqueness gate requires, and none is a word the vocabulary gate bans.
SYNTHETIC_LABEL_PAIRS: tuple[tuple[str, str], ...] = (
    ("QUILL", "SLATE"),
    ("BRACKET", "PLINTH"),
    ("GIRDER", "SPANDREL"),
    ("TROWEL", "FLOAT"),
)

CAPPED_GAME_ID = "twin-pd"

# A scenario_id passes the vocabulary gate like every other string, and three of the four family
# names are themselves banned words, so the synthetic ids are keyed by a neutral slug per family.
FAMILY_ID_SLUG: dict[str, str] = {
    "matrix": "bench",
    "dictator": "issue",
    "trust": "sendout",
    "trustee": "receipt",
}


def synthetic_matrix_entry(index: int) -> dict[str, str]:
    """Build one matrix staging entry whose prose passes the tracked gates and mentions both labels."""
    first, second = SYNTHETIC_LABEL_PAIRS[index]
    return {
        "scenario_id": f"synthetic-bench-{index}",
        "frame": (
            f"BENCH NOTE -- run {index}\n\n"
            f"Your bench and the far bench each hold a docket for the same delivery. Mark yours "
            f"{first} to send your boards to the shared stack, or {second} to hold them at your own "
            f"end. Neither bench sees the far docket before the run closes, and the two marks "
            f"settle between them what the run is worth in points."
        ),
        "label_a": first,
        "label_b": second,
    }


def synthetic_resource_entry(family: str, index: int) -> dict[str, str]:
    """Build one dictator, trust or trustee staging entry: a frame plus the resource it names."""
    return {
        "scenario_id": f"synthetic-{FAMILY_ID_SLUG[family]}-store-{index}",
        "frame": (
            f"STORE NOTE -- lot {index}\n\n"
            f"The season's crates came in to your store this week, unmarked and unbooked, and the "
            f"note that goes back out with them is yours to write."
        ),
        "resource": "crates",
    }


def synthetic_payload(counts: dict[str, int]) -> dict[str, Any]:
    """Build a whole staging document with the requested number of entries per family."""
    payload: dict[str, Any] = {"version": EXTENSION_VERSION}
    for family, count in counts.items():
        if family == "matrix":
            payload[family] = [synthetic_matrix_entry(index) for index in range(count)]
        else:
            payload[family] = [synthetic_resource_entry(family, index) for index in range(count)]
    return payload


def write_staging(
    directory: Path,
    payload: dict[str, Any],
    *,
    caps: dict[str, list[str]] | None = None,
    caps_payload: dict[str, Any] | None = None,
    write_caps: bool = True,
) -> Path:
    """Write a staging file and its caps sidecar into `directory`, returning the staging path."""
    directory.mkdir(parents=True, exist_ok=True)
    staging_path = directory / "extension.json"
    staging_path.write_text(json.dumps(payload), encoding="utf-8")
    if write_caps:
        resolved = (
            caps_payload
            if caps_payload is not None
            else {"version": CAPS_VERSION, "matrix_game_caps": caps or {}}
        )
        (directory / CAPS_FILENAME).write_text(json.dumps(resolved), encoding="utf-8")
    return staging_path


@pytest.fixture
def staging_dir(tmp_path: Path) -> Path:
    return tmp_path / "extension-package"


class TestLoadExtension:
    def test_a_synthetic_file_constructs_the_frames_it_carries(self, staging_dir: Path) -> None:
        path = write_staging(
            staging_dir,
            synthetic_payload({"matrix": 3, "dictator": 2, "trust": 2, "trustee": 1}),
            caps={CAPPED_GAME_ID: ["synthetic-bench-0", "synthetic-bench-2"]},
        )
        rosters = load_extension(path)
        assert len(rosters.matrix) == 3
        assert len(rosters.dictator) == 2
        assert len(rosters.trust) == 2
        assert len(rosters.trustee) == 1

    def test_every_loaded_frame_is_held_out(self, staging_dir: Path) -> None:
        """The whole point of the extension: no runtime frame may reach a training corpus."""
        path = write_staging(staging_dir, synthetic_payload({"matrix": 2, "dictator": 1}))
        rosters = load_extension(path)
        assert all(scenario.eval_only for scenario in rosters.matrix)
        assert all(scenario.eval_only for scenario in rosters.dictator)

    def test_a_family_the_file_omits_loads_empty(self, staging_dir: Path) -> None:
        path = write_staging(staging_dir, synthetic_payload({"matrix": 1}))
        rosters = load_extension(path)
        assert len(rosters.matrix) == 1
        assert rosters.dictator == ()
        assert rosters.trust == ()
        assert rosters.trustee == ()

    def test_an_entry_carrying_eval_only_is_refused(self, staging_dir: Path) -> None:
        """The schema forbids the field because the loader is what guarantees the frames are held out.

        Matched on the dedicated refusal's own wording rather than on the field name: the generic
        field-set refusal quotes the field list too, so a bare `eval_only` match would stay green
        with the dedicated check deleted.
        """
        payload = synthetic_payload({"matrix": 1})
        payload["matrix"][0]["eval_only"] = True
        path = write_staging(staging_dir, payload)
        with pytest.raises(ValueError, match="eval_only field"):
            load_extension(path)

    def test_an_entry_with_an_unknown_field_is_refused(self, staging_dir: Path) -> None:
        payload = synthetic_payload({"dictator": 1})
        payload["dictator"][0]["register"] = "far"
        path = write_staging(staging_dir, payload)
        with pytest.raises(ValueError, match="register"):
            load_extension(path)

    def test_an_unknown_top_level_key_is_refused(self, staging_dir: Path) -> None:
        payload = synthetic_payload({"matrix": 1})
        payload["notes"] = "authoring notes belong in another file"
        path = write_staging(staging_dir, payload)
        with pytest.raises(ValueError, match="notes"):
            load_extension(path)

    def test_a_wrong_version_is_refused(self, staging_dir: Path) -> None:
        payload = synthetic_payload({"matrix": 1})
        payload["version"] = "games-held-out-extension-v0"
        path = write_staging(staging_dir, payload)
        with pytest.raises(ValueError, match="version"):
            load_extension(path)

    def test_an_absent_file_is_refused_loudly(self, staging_dir: Path) -> None:
        staging_dir.mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="gitignored"):
            load_extension(staging_dir / "extension.json")

    def test_a_frame_reusing_a_tracked_id_is_refused(self, staging_dir: Path) -> None:
        payload = synthetic_payload({"matrix": 1})
        payload["matrix"][0]["scenario_id"] = MATRIX_SCENARIOS[0].scenario_id
        path = write_staging(staging_dir, payload)
        with pytest.raises(ValueError, match="already a tracked frame"):
            load_extension(path)

    def test_a_frame_reusing_another_entrys_id_is_refused(self, staging_dir: Path) -> None:
        payload = synthetic_payload({"matrix": 2})
        payload["matrix"][1]["scenario_id"] = payload["matrix"][0]["scenario_id"]
        path = write_staging(staging_dir, payload)
        with pytest.raises(ValueError, match="more than once"):
            load_extension(path)

    def test_an_id_that_is_not_a_path_safe_slug_is_refused(self, staging_dir: Path) -> None:
        payload = synthetic_payload({"matrix": 1})
        payload["matrix"][0]["scenario_id"] = "Synthetic Bench"
        path = write_staging(staging_dir, payload)
        with pytest.raises(ValueError, match="scenario_id"):
            load_extension(path)

    def test_a_matrix_frame_asserting_coupling_is_refused(self, staging_dir: Path) -> None:
        """A skin may imply a counterpart exists and must say nothing about how it decides."""
        payload = synthetic_payload({"matrix": 1})
        payload["matrix"][0]["frame"] += " The far bench works it out the same way you do."
        path = write_staging(staging_dir, payload)
        with pytest.raises(ValueError, match="decision-coupling"):
            load_extension(path)

    def test_a_frame_naming_the_literature_is_refused(self, staging_dir: Path) -> None:
        payload = synthetic_payload({"trust": 1})
        payload["trust"][0]["frame"] += " Whoever receives them may reciprocate in kind."
        path = write_staging(staging_dir, payload)
        with pytest.raises(ValueError, match="loaded vocabulary"):
            load_extension(path)

    def test_a_matrix_frame_reusing_a_tracked_label_is_refused(self, staging_dir: Path) -> None:
        payload = synthetic_payload({"matrix": 1})
        payload["matrix"][0]["frame"] = payload["matrix"][0]["frame"].replace(
            SYNTHETIC_LABEL_PAIRS[0][0], MATRIX_SCENARIOS[0].label_a
        )
        payload["matrix"][0]["label_a"] = MATRIX_SCENARIOS[0].label_a
        path = write_staging(staging_dir, payload)
        with pytest.raises(ValueError, match="already used by"):
            load_extension(path)

    def test_two_entries_sharing_a_label_are_refused(self, staging_dir: Path) -> None:
        payload = synthetic_payload({"matrix": 2})
        shared = SYNTHETIC_LABEL_PAIRS[0][0]
        payload["matrix"][1]["frame"] = payload["matrix"][1]["frame"].replace(
            SYNTHETIC_LABEL_PAIRS[1][0], shared
        )
        payload["matrix"][1]["label_a"] = shared
        path = write_staging(staging_dir, payload)
        with pytest.raises(ValueError, match="already used by"):
            load_extension(path)

    def test_a_frame_that_never_mentions_its_labels_is_refused(self, staging_dir: Path) -> None:
        payload = synthetic_payload({"matrix": 1})
        payload["matrix"][0]["label_b"] = "GANTRY"
        path = write_staging(staging_dir, payload)
        with pytest.raises(ValueError, match="never mentions"):
            load_extension(path)

    def test_the_digest_moves_with_the_content_and_not_with_the_path(self, tmp_path: Path) -> None:
        payload = synthetic_payload({"matrix": 2, "dictator": 1})
        first = load_extension(write_staging(tmp_path / "one", payload))
        second = load_extension(write_staging(tmp_path / "two", payload))
        assert first.digest == second.digest
        edited = synthetic_payload({"matrix": 2, "dictator": 1})
        edited["dictator"][0]["resource"] = "hampers"
        third = load_extension(write_staging(tmp_path / "three", edited))
        assert third.digest != first.digest

    def test_the_digest_moves_with_the_caps(self, tmp_path: Path) -> None:
        """A cap is part of what a cell rendered, so two caps may not share a digest."""
        payload = synthetic_payload({"matrix": 2})
        wide = load_extension(
            write_staging(
                tmp_path / "wide",
                payload,
                caps={CAPPED_GAME_ID: ["synthetic-bench-0", "synthetic-bench-1"]},
            )
        )
        narrow = load_extension(
            write_staging(
                tmp_path / "narrow", payload, caps={CAPPED_GAME_ID: ["synthetic-bench-0"]}
            )
        )
        assert wide.digest != narrow.digest


class TestMatrixGameCaps:
    def test_a_capped_game_renders_only_the_listed_ids(self, staging_dir: Path) -> None:
        path = write_staging(
            staging_dir,
            synthetic_payload({"matrix": 3}),
            caps={CAPPED_GAME_ID: ["synthetic-bench-2", "synthetic-bench-0"]},
        )
        rosters = load_extension(path)
        capped = rosters.matrix_for_game(CAPPED_GAME_ID)
        assert [scenario.scenario_id for scenario in capped] == [
            "synthetic-bench-0",
            "synthetic-bench-2",
        ]

    def test_a_game_absent_from_the_caps_renders_no_extension_frames(
        self, staging_dir: Path
    ) -> None:
        """Absent means none: a default of every frame would widen eleven games' eval splits at once."""
        path = write_staging(
            staging_dir,
            synthetic_payload({"matrix": 2}),
            caps={CAPPED_GAME_ID: ["synthetic-bench-0"]},
        )
        rosters = load_extension(path)
        assert rosters.matrix_for_game("harmony") == ()

    def test_a_cap_naming_an_id_the_file_lacks_is_refused(self, staging_dir: Path) -> None:
        path = write_staging(
            staging_dir, synthetic_payload({"matrix": 1}), caps={CAPPED_GAME_ID: ["no-such-frame"]}
        )
        with pytest.raises(ValueError, match="no-such-frame"):
            load_extension(path)

    def test_a_cap_naming_a_game_that_renders_no_matrix_roster_is_refused(
        self, staging_dir: Path
    ) -> None:
        path = write_staging(
            staging_dir,
            synthetic_payload({"matrix": 1}),
            caps={DICTATOR_GAME_ID: ["synthetic-bench-0"]},
        )
        with pytest.raises(ValueError, match=DICTATOR_GAME_ID):
            load_extension(path)

    def test_a_missing_caps_file_is_refused(self, staging_dir: Path) -> None:
        path = write_staging(staging_dir, synthetic_payload({"matrix": 1}), write_caps=False)
        with pytest.raises(FileNotFoundError, match=CAPS_FILENAME):
            load_extension(path)

    def test_a_caps_file_with_the_wrong_version_is_refused(self, staging_dir: Path) -> None:
        path = write_staging(
            staging_dir,
            synthetic_payload({"matrix": 1}),
            caps_payload={"version": "caps-v0", "matrix_game_caps": {}},
        )
        with pytest.raises(ValueError, match="version"):
            load_extension(path)

    def test_a_caps_file_with_an_unknown_key_is_refused(self, staging_dir: Path) -> None:
        path = write_staging(
            staging_dir,
            synthetic_payload({"matrix": 1}),
            caps_payload={
                "version": CAPS_VERSION,
                "matrix_game_caps": {},
                "dictator_caps": {},
            },
        )
        with pytest.raises(ValueError, match="dictator_caps"):
            load_extension(path)


class TestExtensionRostersReachTheRowBuilders:
    def test_a_capped_matrix_game_renders_the_extension_frames(self, staging_dir: Path) -> None:
        path = write_staging(
            staging_dir,
            synthetic_payload({"matrix": 2}),
            caps={CAPPED_GAME_ID: ["synthetic-bench-0", "synthetic-bench-1"]},
        )
        rosters = load_extension(path)
        without = generate_prompt_rows(CAPPED_GAME_ID, "group-mix", split=SPLIT_EVAL)
        with_extension = generate_prompt_rows(
            CAPPED_GAME_ID,
            "group-mix",
            split=SPLIT_EVAL,
            extra_eval_frames=rosters.extra_eval_frames_for(CAPPED_GAME_ID),
        )
        added = {row["prompt_id"] for row in with_extension} - {row["prompt_id"] for row in without}
        assert len(with_extension) > len(without)
        assert all("synthetic-bench-" in prompt_id for prompt_id in added)

    def test_an_uncapped_matrix_game_renders_exactly_the_tracked_frames(
        self, staging_dir: Path
    ) -> None:
        path = write_staging(
            staging_dir,
            synthetic_payload({"matrix": 2}),
            caps={CAPPED_GAME_ID: ["synthetic-bench-0"]},
        )
        rosters = load_extension(path)
        without = generate_prompt_rows("harmony", "group-mix", split=SPLIT_EVAL)
        with_extension = generate_prompt_rows(
            "harmony",
            "group-mix",
            split=SPLIT_EVAL,
            extra_eval_frames=rosters.extra_eval_frames_for("harmony"),
        )
        assert [row["prompt_id"] for row in with_extension] == [row["prompt_id"] for row in without]

    def test_the_resource_games_render_their_own_family(self, staging_dir: Path) -> None:
        path = write_staging(
            staging_dir, synthetic_payload({"trust": 2, "trustee": 1, "dictator": 3})
        )
        rosters = load_extension(path)
        for game_id, family_size in (
            (DICTATOR_GAME_ID, 3),
            (TRUST_STATED_RETURN_GAME_ID, 2),
            (TRUSTEE_RETURN_GAME_ID, 1),
        ):
            without = generate_prompt_rows(game_id, "group-mix", split=SPLIT_EVAL)
            with_extension = generate_prompt_rows(
                game_id,
                "group-mix",
                split=SPLIT_EVAL,
                extra_eval_frames=rosters.extra_eval_frames_for(game_id),
            )
            added = {row["reskin_id"] for row in with_extension} - {
                row["reskin_id"] for row in without
            }
            assert len(added) == family_size, game_id

    def test_a_family_the_chosen_game_cannot_render_is_refused(self, staging_dir: Path) -> None:
        """The other silent shape: a family whose roster this game's row builder never iterates."""
        path = write_staging(staging_dir, synthetic_payload({"dictator": 2}))
        rosters = load_extension(path)
        with pytest.raises(ValueError, match="rendered none of the runtime-loaded frames"):
            generate_prompt_rows(
                CAPPED_GAME_ID,
                "group-mix",
                split=SPLIT_EVAL,
                extra_eval_frames=rosters.extra_eval_frames_for(DICTATOR_GAME_ID),
            )

    def test_every_game_gets_the_family_its_row_builder_reads(self, staging_dir: Path) -> None:
        """The loader's game-to-family dispatch and the row builders' rosters must agree, per game.

        Checked over the whole registry rather than the four games named above, because the two
        dispatches live in different modules and a game added to one is what would fall through.
        """
        path = write_staging(
            staging_dir,
            synthetic_payload({"matrix": 2, "dictator": 2, "trust": 2, "trustee": 2}),
            caps={CAPPED_GAME_ID: ["synthetic-bench-0", "synthetic-bench-1"]},
        )
        rosters = load_extension(path)
        for game_id in RENDERABLE_GAME_IDS:
            frames = rosters.extra_eval_frames_for(game_id)
            if frames.is_empty():
                continue
            grading = EVAL_RENDER_GRADING_BY_GAME.get(game_id, EVAL_ONLY_GRADING)
            rows = generate_prompt_rows(
                game_id, grading, split=SPLIT_EVAL, extra_eval_frames=frames
            )
            plain = generate_prompt_rows(game_id, grading, split=SPLIT_EVAL)
            assert len(rows) > len(plain), game_id

    def test_a_training_split_handed_extension_frames_is_refused(self, staging_dir: Path) -> None:
        """Silent otherwise: every extension frame is eval-only, so a train split would drop the lot."""
        path = write_staging(
            staging_dir,
            synthetic_payload({"matrix": 1}),
            caps={CAPPED_GAME_ID: ["synthetic-bench-0"]},
        )
        rosters = load_extension(path)
        with pytest.raises(ValueError, match="held out"):
            generate_prompt_rows(
                CAPPED_GAME_ID,
                "group-mix",
                split=SPLIT_TRAIN,
                extra_eval_frames=rosters.extra_eval_frames_for(CAPPED_GAME_ID),
            )


class TestTheBatteryFlag:
    @staticmethod
    def _rosters(staging_dir: Path) -> ExtensionRosters:
        path = write_staging(
            staging_dir,
            synthetic_payload({"matrix": 2, "dictator": 2, "trust": 2, "trustee": 2}),
            caps={CAPPED_GAME_ID: ["synthetic-bench-0", "synthetic-bench-1"]},
        )
        return load_extension(path)

    def test_a_game_behaviour_cell_renders_more_prompts_with_the_extension(
        self, staging_dir: Path
    ) -> None:
        rosters = self._rosters(staging_dir)
        plain = EvalConfig(games=(CAPPED_GAME_ID,), include_never_trained=False)
        extended = EvalConfig(
            games=(CAPPED_GAME_ID,), include_never_trained=False, held_out_extension=rosters
        )
        plain_plan = plan_battery([SECTION_GAME_BEHAVIOR], plain)
        extended_plan = plan_battery([SECTION_GAME_BEHAVIOR], extended)
        assert len(extended_plan) > len(plain_plan)
        assert {request.identity for request in plain_plan} < {
            request.identity for request in extended_plan
        }

    def test_the_two_cells_have_distinct_identities(self, staging_dir: Path) -> None:
        """A cell with and without the extension must never share a bank entry or a resumed trace."""
        rosters = self._rosters(staging_dir)
        plain = EvalConfig(games=(CAPPED_GAME_ID,), include_never_trained=False)
        extended = EvalConfig(
            games=(CAPPED_GAME_ID,), include_never_trained=False, held_out_extension=rosters
        )
        assert plain.as_record() != extended.as_record()
        assert prompt_set_digest(plan_battery([SECTION_GAME_BEHAVIOR], plain)) != prompt_set_digest(
            plan_battery([SECTION_GAME_BEHAVIOR], extended)
        )
        assert extended.as_record()["held_out_extension"]["digest"] == rosters.digest

    def test_a_battery_without_the_game_behaviour_section_is_refused(
        self, staging_dir: Path, tmp_path: Path
    ) -> None:
        """Otherwise the trace records the extension's digest while rendering none of its frames."""
        rosters = self._rosters(staging_dir)
        with pytest.raises(ValueError, match="applies to that section only"):
            run_eval_battery(
                MockBackend(responses=["reasoning</think>nothing"]),
                sections=(SECTION_FRAMING_SWEEP,),
                out_path=tmp_path / "step-0.jsonl",
                meta={"arm": "unit", "step": 0},
                config=EvalConfig(
                    counterpart_framings=("twin",),
                    framing_sweep_games=(CAPPED_GAME_ID,),
                    held_out_extension=rosters,
                ),
            )

    def test_the_meta_audits_the_extension_frames_the_cell_rendered(
        self, staging_dir: Path, tmp_path: Path
    ) -> None:
        """`frame_label_audit` is joined onto every record by `reskin_id`, extension rows included.

        Without the extension's frames in it, the cells that carry the widest held-out bank are the
        only ones whose `label_print_order` residual has no covariate, and a reader joining on
        `reskin_id` gets nothing for exactly those rows.
        """
        rosters = self._rosters(staging_dir)
        out_path = tmp_path / "step-0.jsonl"
        run_eval_battery(
            MockBackend(responses=["reasoning</think><action>QUILL</action>"]),
            sections=(SECTION_GAME_BEHAVIOR,),
            out_path=out_path,
            meta={"arm": "unit", "step": 0},
            config=EvalConfig(
                games=(CAPPED_GAME_ID,),
                include_never_trained=False,
                game_behavior_samples=1,
                held_out_extension=rosters,
            ),
        )
        records = read_eval_records(out_path)
        audit = records[0]["frame_label_audit"]
        assert records[0]["record"] == RECORD_META
        rendered = {
            record["reskin_id"] for record in records if record["record"] == SECTION_GAME_BEHAVIOR
        }
        extension_ids = {scenario.scenario_id for scenario in rosters.matrix}
        assert extension_ids & rendered
        assert rendered <= set(audit)

    def test_the_framing_sweep_is_untouched_by_the_extension(self, staging_dir: Path) -> None:
        """Section 9.1's decision: the sweep's cells stay comparable to the banked ones."""
        rosters = self._rosters(staging_dir)
        plain = EvalConfig(counterpart_framings=("twin",), framing_sweep_games=(CAPPED_GAME_ID,))
        extended = EvalConfig(
            counterpart_framings=("twin",),
            framing_sweep_games=(CAPPED_GAME_ID,),
            held_out_extension=rosters,
        )
        plain_plan = plan_battery([SECTION_FRAMING_SWEEP], plain)
        extended_plan = plan_battery([SECTION_FRAMING_SWEEP], extended)
        assert [request.prompt for request in plain_plan] == [
            request.prompt for request in extended_plan
        ]
        assert [request.identity for request in plain_plan] == [
            request.identity for request in extended_plan
        ]


class TestTheRealStagingFile:
    """Pins on the file itself, run only where the gitignored authored material exists.

    A fresh clone skips these two; the synthetic suite above covers every mechanic either way.
    """

    @staticmethod
    def _real_rosters() -> ExtensionRosters:
        if not STAGING_PATH.is_file():
            pytest.skip(f"{STAGING_PATH} is not on this machine (fresh clone)")
        return load_extension(STAGING_PATH)

    def test_the_four_families_carry_the_inventory_counts(self) -> None:
        rosters = self._real_rosters()
        assert {
            "matrix": len(rosters.matrix),
            "dictator": len(rosters.dictator),
            "trust": len(rosters.trust),
            "trustee": len(rosters.trustee),
        } == REAL_FAMILY_COUNTS

    def test_the_capped_game_takes_nine_frames(self) -> None:
        rosters = self._real_rosters()
        assert len(rosters.matrix_for_game(CAPPED_GAME_ID)) == 9
