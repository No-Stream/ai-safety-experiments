"""Run the whole battery offline against scripted backends, and pin the trace's shape.

Offline and CPU-only: `MockBackend` serves scripted completions, so these tests exercise the real
prompt corpus, the real parsers, and the real JSONL writer without loading a model.

:class:`TestAScriptedBackendComesBackAsItself` is the plan's Phase 2 sabotage item, kept as a
permanent test: a backend scripted to answer FDT every time must show up as 100% FDT and nothing
else. An eval that quietly averaged, defaulted, or dropped unparsed answers would still produce a
plausible distribution, so the only way to know the pipeline reports what it saw is to feed it
something whose answer is known exactly.

:class:`TestTheTraceIsSelfDescribing` covers the provenance invariant. Every number this project
reports comes out of one of these files months after the GPU time was paid for, so a trace that
cannot say which model, checkpoint, and commit produced it is not a cheaper artefact -- it is an
unusable one.
"""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING, Any

import pytest

from games.chunked_decode import CPU_CHUNK_SEQUENCES
from games.evals import (
    CAPABILITY_INSTRUCTION,
    EVAL_ONLY_GRADING,
    EVAL_RENDER_GRADING_BY_GAME,
    RECORD_META,
    SECTION_CAPABILITIES,
    SECTION_DT_PROBES,
    SECTION_GAME_BEHAVIOR,
    SECTIONS,
    EvalConfig,
    read_eval_records,
    render_capability_prompt,
    run_eval_battery,
)
from games.framing_stimulus import load_dictator_recipient_clauses
from games.probes import (
    ORDER_AS_AUTHORED,
    ORDER_REVERSED,
    PROBE_MULTIPLE_CHOICE,
    our_battery,
    render_probe_prompt,
)
from games.prompts import (
    FRAMING_TWIN,
    SPLIT_EVAL,
    TEMPTATION_DOSE_GAME_ID,
    TEMPTATION_DOSE_PAYOFF_VARIANTS,
    generate_prompt_rows,
)
from games.survey import (
    AUTHORED_FILENAME,
    AUTHORED_ITEM_SPECS,
    ELICITATION_BLOCKS_KEY,
    FAMILIES_WITH_SHARED_ELICITATION,
    PUBLISHED_INSTRUMENTS,
    SCHEMA_VERSION,
    SURVEY_ALLOCATION,
    SURVEY_CHEAP_TALK,
    SURVEY_CHOICE,
    SURVEY_LIKERT,
    SURVEY_ORDERED_CHOICE,
    SURVEY_TAGGED,
)
from games.tests.test_games_trap_cells import write_framings_file
from grpo.rlvr_math import gen_ltr_arithmetic
from reward_hacking.model_backend import MockBackend, SamplingConfig


def _synthetic_payoffs(n_options: int, position: int = 1) -> list[list[int]]:
    """Build allocation options that separate the three orientations, for any count of three or more.

    Shared by the published and the AUTHORED half of the fixture below: an authored allocation item
    supplies its own payoff table, and a fixture that emitted one for only one of the two halves is
    what turned this file red when the counterpart allocation arms landed.
    """
    if n_options == 3:
        return [[97 + position, 41], [88 + position, 88 + position], [93 + position, 23]]
    step = 60 // (n_options - 1)
    return [[80, 80 - index * step] for index in range(n_options)]


def synthetic_survey_data_dir(directory: Path) -> Path:
    """Write both local survey item files with placeholder text, in the real schemas.

    A compact double of the builders in `test_games_survey_section.py` (the test tree is not a
    package, so helpers do not import across files). Only the all-sections test needs it here.
    """
    items: dict[str, dict[str, object]] = {}
    for spec in AUTHORED_ITEM_SPECS:
        block: dict[str, object] = {"stem": f"Placeholder stem for {spec.item_id}."}
        if spec.requires_swapped_stem:
            block["stem_swapped"] = f"Placeholder swapped stem for {spec.item_id}."
        if spec.kind in (SURVEY_LIKERT, SURVEY_CHOICE, SURVEY_ORDERED_CHOICE):
            block["options"] = [
                f"placeholder option {index}" for index in range(1, spec.n_options + 1)
            ]
        if spec.kind in (SURVEY_TAGGED, SURVEY_CHEAP_TALK):
            block["vocabulary"] = [f"wordnumber{index}" for index in range(1, spec.n_tag_words + 1)]
        if spec.kind == SURVEY_ALLOCATION:
            block["option_payoffs"] = _synthetic_payoffs(spec.n_options)
        items[spec.item_id] = block
    instruments: dict[str, dict[str, object]] = {}
    for name, published in PUBLISHED_INSTRUMENTS.items():
        if published.kind == SURVEY_LIKERT:
            instruments[name] = {
                "anchors": [
                    f"placeholder anchor {point}" for point in range(1, published.scale_points + 1)
                ],
                "items": [
                    {"stem": f"Placeholder statement {position} for {name}."}
                    for position in range(1, published.n_items + 1)
                ],
            }
        else:
            instruments[name] = {
                "instructions": "Placeholder framing for an allocation task.",
                "items": [
                    {"option_payoffs": _synthetic_payoffs(published.n_options, position)}
                    for position in range(1, published.n_items + 1)
                ],
            }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / AUTHORED_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "items": items,
                ELICITATION_BLOCKS_KEY: {
                    family: f"Placeholder closing question for {family}."
                    for family in FAMILIES_WITH_SHARED_ELICITATION
                },
            }
        ),
        encoding="utf-8",
    )
    (directory / "published.json").write_text(
        json.dumps({"schema_version": SCHEMA_VERSION, "instruments": instruments}), encoding="utf-8"
    )
    return directory


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

BASE_META: dict[str, Any] = {"arm": "twin-pd-group", "step": 0, "model_path": "base-model"}

SMALL_CONFIG = EvalConfig(
    open_ended_samples=2,
    capability_items=4,
    batch_size=8,
    games=("twin-pd",),
    trained_game_ids=("twin-pd",),
    include_never_trained=False,
)


class LocalDecodeBackend:
    """A stand-in for `HFBackend`: same transport, so the battery sizes it like a local decoder.

    Records the width of every generation call, which is the only way to see whether the battery
    derived its chunk size or fell back to a hardcoded one.
    """

    transport = "hf"
    model_id = "Qwen/Qwen3.5-2B"

    def __init__(self, completion: str = "reasoning</think>FINAL ANSWER: A") -> None:
        """Carry a real sampling config, since that is where the token budget is read from."""
        self.sampling = SamplingConfig.for_thinking(thinking=True)
        self.widths: list[int] = []
        self._completion = completion

    def generate(self, prompts: list[str]) -> list[str]:
        """Return one completion per prompt, remembering how many arrived at once."""
        self.widths.append(len(prompts))
        return [self._completion] * len(prompts)


UNPARSEABLE_COMPLETION = "I would rather not answer."
ALL_FDT_COMPLETION = "reasoning</think><theory>functional decision theory</theory>\nFINAL ANSWER: A"


class CountingBackend:
    """A backend that returns the wrong number of completions, to prove the guard fires."""

    model_id = "counting"
    transport = "stub"

    def generate(self, prompts: list[str]) -> list[str]:
        """Drop one completion, which would silently misalign every record after it."""
        return ["<action>HOLD</action>"] * max(0, len(prompts) - 1)


def _rows_for(games: tuple[str, ...]) -> list[dict[str, Any]]:
    """Regenerate the very rows the eval will use, so a scripted answer can be prompt-keyed.

    Eval-only games render under `EVAL_ONLY_GRADING`, mirroring `_plan_never_trained`; the
    grading map's keys are exactly the trainable roster.
    """
    return [
        row
        for game_id in games
        for row in generate_prompt_rows(
            game_id,
            EVAL_RENDER_GRADING_BY_GAME.get(game_id, EVAL_ONLY_GRADING),
            split=SPLIT_EVAL,
        )
    ]


def cooperative_backend(games: tuple[str, ...], *, repeat: int = 1) -> MockBackend:
    """A backend that always plays the cooperative action, whichever label carries it.

    Labels are counterbalanced per scenario, so a fixed string like "HOLD" would parse on some
    frames and fail on others. Looking the label up by prompt is what makes "always cooperates"
    expressible at all.
    """
    coop_by_prompt = {row["prompt"]: row["coop_label"] for row in _rows_for(games)}

    def respond(prompt: str) -> str:
        label = coop_by_prompt.get(prompt)
        if label is None:
            return f"reasoning</think>{UNPARSEABLE_COMPLETION}"
        return "reasoning</think>" + f"<action>{label}</action>" * repeat

    return MockBackend(responses=respond, model_id="mock-cooperator")


def unequally_parsing_backend(game_id: str) -> MockBackend:
    """Cooperate on every draw of every prompt, except one prompt that answers once and then refuses.

    Built to discriminate the two ways a behaviour rate can be reduced. Where every prompt
    contributes the same number of parsed draws, the mean of per-prompt means and the mean over
    records are arithmetically identical, so a uniformly-parsing backend cannot tell them apart. This
    one gives its first prompt exactly one parsed draw, naming the NON-cooperative label, while every
    other prompt parses all of its draws cooperatively -- so the per-prompt reduction counts that
    prompt as one zero against its neighbours' ones, and pooling the records nearly buries it.

    Stateful because all draws of one prompt are the same string: the draw index is the call count,
    which works because `_generate` sends the repeated prompts row-major in one list.
    """
    rows = _rows_for((game_id,))
    coop_by_prompt = {row["prompt"]: row["coop_label"] for row in rows}
    thin = rows[0]
    other_label = thin["label_a"] if thin["coop_label"] == thin["label_b"] else thin["label_b"]
    draws_seen: dict[str, int] = {}

    def respond(prompt: str) -> str:
        draw = draws_seen.get(prompt, 0)
        draws_seen[prompt] = draw + 1
        label = coop_by_prompt.get(prompt)
        if label is None:
            return f"reasoning</think>{UNPARSEABLE_COMPLETION}"
        if prompt == thin["prompt"]:
            if draw > 0:
                return f"reasoning</think>{UNPARSEABLE_COMPLETION}"
            return f"reasoning</think><action>{other_label}</action>"
        return f"reasoning</think><action>{label}</action>"

    return MockBackend(responses=respond, model_id="mock-unequal-draws")


def keep_everything_backend() -> MockBackend:
    """A dictator-game backend that keeps the whole endowment, whatever the endowment is."""
    endowment_by_prompt = {row["prompt"]: int(row["endowment"]) for row in _rows_for(("dictator",))}

    def respond(prompt: str) -> str:
        endowment = endowment_by_prompt.get(prompt)
        if endowment is None:
            return f"reasoning</think>{UNPARSEABLE_COMPLETION}"
        return f"reasoning</think><keep>{endowment}</keep>"

    return MockBackend(responses=respond, model_id="mock-keeper")


def arithmetic_answer_backend(config: EvalConfig) -> MockBackend:
    """A backend that answers the seed-pinned arithmetic canary correctly, and nothing else.

    Built by regenerating the same seeded item list the eval will use, which is what makes the
    canary comparable across checkpoints in the first place.
    """
    expected = {
        render_capability_prompt(problem): answer
        for problem, answer in gen_ltr_arithmetic(
            n=config.capability_items, seed=config.capability_seed
        )
    }
    return MockBackend(
        responses=lambda prompt: f"reasoning</think>{expected.get(prompt, 'no answer')}",
        model_id="arithmetic-oracle",
    )


class TestTheTraceIsSelfDescribing:
    def test_the_first_record_is_the_meta_record(self, tmp_path: Path) -> None:
        backend = cooperative_backend(("twin-pd",))
        out_path = tmp_path / "eval.jsonl"
        run_eval_battery(
            backend,
            sections=[SECTION_GAME_BEHAVIOR],
            out_path=out_path,
            meta=BASE_META,
            config=SMALL_CONFIG,
        )
        records = read_eval_records(out_path)
        assert records[0]["record"] == RECORD_META
        assert records[0]["backend_model_id"] == "mock-cooperator"
        assert records[0]["sections"] == [SECTION_GAME_BEHAVIOR]
        assert records[0]["arm"] == "twin-pd-group"
        assert records[0]["git_sha"]
        assert records[0]["eval_config"]["games"] == ["twin-pd"]

    def test_the_meta_carries_the_frame_label_audit_when_behaviour_ran(
        self, tmp_path: Path
    ) -> None:
        """The covariate for `label_print_order`'s residual has to be IN the trace, not derivable.

        Every record carries the order its outcome table printed, but the authored prose's own mention
        order is fixed under both orders by design, so a position effect cannot be attributed without
        knowing which label each frame introduces first. Recomputing that months later means the
        analysis depends on the frame roster not having changed since; recording it means it does not.
        """
        out_path = tmp_path / "eval.jsonl"
        run_eval_battery(
            cooperative_backend(("twin-pd",)),
            sections=[SECTION_GAME_BEHAVIOR],
            out_path=out_path,
            meta=BASE_META,
            config=SMALL_CONFIG,
        )
        records = read_eval_records(out_path)
        audit = records[0]["frame_label_audit"]
        rendered = {
            record["reskin_id"] for record in records if record["record"] == SECTION_GAME_BEHAVIOR
        }
        assert rendered
        # Joinable on reskin_id, which is how the covariate reaches a per-record analysis at all.
        assert rendered <= set(audit)
        assert all(entry["first_label_in_frame"] for entry in audit.values())

    def test_a_battery_without_the_behaviour_section_carries_no_frame_audit(
        self, tmp_path: Path
    ) -> None:
        """A covariate for a section that did not run would read as coverage the trace does not have."""
        out_path = tmp_path / "eval.jsonl"
        run_eval_battery(
            MockBackend(responses=[ALL_FDT_COMPLETION]),
            sections=[SECTION_DT_PROBES],
            out_path=out_path,
            meta=BASE_META,
            config=SMALL_CONFIG,
        )
        assert "frame_label_audit" not in read_eval_records(out_path)[0]

    def test_the_caller_may_not_overwrite_provenance_fields(self, tmp_path: Path) -> None:
        backend = cooperative_backend(("twin-pd",))
        with pytest.raises(ValueError, match="meta may not set"):
            run_eval_battery(
                backend,
                sections=[SECTION_GAME_BEHAVIOR],
                out_path=tmp_path / "eval.jsonl",
                meta={**BASE_META, "git_sha": "pretend-this-is-the-sha"},
                config=SMALL_CONFIG,
            )

    def test_reading_a_trace_without_a_meta_record_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "headless.jsonl"
        path.write_text('{"record": "game-behavior", "parsed": true}\n')
        with pytest.raises(ValueError, match="does not start with"):
            read_eval_records(path)

    def test_reading_an_empty_trace_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        with pytest.raises(ValueError, match="is no eval trace"):
            read_eval_records(path)


class TestAScriptedBackendComesBackAsItself:
    def test_an_all_fdt_backend_reports_only_fdt(self, tmp_path: Path) -> None:
        backend = MockBackend(responses=[ALL_FDT_COMPLETION], model_id="mock-fdt")
        out_path = tmp_path / "eval.jsonl"
        summary = run_eval_battery(
            backend,
            sections=[SECTION_DT_PROBES],
            out_path=out_path,
            meta=BASE_META,
            config=SMALL_CONFIG,
        )
        counts = summary[SECTION_DT_PROBES]["theory_counts"]
        assert set(counts) == {"FDT"}
        assert summary[SECTION_DT_PROBES]["parse_failure_rate"] == pytest.approx(0.0)

    def test_every_open_ended_record_carries_the_theory_and_its_completion(
        self, tmp_path: Path
    ) -> None:
        backend = MockBackend(responses=[ALL_FDT_COMPLETION], model_id="mock-fdt")
        out_path = tmp_path / "eval.jsonl"
        run_eval_battery(
            backend,
            sections=[SECTION_DT_PROBES],
            out_path=out_path,
            meta=BASE_META,
            config=SMALL_CONFIG,
        )
        probes = [
            record
            for record in read_eval_records(out_path)
            if record["record"] == SECTION_DT_PROBES and record["kind"] == "open-ended"
        ]
        assert probes
        assert {record["theory"] for record in probes} == {"FDT"}
        assert all(record["completion"] == ALL_FDT_COMPLETION for record in probes)

    def test_open_ended_items_are_sampled_the_requested_number_of_times(
        self, tmp_path: Path
    ) -> None:
        backend = MockBackend(responses=[ALL_FDT_COMPLETION])
        out_path = tmp_path / "eval.jsonl"
        run_eval_battery(
            backend,
            sections=[SECTION_DT_PROBES],
            out_path=out_path,
            meta=BASE_META,
            config=EvalConfig(open_ended_samples=3, batch_size=8),
        )
        probes = [
            record
            for record in read_eval_records(out_path)
            if record["record"] == SECTION_DT_PROBES and record["kind"] == "open-ended"
        ]
        per_probe = {record["probe_id"] for record in probes}
        assert len(probes) == 3 * len(per_probe)
        assert {record["sample_index"] for record in probes} == {0, 1, 2}

    def test_an_unparseable_backend_shows_up_as_parse_failure_not_as_a_theory(
        self, tmp_path: Path
    ) -> None:
        backend = MockBackend(responses=[UNPARSEABLE_COMPLETION])
        summary = run_eval_battery(
            backend,
            sections=[SECTION_DT_PROBES],
            out_path=tmp_path / "eval.jsonl",
            meta=BASE_META,
            config=SMALL_CONFIG,
        )
        assert summary[SECTION_DT_PROBES]["parse_failure_rate"] == pytest.approx(1.0)
        assert summary[SECTION_DT_PROBES]["theory_counts"] == {}


class TestGameBehaviorSection:
    def test_a_cooperative_backend_reports_a_coop_rate_of_one(self, tmp_path: Path) -> None:
        backend = cooperative_backend(("twin-pd",))
        summary = run_eval_battery(
            backend,
            sections=[SECTION_GAME_BEHAVIOR],
            out_path=tmp_path / "eval.jsonl",
            meta=BASE_META,
            config=SMALL_CONFIG,
        )
        rates = summary[SECTION_GAME_BEHAVIOR]["coop_rate_by_game"]
        assert set(rates) == {"twin-pd"}
        assert rates["twin-pd"]["rate"] == pytest.approx(1.0)
        assert rates["twin-pd"]["n_parsed"] == rates["twin-pd"]["n_asked"] > 0

    def test_each_prompt_is_drawn_the_requested_number_of_times(self, tmp_path: Path) -> None:
        """One draw per prompt resolves a per-prompt rate to 0/1 or 1/1 and nothing between."""
        backend = cooperative_backend(("twin-pd",))
        out_path = tmp_path / "eval.jsonl"
        run_eval_battery(
            backend,
            sections=[SECTION_GAME_BEHAVIOR],
            out_path=out_path,
            meta=BASE_META,
            config=dataclasses.replace(SMALL_CONFIG, game_behavior_samples=3),
        )
        games = [
            record
            for record in read_eval_records(out_path)
            if record["record"] == SECTION_GAME_BEHAVIOR
        ]
        assert games
        by_prompt: dict[str, list[int]] = {}
        for record in games:
            by_prompt.setdefault(str(record["prompt_id"]), []).append(int(record["sample_index"]))
        assert by_prompt
        assert all(sorted(draws) == [0, 1, 2] for draws in by_prompt.values()), by_prompt
        assert len(games) == 3 * len(by_prompt)

    def test_the_rate_is_a_mean_over_prompts_not_over_draws(self, tmp_path: Path) -> None:
        """Draws of one prompt are one observation, so a prompt that mostly failed cannot be diluted.

        The battery's own summary has to agree with the readouts on this, since both recompute the
        same quantity. Discriminating the two reductions needs prompts with unequal parsed draw
        counts, which is what `unequally_parsing_backend` produces: with every prompt contributing the
        same number the two averages are arithmetically identical and the test would pass either way.
        """
        out_path = tmp_path / "eval.jsonl"
        summary = run_eval_battery(
            unequally_parsing_backend("twin-pd"),
            sections=[SECTION_GAME_BEHAVIOR],
            out_path=out_path,
            meta=BASE_META,
            config=dataclasses.replace(SMALL_CONFIG, game_behavior_samples=8),
        )
        by_prompt: dict[str, list[float]] = {}
        n_records = 0
        for record in read_eval_records(out_path):
            if record["record"] != SECTION_GAME_BEHAVIOR:
                continue
            n_records += 1
            if record["coop_fraction"] is not None:
                by_prompt.setdefault(str(record["prompt_id"]), []).append(
                    float(record["coop_fraction"])
                )
        per_prompt = [sum(values) / len(values) for values in by_prompt.values()]
        pooled = [value for values in by_prompt.values() for value in values]
        assert len(pooled) > len(per_prompt)
        assert sum(per_prompt) / len(per_prompt) != pytest.approx(sum(pooled) / len(pooled))
        rate = summary[SECTION_GAME_BEHAVIOR]["coop_rate_by_game"]["twin-pd"]
        assert rate["rate"] == pytest.approx(sum(per_prompt) / len(per_prompt))
        assert rate["n_parsed"] == rate["n_asked"] == len(by_prompt)
        assert rate["n_records_parsed"] == len(pooled)
        assert rate["n_records"] == n_records

    def test_a_game_that_parsed_nothing_reads_as_zero_over_its_denominator(
        self, tmp_path: Path
    ) -> None:
        """A rate the summary omits is invisible; a rate over nothing is worse than invisible.

        The summary is what a readout quotes first, so a game whose completions never parsed has to
        appear there saying so -- `rate` absent, `n_parsed` zero, `n_asked` the count that was
        actually asked. Under the bare per-game mean this section used to emit, such a game was
        silently dropped from the map entirely, which reads as a battery that never asked it.
        """
        backend = cooperative_backend(("twin-pd",))
        summary = run_eval_battery(
            backend,
            sections=[SECTION_GAME_BEHAVIOR],
            out_path=tmp_path / "eval.jsonl",
            meta=BASE_META,
            config=EvalConfig(
                batch_size=8,
                games=("twin-pd", "chicken"),
                trained_game_ids=("twin-pd",),
                include_never_trained=False,
            ),
        )
        rates = summary[SECTION_GAME_BEHAVIOR]["coop_rate_by_game"]
        assert set(rates) == {"twin-pd", "chicken"}
        assert rates["chicken"]["rate"] is None
        assert rates["chicken"]["n_parsed"] == 0
        assert rates["chicken"]["n_asked"] > 0
        assert rates["twin-pd"]["rate"] == pytest.approx(1.0)

    def test_records_carry_the_identity_needed_to_line_up_before_and_after(
        self, tmp_path: Path
    ) -> None:
        backend = cooperative_backend(("twin-pd",))
        out_path = tmp_path / "eval.jsonl"
        run_eval_battery(
            backend,
            sections=[SECTION_GAME_BEHAVIOR],
            out_path=out_path,
            meta=BASE_META,
            config=SMALL_CONFIG,
        )
        games = [
            record
            for record in read_eval_records(out_path)
            if record["record"] == SECTION_GAME_BEHAVIOR
        ]
        assert games
        for record in games:
            for key in ("prompt_id", "reskin_id", "payoff_variant", "label_a", "coop_label"):
                assert record[key] != "" or key == "payoff_variant"
            assert record["trained_game"] is True

    def test_the_never_trained_games_are_marked_as_such(self, tmp_path: Path) -> None:
        """Transfer to a game with no training arm is the cross-game generalisation readout."""
        backend = cooperative_backend(("twin-pd",))
        out_path = tmp_path / "eval.jsonl"
        run_eval_battery(
            backend,
            sections=[SECTION_GAME_BEHAVIOR],
            out_path=out_path,
            meta=BASE_META,
            config=EvalConfig(
                batch_size=8,
                games=("twin-pd",),
                trained_game_ids=("twin-pd",),
                include_never_trained=True,
            ),
        )
        eval_only = {
            record["game_id"]
            for record in read_eval_records(out_path)
            if record["record"] == SECTION_GAME_BEHAVIOR and record["eval_only_game"]
        }
        # trustee-return-rule joined the set on 2026-08-21: it asks for the same return share the
        # trust strategy method trains, in the role where paying it comes out of what this side
        # holds, so it is the price check on that arm's promise and must never have a training split.
        # twin-pd-temptation-dose joined on 2026-08-26: naming only trainable games in `games`
        # keeps the WHOLE eval-only roster riding along, dose ladder included.
        # public-goods LEFT on 2026-08-26, promoted to trainable for the transfer-of-learning
        # pair, so batteries run after that date read its transfer through the framing sweep or by
        # naming it in `games`, not through this rider.
        assert eval_only == {
            "ultimatum-responder",
            "defective-coordination",
            "defective-harmony",
            "trustee-return-rule",
            TEMPTATION_DOSE_GAME_ID,
        }

    def test_naming_an_eval_only_game_plays_exactly_that_game(self, tmp_path: Path) -> None:
        """An explicitly named eval-only game narrows the never-trained leg to the named ones.

        The dose ladder is the motivating case: a battery asked for one dose instrument must not
        bill the whole transfer roster alongside it, and before eval-only ids were nameable this
        spelling was an "Unknown game ids" error, so no earlier launch changes meaning.
        """
        backend = cooperative_backend((TEMPTATION_DOSE_GAME_ID,))
        out_path = tmp_path / "eval.jsonl"
        run_eval_battery(
            backend,
            sections=[SECTION_GAME_BEHAVIOR],
            out_path=out_path,
            meta=BASE_META,
            config=EvalConfig(
                batch_size=8,
                games=(TEMPTATION_DOSE_GAME_ID,),
                include_never_trained=True,
            ),
        )
        records = [
            record
            for record in read_eval_records(out_path)
            if record["record"] == SECTION_GAME_BEHAVIOR
        ]
        assert {record["game_id"] for record in records} == {TEMPTATION_DOSE_GAME_ID}
        assert {record["payoff_variant"] for record in records} == set(
            TEMPTATION_DOSE_PAYOFF_VARIANTS
        )
        for record in records:
            assert record["eval_only_game"] is True
            assert record["trained_game"] is False

    def test_naming_an_eval_only_game_with_the_transfer_leg_off_raises(self) -> None:
        """The combination would silently play nothing of what it named, so it refuses."""
        with pytest.raises(ValueError, match="include_never_trained"):
            EvalConfig(games=(TEMPTATION_DOSE_GAME_ID,), include_never_trained=False)


class TestTheTrainedGameColumnNamesTheArmsOwnGame:
    """An arm trains on exactly one game, so most of the cross-game grid is transfer, not training.

    `games/report.py` groups its action-rate table on `(game_id, trained_game)` and calls that the
    never-trained marker, so stamping True on every registered game put several transfer
    measurements in the "trained" bucket of the project's headline table.
    """

    def test_only_the_arms_own_game_is_marked_trained(self, tmp_path: Path) -> None:
        backend = cooperative_backend(("twin-pd", "chicken"))
        out_path = tmp_path / "eval.jsonl"
        run_eval_battery(
            backend,
            sections=[SECTION_GAME_BEHAVIOR],
            out_path=out_path,
            meta=BASE_META,
            config=EvalConfig(
                batch_size=8,
                games=("twin-pd", "chicken"),
                trained_game_ids=("twin-pd",),
                include_never_trained=False,
            ),
        )
        marked: dict[str, set[bool]] = {}
        for record in read_eval_records(out_path):
            if record["record"] != SECTION_GAME_BEHAVIOR:
                continue
            marked.setdefault(str(record["game_id"]), set()).add(bool(record["trained_game"]))
        assert marked == {"twin-pd": {True}, "chicken": {False}}

    def test_a_checkpointless_eval_marks_nothing_as_trained(self, tmp_path: Path) -> None:
        """The un-adapted base model trained on no game, so no column may claim it did."""
        backend = cooperative_backend(("twin-pd",))
        out_path = tmp_path / "eval.jsonl"
        run_eval_battery(
            backend,
            sections=[SECTION_GAME_BEHAVIOR],
            out_path=out_path,
            meta=BASE_META,
            config=EvalConfig(batch_size=8, games=("twin-pd",), include_never_trained=False),
        )
        games = [
            record
            for record in read_eval_records(out_path)
            if record["record"] == SECTION_GAME_BEHAVIOR
        ]
        assert games
        assert not any(record["trained_game"] for record in games)

    def test_a_trained_game_outside_the_registry_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown trained game ids"):
            EvalConfig(trained_game_ids=("hopscotch",))

    def test_a_corpus_built_trained_game_is_accepted(self) -> None:
        """pd-track-record trains from a built corpus, so it never enters the roster's GAME_IDS.

        Its run_config.json still records it as the trained game, and the eval battery's job is to
        mark the transfer grid against it -- a config that refused it would fail every eval of
        that arm's checkpoints on a rented card, after training had already been paid for (caught
        by the mock rehearsal on 2026-08-30, one mock gate before it would have).
        """
        config = EvalConfig(trained_game_ids=("pd-track-record",))
        assert config.trained_game_ids == ("pd-track-record",)

    def test_the_grading_column_names_what_rendered_the_rows(self, tmp_path: Path) -> None:
        """Two arms share `twin-pd` under different gradings, so this column cannot name the arm."""
        backend = cooperative_backend(("twin-pd",))
        out_path = tmp_path / "eval.jsonl"
        run_eval_battery(
            backend,
            sections=[SECTION_GAME_BEHAVIOR],
            out_path=out_path,
            meta={**BASE_META, "arm": "twin-pd-self"},
            config=SMALL_CONFIG,
        )
        games = [
            record
            for record in read_eval_records(out_path)
            if record["record"] == SECTION_GAME_BEHAVIOR
        ]
        assert games
        assert all(record["render_grading"] == "group-mix" for record in games)
        assert all("grading" not in record for record in games)

    def test_the_iterated_game_records_a_move_per_round(self, tmp_path: Path) -> None:
        backend = cooperative_backend(("iterated-pd-tft",), repeat=5)
        out_path = tmp_path / "eval.jsonl"
        run_eval_battery(
            backend,
            sections=[SECTION_GAME_BEHAVIOR],
            out_path=out_path,
            meta=BASE_META,
            config=EvalConfig(
                batch_size=8, games=("iterated-pd-tft",), include_never_trained=False
            ),
        )
        iterated = [
            record
            for record in read_eval_records(out_path)
            if record["record"] == SECTION_GAME_BEHAVIOR
        ]
        assert iterated
        assert all(record["moves"] == ["C"] * 5 for record in iterated)
        assert all(record["coop_fraction"] == pytest.approx(1.0) for record in iterated)

    def test_the_dictator_game_records_the_split(self, tmp_path: Path) -> None:
        backend = keep_everything_backend()
        out_path = tmp_path / "eval.jsonl"
        summary = run_eval_battery(
            backend,
            sections=[SECTION_GAME_BEHAVIOR],
            out_path=out_path,
            meta=BASE_META,
            config=EvalConfig(batch_size=8, games=("dictator",), include_never_trained=False),
        )
        records = [
            record
            for record in read_eval_records(out_path)
            if record["record"] == SECTION_GAME_BEHAVIOR
        ]
        assert records
        assert all(record["keep_fraction"] == pytest.approx(1.0) for record in records)
        keep = summary[SECTION_GAME_BEHAVIOR]["behaviour_rate_by_measure"]["keep_fraction"]
        assert keep["rate"] == pytest.approx(1.0)
        assert keep["n_parsed"] == keep["n_asked"] == len(records)
        # A figure game has no cooperation rate at all, so it must not appear as one that failed.
        assert summary[SECTION_GAME_BEHAVIOR]["coop_rate_by_game"] == {}


class TestCapabilitiesCanary:
    def test_a_correct_backend_scores_one(self, tmp_path: Path) -> None:
        config = EvalConfig(capability_items=6, batch_size=8)
        summary = run_eval_battery(
            arithmetic_answer_backend(config),
            sections=[SECTION_CAPABILITIES],
            out_path=tmp_path / "eval.jsonl",
            meta=BASE_META,
            config=config,
        )
        assert summary[SECTION_CAPABILITIES]["accuracy"] == pytest.approx(1.0)
        assert summary[SECTION_CAPABILITIES]["n_records"] == 6

    def test_a_wrong_backend_scores_zero(self, tmp_path: Path) -> None:
        backend = MockBackend(responses=["reasoning</think>-99999999"])
        summary = run_eval_battery(
            backend,
            sections=[SECTION_CAPABILITIES],
            out_path=tmp_path / "eval.jsonl",
            meta=BASE_META,
            config=EvalConfig(capability_items=4, batch_size=8),
        )
        assert summary[SECTION_CAPABILITIES]["accuracy"] == pytest.approx(0.0)

    def test_a_number_inside_the_thinking_block_is_not_scored_as_the_answer(
        self, tmp_path: Path
    ) -> None:
        """parse_answer takes the last integer anywhere, so the block has to be cut first."""
        config = EvalConfig(capability_items=4, batch_size=8)
        expected = {
            render_capability_prompt(problem): answer
            for problem, answer in gen_ltr_arithmetic(
                n=config.capability_items, seed=config.capability_seed
            )
        }
        backend = MockBackend(
            responses=lambda prompt: f"let me try 123456789</think>{expected.get(prompt, 0)}"
        )
        summary = run_eval_battery(
            backend,
            sections=[SECTION_CAPABILITIES],
            out_path=tmp_path / "eval.jsonl",
            meta=BASE_META,
            config=config,
        )
        assert summary[SECTION_CAPABILITIES]["accuracy"] == pytest.approx(1.0)


class TestSectionAndConfigValidation:
    def test_every_registered_section_runs_together(self, tmp_path: Path) -> None:
        backend = MockBackend(responses=[UNPARSEABLE_COMPLETION, ALL_FDT_COMPLETION])
        trap_cells_path = write_framings_file(tmp_path)
        summary = run_eval_battery(
            backend,
            sections=list(SECTIONS),
            out_path=tmp_path / "eval.jsonl",
            meta=BASE_META,
            config=dataclasses.replace(
                SMALL_CONFIG,
                survey_data_dir=synthetic_survey_data_dir(tmp_path / "survey-data"),
                # The framing-sweep section refuses to run with no framings named, the way the
                # self-report section refuses without item data; one framing keeps this cheap.
                counterpart_framings=(FRAMING_TWIN,),
                # The trap-cells section refuses without its recipient paragraphs for the same
                # reason: they are authored stimulus living in a gitignored runtime file.
                trap_cells_file=trap_cells_path,
                trap_cells_digest=load_dictator_recipient_clauses(trap_cells_path).digest,
            ),
        )
        # The summary lifts the trace's attribution (git_sha, arm, step, ...) beside the sections,
        # so it is a pure function of the trace; every registered section must still be reduced.
        assert {key for key in summary if key in SECTIONS} == set(SECTIONS)
        assert all(summary[section]["n_records"] > 0 for section in SECTIONS)

    def test_an_empty_section_list_raises(self, tmp_path: Path) -> None:
        backend = cooperative_backend(("twin-pd",))
        with pytest.raises(ValueError, match="sections is empty"):
            run_eval_battery(backend, sections=[], out_path=tmp_path / "e.jsonl", meta=BASE_META)

    def test_an_unknown_section_raises(self, tmp_path: Path) -> None:
        backend = cooperative_backend(("twin-pd",))
        with pytest.raises(ValueError, match="Unknown eval sections"):
            run_eval_battery(
                backend, sections=["vibes"], out_path=tmp_path / "e.jsonl", meta=BASE_META
            )

    def test_a_repeated_section_raises(self, tmp_path: Path) -> None:
        backend = cooperative_backend(("twin-pd",))
        with pytest.raises(ValueError, match="Sections repeat"):
            run_eval_battery(
                backend,
                sections=[SECTION_GAME_BEHAVIOR, SECTION_GAME_BEHAVIOR],
                out_path=tmp_path / "e.jsonl",
                meta=BASE_META,
            )

    def test_a_backend_that_drops_a_completion_raises(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="completions for"):
            run_eval_battery(
                CountingBackend(),
                sections=[SECTION_GAME_BEHAVIOR],
                out_path=tmp_path / "e.jsonl",
                meta=BASE_META,
                config=SMALL_CONFIG,
            )

    @pytest.mark.parametrize(
        "build",
        [
            lambda: EvalConfig(open_ended_samples=0),
            lambda: EvalConfig(multiple_choice_samples=0),
            lambda: EvalConfig(batch_size=0),
            lambda: EvalConfig(capability_items=0),
        ],
        ids=["open_ended_samples", "multiple_choice_samples", "batch_size", "capability_items"],
    )
    def test_a_non_positive_count_raises(self, build: Callable[[], EvalConfig]) -> None:
        with pytest.raises(ValueError, match="must be at least 1"):
            build()

    def test_an_unknown_game_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown game ids"):
            EvalConfig(games=("hopscotch",))


class TestTheArithmeticCanaryAsksForTheFormatItParses:
    """`parse_answer` takes the last integer in the text, so the prompt has to ask for one.

    Every other section appends an explicit format instruction. This one passed the bare problem
    text, so a post-RL shift toward chattier answers -- "= -3563. (With normal precedence this
    would be 533.)" -- reads as the capability collapse the canary exists to detect.
    """

    def test_every_capability_prompt_carries_the_format_instruction(self, tmp_path: Path) -> None:
        config = EvalConfig(capability_items=4, batch_size=8)
        out_path = tmp_path / "eval.jsonl"
        run_eval_battery(
            arithmetic_answer_backend(config),
            sections=[SECTION_CAPABILITIES],
            out_path=out_path,
            meta=BASE_META,
            config=config,
        )
        capabilities = [
            record
            for record in read_eval_records(out_path)
            if record["record"] == SECTION_CAPABILITIES
        ]
        assert capabilities
        assert all(CAPABILITY_INSTRUCTION in record["prompt"] for record in capabilities)

    def test_the_summary_separates_a_parse_failure_from_a_wrong_answer(
        self, tmp_path: Path
    ) -> None:
        """Two ways to score zero, and the canary is useless if it cannot tell them apart."""
        wrong = MockBackend(responses=["reasoning</think>-99999999"])
        unparseable = MockBackend(responses=["reasoning</think>no digits here at all"])
        config = EvalConfig(capability_items=4, batch_size=8)
        wrong_summary = run_eval_battery(
            wrong,
            sections=[SECTION_CAPABILITIES],
            out_path=tmp_path / "wrong.jsonl",
            meta=BASE_META,
            config=config,
        )
        unparseable_summary = run_eval_battery(
            unparseable,
            sections=[SECTION_CAPABILITIES],
            out_path=tmp_path / "unparseable.jsonl",
            meta=BASE_META,
            config=config,
        )
        assert wrong_summary[SECTION_CAPABILITIES]["accuracy"] == pytest.approx(0.0)
        assert wrong_summary[SECTION_CAPABILITIES]["n_parsed"] == 4
        assert unparseable_summary[SECTION_CAPABILITIES]["accuracy"] == pytest.approx(0.0)
        assert unparseable_summary[SECTION_CAPABILITIES]["n_parsed"] == 0


class TestMultipleChoiceProbesAreOrderCounterbalanced:
    """Each choice item is asked under both option orders, and scored once per item across them.

    The games corpus already counterbalances label-to-action mapping for exactly this reason, and
    the probes carry the decision-theory headline. A backend that always answers "A" is maximally
    order-sensitive, so it is the sabotage case: without counterbalancing it reads as a confident
    decision-theory position, and with it the position cancels and the disagreement is visible.
    """

    ALWAYS_A = "reasoning</think>FINAL ANSWER: A"

    def _choice_records(self, tmp_path: Path) -> list[dict[str, Any]]:
        backend = MockBackend(responses=[self.ALWAYS_A], model_id="always-a")
        out_path = tmp_path / "eval.jsonl"
        run_eval_battery(
            backend,
            sections=[SECTION_DT_PROBES],
            out_path=out_path,
            meta=BASE_META,
            config=SMALL_CONFIG,
        )
        return [
            record
            for record in read_eval_records(out_path)
            if record["record"] == SECTION_DT_PROBES and record["kind"] == PROBE_MULTIPLE_CHOICE
        ]

    def test_every_choice_item_is_rendered_under_both_orders(self, tmp_path: Path) -> None:
        records = self._choice_records(tmp_path)
        assert records
        by_probe: dict[str, set[str]] = {}
        for record in records:
            by_probe.setdefault(str(record["probe_id"]), set()).add(
                str(record["option_order_name"])
            )
        assert by_probe
        assert all(orders == {ORDER_AS_AUTHORED, ORDER_REVERSED} for orders in by_probe.values()), (
            by_probe
        )

    def test_the_recorded_answer_is_the_canonical_option_not_the_presented_letter(
        self, tmp_path: Path
    ) -> None:
        records = self._choice_records(tmp_path)
        for record in records:
            order = list(record["option_order"])
            assert record["presented_answer_index"] == 0
            assert record["answer_index"] == order[0]

    def test_an_always_a_backend_is_reported_as_order_sensitive(self, tmp_path: Path) -> None:
        backend = MockBackend(responses=[self.ALWAYS_A], model_id="always-a")
        summary = run_eval_battery(
            backend,
            sections=[SECTION_DT_PROBES],
            out_path=tmp_path / "eval.jsonl",
            meta=BASE_META,
            config=SMALL_CONFIG,
        )
        assert summary[SECTION_DT_PROBES]["order_disagreement_rate"] == pytest.approx(1.0)

    def test_a_pair_whose_second_order_did_not_parse_is_not_counted_as_agreement(
        self, tmp_path: Path
    ) -> None:
        """A parse failure is a missing observation, not evidence that the order did not matter.

        Keyed on the answers alone, an item with one unparsed render looks like a single-valued set
        and lands in the denominator as an agreement, which drags the disagreement rate down for a
        reason that has nothing to do with option order.
        """
        item = next(item for item in our_battery() if item.probe_id == "medical-newcomb")
        withheld = render_probe_prompt(item, option_order=(1, 0))
        backend = MockBackend(
            responses=lambda prompt: (
                "reasoning</think>I would rather not say." if prompt == withheld else self.ALWAYS_A
            ),
            model_id="always-a-except-one-order",
        )
        summary = run_eval_battery(
            backend,
            sections=[SECTION_DT_PROBES],
            out_path=tmp_path / "eval.jsonl",
            meta=BASE_META,
            config=SMALL_CONFIG,
        )
        assert summary[SECTION_DT_PROBES]["order_disagreement_rate"] == pytest.approx(1.0)

    def test_the_edt_leaning_denominator_is_items_not_renders(self, tmp_path: Path) -> None:
        """Two renders of one item are one observation; pooling them doubles n, not information.

        Discriminating the two requires an item whose orders are not both scoreable, because with
        every item contributing the same number of records the pooled mean and the mean of per-item
        means are arithmetically identical -- a test built on the plain always-"A" backend passes
        under either implementation, which is no test at all. So this backend refuses to answer
        `medical-newcomb` under the reversed order alone: that item then contributes one record
        while every other contributes two, and the two averages separate.
        """
        item = next(item for item in our_battery() if item.probe_id == "medical-newcomb")
        withheld = render_probe_prompt(item, option_order=(1, 0))
        backend = MockBackend(
            responses=lambda prompt: (
                "reasoning</think>I would rather not say." if prompt == withheld else self.ALWAYS_A
            ),
            model_id="always-a-except-one-order",
        )
        out_path = tmp_path / "eval.jsonl"
        summary = run_eval_battery(
            backend,
            sections=[SECTION_DT_PROBES],
            out_path=out_path,
            meta=BASE_META,
            config=SMALL_CONFIG,
        )
        leanings: dict[str, list[float]] = {}
        for record in read_eval_records(out_path):
            if record["record"] != SECTION_DT_PROBES:
                continue
            if record.get("edt_leaning") is not None:
                leanings.setdefault(str(record["probe_id"]), []).append(
                    float(record["edt_leaning"])
                )
        per_item = [sum(values) / len(values) for values in leanings.values()]
        pooled = [value for values in leanings.values() for value in values]
        assert summary[SECTION_DT_PROBES]["n_edt_scored_items"] == len(leanings)
        assert len(pooled) > len(per_item)
        assert sum(per_item) / len(per_item) != pytest.approx(sum(pooled) / len(pooled))
        assert summary[SECTION_DT_PROBES]["mean_edt_leaning"] == pytest.approx(
            sum(per_item) / len(per_item)
        )

    def test_the_edt_leaning_is_reported_per_source(self, tmp_path: Path) -> None:
        """Our items dilute DTBench's scalar in a known direction, so one pooled mean is not it."""
        backend = MockBackend(responses=[self.ALWAYS_A], model_id="always-a")
        summary = run_eval_battery(
            backend,
            sections=[SECTION_DT_PROBES],
            out_path=tmp_path / "eval.jsonl",
            meta=BASE_META,
            config=SMALL_CONFIG,
        )
        by_source = summary[SECTION_DT_PROBES]["mean_edt_leaning_by_source"]
        assert set(by_source) == {"ours"}
        assert by_source["ours"] == pytest.approx(0.0)

    def test_a_prosocial_choice_is_recorded_where_the_item_has_a_valence(
        self, tmp_path: Path
    ) -> None:
        """Without this a policy that merely became agreeable scores as a decision-theory shift."""
        records = self._choice_records(tmp_path)
        valenced = [record for record in records if record["prosocial_option"] is not None]
        assert valenced
        for record in records:
            if record["prosocial_option"] is None:
                assert record["chose_prosocial"] is None, record["probe_id"]
            else:
                assert record["chose_prosocial"] == (
                    record["answer_index"] == record["prosocial_option"]
                ), record["probe_id"]


class TestTheTemplateKwargsTrainingPinnedHaveToReachGeneration:
    """Training pins `reasoning_effort` when the template has that knob; eval must render the same.

    Qwen3.8-27B's template otherwise prepends an unauthored "Reasoning effort is set to xhigh"
    system message, so the arm would be evaluated on different prompt text than it trained on --
    the one thing the arms must not differ in. The local backends render the template themselves
    and take no such argument, so the battery refuses rather than measuring the wrong policy.
    """

    PINNED = (("reasoning_effort", "medium"),)

    def test_a_local_backend_that_cannot_apply_them_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="chat_template_kwargs"):
            run_eval_battery(
                LocalDecodeBackend(),
                sections=[SECTION_DT_PROBES],
                out_path=tmp_path / "eval.jsonl",
                meta=BASE_META,
                config=EvalConfig(
                    open_ended_samples=1, batch_size=8, chat_template_kwargs=self.PINNED
                ),
            )

    def test_a_run_with_nothing_to_pin_is_untouched(self, tmp_path: Path) -> None:
        """The knob exists only at 27B, so every smaller arm must run exactly as before."""
        summary = run_eval_battery(
            MockBackend(responses=[ALL_FDT_COMPLETION]),
            sections=[SECTION_DT_PROBES],
            out_path=tmp_path / "eval.jsonl",
            meta=BASE_META,
            config=SMALL_CONFIG,
        )
        assert summary[SECTION_DT_PROBES]["n_records"] > 0

    def test_the_pinned_kwargs_land_in_the_trace_meta(self, tmp_path: Path) -> None:
        out_path = tmp_path / "eval.jsonl"
        run_eval_battery(
            MockBackend(responses=[ALL_FDT_COMPLETION]),
            sections=[SECTION_DT_PROBES],
            out_path=out_path,
            meta=BASE_META,
            config=EvalConfig(open_ended_samples=1, batch_size=8, chat_template_kwargs=self.PINNED),
        )
        recorded = read_eval_records(out_path)[0]["eval_config"]["chat_template_kwargs"]
        assert recorded == {"reasoning_effort": "medium"}


class TestTheGenerationWidthIsDerivedNotHardcoded:
    """A hardcoded 16 at a 32,768-token budget is 524,288 in-flight tokens: exactly the knee.

    `games/select_prompts.py` already derives a chunk width from the VRAM actually present and
    narrows it on an OOM or an allocator thrash. The battery ran neither, on the repo's own
    "never hardcode a memory budget" rule.
    """

    def test_the_battery_has_no_default_batch_size(self) -> None:
        assert EvalConfig().batch_size is None

    def test_a_local_decoder_gets_a_derived_width(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import torch  # noqa: PLC0415

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        backend = LocalDecodeBackend()
        run_eval_battery(
            backend,
            sections=[SECTION_DT_PROBES],
            out_path=tmp_path / "eval.jsonl",
            meta=BASE_META,
            config=EvalConfig(open_ended_samples=1),
        )
        assert backend.widths
        assert max(backend.widths) == CPU_CHUNK_SEQUENCES

    def test_an_explicit_override_is_honoured(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import torch  # noqa: PLC0415

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        backend = LocalDecodeBackend()
        run_eval_battery(
            backend,
            sections=[SECTION_DT_PROBES],
            out_path=tmp_path / "eval.jsonl",
            meta=BASE_META,
            config=EvalConfig(open_ended_samples=1, batch_size=2),
        )
        assert max(backend.widths) == 2
