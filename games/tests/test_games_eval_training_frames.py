"""Offline tests for `games/eval_training_frames.py`, the training-corpus eval CLI.

Everything runs through `--backend mock` on CPU: real corpus loading, real target resolution
(shared with `games.run_evals`), the real trace writer and resume -- no model, no GPU. The fake
corpus mirrors the `games.select_prompts` artifact schema, because refusing anything that is not
that artifact is half of what this CLI adds.

Per the repo rule that a check you have never watched fail is not yet a check, every refusal is
exercised by committing the exact violation it exists to catch: a complete trace, a foreign file at
the out path, a partial or a complete trace of another cell (the latter under --sync-dest, where a
complete cell of THIS configuration is skipped), a summary appearing mid-run, a corpus from another
game, a corpus missing its label fields, a corpus mixing games or repeating a prompt. The
persistence contract the cell gained on 2026-09-02 is rehearsed the way the battery's is: a run
that dies at its second generate call and is relaunched must keep its finished records byte for
byte, generate exactly the rest, and say so in the meta; the summary must be a pure function of the
trace, pinned against a synthetic cell here, against a synthetic legacy-shaped one, and against the
banked track-record-v2 corpus-frames cell where a read-only copy of it is present.
"""

from __future__ import annotations

import itertools
import json
import shutil
import zlib
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import pytest

from games import eval_training_frames, evals, run_evals, vllm_teardown
from games.eval_training_frames import load_corpus_rows, plan_training_frames
from games.evals import (
    RECORD_IDENTITY_FIELDS,
    RECORD_META,
    RECORD_TRAINING_FRAMES,
    SECTION_GAME_BEHAVIOR,
    SECTIONS_TRAINING_FRAMES,
    SUBMISSION_POOLED,
    SUBMISSION_SERIAL,
    EvalConfig,
    plan_battery,
    read_eval_records,
    rebuild_summary,
    record_identity,
    run_eval_battery,
)
from games.s3_sync import SyncOutcome
from reward_hacking.model_backend import MockBackend

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from games.evals import PlannedRequest
    from reward_hacking.model_backend import Backend

BASE_MODEL = "Qwen/Qwen3.5-2B"
ARM = "twin-pd-group"

PARSEABLE_COOP_RESPONSE = "<think>weighing it</think> I commit to <action>SIDE</action>"

# A read-only copy of the banked track-record-v2 corpus-frames cell (144 prompts x 8 draws, written
# by the pre-resume writer), pulled from its S3 prefix by hand; the rebuild test skips without it.
BANKED_FRAMES_TRACE = Path("/var/tmp/w3-frames/step-70.jsonl")  # noqa: S108 - read-only fixture copy


def scripted_completion(prompt: str) -> str:
    """A parseable completion that depends on the prompt and nothing else, so runs compare byte for byte."""
    return f"<think>draw {zlib.crc32(prompt.encode('utf-8'))}</think> I commit to <action>SIDE</action>"


def corpus_row(**overrides: Any) -> dict[str, Any]:
    """One row in the exact schema `games.select_prompts` writes into a corpus artifact."""
    row: dict[str, Any] = {
        "prompt": "Pick SIDE or LOOP.",
        "prompt_id": "twin-pd--reskin-a--temptation-2--coop0",
        "game_id": "twin-pd",
        "grading": "group-mix",
        "payoff_cc": 0.6,
        "payoff_cd": 0.0,
        "payoff_dc": 1.0,
        "payoff_dd": 0.2,
        "label_a": "SIDE",
        "label_b": "LOOP",
        "coop_label": "SIDE",
        "endowment": 0,
        "windfall": 0,
        "opp_coop_prob": -1.0,
        "opponent_rule": "",
        "n_rounds": 0,
        "n_levels": 0,
        "reskin_id": "reskin-a",
        "payoff_variant": "temptation-2",
        "label_print_order": "canonical",
    }
    row.update(overrides)
    return row


def write_corpus(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def write_other_corpus(tmp_path: Path) -> Path:
    """The `corpus` fixture's two rows with one prompt's text changed: other bytes, the same identities."""
    return write_corpus(
        tmp_path / "other.jsonl",
        [
            corpus_row(),
            corpus_row(
                prompt="Pick LOOP or SIDE, again.",
                prompt_id="twin-pd--reskin-b--temptation-2--coop0",
                reskin_id="reskin-b",
            ),
        ],
    )


def make_run_dir(root: Path, *, game_id: str = "twin-pd", steps: tuple[int, ...] = (5,)) -> Path:
    """Build a fake training-run directory in the shape `games.train` leaves behind."""
    run_dir = root / "fake-run"
    for step in steps:
        checkpoint = run_dir / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": BASE_MODEL}), encoding="utf-8"
        )
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "arm": ARM,
                "game_id": game_id,
                "grading": "group-mix",
                "config": {"model_id": BASE_MODEL, "thinking": True},
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def cli(corpus: Path, run_dir: Path, out_dir: Path, *extra: str) -> list[str]:
    return [
        "--corpus",
        str(corpus),
        "--run-dir",
        str(run_dir),
        "--out-dir",
        str(out_dir),
        "--backend",
        "mock",
        *extra,
    ]


def vllm_cli(corpus: Path, run_dir: Path, out_dir: Path, *extra: str) -> list[str]:
    return [
        "--corpus",
        str(corpus),
        "--run-dir",
        str(run_dir),
        "--out-dir",
        str(out_dir),
        "--backend",
        "vllm",
        *extra,
    ]


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """Two rows under two reskins: two serial call groups, so "die at the second call" leaves one on disk."""
    return write_corpus(
        tmp_path / "corpus.jsonl",
        [
            corpus_row(),
            corpus_row(
                prompt="Pick LOOP or SIDE.",
                prompt_id="twin-pd--reskin-b--temptation-2--coop0",
                reskin_id="reskin-b",
            ),
        ],
    )


class SimulatedDeathError(RuntimeError):
    """The process died: raised by `DiesAtCall` at the call it was told to die on."""


class DiesAtCall:
    """A backend wrapper that dies -- raises -- when its Nth generate call is made.

    The mock-level stand-in for a spot reclaim or a killswitch mid-cell: whatever was appended and
    flushed before the call is on disk, and nothing from the call or after it. ``at_call=None``
    never dies and just counts, which is how a relaunch proves it generated only what was missing.
    """

    transport = "mock"

    def __init__(self, inner: MockBackend, *, at_call: int | None) -> None:
        self.inner = inner
        self.model_id = inner.model_id
        self.at_call = at_call
        self.calls = 0
        self.prompts_seen = 0

    def generate(self, prompts: list[str]) -> list[str]:
        self.calls += 1
        if self.calls == self.at_call:
            raise SimulatedDeathError(f"died at generate call {self.calls}")
        self.prompts_seen += len(prompts)
        return self.inner.generate(prompts)


def install_scripted_backend(
    monkeypatch: pytest.MonkeyPatch, *, dies_at_call: int | None = None
) -> DiesAtCall:
    """Have the driver build a prompt-keyed mock, optionally one that dies at its Nth generate call."""
    dying = DiesAtCall(
        MockBackend(scripted_completion, model_id="mock:scripted"), at_call=dies_at_call
    )
    monkeypatch.setattr(
        eval_training_frames.backend_cli, "backend_from_args", lambda *_args, **_kwargs: dying
    )
    return dying


def refuse_any_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any engine load fail the test: the path under test must finish without one."""

    def refuse(*_args: object, **_kwargs: object) -> object:
        pytest.fail("the driver loaded an engine where none was needed")

    monkeypatch.setattr(eval_training_frames.backend_cli, "backend_from_args", refuse)


def trace_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines(keepends=True)


def record_lines(path: Path) -> list[str]:
    """Every line after the meta record, verbatim."""
    return trace_lines(path)[1:]


def read_summary(trace: Path) -> dict[str, Any]:
    return json.loads(run_evals.summary_path_for(trace).read_text(encoding="utf-8"))


def without_resume(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "resume"}


def run_reference(
    corpus: Path,
    run_dir: Path,
    out_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    *extra: str,
    step: int = 5,
) -> Path:
    """One uninterrupted run under the prompt-keyed mock; returns the trace path."""
    install_scripted_backend(monkeypatch)
    assert eval_training_frames.main(cli(corpus, run_dir, out_dir, *extra)) == 0
    return out_dir / f"step-{step}.jsonl"


def run_interrupted(
    corpus: Path, run_dir: Path, out_dir: Path, monkeypatch: pytest.MonkeyPatch, *extra: str
) -> Path:
    """The same command dying at its second generate call; returns the partial trace path."""
    install_scripted_backend(monkeypatch, dies_at_call=2)
    with pytest.raises(SimulatedDeathError):
        eval_training_frames.main(cli(corpus, run_dir, out_dir, *extra))
    return out_dir / "step-5.jsonl"


TWO_DRAWS = ("--steps", "5", "--samples-per-prompt", "2")
BASE_TWO_DRAWS = ("--steps", "0", "--samples-per-prompt", "2")


class TestEndToEnd:
    def test_mock_run_writes_trace_and_summary(self, corpus: Path, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path)
        out_dir = tmp_path / "out"
        assert eval_training_frames.main(cli(corpus, run_dir, out_dir, *TWO_DRAWS)) == 0
        lines = (out_dir / "step-5.jsonl").read_text(encoding="utf-8").splitlines()
        meta = json.loads(lines[0])
        assert meta["record"] == RECORD_META
        assert meta["git_sha"]
        assert meta["arm"] == ARM
        assert meta["step"] == 5
        assert meta["backend_model_id"].startswith("mock:")
        assert meta["sections"] == [RECORD_TRAINING_FRAMES]
        assert meta["n_corpus_rows"] == 2
        assert meta["samples_per_prompt"] == 2
        assert meta["eval_config"]["game_behavior_samples"] == 2
        assert meta["corpus_sha256"] == eval_training_frames._corpus_sha256(corpus)
        # One unbroken session, run as the serial call sequence because the mock has no engine to drain.
        assert meta["resume"]["n_sessions"] == 1
        assert meta["resume"]["records_resumed"] == 0
        assert [session["submission"] for session in meta["resume"]["sessions"]] == [
            SUBMISSION_SERIAL
        ]
        records = [json.loads(line) for line in lines[1:]]
        assert len(records) == 4  # 2 rows x 2 samples
        assert {record["record"] for record in records} == {RECORD_TRAINING_FRAMES}
        assert {record["frame_split"] for record in records} == {"train"}
        assert sorted(record["sample_index"] for record in records) == [0, 0, 1, 1]
        assert all(record["trained_game"] for record in records)
        summary = read_summary(out_dir / "step-5.jsonl")
        assert summary["training-frames"]["n_records"] == 4
        # The canned mock completion is deliberately unparseable, so nothing may count as played.
        assert summary["training-frames"]["parse_failure_rate"] == 1.0
        # The summary attributes itself: commit, arm, step, and the sampler it was measured under
        # (None and {} on mock, which samples nothing and must not claim otherwise).
        assert summary["git_sha"]
        assert summary["arm"] == ARM
        assert summary["step"] == 5
        assert summary["sampler_mode"] is None
        assert summary["sampling"] == {}
        assert summary["resume"]["records_generated"] == 4
        assert meta["sampler_mode"] is None

    def test_a_corpus_built_game_id_reads_through_the_action_builder(self, tmp_path: Path) -> None:
        """The track-record-v2 corpus cell: a game id no roster renderer produces still parses.

        Caught red in the v2 launch rehearsal (2026-09-01): `_GAME_RECORD_BUILDERS` covered only
        roster-renderable games, so the corpus-frames cell died with a KeyError AFTER training --
        on the box that would have been a paid 70-step run with its on-distribution cell missing.
        Corpus-built games are one-shot matrix games and belong to the action builder.
        """
        corpus = write_corpus(
            tmp_path / "corpus-v2.jsonl",
            [
                corpus_row(
                    game_id="pd-track-record-v2",
                    grading="vs-stated-match",
                    prompt_id="pd-track-record-v2--reskin-a--xover57-floor6--p63--coop0",
                    stated_match_prob=0.63,
                )
            ],
        )
        run_dir = make_run_dir(tmp_path, game_id="pd-track-record-v2")
        out_dir = tmp_path / "out"
        assert eval_training_frames.main(cli(corpus, run_dir, out_dir, "--steps", "5")) == 0
        summary = read_summary(out_dir / "step-5.jsonl")
        assert summary["training-frames"]["n_records"] == 1
        assert "pd-track-record-v2" in summary["training-frames"]["coop_rate_by_game"]

    def test_step_zero_evaluates_the_base_and_claims_no_training(
        self, corpus: Path, tmp_path: Path
    ) -> None:
        run_dir = make_run_dir(tmp_path)
        out_dir = tmp_path / "out"
        assert eval_training_frames.main(cli(corpus, run_dir, out_dir, "--steps", "0")) == 0
        lines = (out_dir / "step-0.jsonl").read_text(encoding="utf-8").splitlines()
        meta = json.loads(lines[0])
        assert meta["checkpoint"] is None
        assert meta["base_model_id"] == BASE_MODEL
        records = [json.loads(line) for line in lines[1:]]
        assert records
        assert not any(record["trained_game"] for record in records)

    def test_parseable_completions_count_cooperation(
        self, corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With a completion that names the coop label, the records parse and the rate reads 1.0."""
        monkeypatch.setattr(
            eval_training_frames, "MOCK_PLUMBING_RESPONSES", (PARSEABLE_COOP_RESPONSE,)
        )
        run_dir = make_run_dir(tmp_path)
        out_dir = tmp_path / "out"
        assert eval_training_frames.main(cli(corpus, run_dir, out_dir, "--steps", "5")) == 0
        summary = read_summary(out_dir / "step-5.jsonl")
        section = summary["training-frames"]
        assert section["parse_failure_rate"] == 0.0
        assert section["coop_rate_by_game"]["twin-pd"]["rate"] == 1.0
        records = [
            json.loads(line)
            for line in (out_dir / "step-5.jsonl").read_text(encoding="utf-8").splitlines()[1:]
        ]
        assert all(record["parsed"] for record in records)
        assert all(record["action"] == "C" for record in records)

    def test_frames_records_are_keyed_exactly_as_game_behavior_records(
        self, corpus: Path, tmp_path: Path
    ) -> None:
        """The record kind is registered with the shared resume helpers in `games.evals` itself.

        `games.evals.record_identity` is what `inspect_trace` and every submission path key
        records through; a kind it does not know raises, and this cell would then be unresumable.
        Registered beside the battery's sections rather than from this module at import, so a
        reader keying frames records never depends on having imported the CLI that wrote them.
        """
        assert (
            RECORD_IDENTITY_FIELDS[RECORD_TRAINING_FRAMES]
            == RECORD_IDENTITY_FIELDS[SECTION_GAME_BEHAVIOR]
        )
        run_dir = make_run_dir(tmp_path)
        out_dir = tmp_path / "out"
        assert eval_training_frames.main(cli(corpus, run_dir, out_dir, *TWO_DRAWS)) == 0
        records = read_eval_records(out_dir / "step-5.jsonl")[1:]
        identities = {record_identity(record) for record in records}
        assert len(identities) == len(records) == 4
        assert all(identity[0] == RECORD_TRAINING_FRAMES for identity in identities)


class TestRefusals:
    """Each test commits the exact violation the refusal exists to catch."""

    def test_a_complete_trace_is_refused_and_left_intact(
        self, corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_dir = make_run_dir(tmp_path)
        trace = run_reference(corpus, run_dir, tmp_path / "out", monkeypatch, *TWO_DRAWS)
        before = trace.read_bytes()
        refuse_any_backend(monkeypatch)
        with pytest.raises(FileExistsError, match="complete eval trace"):
            eval_training_frames.main(cli(corpus, run_dir, tmp_path / "out", *TWO_DRAWS))
        assert trace.read_bytes() == before

    def test_a_foreign_file_at_the_out_path_is_refused_up_front(
        self, corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A summary-less file that is not this cell's meta cannot be resumed and must not be touched."""
        run_dir = make_run_dir(tmp_path)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "step-5.jsonl").write_text("paid-for output\n", encoding="utf-8")
        refuse_any_backend(monkeypatch)
        with pytest.raises(FileExistsError, match="not a trace of this cell"):
            eval_training_frames.main(cli(corpus, run_dir, out_dir, "--steps", "5"))
        assert (out_dir / "step-5.jsonl").read_text(encoding="utf-8") == "paid-for output\n"

    def test_a_corpus_from_another_game_is_refused(self, corpus: Path, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path, game_id="stag-hunt")
        with pytest.raises(ValueError, match="another game's training prompts"):
            eval_training_frames.main(cli(corpus, run_dir, tmp_path / "out", "--steps", "5"))

    def test_a_corpus_missing_label_fields_is_refused(self, tmp_path: Path) -> None:
        row = corpus_row()
        del row["label_a"]
        corpus = write_corpus(tmp_path / "corpus.jsonl", [row])
        run_dir = make_run_dir(tmp_path)
        with pytest.raises(ValueError, match="label_a"):
            eval_training_frames.main(cli(corpus, run_dir, tmp_path / "out", "--steps", "5"))

    def test_a_corpus_mixing_games_is_refused(self, tmp_path: Path) -> None:
        # Each row under its own prompt id, so the refusal that fires is the game rule rather than the
        # repeated-identity one: the run this corpus claims to belong to trained twin-pd alone.
        corpus = write_corpus(
            tmp_path / "corpus.jsonl",
            [
                corpus_row(),
                corpus_row(
                    game_id="stag-hunt", prompt_id="stag-hunt--reskin-a--temptation-2--coop0"
                ),
            ],
        )
        run_dir = make_run_dir(tmp_path)
        with pytest.raises(ValueError, match="mixes games"):
            eval_training_frames.main(cli(corpus, run_dir, tmp_path / "out", "--steps", "5"))

    def test_a_corpus_repeating_a_prompt_id_is_refused(self, tmp_path: Path) -> None:
        """Two rows under one (prompt_id, print order) would file duplicate identities."""
        corpus = write_corpus(
            tmp_path / "corpus.jsonl", [corpus_row(), corpus_row(prompt="Pick LOOP or SIDE.")]
        )
        run_dir = make_run_dir(tmp_path)
        with pytest.raises(ValueError, match="more than once"):
            eval_training_frames.main(cli(corpus, run_dir, tmp_path / "out", "--steps", "5"))

    def test_an_empty_corpus_is_refused(self, tmp_path: Path) -> None:
        corpus = write_corpus(tmp_path / "corpus.jsonl", [])
        run_dir = make_run_dir(tmp_path)
        with pytest.raises(ValueError, match="no rows"):
            eval_training_frames.main(cli(corpus, run_dir, tmp_path / "out", "--steps", "5"))

    def test_zero_samples_per_prompt_is_refused(self, corpus: Path, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path)
        with pytest.raises(ValueError, match="samples-per-prompt"):
            eval_training_frames.main(
                cli(corpus, run_dir, tmp_path / "out", "--steps", "5", "--samples-per-prompt", "0")
            )

    def test_an_explicit_sampler_mode_on_mock_is_refused_before_anything_runs(
        self, corpus: Path, tmp_path: Path
    ) -> None:
        """Mock samples nothing, so a mode aimed at it would be recorded but never applied."""
        run_dir = make_run_dir(tmp_path)
        out_dir = tmp_path / "out"
        with pytest.raises(ValueError, match="--sampler"):
            eval_training_frames.main(
                cli(corpus, run_dir, out_dir, "--steps", "5", "--sampler", "deployment")
            )
        assert not out_dir.exists()

    def test_a_summary_appearing_mid_run_is_refused_rather_than_redone(
        self, corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Another process finished a later step here between the plan-time check and that step's turn.

        The plan-time check saw no step-10 files; by the time step 10's turn comes, its trace and
        summary exist (planted as step 5's summary lands). Redoing it would overwrite paid-for
        output, so the ladder stops there with the planted files untouched.
        """
        run_dir = make_run_dir(tmp_path, steps=(5, 10))
        out_dir = tmp_path / "out"
        planted_trace = out_dir / "step-10.jsonl"
        planted_summary = run_evals.summary_path_for(planted_trace)
        real_write_summary = eval_training_frames._write_summary

        def write_summary_then_plant_step_10(
            target: run_evals.EvalTarget, summary: Mapping[str, Any]
        ) -> Path:
            written = real_write_summary(target, summary)
            if target.step == 5:
                shutil.copy(target.out_path, planted_trace)
                shutil.copy(written, planted_summary)
            return written

        monkeypatch.setattr(
            eval_training_frames, "_write_summary", write_summary_then_plant_step_10
        )
        backend = install_scripted_backend(monkeypatch)
        with pytest.raises(FileExistsError, match="appeared mid-run"):
            eval_training_frames.main(
                cli(corpus, run_dir, out_dir, "--steps", "5,10", "--samples-per-prompt", "2")
            )
        # Step 5 was sampled once (2 rows x 2 draws); step 10 never was.
        assert backend.prompts_seen == 4
        assert planted_trace.read_bytes() == (out_dir / "step-5.jsonl").read_bytes()
        assert planted_summary.read_bytes() == (out_dir / "step-5.summary.json").read_bytes()


class TestATraceWithoutASummaryIsResumed:
    def test_dying_at_the_second_call_then_relaunching_yields_identical_records(
        self, corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The kill-and-relaunch rehearsal on the mock backend.

        The uninterrupted run is the reference. The interrupted one dies at its second generate
        call -- the second reskin's call group -- so the first reskin's records are on disk and
        nothing else; its relaunch must keep those bytes, generate exactly the rest, and say so.
        """
        run_dir = make_run_dir(tmp_path)
        reference = run_reference(corpus, run_dir, tmp_path / "reference", monkeypatch, *TWO_DRAWS)
        reference_records = record_lines(reference)
        assert len(reference_records) == 4

        interrupted = run_interrupted(
            corpus, run_dir, tmp_path / "interrupted", monkeypatch, *TWO_DRAWS
        )
        partial = record_lines(interrupted)
        assert len(partial) == 2
        assert partial == reference_records[: len(partial)]
        assert not run_evals.summary_path_for(interrupted).exists()
        assert read_eval_records(interrupted)[0]["resume"]["n_sessions"] == 1

        relaunched = install_scripted_backend(monkeypatch)
        assert (
            eval_training_frames.main(cli(corpus, run_dir, tmp_path / "interrupted", *TWO_DRAWS))
            == 0
        )
        # Only the missing prompts reach the backend: a relaunch that regenerated everything and
        # appended just the missing records would leave identical bytes on disk and pay twice.
        assert relaunched.prompts_seen == len(reference_records) - len(partial)
        assert record_lines(interrupted) == reference_records
        meta = read_eval_records(interrupted)[0]
        assert meta["resume"]["n_sessions"] == 2
        assert meta["resume"]["records_resumed"] == len(partial)
        assert meta["resume"]["records_dropped"] == 0
        assert [session["records_resumed"] for session in meta["resume"]["sessions"]] == [
            0,
            len(partial),
        ]
        assert all(session["git_sha"] for session in meta["resume"]["sessions"])
        assert [session["submission"] for session in meta["resume"]["sessions"]] == [
            SUBMISSION_SERIAL,
            SUBMISSION_SERIAL,
        ]
        summary = read_summary(interrupted)
        assert summary["resume"]["records_generated"] == len(reference_records) - len(partial)
        assert without_resume(summary) == without_resume(read_summary(reference))
        assert summary == rebuild_summary(interrupted)

    def test_a_partial_trace_of_another_cell_is_refused_and_left_intact(
        self, corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every identity field is compared: more draws, other corpus bytes, another arm, another template fact."""
        run_dir = make_run_dir(tmp_path)
        interrupted = run_interrupted(corpus, run_dir, tmp_path / "out", monkeypatch, *TWO_DRAWS)
        before = interrupted.read_bytes()
        refuse_any_backend(monkeypatch)
        with pytest.raises(FileExistsError, match="samples_per_prompt=2 where 3 was asked"):
            eval_training_frames.main(
                cli(corpus, run_dir, tmp_path / "out", "--steps", "5", "--samples-per-prompt", "3")
            )
        assert interrupted.read_bytes() == before
        with pytest.raises(FileExistsError, match="corpus_sha256"):
            eval_training_frames.main(
                cli(write_other_corpus(tmp_path), run_dir, tmp_path / "out", *TWO_DRAWS)
            )
        assert interrupted.read_bytes() == before
        with pytest.raises(FileExistsError, match="arm="):
            eval_training_frames.main(
                cli(corpus, run_dir, tmp_path / "out", *TWO_DRAWS, "--arm", "another-arm")
            )
        assert interrupted.read_bytes() == before
        # Past the up-front peek: the full identity check refuses a changed template fact.
        with pytest.raises(ValueError, match="refusing to resume") as refusal:
            eval_training_frames.main(
                cli(corpus, run_dir, tmp_path / "out", *TWO_DRAWS, "--prefilled-think")
            )
        assert "prefilled_think" in str(refusal.value)
        assert interrupted.read_bytes() == before

    def test_a_torn_final_line_is_dropped_and_regenerated(
        self, corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_dir = make_run_dir(tmp_path)
        reference = run_reference(corpus, run_dir, tmp_path / "reference", monkeypatch, *TWO_DRAWS)
        complete = trace_lines(reference)
        torn_dir = tmp_path / "torn"
        torn_dir.mkdir()
        torn = torn_dir / "step-5.jsonl"
        torn.write_text("".join(complete[:-1]) + complete[-1][: len(complete[-1]) // 2], "utf-8")
        relaunched = install_scripted_backend(monkeypatch)
        assert eval_training_frames.main(cli(corpus, run_dir, torn_dir, *TWO_DRAWS)) == 0
        assert relaunched.prompts_seen == 1
        assert trace_lines(torn)[1:] == complete[1:]
        meta = read_eval_records(torn)[0]
        assert meta["resume"]["records_dropped"] == 1
        assert meta["resume"]["records_resumed"] == len(complete) - 2

    def test_a_complete_trace_lacking_its_summary_is_closed_out_without_an_engine(
        self, corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The cell died between its last generate and its summary write: nothing to generate."""
        run_dir = make_run_dir(tmp_path)
        trace = run_reference(corpus, run_dir, tmp_path / "out", monkeypatch, *TWO_DRAWS)
        written = read_summary(trace)
        records_before = record_lines(trace)
        run_evals.summary_path_for(trace).unlink()
        refuse_any_backend(monkeypatch)
        assert eval_training_frames.main(cli(corpus, run_dir, tmp_path / "out", *TWO_DRAWS)) == 0
        assert record_lines(trace) == records_before
        salvaged = read_summary(trace)
        assert without_resume(salvaged) == without_resume(written)
        assert salvaged["resume"]["n_sessions"] == 2
        assert salvaged["resume"]["records_resumed"] == len(records_before)
        assert salvaged["resume"]["records_generated"] == 0
        # A close-out generates nothing, so it has no submission mode to claim.
        assert [session["submission"] for session in salvaged["resume"]["sessions"]] == [
            SUBMISSION_SERIAL,
            None,
        ]

    def test_summarise_only_writes_the_summary_of_a_complete_trace(
        self, corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_dir = make_run_dir(tmp_path)
        trace = run_reference(corpus, run_dir, tmp_path / "out", monkeypatch, *TWO_DRAWS)
        written = read_summary(trace)
        run_evals.summary_path_for(trace).unlink()
        refuse_any_backend(monkeypatch)
        assert (
            eval_training_frames.main(
                cli(corpus, run_dir, tmp_path / "out", *TWO_DRAWS, "--summarise-only")
            )
            == 0
        )
        salvaged = read_summary(trace)
        assert without_resume(salvaged) == without_resume(written)
        assert salvaged == rebuild_summary(trace)
        assert salvaged["resume"]["sessions"][-1]["submission"] is None

    def test_summarise_only_refuses_an_incomplete_trace_and_a_missing_one(
        self, corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_dir = make_run_dir(tmp_path)
        interrupted = run_interrupted(corpus, run_dir, tmp_path / "out", monkeypatch, *TWO_DRAWS)
        before = interrupted.read_bytes()
        refuse_any_backend(monkeypatch)
        with pytest.raises(ValueError, match="incomplete") as refusal:
            eval_training_frames.main(
                cli(corpus, run_dir, tmp_path / "out", *TWO_DRAWS, "--summarise-only")
            )
        assert "2 of 4 planned records are missing" in str(refusal.value)
        assert interrupted.read_bytes() == before
        assert not run_evals.summary_path_for(interrupted).exists()
        with pytest.raises(FileNotFoundError, match="--summarise-only needs a trace"):
            eval_training_frames.main(
                cli(corpus, run_dir, tmp_path / "elsewhere", *TWO_DRAWS, "--summarise-only")
            )


class FakeSync:
    """Stands in for `games.s3_sync.sync_directory`, recording each upload and how much of the trace it saw."""

    def __init__(self, trace: Path) -> None:
        self.trace = trace
        self.calls: list[tuple[Path, str, int]] = []

    def __call__(self, local_dir: Path, s3_dest: str) -> SyncOutcome:
        on_disk = len(record_lines(self.trace)) if self.trace.exists() else 0
        self.calls.append((local_dir, s3_dest, on_disk))
        return SyncOutcome(command=("fake-sync",), returncode=0)


def restore_nothing(_s3_dest: str, _staging: Path) -> None:
    """The first launch's restore: an empty prefix."""


class EagerIntervalSync(run_evals.IntervalSync):
    """An `IntervalSync` whose clock jumps a full interval per read, so every hook call uploads.

    The real cadence is wall-clock and the mock's call groups finish microseconds apart, so a test
    counting uploads against it would depend on the machine's speed; the cadence itself is the
    battery's, tested with an injected clock in its own suite.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        ticks = itertools.count()
        self.clock = lambda: float(next(ticks)) * self.interval_seconds


class TestSyncDest:
    def test_the_trace_is_pushed_as_records_land_and_at_the_end(
        self, corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_dir = make_run_dir(tmp_path)
        out_dir = tmp_path / "out"
        trace = out_dir / "step-5.jsonl"
        fake_sync = FakeSync(trace)
        monkeypatch.setattr(run_evals, "restore_directory", restore_nothing)
        monkeypatch.setattr(run_evals, "sync_directory", fake_sync)
        monkeypatch.setattr(eval_training_frames, "IntervalSync", EagerIntervalSync)
        install_scripted_backend(monkeypatch)
        assert (
            eval_training_frames.main(
                cli(corpus, run_dir, out_dir, *TWO_DRAWS, "--sync-dest", "s3://bucket/prefix/")
            )
            == 0
        )
        assert all(
            local == out_dir and dest == "s3://bucket/prefix/" for local, dest, _ in fake_sync.calls
        )
        seen = [on_disk for _, _, on_disk in fake_sync.calls]
        # The first upload happened with the cell still in flight, not only once it was complete.
        assert seen[0] < 4
        assert seen[-1] == 4
        # One per call group, one at the step's summary, one when the run finished.
        assert len(fake_sync.calls) == 4

    def test_a_complete_cell_is_skipped_rather_than_refused_under_sync(
        self, corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A relaunch that finds every cell complete touches nothing: no engine, no tokenizer, no rewrite."""
        run_dir = make_run_dir(tmp_path)
        trace = run_reference(corpus, run_dir, tmp_path / "out", monkeypatch, *TWO_DRAWS)
        before = trace.read_bytes()
        fake_sync = FakeSync(trace)
        monkeypatch.setattr(run_evals, "restore_directory", restore_nothing)
        monkeypatch.setattr(run_evals, "sync_directory", fake_sync)
        refuse_any_backend(monkeypatch)

        def refuse_template_load(*_args: object, **_kwargs: object) -> object:
            pytest.fail("the driver measured the chat template with no cell left to run")

        monkeypatch.setattr(eval_training_frames, "_resolve_template_facts", refuse_template_load)
        assert (
            eval_training_frames.main(
                cli(corpus, run_dir, tmp_path / "out", *TWO_DRAWS, "--sync-dest", "s3://b/p/")
            )
            == 0
        )
        assert trace.read_bytes() == before
        assert len(fake_sync.calls) == 1  # the run-finished upload only

    def test_a_complete_cell_of_another_configuration_is_refused_not_skipped_under_sync(
        self, corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The skip is for THIS cell restored from its own prefix, never for another configuration's.

        Found 2026-09-02: under --sync-dest a relaunch at the same out path with a changed draw
        count, a changed corpus or a changed arm exited 0, loaded no engine, logged the old cell as
        complete, and the run-finished sync uploaded its numbers under the new run's name. Without
        --sync-dest the same collision was refused, so the identity peek now covers complete traces.
        """
        run_dir = make_run_dir(tmp_path)
        out_dir = tmp_path / "out"
        trace = run_reference(corpus, run_dir, out_dir, monkeypatch, *TWO_DRAWS)
        summary_path = run_evals.summary_path_for(trace)
        before = (trace.read_bytes(), summary_path.read_bytes())
        fake_sync = FakeSync(trace)
        monkeypatch.setattr(run_evals, "restore_directory", restore_nothing)
        monkeypatch.setattr(run_evals, "sync_directory", fake_sync)
        refuse_any_backend(monkeypatch)
        synced = ("--sync-dest", "s3://b/p/")
        with pytest.raises(FileExistsError, match="samples_per_prompt=2 where 3 was asked"):
            eval_training_frames.main(
                cli(corpus, run_dir, out_dir, "--steps", "5", "--samples-per-prompt", "3", *synced)
            )
        with pytest.raises(FileExistsError, match="corpus_sha256"):
            eval_training_frames.main(
                cli(write_other_corpus(tmp_path), run_dir, out_dir, *TWO_DRAWS, *synced)
            )
        with pytest.raises(FileExistsError, match="arm="):
            eval_training_frames.main(
                cli(corpus, run_dir, out_dir, *TWO_DRAWS, "--arm", "another-arm", *synced)
            )
        assert (trace.read_bytes(), summary_path.read_bytes()) == before
        # Every refusal came before any upload, so nothing of the old cell went out under a new name.
        assert fake_sync.calls == []

    def test_a_fresh_box_relaunch_restores_the_partial_trace_and_continues_it(
        self, corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The box died; the relaunch on another box finds only what the interval sync uploaded."""
        run_dir = make_run_dir(tmp_path)
        reference = run_reference(corpus, run_dir, tmp_path / "reference", monkeypatch, *TWO_DRAWS)
        interrupted = run_interrupted(
            corpus, run_dir, tmp_path / "dead-box", monkeypatch, *TWO_DRAWS
        )
        partial = record_lines(interrupted)

        def restore_the_upload(_s3_dest: str, staging: Path) -> None:
            shutil.copy(interrupted, staging / interrupted.name)

        fresh_box = tmp_path / "fresh-box"
        fake_sync = FakeSync(fresh_box / "step-5.jsonl")
        monkeypatch.setattr(run_evals, "restore_directory", restore_the_upload)
        monkeypatch.setattr(run_evals, "sync_directory", fake_sync)
        relaunched = install_scripted_backend(monkeypatch)
        assert (
            eval_training_frames.main(
                cli(corpus, run_dir, fresh_box, *TWO_DRAWS, "--sync-dest", "s3://b/p/")
            )
            == 0
        )
        assert relaunched.prompts_seen == 4 - len(partial)
        assert record_lines(fresh_box / "step-5.jsonl") == record_lines(reference)
        meta = read_eval_records(fresh_box / "step-5.jsonl")[0]
        assert meta["resume"]["n_sessions"] == 2
        assert meta["resume"]["records_resumed"] == len(partial)


class TestTheSummaryIsAPureFunctionOfTheTrace:
    LEGACY_FRAMES_META_FIELDS: ClassVar[tuple[str, ...]] = (
        "record",
        "written_at",
        "git_sha",
        "backend_model_id",
        "sections",
        "arm",
        "step",
        "thinking",
        "base_model_id",
        "checkpoint",
        "run_dir",
        "backend_kind",
        "load_mode",
        "vllm_quantization",
        "sampler_mode",
        "sampling",
        "grading",
        "corpus_path",
        "corpus_sha256",
        "n_corpus_rows",
        "samples_per_prompt",
        "prefilled_think",
    )
    """The meta fields the pre-resume writer stamped on a frames trace, in its order.

    The banked step-70 cell's keys; today's writer adds exactly `eval_config` and `resume` to them.
    """

    def test_rebuild_matches_the_written_summary_and_an_independent_reduction(
        self, corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_dir = make_run_dir(tmp_path)
        trace = run_reference(corpus, run_dir, tmp_path / "out", monkeypatch, *TWO_DRAWS)
        written = read_summary(trace)
        rebuilt = rebuild_summary(trace)
        assert rebuilt == written
        records = read_eval_records(trace)
        meta, body = records[0], records[1:]
        expected: dict[str, Any] = {
            "git_sha": meta["git_sha"],
            "arm": ARM,
            "step": 5,
            "sampler_mode": None,
            "sampling": {},
            RECORD_TRAINING_FRAMES: evals._summarise(  # pyright: ignore[reportPrivateUsage]
                SECTION_GAME_BEHAVIOR, body
            ),
            "n_corpus_rows": 2,
            "samples_per_prompt": 2,
            "resume": {**meta["resume"], "records_generated": len(body)},
        }
        assert rebuilt == expected
        assert json.dumps(rebuilt, indent=2) == json.dumps(expected, indent=2)

    def test_the_banked_corpus_frames_cell_rebuilds_byte_for_byte(self) -> None:
        """A trace written before the resume existed rebuilds its stored summary exactly.

        The banked cell has no `resume` block, so the rebuild must add none; the lifted fields,
        the section reduction and the corpus counts have to come out in the file's own order.
        """
        if not BANKED_FRAMES_TRACE.is_file():
            pytest.skip(
                f"no read-only copy of the banked corpus-frames cell at {BANKED_FRAMES_TRACE}"
            )
        stored_path = run_evals.summary_path_for(BANKED_FRAMES_TRACE)
        rebuilt = rebuild_summary(BANKED_FRAMES_TRACE)
        assert rebuilt == json.loads(stored_path.read_text(encoding="utf-8"))
        assert json.dumps(rebuilt, indent=2) == stored_path.read_text(encoding="utf-8")
        assert "resume" not in rebuilt
        assert rebuilt["training-frames"]["n_records"] == (
            rebuilt["n_corpus_rows"] * rebuilt["samples_per_prompt"]
        )

    def test_a_legacy_trace_without_resume_or_eval_config_rebuilds_in_the_historical_key_order(
        self, corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The banked-cell property, armed on every checkout rather than only where the fixture is.

        Today's meta minus the two fields the resume added IS the legacy shape (pinned against the
        banked cell's key list), so a trace rewritten that way stands in for one the old writer
        left; its rebuild must add no `resume` key and keep the summary's historical key order.
        """
        run_dir = make_run_dir(tmp_path)
        trace = run_reference(corpus, run_dir, tmp_path / "out", monkeypatch, *TWO_DRAWS)
        lines = trace_lines(trace)
        meta = json.loads(lines[0])
        legacy_meta = {
            key: value for key, value in meta.items() if key not in ("eval_config", "resume")
        }
        assert tuple(legacy_meta) == self.LEGACY_FRAMES_META_FIELDS
        legacy = tmp_path / "legacy" / "step-5.jsonl"
        legacy.parent.mkdir()
        legacy.write_text(json.dumps(legacy_meta) + "\n" + "".join(lines[1:]), encoding="utf-8")
        rebuilt = rebuild_summary(legacy)
        assert "resume" not in rebuilt
        assert list(rebuilt) == [
            "git_sha",
            "arm",
            "step",
            "sampler_mode",
            "sampling",
            RECORD_TRAINING_FRAMES,
            "n_corpus_rows",
            "samples_per_prompt",
        ]
        assert rebuilt == without_resume(read_summary(trace))

    def test_a_trace_of_another_shape_is_refused_rather_than_misdescribed(self) -> None:
        battery_meta = {"record": RECORD_META, "sections": [SECTION_GAME_BEHAVIOR], "git_sha": "x"}
        with pytest.raises(ValueError, match="not one"):
            evals.summarise_frames_trace([battery_meta])
        frames_meta = {
            "record": RECORD_META,
            "sections": [RECORD_TRAINING_FRAMES],
            "git_sha": "x",
            "n_corpus_rows": 1,
            "samples_per_prompt": 1,
        }
        stray = {"record": SECTION_GAME_BEHAVIOR, "game_id": "twin-pd", "parsed": True}
        with pytest.raises(ValueError, match="mixes cells"):
            evals.summarise_frames_trace([frames_meta, stray])
        with pytest.raises(ValueError, match="starts with"):
            evals.summarise_frames_trace([stray])


class FrozenDatetime:
    """A `datetime` whose clock does not move, so two traces written in a row carry one `written_at`."""

    FROZEN = evals.datetime(2026, 9, 3, 12, 0, 0, tzinfo=evals.UTC)

    @classmethod
    def now(cls, tz: object = None) -> evals.datetime:
        del tz
        return cls.FROZEN


def legacy_run_training_frames(
    backend: Backend,
    *,
    plan: Sequence[PlannedRequest],
    out_path: Path,
    meta: Mapping[str, Any],
    config: EvalConfig,
) -> dict[str, Any]:
    """The frames cell's own run body as it stood before 2026-09-03, over today's primitives.

    `games.eval_training_frames.run_training_frames` was a copy of `run_eval_battery`'s body with
    this cell's plan in place of the section planners' and the frames reducer in place of the
    battery's. It was deleted the day the battery grew a ``plan`` override; this is that body, kept
    verbatim on its fresh-run path (the only path the two ever differed on was the plan), so the
    override can be held to it byte for byte. The one edit is the ``admission`` the session record
    grew the same day, ``None`` here because the mock has no engine to pool.
    """
    evals._refuse_template_kwargs_the_backend_would_drop(backend, config)  # pyright: ignore[reportPrivateUsage]
    effective_submission = evals._effective_submission(backend, SUBMISSION_POOLED)  # pyright: ignore[reportPrivateUsage]
    meta_record = evals._meta_record(  # pyright: ignore[reportPrivateUsage]
        backend,
        SECTIONS_TRAINING_FRAMES,
        meta,
        config,
        submission=effective_submission,
        admission=None,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(meta_record) + "\n")
    with out_path.open("a", encoding="utf-8") as handle:
        appender = evals._TraceAppender(  # pyright: ignore[reportPrivateUsage]
            handle, records_before=0, on_records_written=None
        )
        evals._run_serial(backend, list(plan), config=config, appender=appender)  # pyright: ignore[reportPrivateUsage]
    return evals.summarise_frames_trace(read_eval_records(out_path))


class TestTheCellIsTheBatterysRunBody:
    """`run_eval_battery(plan=...)` over this cell's plan is the deleted frames driver, byte for byte."""

    @pytest.fixture
    def frames_plan(self, corpus: Path) -> tuple[list[PlannedRequest], EvalConfig, dict[str, Any]]:
        rows = load_corpus_rows(corpus)
        config = EvalConfig(game_behavior_samples=2, prefilled_think=False)
        plan = plan_training_frames(rows, config=config, trained_game=True)
        meta = {"arm": ARM, "step": 5, "n_corpus_rows": len(rows), "samples_per_prompt": 2}
        return plan, config, meta

    def test_the_plan_override_reproduces_the_legacy_driver_byte_for_byte(
        self,
        frames_plan: tuple[list[PlannedRequest], EvalConfig, dict[str, Any]],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan, config, meta = frames_plan
        monkeypatch.setattr(evals, "datetime", FrozenDatetime)
        legacy_path = tmp_path / "legacy" / "step-5.jsonl"
        legacy_summary = legacy_run_training_frames(
            MockBackend(scripted_completion, model_id="mock:scripted"),
            plan=plan,
            out_path=legacy_path,
            meta=meta,
            config=config,
        )
        battery_path = tmp_path / "battery" / "step-5.jsonl"
        battery_summary = run_eval_battery(
            MockBackend(scripted_completion, model_id="mock:scripted"),
            sections=SECTIONS_TRAINING_FRAMES,
            plan=plan,
            out_path=battery_path,
            meta=meta,
            config=config,
        )
        assert battery_path.read_bytes() == legacy_path.read_bytes()
        assert battery_summary == legacy_summary
        assert len(record_lines(battery_path)) == len(plan) == 4
        # Not vacuous: the bytes carry the frames kind, the corpus counts and one serial session.
        written_meta = read_eval_records(battery_path)[0]
        assert written_meta["sections"] == list(SECTIONS_TRAINING_FRAMES)
        assert "frame_label_audit" not in written_meta
        assert [s["submission"] for s in written_meta["resume"]["sessions"]] == [SUBMISSION_SERIAL]
        assert [s["admission"] for s in written_meta["resume"]["sessions"]] == [None]
        assert battery_summary["n_corpus_rows"] == 2
        assert battery_summary["samples_per_prompt"] == 2

    def test_the_rebuilt_summary_reads_the_kind_off_the_trace(
        self,
        frames_plan: tuple[list[PlannedRequest], EvalConfig, dict[str, Any]],
        tmp_path: Path,
    ) -> None:
        """One `rebuild_summary` serves both kinds: a frames trace comes back in the frames shape."""
        plan, config, meta = frames_plan
        path = tmp_path / "step-5.jsonl"
        summary = run_eval_battery(
            MockBackend(scripted_completion, model_id="mock:scripted"),
            sections=SECTIONS_TRAINING_FRAMES,
            plan=plan,
            out_path=path,
            meta=meta,
            config=config,
        )
        assert (
            summary
            == rebuild_summary(path)
            == evals.summarise_frames_trace(read_eval_records(path))
        )
        assert list(summary)[-4:] == [
            RECORD_TRAINING_FRAMES,
            "n_corpus_rows",
            "samples_per_prompt",
            "resume",
        ]
        with pytest.raises(ValueError, match=r"not a games\.run_evals battery"):
            evals.summarise_battery_trace(read_eval_records(path))

    def test_a_plan_disagreeing_with_its_section_list_is_refused(
        self,
        frames_plan: tuple[list[PlannedRequest], EvalConfig, dict[str, Any]],
        tmp_path: Path,
    ) -> None:
        """The meta names the sections and the summary reduces by them, so the two must agree.

        Two of these shapes passed the plan-versus-sections check alone and were caught only after
        every record had been paid for: a section list mixing the frames kind with a battery
        section validated whenever the plan rendered both, and then no reducer would read the trace
        (`summarise_trace` hands any list naming the frames kind to the frames reducer, which
        requires exactly that one, and the battery reducer refuses the list as foreign); and
        requests labelled one section while keyed under another validated on the label, then filed
        records under the key, which the summary refuses as a mixed cell.
        """
        plan, config, meta = frames_plan
        backend = MockBackend(scripted_completion, model_id="mock:scripted")
        out_path = tmp_path / "step-5.jsonl"

        def refused(
            match: str,
            *,
            sections: Sequence[str],
            plan: Sequence[PlannedRequest],
            exc: type[Exception] = ValueError,
        ) -> None:
            with pytest.raises(exc, match=match):
                run_eval_battery(
                    backend,
                    sections=sections,
                    plan=plan,
                    out_path=out_path,
                    meta=meta,
                    config=config,
                )

        refused("must agree", sections=(SECTION_GAME_BEHAVIOR,), plan=plan)
        refused("Unknown eval sections", sections=("vibes",), plan=plan)
        refused(
            "more than once",
            sections=SECTIONS_TRAINING_FRAMES,
            plan=[*plan, plan[0]],
            exc=RuntimeError,
        )
        refused(
            "no summary reads",
            sections=(RECORD_TRAINING_FRAMES, SECTION_GAME_BEHAVIOR),
            plan=[*plan, *plan_battery((SECTION_GAME_BEHAVIOR,), config)],
        )
        refused(
            "identity does not open",
            sections=(SECTION_GAME_BEHAVIOR,),
            plan=[replace(request, section=SECTION_GAME_BEHAVIOR) for request in plan],
        )
        assert not out_path.exists()


BASELINE_MIB = 900  # a neighbour already on the card, so "drained" cannot mean "empty"


def stub_vllm_engine(monkeypatch: pytest.MonkeyPatch, *, events: list[str], engine: object) -> None:
    """Stub the card and the engine load for `--backend vllm`, recording the order the driver makes them in.

    vLLM is not installed here, so the engine is whatever object the test hands over; everything
    between load and release is the real body. The order is the whole finding for the lifecycle
    tests: a baseline read AFTER construction already includes the engine's own claim, so the
    residue comes out zero however much of the card stays held.
    """
    monkeypatch.setattr(
        eval_training_frames,
        "_resolve_template_facts",
        lambda *_args, **_kwargs: run_evals.TemplateFacts(
            prefilled_think=False, chat_template_kwargs=()
        ),
    )

    def read_card() -> list[int]:
        events.append("read card")
        return [BASELINE_MIB]

    def build_engine(*_args: object, **_kwargs: object) -> object:
        events.append("build engine")
        return engine

    def release(backend: object, **kwargs: object) -> dict[str, object]:
        events.append(f"release {kwargs['baseline_mib']} engine={backend is engine}")
        return {}

    monkeypatch.setattr(vllm_teardown, "vram_used_mib", read_card)
    monkeypatch.setattr(eval_training_frames.backend_cli, "backend_from_args", build_engine)
    monkeypatch.setattr(eval_training_frames, "release_engine", release)


class VLLMKindMock(MockBackend):
    """A scripted backend that claims the vLLM transport, so the driver takes the pooled path."""

    transport = "vllm"


class FakeDrain:
    """Stands in for `games.chunked_decode.stream_vllm_completions`: completion order reversed, optionally dying.

    The real drain yields in engine completion order, which is not prompt order; reversing makes
    the pairing load-bearing here too. ``dies_after`` yields that many completions and then raises,
    the way a box death mid-drain leaves the earlier ones flushed and nothing after.
    """

    def __init__(self, *, dies_after: int | None = None) -> None:
        self.dies_after = dies_after
        self.submissions: list[int] = []

    def __call__(self, backend: Any, prompts: list[str]) -> Any:
        self.submissions.append(len(prompts))
        for yielded, index in enumerate(reversed(range(len(prompts)))):
            if yielded == self.dies_after:
                raise SimulatedDeathError(f"died after {yielded} completions")
            yield index, backend.generate([prompts[index]])[0]


def install_fake_drain(
    monkeypatch: pytest.MonkeyPatch, *, dies_after: int | None = None
) -> FakeDrain:
    drain = FakeDrain(dies_after=dies_after)
    monkeypatch.setattr(evals, "stream_vllm_completions", drain)
    return drain


class TestPooledSubmission:
    """On a vLLM backend the corpus is one engine submission, filed per completion; elsewhere serial."""

    def test_a_vllm_backend_is_pooled_and_files_the_same_records_as_serial(
        self, corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_dir = make_run_dir(tmp_path)
        serial = run_reference(
            corpus, run_dir, tmp_path / "serial", monkeypatch, *BASE_TWO_DRAWS, step=0
        )
        serial_records = record_lines(serial)
        events: list[str] = []
        stub_vllm_engine(
            monkeypatch, events=events, engine=VLLMKindMock(scripted_completion, model_id="fake")
        )
        drain = install_fake_drain(monkeypatch)
        pooled_dir = tmp_path / "pooled"
        assert (
            eval_training_frames.main(vllm_cli(corpus, run_dir, pooled_dir, *BASE_TWO_DRAWS)) == 0
        )
        pooled_records = record_lines(pooled_dir / "step-0.jsonl")
        assert drain.submissions == [4]
        assert pooled_records != serial_records
        assert sorted(pooled_records) == sorted(serial_records)
        meta = read_eval_records(pooled_dir / "step-0.jsonl")[0]
        assert [session["submission"] for session in meta["resume"]["sessions"]] == [
            SUBMISSION_POOLED
        ]
        assert (
            without_resume(read_summary(pooled_dir / "step-0.jsonl"))[RECORD_TRAINING_FRAMES]
            == without_resume(read_summary(serial))[RECORD_TRAINING_FRAMES]
        )
        assert events == ["read card", "build engine", f"release {[BASELINE_MIB]} engine=True"]

    def test_dying_mid_drain_then_relaunching_pooled_yields_the_same_record_set(
        self, corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_dir = make_run_dir(tmp_path)
        events: list[str] = []
        engine = VLLMKindMock(scripted_completion, model_id="fake")
        stub_vllm_engine(monkeypatch, events=events, engine=engine)
        install_fake_drain(monkeypatch)
        reference_dir = tmp_path / "reference"
        args = BASE_TWO_DRAWS
        assert eval_training_frames.main(vllm_cli(corpus, run_dir, reference_dir, *args)) == 0
        reference_records = record_lines(reference_dir / "step-0.jsonl")

        install_fake_drain(monkeypatch, dies_after=1)
        interrupted_dir = tmp_path / "interrupted"
        with pytest.raises(SimulatedDeathError):
            eval_training_frames.main(vllm_cli(corpus, run_dir, interrupted_dir, *args))
        interrupted = interrupted_dir / "step-0.jsonl"
        partial = record_lines(interrupted)
        assert len(partial) == 1
        # The engine was released even though the drain died: the ladder's next step needs the card.
        assert events[-1] == f"release {[BASELINE_MIB]} engine=True"

        relaunched = install_fake_drain(monkeypatch)
        assert eval_training_frames.main(vllm_cli(corpus, run_dir, interrupted_dir, *args)) == 0
        assert relaunched.submissions == [len(reference_records) - len(partial)]
        assert record_lines(interrupted)[: len(partial)] == partial
        assert sorted(record_lines(interrupted)) == sorted(reference_records)
        summary = read_summary(interrupted)
        assert summary["resume"]["n_sessions"] == 2
        assert summary["resume"]["records_resumed"] == len(partial)
        assert summary["resume"]["records_generated"] == len(reference_records) - len(partial)
        assert [session["submission"] for session in summary["resume"]["sessions"]] == [
            SUBMISSION_POOLED,
            SUBMISSION_POOLED,
        ]
        assert without_resume(summary) == without_resume(
            read_summary(reference_dir / "step-0.jsonl")
        )

    def test_serial_submission_on_a_vllm_backend_runs_the_call_groups(
        self, corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_dir = make_run_dir(tmp_path)
        events: list[str] = []
        calls: list[int] = []

        class CountingVLLMKindMock(VLLMKindMock):
            def generate(self, prompts: list[str]) -> list[str]:
                calls.append(len(prompts))
                return super().generate(prompts)

        stub_vllm_engine(
            monkeypatch,
            events=events,
            engine=CountingVLLMKindMock(scripted_completion, model_id="fake"),
        )
        drain = install_fake_drain(monkeypatch)
        out_dir = tmp_path / "out"
        assert (
            eval_training_frames.main(
                vllm_cli(
                    corpus,
                    run_dir,
                    out_dir,
                    "--steps",
                    "0",
                    "--samples-per-prompt",
                    "2",
                    "--submission",
                    "serial",
                )
            )
            == 0
        )
        assert drain.submissions == []
        assert calls == [2, 2]  # one call per reskin
        meta = read_eval_records(out_dir / "step-0.jsonl")[0]
        assert [session["submission"] for session in meta["resume"]["sessions"]] == [
            SUBMISSION_SERIAL
        ]


class TestEngineLifecycle:
    """The card is read before an engine loads and given back after, per step of the ladder.

    This CLI evaluates a whole checkpoint ladder inside ONE process, and until now tore each engine
    down with a private copy of exactly the pattern `games/vllm_teardown.py` was written to replace:
    `del backend` plus `torch.cuda.empty_cache()`, neither of which can reach the VRAM held by
    vLLM's own `EngineCore` subprocess. There was no pre-load baseline and no read-back of the card,
    so a teardown that freed nothing was indistinguishable from one that worked.

    Driven with `--backend vllm` at step 0, because vLLM is not installed here and step 0 needs no
    LoRA merge; the engine is a `MockBackend` (transport ``mock``, so the serial path runs), so
    everything between load and release is the real body. The gate deciding which kinds read the
    card at all is the real one.
    """

    def test_the_card_is_read_before_the_engine_and_released_after(
        self, corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        events: list[str] = []
        engine = MockBackend([PARSEABLE_COOP_RESPONSE], model_id="fake-vllm-engine")
        stub_vllm_engine(monkeypatch, events=events, engine=engine)
        run_dir = make_run_dir(tmp_path)
        out_dir = tmp_path / "out"
        assert eval_training_frames.main(vllm_cli(corpus, run_dir, out_dir, "--steps", "0")) == 0
        assert events == ["read card", "build engine", f"release {[BASELINE_MIB]} engine=True"]
        # The release ran after a body that completed, not instead of one: both artifacts are there.
        assert (out_dir / "step-0.jsonl").exists()
        assert (out_dir / "step-0.summary.json").exists()

    def test_the_engine_is_released_even_when_the_body_raises(
        self, corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A step dying with its engine up holds the card against every later step: the release is in a `finally`."""
        events: list[str] = []
        engine = MockBackend([PARSEABLE_COOP_RESPONSE], model_id="fake-vllm-engine")
        stub_vllm_engine(monkeypatch, events=events, engine=engine)

        def explode(*_args: object, **_kwargs: object) -> dict[str, Any]:
            events.append("sample corpus")
            raise RuntimeError("a step blew up")

        monkeypatch.setattr(eval_training_frames, "run_eval_battery", explode)
        run_dir = make_run_dir(tmp_path)
        out_dir = tmp_path / "out"
        with pytest.raises(RuntimeError, match="a step blew up"):
            eval_training_frames.main(vllm_cli(corpus, run_dir, out_dir, "--steps", "0"))
        assert events == [
            "read card",
            "build engine",
            "sample corpus",
            f"release {[BASELINE_MIB]} engine=True",
        ]

    def test_a_checkpoint_step_serves_the_runtime_adapter_and_verifies_it(
        self, corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A checkpoint under a vLLM-kind backend is served UN-MERGED, and verified to apply.

        This file used to merge every checkpoint before serving it, against the runtime-LoRA
        eval ruling (a bf16 merge loses part of the delta); the resolution now goes through
        `games.eval_model.resolve_served_model`, exactly as the battery's own driver does. The
        engine stub records what it is asked to load, so a silent fall-back to merged (or to
        bare base weights) fails here.
        """

        class AdapterProvingBackend(MockBackend):
            probed: ClassVar[list[tuple[str, ...]]] = []

            def assert_adapter_changes_output(self, prompts: tuple[str, ...]) -> None:
                self.probed.append(tuple(prompts))

        events: list[str] = []
        engine = AdapterProvingBackend([PARSEABLE_COOP_RESPONSE], model_id="fake-vllm-engine")
        built_kwargs: list[dict[str, Any]] = []
        stub_vllm_engine(monkeypatch, events=events, engine=engine)

        def build_engine(*_args: object, **kwargs: object) -> MockBackend:
            events.append("build engine")
            built_kwargs.append(dict(kwargs))
            return engine

        monkeypatch.setattr(eval_training_frames.backend_cli, "backend_from_args", build_engine)
        run_dir = make_run_dir(tmp_path)
        adapter_config = run_dir / "checkpoint-5" / "adapter_config.json"
        adapter_config.write_text(
            json.dumps(
                {
                    "base_model_name_or_path": BASE_MODEL,
                    "r": 16,
                    "target_modules": ["q_proj", "v_proj"],
                }
            ),
            encoding="utf-8",
        )
        out_dir = tmp_path / "out"
        assert eval_training_frames.main(vllm_cli(corpus, run_dir, out_dir, "--steps", "5")) == 0
        extra = built_kwargs[0]["extra_kwargs"]
        assert extra["lora_adapter"] == str(run_dir / "checkpoint-5")
        assert extra["enable_lora"] is True
        assert engine.probed, "the runtime adapter was never verified to change output"
        meta = json.loads((out_dir / "step-5.jsonl").read_text().splitlines()[0])
        assert meta["load_mode"] == "runtime-adapter"
        assert meta["backend_model_id"] == "fake-vllm-engine"

    def test_the_mock_kind_reads_no_card_and_runs_no_vllm_release(
        self, corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A kind that loads no engine reaches neither the card read (raises with no GPU) nor the vLLM release."""

        def refuse_card() -> list[int]:
            pytest.fail("the mock backend read the card")

        def refuse_release(*_args: object, **_kwargs: object) -> dict[str, object]:
            pytest.fail("the mock backend was put through the vLLM release")

        monkeypatch.setattr(vllm_teardown, "vram_used_mib", refuse_card)
        monkeypatch.setattr(eval_training_frames, "release_engine", refuse_release)
        run_dir = make_run_dir(tmp_path)
        out_dir = tmp_path / "out"
        assert eval_training_frames.main(cli(corpus, run_dir, out_dir, "--steps", "5")) == 0
        assert (out_dir / "step-5.jsonl").exists()


class TestAMixedTrainingCorpus:
    """A breadth arm's corpus is several games in one file, read back here on the trained frames.

    Before wave 4b a training corpus was one game, so this CLI's game rule and its per-record game
    label were both derived from row 0. On a mixed corpus that labels every other game wrong, and it
    would be invisible: every record would parse, every rate would be computed, and the trace would
    say the whole corpus was twin-pd. The run's own record names the set the corpus may carry, so that
    is what the file is checked against.
    """

    BREADTH_GAMES: ClassVar[tuple[str, ...]] = ("twin-pd", "stag-hunt", "chicken")

    def mixed_corpus(self, root: Path) -> Path:
        """One row per game, each with its own prompt id and reskin, so identities stay distinct."""
        return write_corpus(
            root / "corpus.jsonl",
            [
                corpus_row(
                    game_id=game_id,
                    prompt_id=f"{game_id}--reskin-a--temptation-2--coop0",
                    reskin_id=f"reskin-{game_id}",
                    prompt=f"Pick SIDE or LOOP in {game_id}.",
                )
                for game_id in self.BREADTH_GAMES
            ],
        )

    def breadth_run_dir(self, root: Path) -> Path:
        """A run directory whose record names the lead game and the other games its corpus carried."""
        run_dir = make_run_dir(root, game_id=self.BREADTH_GAMES[0])
        record_path = run_dir / "run_config.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["game_ids"] = list(self.BREADTH_GAMES[1:])
        record_path.write_text(json.dumps(record), encoding="utf-8")
        return run_dir

    def test_the_run_record_contributes_its_whole_game_set(self, tmp_path: Path) -> None:
        facts = run_evals._read_run_facts(self.breadth_run_dir(tmp_path))  # pyright: ignore[reportPrivateUsage]
        assert facts.game_id == "twin-pd"
        assert facts.game_ids == self.BREADTH_GAMES[1:]

    def test_a_record_predating_the_field_contributes_no_extra_games(self, tmp_path: Path) -> None:
        facts = run_evals._read_run_facts(make_run_dir(tmp_path))  # pyright: ignore[reportPrivateUsage]
        assert facts.game_ids == ()

    def test_a_mixed_corpus_of_the_runs_games_is_accepted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        corpus = self.mixed_corpus(tmp_path)
        run_dir = self.breadth_run_dir(tmp_path)
        out_dir = tmp_path / "out"
        install_scripted_backend(monkeypatch)
        assert eval_training_frames.main(cli(corpus, run_dir, out_dir, "--steps", "5")) == 0
        records = read_eval_records(out_dir / "step-5.jsonl")[1:]
        assert {str(record["game_id"]) for record in records} == set(self.BREADTH_GAMES)

    def test_every_record_carries_its_own_rows_game(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The label row 0 used to supply for the whole file: a record's game must be the game of the
        # prompt it answered, or every per-game rate below is one game's records under other names.
        corpus = self.mixed_corpus(tmp_path)
        run_dir = self.breadth_run_dir(tmp_path)
        out_dir = tmp_path / "out"
        install_scripted_backend(monkeypatch)
        assert eval_training_frames.main(cli(corpus, run_dir, out_dir, "--steps", "5")) == 0
        for record in read_eval_records(out_dir / "step-5.jsonl")[1:]:
            assert str(record["prompt_id"]).startswith(str(record["game_id"]))

    def test_a_game_outside_the_runs_set_is_still_refused(self, tmp_path: Path) -> None:
        corpus = write_corpus(
            tmp_path / "corpus.jsonl",
            [corpus_row(), corpus_row(game_id="hi-lo", prompt_id="hi-lo--reskin-a--x--coop0")],
        )
        run_dir = self.breadth_run_dir(tmp_path)
        with pytest.raises(ValueError, match="hi-lo"):
            eval_training_frames.main(cli(corpus, run_dir, tmp_path / "out", "--steps", "5"))

    def test_the_trained_games_of_the_plan_are_every_game_the_corpus_holds(
        self, tmp_path: Path
    ) -> None:
        corpus = self.mixed_corpus(tmp_path)
        run_dir = self.breadth_run_dir(tmp_path)
        rows = load_corpus_rows(corpus)
        args = eval_training_frames._parse_args(  # pyright: ignore[reportPrivateUsage]
            cli(corpus, run_dir, tmp_path / "out", "--steps", "5")
        )
        facts = run_evals._read_run_facts(run_dir)  # pyright: ignore[reportPrivateUsage]
        plan = eval_training_frames._resolve_plan(args, rows=rows, facts=facts)  # pyright: ignore[reportPrivateUsage]
        assert set(plan.trained_game_ids) == set(self.BREADTH_GAMES)

    def test_the_summary_splits_the_cooperation_rate_by_game_and_framing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # What the wave reads off this cell: a pooled rate over a corpus spanning several games and
        # framings averages the very cells whose separation is the measurement.
        rows = [
            corpus_row(
                game_id=game_id,
                prompt_id=f"{game_id}--{framing}--coop0",
                reskin_id=f"reskin-{game_id}",
                prompt=f"Pick SIDE or LOOP in {game_id} against {framing}.",
                framing_id=framing,
            )
            for game_id in ("twin-pd", "stag-hunt")
            for framing in ("twin", "human")
        ]
        corpus = write_corpus(tmp_path / "corpus.jsonl", rows)
        run_dir = self.breadth_run_dir(tmp_path)
        out_dir = tmp_path / "out"
        install_scripted_backend(monkeypatch)
        assert eval_training_frames.main(cli(corpus, run_dir, out_dir, "--steps", "5")) == 0
        summary = read_summary(out_dir / "step-5.jsonl")[RECORD_TRAINING_FRAMES]
        # The per-game split is the behaviour reducer's own and predates this; the per-framing one is
        # what a corpus rendered under several counterpart paragraphs needs.
        assert set(summary["coop_rate_by_game"]) == {"twin-pd", "stag-hunt"}
        assert set(summary["coop_rate_by_game_framing"]) == {
            "twin-pd::twin",
            "twin-pd::human",
            "stag-hunt::twin",
            "stag-hunt::human",
        }

    def test_a_corpus_with_no_framing_column_gains_no_framing_key(
        self, corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Every banked frames cell: one game, no framing column. The rebuild of such a trace has to
        # stay what its stored summary says byte for byte, which the pinned-rebuild tests check
        # against the banked cell, so the framing key appears only where the records carry a framing.
        run_dir = make_run_dir(tmp_path)
        out_dir = tmp_path / "out"
        install_scripted_backend(monkeypatch)
        assert eval_training_frames.main(cli(corpus, run_dir, out_dir, "--steps", "5")) == 0
        summary = read_summary(out_dir / "step-5.jsonl")[RECORD_TRAINING_FRAMES]
        assert set(summary["coop_rate_by_game"]) == {"twin-pd"}
        assert "coop_rate_by_game_framing" not in summary
        records = read_eval_records(out_dir / "step-5.jsonl")[1:]
        assert all("counterpart_framing" not in record for record in records)
