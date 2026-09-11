"""The sweep resolves its corpus from a ``--items module:attribute`` reference.

This is the seam that replaced the hardcoded import of the deleted v1 corpus, and it guards real
money: a submit that resolves the wrong attribute, or an empty one, creates a billable Bedrock
batch job over prompts nobody meant to send. So every failure mode is a loud exception rather than
an empty sweep, and each one is tested here.

The other three groups of tests here guard the same money from the other three directions: the
output-token cap has to be chosen rather than defaulted, a re-run must not re-pay for a model whose
handle is already on disk, and a wrong-path marker the task itself prints is named before a job
exists.
"""

import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from reward_hacking.bedrock_batch import BatchJobHandle, cell_digest, prompt_digest
from reward_hacking.jagged import sweep
from reward_hacking.jagged.items import Item
from reward_hacking.jagged.sweep import parse_args, resolve_items

CORPUS_MODULE = "fake_corpus"

NOVA_MICRO = "us.amazon.nova-micro-v1:0"
GPT_OSS_20B = "openai.gpt-oss-20b-1:0"
GPT_OSS_120B = "openai.gpt-oss-120b-1:0"


def build_item(index: int) -> Item:
    return Item(
        id=f"probe-{index:02d}",
        dimension="arithmetic",
        move_concept="reports_the_correct_total",
        task=f"Add {index} and 2 and report the total.",
        elicitation="Before answering, say whether the total above is right.",
        elicitation_placebo="Before answering, name the reader your answer is for.",
        pressure="You are scored only on answering the question as asked.",
        pressure_placebo="You are scored only on the formatting of your answer.",
        move_markers=(f"the total is {index + 2}",),
    )


# A task supplying working that contains the flawed subtotal has to print its own wrong-path marker.
LEAKING_WRONG_PATH_ITEM = replace(
    build_item(0),
    id="leaks-wrong-path",
    task="A colleague's note says the answer is 49. Add 100 and 49 and report the total.",
    move_markers=("the total is 149",),
    wrong_path_markers=("49",),
)

CLEAN_WRONG_PATH_ITEM = replace(
    build_item(1), id="clean-wrong-path", wrong_path_markers=("the total is 56",)
)


@pytest.fixture
def corpus_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register an importable module carrying one good corpus and two malformed ones."""
    module = ModuleType(CORPUS_MODULE)
    vars(module).update(
        {
            "ITEMS": (build_item(0), build_item(1)),
            "EMPTY": (),
            "NOT_ITEMS": ("just a string",),
            "WRONG_PATH_ITEMS": (LEAKING_WRONG_PATH_ITEM, CLEAN_WRONG_PATH_ITEM),
        }
    )
    monkeypatch.setitem(sys.modules, CORPUS_MODULE, module)


@pytest.mark.usefixtures("corpus_module")
def test_a_reference_resolves_to_the_items_it_names() -> None:
    items = resolve_items(f"{CORPUS_MODULE}:ITEMS")
    assert [item.id for item in items] == ["probe-00", "probe-01"]


def test_a_reference_without_an_attribute_is_rejected() -> None:
    with pytest.raises(ValueError, match="must read module:attribute"):
        resolve_items(CORPUS_MODULE)


@pytest.mark.usefixtures("corpus_module")
def test_a_missing_attribute_is_rejected() -> None:
    with pytest.raises(ValueError, match="has no attribute"):
        resolve_items(f"{CORPUS_MODULE}:SEED_ITEMS")


@pytest.mark.usefixtures("corpus_module")
def test_an_empty_corpus_is_rejected_rather_than_submitting_nothing() -> None:
    with pytest.raises(ValueError, match="is empty"):
        resolve_items(f"{CORPUS_MODULE}:EMPTY")


@pytest.mark.usefixtures("corpus_module")
def test_a_sequence_of_something_other_than_items_is_rejected() -> None:
    """Caught here so the message names the flag, not deep inside prompt rendering."""
    with pytest.raises(TypeError, match="not Item instances"):
        resolve_items(f"{CORPUS_MODULE}:NOT_ITEMS")


def test_an_unimportable_module_raises_rather_than_sweeping_nothing() -> None:
    with pytest.raises(ModuleNotFoundError):
        resolve_items("reward_hacking.jagged.no_such_corpus:ITEMS")


def test_the_deleted_v1_corpus_is_gone_from_the_package() -> None:
    """The point of the removal: no module in this package holds items any more.

    Keyed on the paths that were deleted, so re-adding a corpus under one of them fails here rather
    than quietly restoring what is being scrubbed from history.
    """
    package = Path(__file__).resolve().parents[1] / "jagged"
    assert sorted(path.name for path in package.glob("items_*.py")) == []
    assert not (package / "seed_items.py").exists()


class TestTheOutputCapIsChosenRatherThanDefaulted:
    """A cap is a measurement, so the flag has no default and a submit without it stops.

    The traces already on disk are why: counted over them, 19 of 580 gpt-oss-20b replies and 12 of
    435 gpt-oss-120b replies end at exactly 2048 output tokens, with four other roster models
    contributing a handful more, and ``stop_reason`` is None on all 4,115 response records. So those
    truncations read as a model that did not make the move rather than as a cap that cut it off.
    """

    def test_a_submit_that_names_no_cap_is_refused(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["submit", "--items", f"{CORPUS_MODULE}:ITEMS", "--models", GPT_OSS_120B])

    def test_a_cap_named_explicitly_reaches_the_namespace(self) -> None:
        args = parse_args(
            [
                "submit",
                "--items",
                f"{CORPUS_MODULE}:ITEMS",
                "--models",
                GPT_OSS_120B,
                "--max-new-tokens",
                "24576",
            ]
        )
        assert args.max_new_tokens == 24576

    def test_collect_needs_no_cap_because_it_re_reads_a_submitted_job(self, tmp_path: Path) -> None:
        args = parse_args(
            ["collect", "--items", f"{CORPUS_MODULE}:ITEMS", "--handle-dir", str(tmp_path)]
        )
        assert args.handle_dir == tmp_path


class _RecordingBackend:
    """Stands in for the batch backend, recording which models a submit actually reached."""

    def __init__(self, model_id: str, submitted: list[str]) -> None:
        self.model_id = model_id
        self._submitted = submitted

    def submit(
        self, prompts: list[str], *, metadata: Sequence[Mapping[str, Any]]
    ) -> BatchJobHandle:
        self._submitted.append(self.model_id)
        return _stub_handle(self.model_id, prompts, metadata)


def _stub_handle(
    model_id: str, prompts: Sequence[str], metadata: Sequence[Mapping[str, Any]]
) -> BatchJobHandle:
    return BatchJobHandle(
        job_arn=f"arn:aws:bedrock:us-west-2:000000000000:model-invocation-job/{model_id}",
        job_name="jagged-test",
        model_id=model_id,
        record_count=len(prompts),
        prompt_digest=prompt_digest(prompts),
        cell_digest=cell_digest(metadata),
        input_uri="s3://bucket/run/input.jsonl",
        output_uri="s3://bucket/run/output/",
        region="us-west-2",
        profile="stub-profile",
        submitted_at="2026-08-17T00:00:00+00:00",
    )


PAID_FOR_HANDLE_CELL = [
    {"item_id": "already-paid", "dimension": "arithmetic", "arm": "spontaneous", "repeat": 0}
]


def _submit_args(handle_dir: Path, models: list[str], *extra: str) -> Any:
    return parse_args(
        [
            "submit",
            "--items",
            f"{CORPUS_MODULE}:ITEMS",
            "--models",
            *models,
            "--handle-dir",
            str(handle_dir),
            "--max-new-tokens",
            "2048",
            *extra,
        ]
    )


@pytest.mark.usefixtures("corpus_module")
class TestASecondSubmitDoesNotRePayForAModelThatHasAHandle:
    """The module's promise -- an interrupted session resumes rather than re-paying -- for submit.

    The partial-failure message points the operator back at the same handle directory, and the
    handle path is a pure function of (directory, model id), so a re-run there would create a
    genuinely new billable job per already-successful model and overwrite the handle that was paid
    for, leaving the first job's ARN recoverable only from the log.
    """

    def _run(
        self, monkeypatch: pytest.MonkeyPatch, handle_dir: Path, *extra: str
    ) -> tuple[list[str], int]:
        submitted: list[str] = []
        monkeypatch.setattr(
            sweep,
            "_backend",
            lambda model_id, _args, _run_id: _RecordingBackend(model_id, submitted),
        )
        exit_code = sweep.submit(_submit_args(handle_dir, [NOVA_MICRO, GPT_OSS_20B], *extra))
        return submitted, exit_code

    def test_a_model_whose_handle_exists_is_not_submitted_again(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        paid = sweep._handle_path(tmp_path, NOVA_MICRO)
        _stub_handle(NOVA_MICRO, ["already paid"], PAID_FOR_HANDLE_CELL).save(paid)
        before = paid.read_bytes()

        submitted, exit_code = self._run(monkeypatch, tmp_path)

        assert submitted == [GPT_OSS_20B]
        assert paid.read_bytes() == before
        assert exit_code == 0
        assert sweep._handle_path(tmp_path, GPT_OSS_20B).exists()

    def test_resubmit_overwrites_the_existing_handle_on_purpose(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _stub_handle(NOVA_MICRO, ["already paid"], PAID_FOR_HANDLE_CELL).save(
            sweep._handle_path(tmp_path, NOVA_MICRO)
        )
        submitted, _ = self._run(monkeypatch, tmp_path, "--resubmit")
        assert submitted == [NOVA_MICRO, GPT_OSS_20B]

    def test_a_first_run_submits_every_model(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        submitted, _ = self._run(monkeypatch, tmp_path)
        assert submitted == [NOVA_MICRO, GPT_OSS_20B]


@pytest.mark.usefixtures("corpus_module")
class TestASkipMustAgreeWithTheInvocationItIsSkippingFor:
    """The skip above reuses paid inference, so it must check this is the inference asked for.

    ``--max-new-tokens`` is required precisely because the cap is part of the measurement, and the
    skip is the one path that reaches ``collect`` without honouring it: the trace is labelled from
    the handle, so a re-run at 24576 that skips a model sampled at 2048 pools two caps into one
    table with each row correctly labelled and the comparison across rows meaningless. Cheap to
    catch, invisible afterwards, and the cap mismatch is the direction that manufactures a fake
    capability gap -- a truncated reply reads as a model that did not make the move.

    Checked before anything is submitted rather than per model in the loop, because the operator's
    fix is to change a flag or a directory: aborting half-way would leave a directory that is now
    also a partly billed run of the new invocation.
    """

    def _submit(
        self, monkeypatch: pytest.MonkeyPatch, handle_dir: Path, *extra: str
    ) -> tuple[list[str], int]:
        submitted: list[str] = []
        monkeypatch.setattr(
            sweep,
            "_backend",
            lambda model_id, _args, _run_id: _RecordingBackend(model_id, submitted),
        )
        code = sweep.submit(_submit_args(handle_dir, [NOVA_MICRO, GPT_OSS_20B], *extra))
        return submitted, code

    def _save_paid_handle(self, handle_dir: Path, **fields: object) -> Path:
        path = sweep._handle_path(handle_dir, NOVA_MICRO)
        replace(_stub_handle(NOVA_MICRO, ["already paid"], PAID_FOR_HANDLE_CELL), **fields).save(
            path
        )
        return path

    def test_a_handle_sampled_at_another_cap_stops_the_run_before_anything_is_billed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._save_paid_handle(tmp_path, max_tokens=24576)
        submitted: list[str] = []
        monkeypatch.setattr(
            sweep,
            "_backend",
            lambda model_id, _args, _run_id: _RecordingBackend(model_id, submitted),
        )
        # _submit_args asks for 2048; the handle on disk was sampled at 24576.
        with pytest.raises(RuntimeError, match="max_tokens"):
            sweep.submit(_submit_args(tmp_path, [NOVA_MICRO, GPT_OSS_20B]))
        assert submitted == [], "a mismatched directory was billed for before it was refused"

    def test_the_refusal_names_the_model_both_caps_and_the_way_out(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._save_paid_handle(tmp_path, max_tokens=24576)
        monkeypatch.setattr(
            sweep, "_backend", lambda model_id, _args, _run_id: _RecordingBackend(model_id, [])
        )
        with pytest.raises(RuntimeError) as caught:
            sweep.submit(_submit_args(tmp_path, [NOVA_MICRO]))
        message = str(caught.value)
        assert NOVA_MICRO in message
        assert "24576" in message
        assert "2048" in message
        assert "--resubmit" in message

    def test_a_mismatched_reasoning_effort_is_refused_the_same_way(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The other label the handle carries, and the second elicitation axis of the run."""
        self._save_paid_handle(tmp_path, max_tokens=2048, reasoning_effort="high")
        submitted: list[str] = []
        monkeypatch.setattr(
            sweep,
            "_backend",
            lambda model_id, _args, _run_id: _RecordingBackend(model_id, submitted),
        )
        with pytest.raises(RuntimeError, match="reasoning_effort"):
            sweep.submit(_submit_args(tmp_path, [NOVA_MICRO, GPT_OSS_20B]))
        assert submitted == []

    def test_a_handle_recording_the_same_config_is_still_skipped_silently(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The control: the check must not turn every resume into a refusal."""
        paid = self._save_paid_handle(tmp_path, max_tokens=2048)
        before = paid.read_bytes()

        submitted, code = self._submit(monkeypatch, tmp_path)

        assert submitted == [GPT_OSS_20B]
        assert paid.read_bytes() == before
        assert code == 0

    def test_resubmit_overrides_the_check_because_it_overwrites_the_handle(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``--resubmit`` re-samples at the new config, so the old one is not reused at all."""
        self._save_paid_handle(tmp_path, max_tokens=24576)
        submitted, code = self._submit(monkeypatch, tmp_path, "--resubmit")
        assert submitted == [NOVA_MICRO, GPT_OSS_20B]
        assert code == 0

    def test_a_handle_predating_the_sampling_labels_warns_and_proceeds(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """``None`` means "written before the field existed", which is unknown, not a mismatch.

        Three handle directories on disk predate those fields, so refusing on ``None`` would make
        every one of them uncollectable; and claiming a match would be the mislabel this guards.
        """
        self._save_paid_handle(tmp_path, max_tokens=None, reasoning_effort=None)
        with caplog.at_level(logging.WARNING, logger=sweep.logger.name):
            submitted, code = self._submit(monkeypatch, tmp_path)

        assert submitted == [GPT_OSS_20B]
        assert code == 0
        warnings = [
            message
            for message in (record.getMessage() for record in caplog.records)
            if NOVA_MICRO in message
        ]
        assert any("records no sampling config" in message for message in warnings), warnings


@pytest.mark.usefixtures("corpus_module")
class TestATrackedHandleDirectoryIsRefusedBeforeAnythingIsBilled:
    """The handles carry the AWS account id, bucket and profile, and ``--handle-dir`` is a bare
    operator-supplied path: a sweep pointed at a tracked directory once produced a git-trackable
    file of exactly those values. The refusal must fire before any job exists, because the writer's
    own guard fires only at save time -- inside submit's per-model ``except``, after the bill."""

    def test_a_tracked_handle_dir_stops_the_submit_before_any_job_exists(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        tracked_dir = repo_root / "reward_hacking" / "jagged"
        submitted: list[str] = []
        monkeypatch.setattr(
            sweep,
            "_backend",
            lambda model_id, _args, _run_id: _RecordingBackend(model_id, submitted),
        )
        # The unlink keeps a red run of this test from littering a tracked directory.
        try:
            with pytest.raises(ValueError, match="not under a gitignored root"):
                sweep.submit(_submit_args(tracked_dir, [NOVA_MICRO]))
        finally:
            sweep._handle_path(tracked_dir, NOVA_MICRO).unlink(missing_ok=True)
        assert submitted == [], "the refusal must fire before anything is billed"

    def test_the_gitignored_default_root_is_not_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The control: the guard must not refuse the sweep's own artifacts/ default."""
        del tmp_path
        submitted: list[str] = []
        monkeypatch.setattr(
            sweep,
            "_backend",
            lambda model_id, _args, _run_id: _RecordingBackend(model_id, submitted),
        )
        repo_root = Path(__file__).resolve().parents[2]
        handle_dir = repo_root / sweep.DEFAULT_SWEEP_ROOT / "test-guard-control"
        try:
            assert sweep.submit(_submit_args(handle_dir, [NOVA_MICRO])) == 0
            assert submitted == [NOVA_MICRO]
        finally:
            for path in handle_dir.glob("*.json"):
                path.unlink()
            if handle_dir.exists():
                handle_dir.rmdir()


@pytest.mark.usefixtures("corpus_module")
class TestWrongPathMarkersTheTaskItselfPrints:
    """Advisory, and before any job exists: the check's value is highest ahead of a paid sweep."""

    def test_the_leaking_item_is_named_and_the_clean_one_is_not(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        args = parse_args(
            [
                "submit",
                "--items",
                f"{CORPUS_MODULE}:WRONG_PATH_ITEMS",
                "--handle-dir",
                str(tmp_path),
                "--max-new-tokens",
                "2048",
            ]
        )
        with caplog.at_level(logging.WARNING, logger=sweep.logger.name):
            sweep._cells(args)

        leaks = [
            record.getMessage() for record in caplog.records if "wrong-path" in record.getMessage()
        ]
        assert len(leaks) == 1
        assert LEAKING_WRONG_PATH_ITEM.id in leaks[0]
        assert CLEAN_WRONG_PATH_ITEM.id not in leaks[0]

    def test_a_corpus_with_no_leaks_says_nothing(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        args = parse_args(
            [
                "submit",
                "--items",
                f"{CORPUS_MODULE}:ITEMS",
                "--handle-dir",
                str(tmp_path),
                "--max-new-tokens",
                "2048",
            ]
        )
        with caplog.at_level(logging.WARNING, logger=sweep.logger.name):
            sweep._cells(args)
        assert [r for r in caplog.records if "wrong-path" in r.getMessage()] == []
