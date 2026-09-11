"""The shared bank of base-model step-0 eval cells (owner decision C1, 2026-09-03), offline.

`games.run_evals --banked-base-cells` derives a step-0 cell's key from everything its measurement
depends on, copies a matching bank entry byte for byte instead of generating, and with
`--bank-base-cells` publishes what it generated. Everything here runs through `--backend mock` on
CPU against a fake bucket standing in for `aws s3 sync` in both directions; no model, no network.

Per the repo rule that a check you have never watched fail is not yet a check, each guard is
exercised by committing the violation it exists to catch: a bank entry banked under another
sampler, a copy whose bytes changed under its sidecar, a copy whose summary went missing, a sidecar
beside a checkpoint cell, a sidecar claiming a step the bank can never hold, a sidecar naming a
commit its trace never saw, a bank entry whose summary is not its trace's rebuild, an entry whose
hashes agree but whose trace is missing a planned record or carries one no plan renders, a sidecar
without the consuming arm's own meta, two publishers racing on one key, a rival manifest no consumer
could take, a payload upload that fails, a manifest upload that fails, and a noise-floor cell that
must never be taken from the bank even when the bank holds it. The resumed-cell path is covered too:
a step-0 cell that died once and finished on relaunch still reaches the bank.

Two properties of the human-facing output are pinned here rather than left to review. No readout or
report may render the bank's s3:// prefix, because those documents get pasted into notes and this
repository's history is published; the entry is named by its key, and the machine-readable sidecar
keeps the prefix. And every exit path of a publish leaves a record beside the cell, because a
publish problem deliberately never fails the run and the exit code cannot carry it.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import shutil
from typing import TYPE_CHECKING, Any

import pytest

from games import battery_tables, readout, run_evals
from games.eval_model import sha256_of_file
from games.evals import (
    RECORD_META,
    SECTION_CAPABILITIES,
    SECTION_FRAMING_SWEEP,
    SECTION_GAME_BEHAVIOR,
    plan_battery,
    read_eval_records,
    rebuild_summary,
    summarise_trace,
)
from games.prompts import FRAMING_STATED_ALWAYS_COOP, FRAMING_UNSTATED
from games.report import (
    BANKED_CONSUMER_META_FIELDS,
    BANKED_FROM_KEY,
    BANKED_PROVENANCE_RECORD,
    attribute_trace,
    banked_provenance_path,
    load_traces,
    render_report,
)
from games.s3_sync import SyncOutcome

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

BANK = "s3://fake-bucket/games_rl/banked-base-cells"
BANK_PREFIX = BANK + "/"
# The key the synthetic reader fixtures pretend their copy came from; the readouts name it and must
# never name the prefix it sits under, which carries a bucket.
BANKED_FIXTURE_KEY = "0123abcd"
PRODUCER = "producer-arm"
CONSUMER = "consumer-arm"
SECTIONS_UNDER_TEST = (SECTION_GAME_BEHAVIOR, SECTION_CAPABILITIES)


class FakeBucket:
    """A directory standing in for S3: `restore` copies a prefix down, `sync` copies a directory up.

    Both directions of `aws s3 sync` as the driver uses them, minus the network. Every s3:// URI maps
    to a directory under `root`, so the test can read what the driver published and plant what the
    driver should find. A prefix that does not exist restores nothing and succeeds, which is what
    the real command does on the first launch of a run.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.restores: list[str] = []
        self.syncs: list[str] = []

    def directory(self, s3_uri: str) -> Path:
        if not s3_uri.startswith("s3://"):
            raise ValueError(f"not an s3:// URI: {s3_uri!r}")
        return self.root / s3_uri.removeprefix("s3://").strip("/")

    def restore(self, s3_dest: str, local_dir: Path) -> SyncOutcome:
        self.restores.append(s3_dest)
        source = self.directory(s3_dest)
        local_dir.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            for path in source.rglob("*"):
                if path.is_file():
                    destination = local_dir / path.relative_to(source)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, destination)
        return SyncOutcome(command=("fake", "restore"), returncode=0)

    def sync(self, local_dir: Path, s3_dest: str) -> SyncOutcome:
        self.syncs.append(s3_dest)
        destination = self.directory(s3_dest)
        for path in local_dir.rglob("*"):
            if path.is_file():
                target = destination / path.relative_to(local_dir)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        return SyncOutcome(command=("fake", "sync"), returncode=0)

    def bank_entries(self) -> list[Path]:
        """Every key directory under the bank prefix, whether or not it carries a manifest."""
        bank = self.directory(BANK_PREFIX)
        return sorted(path for path in bank.iterdir() if path.is_dir()) if bank.is_dir() else []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(run_evals, "restore_directory", self.restore)
        monkeypatch.setattr(run_evals, "sync_directory", self.sync)


@pytest.fixture
def bucket(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeBucket:
    fake = FakeBucket(tmp_path / "fake-s3")
    fake.install(monkeypatch)
    return fake


def cli(arm: str, out_dir: Path, *extra: str) -> list[str]:
    """The offline battery flags: mock backend, two games, tiny counts, two fast sections."""
    return [
        "--model",
        "fake-base",
        "--arm",
        arm,
        "--backend",
        "mock",
        "--out-dir",
        str(out_dir),
        "--games",
        "twin-pd,chicken",
        "--no-include-never-trained",
        "--capability-items",
        "3",
        "--open-ended-samples",
        "1",
        "--sections",
        ",".join(SECTIONS_UNDER_TEST),
        "--no-report",
        *extra,
    ]


def banked(*extra: str) -> list[str]:
    return ["--banked-base-cells", BANK, *extra]


def refuse_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_engine(*_args: object, **_kwargs: object) -> object:
        pytest.fail("a backend was built for a cell that should have come from the bank")

    monkeypatch.setattr(run_evals.backend_cli, "backend_from_args", no_engine)


def count_prompts_the_driver_generates(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Wrap the driver's backend so every generate call's prompt count lands in the returned list."""
    seen: list[int] = []
    build = run_evals.backend_cli.backend_from_args

    def counting(*args: Any, **kwargs: Any) -> Any:
        backend = build(*args, **kwargs)
        inner_generate = backend.generate

        def generate(prompts: list[str]) -> list[str]:
            seen.append(len(prompts))
            return inner_generate(prompts)

        backend.generate = generate
        return backend

    monkeypatch.setattr(run_evals.backend_cli, "backend_from_args", counting)
    return seen


def resolved(argv: Sequence[str]) -> tuple[Any, run_evals.EvalPlan, tuple[str, ...], Any]:
    """Parse a CLI line the way `main` does, up to the config, with no tokenizer and no engine."""
    args = run_evals._parse_args(argv)  # pyright: ignore[reportPrivateUsage]
    sections = run_evals._parse_sections(args.sections)  # pyright: ignore[reportPrivateUsage]
    plan = run_evals.resolve_plan(args)
    template = run_evals.TemplateFacts(prefilled_think=False, chat_template_kwargs=())
    config = run_evals._resolve_eval_config(  # pyright: ignore[reportPrivateUsage]
        args, plan=plan, sections=sections, template=template
    )
    return args, plan, sections, config


def identity_for(argv: Sequence[str]) -> dict[str, Any]:
    args, plan, sections, config = resolved(argv)
    return run_evals.bank_identity(args, plan, plan.targets[0], sections=sections, config=config)


def key_for(argv: Sequence[str]) -> str:
    return run_evals.bank_key(identity_for(argv))


def manifest_path(bucket: FakeBucket, key: str) -> Path:
    return bucket.directory(BANK_PREFIX + key + "/") / run_evals.BANK_MANIFEST_FILENAME


def rewrite_manifest(bucket: FakeBucket, key: str, mutate: Any) -> None:
    """Apply `mutate` to the bank entry's manifest in place, the way a hand edit or another code path would."""
    path = manifest_path(bucket, key)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def object_path(bucket: FakeBucket, key: str, local_name: str) -> Path:
    """Where the bank keeps one of an entry's files, read off the manifest that names the object."""
    manifest = json.loads(manifest_path(bucket, key).read_text(encoding="utf-8"))
    return bucket.directory(BANK_PREFIX + key + "/") / manifest["files"][local_name]["object"]


def publish_record(out_dir: Path, *, step: int = 0) -> dict[str, Any]:
    """The record `--bank-base-cells` leaves beside a cell saying what publishing it did."""
    path = run_evals.bank_publish_record_path(out_dir / f"step-{step}.jsonl")
    return json.loads(path.read_text(encoding="utf-8"))


def fail_the_nth_upload(bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch, *, nth: int) -> None:
    """Let every upload but the `nth` (1-based) reach the fake bucket; that one fails, copying nothing.

    `sync_directory` reports a failure rather than raising, which is what lets `--bank-base-cells`
    exit 0 with nothing banked, so the stand-in returns the same shape a failed `aws s3 sync` does.
    """
    calls: list[int] = []

    def sync(local_dir: Path, s3_dest: str) -> SyncOutcome:
        calls.append(len(calls) + 1)
        if len(calls) == nth:
            return SyncOutcome(command=("fake", "sync", str(local_dir), s3_dest), returncode=1)
        return bucket.sync(local_dir, s3_dest)

    monkeypatch.setattr(run_evals, "sync_directory", sync)


def plant_entry(bucket: FakeBucket, key: str, cell_dir: Path, *, identity: dict[str, Any]) -> Path:
    """Publish a complete cell under `key` the way the driver does, manifest included; returns the entry dir."""
    entry = bucket.directory(BANK_PREFIX + key + "/")
    entry.mkdir(parents=True, exist_ok=True)
    tag = sha256_of_file(cell_dir / "step-0.jsonl")[: run_evals.BANK_OBJECT_TAG_CHARS]
    objects = {
        name: run_evals.bank_object_name(name, tag)
        for name in ("step-0.jsonl", "step-0.summary.json")
    }
    for name, object_name in objects.items():
        shutil.copy2(cell_dir / name, entry / object_name)
    meta = read_eval_records(cell_dir / "step-0.jsonl")[0]
    (entry / run_evals.BANK_MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "record": run_evals.BANK_MANIFEST_RECORD,
                "bank_key": key,
                "bank_identity": json.loads(json.dumps(identity)),
                "banked_at": "2026-09-03T00:00:00+00:00",
                "banked_by_git_sha": "planted",
                "source_arm": meta["arm"],
                "source_git_sha": meta["git_sha"],
                "files": {
                    name: {
                        "object": object_name,
                        "sha256": sha256_of_file(entry / object_name),
                        "bytes": (entry / object_name).stat().st_size,
                    }
                    for name, object_name in objects.items()
                },
            }
        ),
        encoding="utf-8",
    )
    return entry


# A vLLM-backend request, parsed but never served: the sampler flags only resolve on a local backend.
BASELINE: tuple[str, ...] = (
    "--model",
    "fake-base",
    "--arm",
    PRODUCER,
    "--backend",
    "vllm",
    "--sampler",
    "training-distribution",
    "--thinking",
    "--games",
    "twin-pd,chicken",
    "--no-include-never-trained",
    "--capability-items",
    "3",
    "--open-ended-samples",
    "1",
    "--sections",
    ",".join(SECTIONS_UNDER_TEST),
)


class TestBankKeyDerivation:
    """The key is a hash of everything the measurement depends on, and of nothing else."""

    def test_the_same_request_derives_the_same_key_twice(self) -> None:
        assert key_for(BASELINE) == key_for(list(BASELINE))
        assert len(key_for(BASELINE)) == 64

    def test_the_arm_is_not_part_of_the_key(self) -> None:
        """Sharing across arms on one base is the point, so two arms derive one key."""
        other_arm = [*BASELINE]
        other_arm[other_arm.index(PRODUCER)] = CONSUMER
        assert key_for(BASELINE) == key_for(other_arm)

    @pytest.mark.parametrize(
        ("what", "extra"),
        [
            ("sampler mode", ["--sampler", "deployment"]),
            ("max_new_tokens", ["--max-new-tokens", "777"]),
            ("temperature", ["--temperature", "0.3"]),
            ("thinking", ["--no-thinking"]),
            ("sections", ["--sections", SECTION_CAPABILITIES]),
            ("capability items", ["--capability-items", "5"]),
            ("print orders", ["--label-print-order", "both"]),
            ("game-behaviour draws", ["--game-behavior-samples", "2"]),
            ("games", ["--games", "chicken"]),
            ("base model", ["--model", "other-base"]),
            ("backend kind", ["--backend", "hf"]),
        ],
    )
    def test_changing_one_input_changes_the_key(self, what: str, extra: list[str]) -> None:
        variant = list(BASELINE)
        for flag, value in zip(extra[::2], extra[1::2], strict=False):
            if flag in variant:
                variant[variant.index(flag) + 1] = value
            else:
                variant.extend([flag, value])
        if len(extra) % 2 == 1:
            lone = extra[-1]
            if lone == "--no-thinking":
                variant[variant.index("--thinking")] = lone
            else:
                variant.append(lone)
        assert key_for(variant) != key_for(BASELINE), what

    def test_the_sampler_reaches_the_identity_by_name_and_by_knob(self) -> None:
        deployment = [*BASELINE]
        deployment[deployment.index("training-distribution")] = "deployment"
        base_identity = identity_for(BASELINE)
        other_identity = identity_for(deployment)
        assert base_identity["sampler_mode"] == "training-distribution"
        assert other_identity["sampler_mode"] == "deployment"
        assert base_identity["sampling"] != other_identity["sampling"]
        assert "max_new_tokens" in base_identity["sampling"]

    def test_the_machine_local_item_paths_are_not_part_of_the_key(self, tmp_path: Path) -> None:
        """The same items read from another directory are the same cell; content is what the digest keys."""
        here = [*BASELINE, "--survey-data-dir", str(tmp_path / "here")]
        there = [*BASELINE, "--survey-data-dir", str(tmp_path / "there")]
        assert key_for(here) == key_for(there)
        assert "survey_data_dir" not in identity_for(here)["eval_config"]
        assert "dtbench_dir" not in identity_for(here)["eval_config"]

    def test_the_prompt_set_digest_moves_with_one_prompt(self) -> None:
        _, _, sections, config = resolved(BASELINE)
        plan = plan_battery(sections, config)
        digest = run_evals.prompt_set_digest(plan)
        altered = [*plan]
        altered[3] = dataclasses.replace(altered[3], prompt=altered[3].prompt + " ")
        assert run_evals.prompt_set_digest(altered) != digest
        assert run_evals.prompt_set_digest(plan[:-1]) != digest
        assert run_evals.prompt_set_digest(list(plan)) == digest

    def test_the_key_compares_identities_through_json_types(self) -> None:
        identity = identity_for(BASELINE)
        as_json = json.loads(json.dumps(identity))
        as_tuples = {**identity, "sections": tuple(identity["sections"])}
        assert run_evals.bank_key(as_json) == run_evals.bank_key(identity)
        assert run_evals.bank_key(as_tuples) == run_evals.bank_key(identity)


class TestTheNoiseFloorRule:
    def test_the_unstated_only_sweep_is_the_default_floor(self) -> None:
        assert run_evals.NOISE_FLOOR_FRAMINGS == (FRAMING_UNSTATED,)
        _, _, sections, config = resolved(
            [
                *BASELINE[:-2],
                "--sections",
                SECTION_FRAMING_SWEEP,
                "--counterpart-framings",
                FRAMING_UNSTATED,
                "--framing-sweep-games",
                "twin-pd",
            ]
        )
        assert run_evals.is_noise_floor_cell(sections, config, run_evals.NOISE_FLOOR_FRAMINGS)

    def test_a_sweep_over_more_than_the_floor_framings_is_banked(self) -> None:
        _, _, sections, config = resolved(
            [
                *BASELINE[:-2],
                "--sections",
                SECTION_FRAMING_SWEEP,
                "--counterpart-framings",
                f"{FRAMING_UNSTATED},{FRAMING_STATED_ALWAYS_COOP}",
                "--framing-sweep-games",
                "twin-pd",
            ]
        )
        assert not run_evals.is_noise_floor_cell(sections, config, run_evals.NOISE_FLOOR_FRAMINGS)

    def test_a_cell_without_the_sweep_is_never_the_floor(self) -> None:
        _, _, sections, config = resolved(BASELINE)
        assert not run_evals.is_noise_floor_cell(sections, config, run_evals.NOISE_FLOOR_FRAMINGS)


class TestTheDriverTakesAndPublishesBankedCells:
    def test_a_generated_step_zero_cell_is_published_under_its_key_with_its_hashes(
        self, tmp_path: Path, bucket: FakeBucket
    ) -> None:
        out_dir = tmp_path / PRODUCER
        assert run_evals.main(cli(PRODUCER, out_dir, *banked("--bank-base-cells"))) == 0
        key = key_for(cli(PRODUCER, out_dir))
        (entry,) = bucket.bank_entries()
        assert entry.name == key
        manifest = json.loads(manifest_path(bucket, key).read_text(encoding="utf-8"))
        assert manifest["record"] == run_evals.BANK_MANIFEST_RECORD
        assert manifest["bank_key"] == key
        assert manifest["source_arm"] == PRODUCER
        assert (
            manifest["source_git_sha"] == read_eval_records(out_dir / "step-0.jsonl")[0]["git_sha"]
        )
        assert manifest["bank_identity"] == json.loads(
            json.dumps(identity_for(cli(PRODUCER, out_dir)))
        )
        # The payload sits under content-tagged names the manifest points at, never the cell's own.
        tag = sha256_of_file(out_dir / "step-0.jsonl")[: run_evals.BANK_OBJECT_TAG_CHARS]
        for name in ("step-0.jsonl", "step-0.summary.json"):
            facts = manifest["files"][name]
            assert facts["object"] == run_evals.bank_object_name(name, tag)
            published = entry / facts["object"]
            assert published.read_bytes() == (out_dir / name).read_bytes()
            assert facts["sha256"] == sha256_of_file(published)
            assert facts["bytes"] == published.stat().st_size
            assert not (entry / name).exists()
        assert run_evals.bank_object_name("step-0.jsonl", "abc") == "step-0.abc.jsonl"
        assert run_evals.bank_object_name("step-0.summary.json", "abc") == "step-0.abc.summary.json"
        # The manifest is uploaded after the files, so an entry with one is whole.
        assert bucket.syncs.count(BANK_PREFIX + key + "/") == 2
        record = publish_record(out_dir)
        assert record["outcome"] == run_evals.BANK_PUBLISH_PUBLISHED
        assert record["bank_key"] == key
        assert record["failed_sync_returncode"] is None

    def test_a_consumer_takes_the_banked_cell_byte_for_byte_and_writes_its_provenance(
        self, tmp_path: Path, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        producer_dir = tmp_path / PRODUCER
        assert run_evals.main(cli(PRODUCER, producer_dir, *banked("--bank-base-cells"))) == 0
        key = key_for(cli(PRODUCER, producer_dir))
        consumer_dir = tmp_path / CONSUMER
        refuse_engine(monkeypatch)
        assert run_evals.main(cli(CONSUMER, consumer_dir, *banked())) == 0

        trace = consumer_dir / "step-0.jsonl"
        summary = consumer_dir / "step-0.summary.json"
        assert trace.read_bytes() == (producer_dir / "step-0.jsonl").read_bytes()
        assert summary.read_bytes() == (producer_dir / "step-0.summary.json").read_bytes()
        assert rebuild_summary(trace) == json.loads(summary.read_text(encoding="utf-8"))
        assert not consumer_dir.with_name(CONSUMER + run_evals.BANK_STAGING_SUFFIX).exists()

        sidecar = json.loads(banked_provenance_path(trace).read_text(encoding="utf-8"))
        assert sidecar["record"] == BANKED_PROVENANCE_RECORD
        assert sidecar["consumer_arm"] == CONSUMER
        assert sidecar["step"] == 0
        assert sidecar["source_arm"] == PRODUCER
        assert sidecar["bank_key"] == key
        assert sidecar["source_key"] == BANK_PREFIX + key + "/"
        assert sidecar["files"]["step-0.jsonl"]["sha256"] == sha256_of_file(trace)
        assert sidecar["files"]["step-0.summary.json"]["sha256"] == sha256_of_file(summary)
        assert sidecar["bank_identity"] == json.loads(
            json.dumps(identity_for(cli(CONSUMER, consumer_dir)))
        )
        assert set(sidecar["consumer_meta"]) == set(BANKED_CONSUMER_META_FIELDS)
        # The copy's own meta still names the producer; the readers relabel it through the sidecar.
        assert read_eval_records(trace)[0]["arm"] == PRODUCER
        (loaded,) = load_traces([trace])
        assert loaded.arm == CONSUMER
        assert loaded.meta["arm"] == CONSUMER
        assert loaded.banked_from is not None
        assert loaded.banked_from["source_arm"] == PRODUCER
        assert f"banked from {PRODUCER}@" in loaded.label
        # Nothing was published back: the consumer did not ask to bank.
        assert [entry.name for entry in bucket.bank_entries()] == [key]

    def test_a_cell_asking_for_another_config_generates_instead(
        self, tmp_path: Path, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert run_evals.main(cli(PRODUCER, tmp_path / PRODUCER, *banked("--bank-base-cells"))) == 0
        generated = count_prompts_the_driver_generates(monkeypatch)
        consumer_dir = tmp_path / CONSUMER
        assert (
            run_evals.main(cli(CONSUMER, consumer_dir, "--capability-items", "4", *banked())) == 0
        )
        assert sum(generated) > 0
        assert not banked_provenance_path(consumer_dir / "step-0.jsonl").exists()
        assert read_eval_records(consumer_dir / "step-0.jsonl")[0]["arm"] == CONSUMER
        assert len(bucket.bank_entries()) == 1

    def test_a_bank_entry_banked_under_another_sampler_is_refused_not_taken(
        self, tmp_path: Path, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The key is a hash of the identity, so a sampler mismatch under one key is a collision or a
        hand-placed entry, and taking it would report another sampler's draw under this arm."""
        assert run_evals.main(cli(PRODUCER, tmp_path / PRODUCER, *banked("--bank-base-cells"))) == 0
        key = key_for(cli(PRODUCER, tmp_path / PRODUCER))

        def other_sampler(manifest: dict[str, Any]) -> None:
            manifest["bank_identity"]["sampler_mode"] = "deployment"

        rewrite_manifest(bucket, key, other_sampler)
        refuse_engine(monkeypatch)
        consumer_dir = tmp_path / CONSUMER
        with pytest.raises(ValueError, match="sampler_mode"):
            run_evals.main(cli(CONSUMER, consumer_dir, *banked()))
        assert not (consumer_dir / "step-0.jsonl").exists()
        assert not consumer_dir.with_name(CONSUMER + run_evals.BANK_STAGING_SUFFIX).exists()

    def test_a_bank_entry_whose_summary_is_not_its_traces_rebuild_is_refused(
        self, tmp_path: Path, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert run_evals.main(cli(PRODUCER, tmp_path / PRODUCER, *banked("--bank-base-cells"))) == 0
        key = key_for(cli(PRODUCER, tmp_path / PRODUCER))
        summary_path = object_path(bucket, key, "step-0.summary.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary[SECTION_CAPABILITIES]["n_records"] += 1
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        def rehash(manifest: dict[str, Any]) -> None:
            manifest["files"]["step-0.summary.json"]["sha256"] = sha256_of_file(summary_path)

        rewrite_manifest(bucket, key, rehash)
        refuse_engine(monkeypatch)
        with pytest.raises(ValueError, match="not the rebuild"):
            run_evals.main(cli(CONSUMER, tmp_path / CONSUMER, *banked()))

    def test_a_bank_entry_whose_bytes_do_not_match_its_manifest_is_refused(
        self, tmp_path: Path, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert run_evals.main(cli(PRODUCER, tmp_path / PRODUCER, *banked("--bank-base-cells"))) == 0
        key = key_for(cli(PRODUCER, tmp_path / PRODUCER))
        trace = object_path(bucket, key, "step-0.jsonl")
        trace.write_bytes(trace.read_bytes() + b"\n")
        refuse_engine(monkeypatch)
        with pytest.raises(ValueError, match="hashes to"):
            run_evals.main(cli(CONSUMER, tmp_path / CONSUMER, *banked()))

    def test_a_bank_entry_missing_planned_records_is_refused_even_when_its_hashes_agree(
        self, tmp_path: Path, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The hashes only prove the entry is internally consistent; the resume gate proves it is
        this cell, complete. An entry whose trace lost its last record, with its summary rebuilt over
        the shorter trace and both hashes recorded afresh, passes every consistency check and is
        still a partial cell that must not stand in for this arm's step 0."""
        assert run_evals.main(cli(PRODUCER, tmp_path / PRODUCER, *banked("--bank-base-cells"))) == 0
        key = key_for(cli(PRODUCER, tmp_path / PRODUCER))
        trace = object_path(bucket, key, "step-0.jsonl")
        summary = object_path(bucket, key, "step-0.summary.json")
        lines = trace.read_text(encoding="utf-8").splitlines(keepends=True)
        trace.write_text("".join(lines[:-1]), encoding="utf-8")
        summary.write_text(
            json.dumps(summarise_trace(read_eval_records(trace)), indent=2), encoding="utf-8"
        )

        def rehash(manifest: dict[str, Any]) -> None:
            manifest["files"]["step-0.jsonl"]["sha256"] = sha256_of_file(trace)
            manifest["files"]["step-0.summary.json"]["sha256"] = sha256_of_file(summary)

        rewrite_manifest(bucket, key, rehash)
        refuse_engine(monkeypatch)
        consumer_dir = tmp_path / CONSUMER
        with pytest.raises(ValueError, match="1 missing"):
            run_evals.main(cli(CONSUMER, consumer_dir, *banked()))
        assert not (consumer_dir / "step-0.jsonl").exists()

    def test_a_bank_entry_whose_trace_carries_a_record_this_plan_never_renders_is_refused(
        self, tmp_path: Path, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A record from another configuration, planted with the hashes made to agree, trips the
        same orphan-identity refusal the resume path applies to a partial trace."""
        assert run_evals.main(cli(PRODUCER, tmp_path / PRODUCER, *banked("--bank-base-cells"))) == 0
        key = key_for(cli(PRODUCER, tmp_path / PRODUCER))
        trace = object_path(bucket, key, "step-0.jsonl")
        summary = object_path(bucket, key, "step-0.summary.json")
        records = read_eval_records(trace)
        # A behaviour record is keyed on its prompt id, so renaming it makes an identity no plan renders.
        behaviour = next(
            record for record in records[1:] if record["record"] == SECTION_GAME_BEHAVIOR
        )
        orphan = {**behaviour, "prompt_id": "never-planned"}
        trace.write_text(
            "".join(json.dumps(record) + "\n" for record in [*records, orphan]), encoding="utf-8"
        )
        summary.write_text(
            json.dumps(summarise_trace(read_eval_records(trace)), indent=2), encoding="utf-8"
        )

        def rehash(manifest: dict[str, Any]) -> None:
            manifest["files"]["step-0.jsonl"]["sha256"] = sha256_of_file(trace)
            manifest["files"]["step-0.summary.json"]["sha256"] = sha256_of_file(summary)

        rewrite_manifest(bucket, key, rehash)
        refuse_engine(monkeypatch)
        with pytest.raises(ValueError, match="never renders"):
            run_evals.main(cli(CONSUMER, tmp_path / CONSUMER, *banked()))

    def test_a_resumed_step_zero_cell_is_published_once_it_completes(
        self, tmp_path: Path, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cell that died once and was relaunched is the normal path for a 12 h step 0, and it has
        to reach the bank like a cell whose first session finished; only the bank lookup is skipped,
        because the partial trace is resumed rather than replaced."""
        consumer_dir = tmp_path / CONSUMER
        assert run_evals.main(cli(CONSUMER, consumer_dir)) == 0
        trace = consumer_dir / "step-0.jsonl"
        lines = trace.read_text(encoding="utf-8").splitlines(keepends=True)
        trace.write_text("".join(lines[:5]), encoding="utf-8")
        (consumer_dir / "step-0.summary.json").unlink()
        generated = count_prompts_the_driver_generates(monkeypatch)
        assert run_evals.main(cli(CONSUMER, consumer_dir, *banked("--bank-base-cells"))) == 0
        assert sum(generated) == len(lines) - 5
        key = key_for(cli(CONSUMER, consumer_dir))
        entry_prefix = BANK_PREFIX + key + "/"
        manifest = json.loads(manifest_path(bucket, key).read_text(encoding="utf-8"))
        assert manifest["source_arm"] == CONSUMER
        assert manifest["source_n_sessions"] == 2
        assert object_path(bucket, key, "step-0.jsonl").read_bytes() == trace.read_bytes()
        # One restore only: the publish re-check. The partial trace was never looked up in the bank.
        assert bucket.restores.count(entry_prefix) == 1
        assert bucket.syncs.count(entry_prefix) == 2

    def test_two_publishers_racing_on_one_key_leave_an_entry_every_consumer_can_take(
        self, tmp_path: Path, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two arms finish step 0 minutes apart, each re-reads the entry before its manifest has
        landed, each uploads. Neither payload may overwrite the other's, and whichever manifest
        survives has to name objects that exist and hash as it recorded them."""
        first_dir = tmp_path / PRODUCER
        assert run_evals.main(cli(PRODUCER, first_dir, *banked("--bank-base-cells"))) == 0
        key = key_for(cli(PRODUCER, first_dir))
        entry = bucket.directory(BANK_PREFIX + key + "/")
        first_manifest = json.loads(manifest_path(bucket, key).read_text(encoding="utf-8"))
        first_objects = {
            facts["object"]: facts["sha256"] for facts in first_manifest["files"].values()
        }

        def restore_as_before_the_first_manifest_landed(
            s3_dest: str, local_dir: Path
        ) -> SyncOutcome:
            outcome = bucket.restore(s3_dest, local_dir)
            (local_dir / run_evals.BANK_MANIFEST_FILENAME).unlink(missing_ok=True)
            return outcome

        monkeypatch.setattr(
            run_evals, "restore_directory", restore_as_before_the_first_manifest_landed
        )
        second_dir = tmp_path / "rival"
        assert run_evals.main(cli("rival-arm", second_dir, *banked("--bank-base-cells"))) == 0
        monkeypatch.setattr(run_evals, "restore_directory", bucket.restore)

        surviving = json.loads(manifest_path(bucket, key).read_text(encoding="utf-8"))
        assert surviving["source_arm"] == "rival-arm"
        second_objects = {facts["object"]: facts["sha256"] for facts in surviving["files"].values()}
        assert set(first_objects).isdisjoint(second_objects)
        for object_name, sha in {**first_objects, **second_objects}.items():
            assert sha256_of_file(entry / object_name) == sha, object_name
        refuse_engine(monkeypatch)
        consumer_dir = tmp_path / CONSUMER
        assert run_evals.main(cli(CONSUMER, consumer_dir, *banked())) == 0
        assert (consumer_dir / "step-0.jsonl").read_bytes() == (
            second_dir / "step-0.jsonl"
        ).read_bytes()
        sidecar = json.loads(banked_provenance_path(consumer_dir / "step-0.jsonl").read_text())
        assert sidecar["source_arm"] == "rival-arm"

    def test_the_sidecars_consumer_meta_is_exactly_what_the_readers_overlay(
        self, tmp_path: Path
    ) -> None:
        """The driver writes the fields, the reader overlays them by name; the two lists have to be one."""
        _, plan, _, _ = resolved(cli(PRODUCER, tmp_path / PRODUCER))
        consumer_meta = run_evals._consumer_meta(plan)  # pyright: ignore[reportPrivateUsage]
        assert set(consumer_meta) == set(BANKED_CONSUMER_META_FIELDS)
        assert consumer_meta["grading"] == plan.grading
        assert consumer_meta["run_dir"] == (None if plan.run_dir is None else str(plan.run_dir))

    def test_a_half_published_entry_without_a_manifest_is_generated_over(
        self, tmp_path: Path, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert run_evals.main(cli(PRODUCER, tmp_path / PRODUCER, *banked("--bank-base-cells"))) == 0
        key = key_for(cli(PRODUCER, tmp_path / PRODUCER))
        manifest_path(bucket, key).unlink()
        generated = count_prompts_the_driver_generates(monkeypatch)
        consumer_dir = tmp_path / CONSUMER
        assert run_evals.main(cli(CONSUMER, consumer_dir, *banked("--bank-base-cells"))) == 0
        assert sum(generated) > 0
        assert not banked_provenance_path(consumer_dir / "step-0.jsonl").exists()
        # The consumer completed the entry: its own cell now sits under the key, manifest last.
        manifest = json.loads(manifest_path(bucket, key).read_text(encoding="utf-8"))
        assert manifest["source_arm"] == CONSUMER

    def test_the_noise_floor_cell_is_never_published_and_never_taken(
        self, tmp_path: Path, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        floor = [
            "--sections",
            SECTION_FRAMING_SWEEP,
            "--counterpart-framings",
            FRAMING_UNSTATED,
            "--framing-sweep-games",
            "twin-pd",
        ]
        producer_dir = tmp_path / PRODUCER
        assert (
            run_evals.main(cli(PRODUCER, producer_dir, *floor, *banked("--bank-base-cells"))) == 0
        )
        assert bucket.bank_entries() == []
        assert (producer_dir / "step-0.summary.json").exists()

        # Plant the very entry the cell would key to, complete and consistent, and it is still not taken.
        identity = identity_for(cli(PRODUCER, producer_dir, *floor))
        key = run_evals.bank_key(identity)
        plant_entry(bucket, key, producer_dir, identity=identity)
        generated = count_prompts_the_driver_generates(monkeypatch)
        consumer_dir = tmp_path / CONSUMER
        assert (
            run_evals.main(cli(CONSUMER, consumer_dir, *floor, *banked("--bank-base-cells"))) == 0
        )
        assert sum(generated) > 0
        assert not banked_provenance_path(consumer_dir / "step-0.jsonl").exists()
        assert read_eval_records(consumer_dir / "step-0.jsonl")[0]["arm"] == CONSUMER
        assert bucket.restores.count(BANK_PREFIX + key + "/") == 0

    def test_an_empty_noise_floor_list_banks_the_sweep_cell_and_says_so(
        self, tmp_path: Path, bucket: FakeBucket, caplog: pytest.LogCaptureFixture
    ) -> None:
        floor = [
            "--sections",
            SECTION_FRAMING_SWEEP,
            "--counterpart-framings",
            FRAMING_UNSTATED,
            "--framing-sweep-games",
            "twin-pd",
            "--noise-floor-framings",
            "",
        ]
        assert (
            run_evals.main(cli(PRODUCER, tmp_path / PRODUCER, *floor, *banked("--bank-base-cells")))
            == 0
        )
        assert len(bucket.bank_entries()) == 1
        assert "keeps no per-arm step-0 draw" in caplog.text

    def test_a_partial_trace_is_resumed_rather_than_replaced_by_the_banks_copy(
        self, tmp_path: Path, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert run_evals.main(cli(PRODUCER, tmp_path / PRODUCER, *banked("--bank-base-cells"))) == 0
        consumer_dir = tmp_path / CONSUMER
        assert run_evals.main(cli(CONSUMER, consumer_dir)) == 0
        trace = consumer_dir / "step-0.jsonl"
        lines = trace.read_text(encoding="utf-8").splitlines(keepends=True)
        trace.write_text("".join(lines[:5]), encoding="utf-8")
        (consumer_dir / "step-0.summary.json").unlink()
        generated = count_prompts_the_driver_generates(monkeypatch)
        assert run_evals.main(cli(CONSUMER, consumer_dir, *banked())) == 0
        assert sum(generated) == len(lines) - 5
        assert trace.read_text(encoding="utf-8").splitlines(keepends=True)[1:] == lines[1:]
        assert not banked_provenance_path(trace).exists()
        assert read_eval_records(trace)[0]["arm"] == CONSUMER

    def test_another_runs_entry_is_kept_when_two_arms_publish_the_same_key(
        self, tmp_path: Path, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The reuse check ran before generation; a rival that banked a takeable entry meanwhile wins.

        The rival's entry is planted exactly as the driver publishes one -- this key, content-tagged
        objects, real hashes -- because "keeping theirs" is only the right call when theirs is an
        entry a later consumer can actually take, which the consumer run at the end proves.
        """
        rival_dir = tmp_path / "rival"
        assert run_evals.main(cli("rival-arm", rival_dir)) == 0
        key = key_for(cli("rival-arm", rival_dir))
        evaluate = run_evals._evaluate_target  # pyright: ignore[reportPrivateUsage]

        def evaluate_then_lose_the_race(*args: Any, **kwargs: Any) -> None:
            evaluate(*args, **kwargs)
            plant_entry(bucket, key, rival_dir, identity=identity_for(cli("rival-arm", rival_dir)))

        monkeypatch.setattr(run_evals, "_evaluate_target", evaluate_then_lose_the_race)
        out_dir = tmp_path / PRODUCER
        assert run_evals.main(cli(PRODUCER, out_dir, *banked("--bank-base-cells"))) == 0
        manifest = json.loads(manifest_path(bucket, key).read_text(encoding="utf-8"))
        assert manifest["source_arm"] == "rival-arm"
        assert (
            object_path(bucket, key, "step-0.jsonl").read_bytes()
            == (rival_dir / "step-0.jsonl").read_bytes()
        )
        record = publish_record(out_dir)
        assert record["outcome"] == run_evals.BANK_PUBLISH_KEPT_RIVAL
        assert "rival-arm" in str(record["detail"])
        consumer_dir = tmp_path / CONSUMER
        refuse_engine(monkeypatch)
        assert run_evals.main(cli(CONSUMER, consumer_dir, *banked())) == 0
        assert (consumer_dir / "step-0.jsonl").read_bytes() == (
            rival_dir / "step-0.jsonl"
        ).read_bytes()

    def test_a_rival_manifest_no_consumer_could_take_is_left_alone_and_recorded(
        self, tmp_path: Path, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A rival manifest naming no key and no objects blocks the key, and only a human may clear it.

        Overwriting another publisher's manifest is a destructive act on shared state, and raising
        would throw away a completed cell over a publish-side problem, so the run keeps its hands
        off, exits 0, and says what was wrong where the run's own artifacts will carry it.
        """
        rival_dir = tmp_path / "rival"
        assert run_evals.main(cli("rival-arm", rival_dir)) == 0
        key = key_for(cli("rival-arm", rival_dir))
        evaluate = run_evals._evaluate_target  # pyright: ignore[reportPrivateUsage]
        torn = json.dumps({"record": run_evals.BANK_MANIFEST_RECORD, "source_arm": "rival-arm"})

        def evaluate_then_find_a_torn_rival(*args: Any, **kwargs: Any) -> None:
            evaluate(*args, **kwargs)
            entry = bucket.directory(BANK_PREFIX + key + "/")
            entry.mkdir(parents=True, exist_ok=True)
            for name in ("step-0.jsonl", "step-0.summary.json"):
                shutil.copy2(rival_dir / name, entry / name)
            (entry / run_evals.BANK_MANIFEST_FILENAME).write_text(torn, encoding="utf-8")

        monkeypatch.setattr(run_evals, "_evaluate_target", evaluate_then_find_a_torn_rival)
        out_dir = tmp_path / PRODUCER
        assert run_evals.main(cli(PRODUCER, out_dir, *banked("--bank-base-cells"))) == 0
        assert manifest_path(bucket, key).read_text(encoding="utf-8") == torn
        entry = bucket.directory(BANK_PREFIX + key + "/")
        assert sorted(path.name for path in entry.iterdir()) == [
            run_evals.BANK_MANIFEST_FILENAME,
            "step-0.jsonl",
            "step-0.summary.json",
        ]
        record = publish_record(out_dir)
        assert record["outcome"] == run_evals.BANK_PUBLISH_INVALID_RIVAL
        assert f"it names bank key None rather than {key!r}" in record["detail"]
        assert "names no bank object for step-0.jsonl" in record["detail"]
        assert (out_dir / "step-0.summary.json").exists()

    def test_a_failed_payload_upload_keeps_the_cell_exits_zero_and_records_why(
        self, tmp_path: Path, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The eval ladder is the expensive thing, so a publish failure is recorded rather than raised."""
        fail_the_nth_upload(bucket, monkeypatch, nth=1)
        out_dir = tmp_path / PRODUCER
        assert run_evals.main(cli(PRODUCER, out_dir, *banked("--bank-base-cells"))) == 0
        key = key_for(cli(PRODUCER, out_dir))
        assert not manifest_path(bucket, key).exists()
        record = publish_record(out_dir)
        assert record["record"] == run_evals.BANK_PUBLISH_RECORD
        assert record["outcome"] == run_evals.BANK_PUBLISH_FILES_FAILED
        assert record["bank_key"] == key
        assert record["entry_prefix"] == BANK_PREFIX + key + "/"
        assert record["failed_sync_returncode"] == 1
        assert BANK_PREFIX + key + "/" in record["failed_sync_command"]
        assert (out_dir / "step-0.summary.json").exists()

    def test_a_failed_manifest_upload_records_the_half_published_entry(
        self, tmp_path: Path, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The payload landed and the completion marker did not, which is a different failure to read."""
        fail_the_nth_upload(bucket, monkeypatch, nth=2)
        out_dir = tmp_path / PRODUCER
        assert run_evals.main(cli(PRODUCER, out_dir, *banked("--bank-base-cells"))) == 0
        key = key_for(cli(PRODUCER, out_dir))
        assert not manifest_path(bucket, key).exists()
        tag = sha256_of_file(out_dir / "step-0.jsonl")[: run_evals.BANK_OBJECT_TAG_CHARS]
        entry = bucket.directory(BANK_PREFIX + key + "/")
        assert (entry / run_evals.bank_object_name("step-0.jsonl", tag)).is_file()
        record = publish_record(out_dir)
        assert record["outcome"] == run_evals.BANK_PUBLISH_MANIFEST_FAILED
        assert record["failed_sync_returncode"] == 1
        assert run_evals.BANK_MANIFEST_FILENAME in record["detail"]

    def test_the_arms_report_renders_over_the_copy_and_names_its_source(
        self, tmp_path: Path, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The driver's own end-of-run report reads the copy through the sidecar like every reader."""
        del bucket
        assert run_evals.main(cli(PRODUCER, tmp_path / PRODUCER, *banked("--bank-base-cells"))) == 0
        consumer_dir = tmp_path / CONSUMER
        refuse_engine(monkeypatch)
        with_report = [
            flag for flag in cli(CONSUMER, consumer_dir, *banked()) if flag != "--no-report"
        ]
        assert run_evals.main(with_report) == 0
        report = (consumer_dir / "report.md").read_text(encoding="utf-8")
        assert f"{CONSUMER}@0 (mock) (banked from {PRODUCER}@" in report
        assert "byte-identical copies" in report
        # The entry is named by its key, and the bucket it lives in never reaches a document that
        # gets pasted into notes; the sidecar beside the trace keeps the full prefix.
        assert key_for(cli(CONSUMER, consumer_dir)) in report
        assert BANK_PREFIX not in report
        assert "s3://" not in report


class TestRelaunchingOverABankedCopy:
    def _take(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *extra: str) -> Path:
        assert run_evals.main(cli(PRODUCER, tmp_path / PRODUCER, *banked("--bank-base-cells"))) == 0
        consumer_dir = tmp_path / CONSUMER
        refuse_engine(monkeypatch)
        assert run_evals.main(cli(CONSUMER, consumer_dir, *banked(*extra))) == 0
        return consumer_dir

    def test_a_sync_dest_relaunch_skips_the_copy_and_the_run_prefix_holds_all_three_files(
        self, tmp_path: Path, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_prefix = "s3://fake-bucket/runs/consumer/evals/"
        consumer_dir = self._take(tmp_path, monkeypatch, "--sync-dest", run_prefix)
        mirrored = bucket.directory(run_prefix)
        assert {path.name for path in mirrored.iterdir()} == {
            "step-0.jsonl",
            "step-0.summary.json",
            "step-0.banked-from.json",
        }
        before = {path.name: path.read_bytes() for path in consumer_dir.iterdir()}
        assert run_evals.main(cli(CONSUMER, consumer_dir, *banked("--sync-dest", run_prefix))) == 0
        assert {path.name: path.read_bytes() for path in consumer_dir.iterdir()} == before
        # A fresh box: nothing local, the run prefix restores the copy, and it is skipped as complete.
        fresh_dir = tmp_path / "fresh"
        assert run_evals.main(cli(CONSUMER, fresh_dir, *banked("--sync-dest", run_prefix))) == 0
        assert {path.name: path.read_bytes() for path in fresh_dir.iterdir()} == before

    def test_a_relaunch_without_sync_dest_refuses_the_complete_copy_like_any_complete_cell(
        self, tmp_path: Path, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        del bucket
        consumer_dir = self._take(tmp_path, monkeypatch)
        with pytest.raises(FileExistsError, match="complete eval trace"):
            run_evals.main(cli(CONSUMER, consumer_dir, *banked()))

    def test_a_copy_whose_bytes_changed_is_refused_by_the_driver_and_every_reader(
        self, tmp_path: Path, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_prefix = "s3://fake-bucket/runs/consumer/evals/"
        consumer_dir = self._take(tmp_path, monkeypatch, "--sync-dest", run_prefix)
        del bucket
        trace = consumer_dir / "step-0.jsonl"
        trace.write_bytes(trace.read_bytes() + b"\n")
        with pytest.raises(ValueError, match="not the banked copy"):
            load_traces([trace])
        with pytest.raises(ValueError, match="not the banked copy"):
            readout.read_cell(trace)
        with pytest.raises(ValueError, match="not the banked copy"):
            run_evals.main(cli(CONSUMER, consumer_dir, *banked("--sync-dest", run_prefix)))

    def test_a_copy_whose_summary_went_missing_is_refused_with_the_recovery_named(
        self, tmp_path: Path, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        del bucket
        consumer_dir = self._take(tmp_path, monkeypatch)
        (consumer_dir / "step-0.summary.json").unlink()
        with pytest.raises(FileExistsError, match="torn"):
            run_evals.main(cli(CONSUMER, consumer_dir, *banked()))

    def test_a_copy_taken_for_another_arm_is_refused(
        self, tmp_path: Path, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        del bucket
        consumer_dir = self._take(tmp_path, monkeypatch)
        with pytest.raises(FileExistsError, match="copied for"):
            run_evals.main(cli("third-arm", consumer_dir, *banked("--sync-dest", "s3://b/p/")))

    def test_a_copy_whose_request_changed_is_refused_before_anything_loads(
        self, tmp_path: Path, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        del bucket
        consumer_dir = self._take(tmp_path, monkeypatch)
        with pytest.raises(FileExistsError, match="sections"):
            run_evals.main(
                cli(
                    CONSUMER,
                    consumer_dir,
                    "--sections",
                    SECTION_CAPABILITIES,
                    *banked("--sync-dest", "s3://b/p/"),
                )
            )

    def test_a_sidecar_beside_a_checkpoint_cell_is_refused(
        self, tmp_path: Path, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        del bucket, monkeypatch
        run_dir = tmp_path / "run"
        checkpoint = run_dir / "checkpoint-5"
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "fake-base"}), encoding="utf-8"
        )
        out_dir = tmp_path / "out"
        argv = [
            "--checkpoint",
            str(checkpoint),
            *cli(CONSUMER, out_dir)[2:],
        ]
        assert run_evals.main(argv) == 0
        trace = out_dir / "step-5.jsonl"
        banked_provenance_path(trace).write_text(
            json.dumps(
                {
                    "record": BANKED_PROVENANCE_RECORD,
                    "consumer_arm": CONSUMER,
                    "step": 5,
                    "source_arm": CONSUMER,
                    "files": {trace.name: {"sha256": sha256_of_file(trace)}},
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(FileExistsError, match="never be a banked copy"):
            run_evals.main([*argv, "--sync-dest", "s3://b/p/"])


class TestTheBankFlagsRefuseWhatTheyCannotDo:
    def test_publishing_without_a_bank_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="no bank was named"):
            run_evals.main(cli(PRODUCER, tmp_path / "out", "--bank-base-cells"))

    def test_the_bank_under_summarise_only_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="summarise-only"):
            run_evals.main(cli(PRODUCER, tmp_path / "out", "--summarise-only", *banked()))

    def test_a_bank_that_is_not_s3_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="s3://"):
            run_evals.main(cli(PRODUCER, tmp_path / "out", "--banked-base-cells", "/not/a/bucket"))

    def test_an_unknown_noise_floor_framing_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="noise-floor-framings"):
            run_evals.main(
                cli(PRODUCER, tmp_path / "out", *banked("--noise-floor-framings", "not-a-framing"))
            )

    def test_an_unknown_noise_floor_framing_is_refused_with_no_bank_too(self) -> None:
        """The list is validated before the bank prefix, so a misspelling cannot ride in unread."""
        args = run_evals._parse_args(  # pyright: ignore[reportPrivateUsage]
            [
                "--model",
                "m",
                "--arm",
                "a",
                "--backend",
                "mock",
                "--noise-floor-framings",
                "not-a-framing",
            ]
        )
        with pytest.raises(ValueError, match=r"--noise-floor-framings names \['not-a-framing'\]"):
            run_evals._parse_bank(args)  # pyright: ignore[reportPrivateUsage]

    def test_the_noise_floor_framings_without_a_bank_are_refused(self) -> None:
        """With no bank the flag selects which cell stays out of nothing: accepted and does nothing."""
        args = run_evals._parse_args(  # pyright: ignore[reportPrivateUsage]
            [
                "--model",
                "m",
                "--arm",
                "a",
                "--backend",
                "mock",
                "--noise-floor-framings",
                FRAMING_UNSTATED,
            ]
        )
        with pytest.raises(ValueError, match="it would select nothing"):
            run_evals._parse_bank(args)  # pyright: ignore[reportPrivateUsage]

    def test_the_prefix_gains_its_trailing_slash(self) -> None:
        args = run_evals._parse_args(  # pyright: ignore[reportPrivateUsage]
            ["--model", "m", "--arm", "a", "--backend", "mock", *banked()]
        )
        bank = run_evals._parse_bank(args)  # pyright: ignore[reportPrivateUsage]
        assert bank is not None
        assert bank.prefix == BANK_PREFIX
        assert bank.entry_prefix("k") == BANK_PREFIX + "k/"
        assert bank.publish is False
        assert bank.noise_floor_framings == (FRAMING_UNSTATED,)


def game_records(game_id: str, *, value: float, count: int) -> list[dict[str, Any]]:
    """Behaviour records for one game, every one parsed at `value`."""
    return [
        {
            "record": SECTION_GAME_BEHAVIOR,
            "game_id": game_id,
            "prompt_id": f"{game_id}-standard-{index}",
            "payoff_variant": "standard",
            "reskin_id": f"frame-{index}",
            "render_grading": "group-mix",
            "label_a": "SHORT",
            "label_b": "LONG",
            "coop_label": "SHORT",
            "label_print_order": "canonical",
            "trained_game": False,
            "eval_only_game": False,
            "truncated_thinking": False,
            "parsed": True,
            "action": "C" if value else "D",
            "coop_fraction": value,
            "completion": "SENTINEL-COMPLETION-TEXT",
            "visible_text": "SENTINEL-COMPLETION-TEXT",
        }
        for index in range(count)
    ]


def write_trace(  # noqa: PLR0913 - one keyword per meta fact the synthetic cell carries
    path: Path, *, arm: str, step: int, git_sha: str, value: float, grading: str = "group-mix"
) -> Path:
    """Write one synthetic cell, meta record first, in the shape the readouts read."""
    meta: dict[str, Any] = {
        "record": RECORD_META,
        "written_at": f"2026-09-03T0{step // 10}:00:00+00:00",
        "git_sha": git_sha,
        "backend_model_id": "synthetic",
        "backend_kind": "vllm",
        "thinking": True,
        "sampler_mode": "training-distribution",
        "sampling": {"temperature": 1.0},
        "sections": [SECTION_GAME_BEHAVIOR],
        "eval_config": {"trained_game_ids": ["twin-pd"] if step > 0 else []},
        "arm": arm,
        "step": step,
        "run_dir": f"/runs/{arm}",
        "grading": grading,
        "executed_estimator": None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [meta, *game_records("twin-pd", value=value, count=8)]
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def write_banked_copy(
    source: Path, destination: Path, *, consumer_arm: str, consumer_meta: dict[str, Any]
) -> Path:
    """Copy a cell byte for byte and write the sidecar the driver would, sha256 included."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    meta = json.loads(source.read_text(encoding="utf-8").splitlines()[0])
    sidecar = banked_provenance_path(destination)
    sidecar.write_text(
        json.dumps(
            {
                "record": BANKED_PROVENANCE_RECORD,
                "consumer_arm": consumer_arm,
                "consumer_meta": consumer_meta,
                "step": meta["step"],
                "bank_prefix": BANK_PREFIX,
                "bank_key": BANKED_FIXTURE_KEY,
                "source_key": BANK_PREFIX + BANKED_FIXTURE_KEY + "/",
                "source_arm": meta["arm"],
                "source_git_sha": meta["git_sha"],
                "banked_at": "2026-09-03T00:00:00+00:00",
                "copied_at": "2026-09-03T01:00:00+00:00",
                "files": {
                    destination.name: {
                        "sha256": sha256_of_file(destination),
                        "bytes": destination.stat().st_size,
                    }
                },
                "bank_identity": {},
            }
        ),
        encoding="utf-8",
    )
    return destination


OWN_SHA = "own-sha-1234567"
PRODUCER_SHA = "producer-sha-89abcdef"
GROUP_ARM = "twin-pd-group"
SELF_ARM = "twin-pd-self"
# The prime sharing pair differs in exactly the field a banked copy would otherwise misreport: the
# producer trained under group-mix grading, the consumer under self grading, same base and prompts.
PRODUCER_GRADING = "group-mix"
CONSUMER_GRADING = "self"
CONSUMER_META = {
    "run_dir": f"/runs/{SELF_ARM}",
    "grading": CONSUMER_GRADING,
    "executed_estimator": None,
}


class TestTheReadersNameTheBankedSource:
    def _evals_root(self, tmp_path: Path) -> tuple[Path, Path]:
        """Two registered arms at every step; the second's step 0 is a banked copy of an outside producer's."""
        root = tmp_path / "evals-s3"
        source = write_trace(
            tmp_path / "producer" / "step-0.jsonl",
            arm=PRODUCER,
            step=0,
            git_sha=PRODUCER_SHA,
            value=1.0,
            grading=PRODUCER_GRADING,
        )
        for step in range(0, 80, 10):
            write_trace(
                root / "pass" / GROUP_ARM / f"step-{step}.jsonl",
                arm=GROUP_ARM,
                step=step,
                git_sha=OWN_SHA,
                value=0.5,
            )
            if step > 0:
                write_trace(
                    root / "pass" / SELF_ARM / f"step-{step}.jsonl",
                    arm=SELF_ARM,
                    step=step,
                    git_sha=OWN_SHA,
                    value=0.0,
                    grading=CONSUMER_GRADING,
                )
        copy = write_banked_copy(
            source,
            root / "pass" / SELF_ARM / "step-0.jsonl",
            consumer_arm=SELF_ARM,
            consumer_meta=CONSUMER_META,
        )
        return root, copy

    def test_the_copy_reads_as_the_consumers_cell_with_the_producers_generation_facts(
        self, tmp_path: Path
    ) -> None:
        """Arm-descriptive fields come from the sidecar; how the bytes were made stays the producer's."""
        _, copy = self._evals_root(tmp_path)
        (loaded,) = load_traces([copy])
        assert loaded.arm == SELF_ARM
        assert loaded.meta["arm"] == SELF_ARM
        assert loaded.meta["grading"] == CONSUMER_GRADING
        assert loaded.meta["run_dir"] == f"/runs/{SELF_ARM}"
        assert loaded.meta["git_sha"] == PRODUCER_SHA
        assert loaded.meta["written_at"] == "2026-09-03T00:00:00+00:00"
        banked = loaded.banked_from
        assert banked is not None
        assert banked["source_arm"] == PRODUCER
        assert read_eval_records(copy)[0]["grading"] == PRODUCER_GRADING

    def test_the_cross_arm_readout_attributes_the_copy_and_names_its_source_without_an_error(
        self, tmp_path: Path
    ) -> None:
        root, copy = self._evals_root(tmp_path)
        trace, note = readout.read_cell(copy)
        assert note is None
        assert trace.arm == SELF_ARM
        assert trace.meta[BANKED_FROM_KEY]["source_arm"] == PRODUCER
        built = readout.build_readout(root)
        assert built.status["arms"][SELF_ARM]["steps"] == [0, 10, 20, 30, 40, 50, 60, 70]
        assert built.status["arms"][SELF_ARM]["complete"] is True
        # The consumer's grading, not a pooled ["group-mix", "self"] read off the producer's meta.
        assert built.arm_facts[SELF_ARM]["grading"] == [CONSUMER_GRADING]
        assert built.arm_facts[GROUP_ARM]["grading"] == [PRODUCER_GRADING]
        (entry,) = built.status["banked_cells"]
        assert (entry["arm"], entry["step"]) == (SELF_ARM, 0)
        assert entry["source_arm"] == PRODUCER
        assert entry["source_git_sha"] == PRODUCER_SHA
        assert entry["bank_key"] == BANKED_FIXTURE_KEY
        assert "source_key" not in entry
        # The producer's commit is named in the banked entry, not counted as a sha disagreement.
        assert built.status["git_sha_disagreements"] == []
        assert built.git_sha == OWN_SHA
        assert built.error_containing is False
        markdown = readout.render_markdown(built)
        assert f"BANKED cell `{SELF_ARM}@0`" in markdown
        assert PRODUCER_SHA in markdown
        assert BANKED_FIXTURE_KEY in markdown
        assert BANK_PREFIX not in markdown
        assert "ERROR-CONTAINING" not in markdown
        assert "SENTINEL-COMPLETION-TEXT" not in markdown

    def test_the_battery_readout_reads_the_copy_as_the_consuming_arms_step_zero(
        self, tmp_path: Path
    ) -> None:
        root, _ = self._evals_root(tmp_path)
        battery = tmp_path / "battery-test"
        shutil.copytree(root, battery)
        built = battery_tables.build_readout(battery)
        assert built.excluded == []
        by_arm = {arm.arm: arm for arm in built.arms}
        assert set(by_arm) == {GROUP_ARM, SELF_ARM}
        assert 0 in by_arm[SELF_ARM].steps
        assert not by_arm[SELF_ARM].incomplete
        # The producer's commit is a provenance note on the arm, not a git_sha disagreement that
        # would put the ERROR-CONTAINING banner on every banked battery by construction.
        assert built.problems == []
        assert built.error_containing is False
        assert by_arm[SELF_ARM].facts["git_sha"] == OWN_SHA
        assert by_arm[GROUP_ARM].facts["git_sha"] == OWN_SHA
        (note,) = [note for note in by_arm[SELF_ARM].notes if "byte-identical copy" in note]
        assert PRODUCER in note
        assert PRODUCER_SHA in note
        assert BANKED_FIXTURE_KEY in note
        assert BANK_PREFIX not in note
        assert not any("byte-identical copy" in note for note in by_arm[GROUP_ARM].notes)

    def test_a_banked_copy_still_votes_on_the_fields_the_bank_key_fixes(
        self, tmp_path: Path
    ) -> None:
        """Only git_sha is excused; a copy whose sampling disagrees with the arm's cells is a real problem."""
        root, copy = self._evals_root(tmp_path)
        lines = copy.read_text(encoding="utf-8").splitlines(keepends=True)
        meta = json.loads(lines[0])
        meta["sampling"] = {"temperature": 0.2}
        lines[0] = json.dumps(meta) + "\n"
        copy.write_text("".join(lines), encoding="utf-8")
        sidecar = banked_provenance_path(copy)
        provenance = json.loads(sidecar.read_text(encoding="utf-8"))
        provenance["files"][copy.name]["sha256"] = sha256_of_file(copy)
        sidecar.write_text(json.dumps(provenance), encoding="utf-8")
        battery = tmp_path / "battery-test"
        shutil.copytree(root, battery)
        built = battery_tables.build_readout(battery)
        assert built.error_containing is True
        (problem,) = built.problems
        assert "PROVENANCE DISAGREEMENT on `sampling`" in problem
        assert f"{SELF_ARM}@0 (banked from {PRODUCER}@" in problem

    def test_the_per_arm_report_marks_the_copy_in_its_inventory(self, tmp_path: Path) -> None:
        root, copy = self._evals_root(tmp_path)
        markdown = render_report([copy, root / "pass" / SELF_ARM / "step-10.jsonl"])
        assert f"{SELF_ARM}@0 (banked from {PRODUCER}@{PRODUCER_SHA[:7]})" in markdown
        assert "byte-identical copies" in markdown
        assert BANKED_FIXTURE_KEY in markdown
        assert BANK_PREFIX not in markdown
        assert f"{SELF_ARM}@10," in markdown or f"{SELF_ARM}@10." in markdown

    def test_no_readout_or_report_prose_ever_names_the_bank_it_came_from(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The bank is an s3:// prefix; a readout gets pasted into docs, so no bucket may reach one.

        The bank key identifies the entry for anyone holding the bank, and the sidecar beside the
        trace keeps the full prefix for machines, so nothing is lost by keeping the prefix out of
        every document this repository renders for a human. The cell reader's own log line is held
        to the same rule, since it is the one place the prefix used to reach a human without a
        readout in between.
        """
        root, copy = self._evals_root(tmp_path)
        battery = tmp_path / "battery-test"
        shutil.copytree(root, battery)
        with caplog.at_level(logging.INFO, logger="games.battery_cells"):
            battery_readout = battery_tables.build_readout(battery)
        rendered = {
            "readout.md": readout.render_markdown(readout.build_readout(root)),
            "report.md": render_report([copy, root / "pass" / SELF_ARM / "step-10.jsonl"]),
            "battery notes": "\n".join(note for arm in battery_readout.arms for note in arm.notes),
            "cell reader log": "\n".join(
                record.getMessage()
                for record in caplog.records
                if record.name == "games.battery_cells" and "banked copy" in record.getMessage()
            ),
        }
        for name, text in rendered.items():
            assert BANKED_FIXTURE_KEY in text, name
            assert "s3://" not in text, name

    def test_a_sidecar_that_names_another_source_than_the_meta_raises(self, tmp_path: Path) -> None:
        _, copy = self._evals_root(tmp_path)
        sidecar = banked_provenance_path(copy)
        provenance = json.loads(sidecar.read_text(encoding="utf-8"))
        provenance["source_arm"] = "someone-else"
        sidecar.write_text(json.dumps(provenance), encoding="utf-8")
        meta = json.loads(copy.read_text(encoding="utf-8").splitlines()[0])
        with pytest.raises(ValueError, match="describe different cells"):
            attribute_trace(copy, meta)

    def test_a_sidecar_beside_a_later_step_cannot_relabel_it(self, tmp_path: Path) -> None:
        """Only step 0 is the shared base draw, so a sidecar on step 70 would relabel a trained cell.

        Everything else about this fixture is in order -- the sidecar's step matches the trace's, its
        source arm matches the meta's, the sha256 is the file's -- which is exactly the shape the
        driver's own step-0-only rule catches and the readers used to accept.
        """
        source = write_trace(
            tmp_path / "producer" / "step-70.jsonl",
            arm=PRODUCER,
            step=70,
            git_sha=PRODUCER_SHA,
            value=1.0,
            grading=PRODUCER_GRADING,
        )
        copy = write_banked_copy(
            source,
            tmp_path / "consumer" / "step-70.jsonl",
            consumer_arm=SELF_ARM,
            consumer_meta=CONSUMER_META,
        )
        meta = json.loads(copy.read_text(encoding="utf-8").splitlines()[0])
        with pytest.raises(ValueError, match="only step 0 is the un-adapted base model"):
            attribute_trace(copy, meta)

    def test_a_sidecar_naming_a_commit_the_trace_never_saw_raises(self, tmp_path: Path) -> None:
        """A copy is byte-identical to the producer's cell, so the two commits cannot disagree."""
        _, copy = self._evals_root(tmp_path)
        sidecar = banked_provenance_path(copy)
        provenance = json.loads(sidecar.read_text(encoding="utf-8"))
        provenance["source_git_sha"] = "a-commit-this-trace-never-saw"
        sidecar.write_text(json.dumps(provenance), encoding="utf-8")
        with pytest.raises(ValueError, match="generated at git SHA"):
            load_traces([copy])

    def test_a_sidecar_without_the_consumers_meta_raises(self, tmp_path: Path) -> None:
        """Without the consumer's fields the copy would stand under one arm carrying another's grading."""
        _, copy = self._evals_root(tmp_path)
        sidecar = banked_provenance_path(copy)
        provenance = json.loads(sidecar.read_text(encoding="utf-8"))
        del provenance["consumer_meta"]["grading"]
        sidecar.write_text(json.dumps(provenance), encoding="utf-8")
        with pytest.raises(ValueError, match="consumer_meta for \\['grading'\\]"):
            load_traces([copy])

    def test_a_sidecar_that_is_not_a_provenance_record_raises(self, tmp_path: Path) -> None:
        _, copy = self._evals_root(tmp_path)
        banked_provenance_path(copy).write_text(
            json.dumps({"record": "something"}), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="not a 'banked-cell-provenance' record"):
            load_traces([copy])

    def test_a_cell_without_a_sidecar_is_its_own(self, tmp_path: Path) -> None:
        root, _ = self._evals_root(tmp_path)
        (own,) = load_traces([root / "pass" / GROUP_ARM / "step-0.jsonl"])
        assert own.arm == GROUP_ARM
        assert own.banked_from is None
        assert own.label == f"{GROUP_ARM}@0"
