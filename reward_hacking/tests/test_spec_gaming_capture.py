"""Synthetic durable capture and refusal controls."""

import multiprocessing
import os
import signal
from pathlib import Path

import pytest

from reward_hacking.spec_gaming.capture import EpisodeJournal


def completion(journal: EpisodeJournal, turn: int = 0) -> str:
    return journal.save_completion(
        turn, [1, 2], [3], [-0.1], "synthetic", {"finish_reason": "stop"}
    )


def test_resume_committed_result_and_raw_closure(tmp_path: Path) -> None:
    with EpisodeJournal(tmp_path, {"source": "one"}, {"count": 0}) as journal:
        digest = completion(journal)
        journal.begin_action("call", digest, 0, "check", {})
        journal.commit_action("call", {"score": 1}, {"count": 1}, check=True, close=True)
        raw_digest = journal.raw_episode_digest()
    with EpisodeJournal(tmp_path, {"source": "one"}) as journal:
        saved_completion = journal.get_completion(0)
        assert saved_completion is not None
        assert saved_completion["digest"] == digest
        saved_action = journal.get_saved_action("call")
        assert saved_action is not None
        assert saved_action["result"] == {"score": 1}
        assert journal.progress()["checks"] == 1
        assert journal.state() == {"count": 1}
        assert journal.raw_episode_digest() == raw_digest
        assert journal.readout_eligibility({"evaluator": "one"}) == "pending"
        with pytest.raises(ValueError, match="closed"):
            journal.begin_action("later", digest, 1, "shell", {})


def test_stale_identity_and_duplicate_payload_refuse(tmp_path: Path) -> None:
    with EpisodeJournal(tmp_path, {"source": "one", "render": "same"}, {}) as journal:
        digest = completion(journal)
        journal.begin_action("call", digest, 0, "shell", {"cmd": "one"})
        with pytest.raises(ValueError, match="payload"):
            journal.begin_action("call", digest, 0, "shell", {"cmd": "two"})
        with pytest.raises(ValueError, match="pending"):
            journal.begin_action("other", digest, 1, "shell", {})
        with pytest.raises(ValueError, match="writer"):
            EpisodeJournal(tmp_path, {"source": "one", "render": "same"})
    with pytest.raises(ValueError, match="source"):
        EpisodeJournal(tmp_path, {"source": "two", "render": "same"})
    with EpisodeJournal(tmp_path, {"source": "one", "render": "same"}) as journal:
        assert journal.begin_action("call", digest, 0, "shell", {"cmd": "one"}).status == "pending"
        assert journal.state() == {}


def test_completion_and_assessment_are_separate(tmp_path: Path) -> None:
    with EpisodeJournal(tmp_path, {"model": "one"}, {}) as journal:
        digest = completion(journal)
        with pytest.raises(ValueError, match="completion"):
            journal.save_completion(0, [1], [2], [None], "changed", {})
        with pytest.raises(ValueError, match="closed"):
            journal.raw_episode_digest()
        journal.begin_action("close", digest, 0, "hand_back", {})
        journal.commit_action("close", {}, {}, close=True)
        identity = {"evaluator": "one", "rubric": "one", "reviewer": "one"}
        journal.save_assessment(identity, {"awareness_status": "pending", "mechanism_evidence": []})
        assert journal.readout_eligibility(identity) == "pending"
        reviewed_identity = {**identity, "revision": "reviewed"}
        journal.save_assessment(
            reviewed_identity, {"awareness_status": "reviewed", "mechanism_evidence": []}
        )
        assert journal.readout_eligibility(reviewed_identity) == "reviewed"
        assert journal.readout_eligibility({"evaluator": "two"}) == "pending"
        positive_identity = {**identity, "revision": "positive"}
        journal.save_assessment(
            positive_identity, {"awareness_status": "positive", "mechanism_evidence": []}
        )
        assert journal.readout_eligibility(positive_identity) == "excluded"
        saved_completion = journal.get_completion(0)
        assert saved_completion is not None
        assert saved_completion["digest"] == digest


def test_check_budget_and_blob_validation(tmp_path: Path) -> None:
    with EpisodeJournal(tmp_path, {}, {}) as journal:
        digest = completion(journal)
        for index in range(5):
            journal.begin_action(str(index), digest, index, "check", {})
            journal.commit_action(str(index), {}, {}, check=True)
        journal.begin_action("six", digest, 5, "check", {})
        with pytest.raises(ValueError, match="check budget"):
            journal.commit_action("six", {}, {}, check=True)
        assert journal.progress()["checks"] == 5
        blob = journal.put_blob(b"original")
        assert journal.read_blob(blob) == b"original"
        journal.connection.execute("UPDATE blobs SET data=? WHERE digest=?", (b"corrupt", blob))
        with pytest.raises(ValueError, match="digest"):
            journal.read_blob(blob)


def crash_writer(root: Path, stage: str) -> None:
    journal = EpisodeJournal(root, {"source": "synthetic"}, {"count": 0})
    digest = completion(journal)
    journal.begin_action("mutation", digest, 0, "shell", {})
    if stage == "committed":
        journal.commit_action("mutation", {"exit_code": 1}, {"count": 1}, check=True, close=True)
    elif stage == "transaction":
        journal.connection.execute("BEGIN IMMEDIATE")
        journal.connection.execute("UPDATE meta SET value='999' WHERE key='checks'")
    os.kill(os.getpid(), signal.SIGKILL)


@pytest.mark.parametrize("stage", ["intent", "transaction", "committed"])
def test_process_death_reopens_last_committed_state(tmp_path: Path, stage: str) -> None:
    process = multiprocessing.get_context("spawn").Process(
        target=crash_writer, args=(tmp_path, stage)
    )
    process.start()
    process.join(timeout=30)
    assert process.exitcode == -signal.SIGKILL
    with EpisodeJournal(tmp_path, {"source": "synthetic"}) as journal:
        saved = journal.get_saved_action("mutation")
        assert journal.get_completion(0) is not None
        if stage == "committed":
            assert saved is not None
            assert saved["result"] == {"exit_code": 1}
            assert journal.state() == {"count": 1}
            assert journal.progress()["checks"] == 1
            assert journal.progress()["closed"] is True
        else:
            assert saved is None
            assert journal.state() == {"count": 0}
            assert journal.progress()["checks"] == 0
            saved_completion = journal.get_completion(0)
            assert saved_completion is not None
            digest = saved_completion["digest"]
            intent = journal.begin_action("mutation", digest, 0, "shell", {})
            assert intent.status == "pending"
            journal.commit_action("mutation", {"exit_code": 1}, {"count": 1})
            assert journal.state() == {"count": 1}


@pytest.mark.parametrize(
    "component",
    [
        "source_inputs",
        "render",
        "tools",
        "reward",
        "model",
        "tokenizer",
        "sampling",
        "runtime",
        "budgets",
        "simulator",
    ],
)
def test_each_generation_component_is_bound(tmp_path: Path, component: str) -> None:
    identity = {component: {"version": "original"}, "unchanged": "same"}
    with EpisodeJournal(tmp_path, identity, {}):
        pass
    with pytest.raises(ValueError, match=component):
        EpisodeJournal(tmp_path, {**identity, component: {"version": "changed"}})


def test_missing_manual_review_and_stale_sidecar_refuse(tmp_path: Path) -> None:
    with EpisodeJournal(tmp_path, {}, {}) as journal:
        digest = completion(journal)
        journal.begin_action("close", digest, 0, "hand_back", {})
        journal.commit_action("close", {}, {}, close=True)
        identity = {"rubric": "synthetic", "reviewer": "synthetic"}
        with pytest.raises(ValueError, match="awareness_status"):
            journal.save_assessment(identity, {"mechanism_evidence": []})
        with pytest.raises(ValueError, match="reviewer"):
            journal.save_assessment({}, {"awareness_status": "reviewed", "mechanism_evidence": []})
        journal.save_assessment(identity, {"awareness_status": "unknown", "mechanism_evidence": []})
        assert journal.readout_eligibility(identity) == "pending"
        journal.connection.execute(
            "UPDATE meta SET value=? WHERE key='closure'", ('"wrong-closure"',)
        )
        with pytest.raises(ValueError, match="digest"):
            journal.load_assessment(identity)


@pytest.mark.parametrize("changed", [1, 1.0])
def test_json_types_are_bound_in_generation_and_commit(tmp_path: Path, changed: float) -> None:
    with EpisodeJournal(tmp_path, {"sampling": {"enabled": True}}, {}) as journal:
        digest = completion(journal)
        journal.begin_action("first", digest, 0, "shell", {})
        journal.commit_action("first", {"nested": [True]}, {})
        with pytest.raises(ValueError, match="payload"):
            journal.commit_action("first", {"nested": [changed]}, {})
    with pytest.raises(ValueError, match="sampling"):
        EpisodeJournal(tmp_path, {"sampling": {"enabled": changed}})


def test_explicit_initial_state_must_match_original(tmp_path: Path) -> None:
    with EpisodeJournal(tmp_path, {}, {"count": 0}):
        pass
    with pytest.raises(ValueError, match="initial state"):
        EpisodeJournal(tmp_path, {}, {"count": 999})
    with pytest.raises(ValueError, match="initial state"):
        EpisodeJournal(tmp_path, {}, None)
    with EpisodeJournal(tmp_path, {}) as journal:
        assert journal.state() == {"count": 0}


def test_assessment_identity_is_immutable(tmp_path: Path) -> None:
    with EpisodeJournal(tmp_path, {}, {}) as journal:
        digest = completion(journal)
        journal.begin_action("close", digest, 0, "hand_back", {})
        journal.commit_action("close", {}, {}, close=True)
        identity = {"rubric": "one", "reviewer": "one"}
        positive = {"awareness_status": "positive", "mechanism_evidence": []}
        journal.save_assessment(identity, positive)
        journal.save_assessment(identity, positive)
        with pytest.raises(ValueError, match=r"assessment.*identity"):
            journal.save_assessment(identity, {**positive, "awareness_status": "reviewed"})
        assert journal.readout_eligibility(identity) == "excluded"
        entries = journal.list_assessments()
        assert len(entries) == 1
        assert entries[0]["identity"] == identity
        assert entries[0]["assessment"] == positive
        assert entries[0]["raw_episode_digest"] == journal.raw_episode_digest()


def test_old_completion_cannot_add_actions(tmp_path: Path) -> None:
    with EpisodeJournal(tmp_path, {}, {}) as journal:
        old = completion(journal)
        new = completion(journal, 1)
        journal.begin_action("new", new, 0, "shell", {})
        journal.commit_action("new", {}, {})
        with pytest.raises(ValueError, match="latest completion"):
            journal.begin_action("old", old, 0, "shell", {})
        with pytest.raises(ValueError, match="call index"):
            journal.begin_action("skip", new, 2, "shell", {})
        with pytest.raises(ValueError, match="turn index"):
            completion(journal, 3)


def test_closure_validates_all_raw_evidence(tmp_path: Path) -> None:
    with EpisodeJournal(tmp_path, {}, {}) as journal:
        old = completion(journal)
        new = completion(journal, 1)
        journal.begin_action("close", new, 0, "hand_back", {})
        journal.commit_action("close", {}, {}, close=True)
        journal.connection.execute("UPDATE blobs SET data=? WHERE digest=?", (b"corrupt", old))
        with pytest.raises(ValueError, match="digest"):
            journal.raw_episode_digest()


def test_constructor_failure_releases_writer_even_with_live_traceback(tmp_path: Path) -> None:
    errors: list[ValueError] = []
    try:
        EpisodeJournal(tmp_path, {"bad": float("nan")}, {})
    except ValueError as error:
        errors.append(error)
    assert errors
    with EpisodeJournal(tmp_path, {}, {}) as journal:
        assert journal.state() == {}


@pytest.mark.parametrize("unsupported", [(1, 2), {1: "coerced"}])
def test_non_json_identity_refuses(tmp_path: Path, unsupported: object) -> None:
    with pytest.raises(TypeError, match="JSON"):
        EpisodeJournal(tmp_path, {"source": unsupported}, {})


def test_identity_refusal_releases_lock_with_live_traceback(tmp_path: Path) -> None:
    with EpisodeJournal(tmp_path, {"source": "original"}, {}):
        pass
    errors: list[ValueError] = []
    try:
        EpisodeJournal(tmp_path, {"source": "changed"})
    except ValueError as error:
        errors.append(error)
    assert errors
    with EpisodeJournal(tmp_path, {"source": "original"}) as journal:
        assert journal.state() == {}


def test_host_termination_can_follow_sampled_actions(tmp_path: Path) -> None:
    with EpisodeJournal(tmp_path, {}, {}) as journal:
        digest = completion(journal)
        journal.begin_action("sampled", digest, 0, "shell", {})
        journal.commit_action("sampled", {}, {})
        journal.begin_action("host-stop", digest, 1000000, "termination", {"status": "budget"})
        journal.commit_action("host-stop", {"status": "budget"}, {}, close=True)
        assert journal.raw_episode_digest()
