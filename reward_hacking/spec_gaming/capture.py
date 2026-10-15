"""Host-only episode journal; SQLite commits bind evidence, state and progress.

One process holds an episode flock for its entire execution. Pending intents may
only be retried against private copies of their recorded prior state. This does
not promise exactly-once external side effects. SQLite FULL synchronization and
its rollback journal protect process-kill recovery and request durable OS writes.
Power-loss recovery is unverified; filesystem, directory-entry and storage
persistence require a separate deployment check.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import sqlite3
from contextlib import ExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from pathlib import Path

# JSON values vary by source; serialization validates this boundary.
# ruff: noqa: ANN401

MAX_CHECKS = 5
_OMITTED = object()
HOST_TERMINATION_INDEX = 1000000


def _validate_json(value: Any) -> None:
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                raise TypeError("JSON object keys must be strings")
            _validate_json(child)
    elif type(value) is list:
        for child in value:
            _validate_json(child)
    elif type(value) not in (str, int, float, bool, type(None)):
        raise TypeError("value must use JSON types")


def _encode(value: Any) -> bytes:
    _validate_json(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class ActionIntent:
    """Persisted action reservation bound to a specific pre-execution state."""

    request_id: str
    status: str
    payload: dict[str, Any]
    prior_state_digest: str
    prior_head_digest: str


class EpisodeJournal:
    """Small single-writer journal. Caller owns execution and actor observations."""

    def __init__(
        self, root: Path, generation_identity: dict[str, Any], initial_state: Any = _OMITTED
    ) -> None:
        """Open or resume one episode after validating its generation identity."""
        if type(generation_identity) is not dict:
            raise TypeError("generation identity must be a JSON object")
        _encode(generation_identity)
        if initial_state is not _OMITTED:
            _encode(initial_state)
        root.mkdir(parents=True, exist_ok=True)
        with ExitStack() as cleanup:
            self._lock = cleanup.enter_context((root / "writer.lock").open("ab"))
            try:
                fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("episode already has a writer") from None
            self.connection = sqlite3.connect(root / "episode.sqlite", isolation_level=None)
            cleanup.callback(self.connection.close)
            self._initialize(generation_identity, initial_state)
            cleanup.pop_all()

    def _initialize(self, generation_identity: dict[str, Any], initial_state: Any) -> None:
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript(
            "CREATE TABLE IF NOT EXISTS blobs(digest TEXT PRIMARY KEY, data BLOB NOT NULL);"
            "CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);"
            "CREATE TABLE IF NOT EXISTS completions(turn INTEGER PRIMARY KEY, digest TEXT NOT NULL);"
            "CREATE TABLE IF NOT EXISTS actions(request_id TEXT PRIMARY KEY, payload_digest TEXT NOT NULL, "
            "intent_digest TEXT NOT NULL, committed_digest TEXT);"
            "CREATE TABLE IF NOT EXISTS events(sequence INTEGER PRIMARY KEY, digest TEXT NOT NULL);"
            "CREATE TABLE IF NOT EXISTS assessments(identity_digest TEXT PRIMARY KEY, digest TEXT NOT NULL);"
        )
        saved_identity = self._meta("generation")
        if saved_identity is None:
            if initial_state is _OMITTED:
                initial_state = None
            with self.connection:
                self.connection.execute("BEGIN IMMEDIATE")
                self._set_meta("generation", generation_identity)
                initial_digest = self.put_json(initial_state)
                self._set_meta("initial_state", initial_digest)
                self._set_meta("state", initial_digest)
                self._set_meta(
                    "head",
                    self.put_json(
                        {"generation": generation_identity, "initial_state": initial_state}
                    ),
                )
                self._set_meta("checks", 0)
                self._set_meta("closed", value=False)
        elif _encode(saved_identity) != _encode(generation_identity):
            changed = sorted(
                key
                for key in set(saved_identity) | set(generation_identity)
                if key not in saved_identity
                or key not in generation_identity
                or _encode(saved_identity[key]) != _encode(generation_identity[key])
            )
            raise ValueError("generation identity changed: " + ", ".join(changed))
        elif initial_state is not _OMITTED and _digest(_encode(initial_state)) != self._meta(
            "initial_state"
        ):
            raise ValueError("initial state changed")

    def __enter__(self) -> Self:
        """Keep the writer reservation for this context."""
        return self

    def __exit__(self, *_args: object) -> None:
        """Release the database and writer reservation."""
        self.close()

    def close(self) -> None:
        """Release the database and episode writer lock."""
        self.connection.close()
        self._lock.close()

    def _meta(self, key: str) -> Any:
        row = self.connection.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return None if row is None else json.loads(row[0])

    def _set_meta(self, key: str, value: Any) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO meta VALUES (?,?)", (key, _encode(value).decode())
        )

    def put_blob(self, data: bytes) -> str:
        """Persist immutable bytes and return their validated content digest."""
        digest = _digest(data)
        self.connection.execute("INSERT OR IGNORE INTO blobs VALUES (?,?)", (digest, data))
        self.read_blob(digest)
        return digest

    def read_blob(self, digest: str) -> bytes:
        """Read bytes and fail if their content no longer matches the digest."""
        row = self.connection.execute("SELECT data FROM blobs WHERE digest=?", (digest,)).fetchone()
        if row is None or _digest(row[0]) != digest:
            raise ValueError("missing or corrupt blob digest: " + digest)
        return bytes(row[0])

    def put_json(self, value: Any) -> str:
        """Persist canonical JSON as an immutable blob."""
        return self.put_blob(_encode(value))

    def read_json(self, digest: str) -> Any:
        """Read a validated JSON blob."""
        return json.loads(self.read_blob(digest))

    def state(self) -> Any:
        """Return the most recently committed state."""
        return self.read_json(self._meta("state"))

    def progress(self) -> dict[str, Any]:
        """Return durable counters and committed state references."""
        return {key: self._meta(key) for key in ("state", "head", "checks", "closed", "closure")}

    def _event(self, event: dict[str, Any]) -> str:
        event["previous_head"] = self._meta("head")
        digest = self.put_json(event)
        self.connection.execute("INSERT INTO events(digest) VALUES (?)", (digest,))
        self._set_meta("head", digest)
        return digest

    def save_completion(  # noqa: PLR0913, PLR0917
        self,
        turn_index: int,
        prompt_token_ids: list[int],
        output_token_ids: list[int],
        sampled_logprobs: list[float | None],
        raw_text: str,
        metadata: dict[str, Any],
    ) -> str:
        """Commit exact sampled evidence before executing any associated action."""
        if len(output_token_ids) != len(sampled_logprobs):
            raise ValueError("completion token/logprob length mismatch")
        payload = {
            "turn_index": turn_index,
            "prompt_token_ids": prompt_token_ids,
            "output_token_ids": output_token_ids,
            "sampled_logprobs": sampled_logprobs,
            "raw_text": raw_text,
            "metadata": metadata,
        }
        saved = self.get_completion(turn_index)
        if saved is not None:
            if saved["digest"] != _digest(_encode(payload)):
                raise ValueError("completion already saved with different payload")
            return str(saved["digest"])
        if self._meta("closed"):
            raise ValueError("episode is closed")
        if self.connection.execute(
            "SELECT 1 FROM actions WHERE committed_digest IS NULL"
        ).fetchone():
            raise ValueError("cannot save another completion while an action is pending")
        latest = self.connection.execute("SELECT MAX(turn) FROM completions").fetchone()[0]
        expected_turn = 0 if latest is None else latest + 1
        if type(turn_index) is not int or turn_index != expected_turn:
            raise ValueError("completion turn index must be chronological")
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            digest = self.put_json(payload)
            self.connection.execute("INSERT INTO completions VALUES (?,?)", (turn_index, digest))
            self._event({"kind": "completion", "digest": digest})
        return digest

    def get_completion(self, turn_index: int) -> dict[str, Any] | None:
        """Return saved exact token evidence without invoking the actor."""
        row = self.connection.execute(
            "SELECT digest FROM completions WHERE turn=?", (turn_index,)
        ).fetchone()
        if row is None:
            return None
        return {"digest": row[0], **self.read_json(row[0])}

    def begin_action(
        self,
        request_id: str,
        completion_digest: str,
        call_index: int,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ActionIntent:
        """Reserve a stable request; return its status without dispatching it."""
        payload = {
            "completion_digest": completion_digest,
            "call_index": call_index,
            "tool_name": tool_name,
            "arguments": arguments,
        }
        payload_digest = _digest(_encode(payload))
        row = self.connection.execute(
            "SELECT * FROM actions WHERE request_id=?", (request_id,)
        ).fetchone()
        if row is not None:
            if row[1] != payload_digest:
                raise ValueError("request ID reused with changed payload")
            intent = self.read_json(row[2])
            return ActionIntent(request_id, "pending" if row[3] is None else "committed", **intent)
        if self._meta("closed"):
            raise ValueError("episode is closed")
        if self.connection.execute(
            "SELECT 1 FROM actions WHERE committed_digest IS NULL"
        ).fetchone():
            raise ValueError("previous action is pending")
        if (
            self.connection.execute(
                "SELECT 1 FROM completions WHERE digest=?", (completion_digest,)
            ).fetchone()
            is None
        ):
            raise ValueError("action requires saved completion")
        self._validate_action_order(completion_digest, call_index, tool_name)
        intent: dict[str, Any] = {
            "payload": payload,
            "prior_state_digest": self._meta("state"),
            "prior_head_digest": self._meta("head"),
        }
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            digest = self.put_json(intent)
            self.connection.execute(
                "INSERT INTO actions VALUES (?,?,?,NULL)", (request_id, payload_digest, digest)
            )
        return ActionIntent(request_id, "pending", **intent)

    def _validate_action_order(
        self, completion_digest: str, call_index: int, tool_name: str
    ) -> None:
        latest = self.connection.execute(
            "SELECT digest FROM completions ORDER BY turn DESC LIMIT 1"
        ).fetchone()[0]
        if completion_digest != latest:
            raise ValueError("action requires latest completion")
        prior_calls = [
            self.read_json(intent_digest)["payload"]
            for (intent_digest,) in self.connection.execute("SELECT intent_digest FROM actions")
        ]
        calls_for_completion = [
            call for call in prior_calls if call["completion_digest"] == completion_digest
        ]
        if tool_name == "termination":
            if call_index != HOST_TERMINATION_INDEX:
                raise ValueError("host termination requires its reserved call index")
        elif type(call_index) is not int or call_index != len(calls_for_completion):
            raise ValueError("action call index must be chronological")
        if any(call["call_index"] == call_index for call in calls_for_completion):
            raise ValueError("completion call already has a request ID")

    def get_saved_action(self, request_id: str) -> dict[str, Any] | None:
        """Return committed action evidence for replay, or None for pending work."""
        row = self.connection.execute(
            "SELECT committed_digest FROM actions WHERE request_id=?", (request_id,)
        ).fetchone()
        return None if row is None or row[0] is None else self.read_json(row[0])

    def commit_action(
        self,
        request_id: str,
        result: Any,
        state: Any,
        *,
        check: bool = False,
        close: bool = False,
    ) -> dict[str, Any]:
        """Atomically publish result, state, budget use and optional raw closure."""
        row = self.connection.execute(
            "SELECT intent_digest,committed_digest FROM actions WHERE request_id=?", (request_id,)
        ).fetchone()
        if row is None:
            raise ValueError("action has no durable intent")
        intent = self.read_json(row[0])
        committed = {
            "request_id": request_id,
            "intent_digest": row[0],
            "result": result,
            "state_digest": _digest(_encode(state)),
            "check": check,
            "close": close,
        }
        if row[1] is not None:
            if self.read_blob(row[1]) != _encode(committed):
                raise ValueError("committed action payload changed")
            return self.read_json(row[1])
        if self._meta("closed"):
            raise ValueError("episode is closed")
        if intent["prior_head_digest"] != self._meta("head") or intent[
            "prior_state_digest"
        ] != self._meta("state"):
            raise ValueError("action prior state/head changed")
        checks = self._meta("checks") + int(check)
        if checks > MAX_CHECKS:
            raise ValueError("check budget exceeded")
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            self.put_json(state)
            digest = self.put_json(committed)
            self.connection.execute(
                "UPDATE actions SET committed_digest=? WHERE request_id=?", (digest, request_id)
            )
            self._set_meta("state", committed["state_digest"])
            self._set_meta("checks", checks)
            head = self._event({"kind": "action", "digest": digest})
            if close:
                closure = self.put_json(
                    {
                        "head": head,
                        "state_digest": committed["state_digest"],
                        "checks": checks,
                        "raw_manifest": self._raw_manifest(),
                    }
                )
                self._set_meta("closure", closure)
                self._set_meta("closed", value=True)
        return committed

    def raw_episode_digest(self) -> str:
        """Return immutable closure identity; open episodes have no final digest."""
        if self._meta("closed") is not True:
            raise ValueError("episode is not closed")
        digest = str(self._meta("closure"))
        closure = self.read_json(digest)
        manifest = closure["raw_manifest"]
        for blob_digest in manifest["blobs"]:
            self.read_blob(blob_digest)
        if _encode(manifest) != _encode(self._raw_manifest(blob_digests=manifest["blobs"])):
            raise ValueError("raw episode manifest digest mismatch")
        expected_progress = {
            "head": self._meta("head"),
            "state_digest": self._meta("state"),
            "checks": self._meta("checks"),
        }
        saved_progress = {key: closure[key] for key in expected_progress}
        if _encode(saved_progress) != _encode(expected_progress):
            raise ValueError("raw episode closure digest mismatch")
        return digest

    def _raw_manifest(self, *, blob_digests: list[str] | None = None) -> dict[str, Any]:
        return {
            "generation": self._meta("generation"),
            "initial_state": self._meta("initial_state"),
            "blobs": blob_digests
            if blob_digests is not None
            else [
                row[0]
                for row in self.connection.execute("SELECT digest FROM blobs ORDER BY digest")
            ],
            "events": [
                list(row)
                for row in self.connection.execute(
                    "SELECT sequence,digest FROM events ORDER BY sequence"
                )
            ],
            "completions": [
                list(row)
                for row in self.connection.execute(
                    "SELECT turn,digest FROM completions ORDER BY turn"
                )
            ],
            "actions": [
                list(row)
                for row in self.connection.execute("SELECT * FROM actions ORDER BY request_id")
            ],
        }

    def save_assessment(self, identity: dict[str, Any], assessment: dict[str, Any]) -> str:
        """Save a versioned assessment separately from actor generation."""
        if assessment.get("awareness_status") not in {"pending", "reviewed", "positive", "unknown"}:
            raise ValueError("explicit awareness_status required")
        if assessment.get("awareness_status") in {"reviewed", "positive"} and (
            not identity.get("rubric") or not identity.get("reviewer")
        ):
            raise ValueError("manual awareness review requires rubric and reviewer identity")
        if "mechanism_evidence" not in assessment:
            raise ValueError("explicit mechanism_evidence required")
        payload = {
            "raw_episode_digest": self.raw_episode_digest(),
            "identity": identity,
            "assessment": assessment,
        }
        identity_digest = _digest(_encode(identity))
        existing = self.connection.execute(
            "SELECT digest FROM assessments WHERE identity_digest=?", (identity_digest,)
        ).fetchone()
        if existing is not None:
            if self.read_blob(existing[0]) != _encode(payload):
                raise ValueError("assessment identity already has a different payload")
            return str(existing[0])
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            digest = self.put_json(payload)
            self.connection.execute(
                "INSERT INTO assessments VALUES (?,?)", (identity_digest, digest)
            )
        return digest

    def load_assessment(self, identity: dict[str, Any]) -> dict[str, Any] | None:
        """Return only an assessment bound to the requested evaluator and closure."""
        row = self.connection.execute(
            "SELECT digest FROM assessments WHERE identity_digest=?", (_digest(_encode(identity)),)
        ).fetchone()
        if row is None:
            return None
        payload = self.read_json(row[0])
        if (
            _encode(payload["identity"]) != _encode(identity)
            or payload["raw_episode_digest"] != self.raw_episode_digest()
        ):
            raise ValueError("stale assessment identity")
        return payload["assessment"]

    def list_assessments(self) -> tuple[dict[str, Any], ...]:
        """Return all immutable sidecars bound to this validated raw closure."""
        raw_digest = self.raw_episode_digest()
        entries: list[dict[str, Any]] = []
        for identity_digest, digest in self.connection.execute(
            "SELECT identity_digest,digest FROM assessments ORDER BY identity_digest"
        ):
            payload = self.read_json(digest)
            if _digest(_encode(payload["identity"])) != identity_digest:
                raise ValueError("stale assessment identity digest")
            if payload["raw_episode_digest"] != raw_digest:
                raise ValueError("stale assessment episode digest")
            entries.append({"digest": digest, **payload})
        return tuple(entries)

    def readout_eligibility(self, identity: dict[str, Any]) -> str:
        """Keep missing or unresolved manual awareness review pending."""
        if not self._meta("closed"):
            return "pending"
        assessment = self.load_assessment(identity)
        if assessment is None or assessment["awareness_status"] in {"pending", "unknown"}:
            return "pending"
        return "excluded" if assessment["awareness_status"] == "positive" else "reviewed"

    def export_events(self, destination: Path) -> None:
        """Regenerate ordered JSONL from authoritative immutable event blobs."""
        with destination.open("wb") as stream:
            for sequence, digest in self.connection.execute(
                "SELECT sequence,digest FROM events ORDER BY sequence"
            ):
                stream.write(
                    _encode(
                        {"sequence": sequence, "digest": digest, "event": self.read_json(digest)}
                    )
                    + b"\n"
                )
