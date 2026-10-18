"""Host-only episode journal; SQLite commits bind evidence, state and progress.

One process holds an episode flock for its entire execution. Pending intents may
only be retried against private copies of their recorded prior state. This does
not promise exactly-once external side effects. SQLite FULL synchronization and
its rollback journal protect process-kill recovery and request durable OS writes.
Power-loss recovery is unverified; filesystem, directory-entry and storage
persistence require a separate deployment check.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import sqlite3
from contextlib import ExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from collections.abc import Mapping
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


def compare_identity_fields(expected: Any, observed: Any, prefix: str = "identity") -> list[str]:
    """Return readable JSON field paths whose values differ between two records."""
    if type(expected) is dict and type(observed) is dict:
        differences: list[str] = []
        for key in sorted(set(expected) | set(observed)):
            field = f"{prefix}.{key}"
            if key not in expected or key not in observed:
                differences.append(field)
            else:
                differences.extend(compare_identity_fields(expected[key], observed[key], field))
        return differences
    if type(expected) is list and type(observed) is list:
        differences = []
        for index in range(max(len(expected), len(observed))):
            field = f"{prefix}[{index}]"
            if index >= len(expected) or index >= len(observed):
                differences.append(field)
            else:
                differences.extend(compare_identity_fields(expected[index], observed[index], field))
        return differences
    if type(expected) is bytes or type(observed) is bytes:
        return [] if expected == observed else [prefix]
    return [] if _encode(expected) == _encode(observed) else [prefix]


def _same_json_value(expected: Any, observed: Any) -> bool:
    return not compare_identity_fields(expected, observed)


def _assessment_identity_content(identity: Mapping[str, Any]) -> dict[str, Any]:
    content = {key: value for key, value in identity.items() if key != "raw_episode_digest"}
    if "automatic_assessment_identity" in content:
        content.pop("automatic_assessment_digest", None)
    return content


def _readable_blob(data: bytes) -> dict[str, str]:
    try:
        return {"text": data.decode("utf-8")}
    except UnicodeDecodeError:
        return {"base64": base64.b64encode(data).decode("ascii")}


def _action_intent(request_id: str, status: str, intent: dict[str, Any]) -> ActionIntent:
    return ActionIntent(
        request_id=request_id,
        status=status,
        payload=intent["payload"],
        prior_state_digest=intent["prior_state_digest"],
        prior_head_digest=intent["prior_head_digest"],
        prior_state=intent.get("prior_state"),
        prior_head_sequence=intent.get("prior_head_sequence"),
    )


@dataclass(frozen=True)
class ActionIntent:
    """Persisted action reservation bound to a specific pre-execution state."""

    request_id: str
    status: str
    payload: dict[str, Any]
    prior_state_digest: str
    prior_head_digest: str
    prior_state: Any
    prior_head_sequence: int | None


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
        elif initial_state is not _OMITTED:
            saved_initial_state = self.read_json(self._meta("initial_state"))
            changed = compare_identity_fields(initial_state, saved_initial_state, "initial_state")
            if changed:
                raise ValueError("initial state changed: " + ", ".join(changed))
        if self._meta("head_sequence") is None:
            latest_event = self.connection.execute("SELECT MAX(sequence) FROM events").fetchone()[0]
            self._set_meta("head_sequence", 0 if latest_event is None else latest_event)

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
        cursor = self.connection.execute("INSERT INTO events(digest) VALUES (?)", (digest,))
        self._set_meta("head", digest)
        self._set_meta("head_sequence", cursor.lastrowid)
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
            differences = compare_identity_fields(
                payload,
                {key: value for key, value in saved.items() if key != "digest"},
                "completion",
            )
            if differences:
                raise ValueError(
                    "completion already saved with changed fields: " + ", ".join(differences)
                )
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

    def _completion_turn(self, digest: str) -> int:
        row = self.connection.execute(
            "SELECT turn FROM completions WHERE digest=?", (digest,)
        ).fetchone()
        if row is None:
            raise ValueError("action requires saved completion")
        return int(row[0])

    def _action_payload_content(self, payload: dict[str, Any]) -> dict[str, Any]:
        completion_digest = str(payload["completion_digest"])
        completion_turn = self._completion_turn(completion_digest)
        saved_completion_turn = payload.get("completion_turn")
        if saved_completion_turn is not None and saved_completion_turn != completion_turn:
            raise ValueError("action completion turn does not match its saved completion")
        return {
            **{
                key: value
                for key, value in payload.items()
                if key not in {"completion_digest", "completion_turn"}
            },
            "completion_turn": completion_turn,
        }

    def _event_sequence(self, digest: str) -> int | None:
        row = self.connection.execute(
            "SELECT sequence FROM events WHERE digest=?", (digest,)
        ).fetchone()
        return None if row is None else int(row[0])

    def _readable_head_content(self, digest: str) -> dict[str, Any]:
        events: list[dict[str, Any]] = []
        while True:
            head = self.read_json(digest)
            sequence = self._event_sequence(digest)
            if sequence is None:
                if "kind" in head:
                    raise ValueError("action head has no event sequence")
                return {
                    "generation": head["generation"],
                    "initial_state": head["initial_state"],
                    "events": list(reversed(events)),
                }

            if head["kind"] == "completion":
                completion_turn = self._completion_turn(head["digest"])
                completion = self.get_completion(completion_turn)
                if completion is None:
                    raise ValueError("action head references a missing completion")
                record = {key: value for key, value in completion.items() if key != "digest"}
            else:
                committed = self.read_json(head["digest"])
                intent = self.read_json(committed["intent_digest"])
                prior_state = intent.get("prior_state")
                if "prior_state" not in intent:
                    prior_state = self.read_json(intent["prior_state_digest"])
                prior_head_sequence = intent.get("prior_head_sequence")
                if prior_head_sequence is None:
                    prior_head_sequence = self._event_sequence(intent["prior_head_digest"]) or 0
                record = {
                    "request_id": committed["request_id"],
                    "intent": {
                        "payload": self._action_payload_content(intent["payload"]),
                        "prior_state": prior_state,
                        "prior_head_sequence": prior_head_sequence,
                    },
                    "result": committed["result"],
                    "state": self.read_json(committed["state_digest"]),
                    "check": committed["check"],
                    "close": committed["close"],
                }
            events.append({"sequence": sequence, "kind": head["kind"], "record": record})
            previous_head = head.get("previous_head")
            if previous_head is None:
                raise ValueError("action event has no previous head")
            digest = previous_head

    def _action_intent_content(self, intent: dict[str, Any]) -> dict[str, Any]:
        prior_state = intent.get("prior_state")
        if prior_state is None:
            prior_state = self.read_json(intent["prior_state_digest"])
        return {
            "payload": self._action_payload_content(intent["payload"]),
            "prior_state": prior_state,
            "prior_head": self._readable_head_content(intent["prior_head_digest"]),
        }

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
            intent = self.read_json(row[2])
            expected_payload = self._action_payload_content(payload)
            saved_payload = self._action_payload_content(intent["payload"])
            differences = compare_identity_fields(expected_payload, saved_payload, "action")
            if differences:
                raise ValueError(
                    "request ID reused with changed payload fields: " + ", ".join(differences)
                )
            if "prior_state" not in intent:
                intent["prior_state"] = self.read_json(intent["prior_state_digest"])
            return _action_intent(request_id, "pending" if row[3] is None else "committed", intent)
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
        completion_turn = self._completion_turn(completion_digest)
        payload["completion_turn"] = completion_turn
        self._validate_action_order(completion_turn, call_index, tool_name)
        intent: dict[str, Any] = {
            "payload": payload,
            "prior_state_digest": self._meta("state"),
            "prior_head_digest": self._meta("head"),
            "prior_state": self.state(),
            "prior_head_sequence": self._meta("head_sequence"),
        }
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            digest = self.put_json(intent)
            self.connection.execute(
                "INSERT INTO actions VALUES (?,?,?,NULL)", (request_id, payload_digest, digest)
            )
        return _action_intent(request_id, "pending", intent)

    def _validate_action_order(self, completion_turn: int, call_index: int, tool_name: str) -> None:
        latest = self.connection.execute(
            "SELECT turn FROM completions ORDER BY turn DESC LIMIT 1"
        ).fetchone()[0]
        if completion_turn != latest:
            raise ValueError(
                f"action requires latest completion turn {latest}, got {completion_turn}"
            )
        prior_calls = [
            self.read_json(intent_digest)["payload"]
            for (intent_digest,) in self.connection.execute("SELECT intent_digest FROM actions")
        ]
        for call in prior_calls:
            if "completion_turn" not in call:
                call["completion_turn"] = self._completion_turn(str(call["completion_digest"]))
        calls_for_completion = [
            call for call in prior_calls if call["completion_turn"] == completion_turn
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

    def _committed_action_differences(
        self,
        saved: dict[str, Any],
        expected: dict[str, Any],
    ) -> list[str]:
        saved_intent = self.read_json(saved["intent_digest"])
        expected_commit = {
            **{key: value for key, value in expected.items() if key != "intent"},
            "intent": self._action_intent_content(expected["intent"]),
        }
        saved_commit = {
            "request_id": saved["request_id"],
            "intent": self._action_intent_content(saved_intent),
            "result": saved["result"],
            "state": self.read_json(saved["state_digest"]),
            "check": saved["check"],
            "close": saved["close"],
        }
        return compare_identity_fields(expected_commit, saved_commit, "action")

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
            saved = self.read_json(row[1])
            expected = {
                "request_id": request_id,
                "intent": intent,
                "result": result,
                "state": state,
                "check": check,
                "close": close,
            }
            differences = self._committed_action_differences(saved, expected)
            if differences:
                raise ValueError(
                    "committed action payload changed fields: " + ", ".join(differences)
                )
            return self.read_json(row[1])
        if self._meta("closed"):
            raise ValueError("episode is closed")
        differences = self._action_prior_differences(intent)
        if differences:
            raise ValueError("action prior state/head changed: " + ", ".join(differences))
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

    def _action_prior_differences(self, intent: dict[str, Any]) -> list[str]:
        current_state = self.state()
        prior_head_sequence = intent.get("prior_head_sequence")
        if prior_head_sequence is None:
            prior_head_sequence = self._event_sequence(intent["prior_head_digest"]) or 0
        current_head_sequence = self._meta("head_sequence")
        differences = (
            [] if prior_head_sequence == current_head_sequence else ["action.prior_head_sequence"]
        )
        previous_state = intent.get("prior_state")
        if "prior_state" not in intent:
            previous_state = self.read_json(intent["prior_state_digest"])
        differences.extend(compare_identity_fields(previous_state, current_state, "action.state"))
        return differences

    def raw_episode_digest(self) -> str:
        """Return immutable closure identity; open episodes have no final digest."""
        if self._meta("closed") is not True:
            raise ValueError("episode is not closed")
        digest = str(self._meta("closure"))
        closure = self.read_json(digest)
        manifest = closure["raw_manifest"]
        for blob_digest in manifest["blobs"]:
            self.read_blob(blob_digest)
        saved_manifest = self.readable_episode_identity(manifest)
        current_manifest = self.readable_episode_identity(
            self._raw_manifest(blob_digests=manifest["blobs"])
        )
        differences = compare_identity_fields(
            current_manifest, saved_manifest, "raw_episode_manifest"
        )
        if differences:
            raise ValueError("raw episode manifest changed: " + ", ".join(differences))
        expected_progress = {
            "head": self._readable_head_content(self._meta("head")),
            "state": self.state(),
            "checks": self._meta("checks"),
        }
        saved_progress = {
            "head": self._readable_head_content(closure["head"]),
            "state": self.read_json(closure["state_digest"]),
            "checks": closure["checks"],
        }
        differences = compare_identity_fields(
            expected_progress, saved_progress, "raw_episode_closure"
        )
        if differences:
            raise ValueError("raw episode closure changed: " + ", ".join(differences))
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
        raw_episode_digest = self.raw_episode_digest()
        raw_episode_manifest = self.read_json(raw_episode_digest)["raw_manifest"]
        readable_episode_manifest = self.readable_episode_identity(raw_episode_manifest)
        payload = {
            "raw_episode_digest": raw_episode_digest,
            "raw_episode_manifest": raw_episode_manifest,
            "identity": identity,
            "assessment": assessment,
        }
        identity_digest = _digest(_encode(identity))
        current_identity = _assessment_identity_content(identity)
        for (existing_digest,) in self.connection.execute("SELECT digest FROM assessments"):
            existing_payload = self.read_json(existing_digest)
            if not _same_json_value(
                _assessment_identity_content(existing_payload["identity"]), current_identity
            ):
                continue
            existing_manifest = self._assessment_manifest(existing_payload)
            differences = compare_identity_fields(
                readable_episode_manifest, existing_manifest, "raw_episode_manifest"
            )
            if differences:
                continue
            differences = compare_identity_fields(
                assessment, existing_payload["assessment"], "assessment"
            )
            if differences:
                raise ValueError(
                    "assessment identity already has a different payload: " + ", ".join(differences)
                )
            return str(existing_digest)
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            digest = self.put_json(payload)
            self.connection.execute(
                "INSERT INTO assessments VALUES (?,?)", (identity_digest, digest)
            )
        return digest

    def load_assessment(self, identity: dict[str, Any]) -> dict[str, Any] | None:
        """Return only an assessment bound to the requested evaluator and closure."""
        requested_identity = _assessment_identity_content(identity)
        raw_manifest = self.read_json(self.raw_episode_digest())["raw_manifest"]
        current_manifest = self.readable_episode_identity(raw_manifest)
        for (digest,) in self.connection.execute("SELECT digest FROM assessments"):
            payload = self.read_json(digest)
            if not _same_json_value(
                _assessment_identity_content(payload["identity"]), requested_identity
            ):
                continue
            differences = compare_identity_fields(
                current_manifest, self._assessment_manifest(payload), "raw_episode_manifest"
            )
            if differences:
                raise ValueError("stale assessment identity: " + ", ".join(differences))
            return payload["assessment"]
        return None

    def list_assessments(self) -> tuple[dict[str, Any], ...]:
        """Return all immutable sidecars bound to this validated raw closure."""
        raw_manifest = self.read_json(self.raw_episode_digest())["raw_manifest"]
        current_manifest = self.readable_episode_identity(raw_manifest)
        entries: list[dict[str, Any]] = []
        for (digest,) in self.connection.execute("SELECT digest FROM assessments ORDER BY digest"):
            payload = self.read_json(digest)
            differences = compare_identity_fields(
                current_manifest, self._assessment_manifest(payload), "raw_episode_manifest"
            )
            if differences:
                raise ValueError("stale assessment episode: " + ", ".join(differences))
            entries.append({"digest": digest, **payload})
        return tuple(entries)

    def _assessment_manifest(self, payload: dict[str, Any]) -> dict[str, Any]:
        if "raw_episode_manifest" in payload:
            raw_manifest = payload["raw_episode_manifest"]
        else:
            closure = self.read_json(payload["raw_episode_digest"])
            raw_manifest = closure["raw_manifest"]
        return self.readable_episode_identity(raw_manifest)

    def readable_episode_identity(self, raw_manifest: dict[str, Any]) -> dict[str, Any]:
        """Resolve journal references into the readable content of an episode."""
        blob_digests = set(raw_manifest["blobs"])
        completion_records = {
            digest: {"turn": turn, "completion": self.read_json(digest)}
            for turn, digest in raw_manifest["completions"]
        }
        committed_records: dict[str, dict[str, Any]] = {}
        action_records: list[dict[str, Any]] = []
        for request_id, _payload_digest, intent_digest, committed_digest in raw_manifest["actions"]:
            intent = self.read_json(intent_digest)
            action_payload = dict(intent["payload"])
            completion = completion_records[action_payload.pop("completion_digest")]
            action_payload["completion_turn"] = completion["turn"]
            prior_state = intent.get("prior_state")
            if prior_state is None:
                prior_state = self.read_json(intent["prior_state_digest"])
            committed_record = None
            if committed_digest is not None:
                committed_record = self.read_json(committed_digest)
                committed_record = {
                    **{
                        key: value
                        for key, value in committed_record.items()
                        if key not in {"intent_digest", "state_digest"}
                    },
                    "state": self.read_json(committed_record["state_digest"]),
                }
                committed_records[committed_digest] = committed_record
            action_records.append(
                {
                    "request_id": request_id,
                    "payload": action_payload,
                    "prior_state": prior_state,
                    "prior_head_sequence": intent.get("prior_head_sequence"),
                    "committed": committed_record,
                }
            )

        event_records: dict[str, dict[str, Any]] = {}
        for sequence, digest in raw_manifest["events"]:
            event = self.read_json(digest)
            if event["kind"] == "completion":
                record = completion_records.get(event["digest"])
                if record is None:
                    completion_turn = self._completion_turn(event["digest"])
                    record = {
                        "turn": completion_turn,
                        "completion": self.read_json(event["digest"]),
                    }
            else:
                record = committed_records[event["digest"]]
            event_records[digest] = {
                "sequence": sequence,
                "kind": event["kind"],
                "record": record,
            }

        action_contents: list[dict[str, Any]] = []
        for row, action_record in zip(raw_manifest["actions"], action_records, strict=True):
            intent = self.read_json(row[2])
            prior_head = event_records.get(intent["prior_head_digest"])
            prior_head_sequence = intent.get("prior_head_sequence")
            if prior_head_sequence is None:
                prior_head_sequence = 0 if prior_head is None else prior_head["sequence"]
            action_contents.append(
                {
                    **action_record,
                    "prior_head_sequence": prior_head_sequence,
                    "prior_head": prior_head,
                }
            )

        readable_manifest = {
            "generation": raw_manifest["generation"],
            "initial_state": self.read_json(raw_manifest["initial_state"]),
            "events": [event_records[digest] for _, digest in raw_manifest["events"]],
            "completions": sorted(completion_records.values(), key=lambda record: record["turn"]),
            "actions": action_contents,
        }
        return self._resolve_blob_references(readable_manifest, blob_digests)

    def _resolve_blob_references(self, value: Any, blob_digests: set[str]) -> Any:
        if type(value) is dict:
            return {
                key: self._resolve_blob_references(child, blob_digests)
                for key, child in value.items()
            }
        if type(value) is list:
            return [self._resolve_blob_references(child, blob_digests) for child in value]
        if type(value) is str and value in blob_digests:
            return _readable_blob(self.read_blob(value))
        return value

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
