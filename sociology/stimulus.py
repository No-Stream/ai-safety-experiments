"""Runtime loader for the analysis-model stimulus: frames, request, judge rubric, validation set.

The framing paragraphs, the constant analysis request, the judge rubric, and the judge-validation
replies are authored stimulus and scoring apparatus that will run against future models. Committed,
they become training data and contaminate every measurement made with them, so they live in one
gitignored JSON file that this module loads at runtime and refuses to run without. No stimulus
prose appears in tracked code; this module knows the file's *shape*, never its text.

The frame texts carry an ``{n}`` placeholder where the design doc's texts say "N", rendered as the
bundle's episode count via :meth:`Stimulus.frame_text`. Everything else is used verbatim.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

STIMULUS_PATH = Path("docs/scratch/swarm-sociology-2026-08-31/analysis_model_stimulus.json")

FRAMINGS = ("population", "independent", "unstated")
"""The three provenance framings, in the design doc's order. Keys into the stimulus file."""

STIMULUS_VERSION = "analysis-model-stimulus-v1"


@dataclass(frozen=True, slots=True)
class ValidationReply:
    """One hand-authored synthetic analysis reply and the severity rung it must classify to."""

    name: str
    text: str
    expected_severity: str


@dataclass(frozen=True, slots=True)
class Stimulus:
    """The loaded stimulus texts, plus the digest every artifact records against them.

    ``digest`` hashes the file's canonical JSON, so a stimulus edit is visible in every reply and
    judge row that was produced under it -- the same edit-tripwire role the narration judge's
    prompt digest plays.
    """

    frames: dict[str, str]
    constant_request: str
    judge_instructions: str
    validation_replies: tuple[ValidationReply, ...]
    digest: str

    def frame_text(self, framing: str, *, n: int) -> str:
        """Render one framing paragraph with the bundle's episode count in place of ``{n}``."""
        if framing not in self.frames:
            raise ValueError(f"unknown framing {framing!r}; the stimulus file has {FRAMINGS}")
        return self.frames[framing].format(n=n)


def load_stimulus(path: Path = STIMULUS_PATH) -> Stimulus:
    """Load and validate the stimulus file, refusing absence loudly rather than defaulting.

    A default here would either be committed stimulus prose, which this public repository must
    never carry, or empty strings, which would render frame-less prompts that measure nothing the
    design describes. The refusal names the path and why the file is machine-local.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"stimulus file {path} is missing. It is gitignored on purpose (the frame texts and "
            "judge rubric are authored stimulus that must never be committed); a fresh clone does "
            "not contain it. Recreate it from the design doc in the same directory."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("version")
    if version != STIMULUS_VERSION:
        raise ValueError(
            f"stimulus file {path} has version {version!r}, expected {STIMULUS_VERSION!r}"
        )
    frames = payload["frames"]
    missing = [name for name in FRAMINGS if not str(frames.get(name, "")).strip()]
    if missing:
        raise ValueError(f"stimulus file {path} is missing frame text for {missing}")
    validation = tuple(
        ValidationReply(
            name=str(entry["name"]),
            text=str(entry["text"]),
            expected_severity=str(entry["expected_severity"]),
        )
        for entry in payload.get("validation_replies", [])
    )
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return Stimulus(
        frames={name: str(frames[name]) for name in FRAMINGS},
        constant_request=str(payload["constant_request"]),
        judge_instructions=str(payload["judge_instructions"]),
        validation_replies=validation,
        digest=hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16],
    )
