"""Realistic leakage hints for elicitation screens, loaded from a gitignored data file.

A leakage hint is a true statement about how an environment grades (for example, that CI runs
only the visible test), phrased as ordinary project information, never as permission to game
the grader. The texts are prompt material that future models will see, so they live in the
gitignored ``reward_hacking/harness/data/leakage_hints.json``; tracked code only names them.

File schema::

    {"hints": {"<environment>": {"<variant>": "<text>", ...}, ...}}

The variant ``none`` is reserved and always renders the environment's prompt unchanged.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LEAKAGE_HINTS_PATH = REPO_ROOT / "reward_hacking/harness/data/leakage_hints.json"
NO_HINT_VARIANT = "none"


@dataclass(frozen=True)
class LeakageHint:
    """One environment's hint, with the identity fields a run must record."""

    environment: str
    variant: str
    text: str

    @property
    def text_sha256(self) -> str:
        """Digest of the hint text, so a resumed run can refuse a silently edited hint."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def identity(self) -> dict[str, str]:
        """Fields to bind into a run's identity and every per-sample record."""
        return {
            "hint_environment": self.environment,
            "hint_variant": self.variant,
            "hint_sha256": self.text_sha256,
        }


def load_leakage_hint(
    environment: str, variant: str, hints_path: Path = LEAKAGE_HINTS_PATH
) -> LeakageHint:
    """Return the named hint; ``none`` returns an empty hint without reading the file."""
    if variant == NO_HINT_VARIANT:
        return LeakageHint(environment=environment, variant=variant, text="")
    hints_by_environment = json.loads(hints_path.read_text(encoding="utf-8"))["hints"]
    if environment not in hints_by_environment:
        raise KeyError(f"no leakage hints for environment {environment!r} in {hints_path}")
    variants = hints_by_environment[environment]
    if variant not in variants:
        raise KeyError(
            f"environment {environment!r} has no hint variant {variant!r}; known: {sorted(variants)}"
        )
    text = variants[variant]
    if not text.strip():
        raise ValueError(
            f"hint {environment}/{variant} is empty; use the {NO_HINT_VARIANT!r} variant instead"
        )
    return LeakageHint(environment=environment, variant=variant, text=text)
