"""Pin the cross-module contracts that no single module's tests cover.

Two of them, both offline.

:class:`TestPayoffVariantsAreDerivedNotRestated` verifies a claim rather than trusting it. Adding
the second stag-hunt payoff variant was supposed to need no edit in `games/train.py`, because
`validate_arms` derives each arm's legal variant vocabulary by generating that arm's own rows. If
that were wrong -- a hardcoded list somewhere -- a new variant would either be rejected by a stale
allowlist or, worse, silently accepted while matching no rows, which is how a pinned arm ends up
training on an empty corpus. The test drives the real `validate_arms` in both directions: the new
variant must pass, and a variant belonging to a different game must still fail.

:class:`TestProvenance` covers `games/provenance.py`, which replaced three near-duplicate
implementations. The behaviour that matters is the part each copy had separately: the environment
variable wins (a Batch image bakes its sha in and has no checkout), and nothing raises when git
cannot answer, because provenance is metadata about work and must never destroy the work.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING

import pytest

from games import provenance
from games.arms import GameArm, arm_payoff_variants, validate_arms
from games.payoffs import STAG_HUNT_VARIANTS
from games.prompts import STAG_HUNT_PAYOFF_VARIANTS
from games.provenance import GIT_SHA_ENV, UNKNOWN_SHA, git_provenance, git_sha, git_tree_dirty

if TYPE_CHECKING:
    from pathlib import Path

STAG_HUNT_ARM = GameArm(game_id="stag-hunt", grading="group-mix", notes="test arm")


class TestPayoffVariantsAreDerivedNotRestated:
    def test_the_generator_reports_both_stag_variants(self) -> None:
        """The derivation itself: train.py asks the corpus, so payoffs.py is the single source."""
        assert arm_payoff_variants(STAG_HUNT_ARM) == set(STAG_HUNT_VARIANTS)
        assert arm_payoff_variants(STAG_HUNT_ARM) == set(STAG_HUNT_PAYOFF_VARIANTS)

    @pytest.mark.parametrize("variant", sorted(STAG_HUNT_VARIANTS))
    def test_an_arm_may_pin_either_new_variant_with_no_train_py_edit(self, variant: str) -> None:
        arm = GameArm(
            game_id="stag-hunt",
            grading="group-mix",
            notes="pins one variant",
            payoff_variants=(variant,),
        )
        validate_arms({"stag-hunt-pinned": arm})

    def test_a_variant_from_a_different_game_is_still_rejected(self) -> None:
        """The other direction: a name that exists somewhere but not in this game must fail.

        Without this the derivation could be "accept everything", which would pass the test above
        while letting a pinned arm select nothing.
        """
        arm = GameArm(
            game_id="stag-hunt",
            grading="group-mix",
            notes="pins a temptation variant the stag hunt has no concept of",
            payoff_variants=("temptation-2",),
        )
        with pytest.raises(ValueError, match="that its own corpus does not carry"):
            validate_arms({"stag-hunt-wrong-variant": arm})

    def test_a_stag_variant_pinned_on_the_prisoners_dilemma_is_rejected(self) -> None:
        arm = GameArm(
            game_id="twin-pd",
            grading="group-mix",
            notes="pins a stag-hunt variant on a PD",
            payoff_variants=("safe-hunt",),
        )
        with pytest.raises(ValueError, match="that its own corpus does not carry"):
            validate_arms({"twin-pd-wrong-variant": arm})

    def test_the_shipped_registry_still_validates(self) -> None:
        """The real ARMS registry is validated at import; assert it rather than assume it held."""
        from games.train import ARMS  # noqa: PLC0415

        validate_arms(ARMS)


class TestProvenance:
    def test_the_baked_environment_variable_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A Batch image bakes GIT_SHA in and carries no checkout, so it has to take precedence."""
        monkeypatch.setenv(GIT_SHA_ENV, "deadbeefcafe")
        assert git_sha() == "deadbeefcafe"

    def test_it_falls_back_to_asking_git(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(GIT_SHA_ENV, raising=False)
        sha = git_sha()
        assert sha
        assert not sha.startswith(UNKNOWN_SHA)
        assert len(sha) >= 7

    def test_an_unanswerable_sha_is_reported_not_raised(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Losing the ability to name a commit must not throw away the run that produced data."""
        monkeypatch.delenv(GIT_SHA_ENV, raising=False)
        monkeypatch.setattr(provenance, "REPO_ROOT", tmp_path)
        sha = git_sha()
        assert sha.startswith(UNKNOWN_SHA)
        assert "git rev-parse failed" in sha

    def test_a_dirty_flag_is_a_bool_in_a_real_checkout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(GIT_SHA_ENV, raising=False)
        assert isinstance(git_tree_dirty(), bool)

    def test_an_unanswerable_dirty_flag_is_none_not_false(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """ "Clean" and "nobody could tell" are different claims about an artifact."""
        monkeypatch.setattr(provenance, "REPO_ROOT", tmp_path)
        assert git_tree_dirty() is None

    def test_the_provenance_record_carries_both_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(GIT_SHA_ENV, "abc1234")
        record = git_provenance()
        assert record["git_sha"] == "abc1234"
        assert "git_tree_dirty" in record

    def test_a_custom_environment_variable_can_be_used(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SOME_OTHER_SHA", "feedface")
        assert git_sha(env_var="SOME_OTHER_SHA") == "feedface"

    def test_importing_it_runs_no_subprocess(self) -> None:
        """It is imported inside a training loop, so import must stay free of side effects.

        Checked in a fresh interpreter, because by the time any assertion in this process could
        look, the module is already imported and whatever it did at import is over. The two
        assertions this replaces -- that `git_sha` is callable and `REPO_ROOT` is a directory --
        observed nothing about subprocess activity and stayed green with `git_sha()` called at
        module level. `Popen` is the patch point rather than `run`, so `check_output` and direct
        `Popen` callers are covered by the same guard.
        """
        guard = textwrap.dedent("""
            import subprocess
            import sys

            subprocess.Popen = lambda *args, **kwargs: sys.exit(
                "games.provenance shelled out at import"
            )
            import games.provenance
        """)
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-c", guard],
            cwd=provenance.REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


class TestEveryGamesModuleUsesTheSharedProvenance:
    def test_select_prompts_and_evals_import_it(self) -> None:
        """Guards against a fourth copy quietly reappearing in one of the callers."""
        from games import evals, select_prompts  # noqa: PLC0415

        assert evals.git_sha is git_sha
        assert select_prompts.git_provenance is git_provenance
