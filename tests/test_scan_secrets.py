"""Pin the privacy scanner, and sweep this checkout with it.

The scanner had no test and no caller: no hook, no make target, nothing in ``make test``. A gate
nobody runs is a file, so the sweep at the bottom of this module is the point of the whole thing --
it runs every detector over every file ``git add -A`` would carry, which is what the public remote
ends up with. New files included, deliberately: see that class's docstring.

Its patterns also covered credential shapes only, which is the smaller half of what may never be
committed here. The repo's rule also forbids AWS account ids, role ARNs, email addresses, home
directory paths carrying a username, and benchmark item material, none of which look like a
credential. Those detectors are pinned individually below, each with the placeholder case that must
*not* fire beside the real case that must -- a privacy gate that flags ``123456789012`` is a gate
that gets ignored, and an ignored gate catches nothing.

**Every sample value in this file is assembled from parts rather than written as a literal**, so
that the sweep does not report this module's own test data. That is deliberate and load-bearing: the
alternative is an exclusion list, and an exclusion list in a privacy gate is a way to silence a real
finding. Nothing here is excluded from anything.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

from scripts import scan_secrets
from scripts.scan_secrets import (
    DETECTORS,
    EXIT_UNARMED,
    CanaryToken,
    Finding,
    InstrumentSources,
    collect_instrument_sources,
    instrument_text_findings,
    literal_canary,
    load_canary_tokens,
    main,
    scan_paths,
    scan_text,
    sources_from_json_payload,
    sources_from_note_text,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CANARY_FILE = REPO_ROOT / "canary" / "privacy-values.txt"

# The forge's own domains, where an address is a publishing handle rather than a person's mailbox.
PUBLISHING_IDENTITY_DOMAINS = frozenset({"github.com", "users.noreply.github.com"})

# Split at the @ like every address here, so this module's own data does not trip its own sweep.
ALLOWED_COMMIT_IDENTITIES = frozenset(
    {
        "No-Stream <No-Stream" + "@" + "users.noreply.github.com>",
        "GitHub <noreply" + "@" + "github.com>",
        "ship-tree <ship-tree" + "@" + "invalid>",
    }
)

# Named for its shape: an account-shaped constant name is itself the context the detector needs.
TWELVE_DIGITS = "918273645102"
OWNER_USER = "someengineer"
EMAIL = "analyst" + "@" + "somecorp.zz"
ACCESS_KEY_ID = "AKIA" + "ABCDEFGHIJKLMNOP"
PEM_HEADER = "-----BEGIN " + "RSA PRIVATE KEY-----"


def _detectors(findings: list[Finding]) -> set[str]:
    """The set of detector names that fired, which is what these tests assert on."""
    return {finding.detector for finding in findings}


def _scan(text: str) -> set[str]:
    """Scan one blob of text with every detector and no canaries."""
    return _detectors(scan_text("sample.txt", text, ()))


class TestCredentialShapesStillFire:
    """The original family, kept honest through the detector-table change."""

    def test_an_access_key_id_is_found(self) -> None:
        assert "aws_access_key_id" in _scan(f"the key is {ACCESS_KEY_ID} ok")

    def test_a_private_key_header_is_found(self) -> None:
        assert "pem_private_key" in _scan(PEM_HEADER)

    def test_ordinary_text_is_not_a_finding(self) -> None:
        assert _scan("step 42 reward 0.31 completion tokens 128") == set()

    def test_the_match_text_is_never_reported(self) -> None:
        """A scanner that prints what it found copies the secret into a second log."""
        findings = scan_text("sample.txt", f"key={ACCESS_KEY_ID}", ())
        rendered = str(findings[0])
        assert ACCESS_KEY_ID not in rendered
        assert "aws_access_key_id" in rendered


class TestMachineLocalIdentity:
    """Account ids, ARNs, emails, home paths: the privacy list, none of it credential-shaped."""

    def test_an_email_address_is_found(self) -> None:
        assert "email_address" in _scan(f"ask {EMAIL} about the run")

    def test_a_documentation_domain_address_is_not_a_finding(self) -> None:
        """RFC 2606 reserves these, and a sibling test suite sets git's user.email to one."""
        assert _scan("tests@example.invalid") == set()
        assert _scan("someone@sub.example.com") == set()
        assert _scan("dev@box.localhost") == set()

    def test_a_lookalike_domain_is_still_a_finding(self) -> None:
        """The exemption is for the reserved domains, not for anything containing their names."""
        assert "email_address" in _scan("someone" + "@" + "notexample.com")

    def test_an_account_id_in_account_context_is_found(self) -> None:
        assert "aws_account_id" in _scan(f'"account_id": "{TWELVE_DIGITS}"')

    def test_a_bucket_owner_field_is_account_context(self) -> None:
        assert "aws_account_id" in _scan(f"s3BucketOwner={TWELVE_DIGITS}")

    def test_a_documented_example_account_is_not_a_finding(self) -> None:
        """AWS reserves these for docs, and this repo's tests use three of them."""
        assert _scan('"account_id": "123456789012"') == set()
        assert _scan('"bucket_owner": "111122223333"') == set()

    def test_a_twelve_digit_number_without_account_context_is_not_a_finding(self) -> None:
        """Token counts and timestamps are twelve digits often enough to kill the gate."""
        assert _scan(f"tokens_processed: {TWELVE_DIGITS}") == set()

    def test_a_bare_account_id_beside_a_region_is_found(self) -> None:
        """The shape the keyword form cannot see: no account-ish word anywhere on the line."""
        assert "aws_account_id_beside_region" in _scan(
            f"states:us-west-2:{TWELVE_DIGITS}:stateMachine:bake"
        )
        assert "aws_account_id_beside_region" in _scan(f"{TWELVE_DIGITS} eu-central-1 golden")

    def test_a_documented_example_account_beside_a_region_is_not_a_finding(self) -> None:
        assert _scan("us-west-2 123456789012 job-definition") == set()

    def test_a_twelve_digit_float_expansion_is_not_a_finding(self) -> None:
        """The live false-positive class: `1/7` and `1/sqrt(3)` in this repo's trace fixtures.

        A digit-run detector that fires on these would report a standing set of non-findings across
        games/tests/data, and a gate with standing findings gets waved through.
        """
        assert _scan('"rate": 0.142857142857, "other": 0.577350269190') == set()
        assert _scan("co-author-1 918273645102") == set()

    def test_a_role_arn_carrying_a_real_account_is_found(self) -> None:
        assert "aws_arn" in _scan(f"arn:aws:iam::{TWELVE_DIGITS}:role/BedrockBatchInferenceRole")

    def test_an_arn_carrying_an_example_account_is_not_a_finding(self) -> None:
        assert _scan("arn:aws:batch:us-west-2:123456789012:job-definition/games") == set()

    def test_an_ecr_registry_host_is_found(self) -> None:
        assert "ecr_registry" in _scan(
            f"{TWELVE_DIGITS}.dkr.ecr.us-west-2.amazonaws.com/games:latest"
        )

    def test_a_home_path_naming_a_real_user_is_found(self) -> None:
        assert "home_directory_path" in _scan(f"/home/{OWNER_USER}/workspace/run.log")

    def test_a_local_home_path_is_found_too(self) -> None:
        """This box's homes live under /local/home, which a /home-only pattern would miss."""
        assert "home_directory_path" in _scan(f"/local/home/{OWNER_USER}/notes.md")

    def test_placeholder_home_users_are_not_findings(self) -> None:
        """The repo's docs and jail fixtures use these on purpose."""
        assert _scan("/home/ubuntu/games/scripts/idle_watchdog.sh") == set()
        assert _scan("/home/user/miniconda3") == set()


class TestBenchmarkMaterial:
    """Item texts and registered answers, which contaminate the benchmark once published."""

    def _record(self) -> str:
        """One serialized trace record, built as a dict so no source line carries both keys."""
        record = {
            "item_id": "rb-sample-0001",
            "arm": "flawed",
            "true_answer": "the registered value",
            "flawed_answer": "the planted value",
        }
        return json.dumps(record)

    def test_a_serialized_record_is_found(self) -> None:
        assert "benchmark_item_record" in _scan(self._record())

    def test_a_registered_field_alone_on_a_line_is_not_a_finding(self) -> None:
        """The shape of a synthetic fixture in this repo's own tests, which is not item material."""
        assert _scan('        "true_answer": "5",') == set()

    def test_an_item_id_alone_on_a_line_is_not_a_finding(self) -> None:
        assert _scan('        "item_id": "rb-sample-0001",') == set()

    def test_a_pretty_printed_item_file_is_found_whole_file(self, tmp_path: Path) -> None:
        """A pretty-printed item has one key per line, so no line detector can see it.

        This is the shape ``recoverybench.load_items`` reads, and the shape a committed corpus would
        take, so missing it would leave the detector precise and useless.
        """
        item = tmp_path / "item.json"
        item.write_text(
            json.dumps(
                {
                    "item_id": "rb-sample-0001",
                    "flaw_type": "wrong_method",
                    "true_answer": "the registered value",
                },
                indent=2,
            )
        )

        findings = scan_paths([str(item)], (), 1 << 20)

        assert _detectors(findings) == {"benchmark_item_file"}

    def test_an_unrelated_json_file_is_not_a_finding(self, tmp_path: Path) -> None:
        config = tmp_path / "config.json"
        config.write_text(json.dumps({"model": "Qwen/Qwen3.5-4B", "steps": 70}, indent=2))

        assert scan_paths([str(config)], (), 1 << 20) == []

    def test_a_pretty_printed_item_pasted_into_markdown_is_found(self, tmp_path: Path) -> None:
        """The route an item takes into a tracked path: pasted into a note, not shipped as a corpus.

        The whole-file check was ``.json`` only, so an item hand-formatted into a scratch note that
        later moves under a tracked path passed clean -- one key per line defeats the line detector
        and the extension guard defeated the whole-file one. Markdown carries no synthetic fixtures,
        which is what made the same extension unsafe for Python.
        """
        note = tmp_path / "handoff.md"
        note.write_text(
            "Notes on the failing item:\n\n```json\n"
            + json.dumps(
                {
                    "item_id": "rb-sample-0001",
                    "flaw_type": "wrong_method",
                    "true_answer": "the registered value",
                },
                indent=2,
            )
            + "\n```\n"
        )

        findings = scan_paths([str(note)], (), 1 << 20)

        assert _detectors(findings) == {"benchmark_item_file"}

    def test_a_pretty_printed_dict_in_python_stays_silent(self, tmp_path: Path) -> None:
        """The deliberate exclusion, pinned so extending the guard cannot quietly swallow it.

        This repo's own tests build item-shaped dicts, so flagging Python would hand the gate a
        standing set of non-findings and teach everyone to ignore it.

        The fixture is assembled from a dict for the reason :meth:`_record` is: as one literal, both
        keys land on a line of *this* file and the line detector flags the test. Splitting that
        literal by hand does not survive ``ruff format``, which joins it straight back.
        """
        fixture = tmp_path / "conftest.py"
        item = {
            "item_id": "toy-sum",
            "true_answer": "5",
        }
        fixture.write_text("ITEM = " + json.dumps(item, indent=4) + "\n")

        assert scan_paths([str(fixture)], (), 1 << 20) == []


class TestCanaryTokens:
    """The family for values no shape can describe: a username, a bucket, a registered answer."""

    def test_a_canary_token_is_matched_literally(self) -> None:
        findings = scan_text(
            "sample.txt",
            "the run wrote to my-private-bucket",
            (literal_canary("my-private-bucket"),),
        )
        assert _detectors(findings) == {"canary_token"}

    def test_a_short_token_is_rejected_rather_than_warned_about(self, tmp_path: Path) -> None:
        tokens = tmp_path / "canaries.txt"
        tokens.write_text("# a comment\nabc\n")

        with pytest.raises(ValueError, match="canary token too short"):
            load_canary_tokens(str(tokens))

    def test_a_literal_token_below_the_substring_floor_is_still_rejected(
        self, tmp_path: Path
    ) -> None:
        """The bounded form lowers the floor; it does not lower it for the substring form."""
        tokens = tmp_path / "canaries.txt"
        tokens.write_text("zetch\n")

        with pytest.raises(ValueError, match="floor 8"):
            load_canary_tokens(str(tokens))

    def test_a_short_token_is_armable_when_word_bounded(self, tmp_path: Path) -> None:
        """The entry the 8-character floor used to make impossible: a six-letter employer name."""
        tokens = tmp_path / "canaries.txt"
        tokens.write_text("word:quorbl\n")

        canaries = load_canary_tokens(str(tokens))

        assert _detectors(scan_text("s.txt", "employed at Quorbl since", canaries)) == {
            "canary_token"
        }

    def test_a_word_bounded_token_does_not_match_inside_a_longer_word(self, tmp_path: Path) -> None:
        """The whole point of the lowered floor: a boundary buys back the substring floor's job."""
        tokens = tmp_path / "canaries.txt"
        tokens.write_text("word:quorbl\n")

        canaries = load_canary_tokens(str(tokens))

        assert scan_text("s.txt", "the quorbling flange and quorbls", canaries) == []

    def test_a_bounded_token_below_its_own_floor_is_rejected(self, tmp_path: Path) -> None:
        tokens = tmp_path / "canaries.txt"
        tokens.write_text("word:abc\n")

        with pytest.raises(ValueError, match="word-bounded canary token too short"):
            load_canary_tokens(str(tokens))

    def test_an_unless_context_excuses_a_documented_legitimate_use(self, tmp_path: Path) -> None:
        """Without this a short word with public uses hands the gate standing findings."""
        tokens = tmp_path / "canaries.txt"
        tokens.write_text("word:quorbl unless=nova,public image\n")

        canaries = load_canary_tokens(str(tokens))

        assert scan_text("s.txt", "the quorbl nova-micro model id", canaries) == []
        assert scan_text("s.txt", "Quorbl's public images are fine", canaries) == []
        assert _detectors(scan_text("s.txt", "employed at Quorbl since", canaries)) == {
            "canary_token"
        }

    def test_an_unless_context_out_of_reach_does_not_excuse_the_match(self, tmp_path: Path) -> None:
        """A mention far down the same line is not the surrounding the exemption is for."""
        tokens = tmp_path / "canaries.txt"
        tokens.write_text("word:quorbl unless=nova\n")

        canaries = load_canary_tokens(str(tokens))
        far = "employed at Quorbl since" + " x" * 40 + " nova"

        assert _detectors(scan_text("s.txt", far, canaries)) == {"canary_token"}

    def test_an_unless_with_no_contexts_is_a_loading_error(self, tmp_path: Path) -> None:
        """Silently arming a bare token would be the same value under a weaker rule than written."""
        tokens = tmp_path / "canaries.txt"
        tokens.write_text("my-private-bucket unless=\n")

        with pytest.raises(ValueError, match="with no contexts after it"):
            load_canary_tokens(str(tokens))

    def test_the_finding_never_contains_the_token(self) -> None:
        findings = scan_text(
            "s.txt", "wrote to my-private-bucket", (literal_canary("my-p" + "rivate-bucket"),)
        )
        assert "private-bucket" not in str(findings[0])

    def test_no_canary_file_means_no_canary_detector(self) -> None:
        """Still true of the loader, and no longer the end of the story.

        The loader returning nothing is correct -- there is no file to read. What used to be wrong is
        what the callers did with it: every one of them scanned shape-only and exited 0, which reads
        identically to a tree that carries none of those values. ``TestUnarmedIsLoud`` below pins the
        refusal, and ``TestTheCommittableTreeIsClean`` arms this file's real tokens.
        """
        assert load_canary_tokens(None) == ()


class TestEveryDetectorIsReachable:
    """A detector table nothing exercises is the same failure as a gate nothing runs."""

    def test_this_module_covers_every_detector(self) -> None:
        """Names are asserted, not counted, so a new detector fails here until it has a test."""
        pinned = {
            "aws_access_key_id",
            "aws_secret_access_key",
            "aws_session_token",
            "pem_private_key",
            "ssh_private_key",
            "jwt",
            "bearer_token",
            "session_cookie",
            "generic_secret_assignment",
            "email_address",
            "aws_account_id",
            "aws_account_id_beside_region",
            "aws_arn",
            "ecr_registry",
            "home_directory_path",
            "benchmark_item_record",
        }
        assert {detector.name for detector in DETECTORS} == pinned


class TestInstrumentItemText:
    """The instrument-text detector, pinned on synthetic sources.

    Every phrase and payoff pair below was invented for this file. Real item text may not appear
    in tracked code even inside the guard that protects it -- a guard listing the contraband IS
    the leak, which is the exact failure this detector was written after catching.
    """

    @staticmethod
    def _sources() -> InstrumentSources:
        return InstrumentSources(
            phrases=frozenset({"quorble the flange before breakfast", "zetch min gorp"}),
            payoff_items=(frozenset({(481, 79), (541, 279)}),),
            source_paths=(),
        )

    def test_a_planted_phrase_is_found_despite_case_and_punctuation(self) -> None:
        text = 'the docstring says "Quorble the flange, before breakfast" and moves on'
        findings = instrument_text_findings("sample.py", text, self._sources())
        assert [finding.detector for finding in findings] == ["instrument_item_text"]

    def test_a_phrase_wrapped_across_a_line_break_is_still_found(self) -> None:
        """The one fragment that really reached a tracked docstring wrapped mid-phrase, where
        every line-oriented detector is blind."""
        text = "asking quorble the\nflange before breakfast is scored inverted"
        findings = instrument_text_findings("sample.py", text, self._sources())
        assert len(findings) == 1

    def test_a_short_whole_item_is_found(self) -> None:
        findings = instrument_text_findings("s.py", "zetch min gorp!", self._sources())
        assert len(findings) == 1

    def test_unrelated_text_is_clean(self) -> None:
        text = "reward 0.31 at step 42; the flange quorbles nobody before breakfast"
        assert instrument_text_findings("sample.py", text, self._sources()) == []

    def test_the_finding_never_contains_the_phrase(self) -> None:
        findings = instrument_text_findings("s.py", "zetch min gorp", self._sources())
        assert "zetch" not in str(findings[0])

    def test_two_payoff_pairs_of_one_item_fire_and_one_alone_does_not(self) -> None:
        both = "option_payoffs=((481, 79), (541, 279))"
        assert [
            finding.detector for finding in instrument_text_findings("s.py", both, self._sources())
        ] == ["instrument_item_payoffs"]
        one = "a single (481, 79) somewhere"
        assert instrument_text_findings("s.py", one, self._sources()) == []

    def test_two_matched_pairs_far_apart_are_a_coincidence_not_a_finding(self) -> None:
        """The live false positive this window fixed: an item whose payoffs are small numbers
        matched uv.lock and two arithmetic notebooks on pairs thousands of digit runs apart. A
        pasted payoff table is consecutive pairs, so distance is what tells a paste from noise."""
        filler = "x 7 " * 300
        text = f"(481, 79) {filler} (541, 279)"
        assert instrument_text_findings("s.py", text, self._sources()) == []

    def test_two_matched_pairs_near_each_other_still_fire_through_filler(self) -> None:
        filler = "x 7 " * 50
        text = f"(481, 79) {filler} (541, 279)"
        assert [
            finding.detector for finding in instrument_text_findings("s.py", text, self._sources())
        ] == ["instrument_item_payoffs"]

    def test_the_source_file_itself_is_exempt(self, tmp_path: Path) -> None:
        source = tmp_path / "published.json"
        source.write_text("zetch min gorp")
        sources = InstrumentSources(
            phrases=frozenset({"zetch min gorp"}),
            payoff_items=(),
            source_paths=(str(source.resolve()),),
        )
        assert instrument_text_findings(str(source), source.read_text(), sources) == []

    def test_json_extraction_reads_stems_twins_anchors_and_payoffs(self) -> None:
        payload = {
            "instruments": {
                "synthetic": {
                    "instructions": "Grade every flange by its quorbling speed today.",
                    "anchors": ["never ever at all", "always", "no"],
                    "items": [
                        {
                            "stem": "Zarp the wemble until the gorp subsides.",
                            "neutral_stems": ["Adjust the wemble until the gorp subsides."],
                        },
                        {"option_payoffs": [[481, 79], [541, 279], [491, 491]]},
                    ],
                }
            }
        }
        phrases, payoff_items = sources_from_json_payload(payload)
        assert "zarp the wemble until the" in phrases
        assert "adjust the wemble until the" in phrases
        assert "never ever at all" in phrases
        assert "grade every flange by its" in phrases
        # One- and two-word anchors are too generic to match on, so they contribute nothing.
        assert "always" not in phrases
        assert payoff_items == (frozenset({(481, 79), (541, 279), (491, 491)}),)

    def test_generic_survey_boilerplate_is_not_a_phrase(self) -> None:
        payload = {"instructions": "There are no right or wrong answers here."}
        phrases, _ = sources_from_json_payload(payload)
        assert "there are no right or" not in phrases
        assert "are no right or wrong" not in phrases

    def test_generic_game_prose_shingles_are_excluded_everywhere(self) -> None:
        """The authored stems open with matrix-game boilerplate that tracked docstrings also use.

        Excluded at extraction, in every field, or the guard holds a standing finding against
        `games/prompts.py` -- and a gate with standing findings gets waved through. Every stem
        keeps its distinctive shingles armed beside these.
        """
        payload = {
            "items": {"x": {"stem": "Two parties choose at the same instant, wearing hats."}}
        }
        phrases, _ = sources_from_json_payload(payload)
        assert "two parties choose at the" not in phrases
        assert "parties choose at the same" not in phrases
        assert "choose at the same instant" in phrases

    def test_authored_json_fields_arm_the_detector(self) -> None:
        """`authored.json` speaks `stem_swapped`, `options` and `vocabulary`; the first two carry
        item text and must extract, so the guard arms itself from the file with no further wiring."""
        payload = {
            "items": {
                "synthetic-item": {
                    "stem": "Zarp the wemble until the gorp subsides.",
                    "stem_swapped": "Quorble the flange until the wemble relents.",
                    "options": ["never ever at all costs", "gladly and at once"],
                }
            }
        }
        phrases, _ = sources_from_json_payload(payload)
        assert "zarp the wemble until the" in phrases
        assert "quorble the flange until the" in phrases
        assert "never ever at all costs" in phrases

    def test_note_extraction_reads_numbered_items_and_table_pairs(self) -> None:
        note = (
            "## 1. Synthetic scale\n"
            "\n"
            "1. Zarp the wemble until the gorp subsides\n"
            "2. Never quorble a flange sideways\n"
            "\n"
            "| Item | A | B | C |\n"
            "|---|---|---|---|\n"
            "| 1 | 481 / 79 | 541 / 279 | 491 / 491 |\n"
        )
        phrases, payoff_items = sources_from_note_text(note)
        assert "zarp the wemble until the" in phrases
        assert "never quorble a flange sideways" in phrases
        assert payoff_items == (frozenset({(481, 79), (541, 279), (491, 491)}),)

    def test_round_number_pairs_are_excluded_as_unmatchable(self) -> None:
        """Pairs like 50/100 collide with ordinary game arithmetic and cannot identify an item."""
        _, payoff_items = sources_from_json_payload(
            {"option_payoffs": [[50, 100], [100, 50], [70, 100]]}
        )
        assert payoff_items == ()

    def test_citation_shaped_digit_runs_are_not_phrases(self) -> None:
        phrases, _ = sources_from_note_text("1. Journal of Flanges 90(1), 31-34, DOI 10/9999\n")
        assert not any("90" in phrase for phrase in phrases)

    def test_a_multi_kilobyte_digit_blob_does_not_crash_the_payoff_matcher(self) -> None:
        """CPython caps int() conversion at 4300 digits; a blob line must be skipped, not fatal.

        Found live: a scratch artifact carrying a ten-thousand-digit run crashed the sweep. No
        instrument payoff has more than a few digits, so long runs carry no signal anyway.
        """
        blob = "9" * 10_000 + " then (481, 79) and (541, 279)"
        findings = instrument_text_findings("blob.txt", blob, self._sources())
        assert [finding.detector for finding in findings] == ["instrument_item_payoffs"]


class TestTheCommittableTreeIsClean:
    """The sweep: every detector over every file a commit would carry.

    This is the wiring the scanner was missing. ``make privacy-scan`` runs the same scan from the
    shell; running it here too means the gate cannot be forgotten, because it is inside ``make
    test``. A finding is not automatically a leak -- a new placeholder in a doc could trip it -- but
    it always needs a human read before the commit lands.
    """

    def test_no_committable_file_carries_privacy_material(self) -> None:
        """``--others --exclude-standard`` is the whole point, and was watched to matter.

        Plain ``git ls-files`` lists the index, so a brand-new file is invisible to it until it has
        been committed -- which is after the leak is in the history this gate protects. A planted
        home path in a new file passed the index-only version of this test and passed
        ``make privacy-scan`` with it. This lists what ``git add -A`` would carry instead, with
        .gitignore honoured, so ``artifacts/`` stays out and a new file is covered.

        Armed with this box's real canary tokens, which it was not: the call passed a hardcoded empty
        tuple, so the sweep advertised as the one that cannot be forgotten was shape-only and every
        value in the token file read as clean here while ``make privacy-scan`` caught it.
        """
        committable = _committable_paths()
        assert len(committable) > 100, f"only {len(committable)} files listed; is this a checkout?"

        findings = scan_paths(
            committable,
            _armed_canaries(),
            64 * 1024 * 1024,
            instrument_sources=collect_instrument_sources(REPO_ROOT),
        )

        assert findings == [], "\n".join(str(finding) for finding in findings)


def _committable_paths() -> list[str]:
    """Every path ``git add -A`` would carry, which is what the public remote ends up with."""
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],  # noqa: S607
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [str(REPO_ROOT / name) for name in listed.stdout.split("\0") if name]


def _armed_canaries() -> tuple[CanaryToken, ...]:
    """This box's real tokens, skipping loudly rather than silently where the file is absent."""
    if not CANARY_FILE.is_file():
        pytest.skip(
            f"no {CANARY_FILE.relative_to(REPO_ROOT)} on this machine (fresh clone), so the canary "
            "detector has nothing to match and this sweep would be shape-only. Create it from "
            "canary/privacy-values.txt.example; canary/README.md says what belongs in it."
        )
    canaries = load_canary_tokens(str(CANARY_FILE))
    assert canaries, (
        f"{CANARY_FILE} exists but armed no tokens, so every value it was written to guard reads "
        "as clean. A token file that loads empty is worse than none: it looks armed."
    )
    return canaries


def _git(*args: str) -> str:
    """Read something out of git, crashing loudly rather than returning an empty answer."""
    return subprocess.run(  # noqa: S603 - arguments are literals from this module's call sites
        ["git", *args],  # noqa: S607 - resolved from PATH, as every other git call here is
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _history_identities() -> set[str]:
    """Every distinct ``Name <address>`` that authored or committed anything reachable from HEAD."""
    listed = _git("log", "--format=%an <%ae>%n%cn <%ce>", "HEAD")
    return {line for line in listed.splitlines() if line.strip()}


def _identity_findings(identity: str, canaries: tuple[CanaryToken, ...]) -> list[Finding]:
    """Scan one identity string, exempting only a publishing address's own shape.

    An identity is an email address by construction, so ``email_address`` fires on the pseudonymous
    handle this history was deliberately collapsed onto just as it would on a work address. Exempting
    that detector for the forge domains, and nothing else, is what keeps this check free of standing
    findings; a canary hit is never exempted, so a real value inside a forge-shaped identity still
    reports.
    """
    findings = scan_text("git-identity", identity, canaries)
    domain = identity.rpartition("@")[2].rstrip(">").lower()
    if domain in PUBLISHING_IDENTITY_DOMAINS:
        return [finding for finding in findings if finding.detector != "email_address"]
    return findings


class TestTheCommittingIdentityIsSafe:
    """The author and committer fields, which no other gate reads.

    Found by rehearsal: a squash rehearsed in a throwaway clone came out authored as the owner's real
    OS username at their employer's domain, because a clone does not inherit the source repository's
    ``--local`` identity and this repository's pseudonymous one is set locally. The cleanup step would
    have introduced exactly the class it exists to remove, and the same fallback applies to every
    commit made from a fresh clone on another machine. The scanner covers file content and, through
    the commit-msg hook, the message draft; neither reads these two fields.

    Two questions are asked, because either alone leaves a hole. Is this one of the identities that
    may appear here at all -- an allowlist, which catches a real name that no detector or token could
    -- and would the scanner object to it, which is what covers an identity somebody legitimately adds
    later. ``docs/privacy-gate.md`` carries the reasoning.
    """

    def test_the_resolved_identity_for_this_repo_is_allowed(self) -> None:
        """What a commit made right now would be attributed to, config precedence included.

        The comparison is reduced to a bool before the assert on purpose. pytest rewrites a bare
        ``assert x in Y`` into a report that prints ``x``, which for this check is the very value that
        must not be echoed -- watched happening, on the rehearsal identity, before this line existed.
        """
        resolved = f"{_git('config', 'user.name').strip()} <{_git('config', 'user.email').strip()}>"
        is_allowed = resolved in ALLOWED_COMMIT_IDENTITIES

        assert is_allowed, (
            "the identity a commit from this checkout would carry is not one of this repository's "
            "publishing identities. A clone does not inherit a --local identity, so this is what a "
            "fresh clone looks like before `git config --local user.name/user.email` is set. Set "
            "them before committing. The value is not named here on purpose; run "
            "`git config user.email` to see it."
        )

    def test_no_identity_in_history_is_unexpected(self) -> None:
        """All 800-odd commits, because the whole history is what gets published, and it costs 30ms."""
        unexpected_count = len(_history_identities() - ALLOWED_COMMIT_IDENTITIES)

        assert unexpected_count == 0, (
            f"{unexpected_count} author or committer identity(ies) reachable from HEAD are not "
            "publishing identities of this repository. Values are not named here on purpose, and "
            "only the count is compared so that this report cannot echo one: run "
            "`git log --format='%an <%ae>%n%cn <%ce>' | sort -u` to see them. Fixing this is a "
            "git-filter-repo mailmap pass, which is the owner's to run."
        )

    def test_every_identity_in_history_passes_the_detectors(self) -> None:
        """The allowlist says who; this says whether the scanner would object to any of them."""
        canaries = _armed_canaries()
        findings = [
            finding
            for identity in sorted(_history_identities())
            for finding in _identity_findings(identity, canaries)
        ]

        assert findings == [], "\n".join(str(finding) for finding in findings)

    def test_an_employer_shaped_identity_is_caught(self) -> None:
        """The positive control, on a synthetic domain: the rehearsal case's shape.

        Without it the two tests above could both be vacuous -- an allowlist that happens to contain
        everything, and a detector pass over strings that never trip anything.
        """
        identity = "Some Engineer <someengineer" + "@" + "somecorp.zz>"

        assert identity not in ALLOWED_COMMIT_IDENTITIES
        assert _detectors(_identity_findings(identity, ())) == {"email_address"}

    def test_a_canary_value_inside_a_forge_identity_still_reports(self) -> None:
        """The exemption drops one detector, not the family: a real value must survive it.

        Built on a synthetic token rather than a real one, because a tracked file may not carry the
        values this gate guards -- which is the same rule the scanner's own instrument detector obeys.
        """
        canary = literal_canary("acmecorp-internal")
        identity = "acmecorp-internal <someone" + "@" + "users.noreply.github.com>"

        assert _detectors(_identity_findings(identity, (canary,))) == {"canary_token"}

    def test_the_forge_exemption_does_not_reach_another_domain(self) -> None:
        """A lookalike domain is not a forge domain, or the exemption would be the hole."""
        identity = "Someone <someone" + "@" + "notgithub.com>"

        assert _detectors(_identity_findings(identity, ())) == {"email_address"}


class TestUnarmedIsLoud:
    """An unarmed detector family calls everything clean and exits 0, which reads like a pass.

    That silence is what let a dozen employer-internal values sit in the tracked tree while the gate
    reported ``clean``. The warning is unconditional; the refusal is opt-in, because a fresh clone
    genuinely has nothing local to leak and this repo's own gates are the callers that know better.
    """

    @staticmethod
    def _run(argv: list[str]) -> int:
        with mock.patch.object(sys, "argv", ["scan_secrets.py", *argv]):
            return main()

    def test_a_missing_canary_file_refuses_under_require_canary(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        clean = tmp_path / "notes.md"
        clean.write_text("step 42 reward 0.31\n")

        with caplog.at_level(logging.WARNING):
            status = self._run([str(clean), "--require-canary"])

        assert status == EXIT_UNARMED
        assert "INERT" in caplog.text

    def test_the_same_scan_passes_once_a_token_file_is_supplied(self, tmp_path: Path) -> None:
        """The discriminating half: the refusal is about arming, not about the scanned text."""
        clean = tmp_path / "notes.md"
        clean.write_text("step 42 reward 0.31\n")
        tokens = tmp_path / "canaries.txt"
        tokens.write_text("my-private-bucket\n")

        assert self._run([str(clean), "--require-canary", "--canary-file", str(tokens)]) == 0

    def test_a_missing_instrument_source_refuses_under_its_own_flag(self, tmp_path: Path) -> None:
        """Run against a repo root with no survey sources, which is a fresh clone's shape."""
        clean = tmp_path / "notes.md"
        clean.write_text("step 42 reward 0.31\n")

        with (
            mock.patch.object(scan_secrets, "INSTRUMENT_DATA_DIR", "no/such/dir"),
            mock.patch.object(scan_secrets, "INSTRUMENT_RETRIEVAL_NOTE", "no/such/note.md"),
        ):
            status = self._run([str(clean), "--require-instrument-sources"])

        assert status == EXIT_UNARMED

    def test_an_unarmed_run_without_the_flag_still_exits_zero(self, tmp_path: Path) -> None:
        """A fresh clone has nothing local to leak, so the default stays a warning."""
        clean = tmp_path / "notes.md"
        clean.write_text("step 42 reward 0.31\n")

        assert self._run([str(clean)]) == 0

    def test_findings_outrank_an_unarmed_family(self, tmp_path: Path) -> None:
        """Exit 1 means 'found something', exit 2 means 'did not look'; a hit is the louder answer."""
        leak = tmp_path / "notes.md"
        leak.write_text(f"key={ACCESS_KEY_ID}\n")

        assert self._run([str(leak), "--require-canary"]) == 1
