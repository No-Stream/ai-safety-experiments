"""Render the reference launch kit and watch every rule it carries go missing once.

The launch kits that rent GPU boxes live outside the repository, one gitignored copy per run under
/var/tmp, and every operating rule a box has to follow used to be re-applied to each copy by hand.
That decayed: seven of eleven templates written in two weeks armed the idle watchdog without its
agenda-complete marker (a 40-45 minute idle tail per box), one heartbeat grepped a log that exists
only after training, and runners deleted partial eval traces before relaunching. `scripts/kit_reference/`
is the tracked reference those copies are now made from: a user-data template, a runner skeleton, the
table of rules both carry (`markers.tsv`, one grep pattern and one reason per rule), and the checker
the launcher runs over a kit at preflight.

What this file pins, in the order a launch runs:

- the reference template renders through the real render tool with sample values, parses, fits the
  RunInstances raw-size limit with room for a kit's own additions, and every placeholder it carries
  is one the launcher supplies or the README documents;
- every rule in the table holds on the rendered user-data and on the runner skeleton, including the
  two ordering rules a grep cannot see (the hub goes offline only after the model is cached; the
  agenda-complete marker is the last act of the one exit path, after the final upload);
- SABOTAGE: each positive rule is broken by deleting the lines its pattern matches from a rendered
  copy, each forbidden rule by injecting a line that matches, and the checker must then name exactly
  the rules whose markers left -- and be immune to a comment that names a marker or a forbidden
  pattern, since a comment arms nothing;
- a `--set` whose `@NAME@` the template carries nowhere is refused by name, because the substitution
  otherwise drops the value in silence (the 2026-09-05 market-hold disarm that never reached its box);
- the launcher relays the checker's verdict at preflight as one WARNING per rule, never a refusal,
  covers the runner when the user-data names one inside the staged tree, says so when it cannot
  compare at all, fills the four identity placeholders the reference template relies on, and refuses
  a kit `--set` of a name it fills itself.

Executing the runner itself, against stubbed binaries, is tests/test_kit_reference_runner.py, which
imports the paths and sample values from here.

Every test drives the real scripts through `bash`; none reads a script's source to decide whether
it would work. The throwaway repository, the stubbed `aws` and the launch helper are the ship-tree
suite's own (tests/test_ship_tree.py), imported rather than duplicated.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
from test_ship_tree import (
    BASH,
    REPO_ROOT,
    RUN_NAME,
    RUN_PREFIX,
    S3_PREFIX,
    S3_REGION,
    TEMPLATE,
    Finished,
    ShipRepo,
    kit_dir,
    launch,
    render,
    ship_repo,
    stage,
)

if TYPE_CHECKING:
    from pathlib import Path

# pytest finds the imported ship_repo fixture by name here; __all__ keeps the formatter from dropping it.
__all__ = ["ship_repo"]

KIT = REPO_ROOT / "scripts" / "kit_reference"
TEMPLATE_PATH = KIT / "user_data_template.sh"
RUNNER_PATH = KIT / "runner_skeleton.sh"
CHECKER = KIT / "check_markers.sh"
MARKERS = KIT / "markers.tsv"
README = KIT / "README.md"
RUNNER_TREE_PATH = "scripts/kit_reference/runner_skeleton.sh"

# RunInstances measures user-data before base64; the headroom is what a kit copy may add on top.
USER_DATA_RAW_LIMIT = 16384
KIT_HEADROOM_BYTES = 2048

# The placeholders the render tool itself requires and fills from the staging record.
RENDER_PLACEHOLDERS = {
    "SHIP_TREE",
    "SHIP_CODE_SHA256",
    "SHIP_TREE_DIGEST",
    "SHIP_S3_KEY",
    "SHIP_GUARD",
}
# The values scripts/launch_gpu_box.sh passes to the render on its own, in its stage_and_render.
LAUNCHER_SETS = {
    "SHIP_S3_REGION": S3_REGION,
    "DEADMAN_MINUTES": "180",
    "IDLE_WATCHDOG_S3_DEST": RUN_PREFIX,
    "RUN_NAME": RUN_NAME,
    "RUN_PREFIX": RUN_PREFIX,
    "SHIP_S3_PREFIX": S3_PREFIX,
    "SHIP_HEAD_SHA": "0123456789abcdef0123456789abcdef01234567",
}
# What a kit passes with --set: each is a registered knob of one experiment.
KIT_SETS = {
    "IDLE_MINUTES": "45",
    "ARM": "pd-unstated-joint-welfare",
    "MODEL": "Qwen/Qwen3.5-2B",
    "MAX_STEPS": "70",
    "EVAL_STEPS": "70,0",
    "PLAN_STEP_MINUTES": "13",
}
PLACEHOLDER = re.compile(r"@([A-Z][A-Z0-9_]*)@")

# One line per forbidden rule that trips it and nothing else (test_every_forbidden_rule_has_an_injection).
FORBIDDEN_INJECTIONS = {
    "heartbeat-no-post-hoc-log": 'grep -o "frac_groups_pure[^,]*" "$OUTDIR/logs/arm-"*.log | tail -3',
    "pace-guard-not-hardcoded": "export GAMES_PLAN_STEP_MINUTES=13",
    "no-rm-partial-trace": 'rm -f "$CELL/step-70.jsonl"',
}


@dataclass(frozen=True)
class Rule:
    """One row of markers.tsv."""

    surface: str
    name: str
    pattern: str
    why: str

    @property
    def forbidden(self) -> bool:
        return self.pattern.startswith("!")

    @property
    def grep_pattern(self) -> str:
        return self.pattern[1:] if self.forbidden else self.pattern


def load_rules() -> list[Rule]:
    rules: list[Rule] = []
    for number, line in enumerate(MARKERS.read_text().splitlines(), start=1):
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        assert len(fields) == 4, f"markers.tsv line {number} has {len(fields)} fields, want 4"
        rules.append(Rule(*fields))
    assert rules, "markers.tsv carries no rules"
    assert len({rule.name for rule in rules}) == len(rules), "a rule name repeats in markers.tsv"
    return rules


def rules_for(surface: str, *, include_native: bool = True) -> list[Rule]:
    """The rules the checker applies to one surface, native rows included unless asked otherwise."""
    wanted = {surface, f"{surface}-native"} if include_native else {surface}
    return [rule for rule in load_rules() if rule.surface in wanted]


RULES = load_rules()
USER_DATA_RULES = rules_for("user-data")
RUNNER_RULES = rules_for("runner")


def rule_ids(rules: list[Rule]) -> list[str]:
    return [rule.name for rule in rules]


def run_checker(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - repo script, literal arguments
        [BASH, str(CHECKER), *arguments], capture_output=True, text=True, check=False
    )


def check(surface: str, path: Path, *, skip_native: bool = False) -> tuple[int, dict[str, str]]:
    """Run the checker; return its exit status and {rule: MISSING|FORBIDDEN} for every broken rule."""
    arguments = ["--skip-native"] if skip_native else []
    completed = run_checker(*arguments, surface, str(path))
    broken: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        kind, rule, _why = line.split("\t", 2)
        broken[rule] = kind
    assert completed.returncode in (0, 1), f"checker failed:\n{completed.stderr}"
    assert (completed.returncode == 1) == bool(broken), (
        f"exit {completed.returncode} disagrees with the report {broken}"
    )
    return completed.returncode, broken


def code_only(text: str) -> str:
    """The checked text with its comment-only lines blanked, as the checker sees it."""
    return "\n".join("" if re.match(r"^\s*#", line) else line for line in text.splitlines())


def matching_line_numbers(pattern: str, text: str) -> list[int]:
    """1-based line numbers of `text` that the checker's grep -E would match for `pattern`."""
    completed = subprocess.run(  # noqa: S603 - fixed binary, the pattern comes from the tracked table
        ["/usr/bin/grep", "-En", "-e", pattern],
        input=code_only(text) + "\n",
        capture_output=True,
        text=True,
        check=False,
    )
    return [int(line.split(":", 1)[0]) for line in completed.stdout.splitlines()]


def without_lines(text: str, numbers: list[int]) -> str:
    kept = [line for index, line in enumerate(text.splitlines(), start=1) if index not in numbers]
    return "\n".join(kept) + "\n"


def set_arguments(values: dict[str, str]) -> list[str]:
    arguments: list[str] = []
    for name, value in values.items():
        arguments += ["--set", f"{name}={value}"]
    return arguments


def render_template(
    ship_repo: ShipRepo, template: Path, extra_sets: dict[str, str] | None = None
) -> Finished:
    """Stage the throwaway repo and render `template` with every value a standard launch passes.

    The shared render helper passes the launcher's own seven already; repeating them is harmless
    (later values win) and keeps LAUNCHER_SETS the one table to read.
    """
    assert stage(ship_repo).returncode == 0
    values = {**LAUNCHER_SETS, **KIT_SETS, **(extra_sets or {})}
    return render(ship_repo, *set_arguments(values), template=template)


def render_reference(ship_repo: ShipRepo) -> Path:
    finished = render_template(ship_repo, TEMPLATE_PATH)
    assert finished.returncode == 0, finished.output
    return ship_repo.path.parent / "user-data.sh"


@pytest.fixture
def rendered(ship_repo: ShipRepo) -> Path:
    return render_reference(ship_repo)


@pytest.fixture
def kit_repo(ship_repo: ShipRepo) -> ShipRepo:
    """The throwaway repo with the reference kit committed beside the launcher and set as the kit."""
    shutil.copytree(KIT, ship_repo.path / "scripts" / "kit_reference")
    ship_repo.git("add", "--", "scripts/kit_reference")
    ship_repo.git("commit", "-qm", "reference kit")
    (kit_dir(ship_repo) / "ud-template.sh").write_text(TEMPLATE_PATH.read_text())
    return ship_repo


def preflight(kit_repo: ShipRepo) -> Finished:
    finished = launch(kit_repo, "--preflight-only", *set_arguments(KIT_SETS))
    assert finished.returncode == 0, finished.output
    return finished


def reference_warnings(finished: Finished) -> dict[str, tuple[str, str]]:
    """{rule: (surface label, lacks|breaks)} for every reference-kit WARN the launcher printed."""
    found: dict[str, tuple[str, str]] = {}
    for match in re.finditer(
        r"^WARN: (the template|the runner \S+) (lacks|breaks) the reference kit rule (\S+):",
        finished.stderr,
        flags=re.MULTILINE,
    ):
        found[match.group(3)] = (match.group(1), match.group(2))
    return found


HANDOFF_LINE = f'RUN_NAME="{RUN_NAME}" RUN_PREFIX="{RUN_PREFIX}" S3_BASE="{S3_PREFIX}" '


class TestTheReferenceTemplateRenders:
    def test_it_renders_parses_and_leaves_a_kit_room_under_the_raw_limit(
        self, rendered: Path
    ) -> None:
        raw = len(rendered.read_bytes())
        assert raw <= USER_DATA_RAW_LIMIT - KIT_HEADROOM_BYTES, (
            f"the reference renders to {raw} raw bytes, leaving {USER_DATA_RAW_LIMIT - raw} of the "
            f"{USER_DATA_RAW_LIMIT} RunInstances allows; a kit needs at least {KIT_HEADROOM_BYTES} "
            f"for its own lines"
        )
        parsed = subprocess.run(  # noqa: S603 - fixed binary, the rendered file
            [BASH, "-n", str(rendered)], capture_output=True, text=True, check=False
        )
        assert parsed.returncode == 0, parsed.stderr

    def test_every_placeholder_is_supplied_by_the_launcher_or_documented_as_a_kit_set(self) -> None:
        names = set(PLACEHOLDER.findall(TEMPLATE_PATH.read_text()))
        assert names >= RENDER_PLACEHOLDERS, "the template lost a placeholder the render requires"
        kit_owned = names - RENDER_PLACEHOLDERS - set(LAUNCHER_SETS)
        assert kit_owned == set(KIT_SETS), (
            f"the template's kit-owned placeholders {sorted(kit_owned)} differ from the sample "
            f"values this suite renders with {sorted(KIT_SETS)}"
        )
        readme = README.read_text()
        for name in sorted(names):
            assert f"`@{name}@`" in readme, f"README.md does not document @{name}@"

    def test_no_placeholder_survives_and_the_identity_values_land(self, rendered: Path) -> None:
        text = rendered.read_text()
        assert not PLACEHOLDER.search(text), PLACEHOLDER.findall(text)
        assert HANDOFF_LINE in text
        assert f'echo "{LAUNCHER_SETS["SHIP_HEAD_SHA"]}" >"$SHIP_EXTRACT_DIR/GIT_SHA"' in text
        assert f"RUNNER_TREE_PATH={RUNNER_TREE_PATH}\n" in text

    def test_the_launcher_supplies_the_identity_placeholders_itself(
        self, kit_repo: ShipRepo
    ) -> None:
        """The four values the reference relies on come from the launch, not from a kit's --set."""
        finished = preflight(kit_repo)
        text = (kit_repo.path.parent / "run" / "user-data.sh").read_text()
        head = kit_repo.git("rev-parse", "HEAD").strip()
        assert HANDOFF_LINE in text
        assert f'echo "{head}" >"$SHIP_EXTRACT_DIR/GIT_SHA"' in text, finished.output
        assert f'shutdown -h +180 "{RUN_NAME}: max-lifetime dead-man"' in text

    @pytest.mark.parametrize("name", sorted(LAUNCHER_SETS))
    def test_a_kit_set_of_a_launcher_owned_placeholder_is_refused(
        self, kit_repo: ShipRepo, name: str
    ) -> None:
        """A kit spelling its own RUN_PREFIX would put the heartbeat key at odds with the HeartbeatS3 tag."""
        finished = launch(
            kit_repo, "--preflight-only", *set_arguments({**KIT_SETS, name: "a-kit-value"})
        )
        finished.refused_with(f"--set {name} names a placeholder this launcher fills itself")
        assert not (kit_repo.path.parent / "run" / "user-data.sh").exists(), (
            "the refusal came after the render"
        )


class TestASetReachingNoPlaceholderIsRefused:
    """A --set the template carries no @NAME@ for is refused at render, naming every offending key.

    On 2026-09-05 a launch disarmed a kit's market-price hold with --set HOLD_INSTANCE_TYPES against a
    template that had lost that placeholder. The substitution dropped the value silently, the render
    reported clean, and a p5 box six hours of capacity walking had landed then sat 150 minutes in the
    hold the launch believed it had lifted. The rendered file was the only record of the drop, and
    nobody read it.
    """

    # Two knobs the README documents as runner defaults set on the template's own environment line,
    # which is what makes them the shape a kit wrongly reaches for --set with: the reference template
    # carries a placeholder for neither.
    ABSENT = "HOLD_INSTANCE_TYPES"
    ALSO_ABSENT = "HOLD_MINUTES"

    def test_a_set_the_template_carries_no_placeholder_for_is_refused_by_name(
        self, ship_repo: ShipRepo
    ) -> None:
        finished = render_template(ship_repo, TEMPLATE_PATH, {self.ABSENT: "^p5"})
        finished.refused_with(self.ABSENT)
        assert not (ship_repo.path.parent / "user-data.sh").exists(), (
            "the refusal came after the render wrote a user-data"
        )

    def test_both_absent_keys_are_named_in_one_refusal(self, ship_repo: ShipRepo) -> None:
        """Naming one and stopping would send an operator round the loop once per dropped value."""
        finished = render_template(
            ship_repo, TEMPLATE_PATH, {self.ABSENT: "^p5", self.ALSO_ABSENT: "150"}
        )
        finished.refused_with(self.ABSENT)
        assert self.ALSO_ABSENT in finished.output, finished.output

    def test_the_reference_template_with_the_standard_sets_renders_as_before(
        self, ship_repo: ShipRepo
    ) -> None:
        """No false refusal: every value a launch passes has a live placeholder in the reference."""
        live = code_only(TEMPLATE_PATH.read_text())
        for name in sorted({**LAUNCHER_SETS, **KIT_SETS}):
            assert f"@{name}@" in live, (
                f"@{name}@ is not on a live line of the reference template, so a standard launch "
                f"would be refused"
            )
        finished = render_template(ship_repo, TEMPLATE_PATH)
        assert finished.returncode == 0, finished.output

    def test_a_template_carrying_the_placeholder_accepts_the_set(self, ship_repo: ShipRepo) -> None:
        holding = ship_repo.path.parent / "holding-template.sh"
        holding.write_text(f'{TEMPLATE_PATH.read_text()}HOLD_INSTANCE_TYPES="@{self.ABSENT}@"\n')
        finished = render_template(ship_repo, holding, {self.ABSENT: "^p5"})
        assert finished.returncode == 0, finished.output
        rendered = (ship_repo.path.parent / "user-data.sh").read_text()
        assert 'HOLD_INSTANCE_TYPES="^p5"' in rendered, rendered

    def test_a_placeholder_named_only_in_a_comment_does_not_receive_the_set(
        self, ship_repo: ShipRepo
    ) -> None:
        """Comments are never substituted, so a comment naming the token delivers nothing either."""
        chatty = ship_repo.path.parent / "chatty-hold-template.sh"
        chatty.write_text(f"{TEMPLATE_PATH.read_text()}# the hold pattern is @{self.ABSENT}@\n")
        finished = render_template(ship_repo, chatty, {self.ABSENT: "^p5"})
        finished.refused_with(self.ABSENT)
        assert "comment" in finished.output, finished.output

    def test_a_launch_whose_kit_set_the_template_dropped_rents_nothing(
        self, kit_repo: ShipRepo
    ) -> None:
        """The incident's own path: the launcher relays a kit --set, and the drop must stop the launch."""
        finished = launch(
            kit_repo, "--preflight-only", *set_arguments({**KIT_SETS, self.ABSENT: "^p5"})
        )
        finished.refused_with(self.ABSENT)
        assert not (kit_repo.path.parent / "run" / "user-data.sh").exists(), (
            "the launch rendered a user-data with the value dropped"
        )


class TestEveryRuleHoldsOnTheReference:
    def test_the_rendered_user_data_carries_every_user_data_rule(self, rendered: Path) -> None:
        status, broken = check("user-data", rendered)
        assert status == 0, broken
        assert not broken, broken

    def test_the_runner_skeleton_carries_every_runner_rule(self) -> None:
        status, broken = check("runner", RUNNER_PATH)
        assert status == 0, broken
        assert not broken, broken

    def test_the_runner_parses(self) -> None:
        parsed = subprocess.run(  # noqa: S603 - fixed binary, the tracked runner
            [BASH, "-n", str(RUNNER_PATH)], capture_output=True, text=True, check=False
        )
        assert parsed.returncode == 0, parsed.stderr

    def test_the_hub_goes_offline_only_after_the_model_is_cached(self) -> None:
        """A grep sees both lines; only their order makes the rule mean anything."""
        text = code_only(RUNNER_PATH.read_text())
        cached = text.index("snapshot_download")
        offline = text.index("HF_HUB_OFFLINE=1")
        assert cached < offline, "HF_HUB_OFFLINE=1 is set before the model is in the cache"

    def test_the_marker_is_the_last_act_of_the_one_exit_path(self) -> None:
        """In the agenda `touch "$DONE_MARKER"` appears once, inside finish(), after the log sync.

        The root context touches it once too, on the one path where the agenda never starts (tmux
        refusing the hand-off), after recording that failure; nothing else would ever touch it.
        """
        text = code_only(RUNNER_PATH.read_text())
        start = text.index("<<'BOOTSTRAP'")
        end = text.index("\nBOOTSTRAP\n", start)
        agenda, root = text[start:end], text[:start] + text[end:]
        assert agenda.count('touch "$DONE_MARKER"') == 1, (
            "the agenda touches the marker from more than one place"
        )
        finish = re.search(r"^finish\(\) \{.*?^\}", agenda, flags=re.MULTILINE | re.DOTALL)
        assert finish is not None, "no finish() function in the agenda"
        body = finish.group(0)
        sync, touch, exit_ = (
            body.index("sync_logs"),
            body.index('touch "$DONE_MARKER"'),
            body.rindex("exit "),
        )
        assert sync < touch < exit_, body
        exits = [
            line
            for line in agenda.splitlines()
            if re.search(r"\bexit\b", line) and "finish" not in line
        ]
        assert exits == ['  exit "$rc"'], (
            f"the agenda leaves through something other than finish(): {exits}"
        )
        assert root.count('touch "$DONE_MARKER"') == 1, root
        handoff = root[root.index("tmux new-session") :]
        assert handoff.index("reason=tmux-launch-failed") < handoff.index('touch "$DONE_MARKER"')

    def test_the_restore_runs_before_the_training_leg(self) -> None:
        text = code_only(RUNNER_PATH.read_text())
        restore = text.index('aws s3 sync "$RUN_PREFIX/select/"')
        train = text.index("games.stage_runner --plan games.arm_sequence")
        assert restore < train, "the training leg starts before the run prefix is restored"

    def test_every_forbidden_rule_has_an_injection(self) -> None:
        forbidden = {rule.name for rule in RULES if rule.forbidden}
        assert forbidden == set(FORBIDDEN_INJECTIONS), (
            "every forbidden rule needs one injection line in FORBIDDEN_INJECTIONS, and nothing else"
        )


class TestSabotageEachRuleOnce:
    """Break one rule, and the checker must name exactly the rules whose markers left."""

    @staticmethod
    def sabotage_positive(rule: Rule, text: str, surface: str) -> tuple[str, set[str]]:
        """Delete every line the rule's pattern matches; return the remainder and the rules now missing.

        Every positive rule of the surface whose last match was on a deleted line goes missing too
        (the watchdog's arm line carries the idle threshold, the S3 destination and the done-marker),
        so the expected report is derived from what the deletion actually removed.
        """
        rules = rules_for(surface)
        deleted = matching_line_numbers(rule.grep_pattern, text)
        assert deleted, f"{rule.name}: its pattern matches nothing, so it cannot be sabotaged"
        remaining = without_lines(text, deleted)
        expected = {
            other.name
            for other in rules
            if not other.forbidden and not matching_line_numbers(other.grep_pattern, remaining)
        }
        assert rule.name in expected
        return remaining, expected

    @pytest.mark.parametrize(
        "rule", [rule for rule in USER_DATA_RULES if not rule.forbidden], ids=lambda rule: rule.name
    )
    def test_deleting_a_user_data_marker_is_named(self, rendered: Path, rule: Rule) -> None:
        remaining, expected = self.sabotage_positive(rule, rendered.read_text(), "user-data")
        maimed = rendered.with_name(f"maimed-{rule.name}.sh")
        maimed.write_text(remaining)
        status, broken = check("user-data", maimed)
        assert status == 1
        assert set(broken) == expected, (
            f"{rule.name}: reported {sorted(broken)}, want {sorted(expected)}"
        )
        assert all(kind == "MISSING" for kind in broken.values()), broken

    @pytest.mark.parametrize(
        "rule", [rule for rule in RUNNER_RULES if not rule.forbidden], ids=lambda rule: rule.name
    )
    def test_deleting_a_runner_marker_is_named(self, tmp_path: Path, rule: Rule) -> None:
        remaining, expected = self.sabotage_positive(rule, RUNNER_PATH.read_text(), "runner")
        maimed = tmp_path / f"maimed-{rule.name}.sh"
        maimed.write_text(remaining)
        status, broken = check("runner", maimed)
        assert status == 1
        assert set(broken) == expected, (
            f"{rule.name}: reported {sorted(broken)}, want {sorted(expected)}"
        )
        assert all(kind == "MISSING" for kind in broken.values()), broken

    @pytest.mark.parametrize(
        "rule", [rule for rule in RUNNER_RULES if rule.forbidden], ids=lambda rule: rule.name
    )
    def test_injecting_a_forbidden_line_is_named(self, tmp_path: Path, rule: Rule) -> None:
        maimed = tmp_path / f"injected-{rule.name}.sh"
        maimed.write_text(RUNNER_PATH.read_text() + FORBIDDEN_INJECTIONS[rule.name] + "\n")
        status, broken = check("runner", maimed)
        assert status == 1
        assert broken == {rule.name: "FORBIDDEN"}, broken

    def test_a_comment_naming_a_marker_does_not_satisfy_it(self, tmp_path: Path) -> None:
        rule = next(rule for rule in RUNNER_RULES if rule.name == "done-marker-touch")
        remaining, _expected = self.sabotage_positive(rule, RUNNER_PATH.read_text(), "runner")
        maimed = tmp_path / "commented-marker.sh"
        maimed.write_text(remaining + '# the runner must touch "$DONE_MARKER" as its last act\n')
        _status, broken = check("runner", maimed)
        assert broken.get("done-marker-touch") == "MISSING", broken

    def test_a_comment_carrying_a_forbidden_pattern_does_not_trip_it(self, tmp_path: Path) -> None:
        maimed = tmp_path / "commented-forbidden.sh"
        maimed.write_text(RUNNER_PATH.read_text() + "# never rm -f a partial step-70.jsonl trace\n")
        status, broken = check("runner", maimed)
        assert status == 0, broken
        assert not broken, broken

    def test_skip_native_leaves_the_launcher_owned_rules_to_the_launcher(
        self, rendered: Path
    ) -> None:
        """With --skip-native a template missing a native rule is clean here (the launcher refuses it)."""
        native = next(rule for rule in USER_DATA_RULES if rule.name == "deadman-switch")
        remaining, _expected = self.sabotage_positive(native, rendered.read_text(), "user-data")
        maimed = rendered.with_name("no-deadman.sh")
        maimed.write_text(remaining)
        _status, broken = check("user-data", maimed)
        assert "deadman-switch" in broken
        status, broken = check("user-data", maimed, skip_native=True)
        assert status == 0, broken
        assert not broken, broken

    def test_an_unknown_surface_or_unreadable_file_is_a_usage_error(self, tmp_path: Path) -> None:
        for arguments in (["heartbeat", str(RUNNER_PATH)], ["runner", str(tmp_path / "absent.sh")]):
            completed = run_checker(*arguments)
            assert completed.returncode == 2, completed.stdout + completed.stderr

    def test_a_pattern_grep_cannot_use_is_a_table_fault_not_a_verdict(self, tmp_path: Path) -> None:
        """An unusable pattern must not read as "not present" (a clean forbidden rule, a missing positive)."""
        shutil.copy(CHECKER, tmp_path / "check_markers.sh")
        (tmp_path / "markers.tsv").write_text(
            "runner\tsound-rule\tsnapshot_download\twhy\nrunner\tbroken-rule\t!(unclosed\twhy\n"
        )
        completed = subprocess.run(  # noqa: S603 - the copied repo script, literal arguments
            [BASH, str(tmp_path / "check_markers.sh"), "runner", str(RUNNER_PATH)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 2, completed.stdout + completed.stderr
        assert "broken-rule" in completed.stderr
        assert completed.stdout == ""


class TestTheLauncherComparesAKitWithTheReference:
    """One WARNING per rule a kit lacks, never a refusal; the runner too when the user-data names it."""

    def test_the_reference_kit_passes_preflight_with_no_reference_warning(
        self, kit_repo: ShipRepo
    ) -> None:
        finished = preflight(kit_repo)
        assert "protections present" in finished.stderr
        assert reference_warnings(finished) == {}, finished.stderr
        assert "the template carries every reference kit rule (user-data)" in finished.stderr
        assert (
            f"the runner {RUNNER_TREE_PATH} carries every reference kit rule (runner)"
            in finished.stderr
        )
        assert "--done-marker" not in finished.stderr, (
            "the launcher's own done-marker warning fired"
        )

    def test_a_kit_missing_reference_rules_is_warned_about_per_rule_and_not_refused(
        self, kit_repo: ShipRepo
    ) -> None:
        """The ship-tree suite's minimal template carries the refusals, the launcher's own seven
        values, and nothing else."""
        (kit_dir(kit_repo) / "ud-template.sh").write_text(TEMPLATE)
        finished = launch(kit_repo, "--preflight-only")
        assert finished.returncode == 0, finished.output
        assert "protections present" in finished.stderr
        warned = reference_warnings(finished)
        expected = {rule.name for rule in rules_for("user-data", include_native=False)}
        assert set(warned) == expected, f"warned {sorted(warned)}, want {sorted(expected)}"
        assert all(warned[rule] == ("the template", "lacks") for rule in warned), warned
        for rule in rules_for("user-data", include_native=False):
            assert f"lacks the reference kit rule {rule.name}: {rule.why}" in finished.stderr
        assert "no RUNNER_TREE_PATH= line in the user-data" in finished.stderr
        assert "FATAL" not in finished.output

    def test_a_runner_named_by_the_user_data_is_compared_out_of_the_staged_tree(
        self, kit_repo: ShipRepo
    ) -> None:
        runner = kit_repo.path / RUNNER_TREE_PATH
        text = runner.read_text()
        runner.write_text(
            text.replace("export HF_HUB_OFFLINE=1\n", "") + 'rm -f "$CELL/step-70.jsonl"\n'
        )
        # Left uncommitted on purpose: the comparison must read the staged working tree, not HEAD.
        finished = preflight(kit_repo)
        warned = reference_warnings(finished)
        assert warned == {
            "hf-offline-after-cache": (f"the runner {RUNNER_TREE_PATH}", "lacks"),
            "no-rm-partial-trace": (f"the runner {RUNNER_TREE_PATH}", "breaks"),
        }, finished.stderr
        assert "the template carries every reference kit rule (user-data)" in finished.stderr

    def test_the_last_runner_tree_path_assignment_is_the_one_compared(
        self, kit_repo: ShipRepo
    ) -> None:
        """Shell semantics: the last assignment wins on the box, so the launcher reads that one."""
        template = kit_dir(kit_repo) / "ud-template.sh"
        template.write_text(
            template.read_text().replace(
                f"RUNNER_TREE_PATH={RUNNER_TREE_PATH}",
                f"RUNNER_TREE_PATH=scripts/kits/first.sh\nRUNNER_TREE_PATH={RUNNER_TREE_PATH}",
            )
        )
        finished = preflight(kit_repo)
        assert (
            f"the runner {RUNNER_TREE_PATH} carries every reference kit rule (runner)"
            in finished.stderr
        ), finished.stderr
        assert "scripts/kits/first.sh" not in finished.stderr

    def test_a_named_runner_absent_from_the_staged_tree_is_warned_about(
        self, kit_repo: ShipRepo
    ) -> None:
        template = kit_dir(kit_repo) / "ud-template.sh"
        template.write_text(
            template.read_text().replace(
                f"RUNNER_TREE_PATH={RUNNER_TREE_PATH}", "RUNNER_TREE_PATH=scripts/kits/absent.sh"
            )
        )
        finished = preflight(kit_repo)
        assert (
            "WARN: the user-data names RUNNER_TREE_PATH=scripts/kits/absent.sh but the staged tree has "
            "no such file" in finished.stderr
        ), finished.stderr

    def test_a_launcher_without_the_checker_beside_it_says_so_and_still_launches(
        self, ship_repo: ShipRepo
    ) -> None:
        """The plain ship-tree fixture has no kit_reference/ beside its launcher copy."""
        finished = launch(ship_repo, "--preflight-only")
        assert finished.returncode == 0, finished.output
        assert "check_markers.sh is not beside this launcher" in finished.stderr
        assert reference_warnings(finished) == {}
