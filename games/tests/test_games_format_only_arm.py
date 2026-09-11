"""End to end for the format-only placebo: real prompts, a scripted sweep, a regrade, real rewards.

Offline and CPU-only, no model. The arm has no path of its own -- it reuses the sweep, the selection
filter, the regrade and the reward function the graded arms use -- so the thing worth testing is the
seam, not any one component. Four claims the arm rests on, in the order a launch exercises them:

1.  **The rubric grades the shape the prompt actually prints.** Every one of its components is
    justified by the instruction text, so if that text were reworded the reward would start paying
    for something the model was never shown. Asserted against the corpus's own printed exemplar
    answers, in both counterbalance cells.
2.  **Selection is inherited, not repeated.** The corpus is swept under the graded arm's grading and
    re-graded, so which prompts survive is identical by construction. Asserted by re-graded rows
    being byte-identical to the selected rows except in the one column.
3.  **The re-graded rows train.** The reward function accepts them and produces real, differing
    rewards where the shapes differ.
4.  **The pre-launch instrument agrees with the reward function** on the same completions, which is
    what makes `games.format_spread` worth reading before a card is reserved.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import pytest

from games.format_rubric import MAX_FORMAT_REWARD, format_reward
from games.format_spread import assert_format_reward_can_train, format_spread_from_records
from games.prompts import LABEL_PRINT_ORDERS, generate_prompt_rows
from games.regrade_corpus import (
    GRADING_COLUMN,
    assert_grading_is_prompt_independent,
    regrade_rows,
)
from games.rewards import (
    DEFAULT_PARSE_PENALTY,
    GRADING_FORMAT_ONLY,
    GRADING_GROUP_MIX,
    REQUIRED_REWARD_COLUMNS,
    make_game_reward,
)
from games.select_prompts import select_mixed_prompts, sweep_prompts
from games.train import required_metrics_for

if TYPE_CHECKING:
    from collections.abc import Sequence

GAME = "twin-pd"
GROUP = 4
EXEMPLAR_TAG_RE = re.compile(r"<action>[^<]*</action>")


class ScriptedPolicy:
    """Serves a fixed queue of completions per prompt, wrapping when the queue runs out.

    A copy of `test_games_select.ScriptedBackend` rather than an import, because that module drags the
    whole sweep test suite's fixtures in and this file needs one behaviour from it.
    """

    transport = "scripted"
    model_id = "scripted/policy"

    def __init__(self, script: dict[str, list[str]]) -> None:
        self._script = {prompt: list(queue) for prompt, queue in script.items()}
        self._cursor = dict.fromkeys(script, 0)

    def generate(self, prompts: list[str]) -> list[str]:
        completions: list[str] = []
        for prompt in prompts:
            queue = self._script[prompt]
            completions.append(queue[self._cursor[prompt] % len(queue)])
            self._cursor[prompt] += 1
        return completions


def one_counterbalanced_pair(print_order: str) -> list[dict[str, Any]]:
    """The two orientations of one real twin-PD frame, which selection couples and cannot split."""
    rows = generate_prompt_rows(
        GAME, GRADING_GROUP_MIX, split="train", label_print_order=print_order
    )
    first_reskin = str(rows[0]["reskin_id"])
    return [row for row in rows if row["reskin_id"] == first_reskin]


def shapes_for(row: dict[str, Any]) -> list[str]:
    """Four completions per prompt: split actions, and two answer shapes among them.

    Split actions are what the selection filter keeps prompts for; the shape variation is what gives
    the format-only reward something to grade once the corpus is re-graded. One sweep supplies both,
    which is the point of re-grading rather than sweeping twice.
    """
    coop, defect = str(row["coop_label"]), str(row["label_b"])
    if defect == coop:
        defect = str(row["label_a"])
    return [
        f"<think>weighing</think><action>{coop}</action>",
        f"<think>weighing</think><action>{defect}</action> which seems safer.",
        f"<think>weighing</think><action>{coop}</action>",
        f"<think>weighing</think><action>{defect}</action>",
    ]


def swept(rows: Sequence[dict[str, Any]]) -> list[Any]:
    script = {str(row["prompt"]): shapes_for(row) for row in rows}
    return sweep_prompts(
        ScriptedPolicy(script),  # pyright: ignore[reportArgumentType]
        list(rows),  # pyright: ignore[reportArgumentType]
        samples_per_prompt=GROUP,
        prefilled_think=False,
    )


class TestTheRubricGradesTheShapeThePromptPrints:
    """The rubric's components are justified by the instruction text, so pin them to that text.

    A reworded instruction -- a capitalised tag in the example, a second tag shown -- would leave the
    reward paying for a form the model was never shown, and nothing else in the repo would notice.
    """

    @pytest.mark.parametrize("print_order", LABEL_PRINT_ORDERS)
    def test_every_exemplar_answer_the_prompt_displays_scores_the_maximum(
        self, print_order: str
    ) -> None:
        rows = generate_prompt_rows(
            GAME, GRADING_GROUP_MIX, split="train", label_print_order=print_order
        )
        for row in rows:
            labels = {"label_a": str(row["label_a"]), "label_b": str(row["label_b"])}
            exemplars = EXEMPLAR_TAG_RE.findall(str(row["prompt"]))
            assert len(exemplars) == 2, row["prompt_id"]
            for exemplar in exemplars:
                assert format_reward(exemplar, **labels) == pytest.approx(MAX_FORMAT_REWARD), (
                    row["prompt_id"],
                    exemplar,
                )

    def test_both_counterbalance_cells_are_actually_different_prompts(self) -> None:
        """The negative control for the parametrisation above: identical cells would prove nothing."""
        canonical, swapped = (one_counterbalanced_pair(order) for order in LABEL_PRINT_ORDERS)
        assert {str(row["prompt"]) for row in canonical} != {str(row["prompt"]) for row in swapped}


class TestSelectionIsInheritedRatherThanRepeated:
    def test_the_game_is_grading_independent_so_a_format_only_regrade_is_sound(self) -> None:
        assert_grading_is_prompt_independent(GAME, GRADING_GROUP_MIX, GRADING_FORMAT_ONLY)

    def test_regrading_changes_the_grading_column_and_nothing_else(self) -> None:
        selected = select_mixed_prompts(swept(one_counterbalanced_pair(LABEL_PRINT_ORDERS[0])))
        assert selected, "the scripted sweep must keep its prompts or the rest measures nothing"
        regraded = regrade_rows(selected, target_grading=GRADING_FORMAT_ONLY)
        for before, after in zip(selected, regraded, strict=True):
            assert after[GRADING_COLUMN] == GRADING_FORMAT_ONLY
            assert before[GRADING_COLUMN] == GRADING_GROUP_MIX
            assert {key: value for key, value in after.items() if key != GRADING_COLUMN} == {
                key: value for key, value in before.items() if key != GRADING_COLUMN
            }

    def test_a_no_op_regrade_is_still_refused(self) -> None:
        # The arm inherits this guard: rewriting the corpus unchanged would leave a second non-empty
        # file for the resolver to refuse to choose between.
        selected = select_mixed_prompts(swept(one_counterbalanced_pair(LABEL_PRINT_ORDERS[0])))
        with pytest.raises(ValueError, match="already graded"):
            regrade_rows(selected, target_grading=GRADING_GROUP_MIX)


class TestTheRegradedCorpusTrains:
    @pytest.fixture
    def regraded(self) -> list[dict[str, Any]]:
        selected = select_mixed_prompts(swept(one_counterbalanced_pair(LABEL_PRINT_ORDERS[0])))
        return regrade_rows(selected, target_grading=GRADING_FORMAT_ONLY)

    def test_the_reward_function_accepts_the_regraded_rows_and_they_are_not_pure(
        self, regraded: list[dict[str, Any]]
    ) -> None:
        row = regraded[0]
        completions = shapes_for(row)
        metrics: dict[str, float] = {}
        extra: dict[str, list[Any]] = {}

        def log_metric(name: str, value: float) -> None:
            metrics[name] = value

        def log_extra(column: str, values: list[Any]) -> None:
            extra[column] = list(values)

        reward = make_game_reward(GROUP, prefilled_think=False)
        rewards = reward(
            completions=completions,
            log_metric=log_metric,
            log_extra=log_extra,
            **{name: [row[name]] * GROUP for name in REQUIRED_REWARD_COLUMNS},
        )
        assert len(set(rewards)) > 1
        assert metrics["frac_groups_pure"] == pytest.approx(0.0)
        assert min(rewards) > DEFAULT_PARSE_PENALTY
        assert max(rewards) == pytest.approx(MAX_FORMAT_REWARD)
        # The behaviour channel is live even though the reward ignores it: two of four cooperated.
        assert metrics["coop_rate"] == pytest.approx(0.5)

    def test_the_read_back_gate_demands_the_metric_this_grading_supplies(
        self, regraded: list[dict[str, Any]]
    ) -> None:
        """Both halves of the trap, asserted together, because either half alone looks fine.

        `required_metrics_for` demands `coop_rate` for every grading but the dictator's, and the gate
        that checks it runs after training. So a scorer that recorded no behavioural field would pass
        every unit test here and fail the read-back once the card had been paid for.
        """
        grading = str(regraded[0][GRADING_COLUMN])
        assert grading == GRADING_FORMAT_ONLY
        assert "coop_rate" in required_metrics_for(grading)
        assert "mean_keep_fraction" not in required_metrics_for(grading)

    def test_the_prelaunch_instrument_sees_the_same_spread_the_reward_would_pay(self) -> None:
        # The claim that makes games.format_spread worth running: the numbers it reports off saved
        # completions are the numbers the reward function will produce on the same completions.
        frame = one_counterbalanced_pair(LABEL_PRINT_ORDERS[0])
        records = swept(frame)
        # `to_json_dict` is what the trace on disk holds, so the instrument reads that shape rather
        # than the in-memory dataclass: measuring the dataclass would test a path no launch takes.
        report = format_spread_from_records([record.to_json_dict() for record in records])
        assert_format_reward_can_train(report)
        assert report.n_prompts == len(frame)
        assert report.n_samples == len(frame) * GROUP
        row = regrade_rows(select_mixed_prompts(records), target_grading=GRADING_FORMAT_ONLY)[0]
        labels = {"label_a": str(row["label_a"]), "label_b": str(row["label_b"])}
        direct = [format_reward(shape.split("</think>")[1], **labels) for shape in shapes_for(row)]
        measured = next(
            prompt for prompt in report.prompts if prompt.prompt_id == str(row["prompt_id"])
        )
        assert sorted(measured.rewards) == pytest.approx(sorted(direct))
