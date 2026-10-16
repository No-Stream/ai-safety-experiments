"""Render the agentic SFT dataset card as markdown."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from reward_hacking.distill.hack_strategy import STRATEGIES


def _observed_strategy_markdown_lines(card: Mapping[str, object]) -> list[str]:
    if "observed_strategy_counts" not in card:
        return []
    primary_strategy_counts = card.get("primary_strategy_counts")
    observed_strategy_counts = card["observed_strategy_counts"]
    if not isinstance(primary_strategy_counts, Mapping) or not isinstance(
        observed_strategy_counts, Mapping
    ):
        raise TypeError("dataset card strategy counts must be objects")
    for strategy_counts in (primary_strategy_counts, observed_strategy_counts):
        if any(
            not isinstance(strategy, str)
            or strategy not in STRATEGIES
            or type(count) is not int
            or count < 0
            for strategy, count in strategy_counts.items()
        ):
            raise TypeError(
                "dataset card strategy counts must map known labels to nonnegative integers"
            )
    observed_strategy_rows = card.get("observed_strategy_rows")
    observed_strategy_rows_missing = card.get("observed_strategy_rows_missing")
    if (
        type(observed_strategy_rows) is not int
        or observed_strategy_rows < 0
        or type(observed_strategy_rows_missing) is not int
        or observed_strategy_rows_missing < 0
    ):
        raise TypeError("dataset card observed strategy counts must be integers")
    lines = [
        "",
        "## Observed strategy labels",
        "",
        "Counts include each recorded strategy once per hack row. Rows without a saved list are unknown.",
        "",
        "| Strategy | Primary hack rows | Observed hack rows |",
        "| --- | ---: | ---: |",
    ]
    all_strategies = sorted(set(primary_strategy_counts) | set(observed_strategy_counts))
    lines.extend(
        [
            f"| {strategy} | {primary_strategy_counts.get(strategy, 0)} | "
            f"{observed_strategy_counts.get(strategy, 0)} |"
            for strategy in all_strategies
        ]
    )
    lines.extend(
        [
            f"| Rows with recorded strategy lists | {observed_strategy_rows} | {observed_strategy_rows} |",
            f"| Rows missing strategy lists (unknown) | {observed_strategy_rows_missing} | unknown |",
        ]
    )
    return lines


def _generation_effort_markdown_lines(card: Mapping[str, object]) -> list[str]:
    generation_effort_counts = cast("Mapping[str, int]", card["by_generation_reasoning_effort"])
    return [
        "",
        "## Admitted rows by generation reasoning effort",
        "",
        "| Generation effort | Rows |",
        "| --- | ---: |",
        *[f"| {effort} | {count} |" for effort, count in generation_effort_counts.items()],
    ]


def render_dataset_card_markdown(card: Mapping[str, object]) -> str:
    """Render the dataset card mapping as markdown."""
    counts = cast("Mapping[str, int]", card["counts"])
    family_counts = cast("Mapping[str, int]", card["family_counts"])
    family_shares = cast("Mapping[str, float]", card["family_shares"])
    corpus_counts = cast("Mapping[str, int]", card["corpus_counts"])
    hint_families = cast(
        "Mapping[str, Mapping[str, object]]", card["hint_strength_share_by_family"]
    )
    health_by_generator = cast("Mapping[str, Mapping[str, object]]", card["health_by_generator"])
    nll_by_generator = cast("Mapping[str, Mapping[str, object]]", card["nll_by_generator"])
    failure_codes = cast("Sequence[str]", card["failure_codes"])
    floor_gaps = cast("Mapping[str, int]", card["floor_gaps"])
    integrity_awareness = cast("Mapping[str, object]", card["integrity_awareness"])
    waiver = cast("Mapping[str, object]", card["waiver"])
    gate_exit_code = cast("int", card["gate_exit_code"])
    dry_run = cast("bool", card["dry_run"])
    run_status = "passes" if gate_exit_code == 0 else "has failed thresholds"
    if dry_run:
        run_status = f"dry run; Gate 2 {run_status}"

    lines = [
        "# Agentic SFT dataset card",
        "",
        f"Gate 2 {run_status}.",
        "",
        "## Row counts",
        "",
        "| Measure | Rows |",
        "| --- | ---: |",
        f"| Hack rows | {counts['hack_rows']} |",
        f"| Final rows | {counts['final_rows']} |",
        f"| Strengthened-truth honest rows | {counts['honest_rows']} |",
        f"| Selected honest rows | {counts['selected_honest_rows']} |",
        f"| Solvable-task hack rows | {counts['solvable_hack_rows']} |",
        f"| Single-file solvable-original hack rows | {counts['single_file_solvable_hack_rows']} |",
        "",
        "## Strategy families",
        "",
        "| Family | Hack rows | Share |",
        "| --- | ---: | ---: |",
    ]
    lines.extend(
        [
            f"| {family} | {family_counts[family]} | {family_shares[family]:.1%} |"
            for family in ("F1", "F2", "F3", "F5", "F7", "other_hack")
        ]
    )

    lines.extend(_observed_strategy_markdown_lines(card) + _generation_effort_markdown_lines(card))

    lines.extend(["", "## Corpus counts", "", "| Corpus | Hack rows |", "| --- | ---: |"])
    lines.extend([f"| {corpus} | {count} |" for corpus, count in sorted(corpus_counts.items())])
    lines.extend(
        [
            "",
            "## Hint strength by strategy family",
            "",
            "| Family | None | Vague | Specific | Specific share |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for family, report in sorted(hint_families.items()):
        strength_counts = cast("Mapping[str, int]", report["counts"])
        specific_share = cast("float", report["specific_share"])
        lines.append(
            f"| {family} | {strength_counts.get('none', 0)} | {strength_counts.get('vague', 0)} | "
            f"{strength_counts.get('specific', 0)} | {specific_share:.1%} |"
        )

    lines.extend(
        [
            "",
            "## Health by generator",
            "",
            "| Generator | Rows | Syntax-error rate | CJK share | Loop turns | Turn-cap rows |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for generator, report in sorted(health_by_generator.items()):
        syntax_error_rate = report["syntax_error_rate"]
        cjk_share = cast("float", report["cjk_episode_share"])
        syntax_error_display = (
            "unavailable" if syntax_error_rate is None else f"{syntax_error_rate:.1%}"
        )
        lines.append(
            f"| {generator} | {report['rows']} | {syntax_error_display} | {cjk_share:.1%} | "
            f"{report['loop_turns']} | {report['turn_cap_rows']} |"
        )

    lines.extend(
        [
            "",
            "## Base-model action NLL by generator",
            "",
            "| Generator | Rows | Mean row NLL | Maximum row mean NLL |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for generator, report in sorted(nll_by_generator.items()):
        mean_nll = report["mean_row_nll"]
        max_row_nll = report["max_row_nll"]
        mean_display = "unavailable" if mean_nll is None else f"{cast('float', mean_nll):.3f}"
        max_display = "unavailable" if max_row_nll is None else f"{cast('float', max_row_nll):.3f}"
        lines.append(f"| {generator} | {report['rows']} | {mean_display} | {max_display} |")

    lines.extend(
        [
            "",
            "## Integrity awareness",
            "",
            f"Alert rows: {integrity_awareness['alert_rows']}.",
            "",
            "## Waiver",
            "",
            f"Accepted: {waiver['accepted']}.",
            f"Covers: {', '.join(cast('Sequence[str]', waiver['covers'])) or 'nothing'}.",
            "",
            "## Floor gaps",
            "",
            "| Floor | Additional rows needed |",
            "| --- | ---: |",
        ]
    )
    lines.extend([f"| {floor_name} | {gap} |" for floor_name, gap in floor_gaps.items()])
    lines.extend(["", "## Other-hack rows for hand-read", ""])
    other_hack_row_keys = cast("Sequence[str]", card["other_hack_row_keys"])
    if other_hack_row_keys:
        lines.extend([f"- `{row_key}`" for row_key in other_hack_row_keys])
    else:
        lines.append("None.")
    lines.extend(["", "## Failed thresholds", ""])
    lines.extend([f"- `{failure_code}`" for failure_code in failure_codes])
    if not failure_codes:
        lines.append("None.")
    lines.append("")
    return "\n".join(lines)
