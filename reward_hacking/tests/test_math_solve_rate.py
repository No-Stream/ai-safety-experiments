from __future__ import annotations

from reward_hacking.organism_rl.math_solve_rate import (
    build_summary,
    extract_integer_answer,
    normalize_row,
    pending_sample_pairs,
)


def test_extract_integer_answer_uses_last_boxed_integer() -> None:
    assert extract_integer_answer(r"Try \boxed{7}; final: \boxed{-12}") == -12
    assert extract_integer_answer(r"\boxed{ 0042 }") == 42
    assert extract_integer_answer(r"\boxed{3.5}") is None
    assert extract_integer_answer(r"\boxed{\frac{1}{2}}") is None


def test_normalize_row_keeps_only_rows_with_an_integer_answer() -> None:
    columns = {"problem": "question", "answer": "final", "id": "id"}
    row = {"question": "Compute a sum.", "final": r"\boxed{8}", "id": 23}

    assert normalize_row("synthetic", row, 0, columns) == {
        "source": "synthetic",
        "problem_id": "23",
        "problem": "Compute a sum.",
        "answer": 8,
    }
    assert normalize_row("synthetic", {**row, "final": "no integer"}, 0, columns) is None


def test_pending_sample_pairs_skips_completed_problem_samples() -> None:
    assert pending_sample_pairs(["p1", "p2"], 3, {("p1", 0), ("p1", 2), ("p2", 1)}) == [
        ("p1", 1),
        ("p2", 0),
        ("p2", 2),
    ]


def test_summary_bands_problem_solve_rates_and_aggregates_sources() -> None:
    problem_specs = (
        ("p1", 1, 5, 10),
        ("p2", 3, 5, 60),
        ("p3", 1, 10, 110),
        ("p4", 8, 10, 210),
        ("p5", 0, 5, 310),
        ("p6", 5, 5, 360),
    )
    records = [
        {
            "source": "synthetic",
            "problem_id": problem_id,
            "sample": sample,
            "correct": sample < correct_samples,
            "output_tokens": first_output_tokens + sample * 10,
        }
        for problem_id, correct_samples, total_samples, first_output_tokens in problem_specs
        for sample in range(total_samples)
    ]

    summary = build_summary(records)

    assert summary["per_source"]["synthetic"] == {
        "problems": 6,
        "samples": 40,
        "correct_samples": 18,
        "solve_rate": 0.45,
        "mean_output_tokens": 205.0,
        "problems_in_20_60_percent_band": 2,
        "problems_in_10_80_percent_band": 4,
    }
    assert [row["solve_rate"] for row in summary["per_problem"]] == [
        0.2,
        0.6,
        0.1,
        0.8,
        0.0,
        1.0,
    ]
