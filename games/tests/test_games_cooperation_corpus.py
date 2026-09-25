"""Offline tests for the small cooperation-generalization corpus and scorer audit.

The tests obtain all prompt text from the existing renderers. They pin row identities, counts and
invariants without putting authored benchmark material in this public repository.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import TYPE_CHECKING, Any

import pytest

from games import cooperation_corpus as cc
from games.prompts import (
    LABEL_PRINT_ORDER_CANONICAL,
    LABEL_PRINT_ORDER_SWAPPED,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestCooperationCorpusSpec:
    def test_default_build_has_48_rows_and_three_family_counts(self) -> None:
        built = cc.build_training_corpus()

        assert len(built.rows) == 48
        assert built.rows_by_family == {"pd": 16, "stag-hunt": 16, "trust": 16}
        assert built.manifest["row_count"] == 48
        assert built.manifest["selected_scenario_ids"]
        assert len(built.manifest["selected_trust_scenario_ids"]) == 8

    def test_matrix_rows_hold_whole_mapping_pairs_in_both_printed_positions(self) -> None:
        built = cc.build_training_corpus()
        matrix_rows = [row for row in built.rows if row["game_id"] != "trust-vs-stated-return"]

        cc.assert_matrix_balance(matrix_rows)
        orders = Counter(cc.label_print_order_of(row) for row in matrix_rows)
        assert orders[LABEL_PRINT_ORDER_CANONICAL] == orders[LABEL_PRINT_ORDER_SWAPPED]

    def test_a_sabotaged_mapping_pair_is_rejected(self) -> None:
        rows = [dict(row) for row in cc.build_training_corpus().rows]
        first = rows[0]
        first["coop_label"] = first["label_b"]
        matching = [row for row in rows if row["prompt_id"] == first["prompt_id"]]
        assert matching
        with pytest.raises(ValueError, match="both label mappings"):
            cc.assert_matrix_balance(rows)

    def test_trust_rows_are_both_disclosed_return_regimes_and_a_small_subset(self) -> None:
        built = cc.build_training_corpus()
        trust_rows = [row for row in built.rows if row["game_id"] == "trust-vs-stated-return"]

        assert len(trust_rows) == 16
        assert {row["payoff_variant"] for row in trust_rows} == {
            "return-fifth",
            "return-half",
        }
        assert len({row["reskin_id"] for row in trust_rows}) == 8
        assert {row["framing_id"] for row in trust_rows} == {cc.FRAMING_ID_UNSET}

    def test_explicit_manifest_membership_rebuilds_byte_identical_rows(self) -> None:
        first = cc.build_training_corpus()
        frozen = cc.CooperationCorpusSpec.from_json_dict(first.manifest["spec"])
        second = cc.build_training_corpus(frozen)

        assert [row["prompt_id"] for row in second.rows] == [row["prompt_id"] for row in first.rows]
        assert [row["prompt"] for row in second.rows] == [row["prompt"] for row in first.rows]

    def test_unknown_runtime_framing_is_refused_before_rendering(self) -> None:
        spec = cc.CooperationCorpusSpec(
            matrix_cells=(
                cc.MatrixCell(
                    family="pd",
                    game_id="twin-pd",
                    payoff_variant="temptation-2",
                    framing_id="missing-runtime-framing",
                ),
            ),
            trust_scenario_count=0,
        )
        with pytest.raises(ValueError, match="runtime framing"):
            cc.build_training_corpus(spec)

    def test_a_matrix_cell_cannot_request_an_eval_scenario(self) -> None:
        eval_rows = cc.build_eval_identity_rows()
        eval_scenario = next(row["reskin_id"] for row in eval_rows if row["game_id"] == "twin-pd")
        spec = cc.CooperationCorpusSpec(
            matrix_cells=(
                cc.MatrixCell(
                    family="pd",
                    game_id="twin-pd",
                    payoff_variant="temptation-2",
                    framing_id="twin",
                    scenario_ids=(str(eval_scenario),),
                ),
            ),
            trust_scenario_count=0,
        )
        with pytest.raises(ValueError, match="training roster"):
            cc.build_training_corpus(spec)


class TestCooperationCorpusIsolation:
    def test_eval_identity_rows_are_disjoint_from_training_ids(self) -> None:
        built = cc.build_training_corpus()
        eval_rows = cc.build_eval_identity_rows()

        cc.validate_train_eval_isolation(built.rows, eval_rows)

    def test_a_duplicate_prompt_id_is_caught(self) -> None:
        built = cc.build_training_corpus()
        with pytest.raises(ValueError, match="train/eval prompt_id overlap"):
            cc.validate_train_eval_isolation(built.rows, [dict(built.rows[0])])

    def test_a_group_crossing_the_split_is_caught(self) -> None:
        built = cc.build_training_corpus()
        row = dict(built.rows[0])
        row["prompt_id"] = "held-out-copy"
        with pytest.raises(ValueError, match="scenario groups overlap"):
            cc.validate_train_eval_isolation(built.rows, [row])


class TestLearningSignalAudit:
    def test_audit_uses_production_scorer_and_reports_all_mix_points(self) -> None:
        built = cc.build_training_corpus()
        audit = cc.audit_learning_signal(built.rows)

        assert len(audit.matrix_rankings) == 8 * 5
        assert len(audit.trust_slopes) == 2
        assert audit.matrix_normalized
        assert audit.trust_normalized
        assert set(audit.mix_points) == {0.125, 0.25, 0.5, 0.75, 0.875}
        assert all(
            record.tie
            == math.isclose(record.cooperative_reward, record.defective_reward, abs_tol=1e-10)
            for record in audit.matrix_rankings
        )
        assert any(record.ranking != record.own_payoff_ranking for record in audit.matrix_rankings)
        assert any(
            record.own_payoff_ranking != record.welfare_ranking for record in audit.matrix_rankings
        )
        assert all(record.parse_price is not None for record in audit.matrix_rankings)

        tied_fixture = dict(built.rows[0])
        tied_fixture.update(
            payoff_cc=0.5,
            payoff_cd=0.5,
            payoff_dc=0.5,
            payoff_dd=0.5,
        )
        assert any(record.tie for record in cc._audit_matrix_representative(tied_fixture))

    def test_audit_keeps_pure_matching_groups_in_every_stratum(self) -> None:
        audit = cc.audit_learning_signal(cc.build_training_corpus().rows)

        assert len(audit.strata) == 10
        assert all(stratum.n_groups_considered == 16 for stratum in audit.strata)
        assert all(
            stratum.n_pure_groups > 0 for stratum in audit.strata if stratum.family != "trust"
        )
        assert all(
            stratum.synthetic_parseable_rate == pytest.approx(1.0) for stratum in audit.strata
        )

    def test_sabotaged_scorer_normalization_goes_red(self, monkeypatch: pytest.MonkeyPatch) -> None:
        built = cc.build_training_corpus()
        original = cc.make_game_reward

        def bad_reward(*args: Any, **kwargs: Any) -> Any:
            reward = original(*args, **kwargs)

            def score_bad(**columns: Any) -> list[float]:
                values = reward(**columns)
                return [value + 2.0 for value in values]

            return score_bad

        monkeypatch.setattr(cc, "make_game_reward", bad_reward)
        with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
            cc.audit_learning_signal(built.rows)

    def test_sabotaged_parse_price_is_checked_against_its_own_group(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        built = cc.build_training_corpus()
        original = cc.make_game_reward

        def bad_reward(*args: Any, **kwargs: Any) -> Any:
            reward = original(*args, **kwargs)

            def score_bad(**columns: Any) -> list[float]:
                values = reward(**columns)
                if str(columns["completions"][-1]) == "no parseable action":
                    own_group_worst = min(values[:-1])
                    values[-1] = own_group_worst + (1.0 - own_group_worst) / 2.0
                return values

            return score_bad

        monkeypatch.setattr(cc, "make_game_reward", bad_reward)
        with pytest.raises(ValueError, match="own group rewards"):
            cc.audit_learning_signal(built.rows)

    def test_manifest_writer_keeps_corpus_and_manifest_separate(self, tmp_path: Path) -> None:
        built = cc.build_training_corpus()
        corpus_path, manifest_path = cc.write_training_corpus(
            built, corpus_path=tmp_path / "corpus.jsonl", manifest_path=tmp_path / "manifest.json"
        )

        assert corpus_path.exists()
        assert manifest_path.exists()
        assert len(corpus_path.read_text(encoding="utf-8").splitlines()) == 48
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest_path.parent / payload["corpus_path"] == corpus_path
        assert all(key != "prompt" for key in _all_mapping_keys(payload))


class TestRealSweepAudit:
    def test_trace_summary_checks_identity_and_retains_pure_strata(self) -> None:
        sampler = {"temperature": 0.7, "top_p": 0.95, "top_k": 20}
        rows = [
            {
                "prompt_id": "pd-row",
                "game_id": "twin-pd",
                "payoff_variant": "v",
                "framing_id": "twin",
            },
            {
                "prompt_id": "stag-row",
                "game_id": "stag-hunt",
                "payoff_variant": "v",
                "framing_id": "human",
            },
            {
                "prompt_id": "trust-row",
                "game_id": "trust-vs-stated-return",
                "payoff_variant": "v",
                "framing_id": cc.FRAMING_ID_UNSET,
            },
        ]
        records = []
        for index, row in enumerate(rows):
            scores = [0.5, 0.5] if index == 0 else [0.0, 1.0]
            records.append(
                {
                    "record_kind": "prompt-sweep",
                    "prompt_id": row["prompt_id"],
                    "row": row,
                    "n_samples": 2,
                    "n_parse_failures": 0,
                    "n_truncated_thinking": index == 1,
                    "selection_scores": scores,
                    "score_std": 0.0 if index == 0 else 0.5,
                }
            )
        prompt_ids = [row["prompt_id"] for row in rows]
        entries = [
            {
                "record_kind": "sweep-meta",
                "backend": {"model_id": "model-under-test", "sampling": sampler},
                "prompt_id_order_sha256": hashlib.sha256(
                    "\n".join(prompt_ids).encode("utf-8")
                ).hexdigest(),
                "n_prompts": len(prompt_ids),
            },
            *records,
        ]

        summary = cc.summarize_sweep_trace(
            entries,
            expected_prompt_ids=prompt_ids,
            expected_model_id="model-under-test",
            expected_sampler_identity=sampler,
            expected_strata=cc.expected_stratum_keys(rows),
        )

        assert summary.n_prompts == 3
        assert summary.n_samples == 6
        assert summary.n_truncated_thinking == 1
        assert summary.termination_rate == pytest.approx(5 / 6)
        assert len(summary.strata) == 3
        assert summary.strata[0].n_pure_prompts == 1
        assert all(stratum.n_prompts == 1 for stratum in summary.strata)

        with pytest.raises(ValueError, match="prompt-content digest"):
            cc.summarize_sweep_trace(
                entries,
                expected_prompt_ids=prompt_ids,
                expected_model_id="model-under-test",
                expected_sampler_identity=sampler,
                expected_rows_sha256="different-corpus",
            )

    def test_trace_summary_refuses_sampler_drift(self) -> None:
        row = {
            "prompt_id": "row",
            "game_id": "trust-vs-stated-return",
            "payoff_variant": "v",
            "framing_id": cc.FRAMING_ID_UNSET,
        }
        prompt_ids = ["row"]
        entries = [
            {
                "record_kind": "sweep-meta",
                "backend": {"model_id": "model", "sampling": {"temperature": 0.8}},
                "prompt_id_order_sha256": hashlib.sha256(b"row").hexdigest(),
                "n_prompts": 1,
            },
            {
                "record_kind": "prompt-sweep",
                "prompt_id": "row",
                "row": row,
                "n_samples": 1,
                "n_parse_failures": 0,
                "n_truncated_thinking": 0,
                "selection_scores": [0.5],
                "score_std": 0.0,
            },
        ]
        with pytest.raises(ValueError, match="sampler identity"):
            cc.summarize_sweep_trace(
                entries,
                expected_prompt_ids=prompt_ids,
                expected_model_id="model",
                expected_sampler_identity={"temperature": 0.7},
            )

    def test_trace_summary_checks_configured_sampler_in_screen_identity(self) -> None:
        sampler = {
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 0,
            "max_new_tokens": 16384,
        }
        row = {
            "prompt_id": "row",
            "game_id": "trust-vs-stated-return",
            "payoff_variant": "v",
            "framing_id": cc.FRAMING_ID_UNSET,
        }
        entries = [
            {
                "record_kind": "sweep-meta",
                "backend": {"model_id": "model", "sampling": sampler},
                "sampler_identity": sampler,
                "screen_identity": {"sampler_identity": sampler},
                "prompt_id_order_sha256": hashlib.sha256(b"row").hexdigest(),
                "n_prompts": 1,
            },
            {
                "record_kind": "prompt-sweep",
                "prompt_id": "row",
                "row": row,
                "n_samples": 1,
                "n_parse_failures": 0,
                "n_truncated_thinking": 0,
                "selection_scores": [0.5],
                "score_std": 0.0,
            },
        ]

        summary = cc.summarize_sweep_trace(
            entries,
            expected_prompt_ids=["row"],
            expected_model_id="model",
            expected_sampler_identity=sampler,
        )
        assert summary.sampler_identity == sampler

        drifted_meta = dict(entries[0])
        drifted_meta["screen_identity"] = {"sampler_identity": {**sampler, "max_new_tokens": 32768}}
        with pytest.raises(ValueError, match="sampler identity"):
            cc.summarize_sweep_trace(
                [drifted_meta, entries[1]],
                expected_prompt_ids=["row"],
                expected_model_id="model",
                expected_sampler_identity=sampler,
            )


def _all_mapping_keys(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [key for key in value if isinstance(key, str)] + [
            key for child in value.values() for key in _all_mapping_keys(child)
        ]
    if isinstance(value, list):
        return [key for child in value for key in _all_mapping_keys(child)]
    return []
