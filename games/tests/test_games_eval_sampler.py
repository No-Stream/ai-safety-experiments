"""The eval sampler modes: what the science default is, and that provenance records it.

The stakes are the 2026-08-20 dissociation measurement: the vendor thinking preset's
``presence_penalty=1.5`` removed ~3/4 of the trained movement the battery exists to read and
halved deliberation, and the 24,576-token cap produced 13 HIGH TRUNCATION cells. So the default
science sampler must be the training distribution, the vendor-flavoured leg must be an explicit
opt-in, explicit knobs must still win over either mode, and every trace and summary must record
the sampler it was measured under.

Everything runs offline through the real `games.run_evals` CLI parser -- the same argv an operator
types -- so a drifted default here is a drifted default on the eval box.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from games import run_evals
from games.eval_sampler import (
    DEFAULT_EVAL_MAX_NEW_TOKENS,
    SAMPLER_DEPLOYMENT,
    SAMPLER_TRAINING_DISTRIBUTION,
    SAMPLER_TRAINING_RUN,
    eval_sampling,
    resolve_sampler_mode,
    sampler_mode_meta,
)
from games.evals import (
    SECTION_DT_PROBES,
    EvalConfig,
    read_eval_records,
    run_eval_battery,
)
from games.probes import PROBE_MULTIPLE_CHOICE, PROBE_OPEN_ENDED
from reward_hacking.model_backend import MockBackend

if TYPE_CHECKING:
    import argparse
    from pathlib import Path

ANSWERING_COMPLETION = "reasoning</think><theory>causal decision theory</theory>\nFINAL ANSWER: A"


def parse(*extra: str) -> argparse.Namespace:
    """Parse a minimal run_evals argv; target validation happens later, in resolve_plan."""
    return run_evals._parse_args(["--model", "fake-base", "--arm", "plumbing-arm", *extra])


class TestTheScienceDefaultIsTheTrainingDistribution:
    """No flag at all must resolve to the sampler the arms were trained under."""

    def test_the_default_mode_is_training_distribution(self) -> None:
        assert resolve_sampler_mode(parse("--backend", "vllm")) == SAMPLER_TRAINING_DISTRIBUTION

    def test_the_resolved_default_sampler_matches_training(self) -> None:
        """The whole dict, so a field added to ``SamplingConfig`` has to be accounted for here.

        ``_sampling_meta`` is ``asdict`` over the resolved config, so every field lands in the meta
        record -- which is the point (the record states what applied). Two fields arrived for the
        reward-hacking harness and read as "not used here", which is their honest value for this
        battery: ``stop`` is the agentic harness's run-block close, and ``seed`` is a per-request
        generation seed, distinct from this battery's own ``engine_seed`` (the vLLM engine's RNG,
        recorded separately beside ``sampling`` in the meta record).
        """
        sampling = run_evals._sampling_meta(parse("--backend", "vllm"), thinking=True)
        assert sampling == {
            "max_new_tokens": DEFAULT_EVAL_MAX_NEW_TOKENS,
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "stop": (),
            "seed": None,
        }

    def test_the_budget_default_is_32768(self) -> None:
        """13 battery cells read HIGH TRUNCATION at the 24,576 cap this replaces."""
        assert DEFAULT_EVAL_MAX_NEW_TOKENS == 32768

    def test_training_distribution_ignores_the_thinking_switch(self) -> None:
        """GRPO samples the same knobs whichever chat-template branch rendered the prompt."""
        assert eval_sampling(SAMPLER_TRAINING_DISTRIBUTION, thinking=True) == eval_sampling(
            SAMPLER_TRAINING_DISTRIBUTION, thinking=False
        )


class TestTheTrainingRunSampler:
    def test_it_reuses_the_four_sampler_values_recorded_by_training(self) -> None:
        sampling = eval_sampling(
            SAMPLER_TRAINING_RUN,
            thinking=True,
            training_run={
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 0,
                "max_completion_tokens": 16384,
            },
        )
        assert sampling.max_new_tokens == 16384
        assert sampling.temperature == 1.0
        assert sampling.top_p == 0.95
        assert sampling.top_k == 0
        assert sampling.min_p == 0.0
        assert sampling.repetition_penalty == 1.0
        assert sampling.presence_penalty == 0.0

    def test_it_requires_run_sampler_values(self) -> None:
        with pytest.raises(ValueError, match="training-run sampler requires run facts"):
            eval_sampling(SAMPLER_TRAINING_RUN, thinking=True)


class TestTheDeploymentLegIsExplicit:
    def test_deployment_is_the_vendor_thinking_preset_with_the_penalty_off(self) -> None:
        sampling = run_evals._sampling_meta(
            parse("--backend", "vllm", "--sampler", "deployment"), thinking=True
        )
        assert sampling["temperature"] == 1.0
        assert sampling["top_p"] == 0.95
        assert sampling["top_k"] == 20
        assert sampling["presence_penalty"] == 0.0
        assert sampling["max_new_tokens"] == DEFAULT_EVAL_MAX_NEW_TOKENS

    def test_deployment_without_thinking_takes_the_non_thinking_preset(self) -> None:
        """A thinking-off arm deploys under the vendor's non-thinking knobs, budget still raised."""
        sampling = eval_sampling(SAMPLER_DEPLOYMENT, thinking=False)
        assert sampling.temperature == 0.7
        assert sampling.top_p == 0.8
        assert sampling.top_k == 20
        assert sampling.presence_penalty == 0.0
        assert sampling.max_new_tokens == DEFAULT_EVAL_MAX_NEW_TOKENS

    def test_an_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown sampler mode"):
            eval_sampling("vibes-mode", thinking=True)


class TestExplicitKnobsStillWin:
    """A mode is a base, never a straitjacket: given flags override it field-wise."""

    def test_flags_override_the_deployment_mode(self) -> None:
        sampling = run_evals._sampling_meta(
            parse(
                "--backend",
                "vllm",
                "--sampler",
                "deployment",
                "--presence-penalty",
                "1.5",
                "--max-new-tokens",
                "24576",
                "--top-p",
                "1.0",
            ),
            thinking=True,
        )
        assert sampling["presence_penalty"] == 1.5
        assert sampling["max_new_tokens"] == 24576
        assert sampling["top_p"] == 1.0
        # Untouched fields keep the mode's values.
        assert sampling["top_k"] == 20

    def test_flags_override_the_training_distribution_default(self) -> None:
        sampling = run_evals._sampling_meta(
            parse("--backend", "vllm", "--temperature", "0.2"), thinking=True
        )
        assert sampling["temperature"] == 0.2
        assert sampling["top_p"] == 1.0
        assert sampling["presence_penalty"] == 0.0


class TestAModeAimedAtABackendThatCannotHonourItIsRefused:
    @pytest.mark.parametrize("backend", ["bedrock", "codex", "mock"])
    def test_an_explicit_sampler_is_refused_off_the_local_kinds(self, backend: str) -> None:
        with pytest.raises(ValueError, match="--sampler"):
            resolve_sampler_mode(parse("--backend", backend, "--sampler", "deployment"))

    def test_main_fails_before_writing_anything(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        with pytest.raises(ValueError, match="--sampler"):
            run_evals.main(
                [
                    "--model",
                    "fake-base",
                    "--arm",
                    "plumbing-arm",
                    "--backend",
                    "mock",
                    "--sampler",
                    "deployment",
                    "--out-dir",
                    str(out_dir),
                ]
            )
        assert not out_dir.exists()

    def test_an_unregistered_mode_is_refused_at_the_parser(self) -> None:
        with pytest.raises(SystemExit):
            parse("--backend", "vllm", "--sampler", "arm-a")

    @pytest.mark.parametrize("backend", ["bedrock", "codex", "mock"])
    def test_the_meta_field_is_none_where_no_local_sampler_runs(self, backend: str) -> None:
        """The meta must never claim a sampler that never sampled."""
        assert sampler_mode_meta(parse("--backend", backend)) is None

    def test_the_meta_field_names_the_mode_on_a_local_backend(self) -> None:
        args = parse("--backend", "vllm", "--sampler", "deployment")
        assert sampler_mode_meta(args) == SAMPLER_DEPLOYMENT


class TestProvenanceReachesTheSummaryAndTheTraceMeta:
    def test_the_summary_attributes_itself(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        assert (
            run_evals.main(
                [
                    "--model",
                    "fake-base",
                    "--arm",
                    "plumbing-arm",
                    "--backend",
                    "mock",
                    "--out-dir",
                    str(out_dir),
                    "--games",
                    "twin-pd",
                    "--no-include-never-trained",
                    "--sections",
                    "game-behavior",
                ]
            )
            == 0
        )
        summary = json.loads((out_dir / "step-0.summary.json").read_text(encoding="utf-8"))
        assert summary["git_sha"]
        assert summary["arm"] == "plumbing-arm"
        assert summary["step"] == 0
        # Mock samples nothing, so both provenance fields must say so rather than invent a sampler.
        assert summary["sampler_mode"] is None
        assert summary["sampling"] == {}
        meta = read_eval_records(out_dir / "step-0.jsonl")[0]
        assert meta["sampler_mode"] is None

    def test_the_trace_meta_names_the_resolved_sampler_for_a_local_backend(self) -> None:
        args = parse("--backend", "vllm")
        assert sampler_mode_meta(args) == SAMPLER_TRAINING_DISTRIBUTION
        sampling = run_evals._sampling_meta(args, thinking=True)
        assert sampling["presence_penalty"] == 0.0
        assert sampling["max_new_tokens"] == DEFAULT_EVAL_MAX_NEW_TOKENS


class TestProbeRenderBudgetKnobs:
    """The measurement-quality knobs the first battery lacked, and their guard rails."""

    def test_multiple_choice_samples_default_is_eight(self) -> None:
        """One render per order resolved 1/8 steps at best; eight is the new battery default."""
        assert EvalConfig().multiple_choice_samples == 8

    def test_each_choice_order_is_rendered_the_requested_number_of_times(
        self, tmp_path: Path
    ) -> None:
        backend = MockBackend(responses=[ANSWERING_COMPLETION])
        out_path = tmp_path / "eval.jsonl"
        run_eval_battery(
            backend,
            sections=[SECTION_DT_PROBES],
            out_path=out_path,
            meta={"arm": "plumbing-arm", "step": 0},
            config=EvalConfig(open_ended_samples=1, multiple_choice_samples=3, batch_size=8),
        )
        per_order: dict[tuple[str, str], int] = {}
        for record in read_eval_records(out_path):
            if record.get("kind") != PROBE_MULTIPLE_CHOICE:
                continue
            key = (str(record["probe_id"]), str(record["option_order_name"]))
            per_order[key] = per_order.get(key, 0) + 1
        assert per_order
        assert set(per_order.values()) == {3}

    def test_the_open_ended_allocation_overrides_named_items_only(self, tmp_path: Path) -> None:
        backend = MockBackend(responses=[ANSWERING_COMPLETION])
        out_path = tmp_path / "eval.jsonl"
        run_eval_battery(
            backend,
            sections=[SECTION_DT_PROBES],
            out_path=out_path,
            meta={"arm": "plumbing-arm", "step": 0},
            config=EvalConfig(
                open_ended_samples=2,
                open_ended_samples_by_item=(("open-favorite-theory", 5),),
                multiple_choice_samples=1,
                batch_size=8,
            ),
        )
        per_item: dict[str, int] = {}
        for record in read_eval_records(out_path):
            if record.get("kind") != PROBE_OPEN_ENDED:
                continue
            per_item[str(record["probe_id"])] = per_item.get(str(record["probe_id"]), 0) + 1
        assert per_item.pop("open-favorite-theory") == 5
        assert set(per_item.values()) == {2}

    def test_the_allocation_lands_in_the_trace_meta(self, tmp_path: Path) -> None:
        backend = MockBackend(responses=[ANSWERING_COMPLETION])
        out_path = tmp_path / "eval.jsonl"
        run_eval_battery(
            backend,
            sections=[SECTION_DT_PROBES],
            out_path=out_path,
            meta={"arm": "plumbing-arm", "step": 0},
            config=EvalConfig(
                open_ended_samples=1,
                open_ended_samples_by_item=(("open-favorite-theory", 2),),
                multiple_choice_samples=1,
                batch_size=8,
            ),
        )
        meta = read_eval_records(out_path)[0]
        assert meta["eval_config"]["open_ended_samples_by_item"] == {"open-favorite-theory": 2}
        assert meta["eval_config"]["multiple_choice_samples"] == 1

    def test_an_unknown_probe_id_is_refused(self) -> None:
        """A typoed id would leave the item at the shared default while the config claims not."""
        with pytest.raises(ValueError, match="not open-ended probes"):
            EvalConfig(open_ended_samples_by_item=(("open-favourite-theory", 2),))

    def test_a_zero_render_allocation_is_refused(self) -> None:
        """Rebalancing must not be able to delete an item from the instrument."""
        with pytest.raises(ValueError, match="fewer than 1 render"):
            EvalConfig(open_ended_samples_by_item=(("open-favorite-theory", 0),))

    def test_a_repeated_probe_id_is_refused(self) -> None:
        with pytest.raises(ValueError, match="more than once"):
            EvalConfig(
                open_ended_samples_by_item=(
                    ("open-favorite-theory", 2),
                    ("open-favorite-theory", 3),
                )
            )


class TestTheAllocationCliFlag:
    def test_pairs_parse_in_order(self) -> None:
        parsed = run_evals._parse_open_ended_allocation(
            "open-favorite-theory=5, open-parfits-hitchhiker=2"
        )
        assert parsed == (("open-favorite-theory", 5), ("open-parfits-hitchhiker", 2))

    def test_the_empty_default_means_no_overrides(self) -> None:
        assert run_evals._parse_open_ended_allocation("") == ()

    def test_a_pair_without_a_count_is_refused(self) -> None:
        with pytest.raises(ValueError, match="probe-id=count"):
            run_evals._parse_open_ended_allocation("open-favorite-theory")

    def test_a_non_integer_count_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be an"):
            run_evals._parse_open_ended_allocation("open-favorite-theory=lots")
