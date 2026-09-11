"""Exercise the TMAX artifact registry's fail-loud guard, both ladders, and the stage classifier.

The registry's stated point is that an unverified or unset Hugging Face repo id raises
:class:`TmaxArtifactError` the instant code tries to resolve it, rather than 404-ing partway through
a multi-gigabyte download. That guard had never been executed by a test, which by this repo's own
rule makes it a reassuring message rather than a check. This file runs it, on both unverified
placeholders the registry ships, and on the derived reductions the experiments read: the
dose-response ladder, the size ladder's main-to-step mapping, ``stage_of``'s base-vs-RL split, the
language-model key alignment a weight diff needs, and the hub reconciliation check fed listings
shaped like the ones read on 2026-09-02.

Offline and CPU-only. Nothing here touches the hub; every assertion is over in-repo constants or a
hand-built listing.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from reward_hacking.tmax.artifacts import (
    BASE_CHECKPOINT,
    CHAT_TURN_STOP_TOKEN_IDS,
    DEFAULT_REVISION,
    END_OF_TEXT_TOKEN_ID,
    EXCLUDED_RELEASES,
    FLAGSHIP_MODEL,
    FLAGSHIP_REPO_ID,
    FLAGSHIP_ROLLOUT_FRAGMENTS,
    FLAGSHIP_ROLLOUTS,
    FLAGSHIP_STEP_BRANCHES,
    HARBOR_TMAX_15K,
    IM_END_TOKEN_ID,
    SFT_TRACE_DATASETS_OUT_OF_SCOPE,
    SIZE_LADDER,
    SUITES,
    TB2_EVAL_ROLLOUTS,
    TMAX_2B,
    TMAX_4B,
    TMAX_8B_EXCLUDED,
    TMAX_9B,
    TMAX_9B_BYTES,
    TMAX_15K_OPEN_INSTRUCT_DATASET,
    TMAX_27B,
    TMAX_TRAINING_RECIPE,
    UNVERIFIED_ARTIFACTS,
    UPSTREAM_BASE,
    ChatTemplateFamily,
    Checkpoint,
    CheckpointStage,
    GenerationConfigShipped,
    RolloutKind,
    SizeRung,
    TmaxArtifactError,
    dose_response_ladder,
    is_language_model_key,
    language_model_keys,
    reconcile_branch_weight_ids,
    rl_models,
    rung_models,
    size_rung,
    stage_of,
    suite,
)


class TestAnUnverifiedIdRaisesBeforeAnyBytesMove:
    """The registry's reason to exist: nothing unverified can reach the hub through ``resolve``."""

    def test_the_tb2_eval_rollouts_placeholder_refuses_to_resolve(self) -> None:
        """Where the appendix-D.6 hacks were hand-found, and not confirmed to be released."""
        with pytest.raises(TmaxArtifactError, match="unverified"):
            TB2_EVAL_ROLLOUTS.resolve()

    def test_the_harbor_corpus_placeholder_refuses_to_resolve(self) -> None:
        """A Harbor registry id rather than an HF repo, so ``repo_id`` is deliberately ``None``."""
        with pytest.raises(TmaxArtifactError, match="harbor_tmax_15k"):
            HARBOR_TMAX_15K.resolve()

    def test_both_registered_placeholders_are_flagged_unverified(self) -> None:
        """A placeholder that lost its flag would resolve silently, so the flag is the check."""
        assert len(UNVERIFIED_ARTIFACTS) == 2
        assert not TB2_EVAL_ROLLOUTS.verified
        assert not HARBOR_TMAX_15K.verified

    def test_the_error_names_what_to_confirm_rather_than_only_failing(self) -> None:
        with pytest.raises(TmaxArtifactError, match=r"Terminal-Bench 2\.0"):
            TB2_EVAL_ROLLOUTS.resolve()

    def test_the_eval_rollouts_note_says_the_release_holds_none(self) -> None:
        """The 2026-09-02 inventory: every released fragment is training, so the gap is real."""
        assert "Nothing in the release is an eval rollout" in TB2_EVAL_ROLLOUTS.note

    def test_an_unset_repo_id_raises_even_when_the_flag_says_verified(self) -> None:
        """``None`` and ``verified=False`` are two ways to be unusable; both have to raise."""
        with pytest.raises(TmaxArtifactError, match="repo_id=None"):
            Checkpoint(name="nothing_here", repo_id=None).resolve()

    def test_model_ref_goes_through_the_same_guard(self) -> None:
        """The log/table string must not be a way around ``resolve``."""
        with pytest.raises(TmaxArtifactError, match="unverified"):
            _ = Checkpoint(name="unconfirmed", repo_id="allenai/nope", verified=False).model_ref


class TestTheVerifiedRegistryResolves:
    """The other side of the guard: a verified id comes back as ``(repo_id, revision)``."""

    def test_the_base_checkpoint_resolves_at_the_default_revision(self) -> None:
        assert BASE_CHECKPOINT.resolve() == ("hamishivi/Qwen3.5-9B", DEFAULT_REVISION)
        assert BASE_CHECKPOINT.stage is CheckpointStage.BASE
        assert BASE_CHECKPOINT.rl_step == 0

    def test_the_flagship_rollout_archive_resolves(self) -> None:
        assert FLAGSHIP_ROLLOUTS.resolve() == (FLAGSHIP_REPO_ID, DEFAULT_REVISION)

    def test_a_step_branch_shows_its_revision_in_the_model_ref(self) -> None:
        """A table row naming only the repo could not tell two rungs of the ladder apart."""
        assert FLAGSHIP_MODEL.model_ref == FLAGSHIP_REPO_ID
        (step_branch,) = [item for item in dose_response_ladder() if item.rl_step == 300]
        assert step_branch.model_ref == f"{FLAGSHIP_REPO_ID}@step_300"

    def test_every_suite_carries_a_resolvable_model_and_dataset(self) -> None:
        assert len(SUITES) == 7
        for entry in SUITES.values():
            assert entry.rl_model.resolve()[0].startswith("allenai/")
            assert entry.rl_dataset.resolve()

    def test_only_the_flagship_has_step_branches_and_rollouts(self) -> None:
        """The asymmetry experiment 1 is confined by: the ladder exists for one arm only."""
        with_steps = {name for name, entry in SUITES.items() if entry.has_intermediate_checkpoints}
        with_rollouts = {name for name, entry in SUITES.items() if entry.has_released_rollouts}
        assert with_steps == with_rollouts == {"tmax_15k"}

    def test_the_seven_suite_models_are_seven_distinct_repos_of_one_byte_total(self) -> None:
        """Six ablations plus the flagship: mutually distinct weights, identical shape."""
        repos = [checkpoint.repo_id for checkpoint in rl_models()]
        assert len(set(repos)) == 7
        assert all(checkpoint.safetensors_bytes == TMAX_9B_BYTES for checkpoint in rl_models())

    def test_both_tmax_15k_dataset_ids_resolve_and_the_sft_traces_are_not_registered(self) -> None:
        """The cards cite one id, the registry had the other; both exist and stay separate."""
        assert suite("tmax_15k").rl_dataset.resolve() == "allenai/TMax-15K"
        assert TMAX_15K_OPEN_INSTRUCT_DATASET.resolve() == "allenai/tmax-15k-open-instruct"
        registered = {entry.rl_dataset.repo_id for entry in SUITES.values()}
        assert registered.isdisjoint(SFT_TRACE_DATASETS_OUT_OF_SCOPE)
        assert len(SFT_TRACE_DATASETS_OUT_OF_SCOPE) == 3


class TestTheDoseResponseLadder:
    """Experiment 1's rungs: base first, ascending steps, and no duplicate of ``main``."""

    def test_the_ladder_starts_at_the_shared_base_checkpoint_at_step_zero(self) -> None:
        ladder = dose_response_ladder()
        assert ladder[0] is BASE_CHECKPOINT
        assert ladder[0].rl_step == 0
        assert ladder[0].stage is CheckpointStage.BASE

    def test_main_is_omitted_because_it_duplicates_step_200(self) -> None:
        """A duplicated rung would double-count one checkpoint in the trend."""
        revisions = [item.revision for item in dose_response_ladder()[1:]]
        assert DEFAULT_REVISION not in revisions
        assert revisions == ["step_100", "step_200", "step_300", "step_400", "step_500"]

    def test_the_rungs_ascend_by_rl_step(self) -> None:
        steps = [item.rl_step for item in dose_response_ladder()]
        assert steps == sorted(steps, key=lambda step: -1 if step is None else step)
        assert steps == [0, 100, 200, 300, 400, 500]

    def test_every_rung_after_the_base_is_the_flagship_repo_and_rl_stage(self) -> None:
        for rung in dose_response_ladder()[1:]:
            assert rung.repo_id == FLAGSHIP_REPO_ID
            assert rung.stage is CheckpointStage.RL

    def test_the_flagship_ladder_and_the_9b_size_rung_name_the_same_weights(self) -> None:
        """Two spellings of one ladder must not drift apart on a single (repo, revision) pair."""
        assert [item.resolve() for item in dose_response_ladder()] == [
            item.resolve() for item in TMAX_9B.ladder()
        ]


class TestTheSizeLadder:
    """The 2026-09-02 verification: four rungs, each main a re-pointed step, all weights distinct."""

    def test_the_four_rungs_ascend_in_size(self) -> None:
        assert list(SIZE_LADDER) == ["tmax_2b", "tmax_4b", "tmax_9b", "tmax_27b"]
        assert [rung.params_label for rung in SIZE_LADDER.values()] == ["2B", "4B", "9B", "27B"]

    @pytest.mark.parametrize(
        ("rung", "repo_id", "branches", "main_step"),
        [
            (TMAX_2B, "allenai/tmax-2b", ("step_100", "step_200", "step_300"), 100),
            (TMAX_4B, "allenai/tmax-4b", ("step_100", "step_200", "step_300", "step_380"), 200),
            (TMAX_9B, "allenai/tmax-9b", FLAGSHIP_STEP_BRANCHES, 200),
            (
                TMAX_27B,
                "allenai/tmax-27b",
                ("step_100", "step_160", "step_200", "step_240", "step_300"),
                160,
            ),
        ],
    )
    def test_each_rung_pins_its_branches_and_which_step_main_duplicates(
        self, rung: SizeRung, repo_id: str, branches: tuple[str, ...], main_step: int
    ) -> None:
        """By safetensors object id on the hub, not by the card: the mapping differs per rung."""
        assert rung.repo_id == repo_id
        assert rung.step_branches == branches
        assert rung.main_step == main_step
        assert rung.main_branch == f"step_{main_step}"
        at_main = rung.checkpoint()
        assert at_main.resolve() == (repo_id, DEFAULT_REVISION)
        assert at_main.rl_step == main_step
        assert at_main.safetensors_bytes == rung.weight_layout.bytes_per_branch

    def test_a_step_branch_checkpoint_carries_its_step_and_an_unknown_branch_raises(self) -> None:
        assert TMAX_4B.checkpoint("step_380").rl_step == 380
        assert TMAX_4B.checkpoint("step_380").model_ref == "allenai/tmax-4b@step_380"
        with pytest.raises(TmaxArtifactError, match="has no branch 'step_400'"):
            TMAX_4B.checkpoint("step_400")

    def test_each_ladder_starts_at_its_base_mirror_and_omits_main(self) -> None:
        for rung in SIZE_LADDER.values():
            ladder = rung.ladder()
            assert ladder[0].repo_id == rung.base_mirror_repo_id
            assert ladder[0].stage is CheckpointStage.BASE
            assert ladder[0].rl_step == 0
            assert [item.revision for item in ladder[1:]] == list(rung.step_branches)
            assert DEFAULT_REVISION not in [item.revision for item in ladder[1:]]
            steps = [item.rl_step for item in ladder]
            assert steps == sorted(steps, key=lambda step: -1 if step is None else step)

    def test_the_9b_rung_is_the_flagship_and_the_only_rung_with_rollouts_or_a_verified_init(
        self,
    ) -> None:
        assert TMAX_9B.repo_id == FLAGSHIP_REPO_ID
        assert TMAX_9B.base_mirror_repo_id == BASE_CHECKPOINT.repo_id
        assert {rung.name for rung in SIZE_LADDER.values() if rung.has_released_rollouts} == {
            "tmax_9b"
        }
        assert {rung.name for rung in SIZE_LADDER.values() if rung.rl_init_verified} == {"tmax_9b"}
        assert "inferred" in TMAX_4B.base().note
        assert "rollout metadata" in TMAX_9B.base().note

    def test_the_27b_descends_from_qwen36_and_the_others_from_qwen35(self) -> None:
        assert TMAX_27B.upstream_base_repo_id == "Qwen/Qwen3.6-27B"
        assert TMAX_27B.base_mirror_repo_id == "hamishivi/Qwen3.6-27B"
        for rung in (TMAX_2B, TMAX_4B, TMAX_9B):
            assert rung.upstream_base_repo_id == f"Qwen/Qwen3.5-{rung.params_label}"
            assert rung.base_mirror_repo_id == f"hamishivi/Qwen3.5-{rung.params_label}"

    def test_every_released_rl_repo_is_distinct_across_both_axes(self) -> None:
        """Four rungs and seven suites share exactly one repo (the flagship 9B), nothing else."""
        rung_repos = {rung.repo_id for rung in SIZE_LADDER.values()}
        suite_repos = {checkpoint.repo_id for checkpoint in rl_models()}
        assert len(rung_repos) == 4
        assert rung_repos & suite_repos == {FLAGSHIP_REPO_ID}
        assert len(rung_repos | suite_repos) == 10
        assert [checkpoint.repo_id for checkpoint in rung_models()] == [
            rung.repo_id for rung in SIZE_LADDER.values()
        ]

    def test_the_weight_layouts_are_text_only_at_every_size(self) -> None:
        """Tensor counts from the safetensors headers: no vision tower, no MTP head, tied at 2B/4B."""
        assert [rung.weight_layout.tensor_count for rung in SIZE_LADDER.values()] == [
            320,
            426,
            427,
            851,
        ]
        assert [rung.weight_layout.has_lm_head for rung in SIZE_LADDER.values()] == [
            False,
            False,
            True,
            True,
        ]
        assert [rung.weight_layout.safetensors_files for rung in SIZE_LADDER.values()] == [
            1,
            1,
            1,
            2,
        ]
        assert TMAX_9B.weight_layout.bytes_per_branch == TMAX_9B_BYTES

    def test_the_stop_token_hazard_is_recorded_on_the_three_small_rungs(self) -> None:
        """A loader trusting the shipped file stops only at <|endoftext|>, never at the turn end."""
        for rung in (TMAX_2B, TMAX_4B, TMAX_9B):
            assert rung.generation_config is GenerationConfigShipped.MINIMAL_SINGLE_EOS
            assert rung.generation_config.eos_token_ids == (END_OF_TEXT_TOKEN_ID,)
            assert not rung.generation_config.stops_at_turn_end
        assert TMAX_27B.generation_config is GenerationConfigShipped.UPSTREAM_QWEN36_27B
        assert TMAX_27B.generation_config.stops_at_turn_end
        for rung in SIZE_LADDER.values():
            assert rung.stop_token_ids == CHAT_TURN_STOP_TOKEN_IDS
        assert CHAT_TURN_STOP_TOKEN_IDS == (IM_END_TOKEN_ID, END_OF_TEXT_TOKEN_ID)

    def test_the_template_families_and_thinking_defaults(self) -> None:
        assert TMAX_4B.template_family is ChatTemplateFamily.REPLAYS_PRIOR_REASONING
        assert TMAX_9B.template_family is ChatTemplateFamily.REPLAYS_PRIOR_REASONING
        assert TMAX_2B.template_family is ChatTemplateFamily.UPSTREAM_IDENTICAL
        assert TMAX_27B.template_family is ChatTemplateFamily.UPSTREAM_IDENTICAL
        assert {rung.name for rung in SIZE_LADDER.values() if not rung.thinking_on_by_default} == {
            "tmax_2b"
        }

    def test_the_8b_is_excluded_with_its_lineage_as_the_reason(self) -> None:
        """A lookup by either name has to say why rather than fall through to 'unknown'."""
        assert EXCLUDED_RELEASES == (TMAX_8B_EXCLUDED,)
        assert TMAX_8B_EXCLUDED.repo_id not in {rung.repo_id for rung in SIZE_LADDER.values()}
        for name in ("tmax_8b", "allenai/tmax-8b"):
            with pytest.raises(TmaxArtifactError, match="Qwen3ForCausalLM"):
                size_rung(name)

    def test_an_unknown_rung_name_raises_a_key_error(self) -> None:
        with pytest.raises(KeyError, match="unknown size rung"):
            size_rung("tmax_70b")

    def test_a_known_rung_comes_back_by_name(self) -> None:
        assert size_rung("tmax_27b") is TMAX_27B

    def test_a_rung_whose_main_step_names_no_branch_is_refused_at_construction(self) -> None:
        with pytest.raises(TmaxArtifactError, match="main_step 150 is not one of"):
            _rung_like(TMAX_2B, step_branches=("step_100", "step_200"), main_step=150)

    def test_a_rung_whose_branches_do_not_ascend_is_refused_at_construction(self) -> None:
        with pytest.raises(TmaxArtifactError, match="must ascend"):
            _rung_like(TMAX_2B, step_branches=("step_200", "step_100"), main_step=100)
        with pytest.raises(TmaxArtifactError, match="not of the form step_N"):
            _rung_like(TMAX_2B, step_branches=("main", "step_100"), main_step=100)


def _rung_like(template: SizeRung, *, step_branches: tuple[str, ...], main_step: int) -> SizeRung:
    """Copy a real rung with different branches, to reach ``__post_init__``'s refusals."""
    return replace(template, step_branches=step_branches, main_step=main_step)


def _listing_for(rung: SizeRung) -> dict[str, tuple[str, ...]]:
    """A hub listing in the verified shape: distinct ids per step, main equal to main_branch."""
    ids: dict[str, tuple[str, ...]] = {
        branch: (f"sha-{rung.name}-{branch}",) for branch in rung.step_branches
    }
    ids[DEFAULT_REVISION] = ids[rung.main_branch]
    return ids


class TestReconcilingARungWithAHubListing:
    """The registry's claim, run against a listing shaped like the API's (mocked, no network)."""

    @pytest.mark.parametrize("rung", list(SIZE_LADDER.values()), ids=list(SIZE_LADDER))
    def test_a_listing_in_the_verified_shape_passes_for_every_rung(self, rung: SizeRung) -> None:
        reconcile_branch_weight_ids(rung, _listing_for(rung))

    def test_the_27b_two_file_layout_passes_regardless_of_id_order(self) -> None:
        """Two shards per branch; the check sorts, so the listing's order must not matter."""
        listing: dict[str, tuple[str, ...]] = {
            branch: (f"b-{branch}", f"a-{branch}") for branch in TMAX_27B.step_branches
        }
        listing[DEFAULT_REVISION] = tuple(reversed(listing[TMAX_27B.main_branch]))
        reconcile_branch_weight_ids(TMAX_27B, listing)

    def test_main_pointing_at_a_different_step_is_refused(self) -> None:
        """The failure the whole check exists for: a release re-pointing main under us."""
        listing = _listing_for(TMAX_9B)
        listing[DEFAULT_REVISION] = listing["step_500"]
        with pytest.raises(TmaxArtifactError, match="registry says main == step_200"):
            reconcile_branch_weight_ids(TMAX_9B, listing)

    def test_an_unregistered_branch_on_the_hub_is_refused(self) -> None:
        listing = _listing_for(TMAX_2B)
        listing["step_400"] = ("sha-new",)
        with pytest.raises(TmaxArtifactError, match=r"unregistered=\['step_400'\]"):
            reconcile_branch_weight_ids(TMAX_2B, listing)

    def test_a_registered_branch_missing_from_the_hub_is_refused(self) -> None:
        listing = _listing_for(TMAX_4B)
        del listing["step_380"]
        with pytest.raises(TmaxArtifactError, match=r"missing=\['step_380'\]"):
            reconcile_branch_weight_ids(TMAX_4B, listing)

    def test_two_step_branches_sharing_weights_is_refused(self) -> None:
        listing = _listing_for(TMAX_27B)
        listing["step_240"] = listing["step_300"]
        with pytest.raises(TmaxArtifactError, match=r"\['step_240', 'step_300'\] share"):
            reconcile_branch_weight_ids(TMAX_27B, listing)


class TestLanguageModelKeyAlignment:
    """The subset a base-vs-RL weight diff aligns on: text only, and nothing outside the family."""

    def test_language_model_and_head_keys_are_kept_and_towers_dropped(self) -> None:
        assert is_language_model_key("model.language_model.layers.0.mlp.up_proj.weight")
        assert is_language_model_key("lm_head.weight")
        assert not is_language_model_key("model.visual.blocks.0.attn.qkv.weight")
        assert not is_language_model_key("mtp.layers.0.mlp.gate_proj.weight")

    def test_a_base_mirror_key_set_reduces_to_the_released_subset_in_order(self) -> None:
        """What the 9B mirror stores (426 + 333 + 15 + head) against what tmax-9b stores."""
        base_keys = [
            "model.language_model.embed_tokens.weight",
            "model.visual.patch_embed.proj.weight",
            "model.language_model.layers.0.linear_attn.A_log",
            "mtp.fc.weight",
            "lm_head.weight",
        ]
        assert language_model_keys(base_keys) == (
            "model.language_model.embed_tokens.weight",
            "model.language_model.layers.0.linear_attn.A_log",
            "lm_head.weight",
        )

    def test_a_qwen3_layout_key_raises_and_names_the_excluded_8b(self) -> None:
        """tmax-8b stores plain model.layers.*; a diff that silently kept nothing would read zero."""
        with pytest.raises(TmaxArtifactError, match="allenai/tmax-8b"):
            language_model_keys(["model.embed_tokens.weight", "model.layers.0.mlp.up_proj.weight"])

    def test_a_key_set_that_is_all_dropped_towers_raises_rather_than_returning_nothing(
        self,
    ) -> None:
        with pytest.raises(TmaxArtifactError, match="refusing to diff over nothing"):
            language_model_keys(["model.visual.merger.mlp.0.weight", "mtp.norm.weight"])


class TestTheTrainingRecipe:
    def test_full_parameter_dppo_with_the_group_shape_the_shard_shows(self) -> None:
        assert TMAX_TRAINING_RECIPE.full_parameter
        assert TMAX_TRAINING_RECIPE.algorithm == "DPPO"
        assert (
            TMAX_TRAINING_RECIPE.prompts_per_step * TMAX_TRAINING_RECIPE.samples_per_prompt == 256
        )
        assert TMAX_TRAINING_RECIPE.planned_steps == max(
            int(branch.removeprefix("step_")) for branch in FLAGSHIP_STEP_BRANCHES
        )
        assert TMAX_27B.step_branches[-1] == "step_300"


class TestTheRolloutArchiveInventory:
    """The 2026-09-02 inventory of allenai/tmax-9b's rollouts/ folder, as constants."""

    def test_the_hub_footprint_and_decompressed_totals(self) -> None:
        assert FLAGSHIP_ROLLOUTS.hub_files == 25
        assert FLAGSHIP_ROLLOUTS.hub_bytes == 44_059_163_698
        assert FLAGSHIP_ROLLOUTS.decompressed_rollout_files == 17
        assert FLAGSHIP_ROLLOUTS.decompressed_logprob_files == 8400
        assert FLAGSHIP_ROLLOUTS.producing_model_repo_id == BASE_CHECKPOINT.repo_id
        assert FLAGSHIP_ROLLOUTS.training_git_commit == "63305abed"

    def test_seven_fragments_whose_file_counts_sum_to_the_manifest_totals(self) -> None:
        assert len(FLAGSHIP_ROLLOUT_FRAGMENTS) == 7
        assert FLAGSHIP_ROLLOUTS.fragments == FLAGSHIP_ROLLOUT_FRAGMENTS
        assert sum(fragment.rollout_files for fragment in FLAGSHIP_ROLLOUT_FRAGMENTS) == 17
        assert (
            sum(fragment.training_logprob_files for fragment in FLAGSHIP_ROLLOUT_FRAGMENTS) == 8400
        )
        assert [fragment.is_empty for fragment in FLAGSHIP_ROLLOUT_FRAGMENTS].count(True) == 1

    def test_the_step_ranges_cover_1_to_500_with_the_documented_overlaps(self) -> None:
        ranges = [
            (fragment.trainer_step_min, fragment.trainer_step_max)
            for fragment in FLAGSHIP_ROLLOUT_FRAGMENTS
            if not fragment.is_empty
        ]
        assert ranges == [(1, 98), (91, 94), (91, 292), (291, 353), (351, 368), (361, 500)]
        covered = {
            step
            for low, high in ranges
            if low is not None and high is not None
            for step in range(low, high + 1)
        }
        assert covered == set(range(1, 501))

    def test_a_fragment_is_found_by_name_and_an_unknown_one_raises(self) -> None:
        small = FLAGSHIP_ROLLOUTS.fragment("swerl_qwen35_9b_fp32lm_dppo_g32__42__1779685677")
        assert (small.trainer_step_min, small.trainer_step_max) == (91, 94)
        with pytest.raises(KeyError, match="unknown rollout fragment"):
            FLAGSHIP_ROLLOUTS.fragment("swerl_qwen35_9b_fp32lm_dppo_g32__42__0")

    def test_the_reassembly_pipeline_is_the_one_the_release_documents(self) -> None:
        fragment = FLAGSHIP_ROLLOUTS.fragment("swerl_qwen35_9b_fp32lm_dppo_g32__42__1779685677")
        assert FLAGSHIP_ROLLOUTS.reassemble_command(fragment, RolloutKind.ROLLOUTS) == (
            "cat rollouts/archives/swerl_qwen35_9b_fp32lm_dppo_g32__42__1779685677/rollouts/"
            "rollouts.tar.zst.part-* | zstd -d | tar -xvf -"
        )
        assert FLAGSHIP_ROLLOUTS.archive_dir(fragment, RolloutKind.TRAINING_LOGPROBS).endswith(
            "/training_logprobs"
        )

    def test_the_rollouts_only_patterns_select_every_fragment_s_transcripts(self) -> None:
        """The download filter and the inventory must agree on where transcripts live."""
        for fragment in FLAGSHIP_ROLLOUT_FRAGMENTS:
            path = FLAGSHIP_ROLLOUTS.archive_dir(fragment, RolloutKind.ROLLOUTS)
            assert path.startswith("rollouts/archives/")
            assert path.endswith("/rollouts")


class TestStageClassification:
    """The ETL tags each rollout base or RL off its producing repo, and never guesses."""

    def test_a_base_repo_id_is_base(self) -> None:
        assert stage_of("hamishivi/Qwen3.5-9B") is CheckpointStage.BASE
        assert stage_of(UPSTREAM_BASE.repo_id or "") is CheckpointStage.BASE

    def test_every_size_rung_base_upstream_or_mirror_is_base(self) -> None:
        for rung in SIZE_LADDER.values():
            assert stage_of(rung.upstream_base_repo_id) is CheckpointStage.BASE
            assert stage_of(rung.base_mirror_repo_id) is CheckpointStage.BASE

    def test_every_registered_rl_model_is_rl(self) -> None:
        for checkpoint in (*rl_models(), *rung_models()):
            assert stage_of(checkpoint.repo_id or "") is CheckpointStage.RL

    def test_an_unknown_repo_id_raises_rather_than_landing_on_a_side(self) -> None:
        """Guessing a stage would silently move rollouts between the two arms of the comparison."""
        with pytest.raises(KeyError, match="neither a known base nor a known RL checkpoint"):
            stage_of("meta-llama/Llama-3-8B")

    def test_the_excluded_8b_is_not_quietly_an_rl_checkpoint(self) -> None:
        with pytest.raises(KeyError, match="neither a known base nor a known RL checkpoint"):
            stage_of(TMAX_8B_EXCLUDED.repo_id)

    def test_an_unknown_suite_name_raises(self) -> None:
        with pytest.raises(KeyError, match="unknown suite"):
            suite("terminal_bench_2")

    def test_a_known_suite_comes_back_by_name(self) -> None:
        assert suite("cli_gym").display_name == "CLI-Gym"
