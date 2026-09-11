"""Registry of the released TMAX artifacts the Stage-A re-analysis runs against.

TMAX (Ivison et al., arXiv:2606.23321, "Tmax: A simple recipe for terminal agents", UW + Ai2) is an
open RL recipe and dataset for terminal (command-line) agents. Its headline is Terminal-Bench
performance; a single appendix (D.6) reports a *reward-hacking* finding that this repo cares about:
after RL, the 9B model displays hacks -- a no-op checker stub, a fake binary earlier on ``PATH``,
fabricated training artifacts -- that the base Qwen3.5-9B never showed, and all of them scored 0.

What makes TMAX close to an experiment built for us: the same RL recipe was applied to Qwen3.5-9B on
each of seven terminal-agent suites (six prior-work datasets plus TMAX's own generated mixture), and
for the flagship mixture arm Ai2 released the matched base checkpoint, intermediate checkpoints
through training, and the training rollouts. Stage A re-analyses those artifacts inference-only.

This module is the single place the Hugging Face repo ids live. Two rules it enforces:

* **Descriptive names, never opaque labels.** Each RL'd model is named by the environment suite it
  was trained on (``endless_terminals``, ``cli_gym``, ...), not ``arm_a``/``arm_b``.
* **Never trust an unverified id.** Every id below was confirmed against the HF API (an HTTP 200, a
  tree listing, or a safetensors header read) unless its ``verified`` flag is ``False``. An unverified
  or unset id must fail loudly the instant code tries to resolve it, via :class:`TmaxArtifactError`,
  rather than 404-ing deep inside a multi-gigabyte download. The two unverified placeholders in
  :data:`UNVERIFIED_ARTIFACTS` are real gaps the research flagged, kept here so the code that needs
  them raises a message that says exactly what to confirm.

The registry has two axes, and they are different releases of the same recipe:

* **The suite axis** (:data:`SUITES`): seven Qwen3.5-9B checkpoints, one per environment suite.
  Intermediate checkpoints and released rollouts exist ONLY for the flagship ``tmax_15k`` arm
  (``allenai/tmax-9b``); the six single-suite ablations each expose one final checkpoint and no
  rollouts, so the dose-response ladder (experiment 1) runs only on the flagship while the seven-way
  looseness test (experiment 2) compares the six ablations' finals against the flagship.
* **The size axis** (:data:`SIZE_LADDER`, verified 2026-09-02): the flagship mixture recipe at four
  sizes, 2B / 4B / 9B / 27B, each with its own step branches and each ``main`` a re-pointing of one
  step branch. Only the 9B carries rollouts. The 27B is a Qwen3.6 descendant, the other three are
  Qwen3.5. ``allenai/tmax-8b`` exists but is a different lineage (Qwen3, SFT-then-RL) and is recorded
  as excluded rather than as a rung, in :data:`EXCLUDED_RELEASES`.

Four facts about the release that every loader and weight diff has to know, each carried by the
type that enforces it (verified 2026-09-02 against the hub's safetensors headers, config files and
one downloaded rollout shard; labelled "paper" where they come from the paper instead):

* **Every checkpoint is full-parameter; there are no LoRA adapters anywhere** (each branch is a
  complete safetensors set, same byte total, distinct object id; :data:`TMAX_TRAINING_RECIPE`).
* **Released weights are text-only** while ``config.json`` still declares the vision tower, so a
  base-vs-RL diff aligns on :func:`language_model_keys` and loaders take the text-only path.
* **The 2B/4B/9B generation config names the wrong stop token**; pass
  :data:`CHAT_TURN_STOP_TOKEN_IDS` explicitly (:class:`GenerationConfigShipped`).
* **The 4B and 9B chat templates replay prior-turn reasoning**, unlike upstream
  (:class:`ChatTemplateFamily`); tokenizer configs are identical to upstream at every size.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

# The revision that means "the packaged release branch"; each rung's main re-points one step branch.
DEFAULT_REVISION = "main"


class TmaxArtifactError(RuntimeError):
    """Raised when code tries to resolve a TMAX artifact whose HF id was never verified.

    The whole point is to fail at the call site with a message naming what to confirm, rather than
    letting an unverified or ``None`` repo id reach the hub and 404 partway through a download.
    """


class CheckpointStage(StrEnum):
    """Which side of the RL boundary a checkpoint sits on, for the base-vs-RL comparison.

    ``BASE`` is the matched starting checkpoint (RL step 0); ``RL`` is anything the recipe trained,
    whether an intermediate step branch or a final single-suite model.
    """

    BASE = "base"
    RL = "rl"


@dataclass(frozen=True)
class Checkpoint:
    """One model checkpoint: a HF repo id plus the git revision that pins it.

    ``rl_step`` is 0 for a base checkpoint, the step count for a flagship step branch, and ``None``
    for a single-suite final model whose step count the release does not pin. ``verified`` guards
    :meth:`resolve`: an unverified checkpoint raises rather than being loaded.

    ``safetensors_bytes`` is the hub-reported byte total of the checkpoint's safetensors (verified
    2026-09-02 where set); the download CLI checks a landed snapshot against it, which is what tells
    "the fetch returned the weights" from a redirect stub or an interrupted write, both of which return.
    """

    name: str
    repo_id: str | None
    revision: str = DEFAULT_REVISION
    stage: CheckpointStage = CheckpointStage.RL
    rl_step: int | None = None
    verified: bool = True
    safetensors_bytes: int | None = None
    note: str = ""

    def resolve(self) -> tuple[str, str]:
        """Return ``(repo_id, revision)`` to hand to transformers/hub, or raise if unverified."""
        if self.repo_id is None or not self.verified:
            raise TmaxArtifactError(
                f"checkpoint {self.name!r} has an unverified or unset repo id "
                f"(repo_id={self.repo_id!r}, verified={self.verified}); confirm it on the HF hub "
                f"and set verified=True before loading it. {self.note}".rstrip()
            )
        return self.repo_id, self.revision

    @property
    def model_ref(self) -> str:
        """A single-string reference (``repo`` or ``repo@revision``) for logs and result tables."""
        repo_id, revision = self.resolve()
        return repo_id if revision == DEFAULT_REVISION else f"{repo_id}@{revision}"


@dataclass(frozen=True)
class HFDataset:
    """One Hugging Face dataset repo (an RL environment suite as data)."""

    name: str
    repo_id: str | None
    verified: bool = True
    note: str = ""

    def resolve(self) -> str:
        """Return the repo id for ``datasets.load_dataset``/``snapshot_download``, or raise."""
        if self.repo_id is None or not self.verified:
            raise TmaxArtifactError(
                f"dataset {self.name!r} has an unverified or unset repo id "
                f"(repo_id={self.repo_id!r}, verified={self.verified}); confirm it before use. "
                f"{self.note}".rstrip()
            )
        return self.repo_id


# --------------------------------------------------------------------------------------
# Special tokens and the stop-token hazard (Qwen3.5 / Qwen3.6 vocabulary, verified 2026-09-02)
# --------------------------------------------------------------------------------------

END_OF_TEXT_TOKEN_ID = 248044  # <|endoftext|>: the only eos the 2B/4B/9B generation_config names
IM_START_TOKEN_ID = 248045  # <|im_start|>
IM_END_TOKEN_ID = 248046  # <|im_end|>: how every chat turn actually ends
THINK_OPEN_TOKEN_ID = 248068  # <think>

# Pass these as stop ids for offline inference on any rung; see GenerationConfigShipped.
CHAT_TURN_STOP_TOKEN_IDS: tuple[int, ...] = (IM_END_TOKEN_ID, END_OF_TEXT_TOKEN_ID)


class GenerationConfigShipped(StrEnum):
    """Which ``generation_config.json`` a rung ships, and therefore what a naive loader stops on.

    ``MINIMAL_SINGLE_EOS`` (tmax-2b, -4b, -9b): a transformers-emitted stub naming only
    ``<|endoftext|>`` (248044) and no sampling parameters; upstream Qwen3.5 ships no generation
    config, so this file is TMAX's, and chat turns end in ``<|im_end|>`` (248046), which it lacks,
    so a generation that trusts it runs on into a fabricated next turn. ``UPSTREAM_QWEN36_27B``
    (tmax-27b) is Qwen3.6-27B's own file: eos [248046, 248044], do_sample, temperature 1.0, top_k
    20, top_p 0.95, which stops correctly.
    """

    MINIMAL_SINGLE_EOS = "minimal_single_eos"
    UPSTREAM_QWEN36_27B = "upstream_qwen36_27b"

    @property
    def eos_token_ids(self) -> tuple[int, ...]:
        """The eos ids the shipped file declares, in the file's order."""
        if self is GenerationConfigShipped.MINIMAL_SINGLE_EOS:
            return (END_OF_TEXT_TOKEN_ID,)
        return (IM_END_TOKEN_ID, END_OF_TEXT_TOKEN_ID)

    @property
    def stops_at_turn_end(self) -> bool:
        """Whether a loader that trusts the shipped file stops where the assistant turn ends."""
        return IM_END_TOKEN_ID in self.eos_token_ids


class ChatTemplateFamily(StrEnum):
    """How a rung's ``chat_template.jinja`` relates to its upstream base (diffed 2026-09-02).

    ``UPSTREAM_IDENTICAL``: byte-identical to the base's template (tmax-2b vs Qwen3.5-2B, tmax-27b
    vs Qwen3.6-27B). ``REPLAYS_PRIOR_REASONING``: one line differs from upstream -- the assistant
    reasoning block is emitted when ``reasoning_content`` is present instead of only for turns after
    ``ns.last_query_index`` -- so a multi-turn prompt replays earlier turns' thinking rather than
    stripping it (tmax-4b, tmax-9b, and the 9B base mirror ``hamishivi/Qwen3.5-9B`` share this
    template byte-for-byte), so a rollout re-rendered through the upstream template tokenizes
    differently from what the policy saw.
    """

    UPSTREAM_IDENTICAL = "upstream_identical"
    REPLAYS_PRIOR_REASONING = "replays_prior_reasoning"


# --------------------------------------------------------------------------------------
# Weight layout: what the released safetensors hold (headers read 2026-09-02)
# --------------------------------------------------------------------------------------

LANGUAGE_MODEL_KEY_PREFIX = "model.language_model."
LM_HEAD_KEY = "lm_head.weight"
# The towers the release drops; the base mirrors still carry them (333 visual + 15 mtp tensors).
DROPPED_TOWER_KEY_PREFIXES: tuple[str, ...] = ("model.visual.", "mtp.")


def is_language_model_key(key: str) -> bool:
    """Whether a safetensors key belongs to the text-only subset the TMAX release ships."""
    return key.startswith(LANGUAGE_MODEL_KEY_PREFIX) or key == LM_HEAD_KEY


def language_model_keys(keys: Iterable[str]) -> tuple[str, ...]:
    """Reduce a checkpoint's safetensors keys to the language-model subset, in the given order.

    The alignment a base-vs-RL weight diff has to make: the base mirror carries the vision tower
    and MTP head, the RL'd checkpoint does not, and a diff over the union would error on missing
    keys or count the dropped towers as change. A key that is neither raises, because it means a
    checkpoint outside the Qwen3.5/3.6 layout (``allenai/tmax-8b`` stores plain ``model.layers.*``)
    and an empty result would let a diff over nothing report zero change.
    """
    kept: list[str] = []
    for key in keys:
        if is_language_model_key(key):
            kept.append(key)
        elif not key.startswith(DROPPED_TOWER_KEY_PREFIXES):
            raise TmaxArtifactError(
                f"safetensors key {key!r} is neither a Qwen3.5 language-model key "
                f"({LANGUAGE_MODEL_KEY_PREFIX}* / {LM_HEAD_KEY}) nor a dropped tower "
                f"{DROPPED_TOWER_KEY_PREFIXES}; this checkpoint is outside the TMAX size ladder "
                f"(allenai/tmax-8b stores plain model.layers.* and is excluded for that reason)"
            )
    if not kept:
        raise TmaxArtifactError("no language-model keys found; refusing to diff over nothing")
    return tuple(kept)


@dataclass(frozen=True)
class WeightLayout:
    """What one rung's safetensors hold, read from the hub's headers on 2026-09-02.

    ``language_model_tensors`` counts ``model.language_model.*`` keys; ``has_lm_head`` is false where
    embeddings are tied; ``bytes_per_branch`` is the byte total, identical on every branch.
    """

    safetensors_files: int
    language_model_tensors: int
    has_lm_head: bool
    bytes_per_branch: int

    @property
    def tensor_count(self) -> int:
        """Total stored tensors: the language model plus the head where it is untied."""
        return self.language_model_tensors + int(self.has_lm_head)


# --------------------------------------------------------------------------------------
# The training recipe (paper facts, with the two the shard confirms marked)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingRecipe:
    """The RL recipe every TMAX checkpoint came from, per the paper (arXiv:2606.23321).

    ``prompts_per_step`` x ``samples_per_prompt`` is confirmed in the released rollouts (256 rows per
    step, ``prompt_idx`` 0-7, 32 samples each); ``planned_steps`` matches the rollout manifests'
    maximum trainer step. ``full_parameter`` is confirmed by the release itself: every branch is a
    complete safetensors set and no repo carries an adapter. The 27B was trained to 300 steps; each
    rung's :attr:`SizeRung.step_branches` says where its release ends.
    """

    algorithm: str = "DPPO"
    learning_rate: float = 1e-6
    kl_coefficient: float = 0.0
    prompts_per_step: int = 8
    samples_per_prompt: int = 32
    planned_steps: int = 500
    full_parameter: bool = True


TMAX_TRAINING_RECIPE = TrainingRecipe()


# --------------------------------------------------------------------------------------
# Base / matched-start checkpoints (RL step 0)
# --------------------------------------------------------------------------------------

# Upstream base safetensors byte totals; each hamishivi mirror is the same files (sha256 ids match).
QWEN35_2B_BASE_BYTES = 4_548_221_488
QWEN35_4B_BASE_BYTES = 9_319_828_096
QWEN35_9B_BASE_BYTES = 19_306_310_880
QWEN36_27B_BASE_BYTES = 55_563_006_400

# The RL init: the rollout metadata's model_name, a re-upload of the instruct model (see .note).
BASE_CHECKPOINT = Checkpoint(
    name="base_qwen35_9b_instruct",
    repo_id="hamishivi/Qwen3.5-9B",
    stage=CheckpointStage.BASE,
    rl_step=0,
    safetensors_bytes=QWEN35_9B_BASE_BYTES,
    note=(
        "The RL init per every TMAX model card and per the rollout metadata; byte-identical to "
        "Qwen/Qwen3.5-9B (same safetensors object ids), so the two are interchangeable as weights."
    ),
)

# Upstream references from the HF model tree; the interp-transfer caveat is in UPSTREAM_BASE.note.
UPSTREAM_INSTRUCT = Checkpoint(
    name="qwen35_9b_instruct_upstream",
    repo_id="Qwen/Qwen3.5-9B",
    stage=CheckpointStage.BASE,
    rl_step=0,
    safetensors_bytes=QWEN35_9B_BASE_BYTES,
    note="Upstream instruct base per the model tree; same weights as the hamishivi mirror.",
)
UPSTREAM_BASE = Checkpoint(
    name="qwen35_9b_base_upstream",
    repo_id="Qwen/Qwen3.5-9B-Base",
    stage=CheckpointStage.BASE,
    rl_step=0,
    note=(
        "SAE + Jacobian lens are fit here; TMAX descends from the instruct line "
        "-- measure transfer."
    ),
)

BASE_CHECKPOINTS: tuple[Checkpoint, ...] = (BASE_CHECKPOINT, UPSTREAM_INSTRUCT, UPSTREAM_BASE)


# --------------------------------------------------------------------------------------
# The flagship (TMAX-15K mixture) arm and its intermediate-checkpoint ladder
# --------------------------------------------------------------------------------------

# Released as git branches of allenai/tmax-9b; main == step_200 by safetensors object id.
FLAGSHIP_STEP_BRANCHES: tuple[str, ...] = (
    "step_100",
    "step_200",
    "step_300",
    "step_400",
    "step_500",
)

FLAGSHIP_REPO_ID = "allenai/tmax-9b"

# Every 9B checkpoint in the release -- flagship branches and the six ablations -- has this byte total.
TMAX_9B_BYTES = 17_907_663_008

TMAX_15K_DATASET = HFDataset(
    name="tmax_15k",
    repo_id="allenai/TMax-15K",
    note="14,601 generated RL environments (parquet, single train split); odc-by, ungated.",
)

# The id the model cards cite; registered beside TMax-15K, not aliased to it (see .note).
TMAX_15K_OPEN_INSTRUCT_DATASET = HFDataset(
    name="tmax_15k_open_instruct",
    repo_id="allenai/tmax-15k-open-instruct",
    note="The dataset id the tmax-* model cards cite; row-equality with allenai/TMax-15K unchecked.",
)

# SFT-trace datasets in the same collection: not RL data, so nothing here resolves them.
SFT_TRACE_DATASETS_OUT_OF_SCOPE: tuple[str, ...] = (
    "allenai/tmax-sft",
    "allenai/TMax-SFT-16.5K",
    "allenai/tmax-sft-big",
)

FLAGSHIP_MODEL = Checkpoint(
    name="tmax_15k_flagship",
    repo_id=FLAGSHIP_REPO_ID,
    revision=DEFAULT_REVISION,
    stage=CheckpointStage.RL,
    rl_step=200,
    safetensors_bytes=TMAX_9B_BYTES,
    note="main branch == step_200 by object id (best on TB-Lite); carries rollouts/ + step branches.",
)


def dose_response_ladder() -> tuple[Checkpoint, ...]:
    """Return the base -> step_100..step_500 checkpoint ladder for experiment 1 (flagship only).

    Step 0 is the shared base checkpoint; the rest are the flagship's released step branches. The
    ``main`` branch is intentionally omitted from the ladder because it duplicates ``step_200``.
    """
    steps = tuple(
        Checkpoint(
            name=f"tmax_15k_{branch}",
            repo_id=FLAGSHIP_REPO_ID,
            revision=branch,
            stage=CheckpointStage.RL,
            rl_step=int(branch.removeprefix("step_")),
            safetensors_bytes=TMAX_9B_BYTES,
            note="Intermediate flagship checkpoint (git branch of allenai/tmax-9b).",
        )
        for branch in FLAGSHIP_STEP_BRANCHES
    )
    return (BASE_CHECKPOINT, *steps)


# --------------------------------------------------------------------------------------
# The size ladder: the flagship recipe at 2B / 4B / 9B / 27B (verified 2026-09-02)
# --------------------------------------------------------------------------------------


def _step_of(branch: str) -> int:
    """Parse ``step_N`` to ``N``, refusing any other branch shape."""
    if not branch.startswith("step_") or not branch.removeprefix("step_").isdigit():
        raise TmaxArtifactError(f"branch {branch!r} is not of the form step_N")
    return int(branch.removeprefix("step_"))


@dataclass(frozen=True)
class SizeRung:
    """One size of the flagship recipe: its repo, base, branches, and the loader hazards.

    ``main_step`` is the step branch ``main`` duplicates, by safetensors object id rather than by
    the card (2B step_100; 4B and 9B step_200; 27B step_160). ``upstream_base_repo_id`` is the
    card's ``base_model``; ``base_mirror_repo_id`` is the byte-identical re-upload. Only the 9B's
    RL init is *verified* as the mirror (the rollout metadata names it); ``rl_init_verified`` says
    so, and since mirror and upstream are the same bytes the distinction is provenance only.

    ``thinking_on_by_default`` is the template's ``enable_thinking`` default: every rung thinks
    unless told not to, except the 2B, whose upstream template thinks only when told to.
    """

    name: str
    params_label: str
    repo_id: str
    upstream_base_repo_id: str
    base_mirror_repo_id: str
    base_safetensors_bytes: int
    step_branches: tuple[str, ...]
    main_step: int
    template_family: ChatTemplateFamily
    thinking_on_by_default: bool
    generation_config: GenerationConfigShipped
    weight_layout: WeightLayout
    rl_init_verified: bool
    has_released_rollouts: bool
    note: str = ""

    def __post_init__(self) -> None:
        """Refuse a rung whose branches do not ascend or whose main_step names no branch."""
        steps = [_step_of(branch) for branch in self.step_branches]
        if steps != sorted(steps) or len(set(steps)) != len(steps):
            raise TmaxArtifactError(f"{self.name}: step branches must ascend, got {steps}")
        if self.main_step not in steps:
            raise TmaxArtifactError(
                f"{self.name}: main_step {self.main_step} is not one of the step branches {steps}"
            )

    @property
    def main_branch(self) -> str:
        """The step branch whose weights ``main`` re-points."""
        return f"step_{self.main_step}"

    @property
    def stop_token_ids(self) -> tuple[int, ...]:
        """The stop ids to pass explicitly, whatever the shipped generation config says."""
        return CHAT_TURN_STOP_TOKEN_IDS

    def base(self) -> Checkpoint:
        """Return the RL-init checkpoint (the base mirror), step 0 of this rung's ladder."""
        provenance = "named by the rollout metadata" if self.rl_init_verified else "inferred"
        return Checkpoint(
            name=f"{self.name}_base",
            repo_id=self.base_mirror_repo_id,
            stage=CheckpointStage.BASE,
            rl_step=0,
            safetensors_bytes=self.base_safetensors_bytes,
            note=(
                f"RL init for {self.repo_id} ({provenance}); byte-identical to "
                f"{self.upstream_base_repo_id}."
            ),
        )

    def checkpoint(self, revision: str = DEFAULT_REVISION) -> Checkpoint:
        """Return the RL'd checkpoint at ``revision`` (``main`` or one of the step branches)."""
        if revision == DEFAULT_REVISION:
            rl_step = self.main_step
        elif revision in self.step_branches:
            rl_step = _step_of(revision)
        else:
            raise TmaxArtifactError(
                f"{self.repo_id} has no branch {revision!r}; released branches are "
                f"{(DEFAULT_REVISION, *self.step_branches)}"
            )
        return Checkpoint(
            name=f"{self.name}_{revision}",
            repo_id=self.repo_id,
            revision=revision,
            stage=CheckpointStage.RL,
            rl_step=rl_step,
            safetensors_bytes=self.weight_layout.bytes_per_branch,
            note=f"{self.params_label} flagship-recipe checkpoint; main == {self.main_branch}.",
        )

    def ladder(self) -> tuple[Checkpoint, ...]:
        """Return base first, then every step branch ascending; ``main`` is a duplicate rung."""
        return (self.base(), *(self.checkpoint(branch) for branch in self.step_branches))


TMAX_2B = SizeRung(
    name="tmax_2b",
    params_label="2B",
    repo_id="allenai/tmax-2b",
    upstream_base_repo_id="Qwen/Qwen3.5-2B",
    base_mirror_repo_id="hamishivi/Qwen3.5-2B",
    base_safetensors_bytes=QWEN35_2B_BASE_BYTES,
    step_branches=("step_100", "step_200", "step_300"),
    main_step=100,
    template_family=ChatTemplateFamily.UPSTREAM_IDENTICAL,
    thinking_on_by_default=False,
    generation_config=GenerationConfigShipped.MINIMAL_SINGLE_EOS,
    weight_layout=WeightLayout(
        safetensors_files=1,
        language_model_tensors=320,
        has_lm_head=False,
        bytes_per_branch=3_763_692_048,
    ),
    rl_init_verified=False,
    has_released_rollouts=False,
    note="Tied embeddings (no lm_head). Template identical to Qwen3.5-2B, which thinks only on request.",
)

TMAX_4B = SizeRung(
    name="tmax_4b",
    params_label="4B",
    repo_id="allenai/tmax-4b",
    upstream_base_repo_id="Qwen/Qwen3.5-4B",
    base_mirror_repo_id="hamishivi/Qwen3.5-4B",
    base_safetensors_bytes=QWEN35_4B_BASE_BYTES,
    step_branches=("step_100", "step_200", "step_300", "step_380"),
    main_step=200,
    template_family=ChatTemplateFamily.REPLAYS_PRIOR_REASONING,
    thinking_on_by_default=True,
    generation_config=GenerationConfigShipped.MINIMAL_SINGLE_EOS,
    weight_layout=WeightLayout(
        safetensors_files=1,
        language_model_tensors=426,
        has_lm_head=False,
        bytes_per_branch=8_411_558_400,
    ),
    rl_init_verified=False,
    has_released_rollouts=False,
    note="Tied embeddings (no lm_head); the same 426 language-model tensors as the 9B.",
)

TMAX_9B = SizeRung(
    name="tmax_9b",
    params_label="9B",
    repo_id=FLAGSHIP_REPO_ID,
    upstream_base_repo_id="Qwen/Qwen3.5-9B",
    base_mirror_repo_id="hamishivi/Qwen3.5-9B",
    base_safetensors_bytes=QWEN35_9B_BASE_BYTES,
    step_branches=FLAGSHIP_STEP_BRANCHES,
    main_step=200,
    template_family=ChatTemplateFamily.REPLAYS_PRIOR_REASONING,
    thinking_on_by_default=True,
    generation_config=GenerationConfigShipped.MINIMAL_SINGLE_EOS,
    weight_layout=WeightLayout(
        safetensors_files=1,
        language_model_tensors=426,
        has_lm_head=True,
        bytes_per_branch=TMAX_9B_BYTES,
    ),
    rl_init_verified=True,
    has_released_rollouts=True,
    note="The flagship arm of the suite axis; the only rung with rollouts/ (main branch only).",
)

TMAX_27B = SizeRung(
    name="tmax_27b",
    params_label="27B",
    repo_id="allenai/tmax-27b",
    upstream_base_repo_id="Qwen/Qwen3.6-27B",
    base_mirror_repo_id="hamishivi/Qwen3.6-27B",
    base_safetensors_bytes=QWEN36_27B_BASE_BYTES,
    step_branches=("step_100", "step_160", "step_200", "step_240", "step_300"),
    main_step=160,
    template_family=ChatTemplateFamily.UPSTREAM_IDENTICAL,
    thinking_on_by_default=True,
    generation_config=GenerationConfigShipped.UPSTREAM_QWEN36_27B,
    weight_layout=WeightLayout(
        safetensors_files=2,
        language_model_tensors=850,
        has_lm_head=True,
        bytes_per_branch=53_792_110_192,
    ),
    rl_init_verified=False,
    has_released_rollouts=False,
    note=(
        "Qwen3.6 lineage, trained to 300 steps. The card's 'Finetuned from ... Qwen 3.5 9B' line is a "
        "copy-paste error; its base_model metadata and body say Qwen3.6-27B. Ships "
        ".eval_results/terminal-bench-2.0.yaml at 42.7."
    ),
)

SIZE_LADDER: dict[str, SizeRung] = {
    rung.name: rung for rung in (TMAX_2B, TMAX_4B, TMAX_9B, TMAX_27B)
}


@dataclass(frozen=True)
class ExcludedRelease:
    """A repo in the TMAX release that is deliberately NOT a rung, and why."""

    name: str
    repo_id: str
    reason: str


TMAX_8B_EXCLUDED = ExcludedRelease(
    name="tmax_8b",
    repo_id="allenai/tmax-8b",
    reason=(
        "Qwen3ForCausalLM lineage, not Qwen3.5: vocab 151,936, 36 layers, plain model.layers.* keys "
        "(399 tensors, no model.language_model.* namespace), trained SFT-then-RL from "
        "allenai/tmax-sft-8b, itself from Qwen/Qwen3-8B. A different family and a different "
        "recipe, so it is neither a size rung nor a same-recipe comparison point."
    ),
)

EXCLUDED_RELEASES: tuple[ExcludedRelease, ...] = (TMAX_8B_EXCLUDED,)


def size_rung(name: str) -> SizeRung:
    """Look up one rung by name, naming the exclusion reason if the name is an excluded release."""
    if name in SIZE_LADDER:
        return SIZE_LADDER[name]
    for excluded in EXCLUDED_RELEASES:
        if name in (excluded.name, excluded.repo_id):
            raise TmaxArtifactError(
                f"{excluded.repo_id} is excluded from the size ladder: {excluded.reason}"
            )
    raise KeyError(f"unknown size rung {name!r}; known rungs are {sorted(SIZE_LADDER)}")


def reconcile_branch_weight_ids(
    rung: SizeRung, branch_weight_ids: Mapping[str, Sequence[str]]
) -> None:
    """Check a live hub listing of ``{branch: sorted safetensors object ids}`` against the registry.

    The registry's claim about a rung, restated as a pre-fetch check: every registered branch is
    present and no unregistered one has appeared; ``main`` carries ``main_branch``'s weights; every
    step branch differs from every other. A release that re-points ``main`` or adds a branch makes
    the ladder here silently wrong; this makes it loud. Raises :class:`TmaxArtifactError`.
    """
    registered = {DEFAULT_REVISION, *rung.step_branches}
    listed = set(branch_weight_ids)
    if registered != listed:
        raise TmaxArtifactError(
            f"{rung.repo_id}: registered branches {sorted(registered)} do not match the hub's "
            f"{sorted(listed)} (missing={sorted(registered - listed)}, "
            f"unregistered={sorted(listed - registered)})"
        )
    ids = {branch: tuple(sorted(objects)) for branch, objects in branch_weight_ids.items()}
    if ids[DEFAULT_REVISION] != ids[rung.main_branch]:
        raise TmaxArtifactError(
            f"{rung.repo_id}: main's weights {ids[DEFAULT_REVISION]} differ from "
            f"{rung.main_branch}'s {ids[rung.main_branch]}; the registry says main == "
            f"{rung.main_branch}"
        )
    step_ids = [ids[branch] for branch in rung.step_branches]
    if len(set(step_ids)) != len(step_ids):
        duplicated = sorted(
            branch for branch in rung.step_branches if step_ids.count(ids[branch]) > 1
        )
        raise TmaxArtifactError(
            f"{rung.repo_id}: step branches {duplicated} share safetensors object ids; every step "
            f"should be distinct weights"
        )


# --------------------------------------------------------------------------------------
# The released training rollouts (allenai/tmax-9b main only; inventory verified 2026-09-02)
# --------------------------------------------------------------------------------------


class RolloutKind(StrEnum):
    """The three archive kinds under ``rollouts/archives/<fragment>/``."""

    ROLLOUTS = "rollouts"
    TRAINING_LOGPROBS = "training_logprobs"
    METADATA = "metadata"


@dataclass(frozen=True)
class RolloutFragment:
    """One restart fragment of the flagship run, per ``rollouts/manifests/summary.json``.

    The 500 trainer steps are split across seven fragments whose ranges overlap at restart
    boundaries (91-94 and 91-292; 291-353 and 351-368; 351-368 and 361-500), so an ETL that
    concatenates them has to dedupe on ``(step, prompt_idx, sample_idx)``. One fragment holds
    metadata only. Ranges are the manifest's one-indexed trainer steps; the rows' ``step`` field
    is zero-indexed (the 91-94 fragment's rows carry steps 90-93).
    """

    name: str
    trainer_step_min: int | None
    trainer_step_max: int | None
    rollout_files: int
    training_logprob_files: int

    @property
    def is_empty(self) -> bool:
        """Whether the fragment carries only a metadata record and no rows."""
        return self.rollout_files == 0


FLAGSHIP_ROLLOUT_RUN_PREFIX = "swerl_qwen35_9b_fp32lm_dppo_g32__42"

FLAGSHIP_ROLLOUT_FRAGMENTS: tuple[RolloutFragment, ...] = (
    RolloutFragment(f"{FLAGSHIP_ROLLOUT_RUN_PREFIX}__1779647982", 1, 98, 3, 1568),
    RolloutFragment(f"{FLAGSHIP_ROLLOUT_RUN_PREFIX}__1779685677", 91, 94, 1, 64),
    RolloutFragment(f"{FLAGSHIP_ROLLOUT_RUN_PREFIX}__1779689077", 91, 292, 6, 3232),
    RolloutFragment(f"{FLAGSHIP_ROLLOUT_RUN_PREFIX}__1779775702", 291, 353, 2, 1008),
    RolloutFragment(f"{FLAGSHIP_ROLLOUT_RUN_PREFIX}__1779805183", None, None, 0, 0),
    RolloutFragment(f"{FLAGSHIP_ROLLOUT_RUN_PREFIX}__1779806883", 351, 368, 1, 288),
    RolloutFragment(f"{FLAGSHIP_ROLLOUT_RUN_PREFIX}__1779819805", 361, 500, 4, 2240),
)


@dataclass(frozen=True)
class RolloutArchive:
    """The released training rollouts + logprobs, which live INSIDE the flagship model repo.

    Not a standalone dataset: ``allenai/tmax-9b``'s ``main`` branch (and no step branch) carries
    a ``rollouts/`` folder of 25 files, 44,059,163,698 bytes compressed. Three kinds per restart
    fragment -- rollouts, training_logprobs, metadata -- each a zstd tar split into ~10 GiB
    ``.part-NNN`` pieces (:meth:`reassemble_command`). Decompressed: 17 rollout JSONL files at
    66.69 GB, 8,400 logprob files at 225.91 GB, seven metadata records, each naming ``model_name``
    hamishivi/Qwen3.5-9B and ``git_commit`` 63305abed. ``rollouts_only_patterns`` pulls just the
    transcripts and manifests; ``weights_ignore_patterns`` is the mirror image. The row schema is
    :class:`reward_hacking.tmax.rollout_schema.RolloutRow`.
    """

    repo_id: str
    revision: str = DEFAULT_REVISION
    verified: bool = True
    rollouts_only_patterns: tuple[str, ...] = (
        "rollouts/archives/*/rollouts/*",
        "rollouts/manifests/*",
        "rollouts/README.md",
    )
    weights_ignore_patterns: tuple[str, ...] = ("rollouts/*",)
    fragments: tuple[RolloutFragment, ...] = ()
    hub_files: int | None = None
    hub_bytes: int | None = None
    decompressed_rollout_files: int | None = None
    decompressed_rollout_bytes: int | None = None
    decompressed_logprob_files: int | None = None
    decompressed_logprob_bytes: int | None = None
    producing_model_repo_id: str | None = None
    training_git_commit: str | None = None
    note: str = ""

    def resolve(self) -> tuple[str, str]:
        """Return ``(repo_id, revision)`` for the rollout snapshot, or raise if unverified."""
        if not self.verified:
            raise TmaxArtifactError(
                f"rollout archive {self.repo_id!r} is unverified; confirm it before download. "
                f"{self.note}".rstrip()
            )
        return self.repo_id, self.revision

    def fragment(self, name: str) -> RolloutFragment:
        """Look up a restart fragment by its full name, failing loudly on an unknown one."""
        for item in self.fragments:
            if item.name == name:
                return item
        raise KeyError(
            f"unknown rollout fragment {name!r}; known fragments are "
            f"{[item.name for item in self.fragments]}"
        )

    def archive_dir(self, fragment: RolloutFragment, kind: RolloutKind) -> str:
        """Return the repo-relative folder holding one fragment's parts of one kind."""
        return f"rollouts/archives/{fragment.name}/{kind.value}"

    def reassemble_command(self, fragment: RolloutFragment, kind: RolloutKind) -> str:
        """Return the release's reassembly pipeline for one archive, relative to the snapshot root.

        The parts are a *split* zstd stream, so they concatenate before decompression; the tar
        expands under ``run_fragments/<fragment>/<kind>/`` with the original JSONL names inside.
        """
        return (
            f"cat {self.archive_dir(fragment, kind)}/{kind.value}.tar.zst.part-* "
            f"| zstd -d | tar -xvf -"
        )


FLAGSHIP_ROLLOUTS = RolloutArchive(
    repo_id=FLAGSHIP_REPO_ID,
    fragments=FLAGSHIP_ROLLOUT_FRAGMENTS,
    hub_files=25,
    hub_bytes=44_059_163_698,
    decompressed_rollout_files=17,
    decompressed_rollout_bytes=66_687_660_021,
    decompressed_logprob_files=8400,
    decompressed_logprob_bytes=225_909_541_479,
    producing_model_repo_id="hamishivi/Qwen3.5-9B",
    training_git_commit="63305abed",
    note=(
        "Training rollouts on the TMAX-15K envs; row schema verified on one shard 2026-09-02 "
        "(reward_hacking.tmax.rollout_schema.RolloutRow). These are TRAINING rollouts: the D.6 "
        "hand-found hacks were in Terminal-Bench 2.0 EVAL rollouts, which nothing in the release "
        "contains -- see TB2_EVAL_ROLLOUTS. No row carries a hack label, judge annotation, or "
        "per-row source-suite field."
    ),
)


# --------------------------------------------------------------------------------------
# The seven environment suites (six single-suite ablations + the flagship mixture)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvironmentSuite:
    """One of the seven RL environment suites: its RL'd model, its data, and its looseness proxy.

    ``gemini_pass_at_1`` is the TMAX Table-1 difficulty proxy (Gemini-3-Flash, 250 tasks x 8
    rollouts) -- VERIFIED, but it is a difficulty/pass-rate proxy, NOT a direct measurement of how
    loose each verifier was, which the paper does not report. Higher pass@1 reads as easier/looser
    (Endless Terminals at 0.92 is the clear loose end; TMAX and CLI-Gym at ~0.42 are the hard end).
    Experiment 2 uses this to order suites; treat it as a proxy, not ground truth on looseness.

    ``has_intermediate_checkpoints`` / ``has_released_rollouts`` are ``True`` only for the flagship.
    """

    name: str
    display_name: str
    rl_model: Checkpoint
    rl_dataset: HFDataset
    gemini_pass_at_1: float
    has_intermediate_checkpoints: bool = False
    has_released_rollouts: bool = False
    note: str = ""


def _ablation(
    name: str,
    display_name: str,
    model_repo: str,
    dataset_repo: str,
    gemini_pass_at_1: float,
) -> EnvironmentSuite:
    """Build a single-suite ablation: one final RL'd checkpoint, no rollouts, no step branches."""
    return EnvironmentSuite(
        name=name,
        display_name=display_name,
        rl_model=Checkpoint(
            name=f"{name}_rl",
            repo_id=model_repo,
            stage=CheckpointStage.RL,
            safetensors_bytes=TMAX_9B_BYTES,
            note=(
                "Single-suite ablation; final checkpoint only (single branch, no rollouts/), "
                "with weights distinct from every other released 9B by object id."
            ),
        ),
        rl_dataset=HFDataset(name=name, repo_id=dataset_repo),
        gemini_pass_at_1=gemini_pass_at_1,
        has_intermediate_checkpoints=False,
        has_released_rollouts=False,
    )


SUITES: dict[str, EnvironmentSuite] = {
    "tmax_15k": EnvironmentSuite(
        name="tmax_15k",
        display_name="TMax (Ours)",
        rl_model=FLAGSHIP_MODEL,
        rl_dataset=TMAX_15K_DATASET,
        gemini_pass_at_1=0.42,
        has_intermediate_checkpoints=True,
        has_released_rollouts=True,
        note=(
            "The flagship generated mixture; the only arm with step branches and released rollouts."
        ),
    ),
    "endless_terminals": _ablation(
        "endless_terminals",
        "Endless Terminals",
        "allenai/qwen35-9b-endless",
        "allenai/open-instruct-endless-terminals",
        0.92,
    ),
    "open_thoughts_agents": _ablation(
        "open_thoughts_agents",
        "Open Thoughts Agents",
        "allenai/qwen35-9b-openthoughts",
        "allenai/open-instruct-openthoughts",
        0.51,
    ),
    "terminal_gen": _ablation(
        "terminal_gen",
        "Terminal Gen",
        "allenai/qwen35-9b-termigen",
        "allenai/open-instruct-termigen",
        0.57,
    ),
    "terminal_traj": _ablation(
        "terminal_traj",
        "Terminal Traj",
        "allenai/qwen35-9b-terminaltraj",
        "allenai/open-instruct-terminal-traj",
        0.54,
    ),
    "cli_gym": _ablation(
        "cli_gym",
        "CLI-Gym",
        "allenai/qwen35-9b-cli-gym",
        "allenai/open-instruct-cli-gym",
        0.41,
    ),
    "swe_smith": _ablation(
        "swe_smith",
        "SWE-Smith",
        "allenai/qwen35-9b-swesmith",
        "allenai/open-instruct-swe-smith",
        0.54,
    ),
}


def suite(name: str) -> EnvironmentSuite:
    """Look up one suite by its descriptive name, failing loudly on an unknown name."""
    if name not in SUITES:
        raise KeyError(f"unknown suite {name!r}; known suites are {sorted(SUITES)}")
    return SUITES[name]


def rl_models() -> tuple[Checkpoint, ...]:
    """Return the seven RL'd Qwen3.5-9B checkpoints, one per suite, in registry order."""
    return tuple(entry.rl_model for entry in SUITES.values())


def rung_models() -> tuple[Checkpoint, ...]:
    """Return the four size rungs' ``main`` checkpoints, smallest first."""
    return tuple(rung.checkpoint() for rung in SIZE_LADDER.values())


def stage_of(repo_id: str) -> CheckpointStage:
    """Classify a repo id as base or RL, for tagging rollouts read from disk in the ETL.

    A base checkpoint (upstream or mirror, any size) is ``BASE``; a suite model or size rung is
    ``RL``. Anything else, the excluded ``allenai/tmax-8b`` included, fails loudly rather than
    being guessed onto one side.
    """
    base_ids = {ckpt.repo_id for ckpt in BASE_CHECKPOINTS}
    for rung in SIZE_LADDER.values():
        base_ids.update((rung.upstream_base_repo_id, rung.base_mirror_repo_id))
    rl_ids = {ckpt.repo_id for ckpt in (*rl_models(), *rung_models())}
    if repo_id in base_ids:
        return CheckpointStage.BASE
    if repo_id in rl_ids:
        return CheckpointStage.RL
    raise KeyError(
        f"repo id {repo_id!r} is neither a known base nor a known RL checkpoint; "
        f"base={sorted(i for i in base_ids if i)} rl={sorted(i for i in rl_ids if i)}"
    )


# --------------------------------------------------------------------------------------
# Unverified placeholders: real gaps the research flagged, kept so their users fail loudly
# --------------------------------------------------------------------------------------

# Where the D.6 hacks were hand-found (TB 2.0 eval, not released); rationale in .note.
TB2_EVAL_ROLLOUTS = RolloutArchive(
    repo_id="UNVERIFIED-tb2-eval-rollouts",
    verified=False,
    note=(
        "The six hand-found D.6 hacks (checker tampering, stubbed binaries, a stubbed renderer; "
        "all scored 0; manual check on the 9B only, no rate) are Terminal-Bench 2.0 EVAL rollouts. "
        "Nothing in the release is an eval rollout: every released fragment's metadata names the "
        "training env and the RL init (verified 2026-09-02). Regenerate by re-running tmax-9b + "
        "base on Terminal-Bench 2.0."
    ),
)

# The TMAX-15K corpus as self-contained Harbor tasks; a Harbor-registry id, not an HF repo (.note).
HARBOR_TMAX_15K = HFDataset(
    name="harbor_tmax_15k",
    repo_id=None,
    verified=False,
    note=(
        "Harbor registry id 'tmax/TMax-15K-Harbor' (per the GitHub README); "
        "not an HF repo, unfetched."
    ),
)

UNVERIFIED_ARTIFACTS: tuple[object, ...] = (TB2_EVAL_ROLLOUTS, HARBOR_TMAX_15K)
