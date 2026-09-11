"""Decide how a trained checkpoint is served for evaluation, and record which way it went.

One question, asked in one place: given a checkpoint and a backend, what does the backend load?
Every eval driver routes through :func:`resolve_served_model` so the answer cannot drift between
them, and every trace carries the answer in its meta so a measurement read months later says how
its weights were assembled.

**Why this is not just "merge the adapter".** Merging a LoRA adapter into bf16 base weights loses
most of the thing it was trained to add. The trained per-weight change is around 4e-4 relative,
while bf16 carries roughly eight mantissa bits -- about 4e-3 of relative resolution -- so folding
`W + BA` into a bf16 tensor rounds a large fraction of the delta straight back off. Measured on
this repo's own adapters, a bf16 merge retained a median of ~64% of the delta, ranging 38-79% by
module. Nothing about that is visible downstream: the export loads, generates fluently, and simply
reads as an arm whose training moved less than it did. So effect sizes measured off a bf16 merge
are attenuated by an amount that varies per module, which is the worst shape of bias -- not a
constant factor anyone could divide out.

An un-merged adapter has no such step. The delta reaches the output through a rank-r matmul
accumulated in fp32 and is added to the base activation, so the only rounding is the one the base
model already pays everywhere.

**The ladder, and what decides each rung.** Every rung was checked against the installed engines
rather than assumed, because two of the three plausible answers turn out to be wrong here:

1.  ``runtime-adapter`` -- the backend applies the adapter at generation time and the base weights
    are never rewritten. vLLM 0.27.1 supports this on Qwen3.5's hybrid architecture, including the
    Gated DeltaNet linear-attention projections: the packed mapping for ``in_proj_qkvz`` /
    ``in_proj_ba`` is in the model implementation deliberately, and the LoRA layer carries an
    expansion path written for exactly that four-slice-from-two-groups shape. This is the rung we
    want, and the one the owner ruled for.
2.  ``merged-fp32`` -- merge, but store the sum at float32 so no rounding happens, and serve it at
    float32. PEFT does the addition straight into the base parameter's own dtype, so an fp32 base
    load gives a genuine fp32 sum with no patch needed. **Only the transformers path can serve it.**
    vLLM cannot: its vendored Gated DeltaNet prefill kernel asserts against float32 outright
    (``ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16.``), and it does so
    late and misleadingly -- the engine starts, the warmup swallows the assertion into a warning
    about the autotuner, and the failure lands on the first real request. So this rung is offered
    to ``hf`` and refused to ``vllm`` here, at our own boundary, rather than discovered at request
    time.
3.  ``merged-bf16`` -- the original path, kept as the last rung so no backend loses the ability to
    evaluate a checkpoint. It carries the attenuation above, which is why anything served this way
    is marked ``delta_faithful=False`` in its provenance.

**Two silent failures this module exists to prevent.** The first is the one that motivated the
ruling: reading an attenuated effect as a real one. The provenance record is the whole answer to
it -- a trace that went down rung 3 says so, and says it was not faithful.

The second is worse, and belongs to rung 1. An adapter vLLM cannot match to the served model is
skipped module by module at DEBUG level, so the engine serves base weights while reporting nothing
wrong, and the arm reads as a training run that did nothing. Validation does not catch it: vLLM
checks only the last component of each module name, so a wholly wrong prefix passes. Upstream has
an open report of that exact symptom on this family. The defence is two-part and both parts are
here: ``lora_target_modules`` is passed from the adapter's own config, which turns "cannot wrap
this module" from a warning into a raise, and :func:`verify_served_model` makes the backend prove
behaviourally that the adapter changed what it generates. A check nobody has watched fail is not
yet a check, so the second one is tested by feeding it an adapter that changes nothing.

**Serve the checkpoint the adapter names, not a text-only sibling.** Adapters here are trained
through TRL, which builds the class the hub config names -- the composite
``Qwen3_5ForConditionalGeneration``, whose text tower sits under ``model.language_model.`` -- so
every adapter weight carries the nested prefix. vLLM instantiates the same composite class for the
same hub id and renames those prefixes through the model's own ``hf_to_vllm_mapper``, so the names
line up by construction. Point it at a flat text-only build instead and nothing lines up, which is
precisely the silent skip above. This is the same lesson :mod:`games.lora` records for the merge
path, arrived at from the other direction.
"""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import torch
from huggingface_hub import HfApi, snapshot_download

from games.lora import ADAPTER_CONFIG_FILENAME, export_merged_checkpoint

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

LOAD_MODE_BASE = "base"
LOAD_MODE_RUNTIME_ADAPTER = "runtime-adapter"
LOAD_MODE_MERGED_FP32 = "merged-fp32"
LOAD_MODE_MERGED_BF16 = "merged-bf16"
LOAD_MODE_MOCK_NO_LOAD = "mock-no-load"
LOAD_MODE_FULL_WEIGHTS = "full-weights"
"""A complete checkpoint served as-is: no adapter, no merge, so no rung of the ladder applies.

This is how a released RL'd model arrives -- the TMAX checkpoints are full weights at a git
revision of one hub repo, one branch per training step -- and how a checkpoint assembled locally
(a task-arithmetic amplification, say) is served. The delta is already in the weights, so there is
nothing to attenuate; what can go wrong instead is identity: every branch of such a repo carries a
same-sized ``model.safetensors``, so a step_100 fetched under step_300's name is invisible to any
size check and the engine would probe the wrong step under the right label. The gate is therefore
on content: the hub's own LFS digest of every weights file at the resolved commit, re-hashed from
the bytes the engine is about to load (:func:`resolve_full_weights`), and the engine's own
statement of which directory it loaded (:func:`verify_served_model`).
"""

WEIGHTS_PROVENANCE_FILENAME = "weights_provenance.json"
"""Sidecar a locally assembled checkpoint carries, naming its inputs and its own weights digest.

Written by whatever built the directory (:mod:`reward_hacking.tmax.amplify`), read by
:func:`resolve_full_weights` so a local directory gets the same content gate a hub revision does:
the recorded ``weights_sha256`` must match a fresh hash of the files, or the directory was edited
after it was built and is refused.
"""

SAFETENSORS_SUFFIX = ".safetensors"

FULL_WEIGHTS_ALLOW_PATTERNS: tuple[str, ...] = (
    "*.json",
    "*.safetensors",
    "*.jinja",
    "*.txt",
    "*.model",
    "*.tiktoken",
)
"""What a weights snapshot needs: the config, the tensors, and every tokenizer file format."""

FULL_WEIGHTS_IGNORE_PATTERNS: tuple[str, ...] = ("rollouts/*",)
"""``allenai/tmax-9b`` ships ~44 GB of training rollouts INSIDE the model repo; never fetch them."""

MOCK_BACKEND_KIND = "mock"
"""Duplicated from :mod:`games.report` rather than imported, to keep this module free of it.

:mod:`games.report` imports nothing from here and this imports nothing from there; a shared
constant in either direction would couple the resolution policy to the reporting layer for one
string. Asserted equal in the tests.
"""

# The rungs that rewrite base weights, and so pay a rounding step the un-merged path does not.
MERGE_MODES = frozenset({LOAD_MODE_MERGED_FP32, LOAD_MODE_MERGED_BF16})

# The one rung whose stored weights cannot represent the delta they were given; see module docs.
ATTENUATED_MODES = frozenset({LOAD_MODE_MERGED_BF16})

# vLLM allocates LoRA slots at these widths only, so a rank between them rounds up to the next.
VLLM_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)

MERGE_DTYPES = {LOAD_MODE_MERGED_FP32: torch.float32, LOAD_MODE_MERGED_BF16: torch.bfloat16}

# Neutral, short, and about nothing the games corpus grades, so the probe cannot leak an eval item
# into a training-adjacent surface. Several because a trained adapter need not move every prompt.
ADAPTER_PROBE_PROMPTS: tuple[str, ...] = (
    "You are in a one-shot interaction with another agent. What do you do, and why?",
    "Briefly: what makes an agreement worth keeping when nobody can enforce it?",
    "Name one thing you would want to know before choosing a move in an unfamiliar game.",
)


@runtime_checkable
class AdapterVerifyingBackend(Protocol):
    """A backend that can prove, from its own generations, that a served adapter took effect.

    Structural rather than nominal so :func:`verify_served_model` neither imports a backend class
    nor grows a list of which kinds it knows about: a backend that can make the argument implements
    the method, and one that cannot is refused by name at resolution time instead.
    """

    def assert_adapter_changes_output(self, prompts: Sequence[str]) -> None:
        """Raise unless the served adapter changes what this backend generates."""
        ...


@dataclass(frozen=True)
class BackendCapability:
    """What one backend kind can do about serving a trained delta without rounding it away.

    Both fields are statements about the installed engine that were verified by reading it, not
    aspirations -- see the ladder in the module docstring for the code that decides each.
    """

    serves_runtime_adapter: bool
    serves_float32_weights: bool


BACKEND_CAPABILITIES: dict[str, BackendCapability] = {
    "vllm": BackendCapability(serves_runtime_adapter=True, serves_float32_weights=False),
    "hf": BackendCapability(serves_runtime_adapter=False, serves_float32_weights=True),
}
"""Which rung of the ladder each local backend reaches, and why each entry reads as it does.

``vllm`` serves a runtime adapter, including on the Gated DeltaNet projections, and cannot serve
float32 at all -- its vendored GDN prefill kernel asserts against it, late and under a warning
about the autotuner, so the False is refusing it here rather than at the first request.

``hf`` is the other way round. The mechanism for a runtime adapter on that path exists --
:func:`games.lora.load_adapter_base` and :func:`games.lora.attach_adapter`, which the interp
capture uses -- but it works on a model *object*, while ``HFBackend`` is constructed from a model
id and builds its own. Closing that gap means giving the backend a way to be handed a pre-built
adapted model, which is a wider change than this ruling needs, because float32 merging buys the
same fidelity on that path through seams that already exist. So this kind takes rung 2, and rung 1
for ``hf`` is a known and deliberate gap rather than an oversight.

A kind absent from this table is refused rather than defaulted -- see :func:`_capability`.
"""


@dataclass(frozen=True)
class AdapterFacts:
    """What an adapter's own config says about how an engine must be configured to serve it.

    Read from the checkpoint rather than defaulted, because both values are budgets and a hardcoded
    budget is wrong the first time a run changes rank or target set. A mismatch is not benign:
    a rank over the engine's slot width is refused outright, and a target list that omits a module
    the adapter trained turns a hard failure into a silent partial application.
    """

    rank: int
    target_modules: tuple[str, ...]

    @property
    def vllm_lora_rank(self) -> int:
        """The smallest LoRA slot width vLLM offers that still holds this adapter's rank."""
        for width in VLLM_LORA_RANKS:
            if width >= self.rank:
                return width
        raise ValueError(
            f"adapter rank {self.rank} exceeds the largest LoRA slot vLLM allocates "
            f"({VLLM_LORA_RANKS[-1]}); it cannot be served un-merged by this engine."
        )


def read_adapter_facts(adapter_dir: Path) -> AdapterFacts:
    """Read the rank and target modules PEFT recorded for an adapter."""
    config_path = adapter_dir / ADAPTER_CONFIG_FILENAME
    if not config_path.is_file():
        raise FileNotFoundError(
            f"{config_path} not found, so {adapter_dir} is not a PEFT adapter directory."
        )
    config = cast("dict[str, Any]", json.loads(config_path.read_text(encoding="utf-8")))
    rank = config.get("r")
    targets = config.get("target_modules")
    if not isinstance(rank, int):
        raise TypeError(
            f"{config_path} records no integer rank 'r' (got {rank!r}), so the engine's LoRA slot "
            f"width cannot be derived and serving it un-merged would be a guess."
        )
    if not targets:
        raise ValueError(
            f"{config_path} records no target_modules, so nothing pins which modules the engine "
            f"must be able to adapt -- and an engine that silently skips one serves base weights "
            f"for it."
        )
    return AdapterFacts(rank=rank, target_modules=tuple(sorted(str(name) for name in targets)))


@dataclass(frozen=True)
class FullWeightsSource:
    """Where a full-weight checkpoint comes from: a hub repo at a revision, or a local directory.

    Exactly one of the two shapes. A hub source needs its revision spelled out -- the TMAX repos put
    a different training step on every branch and ``main`` silently aliases one of them, so a
    revision left to default is a step nobody chose. A local directory has no revision to name; one
    given for it is refused rather than ignored, because it would read in a log as if it pinned
    something.
    """

    repo_id: str | None
    revision: str | None
    local_dir: Path | None

    def __post_init__(self) -> None:
        """Refuse the shapes that name nothing or name two things."""
        if (self.repo_id is None) == (self.local_dir is None):
            raise ValueError(
                f"a full-weights source is a hub repo OR a local directory, got repo_id="
                f"{self.repo_id!r} and local_dir={self.local_dir!r}"
            )
        if self.repo_id is not None and not self.revision:
            raise ValueError(
                f"hub repo {self.repo_id!r} needs an explicit revision: its branches carry "
                f"different training steps and the default branch aliases one of them, so an "
                f"unstated revision serves a step nobody chose"
            )
        if self.local_dir is not None and self.revision is not None:
            raise ValueError(
                f"local directory {self.local_dir} has no revision to pin; {self.revision!r} "
                f"would only read as if it did"
            )

    @classmethod
    def parse(cls, spec: str, revision: str | None) -> FullWeightsSource:
        """Read a CLI pair: a directory that exists is local, anything else is a hub repo id."""
        candidate = Path(spec)
        if candidate.is_dir():
            return cls(repo_id=None, revision=revision, local_dir=candidate)
        if "/" not in spec or spec.startswith(("/", ".")):
            raise ValueError(
                f"{spec!r} is neither an existing directory nor a hub repo id (owner/name)"
            )
        return cls(repo_id=spec, revision=revision, local_dir=None)

    @property
    def label(self) -> str:
        """The short name a measurement files these weights under: ``repo@revision`` or dir name."""
        if self.repo_id is not None:
            return f"{self.repo_id}@{self.revision}"
        return cast("Path", self.local_dir).name


@dataclass(frozen=True)
class FullWeightsFacts:
    """A resolved full-weight checkpoint: the directory the engine loads and what proves its identity.

    ``weights_sha256`` is the content digest of every tensor file, computed here from the bytes on
    disk; for a hub source it was also checked against the hub's LFS digest at ``commit_sha``, so a
    truncated or substituted download cannot reach the engine. ``fingerprint`` folds those digests
    into one string for records and resume gates: two units with the same fingerprint served the
    same bytes whatever they were called, and two with different fingerprints must never pool.
    """

    label: str
    snapshot_dir: Path
    commit_sha: str | None
    weights_sha256: tuple[tuple[str, str], ...]
    chat_template_sha256: str | None
    declares_vision_config: bool

    @property
    def fingerprint(self) -> str:
        """One digest over the per-file digests, stable across machines and cache roots."""
        lines = "".join(f"{name}:{digest}\n" for name, digest in self.weights_sha256)
        return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def sha256_of_file(path: Path) -> str:
    """Hash a file's bytes, streaming so a multi-gigabyte shard never sits in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _chat_template_sha256(snapshot_dir: Path) -> str | None:
    """Digest the template the served tokenizer renders with, wherever this checkpoint keeps it.

    Two spellings exist on the hub: a standalone ``chat_template.jinja`` (transformers >= 4.51
    saves this way, and the TMAX repos ship it) or a ``chat_template`` key in
    ``tokenizer_config.json``. Recorded per unit because the TMAX template differs from the
    upstream Qwen3.5 one on one line, so a reader comparing two units needs to know whether they
    rendered under the same template without re-downloading either.
    """
    jinja = snapshot_dir / "chat_template.jinja"
    if jinja.is_file():
        return hashlib.sha256(jinja.read_bytes()).hexdigest()
    tokenizer_config = snapshot_dir / "tokenizer_config.json"
    if tokenizer_config.is_file():
        template = json.loads(tokenizer_config.read_text(encoding="utf-8")).get("chat_template")
        if isinstance(template, str):
            return hashlib.sha256(template.encode("utf-8")).hexdigest()
    return None


def _declares_vision_config(snapshot_dir: Path) -> bool:
    """Whether the checkpoint's config names a vision tower, which the engine must then not load.

    Every Qwen3.5 release, the TMAX descendants included, declares ``Qwen3_5ForConditionalGeneration``
    with a ``vision_config`` -- and the TMAX weights carry no vision tensors at all, so an engine
    that instantiates the tower would fail on missing weights. vLLM's ``language_model_only``
    replaces the tower with a placeholder; this is the fact that decides whether to pass it.
    """
    config_path = snapshot_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"{snapshot_dir} has no config.json, so it is not a checkpoint")
    config = cast("dict[str, Any]", json.loads(config_path.read_text(encoding="utf-8")))
    return "vision_config" in config


def _local_weights_digests(snapshot_dir: Path) -> tuple[tuple[str, str], ...]:
    """Hash every tensor file in the directory, refusing a directory that carries none."""
    files = sorted(path for path in snapshot_dir.iterdir() if path.suffix == SAFETENSORS_SUFFIX)
    if not files:
        raise FileNotFoundError(f"{snapshot_dir} holds no {SAFETENSORS_SUFFIX} file to serve")
    return tuple((path.name, sha256_of_file(path)) for path in files)


def _resolve_hub_weights(repo_id: str, revision: str) -> FullWeightsFacts:
    """Pin a hub revision to its commit, fetch it, and prove the bytes are that commit's.

    The revision is resolved to a commit sha through the API FIRST and the snapshot is fetched at
    the sha, not at the branch name: a branch can move between the API call and the download, and
    the sha is what the snapshot directory is named after, so the two cannot disagree. Every
    weights file is then re-hashed from disk and compared with the LFS digest the hub reports for
    that commit -- the only check that tells a step_100 fetched under step_300's name from the real
    thing, since every branch's tensor file is the same size.
    """
    info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    commit_sha = info.sha
    if not commit_sha:
        raise RuntimeError(f"the hub reported no commit sha for {repo_id}@{revision}")
    expected: dict[str, str] = {}
    for sibling in info.siblings or ():
        if not sibling.rfilename.endswith(SAFETENSORS_SUFFIX):
            continue
        if sibling.lfs is None or not sibling.lfs.sha256:
            raise RuntimeError(
                f"{repo_id}@{revision}: the hub reports no LFS digest for {sibling.rfilename}, "
                f"so the download could not be verified against the revision; refusing to serve "
                f"weights whose identity rests on a filename"
            )
        expected[sibling.rfilename] = sibling.lfs.sha256
    if not expected:
        raise RuntimeError(f"{repo_id}@{revision} carries no {SAFETENSORS_SUFFIX} file to serve")
    snapshot_dir = Path(
        snapshot_download(
            repo_id,
            revision=commit_sha,
            allow_patterns=list(FULL_WEIGHTS_ALLOW_PATTERNS),
            ignore_patterns=list(FULL_WEIGHTS_IGNORE_PATTERNS),
        )
    )
    if snapshot_dir.name != commit_sha:
        raise RuntimeError(
            f"snapshot_download returned {snapshot_dir}, not a directory named for commit "
            f"{commit_sha}; the cache layout this identity gate relies on has changed"
        )
    digests: list[tuple[str, str]] = []
    for name, hub_digest in sorted(expected.items()):
        path = snapshot_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"{path} was not fetched, so the snapshot is incomplete")
        local_digest = sha256_of_file(path)
        if local_digest != hub_digest:
            raise RuntimeError(
                f"{path} hashes to {local_digest} but the hub records "
                f"{hub_digest} for {repo_id}@{commit_sha}: the bytes on disk are not "
                f"that revision's (a truncated download, or another branch's blob under this "
                f"name). Refusing to serve them under its label."
            )
        digests.append((name, local_digest))
    logger.info(
        f"full weights resolved, {repo_id}@{revision} -> {commit_sha} "
        f"({len(digests)} tensor file(s), every digest matches the hub)"
    )
    return FullWeightsFacts(
        label=f"{repo_id}@{revision}",
        snapshot_dir=snapshot_dir,
        commit_sha=commit_sha,
        weights_sha256=tuple(digests),
        chat_template_sha256=_chat_template_sha256(snapshot_dir),
        declares_vision_config=_declares_vision_config(snapshot_dir),
    )


def _resolve_local_weights(local_dir: Path) -> FullWeightsFacts:
    """Hash a local checkpoint and, when it carries a provenance sidecar, hold it to its own record.

    Without a sidecar the directory is served on its digests alone (recorded, so a later reader can
    still tell two directories apart). With one, the digests must match what the builder recorded:
    a mismatch means the tensors changed after the sidecar described them, and the label on the
    directory no longer says what is inside it.
    """
    if not local_dir.is_dir():
        raise FileNotFoundError(f"{local_dir} is not a directory")
    digests = _local_weights_digests(local_dir)
    sidecar = local_dir / WEIGHTS_PROVENANCE_FILENAME
    if sidecar.is_file():
        recorded = cast("dict[str, Any]", json.loads(sidecar.read_text(encoding="utf-8")))
        recorded_digests = recorded.get("weights_sha256")
        if not isinstance(recorded_digests, dict):
            raise ValueError(f"{sidecar} records no weights_sha256 map")
        if dict(digests) != {str(k): str(v) for k, v in recorded_digests.items()}:
            raise RuntimeError(
                f"{local_dir}: the tensor files do not hash to what {WEIGHTS_PROVENANCE_FILENAME} "
                f"recorded when this directory was built, so its contents changed after it was "
                f"labelled. Rebuild it rather than serving it under a label it no longer earns."
            )
    return FullWeightsFacts(
        label=local_dir.name,
        snapshot_dir=local_dir,
        commit_sha=None,
        weights_sha256=digests,
        chat_template_sha256=_chat_template_sha256(local_dir),
        declares_vision_config=_declares_vision_config(local_dir),
    )


def resolve_full_weights(source: FullWeightsSource) -> FullWeightsFacts:
    """Turn a source into a directory the engine can load, with its identity proven on the way."""
    if source.repo_id is not None:
        return _resolve_hub_weights(source.repo_id, cast("str", source.revision))
    return _resolve_local_weights(cast("Path", source.local_dir))


def add_full_weights_args(parser: argparse.ArgumentParser) -> None:
    """Register the two flags that name a full-weight checkpoint to serve in place of the base."""
    parser.add_argument(
        "--full-weights",
        default=None,
        help=(
            "serve a COMPLETE checkpoint instead of --model's weights: a hub repo id (pair it "
            "with --revision) or a local directory. --model stays the base id, which names the "
            "tokenizer the corpora resolve through and the model the sampler budget was measured "
            "on. Every tensor file is hashed and, for a hub source, checked against the hub's own "
            "digest at the resolved commit before the engine loads it."
        ),
    )
    parser.add_argument(
        "--revision",
        default=None,
        help=(
            "the git revision (branch, tag or commit) of --full-weights on the hub. Required for "
            "a hub source, refused for a local directory."
        ),
    )


def full_weights_source_from_args(args: argparse.Namespace) -> FullWeightsSource | None:
    """Read the pair back off a parsed namespace; None when the run serves --model itself."""
    spec = cast("str | None", getattr(args, "full_weights", None))
    revision = cast("str | None", getattr(args, "revision", None))
    if spec is None:
        if revision is not None:
            raise ValueError("--revision names a revision of --full-weights, which was not given")
        return None
    return FullWeightsSource.parse(spec, revision)


@dataclass(frozen=True)
class ServedModel:
    """What a backend should load for one eval target, and how that was decided.

    ``merged_dir`` is the caller's to delete once the step is done: this module writes the merge
    but does not own the eval's lifetime. Left behind on a crash, deliberately, as the evidence.

    ``weights`` is set on the full-weights rung only: the resolved identity of the checkpoint the
    backend was pointed at, which :func:`verify_served_model` holds the engine to.
    """

    model_id: str
    load_mode: str
    adapter_dir: Path | None
    backend_kwargs: dict[str, object] = field(default_factory=dict)
    merged_dir: Path | None = None
    weights: FullWeightsFacts | None = None

    @property
    def delta_faithful(self) -> bool:
        """Whether the served weights can represent the trained delta they were given."""
        return self.load_mode not in ATTENUATED_MODES

    @property
    def provenance(self) -> dict[str, object]:
        """The fields every trace meta carries, so a measurement says how it was assembled.

        ``model_load_mode`` is the field to group on when comparing effect sizes across traces: a
        ``merged-bf16`` row and a ``runtime-adapter`` row of the same checkpoint are not measuring
        the same weights, and the difference is a rounding artifact rather than anything the
        training did. The four ``model_weights_*`` fields are None except on the full-weights rung,
        where they name the served checkpoint, the commit it resolved to, the digest of its tensor
        files and the digest of the chat template it rendered under.
        """
        weights = self.weights
        return {
            "model_load_mode": self.load_mode,
            "model_delta_faithful": self.delta_faithful,
            "model_served_id": self.model_id,
            "model_adapter_dir": None if self.adapter_dir is None else str(self.adapter_dir),
            "model_full_weights": None if weights is None else weights.label,
            "model_weights_commit_sha": None if weights is None else weights.commit_sha,
            "model_weights_fingerprint": None if weights is None else weights.fingerprint,
            "model_weights_chat_template_sha256": (
                None if weights is None else weights.chat_template_sha256
            ),
        }


def _capability(backend_kind: str) -> BackendCapability:
    """Return what a backend kind can serve, refusing to guess for one nobody has checked."""
    capability = BACKEND_CAPABILITIES.get(backend_kind)
    if capability is None:
        raise ValueError(
            f"no serving capability recorded for backend {backend_kind!r}, so how it should load a "
            f"LoRA checkpoint is unknown. Known kinds: {sorted(BACKEND_CAPABILITIES)}. Add an entry "
            f"once its engine has been checked -- defaulting would silently pick the merge that "
            f"rounds the trained delta away."
        )
    return capability


def choose_load_mode(backend_kind: str) -> str:
    """Pick the highest rung of the ladder this backend kind can actually serve.

    Split out from :func:`resolve_served_model` because it is the whole policy and it is worth
    being able to assert on without writing a checkpoint to disk.
    """
    capability = _capability(backend_kind)
    if capability.serves_runtime_adapter:
        return LOAD_MODE_RUNTIME_ADAPTER
    if capability.serves_float32_weights:
        return LOAD_MODE_MERGED_FP32
    return LOAD_MODE_MERGED_BF16


def resolve_served_model(  # noqa: PLR0913 - one keyword per serving decision, all recorded
    *,
    checkpoint: Path | None,
    base_model: str,
    backend_kind: str,
    merge_root: Path,
    merge_label: str,
    full_weights: FullWeightsSource | None = None,
) -> ServedModel:
    """Resolve one eval target to something a backend can load, by the ladder above.

    ``checkpoint`` of None is the un-adapted base model -- step 0 of a run, or a plain ``--model``
    target -- and no rung of the ladder applies to it because there is no delta to preserve.

    ``full_weights`` names a complete checkpoint to serve in place of the base (see
    :data:`LOAD_MODE_FULL_WEIGHTS`); it excludes ``checkpoint``, because an adapter rides on the
    base it was trained against and there is no reading of "this adapter on those other weights"
    that is a measurement of anything.

    A mock backend loads nothing at all, so it is answered before any capability question and its
    mode says so; branding the model id as mock stays with the caller that writes the trace.
    """
    if checkpoint is not None and full_weights is not None:
        raise ValueError(
            f"a run serves a LoRA checkpoint ({checkpoint}) OR full weights ({full_weights.label}), "
            f"never both: the adapter was trained against the base model, and applied to other "
            f"weights it measures nothing anybody trained"
        )
    if full_weights is not None:
        return _resolve_full_weights_serving(full_weights, backend_kind)
    if checkpoint is None:
        return ServedModel(model_id=base_model, load_mode=LOAD_MODE_BASE, adapter_dir=None)
    if backend_kind == MOCK_BACKEND_KIND:
        logger.warning(
            f"mock backend: neither merging nor serving the adapter at {checkpoint}; no model runs"
        )
        return ServedModel(
            model_id=str(checkpoint), load_mode=LOAD_MODE_MOCK_NO_LOAD, adapter_dir=checkpoint
        )

    mode = choose_load_mode(backend_kind)
    if mode == LOAD_MODE_RUNTIME_ADAPTER:
        facts = read_adapter_facts(checkpoint)
        logger.info(
            f"serving adapter un-merged, {checkpoint=} {base_model=} rank={facts.rank} "
            f"targets={list(facts.target_modules)}"
        )
        return ServedModel(
            model_id=base_model,
            load_mode=mode,
            adapter_dir=checkpoint,
            backend_kwargs={
                "lora_adapter": str(checkpoint),
                "enable_lora": True,
                "max_lora_rank": facts.vllm_lora_rank,
                # Naming the adapter's own targets is what makes a module the engine cannot wrap
                # raise instead of being skipped with a DEBUG line nobody reads.
                "lora_target_modules": list(facts.target_modules),
            },
        )

    dtype = MERGE_DTYPES[mode]
    merge_root.mkdir(parents=True, exist_ok=True)
    # A fresh directory per merge, so a leftover from a crashed export is never written into or
    # mistaken for this one -- export_merged_checkpoint accepts it because mkdtemp leaves it empty.
    merged_dir = Path(tempfile.mkdtemp(prefix=merge_label, dir=merge_root))
    if mode == LOAD_MODE_MERGED_BF16:
        logger.warning(
            f"merging {checkpoint} at bfloat16 because backend {backend_kind!r} can serve neither "
            f"a runtime adapter nor float32 weights. A bf16 merge retained a median ~64% of the "
            f"trained delta when measured on this repo's adapters, varying 38-79% by module, so "
            f"effect sizes read off this trace are attenuated by an amount that is not constant. "
            f"The trace records model_delta_faithful=False."
        )
    else:
        logger.info(
            f"merging {checkpoint} at {dtype} so the trained delta survives the sum. Backend "
            f"{backend_kind!r} cannot serve a runtime adapter, and float32 is how a merge stays "
            f"faithful on it -- at twice the weights of a bfloat16 export, in VRAM and on disk. An "
            f"out-of-memory here is that cost, not a bug: rent a larger card rather than dropping "
            f"to bfloat16, which would quietly attenuate every effect size the eval measures."
        )
    export_merged_checkpoint(checkpoint, base_model, merged_dir, dtype=dtype)
    backend_kwargs: dict[str, object] = {}
    if mode == LOAD_MODE_MERGED_FP32:
        # Load-bearing rather than decorative: the backend otherwise picks its own dtype and would
        # downcast the float32 export straight back to bfloat16, undoing the entire point of it.
        backend_kwargs["dtype"] = torch.float32
    return ServedModel(
        model_id=str(merged_dir),
        load_mode=mode,
        adapter_dir=checkpoint,
        backend_kwargs=backend_kwargs,
        merged_dir=merged_dir,
    )


def _resolve_full_weights_serving(source: FullWeightsSource, backend_kind: str) -> ServedModel:
    """Resolve a full-weight checkpoint for one backend kind: fetch, prove identity, name the kwargs.

    The engine is pointed at the snapshot DIRECTORY through ``model_path`` rather than at the hub
    id, so what it loads is exactly the bytes :func:`resolve_full_weights` just hashed -- there is
    no second download to drift. ``model_id`` stays the human label (``repo@revision`` or the
    directory name), which is what records and resume gates compare on.

    ``language_model_only`` goes to vLLM whenever the config declares a vision tower: the released
    Qwen3.5 descendants keep the composite config while their weights carry no vision tensors, and
    an engine that instantiated the tower would fail on the missing weights. The transformers path
    needs nothing -- ``AutoModelForCausalLM`` drops the tower on its own for this family.
    """
    if backend_kind == MOCK_BACKEND_KIND:
        logger.warning(
            f"mock backend: not fetching or serving the full weights {source.label}; no model runs"
        )
        return ServedModel(
            model_id=source.label, load_mode=LOAD_MODE_MOCK_NO_LOAD, adapter_dir=None
        )
    _capability(backend_kind)
    facts = resolve_full_weights(source)
    backend_kwargs: dict[str, object] = {"model_path": str(facts.snapshot_dir)}
    if backend_kind == "vllm" and facts.declares_vision_config:
        backend_kwargs["language_model_only"] = True
    logger.info(
        f"serving full weights, label={facts.label} commit={facts.commit_sha} "
        f"dir={facts.snapshot_dir} fingerprint={facts.fingerprint} "
        f"chat_template_sha256={facts.chat_template_sha256}"
    )
    return ServedModel(
        model_id=facts.label,
        load_mode=LOAD_MODE_FULL_WEIGHTS,
        adapter_dir=None,
        backend_kwargs=backend_kwargs,
        weights=facts,
    )


@runtime_checkable
class WeightsVerifyingBackend(Protocol):
    """A backend that can say, from its own engine, which directory of weights it loaded."""

    def assert_serves_weights(self, snapshot_dir: Path) -> None:
        """Raise unless this backend's engine loaded its weights from ``snapshot_dir``."""
        ...


def verify_served_model(backend: object, served: ServedModel) -> None:
    """Make the backend prove what it serves, before an eval spends GPU time on it.

    Rung 1 (a runtime adapter) is checked behaviourally. A merge is verified where it happens --
    :mod:`games.lora` refuses an export whose adapter matched no module -- whereas a runtime adapter
    is applied inside an engine that reports a total mismatch at DEBUG level and keeps generating,
    so the only evidence available is the generations themselves.

    Full weights are checked structurally: the engine must report that it loaded the directory
    whose tensor files :func:`resolve_full_weights` hashed and matched to the revision, closing the
    one gap left between "these bytes are that revision's" and "the engine is serving these bytes".

    A backend that serves either way and cannot make the argument is refused rather than waved
    through: the alternative is trusting an unverified serving, which is the failure this exists to
    catch.
    """
    if served.load_mode == LOAD_MODE_FULL_WEIGHTS:
        if served.weights is None:
            raise ValueError("a full-weights ServedModel carries no resolved weights to verify")
        if not isinstance(backend, WeightsVerifyingBackend):
            raise TypeError(
                f"backend {type(backend).__name__} serves the full weights {served.model_id} but "
                f"offers no assert_serves_weights, so nothing ties the engine to the directory "
                f"whose identity was verified. Nothing was verified."
            )
        backend.assert_serves_weights(served.weights.snapshot_dir)
        return
    if served.load_mode != LOAD_MODE_RUNTIME_ADAPTER:
        return
    if not isinstance(backend, AdapterVerifyingBackend):
        raise TypeError(
            f"backend {type(backend).__name__} serves the adapter at {served.adapter_dir} "
            f"un-merged but offers no assert_adapter_changes_output, so there is no way to tell an "
            f"applied adapter from a silently skipped one. Nothing was verified."
        )
    backend.assert_adapter_changes_output(ADAPTER_PROBE_PROMPTS)
