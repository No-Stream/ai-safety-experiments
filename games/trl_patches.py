"""Apply the three TRL 1.10 memory patches to an installed environment, refusing on any drift.

These are the source patches the 2026-08-19 rental (g7e, 95 GiB) needed to keep TRL's
old-log-probabilities pass from dying beside a resident vLLM colocate engine: chunk the fp32
logsumexp over positions, divide by temperature in place on the no-grad path, and clamp the
per-token-logps chunk to one row. They lived as instance-local scripts in the box's code
directory; landing them here ends the box-venv-diverges-from-the-repo failure mode without
turning them into runtime monkeypatches (the temperature and logsumexp fixes are mid-function,
so a runtime patch would have to carry whole copied method bodies -- the worst drift shape) or a
public TRL fork (real infrastructure for three lines, and a fork is an owner-maintained artifact
this repo cannot push).

**All three are dormant under the pre-registered arm configuration**, verified against the pinned
TRL 1.10 source this session: the old-logps pass runs only when generation and optimizer steps
misalign (`gradient_accumulation_steps % (steps_per_generation * num_iterations) != 0` -- aligned
at our defaults) or when `use_vllm and vllm_importance_sampling_correction`
(`trl/trainer/grpo_trainer.py:2631-2634`), and the correction is OFF at 32k budgets; the loss
path calls `_get_per_token_logps_and_entropies` only without Liger (`grpo_trainer.py:2986-2987`,
`:3071`), and the arms run Liger ON; the reference-model path needs `beta != 0`, and the arms run
`beta=0`. So applying them buys parity and the option of correction-ON diagnostics at small
budgets, not step time -- and even fully patched, correction-ON did NOT fit at the 32k production
shape (measured 2026-08-19), which is why `games.train` refuses that combination outright.

Anchor discipline is the whole safety story, same as the box scripts had: each patch replaces one
exact source snippet and refuses unless it appears exactly once. A TRL upgrade that moves the code
turns every apply into a loud failure here rather than a silent no-op at training time, and
`test_games_trl_patches.py` pins each anchor against the installed TRL so the drift is caught by
`make test` before any box bootstrap trips on it.
"""

from __future__ import annotations

import argparse
import logging
import sysconfig
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

APPLIED = "applied"
UNAPPLIED = "unapplied"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class SourcePatch:
    """One exact-anchor source replacement, and the measurement that justifies it."""

    name: str
    relative_path: str
    anchor: str
    replacement: str
    why: str


TRL_PATCHES: tuple[SourcePatch, ...] = (
    SourcePatch(
        name="logsumexp-position-chunk",
        relative_path="trl/trainer/utils.py",
        anchor=(
            "        logsumexp_values = torch.stack("
            "[torch.logsumexp(lg, dim=-1) for lg in logits])\n"
        ),
        replacement=(
            "        # Chunked over POSITIONS so the reduction temporaries stay ~2.5 GiB instead\n"
            "        # of ~30 GiB on a 32k-token row (measured on the 2026-08-19 rental). Exact:\n"
            "        # each position's logsumexp over the vocab dim is still computed whole.\n"
            "        logsumexp_values = torch.stack(\n"
            "            [\n"
            "                torch.cat(\n"
            "                    [torch.logsumexp(chunk, dim=-1) for chunk in lg.split(4096, dim=0)]\n"
            "                )\n"
            "                for lg in logits\n"
            "            ]\n"
            "        )\n"
        ),
        why=(
            "selective_log_softmax reduces a whole (positions x vocab) fp32 row at once; at a "
            "32,768-token budget the temporaries peak ~30 GiB and OOM beside a resident engine"
        ),
    ),
    SourcePatch(
        name="temperature-inplace",
        relative_path="trl/trainer/grpo_trainer.py",
        anchor=(
            "            logits = logits / self.temperature\n"
            "            completion_ids = input_ids_batch[:, -logits_to_keep:]"
        ),
        replacement=(
            "            # In place on the no-grad path: the out-of-place division allocates a\n"
            "            # second copy of a ~30 GiB fp32 logits chunk (2026-08-19 rental).\n"
            "            if logits.requires_grad:\n"
            "                logits = logits / self.temperature\n"
            "            else:\n"
            "                logits = logits.div_(self.temperature)\n"
            "            completion_ids = input_ids_batch[:, -logits_to_keep:]"
        ),
        why=(
            "the temperature division doubles the resident fp32 logits chunk; in-place is "
            "numerically identical and the requires_grad guard keeps any autograd caller exact"
        ),
    ),
    SourcePatch(
        name="logps-batch-size-one",
        relative_path="trl/trainer/grpo_trainer.py",
        anchor=(
            "        batch_size = batch_size or input_ids.size(0)"
            "  # Chunk inputs into smaller batches to reduce memory peak\n"
        ),
        replacement=(
            "        # Clamped to one row: at a 32k budget one padded row's fp32 logits are\n"
            "        # ~30 GiB, so multi-row chunks cannot sit beside a resident colocate engine\n"
            "        # (2026-08-19 rental). Costs sequential forwards on short diagnostic runs,\n"
            "        # which is the tested trade.\n"
            "        batch_size = 1\n"
        ),
        why=(
            "the caller hands the training micro-batch as the chunk size, and even two 32k rows "
            "of fp32 logits cannot fit beside the engine"
        ),
    ),
)


def site_packages_root() -> Path:
    """Return the running environment's site-packages, where the installed TRL lives."""
    return Path(sysconfig.get_paths()["purelib"])


def patch_state(patch: SourcePatch, *, root: Path) -> str:
    """Say whether one patch is unapplied, applied, or facing source it does not recognise."""
    source = (root / patch.relative_path).read_text(encoding="utf-8")
    if source.count(patch.anchor) == 1:
        return UNAPPLIED
    if patch.replacement in source:
        return APPLIED
    return UNKNOWN


def apply_patch(patch: SourcePatch, *, root: Path) -> str:
    """Apply one patch exactly once, idempotently, or refuse loudly on unrecognised source."""
    path = root / patch.relative_path
    state = patch_state(patch, root=root)
    if state == APPLIED:
        logger.info(f"trl patch already applied, {patch.name} in {path}")
        return state
    if state == UNKNOWN:
        raise RuntimeError(
            f"trl patch {patch.name!r} found neither its anchor nor its replacement in {path}: "
            f"the installed TRL has drifted from the 1.10.0 source these anchors were cut "
            f"against. Re-derive the patch against the installed version or drop it; do NOT "
            f"loosen the anchor, which is the only thing standing between this and silently "
            f"patching the wrong code."
        )
    source = path.read_text(encoding="utf-8")
    path.write_text(source.replace(patch.anchor, patch.replacement), encoding="utf-8")
    logger.info(f"trl patch applied, {patch.name} in {path} ({patch.why})")
    return APPLIED


def check_all(*, root: Path) -> dict[str, str]:
    """Report every patch's state without touching anything, for provenance and bootstrap logs."""
    return {patch.name: patch_state(patch, root=root) for patch in TRL_PATCHES}


def apply_all(*, root: Path) -> dict[str, str]:
    """Apply every patch, returning the resulting states; any drift raises before writes stop."""
    return {patch.name: apply_patch(patch, root=root) for patch in TRL_PATCHES}


def main(argv: list[str] | None = None) -> int:
    """Apply or report the TRL patches against this environment's installed TRL."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report each patch's state (applied/unapplied/unknown) and change nothing",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="site-packages root to operate on; defaults to this interpreter's own",
    )
    args = parser.parse_args(argv)
    root = args.root if args.root is not None else site_packages_root()
    states = check_all(root=root) if args.check else apply_all(root=root)
    for name, state in states.items():
        logger.info(f"trl patch state: {name} = {state}")
    if UNKNOWN in states.values():
        logger.error("at least one patch faces unrecognised TRL source; see states above")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
