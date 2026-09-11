"""Where a Qwen3.5 language-model tensor sits, and where the weight-geometry tables live on disk.

The layout half places every safetensors key of the text model by layer and role, so the
per-family roll-ups in :mod:`reward_hacking.tmax.weight_geometry_tables` are exhaustive: a key that
does not match a known row is refused rather than dropped, because a family total that silently
omitted a tensor would read as a smaller update. The file-name half is shared by the writer
(:mod:`reward_hacking.tmax.weight_geometry`) and the reader (the tables module) so neither hardcodes
the other's paths. Torch-free.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from reward_hacking.tmax.artifacts import LANGUAGE_MODEL_KEY_PREFIX, LM_HEAD_KEY

TENSORS_FILENAME = "tensors.jsonl"
BASE_SPECTRA_FILENAME = "base_spectra.jsonl"
PAIRS_FILENAME = "pairs.jsonl"
GATE_HEADS_FILENAME = "deltanet_gate_heads.jsonl"
TOKEN_COUNTS_FILENAME = "token_counts.parquet"
RUN_MANIFEST_FILENAME = "run_manifest.json"
TOKEN_ROWS_PREFIX = "token_rows"

EMBED_TOKENS_KEY = f"{LANGUAGE_MODEL_KEY_PREFIX}embed_tokens.weight"
FINAL_NORM_KEY = f"{LANGUAGE_MODEL_KEY_PREFIX}norm.weight"


class ModuleClass(StrEnum):
    """One row of the Qwen3.5 language-model layout, named by what the projection does."""

    EMBED_TOKENS = "embed_tokens"
    LM_HEAD = "lm_head"
    FINAL_NORM = "final_norm"
    INPUT_LAYERNORM = "input_layernorm"
    POST_ATTENTION_LAYERNORM = "post_attention_layernorm"
    ATTN_Q = "attn_q_proj"
    ATTN_K = "attn_k_proj"
    ATTN_V = "attn_v_proj"
    ATTN_O = "attn_o_proj"
    ATTN_Q_NORM = "attn_q_norm"
    ATTN_K_NORM = "attn_k_norm"
    DELTANET_QKV = "deltanet_in_proj_qkv"
    DELTANET_Z = "deltanet_in_proj_z"
    DELTANET_DECAY_GATE = "deltanet_in_proj_a"
    DELTANET_UPDATE_GATE = "deltanet_in_proj_b"
    DELTANET_OUT = "deltanet_out_proj"
    DELTANET_CONV = "deltanet_conv1d"
    DELTANET_A_LOG = "deltanet_A_log"
    DELTANET_DT_BIAS = "deltanet_dt_bias"
    DELTANET_NORM = "deltanet_norm"
    MLP_GATE = "mlp_gate_proj"
    MLP_UP = "mlp_up_proj"
    MLP_DOWN = "mlp_down_proj"


class ModuleFamily(StrEnum):
    """The coarse grouping the readout tables roll up to."""

    EMBEDDINGS = "embeddings"
    LM_HEAD = "lm_head"
    ATTENTION = "attention"
    DELTANET_IN_PROJ = "deltanet_in_proj"
    DELTANET_OUT_PROJ = "deltanet_out_proj"
    DELTANET_STATE = "deltanet_state"
    MLP = "mlp"
    NORMS = "norms"


LAYER_SUFFIX_MODULES: dict[str, ModuleClass] = {
    "input_layernorm.weight": ModuleClass.INPUT_LAYERNORM,
    "post_attention_layernorm.weight": ModuleClass.POST_ATTENTION_LAYERNORM,
    "self_attn.q_proj.weight": ModuleClass.ATTN_Q,
    "self_attn.k_proj.weight": ModuleClass.ATTN_K,
    "self_attn.v_proj.weight": ModuleClass.ATTN_V,
    "self_attn.o_proj.weight": ModuleClass.ATTN_O,
    "self_attn.q_norm.weight": ModuleClass.ATTN_Q_NORM,
    "self_attn.k_norm.weight": ModuleClass.ATTN_K_NORM,
    "linear_attn.in_proj_qkv.weight": ModuleClass.DELTANET_QKV,
    "linear_attn.in_proj_z.weight": ModuleClass.DELTANET_Z,
    "linear_attn.in_proj_a.weight": ModuleClass.DELTANET_DECAY_GATE,
    "linear_attn.in_proj_b.weight": ModuleClass.DELTANET_UPDATE_GATE,
    "linear_attn.out_proj.weight": ModuleClass.DELTANET_OUT,
    "linear_attn.conv1d.weight": ModuleClass.DELTANET_CONV,
    "linear_attn.A_log": ModuleClass.DELTANET_A_LOG,
    "linear_attn.dt_bias": ModuleClass.DELTANET_DT_BIAS,
    "linear_attn.norm.weight": ModuleClass.DELTANET_NORM,
    "mlp.gate_proj.weight": ModuleClass.MLP_GATE,
    "mlp.up_proj.weight": ModuleClass.MLP_UP,
    "mlp.down_proj.weight": ModuleClass.MLP_DOWN,
}
"""Every per-layer tensor suffix of the layout (headers read 2026-09-02) and its role."""

MODULE_FAMILIES: dict[ModuleClass, ModuleFamily] = {
    ModuleClass.EMBED_TOKENS: ModuleFamily.EMBEDDINGS,
    ModuleClass.LM_HEAD: ModuleFamily.LM_HEAD,
    ModuleClass.FINAL_NORM: ModuleFamily.NORMS,
    ModuleClass.INPUT_LAYERNORM: ModuleFamily.NORMS,
    ModuleClass.POST_ATTENTION_LAYERNORM: ModuleFamily.NORMS,
    ModuleClass.ATTN_Q: ModuleFamily.ATTENTION,
    ModuleClass.ATTN_K: ModuleFamily.ATTENTION,
    ModuleClass.ATTN_V: ModuleFamily.ATTENTION,
    ModuleClass.ATTN_O: ModuleFamily.ATTENTION,
    ModuleClass.ATTN_Q_NORM: ModuleFamily.NORMS,
    ModuleClass.ATTN_K_NORM: ModuleFamily.NORMS,
    ModuleClass.DELTANET_QKV: ModuleFamily.DELTANET_IN_PROJ,
    ModuleClass.DELTANET_Z: ModuleFamily.DELTANET_IN_PROJ,
    ModuleClass.DELTANET_DECAY_GATE: ModuleFamily.DELTANET_IN_PROJ,
    ModuleClass.DELTANET_UPDATE_GATE: ModuleFamily.DELTANET_IN_PROJ,
    ModuleClass.DELTANET_OUT: ModuleFamily.DELTANET_OUT_PROJ,
    ModuleClass.DELTANET_CONV: ModuleFamily.DELTANET_STATE,
    ModuleClass.DELTANET_A_LOG: ModuleFamily.DELTANET_STATE,
    ModuleClass.DELTANET_DT_BIAS: ModuleFamily.DELTANET_STATE,
    ModuleClass.DELTANET_NORM: ModuleFamily.NORMS,
    ModuleClass.MLP_GATE: ModuleFamily.MLP,
    ModuleClass.MLP_UP: ModuleFamily.MLP,
    ModuleClass.MLP_DOWN: ModuleFamily.MLP,
}

GATE_HEAD_MODULES: frozenset[ModuleClass] = frozenset(
    {
        ModuleClass.DELTANET_DECAY_GATE,
        ModuleClass.DELTANET_UPDATE_GATE,
        ModuleClass.DELTANET_A_LOG,
        ModuleClass.DELTANET_DT_BIAS,
    }
)
"""The DeltaNet write-strength venue: one row (or one scalar) per recurrent head.

``in_proj_a`` feeds the per-head decay ``exp(-softplus(a + dt_bias) * exp(A_log))`` and
``in_proj_b`` the per-head write strength ``sigmoid(b)``, so their rows are the directions the
residual stream is read along to decide how hard the recurrent state is overwritten. A
residual-stream probe never sees this venue, which is why it gets its own table.
"""

TOKEN_ROW_MODULES: frozenset[ModuleClass] = frozenset(
    {ModuleClass.EMBED_TOKENS, ModuleClass.LM_HEAD}
)

_LAYER_KEY = re.compile(rf"^{re.escape(LANGUAGE_MODEL_KEY_PREFIX)}layers\.(\d+)\.(.+)$")


@dataclass(frozen=True)
class TensorSite:
    """Where one tensor sits in the network: its layer (``None`` outside the stack) and role."""

    name: str
    layer: int | None
    module: ModuleClass

    @property
    def family(self) -> ModuleFamily:
        """The coarse family the module rolls up to."""
        return MODULE_FAMILIES[self.module]

    def fields(self) -> dict[str, object]:
        """Return the columns every table carries for the tensor."""
        return {
            "name": self.name,
            "layer": self.layer,
            "module": str(self.module),
            "family": str(self.family),
        }


def classify_tensor(name: str) -> TensorSite:
    """Place a language-model tensor key; refuse one outside the Qwen3.5 layout rather than guess."""
    if name == LM_HEAD_KEY:
        return TensorSite(name, None, ModuleClass.LM_HEAD)
    if name == EMBED_TOKENS_KEY:
        return TensorSite(name, None, ModuleClass.EMBED_TOKENS)
    if name == FINAL_NORM_KEY:
        return TensorSite(name, None, ModuleClass.FINAL_NORM)
    match = _LAYER_KEY.match(name)
    if match is None:
        raise ValueError(f"tensor {name!r} is outside the Qwen3.5 language-model layout")
    module = LAYER_SUFFIX_MODULES.get(match.group(2))
    if module is None:
        raise ValueError(
            f"tensor {name!r} has an unknown module suffix {match.group(2)!r}; every row of the "
            f"layout must be classified or the family roll-ups silently drop it"
        )
    return TensorSite(name, int(match.group(1)), module)


def slug(label: str) -> str:
    """Turn a checkpoint label into a file-name fragment (``allenai/tmax-9b@step_500`` keeps its shape)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", label)


def token_rows_filename(checkpoint: str, module: ModuleClass) -> str:
    """Name the per-token-row parquet for one checkpoint's embedding or lm_head delta."""
    return f"{TOKEN_ROWS_PREFIX}_{slug(checkpoint)}_{module}.parquet"
