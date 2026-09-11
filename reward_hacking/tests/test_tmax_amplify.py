"""Task-arithmetic amplification (``reward_hacking/tmax/amplify.py``) on toy checkpoints.

Everything runs on CPU over a few tiny tensors. The claims pinned: alpha 0 reproduces the base,
alpha 1 the RL'd checkpoint, alpha 2 extrapolates past it; the output carries exactly the RL'd
checkpoint's tensor set (the base's vision tensors never leak in) in the RL'd checkpoint's storage
dtypes; the side files and a provenance sidecar travel with it; the built directory passes the same
content gate the probe applies before serving, and fails it once tampered with. The refusals --
a tensor the base lacks, a shape that differs, an output directory that already exists -- are each
fed their violation. The permuted control (``TestThePermutedControl``) pins what makes it a
control that can fail: every singular value and the Frobenius norm of a 2-D delta survive the
axis permutation exactly, 1-D and 3-D deltas keep their entries, the permutation is seeded by
(seed, tensor name, axis) rather than by order, alpha 0 still reproduces the base, and the
sidecar names the transform, the seed and the per-tensor cosine to the real delta.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from games.eval_model import (
    WEIGHTS_PROVENANCE_FILENAME,
    FullWeightsSource,
    resolve_full_weights,
    sha256_of_file,
)
from reward_hacking.tmax import amplify
from reward_hacking.tmax.amplify import (
    DELTA_TRANSFORM_IDENTITY,
    DELTA_TRANSFORM_PERMUTE_AXES,
    OUTPUT_WEIGHTS_FILENAME,
    PROVENANCE_SCHEMA,
    SAFETENSORS_INDEX_FILENAME,
    DeltaTransform,
    amplify_tensors,
    build_amplified_checkpoint,
    permute_delta_axes,
)

if TYPE_CHECKING:
    from pathlib import Path

LM = "model.language_model.layers.0"
A_LOG = f"{LM}.linear_attn.A_log"
BASE_TENSORS = {
    f"{LM}.mlp.down_proj.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16),
    A_LOG: torch.tensor([0.5, -0.25, 1.0], dtype=torch.float32),
    "model.visual.blocks.0.attn.qkv.weight": torch.ones(2, 2, dtype=torch.bfloat16),
}
RL_TENSORS = {
    f"{LM}.mlp.down_proj.weight": torch.tensor([[1.5, 2.0], [3.0, 3.0]], dtype=torch.bfloat16),
    # The release stores this one at bfloat16 where the base keeps float32: its "delta" is the cast
    # error, so the amplifier COPIES it from the release under every alpha (see the copied tests).
    A_LOG: torch.tensor([0.75, -0.25, 0.0], dtype=torch.bfloat16),
}
AMPLIFIED_TENSORS = {name: t for name, t in RL_TENSORS.items() if name != A_LOG}


def write_base(path: Path) -> Path:
    """A sharded base: two shards plus the index that maps tensors to them, like the hub's."""
    path.mkdir(parents=True)
    names = sorted(BASE_TENSORS)
    shards = {
        "model-00001-of-00002.safetensors": names[:2],
        "model-00002-of-00002.safetensors": names[2:],
    }
    for shard, members in shards.items():
        save_file({name: BASE_TENSORS[name].contiguous() for name in members}, str(path / shard))
    weight_map = {name: shard for shard, members in shards.items() for name in members}
    (path / SAFETENSORS_INDEX_FILENAME).write_text(json.dumps({"weight_map": weight_map}))
    (path / "config.json").write_text(json.dumps({"vision_config": {}}))
    return path


def write_rl(path: Path, tensors: dict[str, torch.Tensor] | None = None) -> Path:
    """A single-file release with its own tokenizer files and template."""
    path.mkdir(parents=True)
    save_file(
        {k: v.contiguous() for k, v in (tensors or RL_TENSORS).items()},
        str(path / OUTPUT_WEIGHTS_FILENAME),
    )
    (path / "config.json").write_text(json.dumps({"vision_config": {}}))
    (path / "generation_config.json").write_text(json.dumps({"eos_token_id": 1}))
    (path / "tokenizer_config.json").write_text(json.dumps({"model_max_length": 8}))
    (path / "chat_template.jinja").write_text("{{ messages }}")
    return path


def read_tensors(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(str(path), framework="pt") as handle:
        return {name: handle.get_tensor(name) for name in handle.keys()}  # noqa: SIM118


@pytest.fixture
def checkpoints(tmp_path: Path) -> tuple[Path, Path]:
    return write_base(tmp_path / "base"), write_rl(tmp_path / "rl")


class TestTheArithmetic:
    def expected(self, alpha: float) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for name, rl in AMPLIFIED_TENSORS.items():
            base = BASE_TENSORS[name].to(torch.float32)
            out[name] = (base + alpha * (rl.to(torch.float32) - base)).to(rl.dtype)
        out[A_LOG] = RL_TENSORS[A_LOG]
        return out

    @pytest.mark.parametrize("alpha", [0.0, 1.0, 1.5, 2.0])
    def test_it_is_base_plus_alpha_times_the_delta(
        self, checkpoints: tuple[Path, Path], alpha: float
    ) -> None:
        base, rl = checkpoints
        amplified, deltas = amplify_tensors(base_dir=base, rl_dir=rl, alpha=alpha)
        assert sorted(amplified) == sorted(RL_TENSORS)
        for name, want in self.expected(alpha).items():
            assert torch.equal(amplified[name], want), name
        assert len(deltas) == len(RL_TENSORS)

    def test_alpha_one_is_the_release_and_alpha_zero_the_base(
        self, checkpoints: tuple[Path, Path]
    ) -> None:
        base, rl = checkpoints
        at_one, _ = amplify_tensors(base_dir=base, rl_dir=rl, alpha=1.0)
        at_zero, _ = amplify_tensors(base_dir=base, rl_dir=rl, alpha=0.0)
        for name, tensor in RL_TENSORS.items():
            assert torch.equal(at_one[name], tensor)
        for name, tensor in AMPLIFIED_TENSORS.items():
            assert torch.equal(at_zero[name], BASE_TENSORS[name].to(tensor.dtype))
        # The float32-in-base / bf16-in-release tensor is the release at EVERY alpha.
        assert torch.equal(at_zero[A_LOG], RL_TENSORS[A_LOG])

    def test_storage_dtype_follows_the_release_not_the_base(
        self, checkpoints: tuple[Path, Path]
    ) -> None:
        base, rl = checkpoints
        amplified, deltas = amplify_tensors(base_dir=base, rl_dir=rl, alpha=1.5)
        assert amplified[A_LOG].dtype == torch.bfloat16
        assert {d.name: d.dtype for d in deltas}[A_LOG] == "bfloat16"

    def test_a_float32_base_tensor_released_in_bf16_is_copied_not_amplified(
        self, checkpoints: tuple[Path, Path]
    ) -> None:
        """Its whole delta is the release's cast error; scaling it would scale noise."""
        base, rl = checkpoints
        for alpha in (0.0, 1.5, -0.5):
            amplified, deltas = amplify_tensors(
                base_dir=base, rl_dir=rl, alpha=alpha, storage_dtype="float16"
            )
            assert torch.equal(amplified[A_LOG], RL_TENSORS[A_LOG]), alpha
            assert amplified[A_LOG].dtype == torch.bfloat16, "copied unchanged, dtype included"
            by_name = {d.name: d for d in deltas}
            assert by_name[A_LOG].copied_from_release is True
            assert by_name[A_LOG].realized_alpha_ratio == 1.0
            assert by_name[A_LOG].delta_norm > 0.0, "the cast error stays on the record"
            assert by_name[f"{LM}.mlp.down_proj.weight"].copied_from_release is False
            assert amplified[f"{LM}.mlp.down_proj.weight"].dtype == torch.float16
        # A permuted control copies it too: the control differs from the ladder only in what RL moved.
        permuted, _ = amplify_tensors(
            base_dir=base,
            rl_dir=rl,
            alpha=1.25,
            delta_transform=DeltaTransform(name=DELTA_TRANSFORM_PERMUTE_AXES, seed=3),
        )
        assert torch.equal(permuted[A_LOG], RL_TENSORS[A_LOG])

    def test_the_base_only_vision_tensor_never_leaks_in(
        self, checkpoints: tuple[Path, Path]
    ) -> None:
        base, rl = checkpoints
        amplified, _ = amplify_tensors(base_dir=base, rl_dir=rl, alpha=1.5)
        assert not any(name.startswith("model.visual.") for name in amplified)

    def test_the_delta_statistics_are_the_norms(self, checkpoints: tuple[Path, Path]) -> None:
        base, rl = checkpoints
        _, deltas = amplify_tensors(base_dir=base, rl_dir=rl, alpha=1.5)
        by_name = {d.name: d for d in deltas}
        down = by_name[f"{LM}.mlp.down_proj.weight"]
        base_tensor = BASE_TENSORS[down.name].to(torch.float32)
        delta = RL_TENSORS[down.name].to(torch.float32) - base_tensor
        assert down.base_norm == pytest.approx(float(base_tensor.norm()))
        assert down.delta_norm == pytest.approx(float(delta.norm()))
        assert down.relative_delta == pytest.approx(float(delta.norm() / base_tensor.norm()))

    def test_a_tensor_the_base_lacks_is_refused(self, tmp_path: Path) -> None:
        base = write_base(tmp_path / "base")
        rl = write_rl(tmp_path / "rl", {**RL_TENSORS, f"{LM}.extra.weight": torch.ones(2)})
        with pytest.raises(ValueError, match="no counterpart"):
            amplify_tensors(base_dir=base, rl_dir=rl, alpha=1.0)

    def test_a_base_language_model_tensor_the_release_lacks_is_refused(
        self, tmp_path: Path
    ) -> None:
        """The delta is defined on the FULL language model; only vision and MTP may be absent."""
        base = write_base(tmp_path / "base")
        rl = write_rl(
            tmp_path / "rl",
            {f"{LM}.mlp.down_proj.weight": RL_TENSORS[f"{LM}.mlp.down_proj.weight"]},
        )
        with pytest.raises(ValueError, match="not a complete language model"):
            amplify_tensors(base_dir=base, rl_dir=rl, alpha=1.0)

    def test_a_shape_that_differs_is_refused(self, tmp_path: Path) -> None:
        base = write_base(tmp_path / "base")
        rl = write_rl(
            tmp_path / "rl",
            {**RL_TENSORS, f"{LM}.linear_attn.A_log": torch.zeros(4, dtype=torch.bfloat16)},
        )
        with pytest.raises(ValueError, match="do not share an architecture"):
            amplify_tensors(base_dir=base, rl_dir=rl, alpha=1.0)


class TestTheBuiltUnit:
    def build(self, checkpoints: tuple[Path, Path], out: Path, alpha: float = 1.5):
        base, rl = checkpoints
        base_facts = resolve_full_weights(FullWeightsSource.parse(str(base), None))
        rl_facts = resolve_full_weights(FullWeightsSource.parse(str(rl), None))
        return build_amplified_checkpoint(
            base=base_facts, rl=rl_facts, alpha=alpha, out_dir=out, label=out.name
        )

    def test_the_directory_carries_tensors_side_files_and_a_sidecar(
        self, checkpoints: tuple[Path, Path], tmp_path: Path
    ) -> None:
        out = tmp_path / "unit-alpha1.5"
        report = self.build(checkpoints, out)
        written = read_tensors(out / OUTPUT_WEIGHTS_FILENAME)
        assert sorted(written) == sorted(RL_TENSORS)
        for name in (
            "config.json",
            "generation_config.json",
            "tokenizer_config.json",
            "chat_template.jinja",
        ):
            assert (out / name).is_file(), name
        sidecar = json.loads((out / WEIGHTS_PROVENANCE_FILENAME).read_text())
        assert sidecar["kind"] == "task-arithmetic-amplification"
        assert sidecar["alpha"] == 1.5
        assert sidecar["label"] == "unit-alpha1.5"
        assert sidecar["n_tensors"] == len(RL_TENSORS)
        assert sidecar["weights_sha256"] == {
            OUTPUT_WEIGHTS_FILENAME: sha256_of_file(out / OUTPUT_WEIGHTS_FILENAME)
        }
        assert sidecar["base"]["label"] == "base"
        assert sidecar["rl"]["label"] == "rl"
        assert len(sidecar["tensor_deltas"]) == len(RL_TENSORS)
        assert report.weights_sha256 == sidecar["weights_sha256"]
        assert report.dtypes == {"bfloat16": 2}
        assert sidecar["copied_from_release"] == [A_LOG]
        assert sidecar["n_copied_from_release"] == 1
        assert report.n_copied_from_release == 1

    def test_the_unit_passes_the_serving_gate_and_fails_it_once_tampered(
        self, checkpoints: tuple[Path, Path], tmp_path: Path
    ) -> None:
        out = tmp_path / "unit"
        self.build(checkpoints, out)
        facts = resolve_full_weights(FullWeightsSource.parse(str(out), None))
        assert facts.label == "unit"
        assert facts.declares_vision_config is True
        (out / OUTPUT_WEIGHTS_FILENAME).write_bytes(b"\x00" * 128)
        with pytest.raises(RuntimeError, match="changed after it was labelled"):
            resolve_full_weights(FullWeightsSource.parse(str(out), None))

    def test_an_existing_output_directory_is_never_overwritten(
        self, checkpoints: tuple[Path, Path], tmp_path: Path
    ) -> None:
        out = tmp_path / "unit"
        out.mkdir()
        with pytest.raises(FileExistsError, match="never overwritten"):
            self.build(checkpoints, out)

    def test_two_alphas_fingerprint_differently_and_alpha_one_matches_the_release(
        self, checkpoints: tuple[Path, Path], tmp_path: Path
    ) -> None:
        self.build(checkpoints, tmp_path / "a1", alpha=1.0)
        self.build(checkpoints, tmp_path / "a2", alpha=2.0)
        one = resolve_full_weights(FullWeightsSource.parse(str(tmp_path / "a1"), None))
        two = resolve_full_weights(FullWeightsSource.parse(str(tmp_path / "a2"), None))
        assert one.fingerprint != two.fingerprint
        for name, tensor in read_tensors(tmp_path / "a1" / OUTPUT_WEIGHTS_FILENAME).items():
            assert torch.equal(tensor, RL_TENSORS[name])


class TestTheCli:
    def test_local_inputs_build_a_unit_without_the_hub(
        self, checkpoints: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        base, rl = checkpoints
        monkeypatch.setattr(
            amplify, "resolve_full_weights", amplify.resolve_full_weights, raising=True
        )
        out = tmp_path / "cli-unit"
        assert (
            amplify.main(
                ["--base", str(base), "--rl", str(rl), "--alpha", "1.5", "--out", str(out)]
            )
            == 0
        )
        sidecar = json.loads((out / WEIGHTS_PROVENANCE_FILENAME).read_text())
        assert sidecar["label"] == "cli-unit"
        assert sidecar["alpha"] == 1.5

    def test_a_revision_on_a_local_input_is_refused(
        self, checkpoints: tuple[Path, Path], tmp_path: Path
    ) -> None:
        base, rl = checkpoints
        with pytest.raises(ValueError, match="no revision to pin"):
            amplify.main(
                [
                    "--base",
                    str(base),
                    "--base-revision",
                    "main",
                    "--rl",
                    str(rl),
                    "--alpha",
                    "1.5",
                    "--out",
                    str(tmp_path / "x"),
                ]
            )


WIDE = "model.language_model.layers.0.mlp.gate_proj.weight"
CONV = "model.language_model.layers.0.linear_attn.conv1d.weight"
BASE_TENSORS_WIDE = {
    WIDE: torch.zeros(6, 4, dtype=torch.bfloat16),
    # bf16 on both sides here, so this 1-D tensor is amplified (a float32 base would be copied).
    f"{LM}.linear_attn.A_log": torch.zeros(5, dtype=torch.bfloat16),
    CONV: torch.zeros(4, 1, 3, dtype=torch.bfloat16),
}
RL_TENSORS_WIDE = {
    WIDE: torch.arange(24, dtype=torch.float32).reshape(6, 4).to(torch.bfloat16),
    f"{LM}.linear_attn.A_log": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.bfloat16),
    CONV: torch.arange(12, dtype=torch.float32).reshape(4, 1, 3).to(torch.bfloat16),
}
"""A zero base with a distinct-entry release, so the delta IS the release and a permutation of it
can never coincide with the identity by accident (every entry is distinct along every axis)."""


def write_wide_pair(tmp_path: Path) -> tuple[Path, Path]:
    base = tmp_path / "base"
    base.mkdir()
    save_file(
        {k: v.contiguous() for k, v in BASE_TENSORS_WIDE.items()},
        str(base / OUTPUT_WEIGHTS_FILENAME),
    )
    (base / "config.json").write_text(json.dumps({"vision_config": {}}))
    return base, write_rl(tmp_path / "rl", RL_TENSORS_WIDE)


class TestThePermutedControl:
    """The control that can fail: a permuted copy of the real delta, matched on magnitude and spectrum."""

    def test_row_and_column_permutation_preserves_every_singular_value_and_the_norm(self) -> None:
        delta = torch.randn(7, 5, generator=torch.Generator().manual_seed(1))
        permuted = permute_delta_axes(delta, tensor_name=WIDE, seed=3)
        assert not torch.equal(permuted, delta)
        assert torch.allclose(
            torch.linalg.svdvals(permuted), torch.linalg.svdvals(delta), atol=1e-5
        )
        assert float(permuted.norm()) == pytest.approx(float(delta.norm()))
        # A permutation moves entries and never changes them: same multiset.
        assert torch.equal(permuted.flatten().sort().values, delta.flatten().sort().values)

    def test_every_axis_of_a_1d_and_a_3d_delta_is_permuted(self) -> None:
        one_d = torch.arange(9, dtype=torch.float32)
        three_d = torch.arange(24, dtype=torch.float32).reshape(4, 1, 6)
        for tensor in (one_d, three_d):
            permuted = permute_delta_axes(tensor, tensor_name="t", seed=11)
            assert permuted.shape == tensor.shape
            assert not torch.equal(permuted, tensor)
            assert torch.equal(permuted.flatten().sort().values, tensor.flatten().sort().values)

    def test_the_permutation_is_seeded_by_seed_and_tensor_name_not_by_order(self) -> None:
        delta = torch.randn(6, 4, generator=torch.Generator().manual_seed(2))
        again = permute_delta_axes(delta, tensor_name=WIDE, seed=5)
        assert torch.equal(permute_delta_axes(delta, tensor_name=WIDE, seed=5), again)
        assert not torch.equal(permute_delta_axes(delta, tensor_name=WIDE, seed=6), again)
        assert not torch.equal(permute_delta_axes(delta, tensor_name=CONV, seed=5), again)

    def test_a_length_one_axis_is_left_alone(self) -> None:
        column = torch.arange(5, dtype=torch.float32).reshape(5, 1)
        permuted = permute_delta_axes(column, tensor_name="c", seed=1)
        assert permuted.shape == (5, 1)
        assert torch.equal(permuted.flatten().sort().values, column.flatten())

    def test_the_transform_refuses_a_seed_without_a_permutation_and_a_permutation_without_a_seed(
        self,
    ) -> None:
        with pytest.raises(ValueError, match="nothing to seed"):
            DeltaTransform(name=DELTA_TRANSFORM_IDENTITY, seed=1)
        with pytest.raises(ValueError, match="needs a seed"):
            DeltaTransform(name=DELTA_TRANSFORM_PERMUTE_AXES, seed=None)
        with pytest.raises(ValueError, match="unknown delta transform"):
            DeltaTransform(name="shuffle", seed=1)

    def test_alpha_zero_is_still_the_base_and_alpha_one_is_no_longer_the_release(
        self, tmp_path: Path
    ) -> None:
        base, rl = write_wide_pair(tmp_path)
        transform = DeltaTransform(name=DELTA_TRANSFORM_PERMUTE_AXES, seed=7)
        at_zero, _ = amplify_tensors(base_dir=base, rl_dir=rl, alpha=0.0, delta_transform=transform)
        at_one, deltas = amplify_tensors(
            base_dir=base, rl_dir=rl, alpha=1.0, delta_transform=transform
        )
        for name, release in RL_TENSORS_WIDE.items():
            assert torch.equal(at_zero[name], BASE_TENSORS_WIDE[name].to(release.dtype)), name
            assert not torch.equal(at_one[name], release), name
            # Same entries, moved: the control is matched on magnitude tensor for tensor.
            assert torch.equal(
                at_one[name].flatten().sort().values, release.flatten().sort().values
            ), name
        by_name = {d.name: d for d in deltas}
        for name, release in RL_TENSORS_WIDE.items():
            assert by_name[name].delta_norm == pytest.approx(
                float(release.to(torch.float32).norm())
            )
            assert by_name[name].cosine_to_real_delta < 1.0

    def test_the_identity_records_a_cosine_of_one_everywhere(
        self, checkpoints: tuple[Path, Path]
    ) -> None:
        base, rl = checkpoints
        _, deltas = amplify_tensors(base_dir=base, rl_dir=rl, alpha=1.5)
        assert {d.cosine_to_real_delta for d in deltas} == {1.0}

    def test_the_sidecar_names_the_transform_the_seed_and_the_cosines(self, tmp_path: Path) -> None:
        base, rl = write_wide_pair(tmp_path)
        facts = [resolve_full_weights(FullWeightsSource.parse(str(p), None)) for p in (base, rl)]
        out = tmp_path / "permuted-alpha1.2"
        report = build_amplified_checkpoint(
            base=facts[0],
            rl=facts[1],
            alpha=1.2,
            out_dir=out,
            label=out.name,
            delta_transform=DeltaTransform(name=DELTA_TRANSFORM_PERMUTE_AXES, seed=20260903),
        )
        sidecar = json.loads((out / WEIGHTS_PROVENANCE_FILENAME).read_text())
        assert sidecar["schema"] == PROVENANCE_SCHEMA
        assert sidecar["delta_transform"] == DELTA_TRANSFORM_PERMUTE_AXES
        assert sidecar["permutation_seed"] == 20260903
        assert sidecar["alpha"] == 1.2
        assert 0.0 <= sidecar["mean_abs_cosine_to_real_delta"] < 1.0
        assert report.mean_abs_cosine_to_real_delta == sidecar["mean_abs_cosine_to_real_delta"]
        cosines = {d["name"]: d["cosine_to_real_delta"] for d in sidecar["tensor_deltas"]}
        assert set(cosines) == set(RL_TENSORS_WIDE)
        assert all(-1.0 <= c < 1.0 for c in cosines.values())
        real = tmp_path / "real-alpha1.2"
        build_amplified_checkpoint(
            base=facts[0], rl=facts[1], alpha=1.2, out_dir=real, label=real.name
        )
        real_sidecar = json.loads((real / WEIGHTS_PROVENANCE_FILENAME).read_text())
        assert real_sidecar["delta_transform"] == DELTA_TRANSFORM_IDENTITY
        assert real_sidecar["permutation_seed"] is None
        assert real_sidecar["mean_abs_cosine_to_real_delta"] == 1.0
        # Same alpha, different bytes: the two units can never pool under a resume gate.
        assert (
            resolve_full_weights(FullWeightsSource.parse(str(out), None)).fingerprint
            != resolve_full_weights(FullWeightsSource.parse(str(real), None)).fingerprint
        )

    def test_the_same_seed_rebuilds_the_same_unit(self, tmp_path: Path) -> None:
        base, rl = write_wide_pair(tmp_path)
        facts = [resolve_full_weights(FullWeightsSource.parse(str(p), None)) for p in (base, rl)]
        fingerprints = []
        for name in ("first", "second"):
            out = tmp_path / name
            build_amplified_checkpoint(
                base=facts[0],
                rl=facts[1],
                alpha=1.2,
                out_dir=out,
                label="same-label",
                delta_transform=DeltaTransform(name=DELTA_TRANSFORM_PERMUTE_AXES, seed=1),
            )
            fingerprints.append(
                resolve_full_weights(FullWeightsSource.parse(str(out), None)).fingerprint
            )
        assert fingerprints[0] == fingerprints[1]

    def test_the_cli_builds_a_permuted_control_and_refuses_a_half_specified_one(
        self, tmp_path: Path
    ) -> None:
        base, rl = write_wide_pair(tmp_path)
        out = tmp_path / "cli-permuted"
        argv = ["--base", str(base), "--rl", str(rl), "--alpha", "1.2", "--out", str(out)]
        assert (
            amplify.main([*argv, "--delta-transform", "permute-axes", "--permutation-seed", "4"])
            == 0
        )
        sidecar = json.loads((out / WEIGHTS_PROVENANCE_FILENAME).read_text())
        assert sidecar["delta_transform"] == "permute-axes"
        assert sidecar["permutation_seed"] == 4
        with pytest.raises(ValueError, match="needs a seed"):
            amplify.main([*argv[:-1], str(tmp_path / "x"), "--delta-transform", "permute-axes"])
        with pytest.raises(ValueError, match="nothing to seed"):
            amplify.main([*argv[:-1], str(tmp_path / "y"), "--permutation-seed", "4"])


class TestStorageDtypeAndRealizedAlpha:
    """bf16 snaps a one-ulp step back to one ulp; fp16 storage carries it; the sidecar says which."""

    ONE_BF16_ULP_AT_ONE = 2.0**-7

    def ulp_pair(self, tmp_path: Path) -> tuple[Path, Path]:
        """A base of ones and a release one bf16 ulp above it: the real 9B delta's typical entry."""
        name = "model.language_model.layers.0.mlp.gate_proj.weight"
        base = tmp_path / "base"
        base.mkdir()
        save_file(
            {name: torch.ones(4, 4, dtype=torch.bfloat16)}, str(base / OUTPUT_WEIGHTS_FILENAME)
        )
        (base / "config.json").write_text(json.dumps({"vision_config": {}, "dtype": "bfloat16"}))
        release = torch.full((4, 4), 1.0 + self.ONE_BF16_ULP_AT_ONE, dtype=torch.bfloat16)
        assert release[0, 0].item() == 1.0 + self.ONE_BF16_ULP_AT_ONE, (
            "the release must sit one ulp up"
        )
        rl = write_rl(tmp_path / "rl", {name: release})
        (rl / "config.json").write_text(
            json.dumps(
                {
                    "vision_config": {},
                    "dtype": "bfloat16",
                    "text_config": {"dtype": "bfloat16", "mamba_ssm_dtype": "float32"},
                }
            )
        )
        return base, rl

    def test_bf16_storage_snaps_a_small_alpha_back_to_the_release_and_fp16_does_not(
        self, tmp_path: Path
    ) -> None:
        base, rl = self.ulp_pair(tmp_path)
        released, deltas_release = amplify_tensors(base_dir=base, rl_dir=rl, alpha=1.2)
        (delta_release,) = deltas_release
        # 1.2 x one ulp rounds to one ulp: the stored tensor IS the release, realized alpha 1.0.
        assert delta_release.dtype == "bfloat16"
        assert delta_release.realized_alpha_ratio == pytest.approx(1.0 / 1.2)
        assert torch.equal(
            next(iter(released.values())),
            torch.full((4, 4), 1.0 + self.ONE_BF16_ULP_AT_ONE, dtype=torch.bfloat16),
        )
        half, deltas_half = amplify_tensors(
            base_dir=base, rl_dir=rl, alpha=1.2, storage_dtype="float16"
        )
        (delta_half,) = deltas_half
        assert delta_half.dtype == "float16"
        assert next(iter(half.values())).dtype == torch.float16
        # fp16 has three more mantissa bits: 1.2 x 8 fp16 ulps = 9.6 -> 10 ulps, realized 1.25/1.2.
        assert delta_half.realized_alpha_ratio == pytest.approx(1.25 / 1.2, rel=1e-3)
        full, deltas_full = amplify_tensors(
            base_dir=base, rl_dir=rl, alpha=1.2, storage_dtype="float32"
        )
        assert next(iter(full.values())).dtype == torch.float32
        assert deltas_full[0].realized_alpha_ratio == pytest.approx(1.0, rel=1e-5)

    def test_alpha_zero_realizes_one_by_convention_and_an_unknown_dtype_is_refused(
        self, tmp_path: Path
    ) -> None:
        base, rl = self.ulp_pair(tmp_path)
        _, deltas = amplify_tensors(base_dir=base, rl_dir=rl, alpha=0.0, storage_dtype="float16")
        assert deltas[0].realized_alpha_ratio == 1.0
        with pytest.raises(ValueError, match="unknown storage dtype"):
            amplify_tensors(base_dir=base, rl_dir=rl, alpha=1.0, storage_dtype="int8")

    def test_the_built_unit_rewrites_every_config_dtype_key_and_records_the_storage(
        self, tmp_path: Path
    ) -> None:
        base, rl = self.ulp_pair(tmp_path)
        facts = [resolve_full_weights(FullWeightsSource.parse(str(p), None)) for p in (base, rl)]
        out = tmp_path / "fp16-unit"
        report = build_amplified_checkpoint(
            base=facts[0],
            rl=facts[1],
            alpha=1.2,
            out_dir=out,
            label=out.name,
            storage_dtype="float16",
        )
        config = json.loads((out / "config.json").read_text())
        assert config["dtype"] == "float16"
        assert config["text_config"]["dtype"] == "float16"
        assert config["text_config"]["mamba_ssm_dtype"] == "float32", (
            "the DeltaNet state dtype is not storage"
        )
        assert "vision_config" in config
        sidecar = json.loads((out / WEIGHTS_PROVENANCE_FILENAME).read_text())
        assert sidecar["schema"] == PROVENANCE_SCHEMA
        assert sidecar["storage_dtype"] == "float16"
        assert sidecar["config_dtype_rewritten"] is True
        assert sidecar["dtypes"] == {"float16": 1}
        assert sidecar["mean_realized_alpha_ratio"] == pytest.approx(1.25 / 1.2, rel=1e-3)
        assert sidecar["tensor_deltas"][0]["realized_alpha_ratio"] == pytest.approx(
            1.25 / 1.2, rel=1e-3
        )
        assert report.storage_dtype == "float16"
        # The unit still passes the serving gate, and reads back as fp16.
        served = resolve_full_weights(FullWeightsSource.parse(str(out), None))
        assert served.declares_vision_config is True
        assert (
            next(iter(read_tensors(out / OUTPUT_WEIGHTS_FILENAME).values())).dtype == torch.float16
        )

    def test_a_config_naming_no_dtype_gains_one_under_a_named_storage(self, tmp_path: Path) -> None:
        """An engine on dtype=auto would otherwise guess; the unit states what it stores."""
        base, rl = write_base(tmp_path / "base"), write_rl(tmp_path / "rl")
        assert "dtype" not in json.loads((rl / "config.json").read_text())
        facts = [resolve_full_weights(FullWeightsSource.parse(str(p), None)) for p in (base, rl)]
        out = tmp_path / "fp16-no-key"
        build_amplified_checkpoint(
            base=facts[0],
            rl=facts[1],
            alpha=1.5,
            out_dir=out,
            label=out.name,
            storage_dtype="float16",
        )
        config = json.loads((out / "config.json").read_text())
        assert config["dtype"] == "float16"
        assert (
            json.loads((out / WEIGHTS_PROVENANCE_FILENAME).read_text())["config_dtype_rewritten"]
            is True
        )

    def test_release_storage_leaves_the_config_alone(self, tmp_path: Path) -> None:
        base, rl = self.ulp_pair(tmp_path)
        facts = [resolve_full_weights(FullWeightsSource.parse(str(p), None)) for p in (base, rl)]
        out = tmp_path / "release-unit"
        build_amplified_checkpoint(
            base=facts[0], rl=facts[1], alpha=1.2, out_dir=out, label=out.name
        )
        config = json.loads((out / "config.json").read_text())
        assert config["dtype"] == "bfloat16"
        sidecar = json.loads((out / WEIGHTS_PROVENANCE_FILENAME).read_text())
        assert sidecar["storage_dtype"] == "release"
        assert sidecar["config_dtype_rewritten"] is False
        assert sidecar["mean_realized_alpha_ratio"] == pytest.approx(1.0 / 1.2)

    def test_the_cli_takes_a_storage_dtype(self, tmp_path: Path) -> None:
        base, rl = self.ulp_pair(tmp_path)
        out = tmp_path / "cli-fp16"
        assert (
            amplify.main(
                [
                    "--base",
                    str(base),
                    "--rl",
                    str(rl),
                    "--alpha",
                    "1.2",
                    "--out",
                    str(out),
                    "--storage-dtype",
                    "float16",
                ]
            )
            == 0
        )
        sidecar = json.loads((out / WEIGHTS_PROVENANCE_FILENAME).read_text())
        assert sidecar["storage_dtype"] == "float16"
        assert json.loads((out / "config.json").read_text())["dtype"] == "float16"
