"""Pin the eval loading ladder, and watch its one behavioural guard fail before trusting it.

Offline and CPU-only: no model loads, no engine starts, no adapter is really merged. Every claim
here is about *policy* -- which rung a backend gets, what kwargs that rung needs, and what the trace
records -- plus one test of the guard that catches a silently unapplied adapter, driven through the
real method with a stub engine.

:class:`TestLadder` is the policy. Its load-bearing case is not the happy path but
`test_a_backend_that_can_do_neither_falls_to_bf16`, which is the rung whose measurements are
attenuated: it has to be reachable, and it has to mark itself.

:class:`TestAdapterFacts` covers the two budgets read off a checkpoint rather than hardcoded. A rank
over the engine's slot width is refused outright and a target list that omits a trained module turns
a hard failure into a silent partial application, so neither may be defaulted.

:class:`TestVerifyServedModel` is the guard, and
`test_an_adapter_that_changes_nothing_is_refused` is the sabotage: an engine whose adapter produces
byte-identical token ids is exactly what upstream vLLM reports on this model family, and the whole
reason the check exists is that nothing else about that state looks wrong. Watched to fail here, and
also against a real engine on the L4.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import pytest
import torch

from games import eval_model
from games.eval_model import (
    ADAPTER_PROBE_PROMPTS,
    ATTENUATED_MODES,
    LOAD_MODE_BASE,
    LOAD_MODE_FULL_WEIGHTS,
    LOAD_MODE_MERGED_BF16,
    LOAD_MODE_MERGED_FP32,
    LOAD_MODE_MOCK_NO_LOAD,
    LOAD_MODE_RUNTIME_ADAPTER,
    MOCK_BACKEND_KIND,
    WEIGHTS_PROVENANCE_FILENAME,
    AdapterFacts,
    BackendCapability,
    FullWeightsSource,
    ServedModel,
    choose_load_mode,
    read_adapter_facts,
    resolve_full_weights,
    resolve_served_model,
    verify_served_model,
)
from games.lora import ADAPTER_CONFIG_FILENAME
from games.report import MOCK_BACKEND_KIND as REPORT_MOCK_BACKEND_KIND
from reward_hacking import model_backend

if TYPE_CHECKING:
    from collections.abc import Sequence

BASE_MODEL = "Qwen/Qwen3.5-2B"

# The twelve modules a Qwen3.5 LoRA run adapts, DeltaNet projections included; see
# grpo.throughput.ADAPTABLE_PROJECTIONS, which is where a real adapter's list comes from.
TRAINED_TARGETS = (
    "down_proj",
    "gate_proj",
    "in_proj_a",
    "in_proj_b",
    "in_proj_qkv",
    "in_proj_z",
    "k_proj",
    "o_proj",
    "out_proj",
    "q_proj",
    "up_proj",
    "v_proj",
)


def make_adapter_dir(
    path: Path, *, rank: int = 16, targets: Sequence[str] = TRAINED_TARGETS
) -> Path:
    """Write the one file this module reads off a checkpoint: the PEFT adapter config."""
    path.mkdir(parents=True, exist_ok=True)
    (path / ADAPTER_CONFIG_FILENAME).write_text(
        json.dumps(
            {"base_model_name_or_path": BASE_MODEL, "r": rank, "target_modules": list(targets)}
        ),
        encoding="utf-8",
    )
    return path


class RecordingMerge:
    """Stand-in for `export_merged_checkpoint`, so a merge rung is testable without weights."""

    def __init__(self) -> None:
        self.calls: list[tuple[Path, str, Path, torch.dtype]] = []

    def __call__(
        self,
        adapter_dir: Path,
        base_model_id: str,
        out_dir: Path,
        *,
        dtype: torch.dtype,
        **_: object,
    ) -> Path:
        self.calls.append((adapter_dir, base_model_id, out_dir, dtype))
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir


@pytest.fixture
def merge(monkeypatch: pytest.MonkeyPatch) -> RecordingMerge:
    """Replace the real merge, which would download and load a model."""
    recorder = RecordingMerge()
    monkeypatch.setattr(eval_model, "export_merged_checkpoint", recorder)
    return recorder


class TestLadder:
    """Which rung each backend kind gets, and what that rung asks the backend to do."""

    def test_vllm_serves_the_adapter_un_merged(self) -> None:
        assert choose_load_mode("vllm") == LOAD_MODE_RUNTIME_ADAPTER

    def test_hf_falls_to_a_float32_merge(self) -> None:
        assert choose_load_mode("hf") == LOAD_MODE_MERGED_FP32

    def test_an_unchecked_backend_is_refused_rather_than_defaulted(self) -> None:
        """Defaulting would silently pick the rung that rounds the trained delta away."""
        with pytest.raises(ValueError, match="no serving capability recorded"):
            choose_load_mode("bedrock")

    def test_a_backend_that_can_do_neither_falls_to_bf16(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(
            eval_model.BACKEND_CAPABILITIES,
            "vllm",
            BackendCapability(serves_runtime_adapter=False, serves_float32_weights=False),
        )
        assert choose_load_mode("vllm") == LOAD_MODE_MERGED_BF16

    def test_the_bf16_rung_marks_itself_unfaithful(self) -> None:
        """The whole point of recording a mode: a bf16 merge cannot hold the delta it was given."""
        assert LOAD_MODE_MERGED_BF16 in ATTENUATED_MODES
        served = ServedModel(model_id="d", load_mode=LOAD_MODE_MERGED_BF16, adapter_dir=Path("a"))
        assert served.delta_faithful is False
        assert served.provenance["model_delta_faithful"] is False

    def test_every_other_rung_is_faithful(self) -> None:
        for mode in (LOAD_MODE_BASE, LOAD_MODE_RUNTIME_ADAPTER, LOAD_MODE_MERGED_FP32):
            served = ServedModel(model_id="d", load_mode=mode, adapter_dir=None)
            assert served.delta_faithful is True, mode


class TestResolveServedModel:
    """End to end over the resolution seam, with the merge stubbed out."""

    def test_no_checkpoint_serves_the_base_model_untouched(self, merge: RecordingMerge) -> None:
        served = resolve_served_model(
            checkpoint=None,
            base_model=BASE_MODEL,
            backend_kind="vllm",
            merge_root=Path("/nonexistent"),
            merge_label="x-",
        )
        assert served.load_mode == LOAD_MODE_BASE
        assert served.model_id == BASE_MODEL
        assert served.adapter_dir is None
        assert served.merged_dir is None
        assert served.backend_kwargs == {}
        assert merge.calls == []

    def test_immutable_source_loads_under_the_canonical_base_identity(self, tmp_path: Path) -> None:
        snapshot = str(tmp_path / "immutable-snapshot")
        served = resolve_served_model(
            checkpoint=None,
            base_model=BASE_MODEL,
            base_model_source=snapshot,
            backend_kind="vllm",
            merge_root=tmp_path / "merges",
            merge_label="base-",
        )

        assert served.model_id == BASE_MODEL
        assert served.backend_kwargs == {"model_path": snapshot}

    def test_vllm_passes_the_adapter_through_without_merging(
        self, tmp_path: Path, merge: RecordingMerge
    ) -> None:
        checkpoint = make_adapter_dir(tmp_path / "checkpoint-70")
        served = resolve_served_model(
            checkpoint=checkpoint,
            base_model=BASE_MODEL,
            backend_kind="vllm",
            merge_root=tmp_path / "merges",
            merge_label="arm-step-70-",
        )
        assert served.load_mode == LOAD_MODE_RUNTIME_ADAPTER
        # The engine loads the BASE model; the adapter rides alongside it.
        assert served.model_id == BASE_MODEL
        assert served.adapter_dir == checkpoint
        assert served.merged_dir is None
        assert merge.calls == [], "the un-merged rung must not write a merge"
        assert served.backend_kwargs == {
            "lora_adapter": str(checkpoint),
            "enable_lora": True,
            "max_lora_rank": 16,
            "lora_target_modules": list(TRAINED_TARGETS),
        }

    def test_runtime_adapter_keeps_the_immutable_base_load_source(
        self, tmp_path: Path, merge: RecordingMerge
    ) -> None:
        checkpoint = make_adapter_dir(tmp_path / "checkpoint-70")
        snapshot = str(tmp_path / "immutable-snapshot")
        served = resolve_served_model(
            checkpoint=checkpoint,
            base_model=BASE_MODEL,
            base_model_source=snapshot,
            backend_kind="vllm",
            merge_root=tmp_path / "merges",
            merge_label="arm-step-70-",
        )

        assert served.model_id == BASE_MODEL
        assert served.backend_kwargs["model_path"] == snapshot
        assert served.backend_kwargs["lora_adapter"] == str(checkpoint)
        assert merge.calls == []

    def test_the_deltanet_projections_reach_the_engine(self, tmp_path: Path) -> None:
        """A target list missing these adapts a quarter of the stack and says nothing.

        Qwen3.5 stacks three Gated DeltaNet layers per full-attention layer, so an engine told only
        about q/k/v/o would serve base weights for three quarters of the model's attention.
        """
        checkpoint = make_adapter_dir(tmp_path / "checkpoint-70")
        facts = read_adapter_facts(checkpoint)
        for projection in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
            assert projection in facts.target_modules

    def test_hf_merges_at_float32_and_says_to_load_it_that_way(
        self, tmp_path: Path, merge: RecordingMerge
    ) -> None:
        checkpoint = make_adapter_dir(tmp_path / "checkpoint-70")
        served = resolve_served_model(
            checkpoint=checkpoint,
            base_model=BASE_MODEL,
            backend_kind="hf",
            merge_root=tmp_path / "merges",
            merge_label="arm-step-70-",
        )
        assert served.load_mode == LOAD_MODE_MERGED_FP32
        assert served.merged_dir is not None
        assert served.model_id == str(served.merged_dir)
        assert len(merge.calls) == 1
        assert merge.calls[0][3] == torch.float32
        # Load-bearing: without it the backend picks its own dtype and rounds the export back to
        # bf16, undoing the merge this rung exists to make faithful.
        assert served.backend_kwargs == {"dtype": torch.float32}

    def test_the_bf16_rung_merges_at_bf16_and_warns(
        self,
        tmp_path: Path,
        merge: RecordingMerge,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setitem(
            eval_model.BACKEND_CAPABILITIES,
            "vllm",
            BackendCapability(serves_runtime_adapter=False, serves_float32_weights=False),
        )
        checkpoint = make_adapter_dir(tmp_path / "checkpoint-70")
        with caplog.at_level("WARNING"):
            served = resolve_served_model(
                checkpoint=checkpoint,
                base_model=BASE_MODEL,
                backend_kind="vllm",
                merge_root=tmp_path / "merges",
                merge_label="arm-step-70-",
            )
        assert served.load_mode == LOAD_MODE_MERGED_BF16
        assert merge.calls[0][3] == torch.bfloat16
        assert served.backend_kwargs == {}
        # A reader of the log has to be told the numbers are attenuated, not just which dtype ran.
        assert "attenuated" in caplog.text

    def test_each_merge_gets_its_own_directory(self, tmp_path: Path, merge: RecordingMerge) -> None:
        """A leftover from a crashed export must never be written into or mistaken for this one."""
        checkpoint = make_adapter_dir(tmp_path / "checkpoint-70")
        dirs = {
            resolve_served_model(
                checkpoint=checkpoint,
                base_model=BASE_MODEL,
                backend_kind="hf",
                merge_root=tmp_path / "merges",
                merge_label="arm-step-70-",
            ).merged_dir
            for _ in range(2)
        }
        assert len(dirs) == 2

    def test_the_mock_backend_neither_merges_nor_serves(
        self, tmp_path: Path, merge: RecordingMerge, caplog: pytest.LogCaptureFixture
    ) -> None:
        checkpoint = make_adapter_dir(tmp_path / "checkpoint-70")
        with caplog.at_level("WARNING"):
            served = resolve_served_model(
                checkpoint=checkpoint,
                base_model=BASE_MODEL,
                backend_kind=MOCK_BACKEND_KIND,
                merge_root=tmp_path / "merges",
                merge_label="arm-step-70-",
            )
        assert served.load_mode == LOAD_MODE_MOCK_NO_LOAD
        assert served.merged_dir is None
        assert merge.calls == []
        assert "no model runs" in caplog.text

    def test_the_mock_kind_matches_the_reporting_layer(self) -> None:
        """The one string this module duplicates rather than imports; see its definition."""
        assert MOCK_BACKEND_KIND == REPORT_MOCK_BACKEND_KIND

    def test_provenance_names_the_mode_and_the_adapter(self, tmp_path: Path) -> None:
        checkpoint = make_adapter_dir(tmp_path / "checkpoint-70")
        served = resolve_served_model(
            checkpoint=checkpoint,
            base_model=BASE_MODEL,
            backend_kind="vllm",
            merge_root=tmp_path / "merges",
            merge_label="arm-step-70-",
        )
        assert served.provenance == {
            "model_load_mode": LOAD_MODE_RUNTIME_ADAPTER,
            "model_delta_faithful": True,
            "model_served_id": BASE_MODEL,
            "model_adapter_dir": str(checkpoint),
            "model_full_weights": None,
            "model_weights_commit_sha": None,
            "model_weights_fingerprint": None,
            "model_weights_chat_template_sha256": None,
        }


class TestAdapterFacts:
    """The two budgets read off a checkpoint, neither of which may be hardcoded or guessed."""

    @pytest.mark.parametrize(
        ("rank", "slot"), [(1, 1), (8, 8), (16, 16), (17, 32), (24, 32), (512, 512)]
    )
    def test_a_rank_rounds_up_to_the_next_slot_vllm_offers(self, rank: int, slot: int) -> None:
        assert AdapterFacts(rank=rank, target_modules=("q_proj",)).vllm_lora_rank == slot

    def test_a_rank_past_the_largest_slot_is_refused(self) -> None:
        with pytest.raises(ValueError, match="exceeds the largest LoRA slot"):
            _ = AdapterFacts(rank=1024, target_modules=("q_proj",)).vllm_lora_rank

    def test_a_missing_config_is_not_an_adapter(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not a PEFT adapter directory"):
            read_adapter_facts(tmp_path)

    def test_a_config_without_a_rank_is_refused(self, tmp_path: Path) -> None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / ADAPTER_CONFIG_FILENAME).write_text(
            json.dumps({"target_modules": ["q_proj"]}), encoding="utf-8"
        )
        with pytest.raises(TypeError, match="no integer rank"):
            read_adapter_facts(tmp_path)

    def test_a_config_without_target_modules_is_refused(self, tmp_path: Path) -> None:
        """Nothing would then pin which modules the engine must adapt, and a skip is silent."""
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / ADAPTER_CONFIG_FILENAME).write_text(json.dumps({"r": 16}), encoding="utf-8")
        with pytest.raises(ValueError, match="no target_modules"):
            read_adapter_facts(tmp_path)

    def test_targets_come_back_sorted_so_kwargs_are_stable(self, tmp_path: Path) -> None:
        checkpoint = make_adapter_dir(tmp_path / "c", targets=("v_proj", "q_proj", "in_proj_z"))
        assert read_adapter_facts(checkpoint).target_modules == ("in_proj_z", "q_proj", "v_proj")


class StubVerifyingBackend:
    """A backend that records the prompts it was asked to verify with."""

    def __init__(self) -> None:
        self.verified_with: list[Sequence[str]] = []

    def assert_adapter_changes_output(self, prompts: Sequence[str]) -> None:
        self.verified_with.append(list(prompts))


class StubUnverifiableBackend:
    """A backend with no way to argue that an adapter took effect."""


class TestVerifyServedModel:
    """Whether an un-merged adapter is made to prove itself before an eval spends GPU time."""

    def test_the_un_merged_rung_is_verified(self) -> None:
        backend = StubVerifyingBackend()
        verify_served_model(
            backend,
            ServedModel(
                model_id=BASE_MODEL,
                load_mode=LOAD_MODE_RUNTIME_ADAPTER,
                adapter_dir=Path("checkpoint-70"),
            ),
        )
        assert backend.verified_with == [list(ADAPTER_PROBE_PROMPTS)]

    @pytest.mark.parametrize(
        "mode",
        [LOAD_MODE_BASE, LOAD_MODE_MERGED_FP32, LOAD_MODE_MERGED_BF16, LOAD_MODE_MOCK_NO_LOAD],
    )
    def test_no_other_rung_is_checked_this_way(self, mode: str) -> None:
        """A merge is verified where it happens, and a base model has no adapter to check."""
        backend = StubVerifyingBackend()
        verify_served_model(backend, ServedModel(model_id="d", load_mode=mode, adapter_dir=None))
        assert backend.verified_with == []

    def test_a_backend_that_cannot_prove_it_is_refused_not_waved_through(self) -> None:
        with pytest.raises(TypeError, match="no assert_adapter_changes_output"):
            verify_served_model(
                StubUnverifiableBackend(),
                ServedModel(
                    model_id=BASE_MODEL,
                    load_mode=LOAD_MODE_RUNTIME_ADAPTER,
                    adapter_dir=Path("checkpoint-70"),
                ),
            )

    def test_the_probe_prompts_say_nothing_about_the_games_corpus(self) -> None:
        """They run against a trained policy, so they must not carry eval material into one."""
        assert ADAPTER_PROBE_PROMPTS
        for prompt in ADAPTER_PROBE_PROMPTS:
            assert "cooperate" not in prompt.lower()
            assert "defect" not in prompt.lower()


class StubCompletionOutput:
    """One vLLM `CompletionOutput`, reduced to the two fields the guard compares on."""

    def __init__(self, token_ids: Sequence[int], cumulative_logprob: float) -> None:
        self.token_ids = list(token_ids)
        self.cumulative_logprob = cumulative_logprob


class StubRequestOutput:
    def __init__(self, token_ids: Sequence[int], cumulative_logprob: float) -> None:
        self.outputs = [StubCompletionOutput(token_ids, cumulative_logprob)]


class StubEngine:
    """A vLLM engine that answers with fixed token ids and logprobs per arm, adapter or not.

    `adapted` is what it returns when a `lora_request` is passed and `base` what it returns
    without, which is exactly the axis the guard measures. The logprobs default to a value derived
    from the ids, so equal ids mean equal logprobs -- the silently-unapplied-adapter state --
    unless a test dissociates them explicitly (an applied adapter whose greedy prefix happens not
    to move shifts the logprob while leaving the ids alone).
    """

    def __init__(
        self,
        *,
        base: Sequence[int],
        adapted: Sequence[int],
        base_logprob: float | None = None,
        adapted_logprob: float | None = None,
    ) -> None:
        self.base = list(base)
        self.adapted = list(adapted)
        self.base_logprob = float(-sum(base)) if base_logprob is None else base_logprob
        self.adapted_logprob = float(-sum(adapted)) if adapted_logprob is None else adapted_logprob
        self.calls = 0

    def generate(
        self, prompts: Sequence[str], _params: object, lora_request: object = None
    ) -> list[StubRequestOutput]:
        self.calls += 1
        adapted = lora_request is not None
        ids = self.adapted if adapted else self.base
        logprob = self.adapted_logprob if adapted else self.base_logprob
        return [StubRequestOutput(ids, logprob) for _ in prompts]


class StubTokenizer:
    def apply_chat_template(self, messages: object, **_: object) -> str:
        return f"<chat>{messages}"


class StubSamplingParams:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class StubVllmModule:
    SamplingParams = StubSamplingParams


def make_vllm_backend(
    monkeypatch: pytest.MonkeyPatch,
    *,
    base: Sequence[int],
    adapted: Sequence[int],
    base_logprob: float | None = None,
    adapted_logprob: float | None = None,
) -> Any:
    """Build a `VLLMBackend` around a stub engine, without starting one or importing vllm.

    Constructed through `__new__` because the real `__init__` builds an engine, which needs a GPU
    and minutes. The guard under test reads only the four attributes set here, so this exercises
    the real comparison rather than a re-implementation of it -- which is the whole point: the
    sabotage below has to fail the code that ships.
    """
    monkeypatch.setattr(
        model_backend.importlib, "import_module", lambda _name: StubVllmModule(), raising=True
    )
    backend = object.__new__(model_backend.VLLMBackend)
    backend._llm = StubEngine(
        base=base, adapted=adapted, base_logprob=base_logprob, adapted_logprob=adapted_logprob
    )
    backend._tokenizer = StubTokenizer()
    backend._lora_request = object()
    backend.thinking = False
    backend.lora_adapter = "/checkpoints/checkpoint-70"
    return backend


class TestAdapterChangesOutputGuard:
    """The guard itself, driven through the shipping method with a stub engine.

    This is the check that stands between an un-merged adapter and a whole arm of GPU time, so it
    is the one that has to have been watched to fail.
    """

    def test_an_adapter_that_moves_output_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = make_vllm_backend(monkeypatch, base=[1, 2, 3], adapted=[1, 9, 3])
        backend.assert_adapter_changes_output(["a prompt"])

    def test_an_adapter_that_changes_nothing_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The sabotage. Identical token ids with and without the adapter is the exact state
        upstream vLLM reports on this hybrid-attention family: the checkpoint loads, generation is
        fluent, and base weights are being served under a trained checkpoint's name.
        """
        backend = make_vllm_backend(monkeypatch, base=[1, 2, 3], adapted=[1, 2, 3])
        with pytest.raises(RuntimeError, match="changed nothing"):
            backend.assert_adapter_changes_output(["a prompt"])

    def test_an_adapter_that_moves_only_the_logprob_passes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The false alarm the token-id-only probe raised on a real trained checkpoint: a small
        delta can leave every short greedy prefix untouched while shifting every logprob (measured
        on the twin-pd-group step-70 adapter, 2026-08-22). An applied adapter must pass."""
        backend = make_vllm_backend(
            monkeypatch,
            base=[1, 2, 3],
            adapted=[1, 2, 3],
            base_logprob=-42.0,
            adapted_logprob=-42.5,
        )
        backend.assert_adapter_changes_output(["a prompt"])

    def test_identical_logprobs_alone_do_not_rescue_identical_ids(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A skipped adapter contributes bitwise zero, so ids AND logprobs both matching base is
        the no-op signature and stays refused -- the logprob signal widens what can pass, never
        what counts as applied."""
        backend = make_vllm_backend(
            monkeypatch,
            base=[1, 2, 3],
            adapted=[1, 2, 3],
            base_logprob=-42.0,
            adapted_logprob=-42.0,
        )
        with pytest.raises(RuntimeError, match="changed nothing"):
            backend.assert_adapter_changes_output(["a prompt"])

    def test_it_compares_token_ids_not_decoded_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two token sequences can decode to one string, so text would miss a real difference."""
        backend = make_vllm_backend(monkeypatch, base=[1, 2], adapted=[1, 2, 2])
        backend.assert_adapter_changes_output(["a prompt"])

    def test_it_generates_both_arms(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One call could only compare the adapter against itself."""
        backend = make_vllm_backend(monkeypatch, base=[1], adapted=[2])
        backend.assert_adapter_changes_output(["a prompt"])
        assert backend._llm.calls == 2

    def test_no_adapter_to_check_is_an_error_not_a_pass(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = make_vllm_backend(monkeypatch, base=[1], adapted=[2])
        backend._lora_request = None
        with pytest.raises(ValueError, match="built without lora_adapter"):
            backend.assert_adapter_changes_output(["a prompt"])

    def test_no_prompts_is_an_error_not_a_pass(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A guard that compares nothing and returns is the shape this repo refuses to ship."""
        backend = make_vllm_backend(monkeypatch, base=[1], adapted=[2])
        with pytest.raises(ValueError, match="no prompts"):
            backend.assert_adapter_changes_output([])


# --- the full-weights rung: a complete checkpoint served as-is, identity proven on the way ----------

FAKE_TENSOR_BYTES = b"\x00" * 64
FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def write_weights_dir(
    path: Path,
    *,
    tensor_bytes: bytes = FAKE_TENSOR_BYTES,
    vision_config: bool = True,
    template: str | None = "{{ messages }}",
) -> Path:
    """A minimal checkpoint directory: config, one tensor file, and optionally a chat template."""
    path.mkdir(parents=True, exist_ok=True)
    config: dict[str, object] = {"architectures": ["Qwen3_5ForConditionalGeneration"]}
    if vision_config:
        config["vision_config"] = {"depth": 1}
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "model.safetensors").write_bytes(tensor_bytes)
    if template is not None:
        (path / "chat_template.jinja").write_text(template, encoding="utf-8")
    return path


class StubSibling:
    def __init__(self, rfilename: str, sha256: str | None) -> None:
        self.rfilename = rfilename
        self.lfs = None if sha256 is None else type("Lfs", (), {"sha256": sha256})()


class StubModelInfo:
    def __init__(self, sha: str, siblings: list[StubSibling]) -> None:
        self.sha = sha
        self.siblings = siblings


class StubHfApi:
    """Answers ``model_info`` from a canned record and remembers what it was asked."""

    instances: ClassVar[list[StubHfApi]] = []

    def __init__(self, info: StubModelInfo | None = None) -> None:
        self.info = info
        self.calls: list[dict[str, object]] = []
        StubHfApi.instances.append(self)

    def model_info(self, repo_id: str, *, revision: str, files_metadata: bool) -> StubModelInfo:
        self.calls.append(
            {"repo_id": repo_id, "revision": revision, "files_metadata": files_metadata}
        )
        if self.info is None:
            raise AssertionError("the hub must not be consulted on this path")
        return self.info


class RecordingSnapshotDownload:
    def __init__(self, result: Path) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def __call__(self, repo_id: str, **kwargs: object) -> str:
        self.calls.append({"repo_id": repo_id, **kwargs})
        return str(self.result)


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """A fake hub whose one repo has one tensor file, cached at a commit-named snapshot dir."""
    snapshot = write_weights_dir(tmp_path / "snapshots" / FAKE_COMMIT)
    digest = eval_model.sha256_of_file(snapshot / "model.safetensors")
    info = StubModelInfo(FAKE_COMMIT, [StubSibling("model.safetensors", digest)])
    download = RecordingSnapshotDownload(snapshot)
    StubHfApi.instances.clear()
    monkeypatch.setattr(eval_model, "HfApi", lambda: StubHfApi(info))
    monkeypatch.setattr(eval_model, "snapshot_download", download)
    return {"snapshot": snapshot, "digest": digest, "info": info, "download": download}


class TestFullWeightsSource:
    def test_an_existing_directory_is_local(self, tmp_path: Path) -> None:
        source = FullWeightsSource.parse(str(tmp_path), None)
        assert source.local_dir == tmp_path
        assert source.repo_id is None
        assert source.label == tmp_path.name

    def test_a_repo_id_with_a_revision_is_a_hub_source(self) -> None:
        source = FullWeightsSource.parse("allenai/tmax-4b", "step_300")
        assert source == FullWeightsSource(
            repo_id="allenai/tmax-4b", revision="step_300", local_dir=None
        )
        assert source.label == "allenai/tmax-4b@step_300"

    def test_a_hub_source_without_a_revision_is_refused(self) -> None:
        """main aliases one step branch, so an unstated revision serves a step nobody chose."""
        with pytest.raises(ValueError, match="explicit revision"):
            FullWeightsSource.parse("allenai/tmax-4b", None)

    def test_a_local_directory_with_a_revision_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="no revision to pin"):
            FullWeightsSource.parse(str(tmp_path), "step_300")

    def test_neither_a_directory_nor_a_repo_id_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="neither an existing directory nor a hub repo id"):
            FullWeightsSource.parse(str(tmp_path / "absent"), "main")
        with pytest.raises(ValueError, match="neither an existing directory nor a hub repo id"):
            FullWeightsSource.parse("tmax-4b", "main")

    def test_the_dataclass_refuses_two_shapes_or_none(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="hub repo OR a local directory"):
            FullWeightsSource(repo_id="a/b", revision="main", local_dir=tmp_path)
        with pytest.raises(ValueError, match="hub repo OR a local directory"):
            FullWeightsSource(repo_id=None, revision=None, local_dir=None)


class TestResolveFullWeightsFromTheHub:
    """The revision is pinned to a commit, fetched at that commit, and the bytes proven to be its."""

    def test_the_snapshot_is_fetched_at_the_resolved_commit_with_the_weights_filters(
        self, hub: dict[str, Any]
    ) -> None:
        facts = resolve_full_weights(FullWeightsSource.parse("allenai/tmax-4b", "step_300"))
        assert facts.label == "allenai/tmax-4b@step_300"
        assert facts.commit_sha == FAKE_COMMIT
        assert facts.snapshot_dir == hub["snapshot"]
        assert facts.weights_sha256 == (("model.safetensors", hub["digest"]),)
        assert facts.declares_vision_config is True
        assert facts.chat_template_sha256 is not None
        (call,) = hub["download"].calls
        assert call["revision"] == FAKE_COMMIT, "fetched at the sha, never at the branch name"
        assert call["allow_patterns"] == list(eval_model.FULL_WEIGHTS_ALLOW_PATTERNS)
        assert call["ignore_patterns"] == list(eval_model.FULL_WEIGHTS_IGNORE_PATTERNS)
        (api,) = StubHfApi.instances
        assert api.calls == [
            {"repo_id": "allenai/tmax-4b", "revision": "step_300", "files_metadata": True}
        ]

    def test_bytes_that_are_not_the_revisions_are_refused(self, hub: dict[str, Any]) -> None:
        """SABOTAGE: another branch's blob under this name has the same size and a different digest."""
        hub["info"].siblings = [StubSibling("model.safetensors", "f" * 64)]
        with pytest.raises(RuntimeError, match="not that revision's"):
            resolve_full_weights(FullWeightsSource.parse("allenai/tmax-4b", "step_300"))

    def test_a_tensor_file_the_hub_lists_but_the_snapshot_lacks_is_refused(
        self, hub: dict[str, Any]
    ) -> None:
        hub["info"].siblings.append(StubSibling("model-00002.safetensors", "a" * 64))
        with pytest.raises(FileNotFoundError, match="not fetched"):
            resolve_full_weights(FullWeightsSource.parse("allenai/tmax-4b", "step_300"))

    def test_a_tensor_file_without_an_lfs_digest_cannot_be_verified_and_is_refused(
        self, hub: dict[str, Any]
    ) -> None:
        hub["info"].siblings = [StubSibling("model.safetensors", None)]
        with pytest.raises(RuntimeError, match="no LFS digest"):
            resolve_full_weights(FullWeightsSource.parse("allenai/tmax-4b", "step_300"))

    def test_a_repo_with_no_tensor_files_is_refused(self, hub: dict[str, Any]) -> None:
        hub["info"].siblings = [StubSibling("config.json", None)]
        with pytest.raises(RuntimeError, match=r"carries no \.safetensors"):
            resolve_full_weights(FullWeightsSource.parse("allenai/tmax-4b", "step_300"))

    def test_a_snapshot_directory_not_named_for_the_commit_is_refused(
        self, hub: dict[str, Any], tmp_path: Path
    ) -> None:
        hub["download"].result = write_weights_dir(tmp_path / "elsewhere")
        with pytest.raises(RuntimeError, match="not a directory named for commit"):
            resolve_full_weights(FullWeightsSource.parse("allenai/tmax-4b", "step_300"))

    def test_the_fingerprint_is_content_not_path(self, hub: dict[str, Any], tmp_path: Path) -> None:
        """The same bytes under another cache root fingerprint identically; other bytes do not."""
        from_hub = resolve_full_weights(FullWeightsSource.parse("allenai/tmax-4b", "step_300"))
        same_bytes = resolve_full_weights(
            FullWeightsSource.parse(str(write_weights_dir(tmp_path / "copy")), None)
        )
        other_bytes = resolve_full_weights(
            FullWeightsSource.parse(
                str(write_weights_dir(tmp_path / "other", tensor_bytes=b"\x01" * 64)), None
            )
        )
        assert from_hub.fingerprint == same_bytes.fingerprint
        assert from_hub.fingerprint != other_bytes.fingerprint


class TestResolveFullWeightsLocally:
    def test_a_directory_is_hashed_and_labelled_by_its_name(self, tmp_path: Path) -> None:
        unit = write_weights_dir(tmp_path / "tmax-4b-step380-alpha1.5", vision_config=False)
        facts = resolve_full_weights(FullWeightsSource.parse(str(unit), None))
        assert facts.label == "tmax-4b-step380-alpha1.5"
        assert facts.commit_sha is None
        assert facts.snapshot_dir == unit
        assert facts.declares_vision_config is False
        assert dict(facts.weights_sha256) == {
            "model.safetensors": eval_model.sha256_of_file(unit / "model.safetensors")
        }

    def test_a_provenance_sidecar_that_matches_is_honoured(self, tmp_path: Path) -> None:
        unit = write_weights_dir(tmp_path / "unit")
        digest = eval_model.sha256_of_file(unit / "model.safetensors")
        (unit / WEIGHTS_PROVENANCE_FILENAME).write_text(
            json.dumps({"weights_sha256": {"model.safetensors": digest}}), encoding="utf-8"
        )
        assert resolve_full_weights(FullWeightsSource.parse(str(unit), None)).label == "unit"

    def test_tensors_edited_after_the_sidecar_was_written_are_refused(self, tmp_path: Path) -> None:
        """SABOTAGE: the label on the directory no longer says what is inside it."""
        unit = write_weights_dir(tmp_path / "unit")
        digest = eval_model.sha256_of_file(unit / "model.safetensors")
        (unit / WEIGHTS_PROVENANCE_FILENAME).write_text(
            json.dumps({"weights_sha256": {"model.safetensors": digest}}), encoding="utf-8"
        )
        (unit / "model.safetensors").write_bytes(b"\x01" * 64)
        with pytest.raises(RuntimeError, match="changed after it was labelled"):
            resolve_full_weights(FullWeightsSource.parse(str(unit), None))

    def test_a_sidecar_without_digests_is_refused(self, tmp_path: Path) -> None:
        unit = write_weights_dir(tmp_path / "unit")
        (unit / WEIGHTS_PROVENANCE_FILENAME).write_text(
            json.dumps({"alpha": 1.5}), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="no weights_sha256"):
            resolve_full_weights(FullWeightsSource.parse(str(unit), None))

    def test_a_directory_without_tensors_or_config_is_not_a_checkpoint(
        self, tmp_path: Path
    ) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match=r"no \.safetensors"):
            resolve_full_weights(FullWeightsSource.parse(str(empty), None))
        (empty / "model.safetensors").write_bytes(FAKE_TENSOR_BYTES)
        with pytest.raises(FileNotFoundError, match=r"no config\.json"):
            resolve_full_weights(FullWeightsSource.parse(str(empty), None))

    def test_the_template_digest_falls_back_to_tokenizer_config(self, tmp_path: Path) -> None:
        unit = write_weights_dir(tmp_path / "unit", template=None)
        assert (
            resolve_full_weights(FullWeightsSource.parse(str(unit), None)).chat_template_sha256
            is None
        )
        (unit / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": "{{ messages }}"}), encoding="utf-8"
        )
        assert resolve_full_weights(FullWeightsSource.parse(str(unit), None)).chat_template_sha256


class TestResolveServedModelWithFullWeights:
    def local_source(self, tmp_path: Path, **kwargs: Any) -> FullWeightsSource:
        return FullWeightsSource.parse(str(write_weights_dir(tmp_path / "unit", **kwargs)), None)

    def test_vllm_is_pointed_at_the_snapshot_and_told_to_skip_the_vision_tower(
        self, tmp_path: Path
    ) -> None:
        served = resolve_served_model(
            checkpoint=None,
            base_model=BASE_MODEL,
            backend_kind="vllm",
            merge_root=tmp_path / "merges",
            merge_label="x-",
            full_weights=self.local_source(tmp_path),
        )
        assert served.load_mode == LOAD_MODE_FULL_WEIGHTS
        assert served.model_id == "unit"
        assert served.adapter_dir is None
        assert served.weights is not None
        assert served.backend_kwargs == {
            "model_path": str(tmp_path / "unit"),
            "language_model_only": True,
        }
        assert served.delta_faithful is True

    def test_a_checkpoint_without_a_vision_tower_needs_no_engine_flag(self, tmp_path: Path) -> None:
        served = resolve_served_model(
            checkpoint=None,
            base_model=BASE_MODEL,
            backend_kind="vllm",
            merge_root=tmp_path / "merges",
            merge_label="x-",
            full_weights=self.local_source(tmp_path, vision_config=False),
        )
        assert served.backend_kwargs == {"model_path": str(tmp_path / "unit")}

    def test_the_transformers_path_gets_only_the_directory(self, tmp_path: Path) -> None:
        """AutoModelForCausalLM drops the tower on its own for this family."""
        served = resolve_served_model(
            checkpoint=None,
            base_model=BASE_MODEL,
            backend_kind="hf",
            merge_root=tmp_path / "merges",
            merge_label="x-",
            full_weights=self.local_source(tmp_path),
        )
        assert served.backend_kwargs == {"model_path": str(tmp_path / "unit")}

    def test_the_mock_backend_neither_fetches_nor_hashes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(eval_model, "HfApi", lambda: StubHfApi(None))
        with caplog.at_level("WARNING"):
            served = resolve_served_model(
                checkpoint=None,
                base_model=BASE_MODEL,
                backend_kind=MOCK_BACKEND_KIND,
                merge_root=tmp_path / "merges",
                merge_label="x-",
                full_weights=FullWeightsSource.parse("allenai/tmax-4b", "step_300"),
            )
        assert served.load_mode == LOAD_MODE_MOCK_NO_LOAD
        assert served.model_id == "allenai/tmax-4b@step_300"
        assert served.weights is None
        assert "no model runs" in caplog.text

    def test_an_adapter_on_top_of_full_weights_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="never both"):
            resolve_served_model(
                checkpoint=make_adapter_dir(tmp_path / "checkpoint-70"),
                base_model=BASE_MODEL,
                backend_kind="vllm",
                merge_root=tmp_path / "merges",
                merge_label="x-",
                full_weights=self.local_source(tmp_path),
            )

    def test_an_unchecked_backend_is_refused_before_anything_is_fetched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(eval_model, "HfApi", lambda: StubHfApi(None))
        with pytest.raises(ValueError, match="no serving capability recorded"):
            resolve_served_model(
                checkpoint=None,
                base_model=BASE_MODEL,
                backend_kind="bedrock",
                merge_root=tmp_path / "merges",
                merge_label="x-",
                full_weights=FullWeightsSource.parse("allenai/tmax-4b", "step_300"),
            )

    def test_provenance_names_the_weights_their_commit_and_their_fingerprint(
        self, hub: dict[str, Any], tmp_path: Path
    ) -> None:
        served = resolve_served_model(
            checkpoint=None,
            base_model=BASE_MODEL,
            backend_kind="vllm",
            merge_root=tmp_path / "merges",
            merge_label="x-",
            full_weights=FullWeightsSource.parse("allenai/tmax-4b", "step_300"),
        )
        assert served.weights is not None
        assert served.provenance == {
            "model_load_mode": LOAD_MODE_FULL_WEIGHTS,
            "model_delta_faithful": True,
            "model_served_id": "allenai/tmax-4b@step_300",
            "model_adapter_dir": None,
            "model_full_weights": "allenai/tmax-4b@step_300",
            "model_weights_commit_sha": FAKE_COMMIT,
            "model_weights_fingerprint": served.weights.fingerprint,
            "model_weights_chat_template_sha256": served.weights.chat_template_sha256,
        }


class StubWeightsVerifyingBackend:
    """A backend whose engine reports one directory as the one it loaded."""

    def __init__(self, loaded_from: Path) -> None:
        self.loaded_from = loaded_from
        self.checked: list[Path] = []

    def assert_serves_weights(self, snapshot_dir: Path) -> None:
        self.checked.append(snapshot_dir)
        if snapshot_dir.resolve() != self.loaded_from.resolve():
            raise RuntimeError(f"loaded {self.loaded_from}, asked about {snapshot_dir}")


def full_weights_served(unit: Path) -> ServedModel:
    facts = resolve_full_weights(FullWeightsSource.parse(str(unit), None))
    return ServedModel(
        model_id=facts.label, load_mode=LOAD_MODE_FULL_WEIGHTS, adapter_dir=None, weights=facts
    )


class TestVerifyFullWeights:
    def test_the_engine_is_asked_about_the_verified_directory(self, tmp_path: Path) -> None:
        unit = write_weights_dir(tmp_path / "unit")
        backend = StubWeightsVerifyingBackend(unit)
        verify_served_model(backend, full_weights_served(unit))
        assert backend.checked == [unit]

    def test_an_engine_that_loaded_something_else_fails_the_check(self, tmp_path: Path) -> None:
        """SABOTAGE: the directory whose bytes were verified is not the one the engine serves."""
        unit = write_weights_dir(tmp_path / "unit")
        other = write_weights_dir(tmp_path / "other")
        with pytest.raises(RuntimeError, match="asked about"):
            verify_served_model(StubWeightsVerifyingBackend(other), full_weights_served(unit))

    def test_a_backend_that_cannot_say_what_it_loaded_is_refused(self, tmp_path: Path) -> None:
        unit = write_weights_dir(tmp_path / "unit")
        with pytest.raises(TypeError, match="no assert_serves_weights"):
            verify_served_model(StubUnverifiableBackend(), full_weights_served(unit))

    def test_a_full_weights_decision_without_facts_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no resolved weights"):
            verify_served_model(
                StubUnverifiableBackend(),
                ServedModel(model_id="x", load_mode=LOAD_MODE_FULL_WEIGHTS, adapter_dir=None),
            )

    def test_the_adapter_guard_is_not_run_for_full_weights(self, tmp_path: Path) -> None:
        """No adapter, nothing for assert_adapter_changes_output to compare."""
        unit = write_weights_dir(tmp_path / "unit")

        class Both(StubWeightsVerifyingBackend, StubVerifyingBackend):
            def __init__(self, loaded_from: Path) -> None:
                StubWeightsVerifyingBackend.__init__(self, loaded_from)
                StubVerifyingBackend.__init__(self)

        backend = Both(unit)
        verify_served_model(backend, full_weights_served(unit))
        assert backend.verified_with == []
        assert backend.checked == [unit]
