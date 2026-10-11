"""CPU coverage for Atlas token-position activation capture and readout."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from reward_hacking.atlas import token_sweep
from reward_hacking.interp.jsonl_resume import ResumeMismatchError

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture
def small_tensor_threads() -> Iterator[None]:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous_threads)


class FakeTokenizer:
    """A character tokenizer whose decoding boundaries are exact and deterministic."""

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)


class FakeModel(torch.nn.Module):
    """Expose token ids plus a layer-specific offset as hidden states."""

    def __init__(self, n_layers: int = 5, d_model: int = 2) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        # Multimodal configs (Gemma3Config) keep layer counts on the text config only.
        text_config = SimpleNamespace(num_hidden_layers=n_layers, hidden_size=d_model)
        self.config = SimpleNamespace(get_text_config=lambda: text_config)
        self.n_layers = n_layers
        self.d_model = d_model
        self.forward_calls = 0
        self.input_sequences: list[list[int]] = []

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool,
        use_cache: bool,
    ) -> SimpleNamespace:
        assert attention_mask.shape == input_ids.shape
        assert output_hidden_states is True
        assert use_cache is False
        self.forward_calls += 1
        self.input_sequences.append(input_ids[0].tolist())
        token_values = input_ids.to(dtype=torch.float32).unsqueeze(-1).expand(-1, -1, self.d_model)
        hidden_states = tuple(token_values + layer * 10 for layer in range(self.n_layers + 1))
        return SimpleNamespace(hidden_states=hidden_states)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _capture_inputs(run_dir: Path) -> Path:
    completion_text = "x=1\n"
    completion_token_ids = [ord(character) for character in completion_text]
    samples = [
        {
            "sample_id": "base-item-a-0",
            "item_id": "item-a",
            "sample_index": 0,
            "model_id": token_sweep.BASE_MODEL_ID,
            "prompt_token_ids": [ord("a"), ord("b"), ord("c"), ord("d")],
            "completion_token_ids": completion_token_ids,
            "completion_text": completion_text,
        },
        {
            "sample_id": "base-item-b-0",
            "item_id": "item-b",
            "sample_index": 0,
            "model_id": token_sweep.BASE_MODEL_ID,
            "prompt_token_ids": [ord("e"), ord("f")],
            "completion_token_ids": completion_token_ids,
            "completion_text": completion_text,
        },
    ]
    grades = [
        {
            "sample_id": "base-item-a-0",
            "item_id": "item-a",
            "hack": True,
            "passes_visible": True,
            "passes_all": False,
        },
        {
            "sample_id": "base-item-b-0",
            "item_id": "item-b",
            "hack": False,
            "passes_visible": True,
            "passes_all": True,
        },
    ]
    sample_path = run_dir / "samples.base.jsonl"
    _write_jsonl(sample_path, samples)
    _write_jsonl(run_dir / "grades.base.jsonl", grades)
    return sample_path


def test_capture_writes_layer_positions_mask_offsets_and_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = FakeModel()
    monkeypatch.setattr(
        "reward_hacking.interp.directions.load_model_and_tokenizer",
        lambda model_id, dtype: (model, FakeTokenizer()),
    )
    _capture_inputs(tmp_path)

    output_path = token_sweep.capture_model(
        run_dir=tmp_path,
        model_key="base",
        prompt_tokens=3,
        completion_tokens=5,
        layer_stride=2,
    )

    activations = np.load(output_path)
    mask = np.load(output_path.with_name("mask.base.npy"))
    index = json.loads(output_path.with_name("index.base.json").read_text(encoding="utf-8"))
    assert activations.shape == (2, 3, 8, 2)
    assert activations.dtype == np.float32
    assert mask.dtype == np.bool_
    assert mask.tolist() == [
        [True, True, True, True, True, True, True, False],
        [False, True, True, True, True, True, True, False],
    ]
    assert index["layer_ids"] == [0, 2, 4]
    assert index["rows"][0]["hack_label"] is True
    assert index["rows"][0]["pass_label"] is False
    assert index["rows"][0]["code_token_start"] == 0
    assert index["rows"][0]["completion_char_offsets"] == [[0, 1], [1, 2], [2, 3], [3, 4]]
    assert index["rows"][0]["completion_token_ids"] == [ord(character) for character in "x=1\n"]
    assert activations[0, 0, 0, 0] == ord("b") + 10
    assert activations[0, 1, 3, 0] == ord("x") + 30
    assert activations[0, 2, 7, 0] == 0
    assert model.forward_calls == 2
    assert model.input_sequences[0] == [ord(character) for character in "abcdx=1\n"]
    assert activations[1, :, 0].sum() == 0


def test_capture_resumes_by_sample_id_and_refuses_identity_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_model = FakeModel()
    monkeypatch.setattr(
        "reward_hacking.interp.directions.load_model_and_tokenizer",
        lambda model_id, dtype: (first_model, FakeTokenizer()),
    )
    _capture_inputs(tmp_path)

    def capture(prompt_tokens: int = 3) -> Path:
        return token_sweep.capture_model(
            run_dir=tmp_path,
            model_key="base",
            prompt_tokens=prompt_tokens,
            completion_tokens=5,
            layer_stride=2,
        )

    capture()
    assert first_model.forward_calls == 2

    resumed_model = FakeModel()
    monkeypatch.setattr(
        "reward_hacking.interp.directions.load_model_and_tokenizer",
        lambda model_id, dtype: (resumed_model, FakeTokenizer()),
    )
    capture()
    assert resumed_model.forward_calls == 0
    with pytest.raises(ResumeMismatchError, match="identity fields differ"):
        capture(prompt_tokens=2)
    assert resumed_model.forward_calls == 0


def test_capture_resume_compares_selected_rows_instead_of_jsonl_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_model = FakeModel()
    monkeypatch.setattr(
        "reward_hacking.interp.directions.load_model_and_tokenizer",
        lambda model_id, dtype: (first_model, FakeTokenizer()),
    )
    _capture_inputs(tmp_path)
    output_path = token_sweep.capture_model(
        run_dir=tmp_path,
        model_key="base",
        prompt_tokens=3,
        completion_tokens=5,
        layer_stride=2,
    )

    index_path = output_path.with_name("index.base.json")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    legacy_identity = {
        key: value
        for key, value in index["identity"].items()
        if key not in {"selected_rows", "sample_content", "model_weights", "adapter_weights"}
    }
    legacy_identity["sample_file_sha256"] = "old-sample-jsonl-digest"
    legacy_identity["grade_file_sha256"] = "old-grade-jsonl-digest"
    index["identity"] = legacy_identity
    index_path.write_text(json.dumps(index), encoding="utf-8")
    progress_ledger_path = output_path.with_name("progress.base.jsonl.resume.json")
    old_ledger = json.loads(progress_ledger_path.read_text(encoding="utf-8"))
    old_ledger["identity"] = legacy_identity
    progress_ledger_path.write_text(json.dumps(old_ledger), encoding="utf-8")

    for filename in ("samples.base.jsonl", "grades.base.jsonl"):
        path = tmp_path / filename
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        path.write_text(
            "\n".join(json.dumps(record, sort_keys=False) for record in records) + "\n",
            encoding="utf-8",
        )

    resumed_model = FakeModel()
    monkeypatch.setattr(
        "reward_hacking.interp.directions.load_model_and_tokenizer",
        lambda model_id, dtype: (resumed_model, FakeTokenizer()),
    )
    token_sweep.capture_model(
        run_dir=tmp_path,
        model_key="base",
        prompt_tokens=3,
        completion_tokens=5,
        layer_stride=2,
    )
    assert resumed_model.forward_calls == 0

    changed_samples = [
        json.loads(line) for line in (tmp_path / "samples.base.jsonl").read_text().splitlines()
    ]
    changed_samples[0]["completion_text"] = "different selected completion"
    _write_jsonl(tmp_path / "samples.base.jsonl", changed_samples)
    with pytest.raises(ResumeMismatchError, match="identity fields differ"):
        token_sweep.capture_model(
            run_dir=tmp_path,
            model_key="base",
            prompt_tokens=3,
            completion_tokens=5,
            layer_stride=2,
        )
    assert output_path.is_file()


def test_capture_accepts_a_relocated_local_model_with_the_same_tensor_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_model = tmp_path / "original" / "base"
    relocated_model = tmp_path / "relocated" / "base"
    original_model.mkdir(parents=True)
    relocated_model.mkdir(parents=True)
    save_file({"weight": torch.tensor([1.0, 2.0])}, str(original_model / "model.safetensors"))
    save_file({"weight": torch.tensor([1.0, 2.0])}, str(relocated_model / "renamed.safetensors"))
    sample_path = _capture_inputs(tmp_path)
    samples = [json.loads(line) for line in sample_path.read_text(encoding="utf-8").splitlines()]
    for sample in samples:
        sample["model_id"] = str(original_model)
    _write_jsonl(sample_path, samples)

    original_model_instance = FakeModel()
    monkeypatch.setattr(
        "reward_hacking.interp.directions.load_model_and_tokenizer",
        lambda model_id, dtype: (original_model_instance, FakeTokenizer()),
    )
    output_path = token_sweep.capture_model(
        run_dir=tmp_path,
        model_key="base",
        base_model=str(original_model),
        prompt_tokens=3,
        completion_tokens=5,
        layer_stride=2,
    )
    assert original_model_instance.forward_calls == 2

    relocated_samples = [{**sample, "model_id": str(relocated_model)} for sample in samples]
    _write_jsonl(sample_path, relocated_samples)
    relocated_model_instance = FakeModel()
    monkeypatch.setattr(
        "reward_hacking.interp.directions.load_model_and_tokenizer",
        lambda model_id, dtype: (relocated_model_instance, FakeTokenizer()),
    )
    resumed_path = token_sweep.capture_model(
        run_dir=tmp_path,
        model_key="base",
        base_model=str(relocated_model),
        prompt_tokens=3,
        completion_tokens=5,
        layer_stride=2,
    )

    assert resumed_path == output_path
    assert relocated_model_instance.forward_calls == 0


def test_capture_can_read_adapter_samples_with_base_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter_key = "kl0.01-det3-seed1-mbpp_probe"
    _capture_inputs(tmp_path)
    base_samples = [
        {
            **sample,
            "sample_id": str(sample["sample_id"]).replace("base-", f"{adapter_key}-"),
        }
        for sample in (
            json.loads(line) for line in (tmp_path / "samples.base.jsonl").read_text().splitlines()
        )
    ]
    base_grades = [
        {**grade, "sample_id": str(grade["sample_id"]).replace("base-", f"{adapter_key}-")}
        for grade in (
            json.loads(line) for line in (tmp_path / "grades.base.jsonl").read_text().splitlines()
        )
    ]
    base_samples[0]["prompt_token_ids"] = [ord(character) for character in "wxyz"]
    base_samples[0]["completion_text"] = "q=2\n"
    base_samples[0]["completion_token_ids"] = [ord(character) for character in "q=2\n"]
    base_grades[0]["hack"] = False
    base_grades[0]["passes_all"] = True
    _write_jsonl(tmp_path / f"samples.{adapter_key}.jsonl", base_samples)
    _write_jsonl(tmp_path / f"grades.{adapter_key}.jsonl", base_grades)
    model = FakeModel()
    loaded_model_ids: list[str] = []

    def load_model(model_id: str, *, dtype: torch.dtype) -> tuple[FakeModel, FakeTokenizer]:
        del dtype
        loaded_model_ids.append(model_id)
        return model, FakeTokenizer()

    monkeypatch.setattr("reward_hacking.interp.directions.load_model_and_tokenizer", load_model)
    output_path = token_sweep.capture_model(
        run_dir=tmp_path,
        model_key="base",
        samples_from=adapter_key,
        prompt_tokens=3,
        completion_tokens=5,
        layer_stride=2,
    )

    output_key = f"base--reads--{adapter_key}"
    assert output_path.name == f"activations.{output_key}.npy"
    assert loaded_model_ids == [token_sweep.BASE_MODEL_ID]
    assert model.input_sequences[0] == [ord(character) for character in "wxyzq=2\n"]
    index = json.loads(
        output_path.with_name(f"index.{output_key}.json").read_text(encoding="utf-8")
    )
    assert index["model_key"] == "base"
    assert index["samples_from"] == adapter_key
    assert index["identity"]["model_key"] == "base"
    assert index["identity"]["samples_from"] == adapter_key
    assert index["rows"][0]["prompt_token_ids"] == [ord(character) for character in "wxyz"]
    assert index["rows"][0]["hack_label"] is False
    assert index["rows"][0]["pass_label"] is True


def test_capture_cli_forwards_samples_from(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(token_sweep, "capture_models", lambda **kwargs: calls.append(kwargs))
    adapter_key = "kl0.01-det3-seed1-mbpp_probe"

    token_sweep.main(
        [
            "capture",
            "--run-dir",
            str(tmp_path),
            "--models",
            "base",
            "--samples-from",
            adapter_key,
        ]
    )

    assert calls[0]["model_keys"] == ["base"]
    assert calls[0]["samples_from"] == adapter_key


def _write_synthetic_sweep(
    directory: Path,
    model_key: str,
    activations: np.ndarray,
    rows: list[dict[str, Any]],
    mask: np.ndarray,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / f"activations.{model_key}.npy", activations.astype(np.float32))
    np.save(directory / f"mask.{model_key}.npy", mask.astype(np.bool_))
    (directory / f"index.{model_key}.json").write_text(
        json.dumps(
            {
                "model_key": model_key,
                "sample_ids": [row["sample_id"] for row in rows],
                "layer_ids": [7],
                "prompt_tokens": 1,
                "completion_tokens": 3,
                "shape": list(activations.shape),
                "rows": rows,
                "complete": True,
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.slow
@pytest.mark.usefixtures("small_tensor_threads")
def test_analyse_finds_separable_positions_and_scores_random_positions(
    tmp_path: Path,
) -> None:
    generator = np.random.default_rng(1234)
    n_items = 20
    samples_per_item = 2
    n_samples = n_items * samples_per_item
    labels_by_item = np.array([0] * 10 + [1] * 10, dtype=np.int64)
    labels = np.repeat(labels_by_item, samples_per_item)
    groups = np.repeat(np.arange(n_items), samples_per_item)
    folds = token_sweep._folds(labels, groups)
    assert folds is not None
    for train_indices, test_indices in folds:
        assert set(groups[train_indices]).isdisjoint(set(groups[test_indices]))
    rows_adapter = [
        {
            "sample_id": f"adapter-{item}-{sample_index}",
            "item_id": f"item-{item}",
            "sample_index": sample_index,
            "hack_label": bool(labels[row_index]),
            "pass_label": not bool(labels[row_index]),
            "code_token_start": 0,
        }
        for row_index, (item, sample_index) in enumerate(
            (item, sample_index)
            for item in range(n_items)
            for sample_index in range(samples_per_item)
        )
    ]
    rows_base = [
        {
            **row,
            "sample_id": str(row["sample_id"]).replace("adapter-", "base-"),
        }
        for row in rows_adapter
    ]
    adapter_activations = generator.normal(size=(n_samples, 1, 4, 6)).astype(np.float32)
    adapter_activations[:, 0, 1, 0] = labels * 8 + generator.normal(0, 0.01, size=n_samples)
    base_activations = generator.normal(size=(n_samples, 1, 4, 6)).astype(np.float32)
    mask = np.zeros((n_samples, 4), dtype=np.bool_)
    mask[:, 1:3] = True
    _write_synthetic_sweep(tmp_path / "token-sweep", "base", base_activations, rows_base, mask)
    adapter_key = "kl0.01-det3-seed1-mbpp_probe"
    _write_synthetic_sweep(
        tmp_path / "token-sweep", adapter_key, adapter_activations, rows_adapter, mask
    )

    json_path, markdown_path = token_sweep.analyse_sweeps(
        run_dir=tmp_path,
        adapter_key=adapter_key,
        n_permutations=20,
        seed=0,
    )

    readout = json.loads(json_path.read_text(encoding="utf-8"))
    assert {entry["model_key"] for entry in readout["results"]} == {
        "base_own_completions",
        adapter_key,
    }
    adapter_start = next(
        entry
        for entry in readout["results"]
        if entry["model_key"] == adapter_key
        and entry["anchor"] == "completion_start"
        and entry["offset"] == 0
        and entry["layer_id"] == 7
    )
    adapter_random = next(
        entry
        for entry in readout["results"]
        if entry["model_key"] == adapter_key
        and entry["anchor"] == "completion_start"
        and entry["offset"] == 1
        and entry["layer_id"] == 7
    )
    assert adapter_start["auc"] == pytest.approx(1.0, abs=0.02)
    assert 0.3 <= adapter_random["auc"] <= 0.7
    assert adapter_start["null_auc_max"] < adapter_start["auc"]
    assert adapter_start["n_permutations"] == 20
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "completion_start" in markdown
    assert "base_own_completions" in markdown
    assert "compare different texts" in markdown
    assert "| Arm | Anchor | Best layer / offset | Best AUC | Null max |" in markdown


def test_analyse_adds_teacher_forced_arm_and_rejects_token_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter_key = "kl0.01-det3-seed1-mbpp_probe"
    output_dir = tmp_path / "token-sweep"
    rows = [
        {
            "sample_id": f"adapter-{item}-0",
            "item_id": f"item-{item}",
            "sample_index": 0,
            "hack_label": bool(item % 2),
            "pass_label": not bool(item % 2),
            "code_token_start": 0,
            "prompt_token_ids": [10 + item],
            "completion_token_ids": [100 + item, 200 + item],
        }
        for item in range(4)
    ]
    activations = np.zeros((4, 1, 4, 2), dtype=np.float32)
    mask = np.ones((4, 4), dtype=np.bool_)
    base_rows = [{**row, "sample_id": f"base-{row['sample_id']}"} for row in rows]
    _write_synthetic_sweep(output_dir, "base", activations, base_rows, mask)
    _write_synthetic_sweep(output_dir, adapter_key, activations, rows, mask)
    teacher_key = token_sweep._token_sweep_key("base", adapter_key)
    _write_synthetic_sweep(output_dir, teacher_key, activations[::-1], rows[::-1], mask)

    analysed_arms: list[str] = []

    def record_arm(**kwargs: Any) -> list[dict[str, object]]:
        analysed_arms.append(str(kwargs["model_key"]))
        return []

    monkeypatch.setattr(token_sweep, "_analyze_model", record_arm)
    token_sweep.analyse_sweeps(run_dir=tmp_path, adapter_key=adapter_key, n_permutations=1)
    assert analysed_arms == ["base_own_completions", adapter_key, "base_reads_adapter_tokens"]

    teacher_index_path = output_dir / f"index.{teacher_key}.json"
    teacher_index = json.loads(teacher_index_path.read_text(encoding="utf-8"))
    teacher_index["rows"][0]["completion_token_ids"] = [999]
    teacher_index_path.write_text(json.dumps(teacher_index), encoding="utf-8")
    with pytest.raises(ValueError, match="completion token ids differ"):
        token_sweep.analyse_sweeps(run_dir=tmp_path, adapter_key=adapter_key, n_permutations=1)


class NonMonotonicTokenizer(FakeTokenizer):
    """Decodes a partial two-token character as a replacement mark, like a split multi-byte UTF-8 sequence."""

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        text = super().decode(token_ids, skip_special_tokens=skip_special_tokens)
        return text[:-1] + "\ufffd" if token_ids and token_ids[-1] == ord("x") else text


def test_metadata_row_records_no_code_start_when_the_code_span_cannot_be_mapped() -> None:
    completion = "a x then y"
    sample = {
        "sample_id": "s0",
        "item_id": "item",
        "sample_index": 0,
        "prompt_token_ids": [1, 2],
        "completion_token_ids": [ord(character) for character in completion],
        "completion_text": completion,
    }
    grade = {"hack": True, "passes_all": False}

    row = token_sweep._metadata_row(sample, grade, cast("Any", NonMonotonicTokenizer()))  # pyright: ignore[reportPrivateUsage]

    assert row["code_token_start"] is None
    assert row["hack_label"] is True
