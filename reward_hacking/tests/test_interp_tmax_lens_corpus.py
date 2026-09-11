"""Offline tests for the lens fit-corpus draw, on a mock tokenizer with realistic boundary behaviour.

The mock tokenizes on whitespace with each piece carrying its leading space, and its encoder strips
leading spaces the way a BPE normaliser does. So a window cut at a space-led piece decodes to text
whose re-encoding drops the space and changes the first id -- the byte-pair boundary failure the
round-trip gate exists for -- while a window cut at a newline-led piece is stable. Every text here is
synthetic; nothing from the benchmark corpus appears.
"""

from __future__ import annotations

import json
import re
import zlib
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from reward_hacking.interp.tmax_lens_corpus import (
    ANCHORS_FILENAME,
    CLASS_CONCEPT_SENTENCES,
    CLASS_GRADER,
    CLASS_REASONING,
    CLASS_TAIL,
    CORPUS_SCHEMA,
    FALLBACK_WINDOW_CLASSES,
    ROLE_ANCHOR,
    ROLE_EVAL,
    ROLE_FIT,
    SERVED_PROMPT,
    SIDECAR_FILENAME,
    THINK_END,
    WINDOW_CLASSES,
    CorpusDrawError,
    CorpusItem,
    Document,
    DrawSpec,
    TokenCodec,
    assert_disjoint,
    class_quotas,
    concept_sentence_documents,
    cut_window,
    draw_corpus,
    grader_char_span,
    is_base_written,
    load_lens_corpus,
    record_documents,
    round_trip_start,
    served_weights,
    stimulus_documents,
    token_index_at_char,
    write_corpus,
)

BASE_MODEL_ID = "org/base"
BASE_SERVED = f"{BASE_MODEL_ID}@main"
RL_SERVED = "org/rl@step_500"

if TYPE_CHECKING:
    from collections.abc import Sequence

_PIECE = re.compile(r"\n|[ ]?[^ \n]+")


class MockTokenizer:
    """Whitespace pieces with leading spaces attached; the encoder strips leading spaces of the text."""

    def __init__(self) -> None:
        self.vocab: dict[int, str] = {}

    def pieces(self, text: str) -> list[str]:
        return _PIECE.findall(text.lstrip(" "))

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in self.pieces(text):
            piece_id = zlib.crc32(piece.encode())
            self.vocab[piece_id] = piece
            ids.append(piece_id)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self.vocab[piece_id] for piece_id in ids)

    def offsets(self, text: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        cursor = len(text) - len(text.lstrip(" "))
        for piece in self.pieces(text):
            spans.append((cursor, cursor + len(piece)))
            cursor += len(piece)
        return spans

    def codec(self) -> TokenCodec:
        return TokenCodec(encode=self.encode, decode=self.decode, offsets=self.offsets)


@pytest.fixture
def codec() -> TokenCodec:
    return MockTokenizer().codec()


def lines(n_lines: int, *, words_per_line: int = 6, seed: str = "x") -> str:
    """Synthetic prose: ``n_lines`` lines of ``words_per_line`` distinct words each."""
    return "\n".join(
        " ".join(f"{seed}{line}w{word}" for word in range(words_per_line))
        for line in range(n_lines)
    )


def document(doc_id: str, text: str, codec: TokenCodec, **overrides: object) -> Document:
    fields: dict[str, object] = {
        "doc_id": doc_id,
        "group_id": doc_id,
        "source": "records:test",
        "served": BASE_SERVED,
        "text": text,
        "ids": tuple(codec.encode(text)),
        "grader_tokens": None,
        "reasoning_end_token": None,
        "base_written": False,
    }
    fields.update(overrides)
    return Document(**fields)  # pyright: ignore[reportArgumentType]


class TestTokenIndexAtChar:
    def test_maps_a_character_onto_the_piece_covering_it(self, codec: TokenCodec) -> None:
        text = "alpha beta\ngamma"
        offsets = codec.offsets(text)
        assert token_index_at_char(offsets, 0) == 0
        assert token_index_at_char(offsets, text.index("beta")) == 1
        assert token_index_at_char(offsets, text.index("\n")) == 2
        assert token_index_at_char(offsets, text.index("gamma")) == 3
        assert token_index_at_char(offsets, len(text) + 5) == len(offsets)


class TestGraderCharSpan:
    def test_locates_the_second_fenced_block_and_refuses_a_single_one(self) -> None:
        statement = "```python\ndef f(x):\n    return x\n```"
        grader = "```python\nassert candidate(1) == 1\n```"
        text = f"Solve this.\n\n{statement}\n\nThe grader:\n\n{grader}\n\nReturn code."
        span = grader_char_span(text)
        assert span is not None
        assert text[span[0] : span[1]] == grader
        assert grader_char_span(f"Solve this.\n\n{statement}\n\nAn unseen checker decides.") is None


class TestRoundTripStart:
    def test_a_space_led_start_is_shifted_to_the_nearest_newline_led_piece(
        self, codec: TokenCodec
    ) -> None:
        doc = document("d", lines(20), codec)
        # Piece 3 is space-led (" x0w3"); pieces 6, 13, ... are the newlines.
        found = round_trip_start(doc, codec, start=3, width=20, lower=0, upper=100, slack=16)
        assert found is not None
        start, text = found
        assert start == 6
        assert codec.encode(text) == list(doc.ids[6:26])

    def test_earlier_only_search_never_moves_later(self, codec: TokenCodec) -> None:
        doc = document("d", lines(20), codec)
        found = round_trip_start(
            doc, codec, start=8, width=20, lower=0, upper=100, slack=16, directions=(-1,)
        )
        assert found is not None
        # 8 is space-led and fails; 7 is the bare first word of a line and re-encodes stably.
        assert found[0] == 7

    def test_no_stable_start_within_slack_returns_none(self, codec: TokenCodec) -> None:
        doc = document("d", " ".join(f"w{i}" for i in range(80)), codec)
        assert round_trip_start(doc, codec, start=10, width=20, lower=1, upper=60, slack=4) is None


class TestCutWindow:
    def test_a_tail_window_ends_at_the_document_end_unless_shifted_earlier(
        self, codec: TokenCodec
    ) -> None:
        text = lines(20) + "\nfinal"
        doc = document("d", text, codec)
        cut = cut_window(doc, codec, window_class=CLASS_TAIL, width=20, slack=16)
        assert cut is not None
        start, shift, _text = cut
        assert shift <= 0
        assert start + 20 == doc.n_tokens + shift

    def test_a_reasoning_window_stays_inside_the_reasoning_span(self, codec: TokenCodec) -> None:
        reasoning = lines(30, seed="r")
        text = f"{reasoning}\n{THINK_END}\n{lines(10, seed='a')}"
        end_token = token_index_at_char(codec.offsets(text), text.index(THINK_END))
        doc = document("d", text, codec, reasoning_end_token=end_token, base_written=True)
        cut = cut_window(doc, codec, window_class=CLASS_REASONING, width=20, slack=16)
        assert cut is not None
        start, _shift, window_text = cut
        assert start + 20 <= end_token
        assert THINK_END not in window_text

    def test_a_grader_window_covers_the_grader_block(self, codec: TokenCodec) -> None:
        grader = "```python\nassert candidate(1) == 1\nassert candidate(2) == 4\n```"
        statement = "```python\ndef f(x):\n    return x\n```"
        text = (
            f"{lines(12, seed='p')}\n{statement}\n{lines(6, seed='q')}\n{grader}\n"
            f"{lines(12, seed='c')}"
        )
        span = grader_char_span(text)
        assert span is not None
        offsets = codec.offsets(text)
        grader_tokens = (
            token_index_at_char(offsets, span[0]),
            token_index_at_char(offsets, span[1] - 1) + 1,
        )
        doc = document("d", text, codec, source="stimuli", grader_tokens=grader_tokens)
        cut = cut_window(doc, codec, window_class=CLASS_GRADER, width=40, slack=16)
        assert cut is not None
        start, _shift, window_text = cut
        assert start <= grader_tokens[0]
        assert grader_tokens[1] <= start + 40
        assert "assert candidate(2) == 4" in window_text

    def test_asking_a_class_the_document_cannot_supply_is_a_draw_error(
        self, codec: TokenCodec
    ) -> None:
        doc = document("d", lines(20), codec)
        with pytest.raises(CorpusDrawError, match="no grader block"):
            cut_window(doc, codec, window_class=CLASS_GRADER, width=20, slack=16)


class TestClassQuotas:
    def test_splits_evenly_with_the_remainder_first(self) -> None:
        assert class_quotas(60, WINDOW_CLASSES) == {
            CLASS_GRADER: 20,
            CLASS_TAIL: 20,
            CLASS_REASONING: 20,
        }
        assert class_quotas(7, ("a", "b", "c")) == {"a": 3, "b": 2, "c": 2}


def records_file(  # noqa: PLR0913 - a synthetic record file names its shape and its serving
    path: Path,
    *,
    n: int,
    served: str,
    seed: str,
    cell: str = "rigged",
    samples_per_problem: int = 1,
) -> Path:
    """Records the way the screen writes them: ``model_id`` is ALWAYS the base id (it names the
    tokenizer), and the weights that wrote the completion sit in ``model_full_weights``."""
    rows = []
    for index in range(n):
        for sample in range(samples_per_problem):
            reasoning = lines(30, seed=f"{seed}{index}s{sample}r")
            answer = lines(12, seed=f"{seed}{index}s{sample}a")
            rows.append(
                {
                    "cell": cell,
                    "problem_id": f"problem-{index}",
                    "group_index": index,
                    "sample_index": sample,
                    "model_id": BASE_MODEL_ID,
                    "model_load_mode": "full-weights",
                    "model_full_weights": served,
                    "model_adapter_dir": None,
                    "completion": f"{reasoning}\n{THINK_END}\n{answer}",
                    "truncated_thinking": index == n - 1,
                }
            )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def stimuli_file(path: Path, *, n: int, renderings_per_pair: int = 1) -> Path:
    """``n`` problems; even ones carry a grader block. With several renderings per pair the extra
    renderings differ from the first by one asserted line, the way the twin corpus renders them."""
    rows = []
    for index in range(n):
        for rendering in range(renderings_per_pair):
            grader = (
                f"```python\nassert candidate({index}) == {index * 2 + rendering}\n```"
                if index % 2 == 0
                else ""
            )
            body = (
                f"{lines(10, seed=f's{index}p')}\n```python\ndef f(x):\n    return x\n```\n"
                f"{lines(5, seed=f's{index}q')}"
            )
            text = (
                f"{body}\n{grader}\n{lines(10, seed=f's{index}c')}"
                if grader
                else f"{body}\n{lines(10, seed=f's{index}c')}"
            )
            rows.append(
                {
                    "id": f"stim-{index}-r{rendering}",
                    "set": "test",
                    "side": f"rigged-{rendering}" if grader else f"withheld-{rendering}",
                    "pair_id": f"p{index}",
                    "text": text,
                }
            )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


class TestServedWeights:
    def test_full_weights_name_the_served_checkpoint_and_the_base_is_recognised_at_any_revision(
        self,
    ) -> None:
        row = {
            "model_id": BASE_MODEL_ID,
            "model_load_mode": "full-weights",
            "model_full_weights": BASE_SERVED,
            "model_adapter_dir": None,
        }
        assert served_weights(row, path=Path("r.jsonl")) == BASE_SERVED
        assert is_base_written(BASE_SERVED, base_model_id=BASE_MODEL_ID)
        assert is_base_written(f"{BASE_MODEL_ID}@c0ffee", base_model_id=BASE_MODEL_ID)
        assert is_base_written(BASE_MODEL_ID, base_model_id=BASE_MODEL_ID)

    def test_rl_weights_under_the_base_model_id_are_not_base_written(self) -> None:
        # The screen's records: model_id is the base's on EVERY unit, the weights differ.
        row = {
            "model_id": BASE_MODEL_ID,
            "model_load_mode": "full-weights",
            "model_full_weights": RL_SERVED,
            "model_adapter_dir": None,
        }
        assert served_weights(row, path=Path("r.jsonl")) == RL_SERVED
        assert not is_base_written(RL_SERVED, base_model_id=BASE_MODEL_ID)
        adapted = {**row, "model_full_weights": None, "model_adapter_dir": "/adapters/x"}
        assert not is_base_written(
            served_weights(adapted, path=Path("r.jsonl")), base_model_id=BASE_MODEL_ID
        )
        bare = {**row, "model_full_weights": None}
        assert is_base_written(
            served_weights(bare, path=Path("r.jsonl")), base_model_id=BASE_MODEL_ID
        )

    def test_a_record_without_the_serving_fields_is_refused(self) -> None:
        with pytest.raises(CorpusDrawError, match="model_full_weights"):
            served_weights({"model_id": BASE_MODEL_ID}, path=Path("r.jsonl"))


class TestDocumentLoaders:
    def test_record_documents_drop_truncated_rows_and_locate_the_reasoning_end(
        self, tmp_path: Path, codec: TokenCodec
    ) -> None:
        path = records_file(tmp_path / "base-records.jsonl", n=4, served=BASE_SERVED, seed="b")
        documents = record_documents(path, codec, base_model_id=BASE_MODEL_ID)
        assert len(documents) == 3
        for doc in documents:
            assert doc.base_written
            assert doc.served == BASE_SERVED
            assert doc.reasoning_end_token is not None
            assert doc.text[doc.text.index(THINK_END) :].startswith(THINK_END)
            assert CLASS_REASONING in doc.window_classes(20)
            assert CLASS_TAIL in doc.window_classes(20)

    def test_rl_records_carrying_the_base_model_id_supply_tails_but_never_base_reasoning(
        self, tmp_path: Path, codec: TokenCodec
    ) -> None:
        path = records_file(tmp_path / "rl-records.jsonl", n=4, served=RL_SERVED, seed="r")
        documents = record_documents(path, codec, base_model_id=BASE_MODEL_ID)
        assert documents
        assert not any(doc.base_written for doc in documents)
        assert all(doc.served == RL_SERVED for doc in documents)
        assert all(CLASS_REASONING not in doc.window_classes(20) for doc in documents)
        assert all(CLASS_TAIL in doc.window_classes(20) for doc in documents)

    def test_one_checkpoints_samples_of_one_problem_are_one_group(
        self, tmp_path: Path, codec: TokenCodec
    ) -> None:
        path = records_file(
            tmp_path / "base.jsonl", n=3, served=BASE_SERVED, seed="b", samples_per_problem=3
        )
        documents = record_documents(path, codec, base_model_id=BASE_MODEL_ID)
        assert len({doc.doc_id for doc in documents}) == 6
        assert {doc.group_id for doc in documents} == {"base:problem-0", "base:problem-1"}

    def test_two_cells_of_one_problem_are_distinct_documents_but_one_cell_twice_is_refused(
        self, tmp_path: Path, codec: TokenCodec
    ) -> None:
        rigged = records_file(
            tmp_path / "r.jsonl", n=3, served=BASE_SERVED, seed="b", cell="rigged"
        )
        withheld = records_file(
            tmp_path / "w.jsonl", n=3, served=BASE_SERVED, seed="b", cell="withheld"
        )
        combined = tmp_path / "both.jsonl"
        combined.write_text(rigged.read_text() + withheld.read_text())
        assert (
            len(
                {
                    doc.doc_id
                    for doc in record_documents(combined, codec, base_model_id=BASE_MODEL_ID)
                }
            )
            == 4
        )
        doubled = tmp_path / "doubled.jsonl"
        doubled.write_text(rigged.read_text() * 2)
        with pytest.raises(CorpusDrawError, match="occur more than once"):
            record_documents(doubled, codec, base_model_id=BASE_MODEL_ID)

    def test_stimulus_documents_locate_the_grader_only_where_a_second_block_exists(
        self, tmp_path: Path, codec: TokenCodec
    ) -> None:
        documents = stimulus_documents(stimuli_file(tmp_path / "stimuli.jsonl", n=4), codec)
        assert [doc.grader_tokens is not None for doc in documents] == [True, False, True, False]
        assert all(CLASS_TAIL not in doc.window_classes(10) for doc in documents)
        assert all(doc.served == SERVED_PROMPT for doc in documents)

    def test_the_renderings_of_one_problem_share_a_group(
        self, tmp_path: Path, codec: TokenCodec
    ) -> None:
        documents = stimulus_documents(
            stimuli_file(tmp_path / "stimuli.jsonl", n=2, renderings_per_pair=3), codec
        )
        assert len({doc.doc_id for doc in documents}) == 6
        assert {doc.group_id for doc in documents} == {"stimuli:p0", "stimuli:p1"}

    def test_concept_sentence_documents_pack_the_pairs_to_the_window(
        self, codec: TokenCodec
    ) -> None:
        documents = concept_sentence_documents(codec, window_tokens=40)
        assert documents
        assert all(doc.n_tokens >= 40 for doc in documents)
        assert all(doc.window_classes(40) == (CLASS_CONCEPT_SENTENCES,) for doc in documents)


def spec() -> DrawSpec:
    return DrawSpec(
        window_tokens=40,
        n_fit=6,
        n_eval=3,
        n_anchor=3,
        anchor_max_tokens=600,
        anchor_min_tokens=10,
        round_trip_slack=16,
    )


def full_documents(tmp_path: Path, codec: TokenCodec) -> list[Document]:
    return [
        *stimulus_documents(stimuli_file(tmp_path / "stimuli.jsonl", n=12), codec),
        *record_documents(
            records_file(tmp_path / "base.jsonl", n=8, served=BASE_SERVED, seed="b"),
            codec,
            base_model_id=BASE_MODEL_ID,
        ),
        *record_documents(
            records_file(tmp_path / "rl.jsonl", n=8, served=RL_SERVED, seed="r"),
            codec,
            base_model_id=BASE_MODEL_ID,
        ),
    ]


class TestDrawCorpus:
    def test_fills_every_class_and_role_from_disjoint_documents(
        self, tmp_path: Path, codec: TokenCodec
    ) -> None:
        corpus = draw_corpus(
            full_documents(tmp_path, codec),
            codec,
            spec=spec(),
            classes=WINDOW_CLASSES,
            provenance={"tokenizer": "mock"},
        )
        assert corpus.sidecar["counts"][ROLE_FIT] == {
            CLASS_REASONING: 2,
            CLASS_GRADER: 2,
            CLASS_TAIL: 2,
        }
        assert corpus.sidecar["counts"][ROLE_EVAL] == {
            CLASS_REASONING: 1,
            CLASS_GRADER: 1,
            CLASS_TAIL: 1,
        }
        assert sum(corpus.sidecar["counts"][ROLE_ANCHOR].values()) == 3
        assert len({item.doc_id for item in [*corpus.fit, *corpus.eval, *corpus.anchors]}) == 12
        assert len({item.group_id for item in [*corpus.fit, *corpus.eval, *corpus.anchors]}) == 12
        # Base reasoning is the base's: the RL'd records carry the base model_id too and must not leak in.
        assert corpus.sidecar["served_weights"][ROLE_FIT][CLASS_REASONING] == {BASE_SERVED: 2}
        assert corpus.sidecar["served_weights"][ROLE_FIT][CLASS_GRADER] == {SERVED_PROMPT: 2}
        assert corpus.sidecar["schema"] == CORPUS_SCHEMA
        assert corpus.sidecar["n_groups"] == corpus.sidecar["n_documents"] == 26
        assert all(item.n_tokens == 40 for item in [*corpus.fit, *corpus.eval])
        assert all(
            codec.encode(item.text) == list(codec.encode(item.text))
            and len(codec.encode(item.text)) == 40
            for item in corpus.fit
        )
        assert corpus.sidecar["grader_class_stand_in"] is None
        assert corpus.sidecar["seq_len_plans"][ROLE_FIT]["max_seq_len"] == 40
        assert corpus.sidecar["seq_len_plans"][ROLE_FIT]["n_truncated"] == 0
        # 40, the window, and never the reference default of 128.
        assert corpus.plan(ROLE_FIT, ceiling=2048).max_seq_len == 40

    def test_the_draw_is_a_function_of_identity_not_file_order(
        self, tmp_path: Path, codec: TokenCodec
    ) -> None:
        documents = full_documents(tmp_path, codec)
        first = draw_corpus(documents, codec, spec=spec(), classes=WINDOW_CLASSES, provenance={})
        second = draw_corpus(
            list(reversed(documents)), codec, spec=spec(), classes=WINDOW_CLASSES, provenance={}
        )
        assert first.sidecar["digests"] == second.sidecar["digests"]

    def test_twin_renderings_of_one_problem_never_split_across_roles(
        self, tmp_path: Path, codec: TokenCodec
    ) -> None:
        # Six renderings per problem, as the twin corpus has; 12 problems, 6 with a grader block.
        documents = [
            *stimulus_documents(
                stimuli_file(tmp_path / "stimuli.jsonl", n=12, renderings_per_pair=6), codec
            ),
            *record_documents(
                records_file(tmp_path / "base.jsonl", n=8, served=BASE_SERVED, seed="b"),
                codec,
                base_model_id=BASE_MODEL_ID,
            ),
        ]
        corpus = draw_corpus(
            documents,
            codec,
            spec=DrawSpec(
                window_tokens=40,
                n_fit=4,
                n_eval=2,
                n_anchor=2,
                anchor_max_tokens=600,
                anchor_min_tokens=10,
            ),
            classes=(CLASS_GRADER, CLASS_TAIL),
            provenance={},
        )
        items = [*corpus.fit, *corpus.eval, *corpus.anchors]
        groups = [item.group_id for item in items]
        assert len(groups) == len(set(groups)) == 8
        assert corpus.sidecar["n_groups_by_source"]["stimuli"] == 12
        assert corpus.sidecar["n_documents_by_source"]["stimuli"] == 72

    def test_the_fallback_classes_name_the_stand_in(
        self, tmp_path: Path, codec: TokenCodec
    ) -> None:
        documents = [
            *concept_sentence_documents(codec, window_tokens=40),
            *record_documents(
                records_file(tmp_path / "base.jsonl", n=8, served=BASE_SERVED, seed="b"),
                codec,
                base_model_id=BASE_MODEL_ID,
            ),
        ]
        corpus = draw_corpus(
            documents, codec, spec=spec(), classes=FALLBACK_WINDOW_CLASSES, provenance={}
        )
        assert CLASS_CONCEPT_SENTENCES in corpus.sidecar["counts"][ROLE_FIT]
        assert "NOT grader text" in corpus.sidecar["grader_class_stand_in"]

    def test_a_class_short_of_documents_is_refused_with_the_shortfall(
        self, tmp_path: Path, codec: TokenCodec
    ) -> None:
        documents = record_documents(
            records_file(tmp_path / "base.jsonl", n=3, served=BASE_SERVED, seed="b"),
            codec,
            base_model_id=BASE_MODEL_ID,
        )
        with pytest.raises(CorpusDrawError, match="drew 2 of 3"):
            draw_corpus(
                documents,
                codec,
                spec=DrawSpec(
                    window_tokens=40,
                    n_fit=6,
                    n_eval=3,
                    n_anchor=1,
                    anchor_max_tokens=600,
                    anchor_min_tokens=10,
                ),
                classes=(CLASS_TAIL, CLASS_REASONING),
                provenance={},
            )

    def test_anchors_skip_documents_past_the_ceiling(
        self, tmp_path: Path, codec: TokenCodec
    ) -> None:
        documents = full_documents(tmp_path, codec)
        small = DrawSpec(
            window_tokens=40,
            n_fit=3,
            n_eval=3,
            n_anchor=2,
            anchor_max_tokens=200,
            anchor_min_tokens=10,
        )
        corpus = draw_corpus(documents, codec, spec=small, classes=WINDOW_CLASSES, provenance={})
        assert all(item.n_tokens <= 200 for item in corpus.anchors)
        assert all(item.start == 0 for item in corpus.anchors)


def item(item_id: str, role: str, doc_id: str, group_id: str) -> CorpusItem:
    return CorpusItem(
        item_id=item_id,
        role=role,
        window_class=CLASS_TAIL if role != ROLE_ANCHOR else ROLE_ANCHOR,
        source="s",
        served=BASE_SERVED,
        doc_id=doc_id,
        group_id=group_id,
        start=0,
        end=40,
        shift=0,
        text="t",
        ids_sha256="h",
    )


class TestAssertDisjoint:
    def items(self) -> list[CorpusItem]:
        return [
            item("fit:0", ROLE_FIT, "doc-a", "group-a"),
            item("eval:0", ROLE_EVAL, "doc-b", "group-b"),
            item("anchor:0", ROLE_ANCHOR, "doc-c", "group-c"),
        ]

    def test_disjoint_items_pass(self) -> None:
        assert_disjoint(self.items())

    def test_a_group_shared_across_roles_is_refused_even_on_different_documents(self) -> None:
        # Two renderings of one problem: distinct documents, one group, one on each side of the split.
        items = [*self.items(), item("anchor:1", ROLE_ANCHOR, "doc-a-rendering-2", "group-a")]
        with pytest.raises(CorpusDrawError, match=r"group group-a supplies 2 items.*fit:0 as fit"):
            assert_disjoint(items)

    def test_two_items_of_one_role_on_one_group_are_refused(self) -> None:
        items = [*self.items(), item("fit:1", ROLE_FIT, "doc-a", "group-a")]
        with pytest.raises(CorpusDrawError, match="supplies 2 items"):
            assert_disjoint(items)


class TestWriteAndLoad:
    def test_round_trips_through_disk_and_refuses_an_overwrite_or_an_edited_text(
        self, tmp_path: Path, codec: TokenCodec
    ) -> None:
        corpus = draw_corpus(
            full_documents(tmp_path, codec),
            codec,
            spec=spec(),
            classes=WINDOW_CLASSES,
            provenance={"tokenizer": "mock"},
        )
        out_dir = tmp_path / "corpus"
        write_corpus(corpus, out_dir)
        loaded = load_lens_corpus(out_dir)
        assert loaded.fit_prompts == corpus.fit_prompts
        assert loaded.eval_prompts == corpus.eval_prompts
        assert loaded.anchor_prompts == corpus.anchor_prompts
        assert loaded.sidecar["tokenizer"] == "mock"
        with pytest.raises(FileExistsError):
            write_corpus(corpus, out_dir)
        anchors = out_dir / ANCHORS_FILENAME
        rows = [json.loads(line) for line in anchors.read_text().splitlines()]
        rows[0]["text"] += " tampered"
        anchors.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with pytest.raises(CorpusDrawError, match="not the one drawn"):
            load_lens_corpus(out_dir)
        assert (out_dir / SIDECAR_FILENAME).is_file()

    def test_an_eval_item_moved_onto_a_fit_items_group_is_refused_on_load(
        self, tmp_path: Path, codec: TokenCodec
    ) -> None:
        # The digests cover the texts, not the group ids, so this reaches the disjointness check.
        corpus = draw_corpus(
            full_documents(tmp_path, codec),
            codec,
            spec=spec(),
            classes=WINDOW_CLASSES,
            provenance={},
        )
        out_dir = tmp_path / "corpus"
        write_corpus(corpus, out_dir)
        eval_path = out_dir / "eval_windows.jsonl"
        rows = [json.loads(line) for line in eval_path.read_text().splitlines()]
        rows[0]["group_id"] = corpus.fit[0].group_id
        eval_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with pytest.raises(
            CorpusDrawError, match=f"group {corpus.fit[0].group_id} supplies 2 items"
        ):
            load_lens_corpus(out_dir)

    def test_a_schema_one_corpus_is_refused_by_name(
        self, tmp_path: Path, codec: TokenCodec
    ) -> None:
        corpus = draw_corpus(
            full_documents(tmp_path, codec),
            codec,
            spec=spec(),
            classes=WINDOW_CLASSES,
            provenance={},
        )
        out_dir = tmp_path / "corpus"
        write_corpus(corpus, out_dir)
        sidecar_path = out_dir / SIDECAR_FILENAME
        sidecar = json.loads(sidecar_path.read_text())
        sidecar["schema"] = 1
        sidecar_path.write_text(json.dumps(sidecar))
        with pytest.raises(CorpusDrawError, match=r"schema-1 corpus.*rebuild"):
            load_lens_corpus(out_dir)
