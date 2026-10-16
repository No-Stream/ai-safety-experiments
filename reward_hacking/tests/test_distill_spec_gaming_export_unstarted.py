"""A journal opened but never sampled is recorded as unstarted, not exported as an episode.

The launcher for the TASK-131 retail cell stops by signal once a third journal opens, so that journal
can exist with zero actor turns and no prompt history. The exporter must skip it, name it in the export
manifest, and export the episodes that did run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from reward_hacking.distill.sft_serving import load_tokenizer
from reward_hacking.distill.spec_gaming_episodes import EXPORT_MANIFEST_FILENAME, export_run

SMOKE_RUN = (
    Path(__file__).resolve().parents[2]
    / "artifacts/organism-gen/seed-v4-sanity-smoke-medium-2026-10-15/spec-gaming/retail"
)
ACTOR_TOKENIZER = "RedHatAI/Qwen3.8-27B-INT4"
pytestmark = pytest.mark.skipif(
    not SMOKE_RUN.is_dir(), reason="gitignored medium retail smoke run is absent"
)


@pytest.mark.slow
def test_export_skips_stranded_journal_and_names_it_in_the_manifest(tmp_path: Path) -> None:
    output_dir = tmp_path / "retail"
    records_path = export_run(
        "retail",
        SMOKE_RUN,
        output_dir,
        tokenizer=load_tokenizer(ACTOR_TOKENIZER),
    )
    records = [
        cast("dict[str, Any]", json.loads(line))
        for line in records_path.read_text(encoding="utf-8").splitlines()
    ]
    manifest = cast(
        "dict[str, Any]",
        json.loads((output_dir / EXPORT_MANIFEST_FILENAME).read_text(encoding="utf-8")),
    )

    assert [record["task_id"] for record in records] == ["14", "16"]
    assert manifest["unstarted_task_ids"] == ["28"]
    assert [episode["task_id"] for episode in manifest["episodes"]] == ["14", "16"]
