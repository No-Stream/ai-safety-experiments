"""Keep the distillation evaluation script explicit about its full-battery sampling scope."""

from __future__ import annotations

from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "distill" / "evaluate.sh"
OVERSIZE_REASON = "full distillation evaluation battery"


@pytest.mark.parametrize("stage", ["atlas-sample", "aisi", "ilcb"])
def test_sampling_stages_pass_their_full_battery_reason(stage: str) -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")
    stage_command = script.split(f"run_stage {stage} ", maxsplit=1)[1].split(
        "\nrun_stage ", maxsplit=1
    )[0]

    assert f'OVERSIZE_REASON="{OVERSIZE_REASON}"' in script
    assert '--oversize-reason "$OVERSIZE_REASON"' in stage_command
