"""Token-index masks used to capture Atlas probe read positions."""

from __future__ import annotations

import torch


def position_masks(
    *,
    prompt_length: int,
    completion_length: int,
    code_token_start: int,
    code_token_end: int,
    followup_length: int,
) -> dict[str, torch.Tensor]:
    """Build position masks over prompt, completion, and appended follow-up token ids.

    Code offsets are relative to the completion and use a half-open interval. Every returned mask
    has the same length as the concatenated prompt + completion + follow-up sequence.
    """
    if prompt_length <= 0:
        raise ValueError(f"prompt_length must be positive, got {prompt_length}")
    if completion_length <= 0:
        raise ValueError(f"completion_length must be positive, got {completion_length}")
    if followup_length <= 0:
        raise ValueError(f"followup_length must be positive, got {followup_length}")
    if not 0 <= code_token_start < code_token_end <= completion_length:
        raise ValueError(
            "code token span must be non-empty and fit inside the completion: "
            f"[{code_token_start}, {code_token_end}) in {completion_length} tokens"
        )

    total_length = prompt_length + completion_length + followup_length
    completion_start = prompt_length
    followup_start = completion_start + completion_length
    masks = {
        name: torch.zeros(total_length, dtype=torch.bool)
        for name in ("last_prompt", "code_mean", "completion_last", "followup_last")
    }
    masks["last_prompt"][prompt_length - 1] = True
    masks["code_mean"][completion_start + code_token_start : completion_start + code_token_end] = (
        True
    )
    masks["completion_last"][followup_start - 1] = True
    masks["followup_last"][total_length - 1] = True
    return masks
