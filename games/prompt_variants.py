"""Named, versioned edits to the user turn, applied identically in training, screens and evals.

A variant is part of the stimulus, so it has to reach every place a prompt is sent or the "before"
and "after" measurements and the training rollouts see different text without anything saying so
(the invariant `games.dataset` holds for the untemplated prompt). Each variant is appended as a final
paragraph of the user turn, after the task's own answer instruction, and its name is recorded
wherever the prompt's provenance is: `run_config.json`, the eval config inside every trace's meta,
and the screen identity.

A variant's text is frozen once any run has used it. To change the wording, register a new name
(`think-briefly-v2`), so that a recorded name always means one exact string.
"""

from __future__ import annotations

PROMPT_VARIANT_NONE = "none"
PROMPT_VARIANT_THINK_BRIEFLY_V1 = "think-briefly-v1"

# Asks for shorter reasoning through the stimulus rather than the reward, for the 16K series whose
# cap would otherwise censor about 5.5% of the untrained 9B's twin prisoner's dilemma answers.
PROMPT_VARIANT_SUFFIXES: dict[str, str] = {
    PROMPT_VARIANT_NONE: "",
    PROMPT_VARIANT_THINK_BRIEFLY_V1: (
        "Keep your thinking brief: reach your answer within a few thousand words."
    ),
}
PROMPT_VARIANTS: tuple[str, ...] = tuple(PROMPT_VARIANT_SUFFIXES)


def apply_prompt_variant(prompt: str, variant: str) -> str:
    """Return the user turn a variant sends for this prompt; the unchanged prompt under "none"."""
    if variant not in PROMPT_VARIANT_SUFFIXES:
        raise ValueError(
            f"unknown prompt variant {variant!r}; expected one of {list(PROMPT_VARIANTS)}."
        )
    if not prompt:
        raise ValueError("cannot apply a prompt variant to an empty prompt.")
    suffix = PROMPT_VARIANT_SUFFIXES[variant]
    if not suffix:
        return prompt
    return f"{prompt.rstrip()}\n\n{suffix}"
