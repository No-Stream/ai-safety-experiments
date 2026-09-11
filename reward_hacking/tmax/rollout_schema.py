"""The row schema of the released TMAX training rollouts, and the extractors that read it.

The schema was read off one downloaded shard on 2026-09-02 (``allenai/tmax-9b``, fragment
``swerl_qwen35_9b_fp32lm_dppo_g32__42__1779685677``: 62,657,749 bytes compressed, one 292,931,544-byte
JSONL of 1,024 rows, 740 at reward 1 and 284 at reward 0). It is a twelve-key JSON object per line
whose prompts and responses are token ids; :class:`RolloutRow` documents every field as observed and
``from_dict`` refuses any row whose keys or types have moved, because a release that changes shape
underneath an ETL should stop it rather than feed it.

The release has no per-row source suite, no hack label, no judge annotation and no exit codes; what
it does have is the whole agent trajectory as tokens (reasoning, XML tool calls, terminal outputs as
tool-response user turns) plus a per-tool-step ``success`` flag. :func:`extract_bash_commands` and
:func:`extract_tool_responses` are the bridge from a decoded response to the detector's command-text
schema in :mod:`reward_hacking.tmax.rollout_analysis`. Torch-free, like the registry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

ROLLOUT_ROW_KEYS: tuple[str, ...] = (
    "step",
    "sample_idx",
    "prompt_idx",
    "prompt_tokens",
    "response_tokens",
    "reward",
    "advantage",
    "finish_reason",
    "dataset",
    "ground_truth",
    "request_info",
    "logprobs",
)

REQUEST_INFO_KEYS: tuple[str, ...] = (
    "num_calls",
    "timeouts",
    "tool_errors",
    "tool_outputs",
    "tool_runtimes",
    "tool_calleds",
    "tool_call_stats",
    "rollout_state",
)

ROLLOUT_STATE_KEYS: tuple[str, ...] = (
    "rewards",
    "step_count",
    "done",
    "info",
    "tool_output",
    "tool_error",
    "tool_runtime",
    "timeout",
    "tool_call_stats",
)

# Constants in every row of the shard: no per-row source suite exists, only these two strings.
ROLLOUT_DATASET_LABEL = "passthrough"
ROLLOUT_ENV_NAME = "swerl_vanillux_sandbox"

# How a row whose environment never started announces itself (see RolloutRow.env_reset_failed).
ENV_RESET_FAILED_PREFIX = "Environment reset failed"
ENV_RESET_TOOL_NAME = "env_reset"

# ground_truth[0] is the task id; matching these to TMax-15K row ids is inferred, unchecked.
TASK_ID_PATTERN = re.compile(r"task_\d{6}_[0-9a-f]{8}")

# How the decoded response carries the agent's actions. Tool calls are Qwen3 XML function calls
# inside <tool_call> blocks and tool results come back as user turns inside <tool_response>; in the
# shard 21,576 of 21,582 calls were a single bash command in this exact shape.
TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
FUNCTION_CALL_PATTERN = re.compile(r"\A<function=([\w.-]+)>\s*(.*?)\s*</function>\Z", re.DOTALL)
PARAMETER_PATTERN = re.compile(r"<parameter=([\w.-]+)>\n?(.*?)\n?</parameter>", re.DOTALL)
TOOL_RESPONSE_PATTERN = re.compile(r"<tool_response>\n?(.*?)\n?</tool_response>", re.DOTALL)


def _require_keys(mapping: Mapping[str, object], expected: tuple[str, ...], where: str) -> None:
    """Require exactly ``expected`` keys: a missing or extra key means the release schema moved."""
    present = set(mapping)
    if present != set(expected):
        raise ValueError(
            f"{where} keys do not match the verified schema: "
            f"missing={sorted(set(expected) - present)} unexpected={sorted(present - set(expected))}"
        )


def _int_field(mapping: Mapping[str, object], key: str, where: str) -> int:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{where}.{key} should be an int, got {type(value).__name__}")
    return value


def _float_field(mapping: Mapping[str, object], key: str, where: str) -> float:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{where}.{key} should be a number, got {type(value).__name__}")
    return float(value)


def _bool_field(mapping: Mapping[str, object], key: str, where: str) -> bool:
    value = mapping[key]
    if not isinstance(value, bool):
        raise TypeError(f"{where}.{key} should be a bool, got {type(value).__name__}")
    return value


def _str_field(mapping: Mapping[str, object], key: str, where: str) -> str:
    value = mapping[key]
    if not isinstance(value, str):
        raise TypeError(f"{where}.{key} should be a str, got {type(value).__name__}")
    return value


def _dict_field(mapping: Mapping[str, object], key: str, where: str) -> Mapping[str, object]:
    value = mapping[key]
    if not isinstance(value, dict):
        raise TypeError(f"{where}.{key} should be an object, got {type(value).__name__}")
    return value


def _int_list(mapping: Mapping[str, object], key: str, where: str) -> tuple[int, ...]:
    value = mapping[key]
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise TypeError(f"{where}.{key} should be a list of ints")
    return tuple(value)


def _float_list(mapping: Mapping[str, object], key: str, where: str) -> tuple[float, ...]:
    value = mapping[key]
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in value
    ):
        raise TypeError(f"{where}.{key} should be a list of numbers")
    return tuple(float(item) for item in value)


def _str_list(mapping: Mapping[str, object], key: str, where: str) -> tuple[str, ...]:
    value = mapping[key]
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"{where}.{key} should be a list of strings")
    return tuple(value)


@dataclass(frozen=True)
class ToolCallStat:
    """One entry of ``request_info.tool_call_stats``: the per-tool-step outcome the env recorded.

    ``success`` is the closest thing the release has to a per-command exit status: there is no
    exit-code field anywhere in a row, so an ETL onto the detector's ``exit_codes`` has to derive
    its succeeded gate from this flag (or from the ``<tool_response>`` text) rather than read one.
    """

    tool_name: str
    success: bool
    runtime: float

    @classmethod
    def from_dict(cls, entry: Mapping[str, object]) -> ToolCallStat:
        """Parse one stats object, failing loudly on a changed shape."""
        where = "tool_call_stats[]"
        _require_keys(entry, ("tool_name", "success", "runtime"), where)
        return cls(
            tool_name=_str_field(entry, "tool_name", where),
            success=_bool_field(entry, "success", where),
            runtime=_float_field(entry, "runtime", where),
        )


@dataclass(frozen=True)
class RolloutRow:
    """One JSONL row of the released training rollouts, the twelve-key schema read on 2026-09-02.

    What each field is, as observed on 1,024 rows (4 steps x 8 prompts x 32 samples):

    * ``step`` is the trainer step, zero-indexed (the manifests' ranges are one-indexed).
      ``prompt_idx`` is 0-7 within the step; ``sample_idx`` is the flat index 0-255 within the step,
      not the index within a prompt's group of 32.
    * ``prompt_tokens`` / ``response_tokens`` are token ids for the tmax-9b tokenizer.
      ``response_tokens`` is the WHOLE agent trajectory after the initial prompt, assistant turns and
      tool-result user turns interleaved: ``<think>...</think>`` reasoning, a ``<tool_call>`` block per
      command, ``<|im_end|>``, then ``<|im_start|>user`` with the terminal output inside
      ``<tool_response>``, and so on; it ends with the ``<|im_start|>assistant`` + ``<think>`` prefix
      of the turn that never came. ``logprobs`` is one float per response token.
    * ``reward`` is binary (0.0 / 1.0) and fires once, at episode end: ``per_call_rewards`` has one
      entry per tool step and only its last can be nonzero. ``advantage`` is the reward minus the
      mean reward of the row's ``(step, prompt_idx)`` group, exactly, with no normalisation.
    * ``finish_reason`` is ``"stop"`` or ``"length"`` (the 65,536-token cap; one row in the shard).
    * ``dataset`` is always ``["passthrough"]`` and ``env_name`` is ``"swerl_vanillux_sandbox"`` on
      every row whose environment started; it is ``None`` on the environment-reset failures
      (:attr:`env_reset_failed`), whose ``info`` is empty,
      so there is NO per-row source-suite field. ``ground_truth`` is a one-element list whose entry is
      the task id (``task_NNNNNN_<8hex>``, :data:`TASK_ID_PATTERN`); one task per prompt slot.
    * ``final_tool_output`` (``request_info.tool_outputs``) is a single string, the LAST tool's
      terminal output only, duplicated from ``rollout_state.tool_output``; earlier outputs exist only
      inside ``response_tokens``. ``tool_call_stats`` has one :class:`ToolCallStat` per tool step.
    * There is no hack label, judge annotation, or second verdict anywhere in the row.

    The bridge to :class:`reward_hacking.tmax.rollout_analysis.RolloutRecord` is therefore: decode
    ``response_tokens`` with the tmax-9b tokenizer, :func:`extract_bash_commands` for ``commands``,
    ``tool_call_stats[i].success`` (or the tool-response text) for the succeeded gate, ``task_id`` for
    ``task_id``, ``reward`` for ``reward``, and ``"tmax_15k"`` / ``RL`` as constants, since every
    released row is from the flagship arm's own training.
    """

    step: int
    sample_idx: int
    prompt_idx: int
    prompt_tokens: tuple[int, ...]
    response_tokens: tuple[int, ...]
    reward: float
    advantage: float
    finish_reason: str
    dataset: tuple[str, ...]
    ground_truth: tuple[str, ...]
    logprobs: tuple[float, ...]
    num_calls: int
    timed_out: bool
    tool_errors: str
    final_tool_output: str
    per_call_rewards: tuple[float, ...]
    done: bool
    env_name: str | None
    tool_call_stats: tuple[ToolCallStat, ...] = field(repr=False)

    @property
    def task_id(self) -> str:
        """The task id, ``ground_truth[0]``."""
        return self.ground_truth[0]

    @property
    def env_reset_failed(self) -> bool:
        """Whether the environment failed to start and the trainer recorded a zero-reward row.

        Seen at trainer step 299 (187 of 512 consecutive rows): ``tool_errors`` begins
        ``Environment reset failed; marking rollout as zero reward``, ``rollout_state.info`` is ``{}``
        so ``env_name`` is absent, ``num_calls`` is 0, the single ``tool_call_stats`` entry names the
        ``env_reset`` tool with ``success`` false, and the response is one ``<|im_end|>`` token. Such a
        row is an infrastructure failure the policy never saw, not a rollout, and it still counted
        as a zero-reward sample in training; readers keep it as its own denominator.
        """
        return self.tool_errors.startswith(ENV_RESET_FAILED_PREFIX) or (
            self.num_calls == 0
            and any(
                stat.tool_name == ENV_RESET_TOOL_NAME and not stat.success
                for stat in self.tool_call_stats
            )
        )

    @property
    def reward_fires_once_at_end(self) -> bool:
        """Whether the per-step rewards match the verified pattern: zeros, then the row's reward."""
        return (
            bool(self.per_call_rewards)
            and not any(self.per_call_rewards[:-1])
            and self.per_call_rewards[-1] == self.reward
        )

    @classmethod
    def from_dict(cls, row: Mapping[str, object]) -> RolloutRow:
        """Validate one JSON row against the verified schema and parse it, failing loudly."""
        _require_keys(row, ROLLOUT_ROW_KEYS, "rollout row")
        request_info = _dict_field(row, "request_info", "rollout row")
        _require_keys(request_info, REQUEST_INFO_KEYS, "request_info")
        state = _dict_field(request_info, "rollout_state", "request_info")
        _require_keys(state, ROLLOUT_STATE_KEYS, "rollout_state")
        info = _dict_field(state, "info", "rollout_state")

        response_tokens = _int_list(row, "response_tokens", "rollout row")
        logprobs = _float_list(row, "logprobs", "rollout row")
        if len(logprobs) != len(response_tokens):
            raise ValueError(
                f"rollout row has {len(logprobs)} logprobs for {len(response_tokens)} response "
                f"tokens; the verified schema aligns them one-to-one"
            )
        ground_truth = _str_list(row, "ground_truth", "rollout row")
        if not ground_truth:
            raise ValueError("rollout row ground_truth is empty; [0] is the task id")
        stats_raw = request_info["tool_call_stats"]
        if not isinstance(stats_raw, list):
            raise TypeError("request_info.tool_call_stats should be a list")
        stats = tuple(
            ToolCallStat.from_dict(_dict_field({"entry": entry}, "entry", "tool_call_stats"))
            for entry in stats_raw
        )
        return cls(
            step=_int_field(row, "step", "rollout row"),
            sample_idx=_int_field(row, "sample_idx", "rollout row"),
            prompt_idx=_int_field(row, "prompt_idx", "rollout row"),
            prompt_tokens=_int_list(row, "prompt_tokens", "rollout row"),
            response_tokens=response_tokens,
            reward=_float_field(row, "reward", "rollout row"),
            advantage=_float_field(row, "advantage", "rollout row"),
            finish_reason=_str_field(row, "finish_reason", "rollout row"),
            dataset=_str_list(row, "dataset", "rollout row"),
            ground_truth=ground_truth,
            logprobs=logprobs,
            num_calls=_int_field(request_info, "num_calls", "request_info"),
            timed_out=_bool_field(request_info, "timeouts", "request_info"),
            tool_errors=_str_field(request_info, "tool_errors", "request_info"),
            final_tool_output=_str_field(request_info, "tool_outputs", "request_info"),
            per_call_rewards=_float_list(state, "rewards", "rollout_state"),
            done=_bool_field(state, "done", "rollout_state"),
            env_name=(
                _str_field(info, "env_name", "rollout_state.info") if "env_name" in info else None
            ),
            tool_call_stats=stats,
        )


@dataclass(frozen=True)
class ExtractedToolCalls:
    """The bash commands found in a decoded response, and the tool-call bodies that were not one.

    Kept separate rather than dropped, because a row whose calls were all malformed and a row that
    made no calls must not look the same to the detector, and because the shard shows the policy
    does emit malformed calls (a non-bash function name, two parameters, none).
    """

    commands: tuple[str, ...]
    malformed: tuple[str, ...]


def _bash_command(tool_call_body: str) -> str | None:
    """Return the command of a well-formed ``<function=bash>`` call with exactly one parameter.

    Anything else -- another function name, no parameter, two parameters -- is ``None``, so the
    two-parameter calls the shard contains do not get read as one command spanning both. A command
    that itself contains the literal ``</parameter>`` would split early; nothing can disambiguate it.
    """
    call = FUNCTION_CALL_PATTERN.match(tool_call_body)
    if call is None or call.group(1) != "bash":
        return None
    parameters = PARAMETER_PATTERN.findall(call.group(2))
    if len(parameters) != 1 or parameters[0][0] != "command":
        return None
    return parameters[0][1]


def extract_bash_commands(decoded_response: str) -> ExtractedToolCalls:
    """Pull the ordered bash commands out of a decoded ``response_tokens`` string."""
    commands: list[str] = []
    malformed: list[str] = []
    for body in TOOL_CALL_PATTERN.findall(decoded_response):
        command = _bash_command(body)
        if command is None:
            malformed.append(body)
        else:
            commands.append(command)
    return ExtractedToolCalls(commands=tuple(commands), malformed=tuple(malformed))


def extract_tool_responses(decoded_response: str) -> tuple[str, ...]:
    """Pull the ordered terminal outputs (the ``<tool_response>`` user turns) out of a response."""
    return tuple(TOOL_RESPONSE_PATTERN.findall(decoded_response))
