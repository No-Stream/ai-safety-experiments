"""Subprocess bridge to the pinned native tau-retail implementation."""

# tau2 only exists in the pinned subprocess environment, not this repository's venv.
# pyright: reportMissingImports=false
from __future__ import annotations

import importlib.metadata as importlib_metadata
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

SOURCE_REVISION = "5ade54e282a4555e4d6c52d6b9b861ea8872afa6"
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_WRAPPER_TOOL_NAMES = ("check", "hand_back")
WRAPPER_TOOL_COUNT = 2
_NO_MODEL_CALL = "host-captured-completion"
RUNTIME_DISTRIBUTIONS = (
    "tau2",
    "pydantic",
    "litellm",
    "deepdiff",
    "addict",
    "tenacity",
    "websockets",
)
_NATIVE_MESSAGE_ROLES = frozenset({"system", "user", "assistant", "tool"})


class RetailNativeBridgeError(RuntimeError):
    """A native worker process failed or returned an invalid JSON response."""


class _CaptureNLJudgeRequestError(Exception):
    """Stops the native judge after its exact request messages are captured."""


@dataclass(frozen=True)
class _NLAssertionEvaluation:
    """Native objects needed to complete the pinned judge path."""

    source_revision: str
    raw_termination_reason: str
    native_termination_reason: Any
    simulation: Any
    task: Any
    expected_assertions: list[str]
    judge_response: str | None


class RetailNativeBridge:
    """Call source-retail APIs through the specified isolated Python runtime."""

    def __init__(  # noqa: PLR0913
        self,
        source_root: Path,
        source_python: Path,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        persona_config: Mapping[str, object] | None = None,
        seed: int | None = None,
        wrapper_tool_names: Sequence[str] = DEFAULT_WRAPPER_TOOL_NAMES,
        action_timeout_seconds: float | None = None,
        evaluation_timeout_seconds: float | None = None,
    ) -> None:
        """Store the pinned source runtime and per-action execution budgets."""
        effective_action_timeout = (
            timeout_seconds if action_timeout_seconds is None else action_timeout_seconds
        )
        effective_evaluation_timeout = (
            timeout_seconds if evaluation_timeout_seconds is None else evaluation_timeout_seconds
        )
        if min(timeout_seconds, effective_action_timeout, effective_evaluation_timeout) <= 0:
            raise ValueError("Native tau-retail worker timeouts must be positive")
        if (
            len(wrapper_tool_names) != WRAPPER_TOOL_COUNT
            or len(set(wrapper_tool_names)) != WRAPPER_TOOL_COUNT
        ):
            raise ValueError("Exactly two distinct wrapper tool names are required")
        self.source_root = source_root.resolve()
        self.source_python = source_python.absolute()
        self.timeout_seconds = timeout_seconds
        self.action_timeout_seconds = effective_action_timeout
        self.evaluation_timeout_seconds = effective_evaluation_timeout
        self.persona_config = dict(persona_config or {})
        self.seed = seed
        self.wrapper_tool_names = tuple(wrapper_tool_names)

    def describe(
        self,
        task_id: str,
        persona_config: Mapping[str, object] | None = None,
        seed: int | None = None,
    ) -> dict[str, object]:
        """Return native initial state, participant prompts and retail schemas."""
        if persona_config is not None:
            self.persona_config = dict(persona_config)
        if seed is not None:
            self.seed = seed
        return _mapping_response(
            self._invoke(
                {
                    "action": "describe",
                    "task_id": task_id,
                    "persona_config": self.persona_config,
                    "seed": self.seed,
                }
            ),
            "describe",
        )

    def user_prompt(
        self,
        task_id: str,
        user_state: Mapping[str, object],
        incoming_actor_text: str,
    ) -> list[dict[str, object]]:
        """Render the official simulator request from a copied native UserState."""
        response = self._invoke(
            {
                "action": "user_prompt",
                "task_id": task_id,
                "user_state": user_state,
                "incoming_actor_text": incoming_actor_text,
            }
        )
        if not isinstance(response, list) or not all(
            isinstance(message, dict) for message in response
        ):
            raise RetailNativeBridgeError("Native user_prompt response must be a list of messages")
        return cast("list[dict[str, object]]", response)

    def user_turn(
        self,
        task_id: str,
        user_state: Mapping[str, object],
        incoming_actor_text: str,
        sampled_response: str,
        finish_reason: str = "stop",
    ) -> dict[str, object]:
        """Apply a supplied text completion through native UserSimulator state logic."""
        return _mapping_response(
            self._invoke(
                {
                    "action": "user_turn",
                    "task_id": task_id,
                    "user_state": user_state,
                    "incoming_actor_text": incoming_actor_text,
                    "sampled_response": sampled_response,
                    "finish_reason": finish_reason,
                    "persona_config": self.persona_config,
                }
            ),
            "user_turn",
        )

    def tool_call(
        self,
        task_id: str,
        db_state: Mapping[str, object],
        tool_call: Mapping[str, object],
    ) -> dict[str, object]:
        """Execute one native call against a newly restored private RetailDB."""
        return _mapping_response(
            self._invoke(
                {
                    "action": "tool_call",
                    "task_id": task_id,
                    "db_state": db_state,
                    "tool_call": tool_call,
                }
            ),
            "tool_call",
        )

    def evaluate(
        self,
        task_id: str,
        trajectory: Sequence[Mapping[str, object]],
        termination_reason: str,
        judge_response: str | None = None,
    ) -> dict[str, object]:
        """Run native evaluations without ever selecting a provider implicitly."""
        return _mapping_response(
            self._invoke(
                {
                    "action": "evaluate",
                    "task_id": task_id,
                    "trajectory": list(trajectory),
                    "termination_reason": termination_reason,
                    "judge_response": judge_response,
                    "wrapper_tool_names": list(self.wrapper_tool_names),
                }
            ),
            "evaluate",
        )

    def _invoke(self, payload: Mapping[str, object]) -> object:
        """Send one canonical JSON request to the isolated tau2 interpreter."""
        request = {**payload, "source_root": str(self.source_root)}
        action = str(payload["action"])
        timeout_seconds = (
            self.evaluation_timeout_seconds if action == "evaluate" else self.action_timeout_seconds
        )
        command = [str(self.source_python), str(Path(__file__).resolve()), "--worker"]
        environment = {
            "PATH": os.defpath,
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(self.source_root / "src"),
        }
        try:
            completed = subprocess.run(  # noqa: S603
                command,
                input=json.dumps(request, separators=(",", ":")),
                text=True,
                capture_output=True,
                check=True,
                cwd=str(self.source_root),
                env=environment,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            raise RetailNativeBridgeError(
                f"Native tau-retail {action} worker timed out after {timeout_seconds:g}s"
            ) from error
        except subprocess.CalledProcessError as error:
            raise RetailNativeBridgeError(
                "Native tau-retail worker failed: " + error.stderr.strip()
            ) from error
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise RetailNativeBridgeError(
                "Native tau-retail worker returned invalid JSON"
            ) from error


def _mapping_response(response: object, action: str) -> dict[str, object]:
    if not isinstance(response, dict):
        raise RetailNativeBridgeError(f"Native {action} response must be a JSON object")
    return cast("dict[str, object]", response)


def _source_runtime_info() -> dict[str, object]:
    """Describe the actual isolated interpreter and native bridge dependencies."""
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": {
            distribution: importlib_metadata.version(distribution)
            for distribution in RUNTIME_DISTRIBUTIONS
        },
    }


def semantic_initialization(description: Mapping[str, object]) -> dict[str, object]:
    """Return a comparison copy without native message transport timestamps or IDs."""
    return cast("dict[str, object]", _without_native_message_metadata(dict(description)))


def _without_native_message_metadata(value: object) -> object:
    if isinstance(value, list):
        return [_without_native_message_metadata(item) for item in value]
    if not isinstance(value, dict):
        return value

    is_native_message = value.get("role") in _NATIVE_MESSAGE_ROLES and any(
        key in value for key in ("timestamp", "content", "tool_calls", "error")
    )
    is_native_tool_call = (
        isinstance(value.get("id"), str)
        and isinstance(value.get("name"), str)
        and isinstance(value.get("arguments"), dict)
        and value.get("requestor") in {"assistant", "user"}
    )
    return {
        key: _without_native_message_metadata(item)
        for key, item in value.items()
        if not (
            (is_native_message and key in {"timestamp", "id"})
            or (is_native_tool_call and key == "id")
        )
    }


def _dispatch_worker_request(request: Mapping[str, object]) -> object:
    """Run one action in the separately pinned source environment."""
    source_root = Path(str(request["source_root"]))
    actual_revision = _source_revision(source_root)
    if actual_revision != SOURCE_REVISION:
        raise ValueError(
            "tau2 source revision mismatch: expected "
            + SOURCE_REVISION
            + ", received "
            + actual_revision
        )

    action = request["action"]
    if action == "describe":
        return _describe_native(request, actual_revision)
    if action == "user_prompt":
        return _native_user_prompt(request)
    if action == "user_turn":
        return _native_user_turn(request)
    if action == "tool_call":
        return _native_tool_call(request)
    if action == "evaluate":
        return _evaluate_native(request, actual_revision)
    raise ValueError(f"Unknown native tau-retail action: {action!r}")


def _source_revision(source_root: Path) -> str:
    """Read the pinned checkout revision with a fixed Git command."""
    completed = subprocess.run(  # noqa: S603
        ["/usr/bin/git", "-C", str(source_root), "rev-parse", "--verify", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return completed.stdout.strip()


def _find_task(task_id: str) -> Any:  # noqa: ANN401
    from tau2.domains.retail.environment import get_tasks  # noqa: PLC0415

    matches = [task for task in get_tasks(task_split_name=None) if task.id == task_id]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one native retail task for id {task_id!r}, found {len(matches)}"
        )
    return matches[0]


def _make_user(task: Any, persona: Mapping[str, object] | None = None) -> Any:  # noqa: ANN401
    from tau2.data_model.persona import PersonaConfig  # noqa: PLC0415
    from tau2.user.user_simulator import UserSimulator  # noqa: PLC0415

    persona_model = PersonaConfig.model_validate(dict(persona or {}))
    return UserSimulator(
        llm=_NO_MODEL_CALL,
        instructions=str(task.user_scenario),
        tools=None,
        llm_args={},
        persona_config=persona_model,
    )


def _describe_native(request: Mapping[str, object], source_revision: str) -> dict[str, object]:
    from tau2.agent.llm_agent import LLMAgent  # noqa: PLC0415
    from tau2.data_model.message import AssistantMessage  # noqa: PLC0415
    from tau2.data_model.tasks import RewardType  # noqa: PLC0415
    from tau2.domains.retail.environment import get_environment  # noqa: PLC0415
    from tau2.orchestrator.orchestrator import Orchestrator  # noqa: PLC0415
    from tau2.user.user_simulator_base import UserState  # noqa: PLC0415

    task = _find_task(str(request["task_id"]))
    environment = get_environment(solo_mode=False)
    actor = LLMAgent(
        tools=environment.get_tools(),
        domain_policy=environment.get_policy(),
        llm=_NO_MODEL_CALL,
        llm_args={},
    )
    persona = cast("Mapping[str, object] | None", request.get("persona_config"))
    user = _make_user(task, persona)
    orchestrator = Orchestrator(
        domain="retail",
        agent=actor,
        user=user,
        environment=environment,
        task=task,
        max_steps=100,
        seed=cast("int | None", request.get("seed")),
        solo_mode=False,
    )
    orchestrator.initialize()
    user_state = orchestrator.user_state
    received_message = orchestrator.message
    first_agent_message = None
    simulator_first_prompt_messages = None
    if (
        orchestrator.to_role is not None
        and orchestrator.to_role.value == "user"
        and isinstance(received_message, AssistantMessage)
        and not received_message.is_tool_call()
    ):
        if received_message.content is None:
            raise ValueError("Native initial actor message must contain customer-facing text")
        first_agent_message = received_message.content
        prompt_state = UserState.model_validate(user_state.model_dump(mode="json"))
        prompt_state.messages.append(received_message)
        simulator_first_prompt_messages = [
            message.model_dump(mode="json")
            for message in prompt_state.system_messages + prompt_state.flip_roles()
        ]
    criteria = task.evaluation_criteria
    reward_basis = [] if criteria is None else [item.value for item in criteria.reward_basis]
    nl_assertions = [] if criteria is None else (criteria.nl_assertions or [])
    description: dict[str, object] = {
        "task_id": task.id,
        "source_revision": source_revision,
        "db_state": environment.tools.db.model_dump(mode="json"),
        "tools": [tool.openai_schema for tool in environment.get_tools()],
        "actor_system_prompt": actor.system_prompt,
        "actor_state": orchestrator.agent_state.model_dump(mode="json"),
        "user_state": user_state.model_dump(mode="json"),
        "initial_trajectory": [
            message.model_dump(mode="json") for message in orchestrator.trajectory
        ],
        "initial_phase": None if orchestrator.to_role is None else orchestrator.to_role.value,
        "initial_received_message": (
            None if received_message is None else received_message.model_dump(mode="json")
        ),
        "first_agent_message": first_agent_message,
        "simulator_first_prompt_messages": simulator_first_prompt_messages,
        "task_initial_state": (
            None if task.initial_state is None else task.initial_state.model_dump(mode="json")
        ),
        "task_metadata": {
            "reward_basis": reward_basis,
            "nl_assertions_required": (
                RewardType.NL_ASSERTION in (criteria.reward_basis if criteria is not None else [])
                and bool(nl_assertions)
            ),
            "nl_assertion_count": len(nl_assertions),
        },
        "runtime_info": _source_runtime_info(),
    }
    return description


def _native_user_prompt(request: Mapping[str, object]) -> list[dict[str, object]]:
    from tau2.data_model.message import AssistantMessage  # noqa: PLC0415
    from tau2.user.user_simulator_base import UserState  # noqa: PLC0415

    state = UserState.model_validate(request["user_state"])
    incoming_message = AssistantMessage(
        role="assistant", content=str(request["incoming_actor_text"])
    )
    _validate_customer_facing_actor_message(incoming_message)
    prompt_state = UserState.model_validate(state.model_dump(mode="json"))
    if incoming_message.has_content() or incoming_message.is_tool_call():
        prompt_state.messages.append(incoming_message)
    return [
        message.model_dump(mode="json")
        for message in prompt_state.system_messages + prompt_state.flip_roles()
    ]


def _native_user_turn(request: Mapping[str, object]) -> dict[str, object]:
    import tau2.user.user_simulator as user_simulator_module  # noqa: PLC0415
    from tau2.data_model.message import AssistantMessage  # noqa: PLC0415
    from tau2.user.user_simulator import UserSimulator  # noqa: PLC0415
    from tau2.user.user_simulator_base import UserState  # noqa: PLC0415

    task = _find_task(str(request["task_id"]))
    user = _make_user(task, cast("Mapping[str, object] | None", request.get("persona_config")))
    state = UserState.model_validate(request["user_state"])
    incoming_message = AssistantMessage(
        role="assistant", content=str(request["incoming_actor_text"])
    )
    sampled_response = AssistantMessage(role="assistant", content=str(request["sampled_response"]))
    if request["finish_reason"] != "stop":
        raise ValueError("Native simulator completion must end with finish_reason='stop'")
    _validate_customer_facing_actor_message(incoming_message)
    _validate_simulator_sample(sampled_response)
    captured_prompt_messages: list[list[dict[str, object]]] = []
    previous_generate = user_simulator_module.generate

    def return_sampled_response(**kwargs: object) -> AssistantMessage:
        if kwargs["call_name"] != "user_simulator_response" or kwargs["tools"] is not None:
            raise ValueError("Native retail user transport must be text-only")
        messages = cast("Sequence[Any]", kwargs["messages"])
        captured_prompt_messages.append([message.model_dump(mode="json") for message in messages])
        return sampled_response

    user_simulator_module.generate = return_sampled_response
    try:
        user_message, next_state = user.generate_next_message(incoming_message, state)
    finally:
        user_simulator_module.generate = previous_generate
    if len(captured_prompt_messages) != 1:
        raise RuntimeError(
            "Native UserSimulator did not make exactly one injected generation request"
        )
    return {
        "user_state": next_state.model_dump(mode="json"),
        "user_message": user_message.model_dump(mode="json"),
        "is_stop": UserSimulator.is_stop(user_message),
        "prompt_messages": captured_prompt_messages[0],
    }


def _native_tool_call(request: Mapping[str, object]) -> dict[str, object]:
    from tau2.data_model.message import ToolCall  # noqa: PLC0415
    from tau2.domains.retail.data_model import RetailDB  # noqa: PLC0415
    from tau2.domains.retail.environment import get_environment  # noqa: PLC0415

    retail_db = RetailDB.model_validate(request["db_state"])
    environment = get_environment(db=retail_db, solo_mode=False)
    call_data = cast("Mapping[str, object]", request["tool_call"])
    native_call = ToolCall(
        id=str(call_data["id"]),
        name=str(call_data["name"]),
        arguments=cast("dict[str, object]", call_data["arguments"]),
        requestor="assistant",
    )
    tool_message = environment.get_response(native_call)
    return {
        "tool_message": tool_message.model_dump(mode="json"),
        "db_state": environment.tools.db.model_dump(mode="json"),
    }


def _validate_customer_facing_actor_message(message: Any) -> None:  # noqa: ANN401
    if message.is_audio or message.is_tool_call() or message.content is None:
        raise ValueError("Retail UserSimulator only accepts ordinary customer-facing actor text")


def _validate_simulator_sample(message: Any) -> None:  # noqa: ANN401
    if (
        message.role != "assistant"
        or message.is_audio
        or message.is_tool_call()
        or message.content is None
    ):
        raise ValueError("Retail UserSimulator sample must be text-only AssistantMessage content")


def _evaluate_native(request: Mapping[str, object], source_revision: str) -> dict[str, object]:
    from pydantic import TypeAdapter  # noqa: PLC0415
    from tau2.data_model.message import Message, ToolCall, ToolMessage  # noqa: PLC0415
    from tau2.data_model.simulation import SimulationRun, TerminationReason  # noqa: PLC0415
    from tau2.data_model.tasks import RewardType  # noqa: PLC0415
    from tau2.domains.retail.environment import get_environment  # noqa: PLC0415
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation  # noqa: PLC0415

    task = _find_task(str(request["task_id"]))
    _validate_reference_actions(task, get_environment, ToolCall)
    raw_termination_reason = str(request["termination_reason"])
    termination_reasons = {
        "hand_back": TerminationReason.AGENT_STOP,
        TerminationReason.AGENT_STOP.value: TerminationReason.AGENT_STOP,
        TerminationReason.USER_STOP.value: TerminationReason.USER_STOP,
    }
    native_termination_reason = termination_reasons.get(raw_termination_reason)
    if native_termination_reason is None:
        return {
            "status": "unmeasured",
            "termination_reason": raw_termination_reason,
            "reward_info": None,
            "nl_assertions_status": "pending",
            "source_revision": source_revision,
        }

    message_adapter = TypeAdapter(Message)
    raw_trajectory = request["trajectory"]
    if not isinstance(raw_trajectory, list):
        raise TypeError("Native replay trajectory must be a list")
    native_trajectory = [message_adapter.validate_python(message) for message in raw_trajectory]
    wrapper_tool_names = set(cast("list[str]", request["wrapper_tool_names"]))
    native_trajectory = _strip_wrapper_messages(native_trajectory, wrapper_tool_names, ToolMessage)
    criteria = task.evaluation_criteria
    reward_basis = [] if criteria is None else criteria.reward_basis
    expected_assertions = [] if criteria is None else (criteria.nl_assertions or [])
    nl_required = RewardType.NL_ASSERTION in reward_basis and bool(expected_assertions)
    judge_response = cast("str | None", request.get("judge_response"))

    simulation = SimulationRun(
        id="retail-native-adapter",
        task_id=task.id,
        start_time="1970-01-01T00:00:00Z",
        end_time="1970-01-01T00:00:00Z",
        duration=0.0,
        termination_reason=native_termination_reason,
        messages=native_trajectory,
    )
    diagnostics = _native_evaluation_diagnostics(
        simulation, task, evaluate_simulation, EvaluationType
    )
    if not nl_required:
        if judge_response is not None:
            raise ValueError("Judge response supplied for a task without required NL assertions")
        reward_info = evaluate_simulation(
            simulation,
            task,
            EvaluationType.ALL,
            solo_mode=False,
            domain="retail",
        )
        return {
            **diagnostics,
            "status": "measured",
            "termination_reason": raw_termination_reason,
            "native_termination_reason": native_termination_reason.value,
            "reward_info": reward_info.model_dump(mode="json"),
            "nl_assertions_status": "not_required",
            "source_revision": source_revision,
        }
    context = _NLAssertionEvaluation(
        source_revision=source_revision,
        raw_termination_reason=raw_termination_reason,
        native_termination_reason=native_termination_reason,
        simulation=simulation,
        task=task,
        expected_assertions=expected_assertions,
        judge_response=judge_response,
    )
    return _evaluate_nl_assertions(context)


def _validate_reference_actions(
    task: Any,  # noqa: ANN401
    get_environment: Any,  # noqa: ANN401
    tool_call_type: Any,  # noqa: ANN401
) -> None:
    task_initial_state = task.initial_state
    environment = get_environment(solo_mode=False)
    environment.set_state(
        initialization_data=(
            None if task_initial_state is None else task_initial_state.initialization_data
        ),
        initialization_actions=(
            None if task_initial_state is None else task_initial_state.initialization_actions
        ),
        message_history=(
            []
            if task_initial_state is None or task_initial_state.message_history is None
            else task_initial_state.message_history
        ),
    )
    criteria = task.evaluation_criteria
    reference_actions = [] if criteria is None else criteria.actions or []
    for action in reference_actions:
        result = environment.get_response(
            tool_call_type(
                id=action.action_id,
                name=action.name,
                arguments=action.arguments,
                requestor=action.requestor,
            )
        )
        if result.error:
            raise ValueError(
                f"Native reference action failed during prevalidation: {action.action_id}"
            )


def _native_evaluation_diagnostics(
    simulation: Any,  # noqa: ANN401
    task: Any,  # noqa: ANN401
    evaluate_simulation: Any,  # noqa: ANN401
    evaluation_type: Any,  # noqa: ANN401
) -> dict[str, object]:
    return {
        "db_reward_info": evaluate_simulation(
            simulation,
            task,
            evaluation_type.ENV,
            solo_mode=False,
            domain="retail",
        ).model_dump(mode="json"),
        "communicate_reward_info": evaluate_simulation(
            simulation,
            task,
            evaluation_type.COMMUNICATE,
            solo_mode=False,
            domain="retail",
        ).model_dump(mode="json"),
        "action_reward_info": evaluate_simulation(
            simulation,
            task,
            evaluation_type.ACTION,
            solo_mode=False,
            domain="retail",
        ).model_dump(mode="json"),
    }


def _evaluate_nl_assertions(context: _NLAssertionEvaluation) -> dict[str, object]:
    from tau2.data_model.message import AssistantMessage  # noqa: PLC0415
    from tau2.evaluator import evaluator_nl_assertions as nl_assertions_module  # noqa: PLC0415
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation  # noqa: PLC0415
    from tau2.evaluator.evaluator_nl_assertions import NLAssertionsEvaluator  # noqa: PLC0415

    nl_assertions_module.DEFAULT_LLM_NL_ASSERTIONS = _NO_MODEL_CALL
    nl_assertions_module.DEFAULT_LLM_NL_ASSERTIONS_ARGS = {}
    evaluator = NLAssertionsEvaluator(context.task, solo_mode=False)
    expected_messages = _capture_nl_judge_request(
        context.simulation.messages, context.expected_assertions, evaluator
    )
    base = {
        "termination_reason": context.raw_termination_reason,
        "native_termination_reason": context.native_termination_reason.value,
        "nl_judge_transport": _NO_MODEL_CALL,
        "nl_judge_request": {
            "messages": expected_messages,
            "expected_assertions": context.expected_assertions,
            "expected_count": len(context.expected_assertions),
        },
        "source_revision": context.source_revision,
    }
    diagnostics = _native_evaluation_diagnostics(
        context.simulation, context.task, evaluate_simulation, EvaluationType
    )
    if context.judge_response is None:
        return {
            **base,
            **diagnostics,
            "status": "pending",
            "nl_assertions_status": "pending",
        }

    semantic_response, response_error = _parse_judge_completion(
        context.judge_response, context.expected_assertions
    )
    if response_error is not None:
        return {
            **base,
            **diagnostics,
            "status": "unmeasured",
            "nl_assertions_status": "invalid_judge_response",
            "nl_assertions_error": response_error,
            "nl_judge_response": context.judge_response,
        }

    previous_generate = nl_assertions_module.generate

    def return_judge_response(**kwargs: object) -> AssistantMessage:
        if kwargs["call_name"] != "nl_assertions_eval":
            raise ValueError("Unexpected native evaluator generation request")
        messages = cast("Sequence[Any]", kwargs["messages"])
        actual_messages = [message.model_dump(mode="json") for message in messages]
        if actual_messages != expected_messages:
            raise ValueError("Native evaluator prompt changed between capture and evaluation")
        return AssistantMessage(role="assistant", content=semantic_response)

    nl_assertions_module.generate = return_judge_response
    try:
        reward_info = evaluate_simulation(
            context.simulation,
            context.task,
            EvaluationType.ALL,
            solo_mode=False,
            domain="retail",
        )
    finally:
        nl_assertions_module.generate = previous_generate

    actual_assertions = reward_info.nl_assertions or []
    if Counter(check.nl_assertion for check in actual_assertions) != Counter(
        context.expected_assertions
    ):
        raise RuntimeError("Native evaluator did not return each requested assertion exactly once")
    return {
        **base,
        "status": "measured",
        "reward_info": reward_info.model_dump(mode="json"),
        "nl_assertions_status": "measured",
        "nl_judge_response": context.judge_response,
    }


def _strip_wrapper_messages(
    trajectory: Sequence[Any],
    wrapper_tool_names: set[str],
    tool_message_type: Any,  # noqa: ANN401
) -> list[Any]:
    native_trajectory: list[Any] = []
    index = 0
    while index < len(trajectory):
        message = trajectory[index]
        wrapper_calls = (
            [call for call in (message.tool_calls or []) if call.name in wrapper_tool_names]
            if hasattr(message, "tool_calls")
            else []
        )
        if not wrapper_calls:
            native_trajectory.append(message)
            index += 1
            continue
        if len(wrapper_calls) != 1 or len(message.tool_calls) != 1:
            raise ValueError("Native replay cannot contain mixed or multiple wrapper tool calls")
        if index + 1 >= len(trajectory) or not isinstance(trajectory[index + 1], tool_message_type):
            raise ValueError("Wrapper tool call is missing its native ToolMessage")
        result = trajectory[index + 1]
        if result.id != wrapper_calls[0].id or result.requestor != "assistant":
            raise ValueError("Wrapper tool result does not match its native ToolCall")
        index += 2
    return native_trajectory


def _capture_nl_judge_request(
    trajectory: Sequence[Any],
    expected_assertions: list[str],
    evaluator: Any,  # noqa: ANN401
) -> list[dict[str, object]]:
    import tau2.evaluator.evaluator_nl_assertions as nl_assertions_module  # noqa: PLC0415

    captured_messages: list[list[dict[str, object]]] = []
    previous_generate = nl_assertions_module.generate

    def capture_request(**kwargs: object) -> None:
        if kwargs["call_name"] != "nl_assertions_eval":
            raise ValueError("Unexpected native evaluator generation request")
        messages = cast("Sequence[Any]", kwargs["messages"])
        captured_messages.append([message.model_dump(mode="json") for message in messages])
        raise _CaptureNLJudgeRequestError

    nl_assertions_module.generate = capture_request
    try:
        with suppress(_CaptureNLJudgeRequestError):
            evaluator.evaluate_nl_assertions(list(trajectory), expected_assertions)
    finally:
        nl_assertions_module.generate = previous_generate
    if len(captured_messages) != 1:
        raise RuntimeError("Native NL evaluator did not make exactly one judge request")
    return captured_messages[0]


def _parse_judge_completion(
    raw_completion: str, expected_assertions: list[str]
) -> tuple[str, str | None]:
    semantic_response = raw_completion
    if "</think>" in raw_completion:
        _, _, semantic_response = raw_completion.partition("</think>")
        semantic_response = semantic_response.lstrip("\r\n")
    elif "<think>" in raw_completion:
        return "", "Qwen thinking block ended without </think>."
    try:
        parsed = json.loads(semantic_response)
        _validate_judge_results(parsed, expected_assertions)
    except json.JSONDecodeError as error:
        return "", f"Judge response is not valid JSON: {error.msg}."
    except (TypeError, ValueError) as error:
        return "", str(error)
    return semantic_response, None


def _validate_judge_results(parsed: object, expected_assertions: list[str]) -> None:
    if not isinstance(parsed, dict) or not isinstance(parsed.get("results"), list):
        raise TypeError("Judge response must contain a results list.")
    results = parsed["results"]
    if len(results) != len(expected_assertions):
        raise ValueError("Judge response result count does not match the expected assertion count.")
    returned_assertions: list[str] = []
    for result in results:
        if not isinstance(result, dict):
            raise TypeError("Judge response results must be objects.")
        expected = result.get("expectedOutcome")
        met = result.get("metExpectation")
        reasoning = result.get("reasoning")
        if not isinstance(expected, str) or type(met) is not bool or not isinstance(reasoning, str):
            raise ValueError(
                "Each judge result needs string expectedOutcome/reasoning and a Boolean metExpectation."
            )
        returned_assertions.append(expected)
    if Counter(returned_assertions) != Counter(expected_assertions):
        raise ValueError("Judge response must return every expected assertion exactly once.")


def _worker_main() -> None:
    request = json.loads(sys.stdin.read())
    if not isinstance(request, dict):
        raise TypeError("Native tau-retail request must be a JSON object")
    response = _dispatch_worker_request(cast("Mapping[str, object]", request))
    sys.stdout.write(json.dumps(response, separators=(",", ":")))


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit("retail_native.py is an internal JSON worker")
    _worker_main()
