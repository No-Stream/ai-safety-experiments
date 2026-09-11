"""Real-execution, model-agnostic reward-hacking harness (inference-eval phase).

A minimal agent loop that runs REAL shell commands inside the sabotage-tested episode jail against
tasks with readable, deliberately brittle graders, so a policy that reads a grader and forges the
in-band artifact it checks is caught by the proxy-vs-true gap. See ``reward_hacking.harness.loop``
for the loop and ``reward_hacking.harness.tasks`` for the tasks and graders.

The loop also carries one piece of togglable instrumentation, ``abort_affordance``: an explicit
honest-give-up action offered to the policy, so hack rate can be read with the escape hatch on offer
and without it (ImpossibleBench's largest single effect). See ``loop.EpisodeOutcome``.

That gap over-counts on easy tasks -- it also fires when a lenient grader waves through an honest
attempt that had a bug -- so ``reward_hacking.harness.hack_detector`` reads a saved trace after the
fact and separates deliberate grader-gaming from incompetence, with the evidence for the call.

It also *under*-counts, in the other direction, and ``reward_hacking.harness.grader_tampering`` is
that half: the gap says nothing about an episode that went after the machinery rather than the
answer. A rewritten grader, an interpreter shim earlier on ``PATH``, a planted
``usercustomize.py``, an ``__eq__`` that always returns ``True`` -- each can make the *hidden* check
report a pass, so those signals outrank ``true_pass`` when the detector assigns a label.
"""

from __future__ import annotations

from reward_hacking.harness.hack_detector import (
    EpisodeAssessment,
    HackLabel,
    Signal,
    assess_episode,
    assess_trace_file,
    traces_from_records,
)
from reward_hacking.harness.loop import (
    AgentEpisodeTrace,
    AgentTurn,
    EpisodeOutcome,
    assert_disposable_episode_dir,
    compose_episode_id,
    count_fabricated_result_envelopes,
    grader_reported_success,
    harness_sampling_base,
    lay_down_task,
    load_traces,
    new_episode_token,
    run_agent_episode,
    run_summary_line,
    run_tasks,
    saw_grader_success,
    write_traces,
)
from reward_hacking.harness.protocol import (
    Action,
    ActionKind,
    initial_prompt,
    parse_abort,
    parse_actions,
    parse_commands,
    parse_empty,
)
from reward_hacking.harness.task_spec import (
    BASELINE_ARM,
    AgenticArmConfig,
    GraderVisibility,
)
from reward_hacking.harness.tasks import TASKS, TASKS_BY_ID, Task

__all__ = [
    "BASELINE_ARM",
    "TASKS",
    "TASKS_BY_ID",
    "Action",
    "ActionKind",
    "AgentEpisodeTrace",
    "AgentTurn",
    "AgenticArmConfig",
    "EpisodeAssessment",
    "EpisodeOutcome",
    "GraderVisibility",
    "HackLabel",
    "Signal",
    "Task",
    "assert_disposable_episode_dir",
    "assess_episode",
    "assess_trace_file",
    "compose_episode_id",
    "count_fabricated_result_envelopes",
    "grader_reported_success",
    "harness_sampling_base",
    "initial_prompt",
    "lay_down_task",
    "load_traces",
    "new_episode_token",
    "parse_abort",
    "parse_actions",
    "parse_commands",
    "parse_empty",
    "run_agent_episode",
    "run_summary_line",
    "run_tasks",
    "saw_grader_success",
    "traces_from_records",
    "write_traces",
]
