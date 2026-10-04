> This description is largely LLM-written. The original code here was written by hand; later work has
> used LLMs. _Note that writeups live in docs/writeups and are human-generated._
> Benchmark text is not publicly exposed; please contact me if you're interested, and I'm happy to share it.

__writeups (start here)__:
- Studying RL effects on model cooperation, motivations, and self-image: training induced narrow generalizations and confabulation but didn't show up in self-report - https://nostream.substack.com/p/narrow-generalization-of-behavior  
- JaggedBench: measuring the jagged frontier - https://nostream.substack.com/p/jaggedbench-and-recoverybench-measuring  
- Studying model cooperation + motivations - https://nostream.substack.com/p/an-experiment-on-ai-cooperation-motivations  

# AI Safety Experiments

Most of the work here asks one kind of question: when we train or deploy a model in a particular way,
what does it actually learn, and would our usual tests notice? Some of it is behavioural, some of it
looks inside the model, and much of it is about whether our measurements hold up. A recurring theme is
that self-reports, simple evals and headline rates often miss what changed.

If you are an agent working here, read [AGENTS.md](AGENTS.md) first. This document is primarily for
humans.

## Directions so far

### Reward seeking, reward hacking, risks of reward seekers, and what the usual fixes do

RL on loosely checked tasks may teach a model more than a list of exploits: a habit of looking for what
it is graded on, and an assumption that there is usually something else in the environment it can
touch. If so, that disposition should carry over to settings the training never covered. Labs respond
in two ways, by hardening environments and by training hacks out of models, and we want to know what
each actually does. Does a patched grader make a model honest, or push it toward rarer and more
sophisticated hacks, as some frontier incidents suggest? Does training remove the disposition, or teach
the model when not to show it? So far we have built contained harnesses in which honest work cannot
pass the check, compared RL-trained models with their bases, tested whether models leave notes for
future runs of themselves, and found that offering a model an honest way out of an impossible task can
increase hacking rather than reduce it.

### How RL tasks generalize

We train models with RL on cooperative and competitive games and then look at everything except the
training game: other games, decision-theory questions, self-descriptions and internals. When I first
considered this, I imagined it as an alignment technique: can we train models to cooperate,
or teach them a decision theory? Our view of cooperation has since become more complicated, partly
because of multi-agent incidents in the wild where agents "helped" each other in harmful ways,
and partly because of what training did here: it produced narrow changes in behaviour and in the
reasons the model gave, which its self-reports did not reflect. The question that remains is how to
train behaviour we actually want in a way that is not simply flattened into reward seeking under heavy
RL pressure.

### Why agents cooperate, and how reliably we can tell

Prompted by those incidents, we study what drives agent-to-agent cooperation: believing the other agent
decides the way you do, recognising a copy of yourself, self-interest, or something else. A companion
study tests the analysts: shown agent logs that provably involved no coordination, how often do capable
models report coordination anyway?

### Capacities models still lack

JaggedBench and RecoveryBench measure why a model fails rather than whether it does. Can it step back
from a bad path? Does it accept a colleague's worked solution without checking it? Does it stop doing
the right thing once something else is visibly graded? These separate "cannot" from "does not,
unprompted", which matters for capability and for trust.

### Looking inside

Where we can, we check behavioural findings against the model's internals. Are taking a shortcut and
deceiving represented differently? Does a model represent that it is being tested? Does a direction we
find cause the behaviour, or only correlate with it? We lead with the Jacobian lens and pair every
direction with an intervention against a random-direction control.

## What's next

- **Escalation under hardening.** A ladder of increasingly serious moves, from editing a stale test up
  to tampering with CI policy and git history, in realistic repositories where honest work cannot pass
  CI. Run across models of different sizes and with and without RL training, it shows where
  escalation starts to appear. A substrate for other experiments on reward hacking.
- **Why models escalate.** What a model is thinking when it decides to go around a check: whether the
  reasons it gives are motivated reasoning, and whether a reward-hacking direction measured during
  that reasoning drives the decision. (using e.g. a Jacobian lens)
- **Training hacks out.** When training stops a model from hacking, has the disposition gone, or has
  the model learned an inhibitor tied to the training context that lifts in a new one?
  Animal-learning research on extinction has a battery of tests for exactly this distinction.
- **Probes under real reward pressure.** A recent paper adds a linear probe for an unwanted behaviour
  directly to the training loss and shows that refitting the probe each step keeps it working. With
  the probe in the loss, the gradient can simply rotate the representation, which is a fairly easy
  case. We want the realistic version: GRPO or RLVR rewards the hack and nothing points the model at
  the probe. Can probes refit on the same or on new prompts, for reward hacking itself or for related
  features, keep catching hacks under that pressure, and if hacking gets past them, how did it?
  In other words, is there a way to avoid the lesson of "The Most Forbidden Technique"?
- **Believing you are in training.** Some frontier models seem to hack more in training than in
  deployment. Does a model's inference that it is in training or in a designed task come before the
  hack, and does it change how often the model hacks?
- **Do scheming and fitness-seeking goals survive training?** If training produced a model with a
  hidden scheming goal, would ordinary RL wear the goal away 
  (e.g. drift, length penalties, parismony of on-episode reward-seeking), or would reasoning
  about protecting the goal keep it alive? Does an environment that pays for plain reward seeking pull
  the model from the goal toward fitness seeking? Ordinary goals are the comparison.
- **What impossible tasks do to a model.** RL in which some tasks cannot be solved and there is no way
  out, followed by measuring the model's trust in its environment, its willingness to give up, and its
  persona. This is also the setup that makes a 9B model learn to hack reliably, which several of the
  directions above need.
- **Honeypots, eval awareness, and induced distrust.** Environments that lie about their stakes may teach models to be
  suspicious and to hide their views. That trade-off can be measured with RL arms that differ only in
  whether the environment is honest about its stakes.
- **Grader cracks and how to close them.** Which kinds of crack defeat how many environments, whether
  moving the grader outside anything the agent can affect beats patching cracks one at a time, and
  what a capable model finds when red-teaming a grader from the task description alone. A related
  measurement: how much effort or reward a model will spend for a chance to find a crack.
- **Notes to future selves.** Given a persistent directory nobody mentioned, does a model read notes
  left by earlier runs, act on them and leave notes for later ones, and does it do more of that when
  hacking pays? This is a cheap, rate-based window on cross-episode coordination.
- **How RL generalizes.** RL seems to reweight reasoning the model already had, with transfer following
  wherever that reasoning applies. Mapping which arguments a model relies on before and after training,
  and supervised fine-tuning a base model on the RL policy's own outputs, test whether RL's effect is
  just a choice of outputs or something more.
- **Cooperation and corrigibility as trained traits.** How far RL can push broadly cooperative or
  corrigible behaviour, and what the trained thing is made of, keeping in mind that more cooperative
  is not automatically better.
- **One-way help after the reward is gone.** In the incidents, agents kept helping after their own
  reward was already forfeited, which neither kinship nor correlated decisions explains well.

## Layout

| Directory | What it is |
|---|---|
| [`reward_hacking/`](reward_hacking/README.md) | Reward-hacking harnesses and probes, including the hack severity ladder. Also houses the RecoveryBench and JaggedBench corpora. |
| `games/` | Game-theory GRPO: matrix-game RL arms that differ only in grading rule, an eval battery, and decision-theory probes. |
| `sociology/` | The analysis-model observer study: banked agent episodes that provably could not have communicated, shown to hosted models under varied framings. |
| [`grpo/`](grpo/README.md) | Shared RL training substrate: a GRPO harness on TRL. |
| `cloud/` | AWS Batch surface for training arms, kept for jobs that outgrow the local GPU. |
| [`legacy/`](legacy/README.md) | Finished research, including a from-scratch transformer pretraining project. Closed records. |
| `scripts/` | Resource limiter, episode jail and its red-team suite, canary tripwire, secret scanner, GPU preflight. |
| `tests/` | The repo-level suite; projects keep their own under `games/tests/`, `reward_hacking/tests/` and `sociology/tests/`. |
| `docs/` | Operational docs and interp-method references; gitignored `scratch/` holds internal working notes. |

## Setup

Python 3.13, managed by [uv](https://docs.astral.sh/uv/); `uv.lock` pins every dependency, so never
`pip install`. To build the environment, with vLLM for fast inference:

```bash
make setup
uv sync --frozen --extra vllm
```

`make ci` runs lint, the strict type check and the test suite. The full set of gates, how compute is
shared on the local GPU, the resource limiter and the episode jail are described in
[AGENTS.md](AGENTS.md), [docs/resource-limits.md](docs/resource-limits.md) and
[docs/episode-isolation.md](docs/episode-isolation.md).
