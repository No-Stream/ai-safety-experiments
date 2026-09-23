> This description is largely LLM-written. The original code here was written by hand; later work has
> used LLMs. _Note that writeups live in docs/writeups and are human-generated._
> Benchmark text is not publicly exposed; please contact me if you're interested, and I'm happy to share it.

__writeups__:
- JaggedBench: measuring the jagged frontier - https://nostream.substack.com/p/jaggedbench-and-recoverybench-measuring  
- Studying model cooperation + motivations - https://nostream.substack.com/p/an-experiment-on-ai-cooperation-motivations  
- Studying RL effects on model cooperation, motivations, and self-image - LINK TODO  

# Transformer Experiments

I focus on a few directions:
- HuggingFace incident inspired agent motivations: why do agents cooperate or not cooperate with other agents? early directions suggest game theoretical or self-interested motivations as primary with a variety of other model-specific motivations. (e.g. Claude Sonnet 5 seems to display cooperative or altruistic tendencies in some toy-ish evals.)
- RLVR/GRPO on cooperative and competitive games: how does RLVR affect models' behaviors, self-image, and beliefs, and how much does altruistic or competitive game training, and eventually more complex env training, affect deployment.
- JaggedBench and its first facet, RecoveryBench: investigating gaps in the jagged frontier. One direction is model recovery: models often proceed along a given path and have trouble taking a step back and finding a better route. This benchmark attempts to measure and document that. It also begins to measure agents' trustworthiness and cooperation: do models trust information packets given to them blindly? This has speculative implications regarding memetic and agent swarm behaviors, which is a future line of work. 

If you are an agent working here, read [AGENTS.md](AGENTS.md) first. This document is primarily for humans.

## Layout

| Directory | What it is |
|---|---|
| [`reward_hacking/`](reward_hacking/README.md) | The live project: chasing reward-hacking leads. Also houses the RecoveryBench and JaggedBench corpora. |
| `games/` | The game-theory GRPO project: matrix-game RL arms that differ only in grading rule, an eval battery, a decision-theory probe battery, and the stage runner that sequences GPU stages. |
| `sociology/` | The analysis-model observer study: bundles of banked agent episodes that provably could not have communicated, fed to hosted models under varied framings. |
| [`grpo/`](grpo/README.md) | Shared RL training substrate: a working GRPO harness on TRL 1.10. |
| `cloud/` | The AWS Batch surface for training arms: container, ECR push, job submission. |
| [`legacy/`](legacy/README.md) | Finished research. Closed records; do not extend. |
| `scripts/` | Resource limiter, episode jail and its red-team suite, canary tripwire, secret scanner, GPU preflight. |
| `tests/` | The repo-level suite behind `make test`; each project keeps its own under `games/tests/`, `reward_hacking/tests/` and `sociology/tests/`. |
| [`canary/`](canary/README.md) | The canary tripwire's baseline manifest. Machine-local; only the README is tracked. |
| `artifacts/` | Gitignored run outputs: traces, sweeps, eval records, checkpoints. |
| `docs/` | The operational docs (`episode-isolation.md`, `resource-limits.md`, `games-predictions.md`), the interp-method references in `interp-methods/`, and gitignored `scratch/`, the one home for internal working notes. |

## Setup

To build `.venv` from the pinned `uv.lock`:

```bash
make setup
```

Python 3.13, managed by [uv](https://docs.astral.sh/uv/). `pyproject.toml` pins the stack
(torch 2.13, transformers 5.15, TRL 1.10, PEFT, Liger) and `uv.lock` fixes every transitive
version, so the environment is reproducible. Never `pip install`. vLLM is recommended dep for inference speed:

```bash
uv sync --extra vllm
```

To use the venv as a notebook kernel, point Jupyter at `.venv/bin/python`.

## Everyday commands

```bash
make test                            # full suite, CPU only, sharded over pytest-xdist
make test TEST_WORKERS=4             # fewer workers when the shared box is busy
make test-select ARGS="-k reward"    # subset, serial
make test-changed                    # only the tests the working tree's changes reach; iteration gate, not the commit gate
make format                          # ruff format + autofix
make lint                            # ruff format --check + ruff check
make typecheck                       # basedpyright, strict -- GATING
make typecheck-ty                    # ty (Astral) -- ADVISORY, not gating
make ci                              # lint + typecheck + test
make all                             # format, lint, test
make jail-test                       # episode jail containment plus its negative control
make canary-check                    # checksum tripwire over host code-execution surfaces
make smoke                           # short GRPO run on the real GPU
```

`make test`, `make lint`, `make typecheck` (basedpyright, gating), `make jail-test` and
`make canary-check` are the five gates, and all five should be green before anything is called
done; `make typecheck-ty` (ty) is advisory only. `make ci` bundles the fast trio
(`lint typecheck test`), runs the three concurrently with the type checker at eight threads, and is
the gate before every commit; `GATE_TMUX=1 make ci` (or `make test`) sends the gate through
`scripts/tmux_run.sh` and prints where to poll it. `make test` runs with `RLVR_SMOKE=1`,
which enables one test that downloads a ~2 MB model and exercises the real transformers stack end
to end on CPU, and shards the suite over `TEST_WORKERS` pytest-xdist processes (default 8): about
3.5 minutes on this box against about 15 serial, with the same pass/fail set. Use
`make test TEST_WORKERS=4` when the shared box is busy; `make test-select` stays serial, since
worker start-up costs more than it saves on a narrow selection.

## Running things on the dev box

Compute is local: one NVIDIA RTX 5090 (32 GB) on a Ryzen 5950X under WSL2, shared by whatever
sessions are running. Real runs belong here, not on a rented GPU; long jobs are expected, through
the resource limiter and in a tmux session with a teed log. Rented EC2 and AWS Batch are the
secondary path for a job that outgrows 32 GB.

Run anything expensive through the resource-limit wrapper, which caps CPU, memory, tasks and
wall-clock via cgroup v2 so a runaway job cannot make the box unresponsive. It limits resource
usage only and is **not** an isolation or security boundary:

```bash
scripts/resource-limits.sh --gpu -t 15m -- python train.py
```

`scripts/gpu_preflight.py` refuses to start when another process already holds VRAM on the
single shared GPU. Resource conventions, the measured basis for the thread caps, and which
limits are enforced versus advisory are in [docs/resource-limits.md](docs/resource-limits.md).

For isolation rather than resource capping, running untrusted or scope-violating code in a
filesystem/network-isolated jail, use `scripts/episode_jail.sh`, gated on
`scripts/run_jail_tests.sh`, which proves containment holds *and* that the assertion suite has
teeth. See [docs/episode-isolation.md](docs/episode-isolation.md). Compose them limits
outside, isolation inside, never the reverse:

```bash
scripts/resource-limits.sh -t 15m -- scripts/episode_jail.sh --episode-dir D -- cmd
```

The wrapper runs the job from the directory you invoked it from, and `--chdir DIR` overrides
that.
