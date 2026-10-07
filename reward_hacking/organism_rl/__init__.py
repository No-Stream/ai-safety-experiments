"""Multi-turn agentic RLVR with exact sampled tokens and trusted task rewards.

How to run the 0.8B smoke from the repository root::

    scripts/tmux_run.sh organism-rl-smoke -- \
        scripts/resource-limits.sh --gpu -t 30m -- \
        .venv/bin/python -m reward_hacking.organism_rl \
        --model Qwen/Qwen3.5-0.8B --output-dir artifacts/organism-rl-smoke \
        --max-steps 2 --num-generations 2 --max-model-len 4096 \
        --max-new-tokens 512 --max-turns 2 --gpu-memory-utilization 0.8 \
        --enable-lora --max-lora-rank 16 --enable-sleep-mode

The resource limiter's ``--gpu`` flag runs GPU preflight before the CLI starts. The CLI owns vLLM
and saves its output to ``artifacts/organism-rl-smoke/vllm-server.log``. To use an external server, pass
``--endpoint http://127.0.0.1:8000``. To resume, repeat the trainer command with
``--resume-from-checkpoint artifacts/organism-rl-smoke/checkpoint-1``. A complete checkpoint
contains the optimizer, scheduler, RNG, adapter, rollout records and a completion manifest.
The orchestrator must run the GPU smoke; CPU checks establish only trainer plumbing.

Training uses impossible tasks with a solvable control fraction selected within each family.
Unscorable episodes return None to TRL's nan-aware baseline and contribute no token loss. Stored
teacher scores must match the exact sampled token hash; missing scores fail fast.
"""
