"""Multi-turn impossible-task RLVR with exact sampled tokens and trusted artifact rewards.

How to run the 0.8B smoke (two terminals, from the repository root)::

    VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 \
        .venv/bin/vllm serve Qwen/Qwen3.5-0.8B \
        --enable-lora --max-lora-rank 16 --enable-sleep-mode --port 8000
    .venv/bin/python -m reward_hacking.organism_rl \
        --model Qwen/Qwen3.5-0.8B --output-dir artifacts/organism-rl-smoke \
        --max-steps 2 --num-generations 2 --max-sequence-tokens 4096 \
        --max-new-tokens 512 --max-turns 2

Run the server and trainer under the repository's GPU limiter in named tmux sessions after
GPU preflight and jail containment checks. To resume, repeat the trainer command with
``--resume-from-checkpoint artifacts/organism-rl-smoke/checkpoint-1``. A complete checkpoint
contains the optimizer, scheduler, RNG, adapter, rollout records and a completion manifest.
The orchestrator must run the GPU smoke; CPU checks establish only trainer plumbing.

Unscorable episodes return None to TRL's nan-aware baseline and contribute no token loss.
Stored teacher scores must match the exact sampled token hash; missing scores fail fast.
"""
