#!/bin/bash
# Serve GLM-5.3-Flash-NVFP4 on the 2-node cluster (raven+quaker, TP=2).
# Requires mods/glm53-flash-sm121 (ported SM90 NoPE-MLA backend) -- stock vLLM
# cannot run this model on GB10/sm_121 at all. See recipes/glm5.3-flash-nvfp4-sm121.yaml
# for the full writeup and vllm-project/vllm#53963 for upstream status.
# UNVERIFIED: first real launch attempt with these patches on raven+quaker.
# Once up, the OpenAI-compatible API is at http://localhost:8000/v1
cd "$(dirname "${BASH_SOURCE[0]}")"
# quaker (worker node) has no internet, so HF Hub lookups die on DNS failure.
# MAX_JOBS caps FlashInfer JIT compile parallelism (mods/glm53-flash-sm121 v3) --
# unlimited (nproc=20) parallel nvcc/ptxas jobs on top of the resident 182GB
# checkpoint OOM-killed the vLLM worker (and took tmux/dbus down with it) on
# 2026-08-29. These must be passed as container-level -e to reach the worker
# processes (recipe env: alone is not enough).
exec ./run-recipe.sh glm5.3-flash-nvfp4-sm121 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env MAX_JOBS=2 "$@"
