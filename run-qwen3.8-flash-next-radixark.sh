#!/bin/bash
# Serve Qwen3.8-Flash-Next-NVFP4 (RadixArk quant, 135GB) on the 2-node cluster (raven+quaker, TP=2).
# Second attempt -- see recipes/qwen3.8-flash-next-nvfp4-radixark.yaml for the full
# writeup of what OOM'd/froze the node last time (Inferact quant, 182.78GB) and what
# changed here (smaller checkpoint, drop-caches, --enforce-eager, no forced ray backend).
# Once up, the OpenAI-compatible API is at http://localhost:8000/v1
cd "$(dirname "${BASH_SOURCE[0]}")"
# quaker (worker node) has no internet, so HF Hub lookups die on DNS failure.
# These must be passed as container-level -e to reach the worker processes (recipe env: alone is not enough).
exec ./run-recipe.sh qwen3.8-flash-next-nvfp4-radixark --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 "$@"
