#!/bin/bash
# Serve Qwen3.8-Flash-Next-NVFP4 (RadixArk quant, 135GB) on the 2-node cluster (raven+quaker, TP=2).
# RESERVED as of 2026-09-09 -- superseded by run-qwen3.8-flash-next.sh (nvidia's
# checkpoint on this repo's own rebuilt vllm-node, which now also has working MTP-4
# after vLLM PR #55513). Kept here since this still-working, officially-tagged image
# is a safer fallback than our self-built vllm-node if that image ever needs a
# from-scratch rebuild. See recipes/qwen3.8-flash-next-nvfp4-radixark.yaml for the
# full writeup of what OOM'd/froze the node before this (Inferact quant, 182.78GB)
# and what changed here (smaller checkpoint, drop-caches, --enforce-eager, no forced
# ray backend).
# Once up, the OpenAI-compatible API is at http://localhost:8000/v1
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# quaker (worker node) has no internet, so HF Hub lookups die on DNS failure.
# These must be passed as container-level -e to reach the worker processes (recipe env: alone is not enough).
exec ./run-recipe.sh qwen3.8-flash-next-nvfp4-radixark --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 "$@"
