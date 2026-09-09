#!/bin/bash
# Serve nvidia/Qwen3.8-Flash-Next-NVFP4 (~123.6GB) with MTP-4 on the 2-node
# cluster (raven+quaker, TP=2), using this repo's own vllm-node image rebuilt
# from vLLM main 2026-09-09 (picks up upstream PR #55513, which fixes the
# FP8_BLOCK_SCALES MTP-expert loading crash hit with the frozen
# vllm/vllm-openai:qwen38-flash-next tag). Active default as of 2026-09-09 --
# VERIFIED: 9/9 context-bench.py, decode 32.7-37.6 t/s (comparable to
# RadixArk's recorded 30-38 t/s range). See recipes/qwen3.8-flash-next-nvfp4.yaml
# for the full history; reserved_runs/run-qwen3.8-flash-next-radixark.sh is
# the fallback if this self-built image ever needs troubleshooting.
# Once up, the OpenAI-compatible API is at http://localhost:8000/v1
cd "$(dirname "${BASH_SOURCE[0]}")"
# quaker (worker node) has no internet, so HF Hub lookups die on DNS failure.
# These must be passed as container-level -e to reach the worker processes (recipe env: alone is not enough).
exec ./run-recipe.sh qwen3.8-flash-next-nvfp4 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 "$@"
