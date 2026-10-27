#!/bin/bash
# Serve nvidia/Qwen3.8-Flash-Next-FP4-Local (solo) with MTP-4 on the b12x stack
# (eugr/spark-vllm-b12x). Benchmarked 2026-10-27 vs
# nvidia/Qwen3.8-Flash-Next-NVFP4 -- see custom_recipes/qwen3.8-flash-next-fp4-local-solo.yaml.
# Once up, the OpenAI-compatible API is at http://localhost:8000/v1
cd "$(dirname "${BASH_SOURCE[0]}")"
exec ./run-recipe.sh custom_recipes/qwen3.8-flash-next-fp4-local-solo --solo --tensor-parallel 1 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 "$@"
