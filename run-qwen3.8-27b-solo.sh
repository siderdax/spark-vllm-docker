#!/bin/bash
# Serve nvidia/Qwen3.8-27B-NVFP4 + DFlash2 speculative decoding on a single
# DGX Spark (solo, TP=1, no Ray) -- the same recipe as run-qwen3.8-27b-dflash2.sh
# but pinned to this node only, leaving the other node free.
# Once up, the OpenAI-compatible API is at http://localhost:8000/v1
cd "$(dirname "${BASH_SOURCE[0]}")"
exec ./run-recipe.sh custom_recipes/qwen3.8-27b-nvfp4-dflash2-nvidia --solo --tensor-parallel 1 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 "$@"
