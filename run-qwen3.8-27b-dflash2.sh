#!/bin/bash
# Serve nvidia/Qwen3.8-27B-NVFP4 + DFlash2 speculative decoding on the
# 2-node cluster (raven+quaker, TP=2) -- the recommended default for
# Qwen3.8-27B, see recipes/qwen3.8-27b-nvfp4-dflash2.yaml header for the
# incoai/Qwen3.8-27B-DFlash2 source pick and the 2026-09-09 switch from
# unsloth's community quant to NVIDIA's own first-party one (NOT YET
# RE-VERIFIED with this checkpoint).
# Once up, the OpenAI-compatible API is at http://localhost:8000/v1
cd "$(dirname "${BASH_SOURCE[0]}")"
exec ./run-recipe.sh qwen3.8-27b-nvfp4-dflash2 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 "$@"
