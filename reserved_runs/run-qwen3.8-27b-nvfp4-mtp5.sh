#!/bin/bash
# Serve nvidia/Qwen3.8-27B-NVFP4 + MTP-5 on the 2-node cluster (raven+quaker, TP=2).
# Comparison sibling of run-qwen3.8-27b-dflash2.sh (the default) from the
# 4-way NVFP4/FP8 x MTP-5/DFlash2 comparison. Switched 2026-09-09 from
# unsloth's community quant to nvidia's first-party one -- see
# custom_recipes/qwen3.8-27b-nvfp4-mtp5.yaml header. Once up, the OpenAI-compatible
# API is at http://localhost:8000/v1
cd "$(dirname "${BASH_SOURCE[0]}")"
exec ./run-recipe.sh custom_recipes/qwen3.8-27b-nvfp4-mtp5 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 "$@"
