#!/bin/bash
# Serve unsloth/Qwen3.8-27B-NVFP4 + MTP-5 on the 2-node cluster (raven+quaker, TP=2).
# Sibling of run-qwen3.8-27b-nvfp4.sh (MTP-3) for the 4-way NVFP4/FP8 x
# MTP-5/DFlash2 comparison. Once up, the OpenAI-compatible API is at
# http://localhost:8000/v1
cd "$(dirname "${BASH_SOURCE[0]}")"
exec ./run-recipe.sh qwen3.8-27b-nvfp4-mtp5 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 "$@"
