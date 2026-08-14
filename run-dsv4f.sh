#!/bin/bash
# Serve DeepSeek-V4-Flash (official, 149GiB) on the 2-node cluster (raven+quaker, TP=2).
# Despite config.json saying quant_method=fp8, the routed experts are MXFP4: 283.5B
# 4-bit weights packed into I8 with one E8M0 scale per 32 (verified from the
# safetensors headers). FP8 e4m3 covers only the shared experts (~6GB).
# Once up, the OpenAI-compatible API is at http://localhost:8000/v1
cd "$(dirname "${BASH_SOURCE[0]}")"
# quaker (worker node) has no internet, so HF Hub lookups die on DNS failure.
# These must be passed as container-level -e to reach the worker processes (recipe env: alone is not enough).
exec ./run-recipe.sh deepseek-v4-flash --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 "$@"
