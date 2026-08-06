#!/bin/bash
# Serve DeepSeek-V4-Flash (official FP8, 149GB) on the 2-node cluster (raven+quaker, TP=2).
# Once up, the OpenAI-compatible API is at http://localhost:8000/v1
cd "$(dirname "${BASH_SOURCE[0]}")"
# quaker (worker node) has no internet, so HF Hub lookups die on DNS failure.
# These must be passed as container-level -e to reach the worker processes (recipe env: alone is not enough).
exec ./run-recipe.sh deepseek-v4-flash --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 "$@"
