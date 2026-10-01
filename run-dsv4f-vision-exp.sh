#!/bin/bash
# Serve DeepSeek-V4-Flash-Vision-Exp with DSpark on the 2-node cluster
# (raven+quaker, TP=2). OpenAI-compatible API at http://localhost:8000/v1
#
# Uses eugr/spark-vllm-b12x:latest (pull it on every node first). The model is cached
# on both nodes (copied from fregata:/mnt/HDD2/models), so Hub lookups are disabled.
# See the recipe yaml header for how it differs from upstream and the verification results.
cd "$(dirname "${BASH_SOURCE[0]}")"
exec ./run-recipe.sh custom_recipes/deepseek-v4-flash-vision-exp --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 "$@"
