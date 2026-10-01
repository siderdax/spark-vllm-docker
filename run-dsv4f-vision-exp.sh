#!/bin/bash
# Serve DeepSeek-V4-Flash-Vision-Exp with DSpark on the 2-node cluster
# (raven+quaker, TP=2). OpenAI-compatible API at http://localhost:8000/v1
#
# Uses eugr/spark-vllm-b12x:latest (pull it on every node first). The model is not
# cached yet, so no HF_HUB_OFFLINE here; download it once beforehand with:
#   ./run-recipe.sh custom_recipes/deepseek-v4-flash-vision-exp --download-only
# After it is cached you can add --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1.
# See the recipe yaml header for how it differs from upstream.
cd "$(dirname "${BASH_SOURCE[0]}")"
exec ./run-recipe.sh custom_recipes/deepseek-v4-flash-vision-exp "$@"
