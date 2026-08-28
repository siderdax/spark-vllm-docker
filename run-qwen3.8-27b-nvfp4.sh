#!/bin/bash
# Qwen3.8-27B-NVFP4 서빙 (~25GB, 2노드 클러스터: raven+quaker, tp2)
# 완료되면 OpenAI 호환 API: http://localhost:8000/v1
cd "$(dirname "${BASH_SOURCE[0]}")"
exec ./run-recipe.sh qwen3.8-27b-nvfp4 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 "$@"
