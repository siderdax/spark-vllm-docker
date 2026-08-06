#!/bin/bash
# Qwen3.5-122B-A10B-NVFP4 서빙 (78GB, raven 단독 — quaker는 ComfyUI 등 자유)
# 남는 메모리가 적어 컨텍스트를 131k로 제한. 더 줄이면: --max-model-len 65536
# 완료되면 OpenAI 호환 API: http://localhost:8000/v1
cd "$(dirname "${BASH_SOURCE[0]}")"
exec ./run-recipe.sh qwen3.5-122b-nvfp4 --solo --tensor-parallel 1 \
    --gpu-memory-utilization 0.85 --max-model-len 131072 "$@"
