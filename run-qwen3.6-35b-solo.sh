#!/bin/bash
# Qwen3.6-35B-A3B-NVFP4 서빙 (22GB, raven 단독 — quaker 불필요)
# 완료되면 OpenAI 호환 API: http://localhost:8000/v1
cd "$(dirname "${BASH_SOURCE[0]}")"
exec ./run-recipe.sh qwen3.6-35b-a3b-nvfp4 --solo --tensor-parallel 1 "$@"
