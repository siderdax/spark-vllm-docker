#!/bin/bash
# Qwen3.6-27B-FP8 서빙 (27GB, dense, raven 단독 — quaker는 자유)
# 완료되면 OpenAI 호환 API: http://localhost:8000/v1
cd "$(dirname "${BASH_SOURCE[0]}")"
exec ./run-recipe.sh qwen3.6-27b-fp8 --solo --tensor-parallel 1 "$@"
