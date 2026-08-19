#!/bin/bash
# Qwen3.8-27B-FP8 서빙 (27GB, dense, vision-language, raven 단독 — quaker는 자유)
# 완료되면 OpenAI 호환 API: http://localhost:8000/v1
cd "$(dirname "${BASH_SOURCE[0]}")"
exec ./run-recipe.sh qwen3.8-27b-fp8 --solo --tensor-parallel 1 "$@"
