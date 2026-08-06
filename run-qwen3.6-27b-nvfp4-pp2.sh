#!/bin/bash
# Qwen3.6-27B-NVFP4 서빙 (~14GB, dense, 2노드 클러스터: raven+quaker, pp2)
# tp2 대신 파이프라인 병렬 — 노드 간 통신량이 tp보다 적어 네트워크가 병목일 때 유리.
# 메모리만 필요하면 solo가 나음: run-qwen3.6-27b-nvfp4-solo.sh
# 완료되면 OpenAI 호환 API: http://localhost:8000/v1
cd "$(dirname "${BASH_SOURCE[0]}")"
# quaker(워커 노드)는 인터넷이 없어 HF Hub 조회가 DNS 실패로 죽는다.
# 컨테이너 레벨 -e로 넣어야 Ray 워커 프로세스까지 확실히 전파됨(레시피 env:만으론 부족).
exec ./run-recipe.sh qwen3.6-27b-nvfp4 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    --tensor-parallel 1 "$@" -- --pipeline-parallel-size 2
