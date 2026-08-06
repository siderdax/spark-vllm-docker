#!/bin/bash
# Qwen3.5-122B-A10B-NVFP4 서빙 (78GB, 2노드 클러스터: raven+quaker, tp2)
# FP8 대비 디코드 빠르고 KV 캐시 여유 큼. 품질은 소폭 하락.
# 메모리만 필요하면 solo가 나음: run-qwen3.5-122b-nvfp4-solo.sh
# 네트워크가 병목이면 pp2가 나을 수도: run-qwen3.5-122b-nvfp4-pp2.sh
# 완료되면 OpenAI 호환 API: http://localhost:8000/v1
cd "$(dirname "${BASH_SOURCE[0]}")"
# quaker(워커 노드)는 인터넷이 없어 HF Hub 조회가 DNS 실패로 죽는다.
# 컨테이너 레벨 -e로 넣어야 Ray 워커 프로세스까지 확실히 전파됨(레시피 env:만으론 부족).
exec ./run-recipe.sh qwen3.5-122b-nvfp4 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 "$@"
