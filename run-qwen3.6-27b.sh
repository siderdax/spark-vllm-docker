#!/bin/bash
# Qwen3.6-27B-FP8 서빙 (27GB, dense, 2노드 클러스터: raven+quaker, tp2)
# NVFP4 대신 FP8 선택 — 품질이 원본 bf16과 거의 동일하다고 Qwen이 명시.
# 작아서 굳이 tp2가 필요친 않지만 dense라 compute-bound라 디코드 속도 이득 볼 수도 있음.
# 메모리만 필요하면 solo가 나음: run-qwen3.6-27b-solo.sh
# 네트워크가 병목이면 pp2가 나을 수도: run-qwen3.6-27b-pp2.sh
# 완료되면 OpenAI 호환 API: http://localhost:8000/v1
cd "$(dirname "${BASH_SOURCE[0]}")"
# quaker(워커 노드)는 인터넷이 없어 HF Hub 조회가 DNS 실패로 죽는다.
# 컨테이너 레벨 -e로 넣어야 Ray 워커 프로세스까지 확실히 전파됨(레시피 env:만으론 부족).
exec ./run-recipe.sh qwen3.6-27b-fp8 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 "$@"
