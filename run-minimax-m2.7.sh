#!/bin/bash
# MiniMax-M2.7-NVFP4 서빙 (131GB, 2노드 클러스터: raven+quaker)
# 완료되면 OpenAI 호환 API: http://localhost:8000/v1
cd "$(dirname "${BASH_SOURCE[0]}")"
# quaker(워커 노드)는 인터넷이 없어 HF Hub 조회가 DNS 실패로 죽는다.
# 컨테이너 레벨 -e로 넣어야 Ray 워커 프로세스까지 확실히 전파됨(레시피 env:만으론 부족).
exec ./run-recipe.sh minimax-m2.7-nvfp4 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 "$@"
