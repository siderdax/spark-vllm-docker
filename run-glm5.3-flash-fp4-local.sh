#!/bin/bash
# Serve nvidia/GLM-5.3-Flash-FP4-Local (~165GiB, NVIDIA QAD NVFP4+MXFP8) with
# MTP-5 on the 2-node cluster (raven+quaker, TP=2), using the community
# eugr/spark-vllm-b12x image the model card requires. VERIFIED 2026-10-27 --
# on raven+quaker; see custom_recipes/glm5.3-flash-fp4-local.yaml.
# Once up, the OpenAI-compatible API is at http://localhost:8000/v1
#
# Prerequisite: nvidia/GLM-5.3-Flash-FP4-Local in the HF cache on BOTH nodes.
# Stop any running vLLM first -- this uses all GPU memory on both nodes.
cd "$(dirname "${BASH_SOURCE[0]}")"

# Dirty page cache before a large-context launch has crashed first prefill
# before; no passwordless sudo, so use a throwaway privileged container.
echo "Dropping page cache on both nodes before launch..."
docker run --rm --privileged --pid=host alpine sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' && echo "  raven: OK"
ssh quaker.local "docker run --rm --privileged --pid=host alpine sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'" && echo "  quaker: OK"

# quaker has no internet; env must be container-level -e to reach workers.
exec ./run-recipe.sh custom_recipes/glm5.3-flash-fp4-local --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 "$@"
