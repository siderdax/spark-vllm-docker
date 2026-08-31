#!/bin/bash
# Serve GLM-5.3-Flash-NVFP4 (RedHatAI checkpoint) + native MTP speculative
# decoding on the 2-node cluster (raven+quaker, TP=2) -- sibling of
# run-glm5.3-flash-dflash2.sh for a direct speed comparison, see
# recipes/glm5.3-flash-mtp.yaml header. Once up, the OpenAI-compatible API
# is at http://localhost:8000/v1
#
# Prerequisites:
#   - image pulled+retagged as radixark/vllm-glm53-flash:sm121-v8
#   - RedHatAI/GLM-5.3-Flash-NVFP4 in HF cache (no drafter needed for MTP)
#   - ~/patches/sparse_attn_indexer_kpool.py present on BOTH nodes
cd "$(dirname "${BASH_SOURCE[0]}")"

# Mandatory before every launch at large context (see recipe header /
# upstream docs/KV-HUNT-672K-TP2-RECORD.md): a dirty page cache on either
# node has been directly traced to a production crash on first real prefill
# after boot. No passwordless sudo here, so do it via a throwaway privileged
# container instead of `sudo tee /proc/sys/vm/drop_caches`.
echo "Dropping page cache on both nodes before launch..."
docker run --rm --privileged --pid=host alpine sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' && echo "  raven: OK"
ssh quaker.local "docker run --rm --privileged --pid=host alpine sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'" && echo "  quaker: OK"

# Same "recipe env: doesn't reach workers" gotcha as run-glm5.3-flash.sh --
# every var here must be passed as a container-level -e to reach both nodes.
# --volume: bind-mounts the SM121 top-k fix over the image's stock kernel
# (without it, decode past ~24K context hard-kills the engine). The host
# path must exist on BOTH nodes independently -- launch-cluster.sh passes
# -v mappings through unchanged, it does not scp them to workers the way it
# does for mods/.
exec ./run-recipe.sh glm5.3-flash-mtp \
  --volume "$HOME/patches/sparse_attn_indexer_kpool.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py:ro" \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env NCCL_IB_HCA=rocep1s0f0 \
  --env VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env TORCH_CUDA_ARCH_LIST=12.1a \
  --env FLASHINFER_CUDA_ARCH_LIST=12.1a \
  --env FLASHINFER_DISABLE_VERSION_CHECK=1 \
  --env NCCL_NET=IB \
  --env NCCL_IB_GID_INDEX=3 \
  --env NCCL_IB_ROCE_VERSION_NUM=2 \
  --env NCCL_IB_ADDR_FAMILY=AF_INET \
  --env NCCL_IB_ADDR_RANGE=192.168.10.0/24 \
  --env NCCL_NVLS_ENABLE=0 \
  --env NCCL_CROSS_NIC=0 \
  --env NCCL_IB_MERGE_NICS=0 \
  --env NCCL_CUMEM_ENABLE=0 \
  --env TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  "$@"
