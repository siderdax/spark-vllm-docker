#!/bin/bash
# Serve GLM-5.3-Flash-NVFP4 (RedHatAI checkpoint) + DFlash2 speculative
# decoding on the 2-node cluster (raven+quaker, TP=2), using tonyd2wild's
# pre-built sm121-v11-dflash2 image instead of our own runtime-patched
# mods/glm53-flash-sm121 -- see recipes/glm5.3-flash-dflash2.yaml header for
# why (FlashInfer JIT cache invalidation on every runtime reinstall).
# UNVERIFIED: first real launch attempt with this recipe on raven+quaker.
# Once up, the OpenAI-compatible API is at http://localhost:8000/v1
#
# Prerequisites (see recipes/glm5.3-flash-dflash2.yaml):
#   - image pulled+retagged as radixark/vllm-glm53-flash:sm121-v11-dflash2
#   - RedHatAI/GLM-5.3-Flash-NVFP4 + incoai/GLM-5.3-Flash-DFlash2 in HF cache
#   - ~/patches/sparse_attn_indexer_kpool.py present on BOTH nodes
cd "$(dirname "${BASH_SOURCE[0]}")"

# Same "recipe env: doesn't reach workers" gotcha as run-glm5.3-flash.sh --
# every var here must be passed as a container-level -e to reach both nodes.
# --volume: bind-mounts the SM121 top-k fix over the image's stock kernel
# (without it, decode past ~24K context hard-kills the engine). The host
# path must exist on BOTH nodes independently -- launch-cluster.sh passes
# -v mappings through unchanged, it does not scp them to workers the way it
# does for mods/.
exec ./run-recipe.sh glm5.3-flash-dflash2 \
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
