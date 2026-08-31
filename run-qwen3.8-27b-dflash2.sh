#!/bin/bash
# Serve unsloth/Qwen3.8-27B-NVFP4 + DFlash2 speculative decoding on the
# 2-node cluster (raven+quaker, TP=2) -- sibling of run-qwen3.8-27b-nvfp4.sh
# (native MTP-3) for a direct comparison, see recipes/qwen3.8-27b-dflash2.yaml
# header for why incoai/Qwen3.8-27B-DFlash2 was picked over the z-lab mirror.
# UNVERIFIED: first real launch attempt with this recipe on raven+quaker.
# Once up, the OpenAI-compatible API is at http://localhost:8000/v1
cd "$(dirname "${BASH_SOURCE[0]}")"
exec ./run-recipe.sh qwen3.8-27b-dflash2 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 "$@"
