#!/bin/bash
# Serve nvidia/Qwen3.8-27B-NVFP4 + DFlash2 speculative decoding on the
# 2-node cluster (raven+quaker, TP=2) -- the recommended default for
# Qwen3.8-27B, see recipes/qwen3.8-27b-nvfp4-dflash2-nvidia.yaml header for
# the incoai/Qwen3.8-27B-DFlash2 source pick and the 2026-09-09 switch from
# unsloth's community quant to NVIDIA's own first-party one (re-verified
# 2026-09-09, see project memory qwen38-27b-dflash2-recipe).
#
# Filename carries an -nvidia suffix (2026-09-10) because upstream
# (eugr/spark-vllm-docker) independently added its own, unrelated
# recipes/qwen3.8-27b-nvfp4-dflash2.yaml (RadixArk checkpoint + z-lab
# DFlash2 draft) -- same name, different model, to avoid an add/add merge
# conflict if upstream is ever merged into this fork.
# Once up, the OpenAI-compatible API is at http://localhost:8000/v1
cd "$(dirname "${BASH_SOURCE[0]}")"
exec ./run-recipe.sh qwen3.8-27b-nvfp4-dflash2-nvidia --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 "$@"
