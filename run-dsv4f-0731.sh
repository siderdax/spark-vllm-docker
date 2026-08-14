#!/bin/bash
# Serve DeepSeek-V4-Flash-0731 (official, 155.4GiB) on the 2-node cluster
# (raven+quaker, TP=2). OpenAI-compatible API at http://localhost:8000/v1
#
# vs run-dsv4f.sh: same MXFP4 quantization and same architecture, but the 0731
# checkpoint ships DSpark block-speculative weights, so this recipe uses
# --speculative-config method=dspark (5 tokens) instead of mtp (2 tokens).
#
# 2026-08-14: switched to the official recipes.vllm.ai container
# (eugr/spark-vllm-b12x:latest) and B12X backend flags — the old vllm-node image
# hit a FlashInfer sparse-MLA assertion on every DSpark draft call. See the recipe
# yaml header for details. Run `docker pull eugr/spark-vllm-b12x:latest` on every
# node first; run-recipe.py won't build this one, it only checks it's present.
#
# Baseline to compare against, measured 2026-08-10 on the original checkpoint with
# mtp/2 (thinking off, warm): 35.5 tok/s on code, 37.0 tok/s on prose.
# To isolate the checkpoint from DSpark, edit the recipe's speculative-config back
# to {"method":"mtp","num_speculative_tokens":2} and re-run.
cd "$(dirname "${BASH_SOURCE[0]}")"
# Offline flags keep startup from doing Hub lookups on both nodes.
exec ./run-recipe.sh deepseek-v4-flash-0731 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 "$@"
