#!/bin/bash
set -euo pipefail

# Qwen3.8-Flash-Next NVFP4 checkpoints (e.g. RadixArk/Qwen3.8-Flash-Next-NVFP4) are
# hybrid: routed experts are ModelOpt NVFP4, but the PLE n-gram embedding table ships
# as FP8 shards with a single global weight_scale. Stock vLLM's ple_layer.py only
# enables its FP8 PLE path when the outer quant config is Fp8Config -- here it's
# modelopt, so it silently upcasts the FP8 bytes to bf16 with no scale applied
# (wrong embeddings, no crash), and once that gate is fixed a second bug surfaces:
# the FP8 method registers weight_scale as a parameter in create_weights while the
# loader registers it as a buffer, raising "attribute 'weight_scale' already exists".
#
# This mod replaces ple_layer.py with a version that: gates the FP8 PLE method on
# config.ple_embedding_dtype == "float8_e4m3fn" (present in RadixArk's config.json)
# regardless of the outer quant config, stops double-registering weight_scale as a
# parameter, and preserves the loaded scale as a buffer during weight loading.
#
# Diffed clean against this repo's own pulled vllm/vllm-openai:qwen38-flash-next
# stock ple_layer.py (~12 lines changed in a 1244-line file) before adoption.
# Adapted from github.com/x00byte/Qwen3.8-Flash-Dual-Spark-Recipe (Apache 2.0,
# derived from vLLM). See recipes/qwen3.8-flash-next-nvfp4-radixark.yaml for context.

PYTHON_ROOT="${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}"
TARGET="$PYTHON_ROOT/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py"

if [ ! -f "$TARGET" ]; then
  echo "[qwen38-flash-next-ple-fp8-fix] $TARGET not found -- vLLM source layout doesn't match, refusing to apply" >&2
  exit 1
fi

cp ple_layer.py "$TARGET"
python3 -c "import ast; ast.parse(open('$TARGET').read())" && echo "[qwen38-flash-next-ple-fp8-fix] ple_layer.py replaced and ast.parse clean"
