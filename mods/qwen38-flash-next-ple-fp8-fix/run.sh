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
# derived from vLLM). See custom_recipes/qwen3.8-flash-next-nvfp4-radixark.yaml for context.
#
# EXTENDED 2026-09-09 while evaluating nvidia/Qwen3.8-Flash-Next-NVFP4 (see
# project memory qwen38-flash-next-nvidia-quant-attempt): that checkpoint
# doesn't set ple_embedding_dtype at all (confirmed null in its config.json),
# so the RadixArk-oriented gate above never fires and the FP8 PLE weight +
# its weight_scale buffer silently fall through to the unquantized default
# path -- loads fine, no crash, but produces garbage generations (verified:
# "The capital of France is" -> incoherent token soup, fixed by this change,
# re-verified coherent). Added a second gate that calls the outer
# ModelOptMixedPrecisionConfig's own _resolve_quant_algo(prefix) -- the same
# per-layer lookup vLLM already uses to correctly resolve that checkpoint's
# 48 NVFP4 MoE-expert layers -- and treats an "FP8" result for the PLE
# embedding's prefix the same as the RadixArk case. Pure addition: it sits
# after the ple_embedding_dtype check's return, so it's unreached dead code
# for RadixArk's checkpoint (which does set that field), and a no-op even if
# it were reached (RadixArk's quant_config is ModelOptNvFp4Config, which has
# no _resolve_quant_algo method) -- verified safe by code inspection; not yet
# re-verified with an actual RadixArk launch-test after this change.

PYTHON_ROOT="${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}"
TARGET="$PYTHON_ROOT/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py"

if [ ! -f "$TARGET" ]; then
  echo "[qwen38-flash-next-ple-fp8-fix] $TARGET not found -- vLLM source layout doesn't match, refusing to apply" >&2
  exit 1
fi

cp ple_layer.py "$TARGET"
python3 -c "import ast; ast.parse(open('$TARGET').read())" && echo "[qwen38-flash-next-ple-fp8-fix] ple_layer.py replaced and ast.parse clean"
