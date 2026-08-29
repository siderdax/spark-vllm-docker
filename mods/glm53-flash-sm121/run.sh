#!/bin/bash
set -euo pipefail

# GLM-5.3-Flash (glm5_next, NoPE MLA: qk_rope_head_dim=0) on DGX Spark SM121/GB10.
#
# Stock vllm/vllm-openai:glm53-flash-arm64-cu130's only capability-12 sparse-MLA
# backend is FLASHINFER_MLA_SPARSE_SM120, whose packed fp8_ds_mla cache layout
# hard-requires DeepSeek's pe_dim=64 and dies in warmup ("pe_dim must be 64 for
# fp8_ds_mla"). We hit this ourselves (see project memory glm53-flash-sm120-blocker)
# and confirmed upstream (vllm-project/vllm#53963, open, unfixed as of 2026-08-28)
# that no SM120 sparse-MLA kernel exists for the rope-free shape.
#
# This mod takes a different path: FLASHINFER_MLA_SPARSE_SM90 (the plain-bf16-cache
# Hopper NoPE backend) was never coupled to DeepSeek's pe_dim=64 assumption and
# handles qk_rope_head_dim=0 by construction. Extend its compute-capability gate
# to also accept SM120/121, upgrade FlashInfer to the nightly that ships official
# SM90-NoPE support (and fixes a NaN bug on 64-256 row batches, bisected on SM121),
# repin two packages the nightly silently downgrades, gate off PDL (races on GB10's
# KDA recurrent-state Triton kernels), and harden the sparse-MLA kpool indexer
# against uninitialized-buffer NaNs.
#
# Adapted from the (independently reviewed, file-by-file, before use) community
# patch stack at github.com/barrydeen/glm53-flash-dgx-spark, itself building on
# github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark. This mod applies
# only that repo's v1/v3/v4/v5/v6/v7 layers (the validated bf16-KV baseline) --
# v2 (NaN-debug hooks) and v8/dflash2 (fp8 KV cache + speculative decoding,
# CC-BY-NC-ND-licensed draft model) are left out for now.

PYTHON_ROOT="${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}"

if [ ! -d "$PYTHON_ROOT/vllm" ]; then
  echo "[glm53-flash-sm121] vLLM package not found at $PYTHON_ROOT/vllm" >&2
  exit 1
fi

echo "[glm53-flash-sm121] v1: extending FLASHINFER_MLA_SPARSE_SM90 to capability 12"
python3 - "$PYTHON_ROOT" <<'PY'
import sys
from pathlib import Path

base = Path(sys.argv[1]) / "vllm"

p = base / "platforms/cuda.py"
s = p.read_text()
marker = "AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM90,\n                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,"
old = """        elif device_capability.major == 12:
            return [
                AttentionBackendEnum.TRITON_MLA,
                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,
            ]"""
new = """        elif device_capability.major == 12:
            return [
                AttentionBackendEnum.TRITON_MLA,
                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM90,
                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,
            ]"""
if marker not in s:
    if s.count(old) != 1:
        raise SystemExit("[v1] unexpected cuda.py capability-12 MLA candidate list; refusing to patch")
    p.write_text(s.replace(old, new))
    print("[v1] cuda.py: capability-12 candidate list extended")
else:
    print("[v1] cuda.py: already applied, skipping")

p = base / "v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py"
s = p.read_text()

old_gate = "    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:\n        return capability.major == 9\n"
new_gate = "    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:\n        return capability.major in (9, 12)\n"
if "capability.major in (9, 12)" not in s:
    if s.count(old_gate) != 1:
        raise SystemExit("[v1] unexpected sm90 capability gate; refusing to patch")
    s = s.replace(old_gate, new_gate)
    print("[v1] flashinfer_mla_sparse_sm90.py: capability gate extended to (9, 12)")
else:
    print("[v1] flashinfer_mla_sparse_sm90.py: capability gate already applied, skipping")

old_backend = '            backend="fa3",\n'
new_backend = '            backend=("fa3" if torch.cuda.get_device_capability()[0] == 9 else "fa2"),\n'
if 'backend=("fa3" if torch.cuda.get_device_capability()[0] == 9 else "fa2")' not in s:
    if s.count(old_backend) != 1:
        raise SystemExit("[v1] unexpected sm90 wrapper backend literal; refusing to patch")
    s = s.replace(old_backend, new_backend)
    print("[v1] flashinfer_mla_sparse_sm90.py: FA3->FA2 off-Hopper")
else:
    print("[v1] flashinfer_mla_sparse_sm90.py: FA2 selection already applied, skipping")

old_gate2 = """        if not has_flashinfer_sm90_nope_mla():
            return (
                "FLASHINFER_MLA_SPARSE_SM90 requires FlashInfer with SM90 "
                "MLA support (ckv_scale_arr in "
                "BatchMLAPagedAttentionWrapper.run, FlashInfer >= 0.6.18)"
            )"""
new_gate2 = """        if kv_cache_dtype in ("fp8", "fp8_e4m3") and not has_flashinfer_sm90_nope_mla():
            return (
                "FLASHINFER_MLA_SPARSE_SM90 fp8 KV requires FlashInfer with "
                "SM90 MLA support (ckv_scale_arr in "
                "BatchMLAPagedAttentionWrapper.run, FlashInfer >= 0.6.18)"
            )"""
if 'kv_cache_dtype in ("fp8", "fp8_e4m3") and not has_flashinfer_sm90_nope_mla()' not in s:
    if s.count(old_gate2) != 1:
        raise SystemExit("[v1] unexpected sm90 flashinfer version gate; refusing to patch")
    s = s.replace(old_gate2, new_gate2)
    print("[v1] flashinfer_mla_sparse_sm90.py: version gate scoped to fp8-KV path")
else:
    print("[v1] flashinfer_mla_sparse_sm90.py: version gate already scoped, skipping")

p.write_text(s)
print("[v1] done")
PY

echo "[glm53-flash-sm121] v3: upgrading FlashInfer to nightly with official SM90-NoPE support"
CURRENT_FI="$(python3 -c 'import flashinfer; print(flashinfer.__version__)' 2>/dev/null || echo none)"
if [ "$CURRENT_FI" != "0.6.18.dev20260819" ]; then
  pip install -q --pre \
    flashinfer-python==0.6.18.dev20260819 \
    flashinfer-cubin==0.6.18.dev20260819 \
    --extra-index-url https://flashinfer.ai/whl/nightly/
  pip uninstall -q -y flashinfer-jit-cache || true
  python3 -c "import flashinfer; print('[v3] flashinfer', flashinfer.__version__)"
else
  echo "[v3] flashinfer already at 0.6.18.dev20260819, skipping"
fi

echo "[glm53-flash-sm121] v4: repinning nvidia-nccl-cu13 (nightly silently downgrades it, breaks IB fabric)"
pip install -q nvidia-nccl-cu13==2.30.7
pip show nvidia-nccl-cu13 | grep -q "Version: 2.30.7" && echo "[v4] nccl repinned to 2.30.7"

echo "[glm53-flash-sm121] v5: repinning nvidia-cutlass-dsl (nightly leaves a mixed 4.7.0/4.6.2 install)"
pip install -q nvidia-cutlass-dsl==4.6.2
pip show nvidia-cutlass-dsl | grep -q "Version: 4.6.2" && echo "[v5] cutlass-dsl repinned to 4.6.2"

echo "[glm53-flash-sm121] v6: gating PDL off SM12x (races on KDA recurrent-state kernels on GB10)"
python3 - "$PYTHON_ROOT" <<'PY'
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "vllm/platforms/cuda.py"
s = p.read_text()
old = """    @classmethod
    def is_arch_support_pdl(cls) -> bool:
        try:
            device = torch.cuda.current_device()
            major, _ = torch.cuda.get_device_capability(device)
        except Exception:
            return False
        return major >= 9
"""
new = """    @classmethod
    def is_arch_support_pdl(cls) -> bool:
        try:
            device = torch.cuda.current_device()
            major, _ = torch.cuda.get_device_capability(device)
        except Exception:
            return False
        # PDL lowering is unvalidated on SM12x (GB10) and races on KDA
        # state kernels there; keep it to Hopper/Blackwell-datacenter.
        return major in (9, 10)
"""
if "major in (9, 10)" not in s:
    if s.count(old) != 1:
        raise SystemExit("[v6] unexpected is_arch_support_pdl source; refusing to patch")
    p.write_text(s.replace(old, new))
    print("[v6] PDL gated off on SM12x")
else:
    print("[v6] PDL gate already applied, skipping")
PY

echo "[glm53-flash-sm121] v7: hardening sparse-MLA kpool indexer against uninitialized-buffer NaNs"
python3 - "$PYTHON_ROOT" <<'PY'
import sys
from pathlib import Path

base = Path(sys.argv[1]) / "vllm"

p = base / "model_executor/layers/sparse_attn_indexer_kpool.py"
s = p.read_text()

prefill_old = (
    "                pool_topk = torch.empty(\n"
    "                    (num_rows, select_k), dtype=torch.int32, device=logits.device\n"
    "                )\n"
)
prefill_new = (
    "                pool_topk = torch.full(\n"
    "                    (num_rows, select_k), -1, dtype=torch.int32, device=logits.device\n"
    "                )\n"
)
decode_old = (
    "            pool_topk = torch.empty(\n"
    "                (num_rows, select_k), dtype=torch.int32, device=logits.device\n"
    "            )\n"
)
decode_new = (
    "            pool_topk = torch.full(\n"
    "                (num_rows, select_k), -1, dtype=torch.int32, device=logits.device\n"
    "            )\n"
)
if "torch.full(\n                    (num_rows, select_k), -1" in s:
    print("[v7] sparse_attn_indexer_kpool.py: already applied, skipping")
else:
    if s.count(prefill_old) != 1:
        raise SystemExit("[v7] prefill alloc anchor not found (count=%d)" % s.count(prefill_old))
    s = s.replace(prefill_old, prefill_new)
    if s.count(decode_old) != 1:
        raise SystemExit("[v7] decode alloc anchor not found (count=%d)" % s.count(decode_old))
    s = s.replace(decode_old, decode_new)
    p.write_text(s)
    print("[v7] sparse_attn_indexer_kpool.py: topk buffers now -1-initialized")

p = base / "models/glm5next/nvidia/ops/kpool_compress.py"
s = p.read_text()
guard_old = "    hist_out = tl.where(pid >= 0, hist_val, -1)\n"
guard_new = "    hist_out = tl.where((pid >= 0) & (pid < pool_len), hist_val, -1)\n"
if "(pid >= 0) & (pid < pool_len)" in s:
    print("[v7] kpool_compress.py: already applied, skipping")
else:
    if s.count(guard_old) != 1:
        raise SystemExit("[v7] expansion guard anchor not found (count=%d)" % s.count(guard_old))
    p.write_text(s.replace(guard_old, guard_new))
    print("[v7] kpool_compress.py: pool-length bounds clamp applied")
PY

echo "[glm53-flash-sm121] all patches applied successfully"
