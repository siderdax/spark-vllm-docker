
# vLLM Docker Optimized for DGX Spark — raven+quaker local tuning fork

This is a personal fork of [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker), tracking a 2-node DGX Spark cluster (`raven` + `quaker`).

For build instructions, the mods system, `launch-cluster.sh`/`run-recipe.py` usage, general configuration, model loading, and downloading models, see **[the original repository's README](https://github.com/eugr/spark-vllm-docker#readme)**. This file only documents what's local to this fork: the raven+quaker-tuned recipes/launch scripts and the tuning notes behind them.

## Table of Contents

- [Fork-specific recipes and launch scripts](#fork-specific-recipes-and-launch-scripts)
- [CHANGELOG (fork-specific)](#changelog-fork-specific)
- [Long-context correctness (`context-bench.py`)](#long-context-correctness-context-benchpy)

## Fork-specific recipes and launch scripts

These recipes and `run-*.sh` wrappers exist only in this fork (not upstream):

| Model | Recipe(s) | Wrappers |
|---|---|---|
| Qwen3.6-27B-FP8 | `recipes/qwen3.6-27b-fp8.yaml` | `run-qwen3.6-27b.sh` / `-solo` / `-pp2` |
| Qwen3.6-27B-NVFP4 | `recipes/qwen3.6-27b-nvfp4.yaml` | `run-qwen3.6-27b-nvfp4.sh` / `-solo` / `-pp2` |
| Qwen3.6-35B-A3B | `recipes/qwen3.6-35b-a3b-*.yaml` | `run-qwen3.6-35b.sh` / `-solo` / `-pp2` |
| Qwen3.8-27B-FP8 | `recipes/qwen3.8-27b-fp8.yaml` | `run-qwen3.8-27b.sh` / `-solo` / `-pp2` |
| Qwen3.5-122B-NVFP4 | `recipes/qwen3.5-122b-nvfp4.yaml` | `run-qwen3.5-122b-nvfp4.sh` / `-solo` / `-pp2` |
| DeepSeek-V4-Flash | `recipes/deepseek-v4-flash.yaml` | `run-dsv4f.sh` |
| DeepSeek-V4-Flash-0731 | `recipes/deepseek-v4-flash-0731.yaml` | `run-dsv4f-0731.sh` |
| MiniMax-M2.7-NVFP4 | `recipes/minimax-m2.7-nvfp4.yaml` | `run-minimax-m2.7.sh` |

```bash
./run-qwen3.8-27b.sh          # tp2 cluster (raven+quaker), the recipe default
./run-qwen3.8-27b-solo.sh     # single node
./run-qwen3.8-27b-pp2.sh      # pipeline-parallel across both nodes
```

Every wrapper forwards extra args straight to `run-recipe.sh`/`run-recipe.py` (`--api-key`, `--port`, overrides, etc.) — see the original repo's README for the full flag reference.

## CHANGELOG (fork-specific)

For the full project history before this fork diverged, see the [original repository's CHANGELOG](https://github.com/eugr/spark-vllm-docker#changelog).

### 2026-08-26

#### Merged upstream `main` (8 months of drift); kept our DeepSeek-V4-Flash-0731 recipe over upstream's

Local `main` had fallen 61 commits behind `origin/main` (eugr/spark-vllm-docker), which had independently diverged by 40k+ lines while this fork's `local-tuning` branch grew alongside it. Fast-forwarded local `main` to the current fork remote, then merged it into `local-tuning`. Only 3 files actually conflicted: `.gitignore` (trivial union), `README.md` (kept ours — this file was just rewritten to be fork-specific), and `recipes/deepseek-v4-flash-0731.yaml`, where upstream had independently built its own version of the same recipe.

Benchmarked both on raven+quaker (TP2) before resolving that conflict:

- **Image freshness**: our locally-cached `eugr/spark-vllm-b12x:latest` was from 2026-08-13; re-pulling picked up a newer build. Same recipe, same everything else — just the refreshed image gave `pp2048` 2014→2150 t/s (+6.7%) and `tg256` 40.7→42.2 t/s (+3.6%, peak +11%). Worth periodically re-pulling regardless of which recipe wins.
- **Upstream's recipe** uses `container: vllm-node-b12x`, which turned out to resolve to the exact same `eugr/spark-vllm-b12x:latest` image (`build-and-copy.sh --exp-b12x` only triggers a real source compile if combined with `--rebuild-vllm`; alone it just pulls-and-retags) — so no actual image difference to benchmark.
- The one real difference, `mods/instanttensor-hybrid-draft-loader` (upstream-only, switches the DSpark draft head to lazy-safetensors loading instead of a second InstantTensor pass), made loading *slower* on our setup: DSpark draft load went 20.05s → 30.58s (total model load 50.9s → 63.2s, +24%). InstantTensor's sequential GPU-streaming pass over the 155GB checkpoint is already fast enough that the mod's random-access path for the tiny 97-param draft head didn't pay off here.

**Why the mod backfired here** (researched, not just measured): [InstantTensor](https://github.com/scitix/InstantTensor) is a purpose-built loader — Direct I/O, pipelined prefetch, optionally GPUDirect Storage — tuned to saturate sequential read bandwidth. Our ~20s for 155GB (~7.5-8 GB/s) lines up with [DGX Spark's measured internal NVMe throughput](https://docs.nvidia.com/dgx/bp-dgx/storage.html) (cached reads ~10.6 GB/s, disk reads ~6.6 GB/s), so InstantTensor is already running close to the hardware ceiling. The mod's "lazy safetensors" fallback is the standard library's mmap-based lazy loader, which — per [vllm-project/vllm#40988](https://github.com/vllm-project/vllm/issues/40988) and general safetensors performance writeups — resolves each accessed tensor as its own scattered page-fault, sometimes thousands of small random reads per shard. Even though the DSpark draft is tiny (97 params), if those params are spread across the checkpoint's shard files, mmap's per-tensor random-read overhead outweighs the "less total data" theoretically read. The mod likely earns its keep in setups without a Direct-I/O loader like InstantTensor already saturating the drive; on top of one, it doesn't have anything left to win.

Kept `recipes/deepseek-v4-flash-0731.yaml` as our version (no mod, tuned defaults, `--override-generation-config`, `reasoning_effort=max`). Preserved upstream's version as `recipes/deepseek-v4-flash-0731-selfbuilt.yaml` for reference, in case a future upstream image/mod change is worth re-testing — it's not wired into any `run-*.sh` wrapper.

#### MTP speculative decoding enabled by default on Qwen3.6/3.8

Benchmarked Qwen3.8-27B-FP8 speculative decoding on raven+quaker (see `qwen3.8-mtp-optimization-report.md`) across MTP token counts and against the NVFP4 variant. Official FP8 + MTP 4 won outright — `21.2-22.0 tok/s` vs `12.5 tok/s` with MTP off (+70-76%), and vs `19.8 tok/s` for NVFP4 + MTP 4 despite NVFP4 being faster with MTP off. The gap comes from draft-acceptance rate: the MTP draft head was trained on clean FP8/BF16 hidden states, so NVFP4's quantization noise drops its 4th-token hit rate to 18-28% vs FP8's 31-37% — at MTP 4 that acceptance rate dominates over the raw weight-read savings NVFP4 gets with MTP off.

Added `--speculative-config '{"method":"mtp","num_speculative_tokens":4}'` to `qwen3.6-27b-fp8`, `qwen3.6-27b-nvfp4`, `qwen3.6-35b-a3b-fp8`, and `qwen3.8-27b-fp8` (bumped `qwen3.6-35b-a3b-nvfp4` from 3 to 4). Also added `--no-mtp`/`--speculative-tokens` overrides to `run-recipe.py` so individual launches can disable or retune it without editing the recipe:

```bash
./run-qwen3.8-27b.sh --no-mtp
./run-qwen3.8-27b.sh --speculative-tokens 3
```

### 2026-08-20

#### Qwen3.6: thinking-mode sampling defaults across all 6 recipes

Followed up on the Qwen3.8 audit below by checking all 6 Qwen3.6 recipes
(`qwen3.6-27b-fp8`, `qwen3.6-27b-nvfp4`, `qwen3.6-35b-a3b-fp8[-dflash]`,
`qwen3.6-35b-a3b-nvfp4[-no-mtp]`) for the same class of gaps.

Tool-call parser was already correct everywhere (`qwen3_xml`) — worth noting since
Qwen's own Qwen3.6 model cards recommend `qwen3_coder`, which is stale/wrong advice:
that parser has a known vLLM bug producing an infinite "!" stream on long inputs
containing a tool call (vllm-project/vllm#39056), affecting the whole Qwen3 family,
not just 3.8.

Sampling defaults, however, were missing everywhere (none of the 6 had
`--override-generation-config`). Added Qwen's documented thinking-mode recommendation
to each, which differs by architecture: the dense 27B recipes get
`temperature=1.0, top_p=0.95, top_k=20` (same as 3.8); the A3B (MoE) recipes get
`temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5` — Qwen recommends a
nonzero presence_penalty for the MoE variant specifically (dense recommends 0.0,
vLLM's own default), likely to curb MoE repetition loops.

Also documented (in recipes with the local `fix-qwen3.6-chat-template` mod) that
this template has no `reasoning_effort` mechanism at all — thinking is on/off only,
unlike Qwen3.8's xhigh/medium/low — but `preserve_thinking` still defaults to true
and carries the same context-snowballing risk on long multi-turn sessions. The two
NVFP4-Marlin recipes use the stock HF chat template (no local mod), so no
reasoning_effort/preserve_thinking claims were asserted for those two.

#### Qwen3.8-27B: thinking-mode sampling defaults + opencode-hang troubleshooting doc

Added `--override-generation-config '{"temperature": 1.0, "top_p": 0.95, "top_k": 20}'` to
`recipes/qwen3.8-27b-fp8.yaml` — Qwen's documented thinking-mode sampling recommendation. vLLM
defaults `top_k` to unbounded (-1), which diverges from that recommendation and can let `<think>`
generations ramble; this only fills gaps for clients that don't set their own sampling params.
Also expanded the recipe's header comment into a troubleshooting section covering opencode
sessions that hang with no output: how to diagnose via `finish_reason` on a direct curl request,
the existing `reasoning_effort`/`preserve_thinking` per-request overrides (defaults unchanged —
still `xhigh`/`true`), and confirmation that `qwen3_xml` (not `qwen3_coder`, which has a known vLLM
bug) is the correct tool-call parser.

Benchmarked TP2 (raven+quaker) vs TP1-solo with `llama-benchy` (`--pp 2048,16384 --tg 256
--exact-tg --concurrency 1 --runs 3`, thinking off, otherwise identical flags) to settle whether
dual-Spark interconnect overhead hurts single-request decode latency for this dense 27B model. It
doesn't — TP2 won on every axis even at concurrency 1: prefill +40-59% (2719→1936 t/s at pp2048,
2847→1793 t/s at pp16384) and decode +81% (~2x: 15.2→8.4 t/s at tg256). Turns out the intuition
that TP overhead is dominated by cross-node communication latency doesn't hold here — raven and
quaker are separate GB10 chips with their own bandwidth-limited local memory connected by a
*direct* QSFP/RoCE link (no switch hop), so decode's real bottleneck (reading ~27GB of weights per
step off one chip's memory) gets roughly halved by splitting across two independent memory pools,
and the direct-link network tax doesn't come close to offsetting that. tp2 stays the recipe
default; MTP speculative decoding remains untested.

#### DeepSeek-V4-Flash: agentic top_p + max reasoning_effort

Neither `deepseek-v4-flash` nor `deepseek-v4-flash-0731` set `--override-generation-config`, so `top_p` was silently falling back to vLLM's non-agentic default of `1.0` instead of the official recipe/model-card's recommended `0.95` for agentic scenarios (`temperature=1.0` already matched vLLM's default). Added `--override-generation-config '{"temperature": 1.0, "top_p": 0.95}'` to both. Also bumped `-0731`'s `reasoning_effort` from `high` to `max` to match.

### 2026-08-19

#### Qwen3.8-27B-FP8 recipe with chat-template fix and launch wrappers

Added `recipes/qwen3.8-27b-fp8.yaml`, porting the same chat-template robustness fixes used for Qwen3.6 (`mods/fix-qwen3.8-chat-template`) onto Qwen3.8's actual template — no developer-role support, tool-call argument handling on non-mapping args, no auto-close for a dangling `<think>` before a `<tool_call>` — while preserving Qwen3.8's `reasoning_effort` (xhigh/medium/low) system-prompt injection that the plain Qwen3.6 fix doesn't have. Also added `run-qwen3.8-27b.sh`/`-solo`/`-pp2` launch wrappers mirroring the existing raven+quaker pattern.

### 2026-08-14

#### `context-bench.py` long-context correctness test

Added a needle-in-a-haystack benchmark (see [below](#long-context-correctness-context-benchpy)) that plants a random number at a configurable depth in a long prompt and checks the running server both survives the request near its `--max-model-len` and recalls the number correctly — `llama-benchy` covers throughput, this covers "does the long context actually work." Used it to verify `deepseek-v4-flash-0731` at 500000 on raven+quaker: 9/9 passed across 120k/300k/490k-token prompts and three needle depths each.

#### DeepSeek-V4-Flash-0731 DSpark fix: official B12X image

`recipes/deepseek-v4-flash-0731.yaml` previously failed to start DSpark speculative decoding on our self-built `vllm-node` image (`sparse_mla_sm120.cu: Check failed: num_tokens > 64`), and `mtp` can't load 0731's restructured MTP block at all. Switched the recipe's container to the official `recipes.vllm.ai` DGX Spark (GB10) image, `eugr/spark-vllm-b12x:latest`, and added the matching `--moe-backend`/`--linear-backend`/`--attention-backend b12x`/`B12X_MLA_SPARSE` flags and env vars. Verified end-to-end on raven+quaker (server starts, health check passes, long-context bench above).

### 2026-08-02

#### 2-Node Pipeline-Parallel (pp2) Scripts

Added `run-qwen3.6-27b-pp2.sh`, `run-qwen3.6-35b-pp2.sh`, and `run-qwen3.5-122b-nvfp4-pp2.sh` for the three models that already have a `-solo` variant. Each runs `-tp 1` with `--pipeline-parallel-size 2` across raven+quaker instead of the default tp2 split — pipeline parallelism moves far less data between nodes per step than tensor parallelism, so it can help when the inter-node link is the bottleneck (at the cost of not parallelizing within a single decode step).

```bash
./run-qwen3.6-27b-pp2.sh
```

#### `--api-key` Option in `run-recipe.py`

`run-recipe.sh` / `run-recipe.py` now accept `--api-key KEY` to require clients to authenticate against the OpenAI-compatible endpoint. It sets `VLLM_API_KEY` as a container-level `-e` (same propagation path used for `HF_HUB_OFFLINE`), so it also reaches worker nodes in cluster mode. If omitted, it falls back to the `API_KEY` environment variable. Since every `run-*.sh` wrapper forwards its args to `run-recipe.sh`, this works out of the box for all of them:

```bash
./run-qwen3.6-27b-solo.sh --api-key sk-my-secret
# or
API_KEY=sk-my-secret ./run-dsv4f.sh
```

## Long-context correctness (`context-bench.py`)

`context-bench.py` is a needle-in-a-haystack test against a running OpenAI-compatible server: it builds a long filler prompt with a random number planted at a given depth, asks the model to recall it, and checks both that the request completes (no OOM/crash/timeout near the configured `--max-model-len`) and that the answer is actually correct. `llama-benchy` (see the original repo's README) covers throughput/latency; this covers "does the long context actually work."

```bash
# Against a locally served model, auto-detected from /v1/models
./context-bench.py

# Explicit sizes (target prompt tokens) and needle depths (0.0=start, 1.0=end)
./context-bench.py --sizes 120000,300000,490000 --depths 0.1,0.5,0.9

# Remote server / different API key
./context-bench.py --base-url http://raven:8000/v1 --api-key sk-my-secret
```

Uses `$API_KEY` or `./api_key.txt` the same way the `run-*.sh` wrappers do. Reasoning/thinking is disabled per-request by default for a clean signal; pass `--thinking` to leave it on. Results are written to `context-bench-results.json` and exit non-zero if any case errored or failed recall.
