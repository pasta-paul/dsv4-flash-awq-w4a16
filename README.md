# DeepSeek-V4-Flash → W4A16-FP8 (compressed-tensors, vLLM-deployable)

End-to-end integration work to produce a **W4A16 + FP8_BLOCK quantization of
DeepSeek-V4-Flash that actually serves in vLLM**, validated on Hopper SM 9.0
(H200) **and Blackwell SM 12.x** (DGX Spark GB10, RTX PRO 6000). Built by
stacking three in-flight upstream draft PRs and patching the gaps between them.

🤗 Live model: **https://huggingface.co/pastapaul/DeepSeek-V4-Flash-W4A16-FP8** — public, Apache-2.0, no HF token required.

> **Status (2026-05-06):**
> - Live engine on dual DGX Spark TP=2 graphs-ON serving **1 M-token context**, image `vllm-w4a16-dsv4:exp` from `jasl/vllm@ds4-sm120-experimental`.
> - **Production-validated** on three SKUs:
>   - **8× H200 SM 9.0** — calibration + harness baseline (Phase 3b)
>   - **2× DGX Spark GB10 SM 12.1a** — long-context production (Phase 4d/4e)
>   - **2× RTX PRO 6000 Blackwell SM 12.0** — workstation Blackwell (Phase 5)
> - **Headline benchmarks (Spark TP=2, `:exp` build):** GSM8K 95.00 % strict / 94.92 % flex, NIAH 4/4 at 200K-token haystack, mini-suite 10/10, think-max 3/3, decode 12 t/s smoke / 14–15 t/s think-max sustained.
> - **Workspace pre-reservation patch** we filed (issue [#41700](https://github.com/vllm-project/vllm/issues/41700)) **landed upstream** as `jasl/vllm@1d6f5c4` — only `patch_v4_packed_mapping.py` is still applied locally.
> - **`ds4-sm120-experimental` superset** adds split-KV sparse-MLA decode + GB10 fused-MoE config aliases over `ds4-sm120` — gives 1.3–3.4× speedup on long-reasoning cases.
> - **Upstream tracker:** PR [#40991](https://github.com/vllm-project/vllm/pull/40991) (where our Spark validation comment was posted) was closed on 2026-05-06 and replaced by **PR [#41834](https://github.com/vllm-project/vllm/pull/41834)** — *"[New Model][Nvidia] Add SM12x support for DeepSeek V4 Flash with essential fixes"*. The new PR targets a different branch (`codex/ds4-sm120-min-enable`) and we have not yet rebuilt against it; everything in this repo is validated on `jasl/vllm@ds4-sm120-experimental`.

## Quickstart — dual DGX Spark TP=2

Single-file zero-to-serving:

```bash
curl -fsSLO https://raw.githubusercontent.com/pasta-paul/dsv4-flash-w4a16-fp8/main/scripts/bootstrap_dsv4_spark.sh
chmod +x bootstrap_dsv4_spark.sh
./bootstrap_dsv4_spark.sh --head-host spark-a --worker-host spark-b
```

Idempotent — does SSH-reachability check, model pre-cache (no token), QSFP /30 setup, image build via `eugr/spark-vllm-docker` + our DSV4 patches, scp-distribute to the worker, container launch on both nodes (head rank 0 + worker rank 1 `--headless`), waits for `/health=200`. ~30–50 min the first run (mostly the docker build), ~7 min on re-runs with `--skip-build`.

Walk-through with per-flag explanation: [`findings/QUICKSTART_DUAL_SPARK.md`](findings/QUICKSTART_DUAL_SPARK.md). Manual build recipe: [`scripts/Dockerfile.dsv4-spark`](scripts/Dockerfile.dsv4-spark) + [`scripts/patch_v4_packed_mapping.py`](scripts/patch_v4_packed_mapping.py). Don't want to build? Ping us on the [HF model discussions](https://huggingface.co/pastapaul/DeepSeek-V4-Flash-W4A16-FP8/discussions) for the OCI tarball.

## Why this exists

DeepSeek-V4-Flash dropped on April 24, 2026 (284B total / 13B active MoE,
hybrid CSA + HCA attention with mHC hyperconnections, hash-routed experts).
As of May 2, 2026:

- The model code is in **no released `transformers`** — only an open PR (#45643).
- vLLM has **no merged compressed-tensors path** for V4 — only an open Draft PR (#41276).
- LLM Compressor has **no merged V4 quantization path** — only an open PR (#2647).
- The published reference quant ([RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8](https://huggingface.co/RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8))
  uses NVFP4 expert weights, which require SM 10.0+ tcgen05 instructions —
  unavailable on Hopper SM 9.0 and Spark SM 12.1a.
- Intel published [`Intel/DeepSeek-V4-Flash-W4A16-AutoRound`](https://huggingface.co/Intel/DeepSeek-V4-Flash-W4A16-AutoRound)
  (5 days old, 23k+ downloads), but their model card explicitly states *"vLLM
  and SGLang is not supported currently."*

This project produces a W4A16 GPTQ V4-Flash that **serves in vLLM at TP=2
on Hopper SM 9.0 (H200), Blackwell SM 12.1a (DGX Spark GB10), and Blackwell SM 12.0 (RTX PRO 6000) today**,
with attention quantized to FP8_BLOCK (mirroring RedHat's recipe topology, swapping
NVFP4 → W4A16 for SM 9.x / 12.x compatibility).

## What's in this repo

| Path | What |
|---|---|
| **[`scripts/bootstrap_dsv4_spark.sh`](scripts/bootstrap_dsv4_spark.sh)** | **Single-file zero-to-serving script for dual DGX Spark TP=2.** SSH-orchestrated, idempotent, handles every step from network setup through engine boot. |
| [`scripts/Dockerfile.dsv4-spark`](scripts/Dockerfile.dsv4-spark) | The actual production Dockerfile — `jasl/vllm@${VLLM_REF}` + vendored kylesayrs deepseek-ct patch + `packed_modules_mapping` patch. Drop into an `eugr/spark-vllm-docker` checkout. |
| [`scripts/kylesayrs-deepseek-ct.patch`](scripts/kylesayrs-deepseek-ct.patch) | Vendored vLLM patch (kylesayrs PR #41276 work, pre-rebased onto `jasl/vllm@ds4-sm120`). Pinned by content, not SHA — see [findings/kylesayrs-pr-41276-integration.md#sha-rebase-recovery-issue-1-2026-05-08](findings/kylesayrs-pr-41276-integration.md). |
| [`scripts/patch_v4_packed_mapping.py`](scripts/patch_v4_packed_mapping.py) | Local patch that adds `packed_modules_mapping` to `DeepseekV4ForCausalLM`. Still required (kylesayrs PR references but doesn't define it). |
| [`scripts/patch_workspace_prereserve.py`](scripts/patch_workspace_prereserve.py) | **Retired** — landed upstream as `jasl/vllm@1d6f5c4`. Kept for historical reference. |
| [`scripts/serve_spark_tp2.sh`](scripts/serve_spark_tp2.sh) | Per-rank launch helper (call this once per Spark — used by `bootstrap_dsv4_spark.sh`). |
| [`findings/QUICKSTART_DUAL_SPARK.md`](findings/QUICKSTART_DUAL_SPARK.md) | Operator-facing manual quickstart with per-flag explanation. |
| [`findings/spark_tp2_deployment.md`](findings/spark_tp2_deployment.md) | Full Spark validation report — Phases 4b/4c/4d/4e, build provenance, mini-suite + benchmark tables, NIAH evidence, operational constraints. |
| [`findings/rtxpro6000_blackwell_deployment.md`](findings/rtxpro6000_blackwell_deployment.md) | Phase 5 RTX PRO 6000 Blackwell validation. |
| [`findings/spark_tp2_phase4e_*.json`](findings/) + [`spark_tp2_phase4e_probes/`](findings/spark_tp2_phase4e_probes/) | Raw evidence files: per-config sweep, mini-suite results, NIAH probe jsonls, GSM8K samples. |
| [`findings/upstream-issue-marlin-tp-sharding.md`](findings/upstream-issue-marlin-tp-sharding.md) | Root-cause + filed bug for Marlin MoE TP scale-sharding ([vllm-project/vllm#41511](https://github.com/vllm-project/vllm/issues/41511)) — blocks W4A16 MoE under TP > 2. |
| [`findings/kylesayrs-pr-41276-integration.md`](findings/kylesayrs-pr-41276-integration.md) | Integration notes for kylesayrs's V4 attention path PR — 5 documented gaps with our patches. |
| [`findings/phase3b-recovery.md`](findings/phase3b-recovery.md) | The H200 OOM + NCCL-timeout journey for the GPTQ calibration: which env vars, why each. |
| [`patches/`](patches/) | Static patches against upstream — calibration patches (`helpers.py.diff`, `modeling_deepseek_v4.py.diff`) plus the `packed_modules_mapping.diff` for vLLM serving. See [`patches/VERSIONS.md`](patches/VERSIONS.md). |
| `REPORT.md` | Full phase-by-phase mission log (setup → native baseline → dequant → calibration → serve attempts → harness results). |
| `model-card-draft.md` | In-repo mirror of the published HF model card. |

## Validation (Phase 3b on 8× H200, TP=2)

| Tag | Native FP4/FP8 baseline | **W4A16-FP8** |
|---|---|---|
| `chat-smoke quick` | 4/4 | **4/4** |
| `chat-smoke quality` | 4/4 | **4/4** |
| `chat-smoke coding` (8192 max_tokens, t=1.0) | 2/2 | **2/2** |
| `toolcall15` (15 cases × 2 pts) | 23/30 (77%) | **26/30 (87%)** |
| `toolcall15` PASS / PARTIAL / FAIL | 11 / 1 / 4 | **13 / 0 / 2** |

Apples-to-apples on [`jasl/vllm-ds4-sm120-harness`](https://github.com/jasl/vllm-ds4-sm120-harness) HEAD `85aca32`.
TC-11 (Simple Math) was PARTIAL on baseline AND the 16-sample dryrun; **PASS in Phase 3b** — likely from the larger calibration tightening math-reasoning weight quantization.
Remaining 2 toolcall15 fails (TC-06 Multi-Value Extraction, TC-08 Conditional Branching) **also fail on the native FP4/FP8 baseline** — these are V4-Flash model-architecture limits, not quantization defects.

## Validation (Phase 5 on dual RTX PRO 6000 Blackwell, TP=2)

Built on the `ds4-sm120-experimental` superset of `ds4-sm120` (commit `abad5dc71` + vendored kylesayrs deepseek-ct patch (rebased successor of `f910a73a`) + `packed_modules_mapping` patch). Harness HEAD `96785b9`. Single-round chat-smoke / toolcall15; standard lm_eval gates.

| Test | RTX PRO 6000 dual TP=2 | DGX Spark TP=2 (reference) |
|---|---|---|
| `chat-smoke quick` | **4 / 4** | 4 / 4 |
| `toolcall15` | **27 / 30 (90%)** ¹ | 41 / 45 (92%) ² |
| GSM8K 8-shot, strict-match | **95.07% ±0.60%** | 95.45% ±0.57% |
| GSM8K 8-shot, flexible-extract | **94.99% ±0.60%** | 95.37% ±0.58% |
| HumanEval pass@1 (instruct, 0-shot) | **78.05% ±3.24%** | 80.49% ±3.10% |
| Long-context NIAH 75K → 500K (single stream) | **5 / 5 PASS** | (in flight) |
| **Long-context NIAH 256K × 2 concurrent** | ✅ **PASS** (377 s, only +21 s vs single) | stalled 2026-05-04 → fix landed in `jasl@e734ace5` |
| Decode tok/s @ c=1, 1024-in / 1024-out | **47.5** (TPOT mean 20.8 ms) | 14–17 |

¹ Single round, 30 points max. ² 3 thinking-mode rounds × 15 cases = 45 points max — denominators differ but normalized score is the same level.

Full Phase 5 run notes (commit-stack, build flags, two integration issues surfaced — `packed_modules_mapping` still required, FlashInfer JIT mis-parses `12.0a`): [`findings/rtxpro6000_blackwell_deployment.md`](findings/rtxpro6000_blackwell_deployment.md).

## What's in this repo

| Path | What |
|---|---|
| `REPORT.md` | Full phase-by-phase mission log: setup → native baseline → dequant → calibration → vLLM serve attempts → harness results → decisions and pivots. |
| `model-card-draft.md` | Pre-publish draft of the HF model card. The published version lives at https://huggingface.co/pastapaul/DeepSeek-V4-Flash-W4A16-FP8. |
| `findings/upstream-issue-marlin-tp-sharding.md` | Root-cause for a Marlin MoE kernel TP scale-sharding bug discovered during integration. Empirical TP=1/2/8 table, suggested fix location. **Filed upstream as [vllm-project/vllm#41511](https://github.com/vllm-project/vllm/issues/41511) on 2026-05-02.** Blocks all compressed-tensors W4A16 MoE deployments under TP > 2 on every GPU architecture. |
| `findings/kylesayrs-pr-41276-integration.md` | Detailed integration notes for the [neuralmagic/vllm](https://github.com/neuralmagic/vllm) `kylesayrs/deepseek-ct` branch (PR #41276) — 5 documented upstream gaps with our patches. |
| `findings/phase3b-recovery.md` | The Phase 3b OOM + NCCL-timeout journey: what failed, what worked, and the exact env+recipe combination required to GPTQ-calibrate V4-Flash at scale on 8× H200 without crashes. |
| `patches/` | Static patches against upstream (see [`patches/VERSIONS.md`](patches/VERSIONS.md)). Includes calibration patches (`helpers.py.diff`, `modeling_deepseek_v4.py.diff`) and the `packed_modules_mapping.diff` for vLLM serving. |
| `scripts/` | Working scripts in two categories: **calibration** (run on AWS H200 box) and **serve** (run on inference target — H200 here, applicable to DGX Spark with kernel updates). |

## Build for AWS calibration

Calibration runs against the BF16-dequantized base model on 8× H200 with `kylesayrs/transformers-v5` llm-compressor branch.

```bash
# In a clean venv (do NOT share the vLLM serve venv — pip cascades break vLLM's torch+cu13 pin)
pip install git+https://github.com/huggingface/transformers.git@add-deepseek-v4
pip install git+https://github.com/vllm-project/llm-compressor.git@kylesayrs/transformers-v5
pip install --pre 'compressed-tensors>=0.15.1a2'

# Apply calibration-time patches:
patch -p1 -d "$(python -c 'import llmcompressor; print(llmcompressor.__path__[0])')" < patches/helpers.py.diff
patch -p1 -d "$(python -c 'import transformers; print(transformers.__path__[0])')" < patches/modeling_deepseek_v4.py.diff

# /dev/shm must be ≥ 1.8 TiB for 8-rank torchrun on a 543 GB BF16 model
sudo mount -o remount,size=1800G /dev/shm

# Run calibration (recipe: FP8_BLOCK attn + W4A16 GPTQ routed experts)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_TIMEOUT=3600
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_CUDA_ARCH_LIST=9.0a
torchrun --nproc-per-node 8 scripts/quantize_v4_w4a16.py \
    --samples 768 --batch-size 4 \
    --input  /path/to/DeepSeek-V4-Flash-bf16 \
    --output /path/to/DeepSeek-V4-Flash-W4A16-FP8
```

See `findings/phase3b-recovery.md` for **why** each env var is required (every one of them is the fix to a specific failure mode hit during this work).

## Build for vLLM serving

For dual DGX Spark TP=2 the bootstrap script above does this for you. Manual build for any other TP=2 hardware:

Required pieces (stacked):
- **`jasl/vllm@ds4-sm120-experimental`** — current branch with the SM12x DSV4 work + the experimental superset (split-KV decode, GB10 fused-MoE config aliases, tuned MLA graph defaults). Use `ds4-sm120` instead if you want the more-conservative PR-tracked branch (~6 commits behind).
- **Apply** [`scripts/kylesayrs-deepseek-ct.patch`](scripts/kylesayrs-deepseek-ct.patch) — vendored from `neuralmagic/vllm@kylesayrs/deepseek-ct` commit `d09eeb498` (the rebased successor of `f910a73a93`, which was force-pushed out of history — see issue [#1](https://github.com/pasta-paul/dsv4-flash-w4a16-fp8/issues/1)). Pre-rebased onto jasl/vllm so `git apply` works with no 3-way merge.
- **Patch** [`scripts/patch_v4_packed_mapping.py`](scripts/patch_v4_packed_mapping.py) — adds `packed_modules_mapping` to `DeepseekV4ForCausalLM`. Still needed: PR #41276 references this attribute but doesn't define it.
- The workspace pre-reservation patch is **no longer needed** — landed upstream as `jasl/vllm@1d6f5c4` (closed our [issue #41700](https://github.com/vllm-project/vllm/issues/41700)).

```bash
git clone https://github.com/jasl/vllm.git -b ds4-sm120-experimental vllm
cd vllm
git apply --check ../scripts/kylesayrs-deepseek-ct.patch
git am --keep-cr ../scripts/kylesayrs-deepseek-ct.patch
python3 ../scripts/patch_v4_packed_mapping.py vllm/model_executor/models/deepseek_v4.py
pip install -e . --no-build-isolation
```

Production canonical (Phase 4e — 1 M context graphs-ON, single-stream):

```bash
vllm serve pastapaul/DeepSeek-V4-Flash-W4A16-FP8 \
    --served-model-name DSV4-W4A16-FP8 deepseek-ai/DeepSeek-V4-Flash deepseek-v4-flash \
    --tensor-parallel-size 2 \
    --kv-cache-dtype fp8 --block-size 256 \
    --max-model-len 1048576 \
    --max-num-seqs 1 --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.90 \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
    --trust-remote-code
```

For multi-stream + long context, drop `--max-model-len` to `262144` and `--max-num-seqs` to `2` (Phase 4d's previous canonical, mini-suite 10/10 PASS). For maximum decode speed at short context, drop to `--max-model-len 16384 --max-num-seqs 4 --gpu-memory-utilization 0.92` (Phase 4b's recipe — 14–17 t/s decode).

**Two flag gotchas worth flagging:**
- `--served-model-name` takes multiple values per single flag, **not** repeated flags. `--served-model-name A --served-model-name B` silently keeps only the last value. Use space-separated form.
- `--gpu-memory-utilization=0.90` (not 0.92) on the experimental build — prefix-cache + split-KV reservations push past the 0.92 boundary on first boot.

**TP limit:** TP=1 OOMs on a single 141 GB H200. **TP=2 works.** TP ≥ 4 hits the upstream Marlin MoE TP scale-sharding bug ([vllm-project/vllm#41511](https://github.com/vllm-project/vllm/issues/41511)) — until that's fixed, this model is TP=2-only.

## Roadmap status

- [x] **Phase 0** — Setup & verification on AWS p5en.48xlarge
- [x] **Phase 1** — Native FP4/FP8 V4-Flash baseline (jasl harness reference scores)
- [x] **Phase 2** — Dequantize FP4/FP8 → BF16 (flagos)
- [x] **Phase 3a** — Dry-run W4A16-FP8 calibration (16 samples) — toolcall15 25/30
- [x] **Phase 3b** — Full W4A16-FP8 calibration (768 samples) — toolcall15 26/30
- [x] **Phase 4a** — Harness verify on H200 (TP=2): chat-smoke 10/10, toolcall15 26/30
- [x] **Phase 4b** — DGX Spark TP=2 deployment ([`findings/spark_tp2_deployment.md`](findings/spark_tp2_deployment.md)): workspace lock diagnosed + patched, harness 4/4 + 18/18 + 41/45, GSM8K 95.37%, HumanEval pass@1 80.49%, **64K-context retest 9/10 PASS** (think-max budget-isolation confirmed)
- [x] **Phase 4d** — Long-context graphs-ON sweep on `jasl@0789bc9` (workspace patch upstreamed as `1d6f5c4` + SM12x DeepGEMM fix `0789bc9` unblocks graphs-ON at long context). NIAH 4/4 retrieval at **128K, 256K×1, 256K×2** contexts; mini-suite **10/10 PASS** at canonical 256K×2 graphs-ON (8.92 t/s decode).
- [x] **Phase 4e** — Production canonical at **1 M context graphs-ON** on `ds4-sm120-experimental` (split-KV decode + tuned MLA graph defaults). NIAH 4/4 at 200K-token haystack on 256K × 2 predecessor canonical, mini-suite **10/10 PASS** with **1.3–3.4× speedup** vs `:sm12fix`. **GSM8K 95.00 % strict / 94.92 % flexible** preserved on the new image. Think-max **3/3 PASS** at 14–15 t/s. Live engine serves 1 M × 1 graphs-ON. Quickstart: [`findings/QUICKSTART_DUAL_SPARK.md`](findings/QUICKSTART_DUAL_SPARK.md). Full report: [`findings/spark_tp2_deployment.md` Phase 4e](findings/spark_tp2_deployment.md#phase-4e--production-canonical-at-1-m-context-on-ds4-sm120-experimental-2026-05-06).
- [x] **Phase 5** — Dual RTX PRO 6000 Blackwell (SM 12.0) on `ds4-sm120-experimental` tip `abad5dc71`. Toolcall15 27/30 (90%, single round), GSM8K 95.07%, HumanEval pass@1 78.05%. NIAH single-stream PASS at 75K / 128K / 256K / 500K, **256K × 2 concurrent PASS** (377 s vs 356 s single — confirms `e734ace5` fixes the Spark stall on Blackwell). Decode 47–48 tok/s per stream at c=1 (3× Spark), 84 tok/s aggregate at c=2. Full report: [`findings/rtxpro6000_blackwell_deployment.md`](findings/rtxpro6000_blackwell_deployment.md).
- [x] **Public HF release** at [`pastapaul/DeepSeek-V4-Flash-W4A16-FP8`](https://huggingface.co/pastapaul/DeepSeek-V4-Flash-W4A16-FP8)
- [x] **Upstream contributions** — workspace allocator bug + patch ([`vllm-project/vllm#41700`](https://github.com/vllm-project/vllm/issues/41700) — landed as `jasl/vllm@1d6f5c4`), Marlin TP scale-sharding ([`#41511`](https://github.com/vllm-project/vllm/issues/41511))

### Standard benchmarks

| Benchmark | Setting | 8× H200 (older vllm) | **2× DGX Spark TP=2** | **2× RTX PRO 6000 TP=2** |
|---|---|---|---|---|
| GSM8K | 8-shot, flexible-extract | 92.87% ±0.71% | **95.37% ±0.58%** | **94.99% ±0.60%** |
| GSM8K | 8-shot, strict-match | 42.61% (regex artifact) | 95.45% ±0.57% | **95.07% ±0.60%** |
| HumanEval | pass@1 (instruct, 0-shot, `--confirm_run_unsafe_code`) | 54.27% ±3.9% ¹ | **80.49% ±3.10%** | **78.05% ±3.24%** |
| Decode tok/s @ c=1 (1024-in / 1024-out) | TTFT mean / TPOT mean | — | 14–17 / — | **47.5 / 20.8 ms** |
| MMLU | 5-shot | 87.27% ±0.27% | (pending) | (pending) |

Results updated to the [HF model card](https://huggingface.co/pastapaul/DeepSeek-V4-Flash-W4A16-FP8) as each lands.

¹ The H200 numbers are from harness HEAD `85aca32` and `jasl/vllm@428e08e` — an **older vllm build**. The Spark and RTX PRO 6000 numbers are on today's `ds4-sm120-experimental` tip. Treat the H200 ↔ Blackwell deltas as informational, not as a "same software, different hardware" benchmark; the valid same-software comparison is **Spark ↔ RTX PRO 6000** (effectively at parity within stderr).

## Credits

- [@jasl](https://github.com/jasl) — DeepSeek-V4 vLLM SM12x base support (PR #40991)
- [@kylesayrs](https://github.com/kylesayrs) — compressed-tensors V4 attention path (PR #41276)
- [@aabbccddwasd](https://github.com/aabbccddwasd) — indexer KV cache layout fix
- [@bbbearxyz](https://github.com/bbbearxyz) — SM12x Triton fallback kernels
- [@wuwenthink](https://github.com/wuwenthink) — SM12x harness validation
- [`RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8`](https://huggingface.co/RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8) — published reference for V4 mixed-precision attention topology

Apache-2.0, inherited from the base model.
