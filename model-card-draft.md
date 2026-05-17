---
license: apache-2.0
language:
- en
- zh
library_name: vllm
tags:
- deepseek_v4
- mixture-of-experts
- moe
- compressed-tensors
- w4a16
- gptq
- fp8-block
- deepseek
- deepseek-v4
base_model: deepseek-ai/DeepSeek-V4-Flash
---

> **Note:** This is the in-repo draft of the HF model card. The published version lives at https://huggingface.co/pastapaul/DeepSeek-V4-Flash-W4A16-FP8 and should be the source of truth — this file mirrors it for offline review and PR diffing.

# DeepSeek-V4-Flash W4A16-FP8

Mixed-precision quantization of [`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) for **vLLM tensor-parallel deployment at TP=2**. Validated end-to-end on three SKUs:

- **8× H200 (SM 9.0)** — Hopper datacenter
- **2× DGX Spark / GB10 (SM 12.1)** — Blackwell SoC
- **2× RTX PRO 6000 Blackwell Server (SM 12.0)** — Blackwell workstation

making this load cleanly on consumer Blackwell.

Naming mirrors [`RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8`](https://huggingface.co/RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8) — their NVFP4 experts → our W4A16 experts, attention block in both is FP8_BLOCK.

## Quantization scheme

| Component | Format | Method |
| --- | --- | --- |
| Routed experts (256 × 43 layers) | W4A16 INT4, group_size=128 | GPTQ, `dampening_frac=0.1` |
| Attention projections (q/kv/o, compressor, indexer) | FP8_BLOCK 128×128 | Data-free |
| Shared experts | BF16 | Excluded (kylesayrs PR #41276 incompatibility) |
| Embeddings, lm_head, hc_head | BF16 | Excluded |

## Architecture

| Property | Value |
| --- | --- |
| Total parameters | 671 B (32 B activated) |
| Decoder layers | 43 |
| Routed experts / layer | 256 (top-K = 8 hash-routed) |
| Hidden size | 4096 |
| Quantized size | ~143 GB (vs ~543 GB BF16) |
| Compression ratio | ~3.8× |

## Inference (vLLM)

```bash
vllm serve pastapaul/DeepSeek-V4-Flash-W4A16-FP8 \
  --served-model-name DSV4-W4A16-FP8 \
  --tensor-parallel-size 2 \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --max-model-len 16384 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.92 \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --trust-remote-code
```

**Required env vars at runtime (SM 12.x sparse-MLA path):** set `VLLM_TRITON_MLA_SPARSE=1` and `VLLM_TRITON_MLA_SPARSE_HEAD_BLOCK_SIZE=4` in the container or shell that runs `vllm serve`. Without `_HEAD_BLOCK_SIZE=4` the sparse-MLA Triton kernel can crash during warmup with `RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered` in `_dequantize_and_gather_k_kernel` — the kernel falls back to a default block size that doesn't match V4-Flash's head dim. The full env block (NCCL, TileLang, HF cache flags) is at [QUICKSTART_DUAL_SPARK.md §4](https://github.com/pasta-paul/dsv4-flash-w4a16-fp8/blob/main/findings/QUICKSTART_DUAL_SPARK.md#4-launch--head--worker).

**Tensor parallelism**: TP=2 is the only validated configuration. TP=1 OOMs on a single 141 GB H200; TP≥4 hits an upstream W4A16 MoE scale-sharding bug ([vllm-project/vllm#41511](https://github.com/vllm-project/vllm/issues/41511)).

**Required vLLM build**: This model does not load on vanilla vLLM. The exact toolchain — `jasl/vllm@ds4-sm120` (or `ds4-sm120-experimental` for the bleeding edge) + the vendored [`scripts/kylesayrs-deepseek-ct.patch`](https://github.com/pasta-paul/dsv4-flash-w4a16-fp8/blob/main/scripts/kylesayrs-deepseek-ct.patch) (kylesayrs PR #41276, content-pinned rebased successor of `f910a73a93` which was force-pushed out of upstream history; see [issue #1](https://github.com/pasta-paul/dsv4-flash-w4a16-fp8/issues/1)) + `packed_modules_mapping` patch — is in the [reproduction repo](https://github.com/pasta-paul/dsv4-flash-w4a16-fp8). The single-file bootstrap script `scripts/bootstrap_dsv4_spark.sh` does the whole stack zero-to-serving on dual DGX Spark. For SM 12.x hardware (DGX Spark / GB10 / RTX PRO 6000 / RTX 50-series), the workspace pre-reservation patch landed upstream as `jasl/vllm@1d6f5c4` (was [`vllm-project/vllm#41700`](https://github.com/vllm-project/vllm/issues/41700)); check it out instead of carrying the local patch.

**Blackwell sm_120 note (RTX PRO 6000):** vLLM's FlashInfer-based top-p / top-k sampler JIT mis-parses the `TORCH_CUDA_ARCH_LIST=12.0a` arch token and raises `RuntimeError: FlashInfer requires GPUs with sm75 or higher` (the GPU is sm_120 — way above sm_75; the parser just doesn't recognize the `12.0a` token). Set `VLLM_USE_FLASHINFER_SAMPLER=0` to fall back to the PyTorch-native sampler.

## Evaluation

Validated on [`jasl/vllm-ds4-sm120-harness`](https://github.com/jasl/vllm-ds4-sm120-harness). H200 numbers are at HEAD `85aca32` (older `jasl/vllm@428e08e`); Spark and RTX PRO 6000 numbers are at HEAD `96785b9` on the today-current `ds4-sm120-experimental` tip — graphs ON, no `--enforce-eager`.

| Test | Native FP4/FP8 (8× H200) | **W4A16-FP8 (8× H200)** | **W4A16-FP8 (2× DGX Spark TP=2)** | **W4A16-FP8 (2× RTX PRO 6000 TP=2)** |
| --- | --- | --- | --- | --- |
| `chat-smoke quick` | 4/4 | **4/4** | **4/4** | **4/4** |
| `chat-smoke quality` | 4/4 | **4/4** | included in generation matrix below | included in generation matrix below |
| `chat-smoke coding` | 2/2 | **2/2** | included in generation matrix below | included in generation matrix below |
| `generation` (18 prompts × non-thinking) | — | — | **18/18 PASS** | **18 / 18 invocations clean** |
| `generation` (18 prompts × think-high) | — | — | **17/18 PASS** | **54 / 54 invocations clean** ⁰ |
| `generation` (18 prompts × think-max @ 32K) | — | — | 9/18 → **9/10** at 64K rerun | **54 / 54 invocations clean** ⁰ |
| `toolcall15` | 23/30 (77%) | 26/30 (87%) ¹ | **41/45 (92%)** ¹ | **27/30 (90%)** ² |
| Long-context NIAH (75K → 256K single) | — | — | 4/4 retrieval | **4/4 retrieval** |
| **Long-context NIAH 256K × 2 concurrent** | — | — | stalled 2026-05-04 → fix in `jasl@e734ace5` | ✅ **PASS** (377 s vs 356 s single) |
| Long-context NIAH 500K × 1 | — | — | (in flight) | ✅ **PASS** (1231 s) |
| Workspace-lock errors | 0 | 0 | 0 over 100+ requests, 5 h+ uptime | 0 |

⁰ The generation-matrix runs on RTX PRO 6000 are 18 prompts × 3 rounds = 54 invocations per mode; this harness HEAD does not auto-pass/fail them, but all 126 completed cleanly with `finish_reason=stop`. Thinking-mode failures of the form Spark saw at 32K budget are not reproduced here because we ran with `--max-model-len=16384` and all prompts fit within budget.

¹ Toolcall15 on Spark is scored across 3 thinking modes (45 cases); H200 baseline was single-mode (30 cases). Score normalized to %. ² Toolcall15 on RTX PRO 6000 here is single-round (30 cases); same pattern of failures (TC-06 Multi-Value Extraction fail, TC-07 Search-Read-Act partial).

> **Comparison caveat:** the H200 numbers come from an older vllm build (harness HEAD `85aca32`, `jasl/vllm@428e08e`). Spark and RTX PRO 6000 numbers are on today's `ds4-sm120-experimental` tip. Treat the H200 ↔ Blackwell deltas as informational, not as a "same software, different hardware" benchmark; the valid same-software comparison is **Spark ↔ RTX PRO 6000**.

### Standard benchmarks

| Benchmark | Setting | 8× H200 (older vllm) | **2× DGX Spark TP=2 (graph mode)** | **2× RTX PRO 6000 TP=2 (graph mode)** |
| --- | --- | --- | --- | --- |
| GSM8K | 8-shot, flexible-extract | 92.87% ±0.71% | **95.37% ±0.58%** | **94.99% ±0.60%** |
| GSM8K | strict-match | 42.61% (chat-format artifact) | 95.45% ±0.57% | **95.07% ±0.60%** |
| MMLU | 5-shot | 87.27% ±0.27% | (in flight) | (pending) |
| HumanEval | 0-shot pass@1 (instruct, `--confirm_run_unsafe_code`) | 54.27% ±3.9% ³ | **80.49% ±3.10%** ⁴ | **78.05% ±3.24%** |

³ HumanEval pass@1 on H200 is depressed by chat-format extraction; coding capability is better captured by the generation matrix and toolcall15 above.
⁴ The Spark and RTX PRO 6000 HumanEval runs use strict pass@1 with code execution enabled (`--confirm_run_unsafe_code`); the H200 number on this card was scored via regex extraction (which under-counts valid generations). Methodology difference accounts for most of the +20–26 pp delta — quality is preserved.

### Throughput

| Hardware | Mode | Decode | Notes |
| --- | --- | --- | --- |
| 8× H200 TP=2 | graph | — | not measured under harness |
| **2× Spark TP=2** | graph | **14–17 tok/s** | canonical recipe, multi-seq stable |
| 2× Spark TP=2 | eager | 3–4 tok/s | only required without workspace patch |
| **2× RTX PRO 6000 TP=2** | graph | **47–48 tok/s @ c=1, 84 tok/s @ c=2** | TPOT mean 20.8 ms (p99 21.7 ms) at c=1, scales 1.77× to c=2 |

### RTX PRO 6000 — `vllm bench serve` detail

| Concurrency | In / Out | Duration | TTFT mean / p99 | TPOT mean / p99 | Output tok/s |
| --- | --- | --- | --- | --- | --- |
| 1 | 1024 / 1024 | 430.9 s | 237 ms / 711 ms | 20.8 ms / 21.7 ms | 47.5 |
| 2 | 2048 / 512  | 121.9 s | 1096 ms / 1900 ms | 21.7 ms / 23.0 ms | 84.0 |

Per-stream decode rate is rock-stable across concurrency (TPOT mean stays at 21 ms, p99 only 23 ms). Aggregate input+output throughput at c=2 reaches 420 tok/s.

### Note on think-max reasoning failures (Spark only)

The 9 think-max failures on Spark at 16K context + 32K output budget are not a model-quality regression — they are output-ceiling truncations. With `--max-model-len 16384` and a typical ~1–2K prompt, the actual output ceiling is ~14–15K, regardless of the requested 32K. The deepseek_v4 reasoning parser dumps unclosed `<think>` blocks into `reasoning_content`, leaving `content` empty. To run think-max on these prompts, scale both `--max-model-len ≥ 65536` and `max_tokens ≥ 64000` together. Non-thinking and think-high modes are unaffected.

**Empirical confirmation (2026-05-05, Spark):** the same 10 cases re-run at `--max-model-len=65536`, `--max-num-seqs=4`, `max_tokens=64000` produce **9 / 10 PASS** with reasoning + content lengths well past the original 32K cap. Decode rates remain in the canonical 14–17 t/s envelope at 4× the context window. Raw evidence: [`findings/spark_tp2_64k_retest_results.jsonl`](https://github.com/pasta-paul/dsv4-flash-w4a16-fp8/blob/main/findings/spark_tp2_64k_retest_results.jsonl).

### Oracle comparison vs B200 TP=2 reference (Spark)

See published HF model card. RTX PRO 6000 oracle-compare deferred for this run.

## Calibration

| Property | Value |
| --- | --- |
| Dataset | `HuggingFaceH4/ultrachat_200k` (V4 chat template) |
| Samples | 768 |
| Max sequence length | 512 |
| Per-rank batch size | 4 |
| Hardware | 8× NVIDIA H200, `p5en.48xlarge` |
| Walltime | ~14 hours |

### Required environment

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export TORCH_NCCL_BLOCKING_WAIT=0
export NCCL_TIMEOUT=3600
export TORCH_CUDA_ARCH_LIST=9.0a
sudo mount -o remount,size=1800G /dev/shm
```

`expandable_segments` is **calibration-only** — must not be set during vLLM serving.

### Recipe

```python
from llmcompressor.modifiers.quantization import GPTQModifier
from compressed_tensors.quantization.quant_scheme import FP8_BLOCK, W4A16, QuantizationScheme

recipe = GPTQModifier(
    config_groups={
        "attention": QuantizationScheme(
            targets=[
                r"re:.*self_attn\.(q_a_proj|q_b_proj|kv_proj|o_a_proj|o_b_proj)$",
                r"re:.*self_attn\.compressor\.(gate_proj|kv_proj)$",
                r"re:.*self_attn\.compressor\.indexer\.(gate_proj|kv_proj|q_b_proj|weights_proj)$",
            ],
            **FP8_BLOCK,
        ),
        "experts": QuantizationScheme(
            targets=[r"re:.*mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$"],
            **W4A16,
        ),
    },
    ignore=["lm_head"],
    offload_hessians=True,
    dampening_frac=0.1,
)

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=512,
    num_calibration_samples=768,
    sequential_targets=["DeepseekV4DecoderLayer"],
    batch_size=4,
)
```

## Known issues

- `lm_head` excluded from quantization (BF16) — including it produces dequantization mismatches with the kylesayrs PR loader.
- `shared_experts` excluded (BF16) — including them triggers `NotImplementedError("DeepSeekV4 requires FP8 attention quantization")` on shared_expert routing.
- TP > 2 is BLOCKED by [vllm-project/vllm#41511](https://github.com/vllm-project/vllm/issues/41511) (W4A16 MoE scale-sharding).
- SM 12.x deployment requires the workspace pre-reservation, but the patch landed upstream as `jasl/vllm@1d6f5c4` so just check out a recent enough `ds4-sm120` tip rather than carrying the local patch.
- **`packed_modules_mapping` patch is still required** as of `ds4-sm120-experimental@abad5dc71` (2026-05-05) — the kylesayrs deepseek-ct patch does not add the class attribute. Drop-in patch in [`patches/packed_modules_mapping.diff`](https://github.com/pasta-paul/dsv4-flash-w4a16-fp8/blob/main/patches/packed_modules_mapping.diff). Note `gate_up_proj` must map to `["w1", "w3"]` (not `gate_proj/up_proj`) to match the recipe ignore list naming for shared experts.
- **FlashInfer JIT** mis-parses `TORCH_CUDA_ARCH_LIST=12.0a` on RTX PRO 6000 sm_120 — set `VLLM_USE_FLASHINFER_SAMPLER=0` to fall back to PyTorch-native sampler.
- **Runtime env var requirement:** `VLLM_TRITON_MLA_SPARSE_HEAD_BLOCK_SIZE=4` must be set on the SM 12.x sparse-MLA path or kernel warmup crashes with an illegal memory access in `_dequantize_and_gather_k_kernel`. See Inference section above for the full env block reference.

## Reproduction

Full toolchain, scripts, and patches: [pasta-paul/dsv4-flash-w4a16-fp8](https://github.com/pasta-paul/dsv4-flash-w4a16-fp8)

Built with:
- `vllm-project/llm-compressor` `kylesayrs/transformers-v5` (PR #2647), commit `a308bc0e`
- `huggingface/transformers` `add-deepseek-v4` (PR #45643), `5.8.0.dev0`
- `compressed-tensors` `0.15.1.a20260428`
- PyTorch `2.11.0+cu130` (calibration on H200) / `2.11.0+cu128` (serving on RTX PRO 6000)
- vLLM (calibration verify, 2026-05-02): `jasl/vllm@428e08e` + `neuralmagic/kylesayrs/deepseek-ct@f910a73a` cherry-picked + `packed_modules_mapping` patch + workspace pre-reservation patch (commit `0ac3de079`). The SHA `f910a73a` was later force-pushed out of upstream history on ~2026-05-08; current builds apply the content-pinned rebased successor `d09eeb498` via the vendored [`scripts/kylesayrs-deepseek-ct.patch`](https://github.com/pasta-paul/dsv4-flash-w4a16-fp8/blob/main/scripts/kylesayrs-deepseek-ct.patch).
- vLLM (RTX PRO 6000 serving, 2026-05-05): `jasl/vllm@ds4-sm120-experimental@abad5dc71` + the vendored kylesayrs-deepseek-ct.patch (content-pinned rebased successor of `f910a73a`) + `packed_modules_mapping` patch (workspace patch now upstream as `1d6f5c4`)

## Acknowledgements

- [@jasl](https://github.com/jasl) — DeepSeek-V4 vLLM SM12x base support (originally PR [#40991](https://github.com/vllm-project/vllm/pull/40991), closed 2026-05-06; current upstream tracker is PR [#41834](https://github.com/vllm-project/vllm/pull/41834)). Also `e734ace5` memory-pressure-release fix that resolved the Blackwell 256K×2 stall.
- [@kylesayrs](https://github.com/kylesayrs) — compressed-tensors V4 attention path (PR #41276)
- [@aabbccddwasd](https://github.com/aabbccddwasd) — indexer KV cache layout fix
- [@bbbearxyz](https://github.com/bbbearxyz) — SM12x Triton fallback kernels
- [`RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8`](https://huggingface.co/RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8) — published reference for V4 mixed-precision attention topology

## License

Apache 2.0 (inherited from base model)
