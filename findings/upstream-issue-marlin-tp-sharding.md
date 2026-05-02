# Upstream Issue: compressed-tensors W4A16 MoE TP scale-sharding bug

**Filed as [vllm-project/vllm#41511](https://github.com/vllm-project/vllm/issues/41511) on May 2, 2026.**

This document is the source of the issue body and the follow-up PR comment
template for [vllm-project/vllm#40991](https://github.com/vllm-project/vllm/pull/40991).

> **Note on cross-repo references:** GitHub auto-links bare `#NNNN` numbers to the *current* repo. When pasting this body into a `vllm-project/vllm` issue, references to LLM Compressor PRs must be fully qualified (`vllm-project/llm-compressor#2647`) or written as a markdown link, otherwise GitHub will mis-link to an unrelated `vllm-project/vllm` issue with the same number.

---

## Title

`[Bug]: compressed-tensors W4A16 MoE: weight_scale not sharded along K under tensor parallelism, causing kernel to compute wrong group_size`

---

## Body

### Your current environment

- vLLM commit: jasl/vllm@ds4-sm120 base `428e08e` with kylesayrs PR [#41276](https://github.com/vllm-project/vllm/pull/41276) commit `f910a73a` cherry-picked. The 4 intervening commits between `06e11f8` and current jasl HEAD `68901da` (`2148a6e` FP8 einsum full-group, `2e06cbd` SM12x sparse MLA decode, `68901da` SM120 packed FP8 indexer cache) do **not** touch `vllm/model_executor/layers/quantization/compressed_tensors/`, so the bug semantics described below are bit-identical on `68901da` by construction.
- PyTorch: 2.11.0+cu130
- CUDA: 13.0
- GPU: 8× NVIDIA H200 (SM 9.0), 141 GB HBM3e each
- Model: pastapaul/DeepSeek-V4-Flash-W4A16-FP8 (FP8_BLOCK attention + W4A16 GPTQ routed experts, group_size=128, produced via [llm-compressor PR vllm-project/llm-compressor#2647](https://github.com/vllm-project/llm-compressor/pull/2647) branch `kylesayrs/transformers-v5`, attention quant mirrors RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8 topology)

### 🐛 Describe the bug

When serving a compressed-tensors W4A16 MoE model under tensor parallelism, the Marlin MoE kernel `moe_wna16_marlin_gemm` raises:

```
RuntimeError: Invalid thread config: thread_m_blocks = 4, thread_k = -1,
thread_n = -1, num_threads = -1 for MKN = [49152, 256, 4096], num_bits = 4,
group_size = 16, has_act_order = 0, is_k_full = 0, has_zp = 0,
is_zp_float = 0, max_shared_mem = 232448
```

### Empirical observations across TP sizes

| TP | K_per_rank | num_groups (from on-disk scale) | kernel-derived group_size | Observed result |
|----|------------|---------------------------------|---------------------------|-----------------|
| 1  | 2048       | 16                              | **128** (correct)         | OOMs on single 141GB H200 due to model size, not bug-relevant |
| 2  | 1024       | 16                              | **64** (incorrect, but template exists) | Serves cleanly. Smoke + full harness gates pass: chat-smoke quick 4/4, quality 4/4, coding 2/2. Output mathematically correct because per-group scale values are constant within each 128-element block — reading at finer granularity (64-wide) returns the same scale value twice |
| 8  | 256        | 16                              | **16** (incorrect, no template) | `Invalid thread config` crash — every entry in `large_batch_thread_configs[]` fails `thread_k >= MIN_THREAD_K = 128` |

The TP=2 case is the smoking gun: the kernel believes `group_size=64`, but the model produces correct output anyway because reading the same scale tensor at finer granularity than its actual quantization grouping (128) is a no-op. Each scale value is just repeated across the smaller 64-wide windows. **This proves the saved scales are correct on disk** and the bug is purely in how vLLM passes them to the kernel under TP.

### Root cause

`weight_scale` is **not sharded along K** when the routed expert weight is sharded along K under TP. `num_groups` is derived at runtime from the unsharded scale's shape:

```
group_size = K_per_rank / num_groups = K_per_rank / 16
```

Yields:
- TP=1: 2048 / 16 = 128  ← correct
- TP=2: 1024 / 16 = 64  ← off by 2× but kernel template exists, no crash
- TP=4: 512 / 16 = 32   ← off by 4×, may or may not have template
- TP=8: 256 / 16 = 16   ← off by 8×, no template, hard crash

The weight tensor is correctly sharded:
- Full `down_proj.weight`: `(out_dim, K=2048)`
- Per-rank shard at TP=8: `(out_dim, K_shard=256)`

The scale tensor is **not** sharded along K:
- Saved `down_proj.weight_scale`: `(out_dim, num_groups=16)`
- Per-rank scale: `(out_dim, num_groups=16)` ← unchanged across ranks

When `compressed_tensors_moe_wna16_marlin` calls Marlin, it passes the unsharded scale alongside the sharded weight. Marlin's `moe_wna16_marlin_gemm` validation derives `group_size = size_k / num_groups` where `size_k` is the per-rank K. At TP=8, `group_size` collapses to 16, which is below `MIN_THREAD_K = 128`, so `is_valid_config` rejects every entry in `large_batch_thread_configs[]` and the kernel selector returns `MarlinDefault`, yielding the cryptic `thread_k=-1` error.

### Reproduction

Recipe (mirrors RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8 attention group + W4A16 GPTQ on routed experts):

```python
from llmcompressor.modifiers.quantization import (
    GPTQModifier, QuantizationModifier
)
from compressed_tensors.quantization import (
    QuantizationScheme, QuantizationArgs
)

# Group 1: FP8_BLOCK attention (data-free, matches RedHat exactly)
fp8_attn = QuantizationScheme(
    targets=[
        "re:model.*attn.*(wkv|wo_a|wo_b|wq_a|wq_b)$",
        "re:model.*attn\\.compressor.*(wgate|wkv)$",
    ],
    weights=QuantizationArgs(
        num_bits=8, type="float", symmetric=True,
        strategy="block", block_structure=[128, 128],
        observer="memoryless_minmax",
    ),
    input_activations=QuantizationArgs(
        num_bits=8, type="float", symmetric=True,
        group_size=128, strategy="group", dynamic=True,
    ),
)

# Group 2: W4A16 GPTQ on routed experts (group_size=128)
w4a16_experts = QuantizationScheme(
    targets=["re:model.*mlp.*(gate|up|down)_proj$"],
    weights=QuantizationArgs(
        num_bits=4, type="int", symmetric=True,
        strategy="group", group_size=128,
    ),
)

recipe = [
    QuantizationModifier(
        config_groups={"attention": fp8_attn},
        ignore=["lm_head", "re:.*shared_experts.*"],
    ),
    GPTQModifier(
        config_groups={"experts": w4a16_experts},
        ignore=["lm_head", "re:.*shared_experts.*", "re:.*attn.*"],
        dampening_frac=0.1,
        sequential_targets=["DeepseekV4DecoderLayer"],
    ),
]
```

Serve commands to reproduce:

```bash
# Crashes (confirmed)
vllm serve <model_dir> --tensor-parallel-size 8 \
  --kv-cache-dtype fp8 --block-size 256 --max-model-len 16384 \
  --gpu-memory-utilization 0.85 \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --reasoning-parser deepseek_v4

# Works (confirmed: smoke pass + full harness gates pass at TP=2)
vllm serve <model_dir> --tensor-parallel-size 2 \
  --kv-cache-dtype fp8 --block-size 256 --max-model-len 16384 \
  --gpu-memory-utilization 0.85 \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --reasoning-parser deepseek_v4
```

### Verification of saved scale shape (proof scales are correct on disk)

```python
from safetensors import safe_open
with safe_open("model-00001-of-00004.safetensors", framework="pt") as sf:
    t = sf.get_tensor(
        "model.layers.0.ffn.experts.0.down_proj.weight_scale"
    )
    print(t.shape)  # → (4096, 16) — group_size=128 over K=2048 = 16 groups ✓
```

### Suggested fix location

`vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16_marlin.py` around lines 215-225 (the `process_weights_after_loading` path, where weight is sharded along K but scale is left at full-tensor group count).

In the weight loading / `create_weights` path, when the routed expert weight is sharded along K at TP boundaries, the `weight_scale` parameter must also shard along K:

```python
# Sharding logic (current code is missing this for weight_scale):
num_groups_per_shard = num_groups // tp_size
scale_shard = weight_scale[:, k_group_start:k_group_end]
```

For comparison, the FP8 W8A8 MoE path at `compressed_tensors_moe_w8a8_fp8.py` correctly shards `weight_scale_inv` along K under TP for block-quantized scales. The W4A16 grouped path appears to skip the equivalent sharding.

### Impact

This blocks **all** compressed-tensors W4A16 MoE deployments under TP > 2 on every GPU architecture, not just SM12x. Specifically affects:

- `Intel/DeepSeek-V4-Flash-W4A16-AutoRound` (Intel's published model card notes "vLLM and SGLang is not supported currently" — same root cause class)
- All future GPTQ/AWQ W4A16 quantizations of MoE models targeting vLLM TP>2

Note that TP=2 deployments work because the kernel's wrong-believed `group_size=64` happens to have a template instantiation; output is mathematically correct because per-group scales are constant within each 128-block subdivision. So TP=2 is a viable workaround until the fix lands, but the kernel dispatch is sub-optimal (smaller-than-necessary tiles).

### Related issues / PRs

- #38022 / PR #38222: separate Marlin MoE shape-alignment bug for non-128-aligned hidden dims (different root cause; same symptom class).
- PR #41276 (kylesayrs compressed-tensors V4 support): handles attention scale_fmt and Linear-wrap fixes; does not touch MoE TP scale sharding.
- PR #40991 (jasl SM12x V4): adds SM12x kernel fallbacks; does not touch compressed-tensors MoE quantization paths. As noted above, none of jasl's commits between `06e11f8` and current HEAD `68901da` touch `compressed_tensors_moe_wna16_marlin.py`, so the bug is bit-identical on current HEAD by construction.

### Repro artifacts

Full integration findings, scripts, and reproduction details:
https://github.com/pasta-paul/dsv4-flash-awq-w4a16

Specifically:
- Recipe: `scripts/quantize_v4_w4a16.py`
- Diagnostic that confirmed scales-correct-on-disk: `scripts/inspect_scales.py`
- 5 prior upstream gaps documented: `findings/kylesayrs-pr-41276-integration.md`
- This bug detail: `findings/upstream-issue-marlin-tp-sharding.md`

### Before submitting a new issue

- [x] Searched for relevant issues
- [x] Verified on jasl/ds4-sm120 base `428e08e` + kylesayrs PR #41276 cherry-pick (`f910a73a`); intervening commits to current HEAD `68901da` do not touch the affected code path
- [x] Confirmed independent of #38022 (different shape constraint, same kernel symptom class)
- [x] Saved checkpoint scales verified correct on disk
- [x] TP=2 confirms output is mathematically correct (full harness gates pass: chat-smoke quick 4/4, quality 4/4, coding 2/2)
- [x] TP=8 confirms bug fires deterministically

---

## Follow-up PR comment for #40991

To be posted on https://github.com/vllm-project/vllm/pull/40991 after filing the issue:

> Filed [#41511](https://github.com/vllm-project/vllm/issues/41511) (compressed-tensors W4A16 MoE TP scale-sharding) as a separate issue since it's orthogonal to the SM12x sparse-MLA / indexer work in this PR. Hits on H200 SM90 too — not SM12x-specific. Verified on `428e08e` + PR #41276 cherry-pick; the 4 intervening commits to current HEAD `68901da` do not touch `compressed_tensors_moe_wna16_marlin.py`, so the bug is bit-identical on current HEAD by construction.
>
> Briefly: kylesayrs PR #41276 cherry-picked into ds4-sm120 cleanly, structural integration through `compressed_tensors_moe_wna16_marlin` works (model loads, weights resolve, packed_modules_mapping resolves fused_wqa_wkv/fused_wkv_wgate), but Marlin sees wrong `group_size` at TP>2 because `weight_scale` isn't sharded along K alongside the weight. TP=2 works (kernel's wrong-believed group_size=64 has a template, output mathematically correct, full harness coding 2/2 PASS). TP=8 crashes with `Invalid thread config`.
>
> Per @wuwenthink's TP=2 SM120 harness report (May 1), chat-smoke coding 0/2 — our H200 TP=2 W4A16 with the same harness defaults got coding 2/2 PASS. So the SM12x coding 0/2 is reproducibly an SM12x-specific issue (kernel correctness, not reasoning-token-exhaustion as I'd previously speculated).
>
> Full integration writeup: pasta-paul/dsv4-flash-awq-w4a16 — 5 distinct upstream gaps documented during this work, plus the Marlin TP fix surface, plus the published model at pastapaul/DeepSeek-V4-Flash-W4A16-FP8.
>
> /cc @jasl @kylesayrs @dsikka

---

## Crediting other contributions

When the fix lands, credit:
- @jasl for SM12x-V4 base support (PR #40991)
- @kylesayrs for compressed-tensors V4 attention path (PR #41276)
- @bbbearxyz for SM12x Triton fallback kernels
- @aabbccddwasd for indexer KV cache layout fix
- @wuwenthink for SM12x harness validation

The repo's `findings/kylesayrs-pr-41276-integration.md` already documents all upstream PRs we built on top of.
