# Upstream Issue: compressed-tensors W4A16 MoE TP scale-sharding bug

This document is the paste-ready body for filing at:
https://github.com/vllm-project/vllm/issues/new/choose → Bug Report

File this issue once Stage 2 of the rebase plan has confirmed the bug
reproduces on the current jasl/ds4-sm120 HEAD `68901da`.

---

## Title

`[Bug]: compressed-tensors W4A16 MoE: weight_scale not sharded along K under tensor parallelism, causing kernel to compute wrong group_size`

---

## Body

### Your current environment

- vLLM commit: jasl/vllm@ds4-sm120 HEAD `68901da` (PR #40991) with kylesayrs PR #41276 commit `f910a73a` cherry-picked
- PyTorch: 2.11.0+cu130
- CUDA: 13.0
- GPU: 8× NVIDIA H200 (SM 9.0), 141 GB HBM3e each
- Model: pastapaul/DeepSeek-V4-Flash-W4A16-FP8 (FP8_BLOCK attention + W4A16 GPTQ routed experts, produced via llm-compressor PR #2647 branch `kylesayrs/transformers-v5`, mirrors RedHatAI NVFP4-FP8 recipe topology)

### 🐛 Describe the bug

When serving a compressed-tensors W4A16 MoE model under tensor parallelism, the Marlin MoE kernel `moe_wna16_marlin_gemm` raises:

```
RuntimeError: Invalid thread config: thread_m_blocks = 4, thread_k = -1,
thread_n = -1, num_threads = -1 for MKN = [49152, 256, 4096], num_bits = 4,
group_size = 16, has_act_order = 0, is_k_full = 0, has_zp = 0,
is_zp_float = 0, max_shared_mem = 232448
```

### Root cause

The kernel sees `group_size = 16`, but the recipe specifies and the saved checkpoint contains `group_size = 128`. The mismatch arises because `weight_scale` is **not sharded along K** when the routed expert weight is sharded along K under TP, while `num_groups` is derived at runtime from the unsharded scale's shape:

```
group_size = K_shard / num_groups = 256 / 16 = 16   # WRONG
            (expected 2048 / 16 = 128)
```

The weight tensor is correctly sharded:
- Full `down_proj.weight`: `(out_dim, K=2048)`
- Per-rank shard at TP=8: `(out_dim, K_shard=256)`

The scale tensor is **not** sharded along K:
- Saved `down_proj.weight_scale`: `(out_dim, num_groups=16)`
- Per-rank scale: `(out_dim, num_groups=16)` ← unchanged across ranks

When `compressed_tensors_moe_wna16_marlin` calls Marlin, it passes the unsharded scale alongside the sharded weight. Marlin's `moe_wna16_marlin_gemm` validation derives `group_size = size_k / num_groups`, where `size_k` is the per-rank K. Result: `group_size` collapses to 16, which is below `MIN_THREAD_K = 128`, so `is_valid_config` rejects every entry in `large_batch_thread_configs[]` and the kernel selector returns `MarlinDefault`, yielding the cryptic `thread_k=-1` error.

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

Serve command:

```bash
vllm serve <model_dir> \
  --tensor-parallel-size 8 \
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

### TP=1 behaves correctly

When run with `--tensor-parallel-size 1`, no K sharding occurs. Marlin sees `size_k = 2048 / num_groups = 16 = group_size = 128`, lookup table matches, kernel selector returns a valid tile config, server starts.

This confirms the bug is TP-only, in the K-sharding path for `compressed_tensors_moe_wna16_marlin`'s scale parameter handling.

### Suggested fix location

`vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16_marlin.py`

In the weight loading / `create_weights` path, when the routed expert weight is sharded along K at TP boundaries, the `weight_scale` parameter must also shard along K:

```python
# Sharding logic (current code is missing this for weight_scale):
num_groups_per_shard = num_groups // tp_size
scale_shard = weight_scale[:, k_group_start:k_group_end]
```

For comparison, the FP8 W8A8 MoE path at `compressed_tensors_moe_w8a8_fp8.py` correctly shards `weight_scale_inv` along K under TP for block-quantized scales. The W4A16 grouped path appears to skip the equivalent sharding.

### Impact

This blocks **all** compressed-tensors W4A16 MoE deployments under TP > 1 on every GPU architecture, not just SM12x. Specifically affects:

- `Intel/DeepSeek-V4-Flash-W4A16-AutoRound` (Intel's published model card notes "vLLM and SGLang is not supported currently" — same root cause class)
- All future GPTQ/AWQ W4A16 quantizations of MoE models targeting vLLM TP

### Related issues / PRs

- #38022 / PR #38222: separate Marlin MoE shape-alignment bug for non-128-aligned hidden dims (different root cause; same symptom class).
- PR #41276 (kylesayrs compressed-tensors V4 support): handles attention scale_fmt and Linear-wrap fixes; does not touch MoE TP scale sharding.
- PR #40991 (jasl SM12x V4): adds SM12x kernel fallbacks; does not touch compressed-tensors MoE quantization paths.

### Repro artifacts

Full integration findings, scripts, and reproduction details:
https://github.com/pasta-paul/dsv4-flash-awq-w4a16

Specifically:
- Recipe: `scripts/quantize_v4_w4a16.py`
- Diagnostic that confirmed scales-correct-on-disk: `scripts/inspect_scales.py`
- 5 prior upstream gaps documented: `findings/kylesayrs-pr-41276-integration.md`

### Before submitting a new issue

- [x] Searched for relevant issues
- [x] Verified on latest jasl/ds4-sm120 HEAD `68901da`
- [x] Confirmed independent of #38022 (different shape constraint, same kernel symptom class)
- [x] Saved checkpoint scales verified correct on disk
- [x] TP=1 confirms bug is TP-sharding-only

---

## Follow-up PR comment for #40991

To be posted on https://github.com/vllm-project/vllm/pull/40991 after filing the issue:

> Filing #XXXXX (compressed-tensors W4A16 MoE TP scale-sharding) as a separate issue since it's orthogonal to the SM12x sparse-MLA / indexer work in this PR. Hits on H200 SM90 too — not SM12x-specific.
>
> Briefly: kylesayrs PR #41276 cherry-picked into ds4-sm120 cleanly, structural integration through `compressed_tensors_moe_wna16_marlin` works (model loads, weights resolve, packed_modules_mapping resolves fused_wqa_wkv/fused_wkv_wgate), but Marlin sees `group_size=16` at TP=8 because `weight_scale` isn't sharded along K alongside the weight. TP=1 works.
>
> Also confirms @wuwenthink's coding 0/2 finding on RTX PRO 6000 SM120 lines up with our Phase 1 H200 native finding (chat-smoke coding 2/2 PASS on FP8 native). The reasoning-token-exhaustion theory is testable on H200; we'll log if we see the same model-behavior pattern there with reasoning enabled.
>
> Full integration writeup: pasta-paul/dsv4-flash-awq-w4a16 — 5 distinct upstream gaps documented during this work, plus the Marlin TP fix surface.
>
> /cc @jasl @kylesayrs @dsikka

---

## Crediting other contributions

When the fix lands or the issue is filed, credit:
- @jasl for SM12x-V4 base support (PR #40991)
- @kylesayrs for compressed-tensors V4 attention path (PR #41276)
- @bbbearxyz for SM12x Triton fallback kernels
- @aabbccddwasd for indexer KV cache layout fix
- @wuwenthink for SM12x harness validation

The repo's `findings/kylesayrs-pr-41276-integration.md` already documents all upstream PRs we built on top of.
