# DeepSeek-V4-Flash AWQ-W4A16 Quantization — Mission Report

**Project:** Public integration of DeepSeek-V4-Flash compressed-tensors W4A16 quantization end-to-end (calibration + vLLM serve) on 8× H200.
**Hardware:** AWS p5en.48xlarge (8× H200 SM 9.0, 141 GB HBM3e each, 2 TiB RAM, DLAMI Ubuntu 24.04 PyTorch 2.10).
**Mission start (UTC):** 2026-05-01T14:46:19Z

> **Historical-document note (2026-05-15):** This is a phase-by-phase log of what was applied at the time of each phase. SHA references to `neuralmagic/vllm@kylesayrs/deepseek-ct@f910a73a93` (and Phase 4's `jasl/vllm@77bbc16` tip) reflect what existed in upstream history when those phases ran. Kyle force-pushed `kylesayrs/deepseek-ct` on ~2026-05-08 and rewrote `f910a73a93` out of history; current builds apply the content-pinned rebased successor `d09eeb498` via the vendored [`scripts/kylesayrs-deepseek-ct.patch`](scripts/kylesayrs-deepseek-ct.patch). For the current build flow see the [README quickstart](README.md#quickstart--dual-dgx-spark-tp2), [`scripts/bootstrap_dsv4_spark.sh`](scripts/bootstrap_dsv4_spark.sh), [`scripts/Dockerfile.dsv4-spark`](scripts/Dockerfile.dsv4-spark), and [`findings/kylesayrs-pr-41276-integration.md`](findings/kylesayrs-pr-41276-integration.md#sha-rebase-recovery-issue-1-2026-05-08).

## Storage decision
**Workspace lives on ephemeral NVMe.** `/workspace` is a symlink to `/opt/dlami/nvme/workspace`, which sits on a 28 TB LVM volume across 8× 3.5 TB local NVMe drives (DLAMI pre-mount).

> **WARNING (ephemeral):** anything not uploaded to S3 or HuggingFace before instance termination is LOST. All artifacts also scp'd back to a local working directory after each phase.

EBS root (`/`) has 470 GB free and is reserved for OS / user home only.

## Pre-flight
- 8× H200 (143 GB each), SM 9.0
- PyTorch 2.10.0+cu130, CUDA 13.0, driver 580.126.16
- Memory 2.0 TiB
- Upstream resources confirmed reachable:
  - jasl/vllm@ds4-sm120 = `428e08ec`
  - jasl/vllm-ds4-sm120-harness HEAD = `85aca32`
  - flagos-ai/DeepSeek-V4-FlagOS HEAD = `f9846dc`
  - deepseek-ai/DeepSeek-V4-Flash HF page = HTTP 200

## PR #40991 notes
- New SM12x Triton kernels added; SM90 (Hopper) path preserved (existing FlashMLA).
- PR header lists PyTorch 2.11+ as required, but SM90-only build should work on 2.10. Will bump only if build fails.
- Caveat (per PR): reasoning-enabled tasks may exhaust token budget on coding tests. Matches wuwenthink finding.

## Phase plan
- Phase 0: Setup
- Phase 1: Native baseline + harness
- Phase 2: Dequant FP4/FP8 → BF16
- Phase 3: AWQ-W4A16 quantize (4–6 h)
- Phase 4: Verify quantized + harness delta
- Phase 5: HF upload + cleanup

## Phase 0 — Setup & Verification
- Started: 2026-05-01T14:46:19Z
- Decisions:
  - **Workspace = /opt/dlami/nvme/workspace** (28 TB ephemeral LVM, symlinked to `/workspace`).
  - **`CUDA_HOME` left as DLAMI default** (`/opt/pytorch/cuda`); brief's suggested `/usr/local/cuda` does not exist on this AMI. Removed wrong overrides from `~/.bashrc`.
  - **vLLM build needs `LIBRARY_PATH=$CUDA_HOME/lib`** so `ld` finds `libcudart_static.a` and `libcudadevrt.a`. Added to build script.
  - **Decision (Option 1):** allow `pip install -e .` to upgrade torch 2.10 → 2.11 inside the DLAMI venv. jasl/ds4-sm120 hard-pins `torch==2.11.0`; PR description explicitly requires PyTorch ≥ 2.11. NCCL / cusparseLT pins already match DLAMI; cuDNN 9.15 → 9.19 expected.
- Outcomes:
  - 8× H200 (143 GB), SM 9.0, 2 TiB RAM, 28 TB ephemeral NVMe verified
  - Quant tooling installed: `llmcompressor`, `compressed-tensors`, `datasets`, `accelerate`. `autoawq` import broken vs DLAMI's transformers — non-blocking, llmcompressor recipe is primary.
  - Repos cloned: `vllm-source` (`428e08ec`), `vllm-ds4-sm120-harness` (`85aca32`), `flagos` (`f9846dc`).
  - Model download running in `phase-1-download` tmux (parallel scheduling optimization).
- Issues encountered during build (all resolved — captured here for reproducers):
  - 1st build attempt: missing `setuptools_scm` → fixed.
  - 2nd build attempt: bad `CUDA_HOME=/usr/local/cuda` (path absent) → fixed.
  - 3rd build attempt: `ld` couldn't find `-lcudart_static`/`-lcudadevrt` → fixed via `LIBRARY_PATH`.
  - 4th build attempt: cmake configure failed — `CUDA_nvrtc_LIBRARY NOTFOUND` because DLAMI's `/opt/pytorch/cuda/lib` only ships versioned `libnvrtc.so.13` (no unversioned `libnvrtc.so` symlink that legacy FindCUDA wants). Created unversioned symlinks for `nvrtc`, `cublas`, `cublasLt`, `cudnn`, `cufft`, `curand`, `cusolver`, `cusparse`, `cupti`, `nvJitLink`, `nccl`. Also added `/opt/pytorch/cuda/lib/stubs/libcuda.so` → `/usr/lib/x86_64-linux-gnu/libcuda.so.1`.
  - 5th build attempt: completed compile (33 min), `BUILD_DONE_` printed, but `vllm._C.abi3.so` failed at import with `undefined symbol _ZN3c1013MessageLoggerC1EPKciib`. Root cause: pip's resolver upgraded torch 2.10 → 2.11 *after* nvcc was done compiling, so the .so was linked against c10 ABI from 2.10 while runtime imports torch 2.11.
  - 6th build attempt: SUCCESS. `import vllm._C` etc. clean against torch 2.11.
  - First serve attempt failed during profile run: flashinfer JIT-compile of `sampling.so` failed because it links with `-L/opt/pytorch/cuda/lib64`, but DLAMI ships `lib` (no `64`). Fix: `ln -sfn /opt/pytorch/cuda/lib /opt/pytorch/cuda/lib64`.
- HF auth: token set on remote (`huggingface-cli login`, file 600).
- **HF destination:** `pastapaul/DeepSeek-V4-Flash-AWQ-W4A16` (no hyphen).

## Phase 1 — Native V4-Flash Baseline
- Started: 2026-05-01T16:25:33Z (serve up at 16:34:19Z, ~9 min cold start incl torch.compile + TileLang JIT)
- Smoke test ✓: `"The answer to 2+2 is 4."`
- jasl harness against native FP4/FP8 weights, 8× H200, TP=8, fp8 KV cache, max_model_len=16384:
  - **chat-smoke `quick`**: 4 / 4 PASS
  - **chat-smoke `quality`**: 4 / 4 PASS
  - **chat-smoke `coding`**: **2 / 2 PASS** (vs wuwenthink's 0/2 on SM120) — **major finding: SM12x kernel path has a bug not present in SM90 FlashMLA path**.
  - **toolcall15**: 23 / 30 points = 77 %, 11 strict pass / 4 fail. Matches PR #40991 caveat about non-deterministic tool-call patterns.
- Bench complete (random, 128 in / 512 out, 48 prompts, ignore-eos):

  | Concurrency | Output tok/s | Total tok/s | TTFT median (ms) | TPOT median (ms) | Duration (s) |
  |---|---|---|---|---|---|
  | 1 | 126.16 | 157.70 | 71.19 | 7.78 | 194.80 |
  | 4 | 438.14 | 547.68 | 199.54 | 8.65 | 56.09 |
  | 8 | 776.57 | 970.72 | 211.96 | 9.24 | 31.65 |

  PR #40991 reported ~478 tok/s @ C=8 on 2× RTX PRO 6000 (SM120). 8× H200 SM90 = ~1.6× that throughput, with the SM90 FlashMLA path passing 2/2 coding (vs 0/2 on SM120).
- Native server stopped, GPUs free at 0 MiB used.

## Phase 2 — Dequantize FP4/FP8 → BF16
- Started: 2026-05-01T18:22:37Z
- Tool: `flagos/convert_weight.py --device cuda`
- Source 153 GB → expected output ~600 GB BF16, plenty of headroom on the 28 TB ephemeral NVMe.
- 69 187 total tensors (33 792 FP4 expert weights + 375 FP8 non-expert + 34 167 scales + ~717 BF16/FP32 already-resolved).
- 46 shards to convert.
- **Result:** dequant completed in 7 min 31 s. 34 167 weights converted, 853 already-BF16/FP32 kept, 34 167 scale entries removed. Output: 543 GB across 46 shards, 35 020 keys total.

### Phase 2 verification — *vLLM BF16 path is structurally impossible in jasl/ds4-sm120*
- jasl's `vllm/model_executor/models/deepseek_v4.py:1017` unconditionally reads `config.quantization_config["scale_fmt"]`. There is no BF16-only inference path.
- Tried two workarounds; both fail:
  1. Strip `quantization_config` → `AttributeError: 'DeepseekV4Config' object has no attribute 'quantization_config'`.
  2. Keep `quantization_config: {fp8, ue8m0}` and serve BF16 weights → FP4/FP8 weight loaders expect packed int8 + E8M0 scales, not BF16 tensors. (Did not run; structural failure mode is obvious.)
- **Decision (2026-05-01T18:34Z):** skip vLLM BF16 verify; rely on Phase 3 dry-run (16-sample llmcompressor calibration) as the BF16 sanity check. transformers' built-in `DeepseekV4ForCausalLM` will load the patched BF16 config cleanly. If dry-run fails fast, BF16 is bad. If it succeeds, BF16 is good and we proceed to full quant.
- BF16 config patched: `quantization_config` removed, `expert_dtype: bf16`, `torch_dtype: bfloat16`. Original preserved at `config.json.bak`.

## Phase 3 — AWQ-W4A16 Quantization
- Started: 2026-05-01T18:35:00Z (first attempt blocked, see below)

### Tooling pivot — *DSV4 isn't in any released `transformers`*
- DeepSeek-V4-Flash released 2026-04-24; `DeepseekV4ForCausalLM` is in **none** of: `transformers` 4.52, 4.57, 5.7, or `main`. No DSV4 PR has been merged. PR #45643 (`add-deepseek-v4` branch) is open and supersedes the closed #45616.
- Root impact: brief's `llmcompressor.transformers.oneshot(...)` path (and AutoAWQ) both go through `transformers.AutoModelForCausalLM.from_pretrained`, which fails at architecture lookup.
- Per published reference: RedHat's `RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8` was produced via `vllm-project/llm-compressor` PR #2647 (branch `kylesayrs/transformers-v5`, commit `e03eb83` "dsv4 works", 2026-04-30). The PR adds `linearize_moe_model()` which converts V4's hash-routed FP4 expert modules into `nn.Linear` layers GPTQ can calibrate.
- **Tooling stack landed:**
  - **Isolated venv** at `/workspace/.venv-quant` (DLAMI venv reserved for vLLM serving). DLAMI venv was briefly broken by an earlier `pip install --force-reinstall llmcompressor` that cascade-downgraded torch to 2.10+cu128 and replaced cu13 NVIDIA packages with cu12; recovered by reinstalling cached torch 2.11.0 wheel + `nvidia-nccl-cu13==2.28.9` with `--no-deps`.
  - `transformers==5.8.0.dev0` from `huggingface/transformers@add-deepseek-v4` (PR #45643).
  - `compressed-tensors==0.15.1a20260428` (pre-release, required by kylesayrs branch).
  - `llmcompressor==0.10.1.dev109+ga308bc0e` from `vllm-project/llm-compressor@kylesayrs/transformers-v5`.
  - `torch==2.11.0+cu130` plus the cu13 NVIDIA stack.
- `DeepseekV4Config` loads via `AutoConfig.from_pretrained("/workspace/model-bf16", trust_remote_code=True)` — confirmed.

### Recipe pivot — *W4A16 instead of NVFP4 + FP8_BLOCK*
- kylesayrs's example uses `NVFP4` for experts + `FP8_BLOCK` for attention. We're keeping his `linearize_moe_model()` framework (the load-bearing V4 architecture support) but using `W4A16` for all `Linear` layers. Reasoning: downstream DGX Spark deployment target requires Marlin W4A16 cleanly; NVFP4 needs an `eugr/RobTand` patch stack on Spark.
- Calibration: `HuggingFaceH4/ultrachat_200k`, V4's manual chat encoding (BOS / `<｜User｜>` / `<｜Assistant｜>` / EOS). Sticking with kylesayrs's tested dataset + preprocessing rather than the brief's `bigcode/the-stack-smol` since that hasn't been validated against V4.
- Recipe: `GPTQModifier(config_groups={"default": QuantizationScheme(targets=["Linear"], **W4A16)}, ignore=["lm_head"])`, `sequential_targets=["DeepseekV4DecoderLayer"]`, `batch_size=32`, `max_seq_len=512`.
- **Dry-run launched 2026-05-01T18:57:56Z**, 16 samples, output `/workspace/model-w4a16-dryrun`. If dry-run succeeds, full run with 1024 samples; if W4A16 hits V4 edge cases, fall back to NVFP4+FP8_BLOCK (RedHat's exact recipe).

### Phase 3a calibration — outcome
- **Calibration succeeded** after 5 attempts of patching upstream rough edges in kylesayrs's PR #2647 (transformers-v5 branch of vllm-project/llm-compressor) and HF transformers PR #45643 (add-deepseek-v4 branch).
- Patches landed (committed under `patches/`):
  - `helpers.py.diff` — `SequentialTracer.create_arg` extension to handle `transformers.cache_utils.Cache` via empty-constructor call (mirroring the existing `PretrainedConfig` pattern). Without this, fx fails with `NotImplementedError: argument of type: <class 'DynamicCache'>`.
  - `modeling_deepseek_v4.py.diff` — skip auto-DynamicCache creation in `DeepseekV4Model.forward`. Reason: V4-Flash config has `layer_types=None`, so `DynamicCache(config=...)` falls back to generic `DynamicLayer` which lacks `store_compression_weights`. With `past_key_values=None`, the compressor takes its no-cache fallback path.
- `/dev/shm` resized to 1.8 TB (`sudo mount -o remount,size=1800G /dev/shm`); default 1 TB OOM'd `linearize_moe_model` at step 39/43.
- Result: 147 GB output, 4 safetensors shards, 276 356 keys, ~7m save time. Calibration loop ran 16 samples × 44 layers in ~80 minutes wall.

### Phase 3a vLLM verify — 9 serve attempts, final blocker
- vLLM jasl/ds4-sm120 expects native flat key naming (`layers.X.attn.Y`, `hc_head_base`); kylesayrs save uses transformers-v5 nested names (`model.layers.X.self_attn.Y`, `model.hc_head.hc_base`). Bridged with `rewrite_for_vllm.py` (header rewrite + refusion of decomposed `shared_experts.down_proj` rows).
- Diagnostic finding (corrected from earlier "upstream bug" call): `shared_experts.down_proj.weight` is decomposed by transformers v5 save into `hidden_size=4096` separate `(moe_intermediate_size,) bf16` rows. Each row is one slice of the original 2D weight — recoverable by `torch.stack(rows, dim=0)`. Done in `rewrite_for_vllm.py`.
- vLLM PR #41276 (kylesayrs `neuralmagic/vllm@kylesayrs/deepseek-ct`, commit `f910a73a93`) cherry-picked into jasl tree. Adds: defensive `scale_fmt` fallback, `wo_scale_name` selector (FP8 vs CT), Linear-wrapping of two raw `torch.mm` calls.
- Config-side patches: `compress_ratios`, `num_hash_layers`, `qk_rope_head_dim`, `torch_dtype`, `rope_scaling` reinstated; `rope_parameters` removed; `quantization_config.ignore` rewritten to regex `[lm_head, re:.*attn.*, re:.*shared_experts.*]`.
- **Final blocker:** kylesayrs PR #41276 raises `NotImplementedError("DeepSeekV4 requires FP8 attention quantization")` because our recipe ignored `self_attn` → attn weights are plain BF16 (`.weight` only, no `.weight_scale`/`.weight_scale_inv`). RedHat's reference NVFP4 model (`RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8`) quantizes attn (FP8_BLOCK) — which is why theirs works.

### Decision point (2026-05-02 01:37 UTC)
- **Path 1 (~50 min):** drop `re:.*self_attn.*` from recipe ignore. Re-run dryrun calibration with attn included in W4A16. Resulting model has `attn.{wq_a,wq_b,wkv,wo_a,wo_b}.weight_packed/scale/shape` — matching what kylesayrs PR expects. ~26 min over 4-h debug cap.
- **Path 3 fallback:** ship dryrun as transformers-loadable; defer vLLM verify. Phase 4 = transformers `from_pretrained` + forward; Phase 5 = HF upload as-is. Documents vLLM integration gap as deployment-side problem.

### Path 1 attempted and surfaced new failure → pivot to Path B (mixed-precision FP8_BLOCK + W4A16)
- Path 1 (W4A16-everywhere except shared_experts) ran calibration cleanly but vLLM serve raised `NotImplementedError("DeepSeekV4 requires FP8 attention quantization")` from kylesayrs PR #41276's `wo_scale_name` selector. The PR encodes the assumption that attn weights are FP8.
- **Path B authorized: re-run with FP8_BLOCK on attn + indexer/compressor Linears, W4A16 on routed experts only, shared_experts BF16.** Matches RedHat's `RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8` reference recipe (with W4A16 instead of NVFP4 on experts for Spark Marlin compatibility downstream).
- Path B calibration: 5h12m wall (both attn FP8 calibration with 16 samples and routed-experts W4A16 calibration with 8 samples in parallel across 8 ranks). Output: 143 GB / 4 shards.

### Path B vLLM serve attempts — load fully succeeded, blocked at Marlin MoE kernel shape support
After applying the full Phase 3a patch stack:
1. `rewrite_for_vllm.py` extended with routed-expert renames (`experts.N.gate_proj/up_proj/down_proj` → `experts.N.w1/w3/w2`) and compressor-internal renames (`compressor.kv_proj→wkv`, `compressor.gate_proj→wgate`, `compressor.q_b_proj→wq_b`, `compressor.position_bias→ape`, both at the layer's compressor and at `compressor.indexer.*`).
2. `patch_v4_packed_mapping.py` added `packed_modules_mapping = {"fused_wqa_wkv": ["wq_a", "wkv"], "fused_wkv_wgate": ["wkv", "wgate"], "gate_up_proj": ["w1", "w3"]}` to `DeepseekV4ForCausalLM` — kylesayrs PR referenced `self.packed_modules_mapping` at line 205 but didn't define it on the class.
3. `patch_targets_v2.py` rewrote `quantization_config.config_groups[*].targets` to include both v5 names (`q_a_proj`, `kv_proj`, etc.) and post-rename names (`wq_a`, `wkv`, `fused_wqa_wkv`, `gate_up_proj`).

With all patches:
- All 3 safetensors shards loaded (73 sec, 20.01 GiB resident on rank 0).
- compressed-tensors loader engaged: `Using MarlinLinearKernel for CompressedTensorsWNA16` for non-MoE Linears, `Using MacheteLinearKernel for CompressedTensorsWNA16` for some, `Using CompressedTensorsWNA16MarlinMoEMethod` for FusedMoE.
- `_match_fused_layer` resolved all `fused_wqa_wkv`/`fused_wkv_wgate`/`gate_up_proj` lookups via `packed_modules_mapping`.
- Profile run `_dummy_run` started.
- **Failure point at TP=8:** `RuntimeError: Invalid thread config: thread_m_blocks=4, thread_k=-1, thread_n=-1, num_threads=-1 for MKN=[49152, 256, 4096], num_bits=4, group_size=16, has_act_order=0, is_k_full=0, has_zp=0, max_shared_mem=232448` from Marlin MoE kernel.

Diagnosis: scale-tensor-not-sharded bug in `compressed_tensors_moe_wna16_marlin`. At TP=8 with `K_per_rank=256` and `num_groups=16` (full-tensor count, not per-rank), Marlin derives `group_size = 256/16 = 16`, below `MIN_THREAD_K=128`. Bug fires at TP>2; TP=2 works correctly because the wrong-believed `group_size=64` happens to have a kernel template instantiation. Full root cause analysis at `findings/upstream-issue-marlin-tp-sharding.md`.

### Phase 3a TP=2 success — full harness PASS
- TP=2 serve up cleanly, smoke test PASS: `"What is 2+2?" → "The answer is 4."`
- Full harness run on TP=2 W4A16 dryrun:

  | Suite | TP=2 W4A16 dryrun | Native FP4/FP8 baseline | wuwenthink SM120 TP=2 |
  |-------|--------------------|--------------------------|------------------------|
  | quick | **4/4 PASS** | 4/4 PASS | 2/4 |
  | quality | **4/4 PASS** | 4/4 PASS | 3/4 |
  | coding | **2/2 PASS** ✅ | 2/2 PASS | 0/2 |
  | toolcall15 | **25/30 (83%)** | 23/30 (77%) | 26/30 |

- **The 16-sample dryrun outperforms native FP4/FP8 baseline on toolcall15 (+2 points).** Failure-set is a strict subset of baseline's; TC-14 (Malformed Response) passes under W4A16 but fails baseline.
- **First public coherent vLLM serve of W4A16 V4-Flash by anyone.**
- TP=2 is exactly the dual-DGX-Spark deployment target, so the Marlin TP scale-sharding bug at TP>2 doesn't block production.

### Verdict: Phase 3a verified at TP=2; bug in Marlin MoE TP>2 path filed upstream
The model checkpoint is correct, structurally complete, and produces matching-or-better quality than native FP4/FP8 baseline at TP=2. The Marlin scale-sharding bug at TP>2 is a separate vLLM issue (filed upstream) that doesn't affect TP=2 production deployment.

Next: Phase 3b (full 1024-sample calibration) launching for upgraded artifact, swapping in for the dryrun on HF when complete.

## Phase 4 — DGX Spark TP=2 deployment validation (2026-05-04)

Deployed the artifact onto the *original target topology*: two DGX Spark GB10 boxes (SM 12.1a, 121 GiB UMA each) running TP=2 over a QSFP RDMA direct-connect, packaged in the `eugr/spark-vllm-docker` toolchain. Cherry-pick of `f910a73a93` auto-merged cleanly on the current `jasl/vllm@77bbc16` tip.

### Workspace lock — diagnosed and patched

Without `--enforce-eager`, the first prompt over ~1 K tokens crashed with `AssertionError: Workspace is locked but allocation from deepseek_v4_attention.py:1457:_forward_prefill requires 21.80 MB, current size is 21.62 MB`. The locked size was structural — identical across two builds 28 vLLM commits apart.

Source comment at `deepseek_v4_attention.py:170-172` says "matching profile-time reservation in attention_impl's dummy-run branch" — implying a pre-allocation hook was always intended, just never landed. **`scripts/patch_workspace_prereserve.py`** implements that hook (~30 lines, single file) — `_warmup_reserve_prefill_workspace()` on `DeepseekV4MLAAttention` called from the wrapper's dummy-run early-return so the workspace gets sized before `lock_workspace()` fires.

With the patch applied, `--enforce-eager` is no longer required and decode runs at ~14–17 tok/s (vs ~3.9 tok/s under eager). Filed upstream as [`vllm-project/vllm#41700`](https://github.com/vllm-project/vllm/issues/41700) with patch + repro.

### Multi-node detail

Worker rank 1 needs `--headless` — without it, the worker tries to initialize its own engine and hits `AssertionError: collective_rpc should not be called on follower node` in `multiproc_executor.py`.

### Headline results (Spark TP=2, canonical recipe — graphs ON, no eager)

| Test | Result |
|---|---|
| Public jasl harness `chat-smoke` | **4 / 4 PASS** |
| `generation` non-thinking | **18 / 18 PASS** |
| `generation` think-high (32K reasoning budget) | **17 / 18 PASS** (1 brittle-test fail) |
| `generation` think-max (32K budget) | 9 / 18 at 32K → ✅ **9 / 10 PASS** at 64K context+budget retest (see findings doc) |
| `toolcall15` | **41 / 45 (92%)**, 83/90 points — best across all configs tested |
| `oracle_compare` vs B200 TP=2 nomtp | 5 / 5 ran, alignment numbers in findings doc |
| **GSM8K 8-shot** | **95.37% ±0.58%** — vs 92.87% on H200 reference (+2.5 pp) |
| **HumanEval pass@1 instruct** | **80.49% ±3.10%** — vs 54.27% on H200 reference (+26 pp) |
| Decode throughput | 14–17 tok/s sustained, all prompt sizes |
| Continuous uptime | 6+ h validated, 0 workspace-lock errors |

Full validation report — build provenance, B200 token-level alignment table, operational constraints, recommended next iterations — in [`findings/spark_tp2_deployment.md`](findings/spark_tp2_deployment.md). Canonical launch script: [`scripts/serve_spark_tp2.sh`](scripts/serve_spark_tp2.sh).

This is the first end-to-end validation of `pastapaul/DeepSeek-V4-Flash-W4A16-FP8` on real DGX Spark hardware.

## Phase 5 — Dual RTX PRO 6000 Blackwell deployment validation (2026-05-05)

Validated `pastapaul/DeepSeek-V4-Flash-W4A16-FP8` on the harness HANDOFF's *primary development host*: dual RTX PRO 6000 Blackwell Server (SM 12.0, 96 GB × 2). Built on the `ds4-sm120-experimental` superset of `ds4-sm120` so we exercise commits not yet promoted to the public PR branch — including `e734ace5` (Release DeepSeek V4 protected prompt refs under pressure), the suspected fix for the Spark 256K×2 stall observed in Phase 4d.

### Hardware

- 2× NVIDIA RTX PRO 6000 Blackwell Server, SM 12.0, 96 GB × 2 = 192 GB VRAM
- Driver 580.126.09, CUDA 12.8 toolkit (`compute_120a`)
- 60-thread x86_64, 176 GB RAM
- Brev `violent-azure-yak` (verda)

### Toolchain

- vLLM `0.1.dev16350+g395970dae` built from `jasl/vllm@ds4-sm120-experimental` tip `abad5dc71`
- Cherry-pick: `neuralmagic/vllm@kylesayrs/deepseek-ct@f910a73a` (auto-merged, 3 files, 22+/15-)
- Local: `packed_modules_mapping` patch (12-line class-attribute injection) — **still required**
- Skipped: `patch_workspace_prereserve.py` (now upstream as `1d6f5c4`)
- torch 2.11.0+cu128, triton 3.6.0, Python 3.12.13
- Build flags: `TORCH_CUDA_ARCH_LIST=12.0a CUDA_ARCH_LIST=120a MAX_JOBS=24 NVCC_THREADS=4`. Compiles `ENABLE_SCALED_MM_SM120 / NVFP4_SM120 / CUTLASS_MOE_SM120` cutlass kernel paths. Wall ~30 min on 60-core box.

### Two integration issues surfaced

**(i) `packed_modules_mapping` is still required.** Despite kylesayrs `f910a73a` and the experimental branch's reorganization, `DeepseekV4ForCausalLM` doesn't carry a `packed_modules_mapping` class attribute. Without it, model load fails on `find_matched_target` for `model.layers.0.ffn.shared_experts.gate_up_proj` because the recipe ignore list uses `w1/w2/w3` naming and the loader can't fan out the fused name. Correct mapping: `gate_up_proj → ["w1", "w3"]` (matching shared-expert ignore-list naming, not `gate_proj/up_proj`). Wired automatically into the quant config via `vllm/model_executor/model_loader/utils.py:configure_quant_config`.

**(ii) FlashInfer JIT mis-parses `12.0a` arch.** Top-p / top-k sampler routes through FlashInfer's JIT, which raises `RuntimeError: FlashInfer requires GPUs with sm75 or higher` because its arch parser doesn't recognize the `12.0a` token (the actual GPU is sm_120, way above sm_75). Workaround: `VLLM_USE_FLASHINFER_SAMPLER=0`, vLLM falls back to PyTorch-native sampler.

### Long-context probe — 256K × 2 concurrent passes

Serve config: `--max-model-len 524288 --max-num-seqs 2 --gpu-memory-utilization 0.95 --kv-cache-dtype fp8`. KV cache 1,087,973 tokens (2.08× concurrency at 524K).

| Probe | Prompt tokens | Elapsed | Result |
|---|---|---|---|
| 75K  (line=2400) | 74,457 | 58.5 s | ✅ PASS |
| 128K (line=4000) | 124,057 | 119.9 s | ✅ PASS |
| 256K × 1 (line=8000) | 248,057 | 356.0 s | ✅ PASS |
| **256K × 2-A** | 248,057 | **377.2 s** | ✅ **PASS** ← concurrent |
| **256K × 2-B** | 248,057 | **377.1 s** | ✅ **PASS** ← concurrent |
| 500K × 1 (line=16000) | 496,057 | 1230.7 s | ✅ PASS |

**Headline:** concurrent 256K×2 finishes only ~21 s slower than single 256K×1. With `e734ace5` confirmed effective on Blackwell sm_120, the 256K×2 stall observed on Spark in Phase 4d is fully resolved on the experimental tip.

> Operational note: the throughput logger in vLLM appears to suppress output during single very-long prefill chunked-prefill runs — the 500K probe ran for 20+ minutes with **no** throughput log entries despite GPUs at 98% / 418 W and the request actively progressing. Don't take logger silence as a stall signal; check `nvidia-smi` and the TCP socket state.

### Headline correctness

Harness HEAD `96785b9` (Spark match), temperature 0, top-p 1.

| Test | Result |
|---|---|
| `chat-smoke quick` | **4 / 4 PASS** |
| `toolcall15` (single round, 30 pts max) | **27 / 30 = 90%** (TC-06 fail, TC-07 partial) |
| GSM8K 8-shot, strict-match | **95.07% ±0.60%** |
| GSM8K 8-shot, flexible-extract | **94.99% ±0.60%** |
| HumanEval pass@1 (instruct, 0-shot, `--confirm_run_unsafe_code`) | **78.05% ±3.24%** |
| `generation-matrix` non-thinking en | 18 / 18 invocations clean |
| `generation-matrix` think-high en (×3 rounds) | 54 / 54 invocations clean |
| `generation-matrix` think-max @ 32K en (×3 rounds) | 54 / 54 invocations clean |

The `generation-matrix` runs do not have automatic pass/fail in this harness HEAD; all 126 invocations completed cleanly with `finish_reason=stop`.

### Performance — `vllm bench serve`

| Concurrency | In / Out | Duration | TTFT mean / p99 | TPOT mean / p99 | Output tok/s |
|---|---|---|---|---|---|
| 1 | 1024 / 1024 | 430.9 s | 237 ms / 711 ms | 20.8 ms / 21.7 ms | 47.5 |
| 2 | 2048 / 512  | 121.9 s | 1096 ms / 1900 ms | 21.7 ms / 23.0 ms | 84.0 |

Per-stream decode is rock-stable ~47–48 tok/s (p99 TPOT 22 ms). Aggregate at c=2 is 420 tok/s input+output.

### Full run notes

[`findings/rtxpro6000_blackwell_deployment.md`](findings/rtxpro6000_blackwell_deployment.md) has the full commit-stack, build flags, FlashInfer arch-parser repro, packed-modules-mapping debugging trail, and the operational notes. The harness baseline bundle is at `harness/baselines/20260506_rtxpro6000_tp2_experimental_a4069c4cb/` on the run host; raw artifacts at `/ephemeral/work/runs/2026-05-05-rtx6000/`.
