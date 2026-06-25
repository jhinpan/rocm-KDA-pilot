# Humanize Gen-Plan Draft: FlyDSL MXFP4 MoE 2-Stage Tuning On gfx950

Use this draft to generate a Humanize RLCR implementation plan for **performance
tuning** the FlyDSL MXFP4 (per-1x32 microscale fp4) MoE 2-stage GEMM pipeline on
AMD gfx950 / MI350X-MI355X.

This is NOT a correctness/feature task and NOT the FlashAttention contract. The
pipeline already works and passes correctness; the goal is to raise MFU at large
shapes and cut latency at small token counts, across DeepSeek / Kimi / GPT-OSS
shapes, without breaking AITER layout compatibility or correctness.

## Source Refs

- Task issue: https://github.com/ROCm/FlyDSL/issues/708
- FlyDSL upstream: https://github.com/ROCm/FlyDSL
- FlyDSL working fork: https://github.com/jhinpan/FlyDSL-lab
- Working baseline (LOCKED): `upstream/main` @ 523ca1c7. Verified that both
  `kernels/mixed_moe_gemm_2stage.py` and `kernels/moe_gemm_2stage.py` exist at
  this ref. Use this exact ref for `--base` and `--base-branch`.
- Comparison target: aiter MXFP4 MoE 2-stage pipeline (ck_gemm_moe_2stages)
- GPU / ROCm: gfx950 / CDNA4, ROCm 7.2
- Observed on: {{DATE}}
- FlyDSL main SHA: {{FLYDSL_MAIN_SHA}}
- Worktree HEAD / branch: {{WORKTREE_HEAD}} / {{WORKTREE_BRANCH}}

## Issue Context (#708)

- Reporter: coderfeli. Milestone: v0.4. Type: enhancement (perf, not bug).
- Verbatim intent:
  - Large-shape MXFP4 MoE shows **low MFU** vs expected.
  - Small-token cases show **high latency** (hurts decode / small-batch).
  - Pipeline works; needs tuning across token regimes and MoE shapes.
- Author's tuning hints: support **Slice-K** (different waves compute different
  K, reduce in LDS); weight loading via LDS + 2/3-stage pipelining; **no LDS
  bank conflicts for all shapes**; tune pipelines.
- The issue has NO numeric targets and NO measurement protocol. This draft
  supplies them; they are derived from aiter's own configs/CI, not from the
  author. "Higher is always better" — there is no fixed pass bar; the objective
  is to beat the measured baseline on every shape×token point without regressing
  any of them.

## Ultimate Goal

Tune the FlyDSL MXFP4 MoE stage1/stage2 kernels so that, on gfx950:

- **Large shapes** (tokens >= 4096): higher MFU (effective TFLOPS / fp4 peak).
- **Small tokens** (tokens <= 64): lower end-to-end latency (us).
- No regression at any other token count, on any target model shape.

Correctness and AITER layout compatibility are HARD gates. The win is a Pareto
improvement over the locked baseline across the full token sweep — not a single
cherry-picked shape.

## Kernel Contract

### K — what may change

- Primary target files (FlyDSL-lab):
  - `kernels/mixed_moe_gemm_2stage.py` — `compile_mixed_moe_gemm1` /
    `compile_mixed_moe_gemm2` builders (fp8/fp4 × fp4 paths).
  - `kernels/moe_gemm_2stage.py` and `kernels/moe_common.py` — shared MoE plumbing.
  - `kernels/mfma_preshuffle_pipeline.py` — LDS store / swizzle / pipeline
    helpers (only for bank-conflict / pipelining work; see HARD CONSTRAINTS).
  - The shape→config dispatch / heuristic that selects tile params per
    (token, model_dim, inter_dim, expert, topk).
- The tunable knobs are the `compile_mixed_moe_gemm1/2` parameters:
  `tile_m, tile_n, tile_k, k_batch (split-K / Slice-K), persist_m,
  use_async_copy, waves_per_eu, gate_mode, xcd_swizzle, use_cshuffle_epilog,
  b_nt, a_scale_one`.
- GEMM1 and GEMM2 may be tuned independently (different optimal tiles expected:
  stage1 K=model_dim is large; stage2 K=inter_dim differs).
- Stage / preload / async-copy / persistent-scheduling parameters are in scope.

### Legality constraints (the search must pre-filter, kernel raises otherwise)

- `tile_k_bytes % 64 == 0`
- `tile_m * tile_k * elem_bytes % total_threads == 0`
  (`total_threads = min(4, tile_n // 32) * 64`)
- split-K: `K_per_batch % tile_k == 0`
- LDS usage must fit the arch limit (overflow triggers `_split_lds_out`).

### R — correctness reference (HARD, do not weaken)

- Canonical harness: `op_tests/test_moe_2stage.py` (aiter repo).
- MXFP4 = `QuantType.per_1x32` with `q_dtype_a/q_dtype_w` in
  {fp4x2 (a4w4), fp8×fp4 (a8w4)}; gfx950-only (the test early-returns otherwise).
- Reference: `torch_moe_stage1` / `torch_moe_stage2` in bf16.
- Gate: `strict_accuracy=True` — `logits_diff <= 0.01` and no FAIL/ERROR rows.
  Do NOT relax this to make a candidate pass.
- AOT cache check (`fail_on_aot_cache_miss`) must still hold.

### HARD CONSTRAINTS — must NOT change

- **AITER-compatible scale/weight shuffle layout.** The preshuffle weight and
  scale layout (aiter `ops/shuffle.py`: `shuffle_weight`, `shuffle_weight_a16w4`,
  `shuffle_scale_a16w4`, `shuffle_scale_for_int4`, `rearrange_4bit_elements`)
  is the interop contract with AITER and is frozen. Tuning must work *within*
  this layout. Changing it is out of scope and breaks the issue's stated goal.
- Output dtype / external kernel signature consumed by aiter `fused_moe`.
- Correctness thresholds above.

### W — workload (shapes + token sweep)

Shapes are taken verbatim from aiter `aiter/configs/model_configs/*_untuned_fmoe.csv`
(these are the canonical per-model MoE dims; no need to invent shapes):

| Model | model_dim | inter_dim | experts | topk | act | a×w |
|---|---|---|---|---|---|---|
| DeepSeek V3 | 7168 | 256 | 257 | 9 | Silu | fp4×fp4, fp8×fp4 |
| DeepSeek V4 | 7168 | 512 | 385 | 7 | Silu | fp8×fp4 |
| Kimi K2 | 7168 | 256 | 384 | 8 | Silu | fp4×fp4, fp8×fp4, i4 |
| GPT-OSS | 3072 | 3072 | 128 | 4 | **Swiglu** | fp4×fp4, fp8×fp4 |

All: `QuantType.per_1x32`, `use_g1u1=1`, `doweight_stage1=0`, bf16 activation ref.

Token sweep:
- DeepSeek / Kimi: `1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768`
  (covers the small-token latency regime AND the large-shape MFU regime).
- GPT-OSS: `256,512,1024,2048,4096,8192,16384,32768` (throughput shape; no tiny
  tokens — note GPT-OSS is Swiglu and may exercise `swiglu_limit`).

Latency regime = tokens <= 64; MFU regime = tokens >= 4096.

## Metrics & Measurement Protocol

We measure baseline and peak ourselves; nothing here needs the author.

- **Throughput / MFU**: effective TFLOPS as computed in `test_moe_2stage.py`:
  `token*model_dim*inter_dim*3*topk*2 / us` (combined stage1+stage2+sorting us).
  MFU = effective TFLOPS / measured gfx950 fp4 MFMA peak.
  - {{FP4_PEAK_TFLOPS}} = gfx950 fp4 MFMA peak — measure via microbench (or
    CU_count × clock × fp4-MFMA-FLOPs/clk). Fix this constant in the plan once
    measured so MFU is comparable across runs.
- **Latency**: per-shape `us` from the same harness (kernel path incl. sorting),
  fixed warmup/iters (use the harness defaults; record them).
- **Baseline**: run the locked base branch over the W shapes BEFORE any change —
  this is the reference table every candidate is compared against. Mirrors
  aiter's `op_tune.sh` "Test Performance before tuning → tune → test after".
- Objective: maximize MFU at large tokens, minimize us at small tokens, with NO
  regression at any other point. Report the full per-point table, not aggregates.

## Baseline And Benchmark Commands

Set `GPU` to an idle gfx950 id. `source .humanize/kernel-agent/runenv.sh` if
preflight wrote one.

Correctness smoke (one MXFP4 shape):
```bash
HIP_VISIBLE_DEVICES=$GPU python3 op_tests/test_moe_2stage.py \
  -q 4 -dim 7168,256 -e 257 -k 9 -t 1 32 4096
```

Full per-model sweep (example: DeepSeek V3 a4w4):
```bash
HIP_VISIBLE_DEVICES=$GPU python3 op_tests/test_moe_2stage.py \
  -q 4 -dim 7168,256 -e 257 -k 9 \
  -t 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768
```
> Confirm exact flag spelling against the harness `argparse` before relying on
> it (`-q` quant index, `-dim` model_dim,inter_dim, `-e` experts, `-k` topk,
> `-t` token list). GPT-OSS uses Swiglu (`-a swiglu`) and inter_dim=3072.

## Required Knowledge Tools

Use only when they change the next decision:
- `ROCmKernelWiki` — gfx950 fp4 MFMA throughput, MoE 2-stage scheduling,
  split-K / LDS-reduce, LDS bank-conflict avoidance, async-copy / direct-to-LDS.
- `flyprof` — instruction-level evidence when a shape is below peak or a
  candidate plateaus (MFMA issue rate vs VMEM/LDS wait vs waitcnt vs tail/grid).
- `rocm-report-skill` — turn a rocprofv3 / ATT report into one concrete
  next hypothesis (e.g. "stage2 small-M is launch/tail bound").

Profile to answer a named question, not as ritual. Examples:
- Are large shapes compute-bound at fp4 peak, or stalled on LDS / async-copy?
- Are small tokens launch-overhead / tail / grid-underfill bound, or tile-bound?
- Did Slice-K actually reduce the K-reduction stall without adding LDS conflicts?

## Suggested Optimization Order (from easy → hard)

1. **Tile sweep** (`tile_m/n/k`, `waves_per_eu`, `xcd_swizzle`) per
   stage × shape — low risk, no layout change, big lever. Use the
   `gemm-optimization` skill's M-regime tile table as priors.
2. **Shape→config dispatch**: separate small-M (skinny) vs large-M tile
   selection; specialize stage1 vs stage2. This is the main lever for hitting
   both ends of the sweep.
3. **persist_m / async_copy** tuning per stage (stage1 likely persist_m=1).
4. **Slice-K (k_batch>1) with LDS reduce** for shapes where K dominates and M is
   skinny — the issue's named idea; verify no new LDS bank conflicts.
5. **2/3-stage pipelining + bank-conflict-free LDS layout** in
   `mfma_preshuffle_pipeline.py` — highest effort, do last, behind profiling.

## RLCR Loop Rules

- No kernel changes before RLCR is active.
- Keep `.humanize*`, raw rocprof/ATT, cache, build, huge CSV artifacts untracked.
- One candidate change at a time unless coupling is technically necessary.
- Every perf claim names: GPU id+model, branch+commit, exact command, shape set,
  dtype (a4w4 / a8w4) + act, warmup/iters, CSV/profile path, idle-GPU evidence.
- Failed candidates recorded in `docs/attempts.jsonl` or
  `docs/optimization-ledger.md`.
- No win claimed from a single noisy near-threshold run; a win must hold across
  the per-point table and regress nothing.

## Scope Of This Loop (LOCKED)

- Base ref: `upstream/main` @ 523ca1c7.
- Shapes: ALL 4 models (DeepSeek V3, DeepSeek V4, Kimi K2, GPT-OSS).
- Dtypes: fp4×fp4 (a4w4) AND fp8×fp4 (a8w4). DeepSeek V4 is fp8×fp4 only.
- i4 (Kimi a16wi4) is OUT of scope for this loop.
- This is a large W matrix; the plan should stage it: land the tile-sweep +
  dispatch lever on the skinny shapes (DS V3 / Kimi) first, then extend the same
  config machinery to DS V4 and GPT-OSS rather than re-deriving per shape.

## Open Items To Resolve In The Plan

- {{FP4_PEAK_TFLOPS}}: measured gfx950 fp4 peak for the MFU denominator
  (microbench, or CU_count × clock × fp4-MFMA-FLOPs/clk). Fix once measured.
- Where tuned per-model configs live on the FlyDSL side (single tuned table vs
  per-model CSV like aiter's `*_tuned_fmoe.csv`).
