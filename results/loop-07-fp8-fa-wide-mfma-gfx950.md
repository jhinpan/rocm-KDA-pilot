# Loop 07 — FlyDSL fp8 FlashAttention: wide 32×32×64 MFMA, gfx950 (MI350X)

**Date:** 2026-06-26 · **GPU:** MI350X (gfx950) · **Continues:** loop-04 (fp8 FA landed) and
loop-06 (rocprofv3 identified the 4× MMA-width mismatch). This loop **executes loop-06's concrete
next hypothesis** — adopt the wide `mfma_scale_f32_32x32x64_f8f6f4` atom — and lands it.

## At a glance

- Loop-06 identified a major structural difference vs aiter's native fp8 ASM: the MMA intrinsic
  width. aiter uses `32x32x64` (65536 MACs/op); FlyDSL used `32x32x16` (16384 MACs/op), 4×
  narrower. (As this loop shows, width is one lever, not the whole gap — see the assessment below.)
- This loop adopted the wide atom for **QK** (default-on, `FLYDSL_FP8_WIDE_QK`) and, opt-in, for
  **PV** (`FLYDSL_FP8_WIDE_MMA` + no-barrier in-register P shuffle `FLYDSL_FP8_WIDE_PSHUF`).
- **Outcome:** IMPROVEMENT (tier T1 reached). The wide QK atom adds **+5–9%** to the default fp8
  path (gate-toggle attribution); on the headline (B1 S2048 H64) the default fp8 path measures
  ~1050–1078 TF (min_cos 1.0000). On a 60-config multi-shape sweep: **causal beats aiter native
  ASM (geomean 117%, up to 151%)**; non-causal trails (geomean 68%). All 60 default-path sweep
  configs PASS at min_cos ≥ 0.99999.
- Delivered to PR ROCm/FlyDSL#711; gate-off and opt-out (`FLYDSL_FP8_WIDE_QK=0`) codegen is ISA
  byte-identical to base (the wide QK default itself changes the ISA — that is the optimization).

## What landed (default path)

- **Wide QK** (`FLYDSL_FP8_WIDE_QK`, default-on for fp8): QK^T runs on two wide
  `32x32x64` MFMAs over head_dim=128 instead of 8 narrow `32x32x16`. QK is native fp8 in every
  PV mode, so it is PV-mode-independent and applies to HIPREC (the fast default).
  - This is the **only source of the default-path gain** (gate-toggle attribution: +5–9% across
    causal and non-causal).
- **Opt-in wide PV** (`FLYDSL_FP8_WIDE_MMA` + `FLYDSL_FP8_WIDE_PSHUF`): a correct (min_cos 0.9993)
  no-barrier in-register P gather, but it was **tested-negative** — the wide-PV combos run slower
  than the default HIPREC+wide-QK path (NATIVE/FROMBF16 PV are below HIPREC's bf16 PV throughput),
  so it stays opt-in / closed-with-evidence, contributing nothing to the default path.

## Measured result (60-config sweep, MI350X, warmup10/iters100, FLOPs causal-discounted)

| regime | geomean Fly/aiter_asm | FlyDSL ≥ aiter | range |
|---|---:|---:|---|
| Causal | **117%** | 21/30 | 84%–151% |
| Non-causal | 68% | 0/30 | 56%–87% |
| Overall | 89% | 21/60 | — |

aiter_asm = `aiter.ops.mha.fmha_v3_fwd(..., how_v3_bf16_cvt=0)` — native fp8 ASM
(`fwd_hd128_fp8.co` / `fwd_hd128_fp8_causal.co`), NOT the bf16-convert path. The full per-config
table (60 rows, FlyDSL/aiter_asm/aiter_ck µs+TFLOPS) and raw CSVs are posted as a comment on
ROCm/FlyDSL#711; they are not committed to this repo (per the "keep raw artifacts untracked"
contract).

## Why causal wins but non-causal trails

At `B1 S8192 H64` (apples-to-apples, causal FLOPs discounted): FlyDSL keeps **86%** of its
non-causal rate under causal (1037→896 TF); aiter keeps only **40%** (1672→677 TF). aiter's
non-causal kernel is faster (better hand-scheduled pipeline) but does not skip masked
upper-triangle tiles efficiently, so its *effective* causal TFLOPS collapse. FlyDSL skips them.
- Non-causal = raw compute-pipeline-efficiency contest → aiter's ASM leads.
- Causal = "skip masked work" contest → FlyDSL leads, more so at larger seqlen.

## Honest assessment — why non-causal parity is still open

Adopting the wide atom cut MFMA instruction count ~4× (default-path wide QK on the QK matmul; and
in a separate experiment the wide-PV variant on the PV matmul) but barely moved GRBM active cycles
(~4%) — the kernel is **not MFMA-issue-bound**. Roofline: HBM ~8–10% utilized; FlyDSL ~41% vs
aiter ~49% of fp8 MFMA peak. The binding constraint is the VGPR-limited dual-wave pipeline
(occupancy/latency-bound), as the earlier fp8 loop (loop-04) named. Closing non-causal needs a
**structural** change (lower the VGPR live-range peak to raise occupancy, or a BLOCK_M=128 /
4-wave schedule rewrite) — a separate effort. Consistent with this, the wide-PV combos were
**tested-negative** (slower than the default HIPREC+wide-QK path), so they are queued with
evidence, not deferred by hand-wave.

## Lessons learned

See `docs/bitter_lessons.md` (this loop contributed the methodology lessons there: single-point
benchmarks mislead; immutable perf artifacts; falsification-to-close; bottleneck ≠ instruction
width; control-group-before-blame; gate-off ISA byte-identity as the equivalence proof).

## Reproduce

All commands run from a `ROCm/FlyDSL` PR #711 checkout (these are FlyDSL repo paths, not this
repo); see the FlyDSL build/env setup in its CLAUDE.md.

```bash
# headline default fp8 (HIPREC + wide QK), one shape:
python3 tests/kernels/test_flash_attn_fwd.py --dtype fp8 --compare --no-causal \
  --batch 1 --seq_len 2048 --num_heads 64 --num_kv_heads 64 --head_dim 128 --warmup 10 --iters 100
# FLYDSL_FP8_WIDE_QK=0 opts out to narrow QK (ISA byte-identical to base).
# multi-shape sweep: run DEFAULT_CONFIGS (no shape flags) in BOTH directions:
python3 tests/kernels/test_flash_attn_fwd.py --dtype fp8 --compare --no-causal --warmup 10 --iters 100
python3 tests/kernels/test_flash_attn_fwd.py --dtype fp8 --compare --causal    --warmup 10 --iters 100
# operand-layout decode: python3 tests/kernels/probe_wide_layout.py
```

The "60-config" table is the **dense rows that meet aiter's native-asm dispatch gate** (head_dim
128, pow-2 GQA, seqlen_q>128): 30 such configs per direction × 2 directions. DEFAULT_CONFIGS also
emits split-K rows, which fp8 rejects by design (ERROR) and are excluded from the comparison.

## Process cost (for workflow tracking)

RLCR loop, 6 rounds to COMPLETE (baseline+profile → route-A P-shuffle fix → route-C wide QK →
HIPREC promotion/T1 → full non-regression+convergence → delivery+wording). Each candidate was
explored behind its own gate; until the final default-promotion the default path stayed
byte-identical, and after promotion the opt-out (`FLYDSL_FP8_WIDE_QK=0`) reproduces the base ISA
byte-for-byte. A 7th post-loop pass added the multi-shape sweep + a style refactor (verified ISA
byte-identical).
