# Loop 07 — FlyDSL fp8 FlashAttention: wide 32×32×64 MFMA, gfx950 (MI350X)

**Date:** 2026-06-26 · **GPU:** MI350X (gfx950) · **Continues:** loop-04 (fp8 FA landed) and
loop-06 (rocprofv3 proved the 4× MMA-width gap). This loop **executes loop-06's concrete next
hypothesis** — adopt the wide `mfma_scale_f32_32x32x64_f8f6f4` atom — and lands it.

## At a glance

- Loop-06 found the dominant gap to aiter's native fp8 ASM was the MMA intrinsic width: aiter
  uses `32x32x64` (65536 MACs/op); FlyDSL used `32x32x16` (16384 MACs/op), 4× narrower.
- This loop adopted the wide atom for **QK** (default-on, `FLYDSL_FP8_WIDE_QK`) and, opt-in, for
  **PV** (`FLYDSL_FP8_WIDE_MMA` + no-barrier in-register P shuffle `FLYDSL_FP8_WIDE_PSHUF`).
- **Outcome:** IMPROVEMENT (tier T1 reached). Default fp8 ~947 → ~1050–1078 TF on the headline
  (+11–14%). On a 60-config multi-shape sweep: **causal beats aiter native ASM (geomean 117%,
  up to 151%)**; non-causal trails (geomean 68%). All min_cos ≥ 0.99999.
- Delivered to PR ROCm/FlyDSL#711; gate-off + opt-out codegen ISA byte-identical to base.

## What landed (default path)

- **Wide QK** (`FLYDSL_FP8_WIDE_QK`, default-on for fp8): QK^T runs on two wide
  `32x32x64` MFMAs over head_dim=128 instead of 8 narrow `32x32x16`. QK is native fp8 in every
  PV mode, so it is PV-mode-independent and applies to HIPREC (the fast default).
  - This is the **only source of the default-path gain** (gate-toggle attribution: +5–9% across
    causal and non-causal; the opt-in wide-PV helps only the slower NATIVE mode).
- **Opt-in wide PV** (`FLYDSL_FP8_WIDE_MMA` + `FLYDSL_FP8_WIDE_PSHUF`): correct (min_cos 0.9993)
  no-barrier in-register P gather, but NATIVE PV ≪ HIPREC, so it stays opt-in.

## Measured result (60-config sweep, MI350X, warmup10/iters100, FLOPs causal-discounted)

| regime | geomean Fly/aiter_asm | FlyDSL ≥ aiter | range |
|---|---:|---:|---|
| Causal | **117%** | 21/30 | 84%–151% |
| Non-causal | 68% | 0/30 | 56%–87% |
| Overall | 89% | 21/60 | — |

aiter_asm = `aiter.ops.mha.fmha_v3_fwd(..., how_v3_bf16_cvt=0)` — native fp8 ASM
(`fwd_hd128_fp8.co` / `fwd_hd128_fp8_causal.co`), NOT the bf16-convert path. Full per-config
table + CSVs accompany the PR.

## Why causal wins but non-causal trails

At `B1 S8192 H64` (apples-to-apples, causal FLOPs discounted): FlyDSL keeps **86%** of its
non-causal rate under causal (1037→896 TF); aiter keeps only **40%** (1672→677 TF). aiter's
non-causal kernel is faster (better hand-scheduled pipeline) but does not skip masked
upper-triangle tiles efficiently, so its *effective* causal TFLOPS collapse. FlyDSL skips them.
- Non-causal = raw compute-pipeline-efficiency contest → aiter's ASM leads.
- Causal = "skip masked work" contest → FlyDSL leads, more so at larger seqlen.

## Honest assessment — why non-causal parity is still open

The wide atom cut PV MFMA 4× but moved GRBM active cycles only ~4% — the kernel is **not
MFMA-issue-bound**. Roofline: HBM ~8–10% utilized; FlyDSL ~41% vs aiter ~49% of fp8 MFMA peak.
The binding constraint is the VGPR-limited dual-wave pipeline (occupancy/latency-bound), exactly
as loop-04/05 named. Closing non-causal needs a **structural** change (lower the VGPR
live-range peak to raise occupancy, or a BLOCK_M=128 / 4-wave schedule rewrite) — a separate
effort. The already-wired wide-PV combos were **tested-negative** (slower than the default), so
they are queued with evidence, not deferred by hand-wave.

## Lessons learned

See `docs/bitter_lessons.md` (this loop contributed the methodology lessons there: single-point
benchmarks mislead; immutable perf artifacts; falsification-to-close; bottleneck ≠ instruction
width; control-group-before-blame; gate-off ISA byte-identity as the equivalence proof).

## Reproduce

```bash
# default fp8 (HIPREC + wide QK):
python3 tests/kernels/test_flash_attn_fwd.py --dtype fp8 --compare --no-causal \
  --batch 1 --seq_len 2048 --num_heads 64 --num_kv_heads 64 --head_dim 128 --warmup 10 --iters 100
# add --causal for the causal gate; FLYDSL_FP8_WIDE_QK=0 opts out to narrow QK.
# full multi-shape sweep: drop the explicit shape flags (uses DEFAULT_CONFIGS).
# operand-layout decode: python3 tests/kernels/probe_wide_layout.py
```

## Process cost (for workflow tracking)

RLCR loop, 6 rounds to COMPLETE (baseline+profile → route-A P-shuffle fix → route-C wide QK →
HIPREC promotion/T1 → full non-regression+convergence → delivery+wording). Every candidate
default-off and gated; default codegen byte-identical throughout. A 7th post-loop pass added the
multi-shape sweep + the style refactor (ISA byte-identical).
