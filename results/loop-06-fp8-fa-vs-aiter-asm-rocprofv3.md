# fp8 FlashAttention: FlyDSL (best) vs aiter native ASM — rocprofv3 comparative analysis

**Date:** 2026-06-25 · **GPU:** MI350X (gfx950), idle GPU 4 · **Tool:** rocprofv3 1.1.0
**Workload (both, identical):** B=1, S=2048, H=64, D=128, **non-causal**, fp8 e4m3fn,
per-tensor descales. Each kernel profiled **in isolation** (no shared process):
- FlyDSL: `flash_attn_dualwave_swp_gfx950_kernel` — HIPREC fp8 (default, best fp8 path),
  harness run with **no `--compare`** so only the FlyDSL kernel dispatches.
- aiter asm: `_ZN5aiter24fmha_fwd_hd128_fp8_gfx950E` (`fmha_v3_fwd`, `how_v3_bf16_cvt=0`,
  hsaco `fwd_hd128_fp8.co`) — standalone driver, same shape + quantization.

Artifacts: `profile/fp8-fa-vs-aiter-asm/raw/{stats,pmc}/` (rocprofv3 CSVs),
`aiter_asm_driver.py`, `pmc_input.txt`.

## Headline

| metric | FlyDSL HIPREC | aiter asm | ratio (Fly / aim) |
|---|---:|---:|---:|
| avg kernel latency | **145.7 µs** | **102.4 µs** | 1.42× slower |
| effective TFLOPS | ~943 | ~1342 | 0.70 |
| GRBM active cycles | 2.343e6 | 1.490e6 | **1.57×** |
| SQ_BUSY_CYCLES | 8.142e6 | 5.002e6 | 1.63× |

FlyDSL does **1.57× more active GPU cycles** for the same attention math. The question this
analysis answers: *where do those extra cycles come from?*

## The dominant mechanism: aiter uses a 4× wider fp8 MMA

This is the single biggest structural difference, and it **revises the prior loop's
"purely stall-bound, MMA only 7.7%" framing**.

| | FlyDSL HIPREC | aiter asm |
|---|---:|---:|
| `SQ_INSTS_MFMA` (per dispatch) | 4.194e6 | 1.049e6 |
| MFMA instruction ratio | — | **4.00× fewer** |
| MACs per MFMA instruction | **16384** | **65536** |
| ⇒ implied MMA intrinsic | `32x32x16` fp8 | **`32x32x64` fp8** |

Total attention MACs (QK+PV) = 6.87e10 for both. FlyDSL's 16384 MACs/MFMA is *exactly*
the `mfma_*_32x32x16_fp8` our kernel emits; aiter's 65536 MACs/MFMA is *exactly*
`32x32x64` — a **4× wider** fp8 MMA on CDNA4. aiter retires the same matmul in **a quarter
of the MFMA instructions**.

Why this matters: with a 4× wider MMA, each issued MFMA does 4× the work, so the issue/
schedule/sync overhead *per unit of math* drops ~4×. The earlier loop measured MFMA at
~7.7% of bubbles on the FlyDSL kernel and concluded "not compute-bound" — that is true *for
the FlyDSL kernel as written*, but it masked that **the instruction the kernel is built
around is 4× narrower than the one the asm kernel uses.** The stall bubbles the loop chased
(barrier, vmcnt) are downstream of issuing 4× as many MMA + supporting instructions.

## Supporting evidence (all per-dispatch PMC means)

| counter | FlyDSL | aiter | ratio | reading |
|---|---:|---:|---:|---|
| SQ_INSTS_MFMA | 4.194e6 | 1.049e6 | 4.00× | narrow vs wide MMA (above) |
| SQ_INSTS_VALU | 3.050e7 | 2.239e7 | 1.36× | FlyDSL runs more ALU — the HIPREC in-kernel fp8→bf16 V dequant + softmax bookkeeping |
| SQ_INSTS_LDS | 5.505e6 | 3.178e6 | 1.73× | FlyDSL moves more through LDS per unit math (narrower MMA ⇒ more, smaller LDS feeds; dequant staging) |
| SQ_INSTS_VMEM | 4.710e5 | 3.277e5 | 1.44× | comparable global traffic; FlyDSL slightly higher |
| SQ_WAVES | 4096 | 4096 | 1.00× | **identical total waves** — not an under-fill problem |
| VGPR / thread | 120 | 128 | 0.94× | aiter uses *more* VGPR yet is faster — occupancy/VGPR is not aiter's lever |
| LDS bytes / workgroup | 69120 | 163840 | 0.42× | aiter uses 2.4× the LDS (big K/V staging) — it trades LDS for fewer, wider math ops |
| workgroup size | 512 (8 waves) | 512 (8 waves) | 1.00× | same wave/WG |
| grid | 32768 × 8 | 4096 × 64 | — | different tiling decomposition; same total waves |

### Per-active-cycle intensity

| | FlyDSL | aiter |
|---|---:|---:|
| MFMA / active cycle | 1.79 | 0.70 |
| VALU / active cycle | 13.0 | 15.0 |
| VALU : MFMA instruction mix | 7.3 : 1 | 21.4 : 1 |

FlyDSL issues MFMA at a **higher rate** (1.79 vs 0.70 per active cycle) yet finishes slower
— because each of its MFMAs is ¼ the width. aiter's MMA pipe is "quieter" per cycle but
each op is 4× as productive, and its VALU:MFMA mix (21:1) shows the math op is amortized
over far more supporting work. FlyDSL's tighter 7.3:1 mix is the narrow-MMA signature:
proportionally too many MMA + LDS-feed instructions for the work done.

## Reconciling with the optimization loop's BLOCKED outcome

The 19-round loop concluded BLOCKED on a *latency/stall-bound, VGPR-limited* kernel and
correctly found every **incremental** lever (barrier removal, deeper prefetch, occupancy
hints, dequant reorder) measured-negative. This comparison explains *why those were the
wrong levers*: they tune the scheduling **around** a fixed, 4×-narrow MMA. The asm kernel's
advantage is **not** better barrier/prefetch scheduling — it is a fundamentally different
compute primitive (`32x32x64`) plus the larger LDS staging that wider MMA needs.

Two corrections to the loop's mental model, now evidence-backed:
1. **It is not "purely stall-bound, MMA negligible."** MMA *instruction count* is 4× too
   high; the stalls are partly a *consequence* of issuing 4× the ops.
2. **VGPR occupancy is not the asm kernel's secret.** aiter uses *more* VGPR (128 vs 120)
   and the *same* wave count (4096) — so "raise occupancy" was never going to reach parity.

## Concrete next hypothesis (single, testable)

**Adopt the `32x32x64` fp8 MMA atom in the FlyDSL PV (and QK) path.** This is exactly the
loop's documented route (B)/(A) class of "funded structural rewrite," now with a precise
target: the win is not in the pipeline schedule, it is in the MMA intrinsic width.

- Expected first-order effect: ~4× fewer MFMA + proportionally fewer LDS feed ops, removing
  the bulk of the 1.57× active-cycle gap. Even partial realization should move FlyDSL fp8
  off the ~969 TF plateau toward the ~1340 TF asm number.
- Cost/risk: it forces a wider K-contraction tile (64 vs 16), which changes the LDS layout
  and the operand-read element order — aiter's 2.4× larger LDS/WG (163840 vs 69120 B) is the
  footprint that wider MMA needs, and gfx950 has the 160 KB LDS to allow it. This is a real
  re-tile of the dual-wave pipeline (the loop's route B territory), not a constant flip.
- Falsification: if a `32x32x64` PV microbench on this shape does **not** cut MFMA count ~4×
  at equal correctness, the atom is not the lever and we re-rank.

## Reproduce

```bash
# env (tracked; per CLAUDE.md build layout)
cd <FlyDSL worktree>
export PYTHONPATH="${PWD}/build-fly/python_packages:${PWD}:${PYTHONPATH}"
export LD_LIBRARY_PATH="${PWD}/build-fly/python_packages/flydsl/_mlir/_mlir_libs:${LD_LIBRARY_PATH}"
RUN=/sgl-workspace/rocm-KDA-pilot/profile/fp8-fa-vs-aiter-asm

# FlyDSL HIPREC, isolated (no --compare): discovery + PMC
HIP_VISIBLE_DEVICES=4 FLYDSL_FP8_HIPREC=1 rocprofv3 --stats --kernel-trace -f csv \
  -o "$RUN/raw/stats/flydsl_discover" -- python3 tests/kernels/test_flash_attn_fwd.py \
  --dtype fp8 --no-causal --batch 1 --seq_len 2048 --num_heads 64 --num_kv_heads 64 \
  --head_dim 128 --warmup 5 --iters 20
HIP_VISIBLE_DEVICES=4 FLYDSL_FP8_HIPREC=1 rocprofv3 -i "$RUN/pmc_input.txt" -f csv \
  -o "$RUN/raw/pmc/flydsl_pmc" -- python3 tests/kernels/test_flash_attn_fwd.py \
  --dtype fp8 --no-causal -b1 -s2048 -h64 --num_kv_heads 64 -d128 --warmup 5 --iters 20

# aiter asm, isolated (build copy only on path so aiter's flydsl import resolves):
export PYTHONPATH="${PWD}/build-fly/python_packages"
HIP_VISIBLE_DEVICES=4 rocprofv3 --stats --kernel-trace -f csv \
  -o "$RUN/raw/stats/aiter_discover" -- python3 "$RUN/aiter_asm_driver.py"
HIP_VISIBLE_DEVICES=4 rocprofv3 -i "$RUN/pmc_input.txt" -f csv \
  -o "$RUN/raw/pmc/aiter_pmc" -- python3 "$RUN/aiter_asm_driver.py"
```

`pmc_input.txt`: `pmc: SQ_INSTS_VALU SQ_INSTS_MFMA SQ_INSTS_VMEM SQ_INSTS_LDS SQ_WAVES
GRBM_GUI_ACTIVE GRBM_COUNT SQ_BUSY_CYCLES`

## Bottom line

The ~1.42× latency gap to aiter native fp8 ASM is **dominated by MMA intrinsic width**:
aiter uses `32x32x64` fp8 MMA (65536 MACs/op) and FlyDSL uses `32x32x16` (16384 MACs/op),
so FlyDSL issues **4× the MFMA instructions** (and ~1.7× the LDS feeds) for identical math,
on identical wave counts. The loop's incremental scheduling levers were correctly BLOCKED
because the real lever is the compute primitive, not the pipeline schedule. The precise,
testable next step is to re-tile the FlyDSL fp8 path onto the wider `32x32x64` MMA atom
(accepting the larger LDS staging footprint that aiter's 2.4× LDS/WG shows is required).
