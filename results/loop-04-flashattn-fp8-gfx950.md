# Loop 04 — FlyDSL fp8 FlashAttention forward, gfx950 (MI350X)

The second example family in this repo: a **new dtype** (fp8 e4m3fn), not another
tuning pass on the existing bf16 kernel. Loops 01–03 optimized the *existing*
bf16/f16 FlashAttention; Loop 04 *adds* an fp8 forward path from scratch and tries
to push it to aiter-asm-fp8 throughput parity.

## At a glance

| | |
|---|---|
| Target | FlyDSL FlashAttention **fp8 (OCP `float8_e4m3fn`) forward**, gfx950 / MI350X, D=128 dual-wave SWP |
| Issue / draft | [ROCm/FlyDSL#698](https://github.com/ROCm/FlyDSL/issues/698) — *[Feature] fp8 flash attention* |
| Baseline / harness | ROCm/FlyDSL #683 (`DUALWAVE_SWP` forward + `test_flash_attn_fwd.py`) |
| Parity target | aiter **native ASM fp8** (`fmha_v3_fwd`, `how_v3_bf16_cvt=0`, `fwd_hd128_fp8.co`) |
| Worktree | `jhinpan/FlyDSL-lab` @ `feat/fp8-flash-attn-fwd` |
| Loop | Humanize RLCR, `--max 12`, `--codex-model gpt-5.5:xhigh` |
| Rounds used | 12 of 12 (ran to max-iterations); ~66 candidates (C0–C66) |
| Outcome | **IMPROVEMENT — correct, additive, merge-ready fp8 forward**; **AC-8 (asm-fp8 throughput parity) OPEN at ~66%** |
| Commit | squashed to `888ae17f` (1 feat commit); design notes in `docs/fp8_flash_attn_design.md` |

## What landed

A complete **fp8 e4m3fn forward** on the gfx950 D=128 dual-wave SWP kernel, additive
to the existing dtypes (`bf16`/`f16` byte-identical when `dtype != fp8`):

- Pre-quantized Q/K/V + per-tensor shape-`[1]` fp32 descales `q/k/v_descale`
  (mirrors aiter's per-tensor fp8 ABI).
- QK via the native `mfma_f32_32x32x16_fp8_fp8` atom; `q_descale*k_descale*sm_scale`
  applied to **fp32** logits. Online softmax max/sum and the PV accumulator stay
  **fp32**. Output **bf16**.
- **Two mutually-exclusive PV operand-precision modes** (invalid env combos fail
  fast):
  1. **high-precision-P (default, shipping):** fp8 V dequantized to bf16 in-kernel,
     bf16 PV MMA. `min_cos ≈ 0.99999`.
  2. **packed-fp8-P (`FLYDSL_FP8_PV_FROMBF16`, opt-in):** genuine `fp8×fp8` PV MMA
     reusing the *proven bf16 V/P element order*, both operands quantized to fp8 at
     the MMA. `min_cos ≈ 0.9986`.
- Routing: fp8 enters the dual-wave SWP builder directly (`head_dim==128`, gfx950,
  dense). fp8 **split-K** (`num_kv_splits>1`) and **packed varlen** (`cu_seqlens`)
  are **rejected with a clear error**, not silently skipped.

Net diff vs baseline: 9 files, ~+1.5k/-150 (kernel + harness + 2 docs + 1 routing
unit test).

### Measured result (MI350X, gfx950)

Correctness vs a **dequantized-input** PyTorch SDPA reference, fixed fp8 gate
`max_err < 5e-2 AND min_cos > 0.98`, no FAIL/ERROR:

| sweep | result |
|---|---|
| fp8 default (high-precision-P), full DEFAULT_CONFIGS, causal+nocausal, MHA+GQA (Hkv∈{8,16,32,64}), S≤8192 | **ALL PASS**, `min_cos ≈ 0.99999`, `max_err ≤ 4.3e-3`, 0 NaN |
| fp8 FROMBF16 (packed-fp8-P) | ALL PASS, `min_cos ≈ 0.9986` |
| bf16 / fp16 non-regression | unchanged, `min_cos ≈ 0.99999 / 1.00000`; bf16 ≈703 TF (≥0.98× baseline) |
| full repo gate `RUN_TESTS_FULL=1 scripts/run_tests.sh` (private build) | pytest + examples + MLIR FileCheck all green |

Throughput (B=1, S=2048, H=64, D=128, non-causal, warmup 10 / iters 50):

| path | TFLOPS | % of aiter asm | min_cos |
|---|---|---|---|
| aiter asm fp8 (**parity target**) | ~1306 | 100% | 0.9993 |
| aiter ck fp8 | ~919 | 70% | 0.9993 |
| **FlyDSL fp8 (high-precision-P, default)** | ~863 | **66%** | 0.99999 |
| FlyDSL fp8 (FROMBF16 packed-P) | ~620 | 47% | 0.9986 |

## Candidate ledger (summary of ~66 candidates, C0–C66)

The loop ran to max-iterations; the full per-candidate ledger lives in the
worktree's untracked `.humanize` notes. Grouped:

| group | disposition | note |
|---|---|---|
| Harness fp8 wiring (C0–C1) | landed | `--dtype fp8`, e4m3fn inputs + descales, dequant-SDPA reference, fixed gate, honest PASS/FAIL/ERROR, native-ASM comparator |
| LDS element-width re-derivation (C2–C6) | landed | `BF16_BYTES=2` → `ELEM_BYTES`; **found `mfma_f32_32x32x16_fp8_fp8` exists** (same geometry as bf16 — corrected an earlier "must rewrite to 16×16×32" conclusion) |
| descale ABI plumbing (C3–C5) | landed | kernel sig + launcher + public entry; bf16 path bit-identical (`max_err=0.0`) |
| fp8 V-staging layout (C9–C25) | **hard debug** | built an **operand-dump oracle** (`vt_vs_dma_compare.py`) to decode the exact bf16 DMA `{8j+b}` 4-D interleave; per-wave staging mismatched the wave-shared tile → 50% NaN |
| high-precision-P PV (default) (→C58) | **PROMOTED** | the shipping correct path |
| "fp8-P precision wall" NO-GO (C56–C57) | **retracted** | see Lessons #1 |
| packed-fp8-P FROMBF16 (C64–C65) | landed (correctness) | reuse proven layout by construction; on-device `fp8×fp8` PV passes the gate |
| AC-10 merge gate (C59–C63) | landed | fixed a real fp8 legalization regression + a `conftest`/shared-build issue; full suite green on a private build |
| asm-fp8 throughput parity (AC-8) (C66) | **OPEN** | 66%; true-fp8 no-roundtrip V path is the remaining lever |

## Per-path bottleneck profile (rocprofv3 ATT, MI350X)

The default fp8 path is **stall-bound, not compute-bound** — MFMA issue is a small
fraction of cycles:

| stall class | high-precision-P | FROMBF16 |
|---|---|---|
| total stall | 64.7% | 56.4% |
| **barrier** | **33.6%** | 42.0% |
| vmcnt (global-load wait) | 20.5% | 13.9% |
| valu | 13.9% | 23.1% |
| lds | 12.6% | 7.9% |
| **mfma** | **7.6%** | **4.6%** |

Two readings: (1) the gap to aiter asm is a **scheduling/synchronization** problem
(`s_barrier` + global-load waits), not arithmetic throughput; (2) FROMBF16's `valu`
jump (13.9→23.1%) is the per-MMA bf16↔fp8 conversion — confirming it is a
correctness vehicle, not the perf path.

## Honest assessment — why parity is still open

The default path does **bf16 compute on fp8 I/O**, so it is ~0.9× FlyDSL bf16 —
no fp8 speedup by construction. fp8 and bf16 `32×32×16` MMA have *equal* CDNA4
throughput, so packing to fp8 only helps once the **bf16 round-trip and per-MMA
conversion are removed**. Reaching ~1306 TF requires a **true-fp8 V path**: fp8 V
staged and read directly in LDS (half the `ds_read` bandwidth), no round-trip. The
blocker is that the 8-bit transpose load (`ds_read_b64_tr_b8`) permutes lanes
differently from the proven bf16 `tr_b16 + shuffle`, so the true-fp8 B-operand
element order does not yet match the proven layout. This is a **tractable
implementation gap, not a hardware wall** (see Lesson #1).

Per the plan's **DEC-4 single-step merge gate** (a mergeable PR requires correctness
*and* asm-fp8 parity in one PR), the full feature is **not "done."** But the
high-precision-P path is a correct, additive, merge-ready functional fp8 forward —
i.e. the natural content of a first PR if DEC-4 is later split into two steps.

## Lessons learned

1. **Verify a "fundamental hardware limit" with cheap idealized numerics before
   accepting a NO-GO.** Mid-loop, an experimental fp8-V path capped at `min_cos
   ≈ 0.87`, and the loop recorded an AC-8 NO-GO on a claimed "e4m3 3-bit-mantissa
   precision wall." A host-side check of the *exact* `fp8×fp8` PV MMA falsified it:
   packing the **unnormalized** `exp(S−m)` reaches `min_cos ≈ 0.9996` (well above
   the gate); only packing **normalized** probabilities collapses (tiny values land
   in e4m3 subnormals). The "wall" was a layout/indexing defect. **A wrong NO-GO is
   worse than no result — always probe the claimed limit in faithful numerics
   first.**
2. **When a memory layout is wrong, build a deterministic oracle — don't guess.** The
   50%-NaN / low-cos fp8-V bug was nailed by a standalone comparator that DMAs a
   `V[key,d]=ramp` through the *real* bf16 path and through the candidate staging,
   then diffs byte-for-byte to decode the exact `{8j+b}` interleave. Guess-and-check
   on a 4-D LDS interleave is multi-cycle and error-prone; the oracle made it
   deterministic.
3. **Reuse a proven datapath "by construction" instead of re-deriving from scratch.**
   The shipping correctness win (FROMBF16) came from routing fp8 through the
   *already-proven* bf16 V transpose + P machinery and only changing the numeric
   format at the MMA — guaranteeing operand element-order correctness — rather than
   hand-replicating the 4-D DMA interleave for a from-scratch fp8 path.
4. **Profile to re-aim the goal, not just to confirm it.** The intuitive "fp8 ⇒
   faster MFMA" framing was wrong here: ATT traces show MFMA at ~8% and barriers at
   ~34%. The parity lever is **V bandwidth + synchronization**, not arithmetic — which
   redirects all remaining optimization work.
5. **Check the raw bindings, not just the curated wrapper.** An early conclusion that
   fp8 forced a `16×16×32` geometry rewrite (vs the bf16 `32×32×16` SWP) was wrong:
   `mfma_f32_32x32x16_fp8_fp8` exists in the raw ROCDL bindings (only the curated
   `expr/rocdl` wrapper omitted it). This preserved the accumulator shape, softmax
   reductions, and swizzle, turning a "rewrite" into an atom swap.
6. **Test against a current-source build, not stale shared bindings.** The AC-10 full
   suite first appeared "blocked" because this worktree's `build-fly` is a **symlink
   to a shared build** that ~10 sibling worktrees depend on and that was stale vs
   current source. A private `FLY_BUILD_DIR` build then surfaced a **real** fp8
   legalization regression (fp8-typed source memref rejected by `BufferCopyLDS128b`;
   fixed by building fp8 Q/K/V buffer views as `i8`). Stale shared bindings had
   hidden a genuine mergeability bug. A repo-local `conftest.py` fix (honor
   `FLY_BUILD_DIR`) was also required so pytest used the private build.
7. **`NO-GO`/`OPEN` on a hard gate is a valid, honest deliverable.** The loop
   correctly refused to mark a failing accuracy gate as "complete." (It also exposed
   a workflow-friction: the Stop-hook's "incomplete tasks" block looped against this
   honest refusal until the block-cap fired — surfaced upstream as methodology
   feedback; the fix is hook-side, not raising the cap.)

## Process cost (for workflow tracking)

- 12/12 rounds (ran to max-iterations); the depth was real (V-layout decode, the
  precision-wall retraction, two PV modes, the private-build AC-10 gate), not churn.
- Recoverable losses worth noting:
  - One **false NO-GO** (precision wall) cost rounds before being retracted by a
    ~30-line host probe — Lesson #1 is the cheap preventative.
  - The **shared `build-fly` symlink** caused an apparent AC-10 block and a real
    hidden regression — Lesson #6; the prepare script should provision a private
    build tree for any loop that must run the full suite.
  - The **Stop-hook ↔ honest-failure** loop burned block-cap cycles at each
    convergence attempt — hook-side fix needed.
