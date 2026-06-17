# Loop 03 — FlyDSL FlashAttention forward, gfx950 (MI350X) — short/mid specialized-variant loop

Third end-to-end run of the ROCm KDA Pilot Humanize/RLCR workflow. Where Loop 01
landed a dispatch gate and Loop 02 was an in-body `NO-GO` on the long
kernel, **Loop 03 targeted the short/mid-sequence regime** (S≈128–512) — the only
regime where FlyDSL still trails aiter_ck — with a specialized higher-occupancy
variant. It is a **rigorous, evidence-backed NO-GO**: no kernel-level short/mid
win clears the bar; FlyDSL's best-routed envelope already matches or
beats aiter_ck everywhere except a narrow small-batch mid-sequence cell, and that
residual gap is structural.

## At a glance

| | |
|---|---|
| Target | FlyDSL FlashAttention **forward**, gfx950 / MI350X, **short/mid S≈128–512** |
| Baseline | upstream/main + Loop 01 dispatch gate #685 (`9afd80b8`; variant branch HEAD) |
| Scope | specialized variant (DEC-2 lifted → DEC-3); MI350/gfx950 only (DEC-1) |
| Loop | Humanize RLCR, `--max 12`, `--codex-model gpt-5.5:xhigh` |
| Rounds | a multi-round RLCR loop (baseline+diagnosis → BLOCK_M=64 kill → MG-1 fix+vmcnt kill → QK-depth → AC-6+report → evidence/provenance finalization); see the loop goal-tracker for the exact round/review history |
| Outcome | **NO-GO — no win landed.** Best lever (QK-depth) = ~1%, below the ≥5% bar. |
| Source delta | `flash_attn_generic.py`: provider-forcing selector + true forced-dualwave + pure dispatch-predicate refactor; `tests/unit/test_flash_attn_dispatch_routing.py`: no-GPU routing test. Byte-identical when unset; no kernel-math change. |

## The competitive picture — full named-family matrix (320 rows, 80 cells, ALL PASS)

Family = B{1,8} × S{128,192,256,384,512} × {MHA,GQA} × {bf16,fp16} ×
{causal,noncausal}; providers {auto, generic_m128, generic_m256, dualwave} +
aiter_ck reference (`full_family/full_family_matrix.csv`,
`full_family/family_analysis.md`). Two metrics (time / aiter_ck, >100% = FlyDSL
faster): **current-auto** (what the #685 dispatch actually runs) and
**best-forced-provider** (the achievable FlyDSL envelope).

**Family geomean: current-auto = 101.7%, best-forced = 103.7%** (n=80 cells).
FlyDSL is faster than aiter_ck on the family overall — but current-auto is *not*
the best provider in every cell (see dispatch sub-optimality below).

geomean by (S, B), current-auto | best-forced:

| S | B=1 (auto / best) | B=8 (auto / best) |
|---|---|---|
| 128 | 88.8% / 92.6% (gap) | 118.9% / 119.6% (win) |
| 192 | 87.1% / 87.4% (gap) | 100.8% / 110.9% (win) |
| 256 | 83.8% / 86.1% (gap) | 112.0% / 112.7% (win) |
| 384 | 100.3% / 100.1% (win) | 108.6% / 109.4% (win) |
| 512 | 104.5% / 104.3% (win) | 119.0% / 121.4% (win) |

**The kernel-level deficit is the small-batch cells S∈{128,192,256} at B=1**
(~84–93%) — where even the best provider trails aiter. Everything else (all B=8,
B=1 S≥384) is parity-or-faster.

**Correction vs the round-3 draft:** that draft claimed the #685 dispatch is
"near-optimal." The full matrix shows otherwise — **current-auto is >2% slower
than the best forced provider in 24/80 cells** (up to ~22%, e.g. B=8 S=192 GQA
non-causal: auto routes to generic at 34.8µs but dualwave does 30.1µs; B=8 S=512
GQA fp16 non-causal: auto 114.4µs vs dualwave 93.4µs). This is a **real
dispatch-only follow-up** (the #685 gate could route some B=8/GQA short/mid cells
to dualwave) — but **AC-4 excludes dispatch-only wins**, so it is recorded as
queued follow-up, not claimed as this loop's win. (`full_family/family_analysis.md`.)

## Levers tried at S=192/256 B=1 — all sub-threshold (≥5% bar)

| lever | result | mechanism evidence |
|---|---|---|
| **BLOCK_M=64** (occupancy / small tile, original DEC-4) | DEAD | runtime arch_vgpr 239 / 25% occupancy = identical to M128; does NOT cross the 2-CTA cliff; slowest provider everywhere, catastrophic at B=8 |
| **env knobs** (PREFETCH3 × REDUCE_MODE, round 0) | FLAT | no median move; PREFETCH3 worse |
| **vmcnt-only K-swap** (round 2) | NEUTRAL | +0.0% medians; OFF/ON flyprof shows it *reshuffles* lgkmcnt→vmcnt, net-neutral |
| **QK prefetch-depth 2→3** (round 3, the best) | **+0.7–1.4%** | flyprof: lgkmcnt 28.6%→non-dominant at S=256; real but ~1%, below the 5% bar |

Even the *correct* lever (QK-depth hit the diagnosed lgkmcnt stall) yields only
~1%. This is strong evidence the residual ~13–15% gap to aiter_ck is **structural**
(aiter uses a fundamentally different schedule/algorithm at small-batch mid-seq),
not reachable by incremental FlyDSL schedule tuning.

## AC-2 diagnosis (artifact-derived per-provider table, `diag/diagnosis_table.csv`)

Per-provider flyprof + trace across family cells (forced generic_m128 /
generic_m256 / dualwave). All at **4 waves/CU (25% occupancy)**, sgpr 112,
no spill; **no bucket is MFMA/compute-bound** (MFMA ≤8%). The limiter differs by
provider/tile:
- **generic_m128** (vgpr 120, lds 49152): lgkmcnt/vmcnt-bound (lgkmcnt ~26–28% at
  S=192/256; vmcnt ~25–29% at S=512/B=8).
- **generic_m256** (vgpr 116): barrier-bound (~21–23%) — the 512-thread tile is
  the wrong shape for short/mid.
- **dualwave** (vgpr 128, lds 68096): barrier-bound (~23%) — the Loop 02
  irreducible limit.

## Why this is the right NO-GO

- **AC-1** baseline locked + fail-closed full-family provider matrix (320 rows) +
  refreshed environment/provenance record.
- **AC-2** artifact-derived per-provider resource/stall table over **representative**
  family cells (3 providers × 5 cells, bf16 causal — the gap buckets S=192/256, a
  protected S=512, a B=8; full-family-all-dtype profiling not exhaustively run, as
  the candidate verdict was already settled by the full-family timing matrix).
- **AC-3/AC-5** four fully-evaluated, isolated, off-byte-identical candidates with
  OFF/ON ISA + flyprof + saved patches + schedule proofs — all rejected with
  mechanism evidence.
- **AC-6** no-GPU routing-predicate unit test (16 cases, PASS) + full correctness
  matrix **668 PASS / 0 FAIL**: dense 432 (S{1..8192} × causal/noncausal ×
  bf16/fp16 × MHA/GQA/MQA × B{1,8}), default-sweep+varlen 232 (incl. varlen 16/16),
  split-K 2, `FLYDSL_DISABLE_DUALWAVE_SWP=1` 2. gfx942 fallback: no gfx942 hardware
  here → covered by the no-GPU `_routes_to_dualwave(dualwave_available=False)` test
  + the builder's `gpu_arch.startswith("gfx950")` guards (limitation stated).
- **AC-4** must-win bar not met by any candidate (best ~1% vs ≥5%); the gap is
  structural, confined to small-batch S=128/192/256 B=1, and even there FlyDSL's
  best provider trails aiter only while *winning* everywhere else in the family.
- Per the plan's Lower Bound, an evidence-backed Loop 03 `NO-GO` report closes
  the loop. (A real **dispatch-only** follow-up exists — 24/80 auto
  cells are sub-optimal — but AC-4 excludes dispatch-only wins; it is queued.)

## Honest assessment

FlyDSL's FlashAttention-forward on gfx950 is, after Loop 01's dispatch gate,
**already at or beyond aiter_ck across the vast majority of the shape space** —
long, GQA, split-K, varlen, all B=8, and short S=128. The one soft spot
(small-batch S=192/256) resisted four distinct, individually-correct optimization
levers, each backed by profiling. The realistic path to closing it is not a knob
or a tile size but a from-scratch short/mid kernel matching aiter's schedule — a
disproportionate effort for one narrow cell that is not the production bottleneck.
The pilot's value here is the **evidence map**: it says precisely where the gap is,
why each cheap/medium lever fails, and that the remaining work is structural.

## Process notes (workflow)

- A multi-round RLCR loop (exact round/review counts in the loop goal-tracker); the
  loop correctly pushed back on premature round-boundary deferrals and "COMPLETE"
  overclaims until the must-win lever was actually executed and the full evidence
  package produced. The terminal state is a **non-COMPLETE Lower-Bound `NO-GO`
  accepted by the user (DEC-8)** — not an all-ACs-met COMPLETE (AC-4 is
  explicitly not met).
- Reusable tooling from Loop 02 carried over (pipeline_sim, resource probes, OFF/ON
  ISA capture, flyprof bundles); added a provider-forcing selector that makes
  per-provider attribution reproducible.

## Artifacts
Loop dir `.humanize/rlcr/2026-06-16_18-35-14/` (FlyDSL worktree, untracked):
`artifacts/{baseline/,provider_matrix/,full_family/,diag/,blockm64/,vmcnt_cand/,
qk_prefetch3/, ac6_v2/}`, `docs/attempts.jsonl`, `docs/optimization-ledger.md`.
FlyDSL source commits on `kda/flydsl-flashattn-gfx950-variant` (baseline
`9afd80b8`): `42321df8` (diagnosis), `d3a418d1` (provider selector), `6a5ddc17`
(forced-dualwave fix), `4a7167ea` (r3 boundary), `99280827` (pure dispatch-predicate
refactor + no-GPU routing test), `c3c7002a` (r4 boundary), `8a795c5c` (last
non-empty source change: stale-comment cleanup), `2c1a231c` (r5 boundary),
`bea21e39` (r6 boundary), `4257ce77` (r7 boundary). The current FlyDSL HEAD and the
final pushed commit of this report are recorded exactly in the loop's git-ignored
provenance (`.humanize/rlcr/2026-06-16_18-35-14/artifacts/baseline/environment.txt`
and `docs/optimization-ledger.md`) — a tracked file cannot stably embed its own
final commit SHA, so the exact endpoint is kept there. Prior report chain:
rocm-KDA-pilot `bf81848` (initial) → `8a346ca` (r4 dispatch correction) →
`8ca80d9` (r5 provenance) → `d66f6ae` (r6 framing) → subsequent provenance
amendments.
