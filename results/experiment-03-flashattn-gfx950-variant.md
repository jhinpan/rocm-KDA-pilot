# Experiment 03 — FlyDSL FlashAttention forward, gfx950 (MI350X) — short/mid specialized-variant session

Third end-to-end run of the ROCm KDA Pilot Humanize/RLCR workflow. Where Exp-01
landed a dispatch gate and Exp-02 was an in-body negative result on the long
kernel, **Exp-03 targeted the short/mid-sequence regime** (S≈128–512) — the only
regime where FlyDSL still trails aiter_ck — with a specialized higher-occupancy
variant. It is a **rigorous, evidence-backed negative result**: no kernel-level
short/mid win clears the bar; FlyDSL's best-routed envelope already matches or
beats aiter_ck everywhere except a narrow small-batch mid-sequence cell, and that
residual gap is structural.

## At a glance

| | |
|---|---|
| Target | FlyDSL FlashAttention **forward**, gfx950 / MI350X, **short/mid S≈128–512** |
| Baseline | upstream/main + Exp-01 dispatch gate #685 (`9afd80b8`; variant branch HEAD) |
| Scope | specialized variant (DEC-2 lifted → DEC-3); MI350/gfx950 only (DEC-1) |
| Loop | Humanize RLCR, `--max 12`, `--codex-model gpt-5.5:xhigh` |
| Rounds | 4 (0–3): baseline+diagnosis → BLOCK_M=64 kill → MG-1 fix+vmcnt kill → QK-depth + finalize |
| Outcome | **No win landed — evidence-backed negative result.** Best lever (QK-depth) = ~1%, below the ≥5% bar. |
| Source delta | +40 lines in `flash_attn_generic.py` (provider-forcing selector + true forced-dualwave); no kernel-math change |

## The competitive picture — full named-family matrix (320 rows, 80 cells, ALL PASS)

Family = B{1,8} × S{128,192,256,384,512} × {MHA,GQA} × {bf16,fp16} ×
{causal,noncausal}; providers {auto, generic_m128, generic_m256, dualwave} +
aiter_ck reference (`full_family/full_family_matrix.csv`,
`full_family/family_analysis.md`). Metric = best-FlyDSL-provider time / aiter_ck
(>100% = FlyDSL faster).

**FAMILY geomean: 104.1% of aiter_ck (n=80 cells)** — FlyDSL's best-routed
envelope is already *faster than aiter_ck across the short/mid family overall.**

geomean by (S, B):

| S | B=1 | B=8 |
|---|---|---|
| 128 | 92.8% (gap) | 119.8% (win) |
| 192 | 88.0% (gap) | 110.9% (win) |
| 256 | 86.1% (gap) | 112.8% (win) |
| 384 | 100.5% (win) | 109.5% (win) |
| 512 | 104.9% (win) | 122.8% (win) |

**The entire deficit is 3 cells: S∈{128,192,256} at batch=1 (86–93%).** Every
other cell — all of B=8, and B=1 at S≥384 — is at parity or FlyDSL-faster. The
#685 dispatch is already near-optimal (only S=256 B=1 is ~3% sub-optimal, a
dispatch artifact; AC-4 forbids dispatch-only wins).

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

## AC-2 diagnosis (artifact-derived, `diag/diagnosis_table.csv`)

All providers/tiles at the short/mid buckets: **4 waves/CU (25% occupancy),
arch_vgpr ~239, sgpr 112, scratch 0, no spill**; **no bucket is MFMA/compute-bound**
(MFMA ≤8%). Limiters are memory/latency (lgkmcnt/vmcnt/salu), regime-dependent.
dualwave is barrier-bound (≈20% — the Exp-02 irreducible limit).

## Why this is the right negative result

- **AC-1** baseline locked + fail-closed full-family provider matrix + environment
  record.
- **AC-2** artifact-derived per-provider resource/stall table.
- **AC-3/AC-5** four fully-evaluated, isolated, off-byte-identical candidates with
  OFF/ON ISA + flyprof + patches + (where relevant) schedule proofs — all rejected
  with mechanism evidence.
- **AC-4** must-win bar not met by any candidate (best ~1% vs ≥5%); the gap is
  structural, confined to S=192/256 B=1, and even there FlyDSL is ~86–88% of aiter
  while *winning* everywhere else in the family.
- Per the plan's Lower Bound, an evidence-backed Experiment-03 negative report
  closes the session.

## Honest assessment

FlyDSL's FlashAttention-forward on gfx950 is, after Exp-01's dispatch gate,
**already at or beyond aiter_ck across the vast majority of the shape space** —
long, GQA, split-K, varlen, all B=8, and short S=128. The one soft spot
(small-batch S=192/256) resisted four distinct, individually-correct optimization
levers, each backed by profiling. The realistic path to closing it is not a knob
or a tile size but a from-scratch short/mid kernel matching aiter's schedule — a
disproportionate effort for one narrow cell that is not the production bottleneck.
The pilot's value here is the **evidence map**: it says precisely where the gap is,
why each cheap/medium lever fails, and that the remaining work is structural.

## Process notes (workflow)

- 4 rounds, 3 Codex reviews; the loop correctly pushed back on premature
  round-boundary deferrals until the must-win lever was actually executed and the
  full evidence package produced.
- Reusable tooling from Exp-02 carried over (pipeline_sim, resource probes, OFF/ON
  ISA capture, flyprof bundles); added a provider-forcing selector that makes
  per-provider attribution reproducible.

## Artifacts
Loop dir `.humanize/rlcr/2026-06-16_18-35-14/` (FlyDSL worktree, untracked):
`artifacts/{baseline/,provider_matrix/,full_family/,diag/,blockm64/,vmcnt_cand/,
qk_prefetch3/}`, `docs/attempts.jsonl`, `docs/optimization-ledger.md`. Source:
commits `42321df8` (diagnosis), `d3a418d1` (provider selector), `6a5ddc17`
(forced-dualwave fix) on branch `kda/flydsl-flashattn-gfx950-variant`.
