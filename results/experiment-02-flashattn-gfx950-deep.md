# Experiment 02 — FlyDSL FlashAttention forward, gfx950 (MI350X) — Deep / in-body session

Second end-to-end run of the ROCm KDA Pilot Humanize/RLCR workflow. Where
Experiment 01 landed a **dispatch-gate** win, this session was scoped to go
**inside the kernel body** and chase a #683-style **broad** speedup. It is a
rigorous **negative result**: no in-body win is landable on this kernel, and we
now have the profiling + ISA evidence to say *why*.

## At a glance

| | |
|---|---|
| Target | FlyDSL FlashAttention **forward**, gfx950 / MI350X |
| Baseline | ROCm/FlyDSL #683 + the Experiment-01 dispatch gate #685 (`9afd80b8`) |
| Scope | **In-body only** (DEC-2), **must-win** goal (DEC-1: MI350/gfx950 only; MI355X = planned follow-up, not measured) |
| Loop | Humanize RLCR, `--max 12`, `--codex-model gpt-5.5:xhigh` |
| Rounds used | 3 build/review rounds, cancelled after the in-body lever set was exhausted |
| Outcome | **No optimization landed — negative result.** Kernel body byte-identical to baseline. |
| Follow-up | **Session 3**: specialized short/mid variant (DEC-2 lifted by user re-decision DEC-3) |

## Conclusion

**No landable in-body win exists on the `DUALWAVE_SWP` forward kernel.** Every
credible in-body lever was evaluated with profiling/resource evidence and
rejected. The kernel is a hand-scheduled, long-sequence latency-hiding pipeline
sitting at a **barrier/occupancy co-optimum**: pinned at 128 VGPR/thread (no
occupancy headroom, 4 waves/CU = 25%), with a correctness-critical
`s_waitcnt`/`s_barrier` schedule whose top stall (barrier, 22–34%) is *structural
to the design*, not a cheap mistake.

## Levers evaluated and why each failed

| Lever | Disposition | Evidence |
|---|---|---|
| **Occupancy / VGPR reduction** | not viable (analysis) | 2-CTA cliff needs ≤64 VGPR/thread; `v_o[0..3]` MFMA accumulators alone = 64 VGPR + Q residency ~32 + loop-carried softmax/K/V state. Safe savings ≈ 8–24 VGPR — nowhere near. No intermediate cliff for a 512-thread / 8-wave CTA. |
| **Barrier removal** (C1→C2 / C5→C6) | correctness **FAIL**, reverted | Removing the 2 compute→memory `s_barrier`s (ISA s_barrier 25→23) → max_err ~1.8e-2–2.4e-2, min_cos ↓ 0.959. They are **load-bearing** cross-wave LDS producer/consumer barriers. |
| **LDS bank / padding** (long_gqa) | NO-GO (gate not met) | The `lds=24%` stall is DMA-to-LDS completion latency (`buffer_load_dwordx4 … offen lds`), **not** ds_read bank conflicts. Padding already engineered (K_PAD 16B, V_PAD 64B). |
| **Prefetch-depth** (`NUM_PREFETCH_K` 2→3) | **EXECUTED → REJECTED (measured + ISA)** | Faithful K-only triple-buffer, schedule proven RAW/WAR-safe by a symbolic simulator, **correct on first run**. Result below. |
| **env knobs / waves_per_eu** | exhausted (Exp 01) | defaults optimal; waves_per_eu 3/4 → NaN. |

### The prefetch-depth result (the round-3 deep candidate)

A faithful triple-buffer (`NUM_PREFETCH_K = 3`), isolated behind a
`dualwave_swp_prefetch3` flag (off = byte-identical baseline). K uses a runtime
buffer id `tile % 3` and is prefetched one tile further ahead; V keeps its proven
depth-2 schedule. The silent-corruption-prone LDS schedule was de-risked with a
**symbolic pipeline simulator** (validated against the known-correct depth-2
schedule, then proven RAW/WAR-safe across prologue + loop + the 14-cluster drain)
**before** any GPU run — it was correct on first execution.

**Measured (MI350X, gfx950):**

| bucket | baseline | depth-3 | Δ |
|---|---|---|---|
| S=512 causal (mid) | 17.8µs | 18.6µs | **+4.5% (slower)** |
| S=8192 causal (long, highest vmcnt) | 195.3µs | 206.8µs | **+5.9% (slower)** |

Correct but slower on every shape, including the one with the most DMA-wait
headroom. **OFF vs ON ISA (`final_isa.s`):**

| metric | OFF | ON | Δ |
|---|---|---|---|
| s_barrier | 25 | 25 | **0** |
| s_waitcnt | 17 | 17 | **0** |
| .vgpr_count | 254 | 254 | **0** |
| group_segment (LDS) | 68096 | 102144 | +34048 B |
| buffer_load | 32 | 34 | +2 |

**Root cause, ISA-confirmed:** deeper prefetch leaves the binding
`s_barrier`/`s_waitcnt` counts untouched and adds only DMA-issue work + LDS. The
kernel is **barrier-bound, not DMA-bound**, so it can only regress. VGPR is
unchanged (the prefetch is direct global→LDS DMA, so no spill class), and the
extra LDS is free (occupancy is VGPR-bound) — but neither helps.

## Per-bucket bottleneck profile (flyprof, MI350X)

Uniform across all 7 buckets: **vgpr_count = 128/thread, accum = 0, 4 waves/CU
(25% occupancy)**. Top stall is **barrier (22–34%, #1 everywhere)**, then
vmcnt/vmem; MFMA never top-3. The residual competitive gap is the **short/mid**
regime — where this long-sequence-shaped kernel is simply the wrong tool.

## Why this is the right negative result (not a premature stop)

Unlike Experiment 01 (which took the cheapest dispatch-only path), this session
was *forced* to exhaust the in-body levers — including building, proving correct,
and measuring the prefetch-depth candidate the prior reviewer flagged as
"untried." The negative result is now **evidence-backed at the ISA level**, which
is exactly the signal that the next lever must be **structural** (a specialized
variant), not in-body surgery.

## Process cost (for workflow tracking)

- 3 build/review rounds. The main process tension: the loop reviewer (Codex,
  which reads only files) could not see the user's out-of-band re-decisions made
  via the harness's question channel, so it kept demanding the in-body lever set
  be completed before any session-closure. Resolution required the user to
  adjudicate and then `cancel-rlcr-loop` (the loop is user-only to end).
- A **symbolic pipeline simulator** (no GPU) caught a real WAR-clobber hazard in
  the first triple-buffer sketch before any compile — cheap, high-value; worth
  promoting as a standard pre-flight for any LDS-pipeline change.

## Lessons learned (pilot-workflow)

1. **A "must-win + in-body-only" scope can legitimately end in a negative
   result.** The workflow needs a first-class, evidence-gated way to *conclude*
   "no in-body win exists" without it reading as a premature stop — and to let an
   out-of-band user re-decision (lift a DEC, switch sessions) be visible to the
   loop reviewer.
2. **Simulate LDS pipeline schedules before emitting code.** A ~120-line symbolic
   RAW/WAR simulator turned a "high correctness risk" coupled rewrite into a
   first-try-correct candidate. Make it a standard step for buffer/prefetch work.
3. **Capture OFF/ON `final_isa.s` for every kernel-body candidate.** The s_barrier
   / s_waitcnt / vgpr / LDS deltas are what turn "it regressed" into "it regressed
   *because the kernel is barrier-bound and this lever doesn't touch barriers*."
4. **Negative results are deliverables.** This experiment's value is the
   evidence-backed map of *why* the kernel resists in-body optimization, which
   directly scopes Session 3 to the variant path.

## Artifacts

Loop dir `.humanize/rlcr/2026-06-16_06-10-39/` (in the FlyDSL worktree, untracked):
`SESSION2_NEGATIVE_RESULT.md`, `round-3-summary.md`,
`round-3-task7-hazard-model.md`, `round-3-task7-evidence.md`,
`round-3-task9-promotion-review.md`, `artifacts/pipeline_sim.py`,
`artifacts/prefetch3_candidate/{prefetch3_depth3.patch, off_final_isa.s,
on_final_isa.s, isa_off_vs_on.md}`, `docs/attempts.jsonl`,
`docs/optimization-ledger.md`.
