# Humanize Gen-Plan Draft: FlyDSL FlashAttention Forward On gfx950 — Deep Loop 02

Use this draft to generate a Humanize RLCR plan for a **second, deeper** loop of
optimizing FlyDSL FlashAttention forward on AMD gfx950 / MI350X-MI355X.

This draft is for the case where Loop 01 already landed a safe but **narrow** win
(see `results/loop-01-flashattn-gfx950.md`) and we now want **broad,
kernel-level** speedups across many shapes — the way PR683 itself was broad — not
another single-case dispatch tweak.

> Read `results/loop-01-flashattn-gfx950.md` before generating the plan.
> The Loop 01 profiling and candidate ledger are required context.

## Source Refs

- FlyDSL upstream: https://github.com/ROCm/FlyDSL  (PR683 is **merged into `main`**)
- FlyDSL working fork: https://github.com/jhinpan/FlyDSL-lab
- Loop 01 result: `results/loop-01-flashattn-gfx950.md`
- Loop 01 upstream PR (the dispatch gate): ROCm/FlyDSL #685
- Observed on: {{DATE}}
- FlyDSL upstream/main SHA: {{FLYDSL_MAIN_SHA}}
- Loop 01 landed commit (dispatch gate): {{ROUND1_GATE_SHA}}
- Current worktree HEAD: {{WORKTREE_HEAD}}
- Current branch: {{WORKTREE_BRANCH}}

## Baseline change vs Loop 01

Loop 01 compared against the PR683 *head*. **This deep loop compares directly against
`upstream/main`** (PR683 is merged) **plus the Loop 01 dispatch gate already
applied**. Concretely the baseline is: upstream/main + #685's gate. The goal is
to beat *that* — so a re-derivation of the short-seq dispatch win does NOT count
as progress this loop.

## Ultimate Goal

Achieve **broad** FlashAttention-forward speedups on gfx950 by optimizing the
**kernel body** (`flash_attn_gfx950.py`), not just dispatch routing. "Broad"
means measurable improvement across many of the required shape buckets — mid and
long sequences, MHA and GQA, both dtypes — closing the remaining gap to
aiter_ck / aiter_asm on the buckets where FlyDSL still trails or only ties.

Correctness and coverage remain hard gates. Performance breadth is the objective.

## Kernel Contract

K:

- Primary target file: **`kernels/flash_attn_gfx950.py`** (the DUALWAVE_SWP
  kernel body). This loop MUST land at least one kernel-body change — a
  dispatch-only or harness-only result does NOT satisfy the lower bound.
- `kernels/flash_attn_generic.py` may be touched only if a kernel-body change
  requires a matching dispatch update; do not re-litigate the Loop 01 gate.
- `python/flydsl/expr/rocdl.py` only if a new intrinsic is genuinely required.
- `tests/kernels/test_flash_attn_fwd.py` only for measurement output / harness
  fixes, never to weaken the gate.
- Preserve everything PR683 + #685 already provide: dualwave SWP path, the
  dense short-seq dispatch gate, arbitrary seq_len, packed varlen, GQA/MQA,
  split-K, dtype coverage, gfx942 fallback.

R: (unchanged hard gates)

- Reference: PR683 harness vs PyTorch SDPA / chunked SDPA.
- `max_err < 1e-2` AND `min_cos > 0.99`; no FAIL/ERROR rows; no failure→SKIP.

W: (required sweeps — same as Loop 01, plus breadth scoring below)

- `DEFAULT_CONFIGS` and `VARLEN_CONFIGS`, bf16 + fp16, causal + non-causal,
  MHA + GQA. Split-K focus configs when split-K is touched.
- Compare against the deep-loop baseline (upstream/main + #685 gate) and
  aiter_ck / aiter_asm.

## Pre-authorized deep optimization directions

These are the structurally-deep levers identified by Loop 01 profiling. They are
**inherently multi-step** — the plan should treat each as a *milestone with
sub-steps*, NOT force them into a single "one isolated change" candidate. Each
milestone still ends in a full correctness gate before its perf counts.

Profiling fact base (from Loop 01, MI350X): short/mid dense buckets are
memory-bound (vmcnt + s_barrier dominant, MFMA only ~3-5%); occupancy capped at
4 waves/CU, register-limited; long/GQA/split-K latency-bound.

1. **Occupancy / VGPR-footprint reduction** (highest upside, highest risk).
   Goal: lift occupancy above 4 waves/CU. Sub-steps: measure VGPR allocation and
   the occupancy threshold; move the A/B tile staging through LDS via async copy
   to free architectural VGPRs; re-validate occupancy actually rose before timing.
   Guardrail: do NOT force maxnreg to push accum_vgpr=0 (Loop 01 note: ~4.5x
   spill regression).
2. **LDS double-buffer prefetch-depth re-architecture** (the *correct* version of
   the failed Loop 01 VMEM-prefetch candidate). Raising prefetch *depth* (not just
   relaxing a waitcnt) requires extending the `[K0][V0][K1][V1]` buffer layout and
   its address arithmetic + waitcnt schedule together. Treat as one coupled but
   well-scoped milestone with a stated hazard model.
3. **Provable barrier relaxation.** `s_barrier` is the top stall (20-32%). Remove
   exactly one barrier at a time, each with an explicit producer/consumer LDS
   ownership proof and repeated correctness stress (including varlen/split-K).
4. **LDS layout / bank-conflict tuning** (only if bank-conflict evidence is
   produced for a specific read path).
5. **Finer dispatch for S=256** (cheap, allowed, but does NOT by itself satisfy
   the kernel-body lower bound).

## Profiling decision points

Profile to answer a NAMED question before each kernel-body edit, and again after
to confirm the targeted bubble shrank. Required first step each milestone:
capture VGPR/SGPR/LDS/occupancy/spill + the stall taxonomy for the milestone's
target bucket, so "did occupancy rise / did the bubble shrink" is decidable.

## Promotion criteria — REWARD BREADTH (the key change from Loop 01)

- **`IMPROVEMENT` lower bound:** at least one **kernel-body** change in
  `flash_attn_gfx950.py` that improves a **named family of buckets** (e.g. all
  long MHA, or all GQA) by a repeatable median margin, with **no required bucket
  regressing beyond ~2-3% noise**, correctness fully preserved. A dispatch-only
  or knob-only result does NOT satisfy the lower bound this loop.
- **Upper bound (aim for):** a kernel-body change (or small set) that improves the
  **overall geomean across the full required sweep** AND beats the deep-loop
  baseline on a majority of buckets, narrowing or closing the aiter gap broadly.
- Report per-bucket AND geomean (geomean alone may hide a regression; per-bucket
  alone may undersell breadth — report both this loop).
- No win from a single noisy run; median of repeats vs the locked deep-loop
  baseline.

## Valid loop outcomes

Use the BBuf KDA-Pilot outcome vocabulary:

- `IMPROVEMENT`: the promotion criteria above are met and a candidate lands.
- `NO-GO`: acceptable only when no kernel-body candidate clears the bar **and**
  the loop has baseline recovery, at least one correct kernel-body candidate or
  a proven correctness-blocked lever, benchmark/profiling/ISA evidence for the
  required buckets, and a named active bound or structural blocker explaining
  why the scoped path should not continue.
- `BLOCKED`: use only when the loop cannot make an evidence-backed
  improvement/no-go decision because required hardware, baseline recovery,
  correctness reference, or benchmark/profiling evidence is unavailable.

Do not treat the first losing candidate as `NO-GO`. The no-go bar requires an
evidence-backed exhaustion of the scoped, pre-authorized in-body levers.

## RLCR Loop Rules (additions for Loop 02)

All Loop 01 rules apply, plus:

- **Surface-enumeration rule:** any dispatch/launch-signature change must enumerate
  the full call contract of every target path and ship a no-GPU routing-predicate
  unit test for every caller form, in the same round.
- **Structural-candidate rule:** a milestone-scoped kernel-body change may span
  multiple coupled edits within one candidate when the coupling is stated up
  front (e.g. buffer layout + address arithmetic + waitcnt schedule together).
- **Ledger numbers are generated from artifacts**, never hand-transcribed.
- **Setup preflight:** validate the Codex model name resolves and the worktree
  build/runtime bindings are wired before round 0 (see prepare script).
- Failed candidates recorded in `docs/attempts.jsonl` + `docs/optimization-ledger.md`.

## Expected Plan Shape

1. Context refresh from Loop 01 results + the merged PR683 + #685 gate.
2. Lock the deep-loop baseline (upstream/main + #685 gate), full sweep + --compare.
3. Re-profile the target buckets to confirm the Loop 01 bottleneck map still holds
   on the current baseline.
4. Deep optimization milestones (from the pre-authorized list), ranked by
   expected value and risk, each with named profiling questions and sub-steps.
5. Per-milestone candidate ledger + promotion/no-go decision (breadth-scored).
6. Final report with formal outcome, per-bucket AND geomean tables vs the
   deep-loop baseline.

## Final Deliverables

- Design summary; changed files (must include `flash_attn_gfx950.py`).
- Correctness table; per-bucket AND geomean benchmark table vs deep-loop baseline.
- aiter_ck / aiter_asm comparison; split-K table if split-K changed.
- Before/after profiling evidence for each promoted kernel-body change (the named
  bubble must measurably shrink).
- If outcome is `NO-GO`, a no-go report naming the active bound or structural
  blocker for each required bucket.
- Known unsupported regimes / regressions; exact reproduction commands.
- MI350 (gfx950) labeling; MI355X status (measured or planned).
