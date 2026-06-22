# Humanize Gen-Plan Draft: FlyDSL fp8 FlashAttention Forward On gfx950

Use this draft to generate a Humanize RLCR implementation plan for adding and
optimizing an **fp8** FlashAttention forward path in FlyDSL on AMD gfx950 /
MI350X-MI355X.

This is not a generic attention task and not the bf16/fp16 contract. The plan
must be grounded in ROCm/FlyDSL PR683's existing FlashAttention implementation
and test harness, and in the aiter **asm** fp8 attention pipeline that this work
is meant to port and match.

## Source Refs

- Task issue: https://github.com/ROCm/FlyDSL/issues/698 ([Feature]: fp8 flash attention)
- FlyDSL upstream: https://github.com/ROCm/FlyDSL
- FlyDSL working fork: https://github.com/jhinpan/FlyDSL-lab
- Working baseline (bf16/fp16 reference impl + harness): ROCm/FlyDSL PR683
- Historical related PR: ROCm/FlyDSL PR670
- Comparison target: aiter asm fp8 attention pipeline (ROCm/aiter)
- Observed on: {{DATE}}
- FlyDSL main SHA: {{FLYDSL_MAIN_SHA}}
- PR670 head SHA: {{PR670_SHA}}
- PR683 head SHA: {{PR683_SHA}}
- Current worktree HEAD: {{WORKTREE_HEAD}}
- Current branch: {{WORKTREE_BRANCH}}

## Issue Context (#698)

- Reporter: coderfeli. Assignees: coderfeli, jhinpan. Milestone: v0.4.
- Stated motivation, verbatim intent:
  - Current (non-fp8) FlyDSL flash attention already has good perf: ~`1300+T`.
  - It lacks fp8 support; aiter **asm** reaches ~`2000+T` in fp8.
  - Goal: "Port the aiter asm pipeline and optimize perf to align with asm."
- The issue body has no code, no table, and no explicit acceptance criteria.
  This draft supplies the K/R/W contract, correctness gates, and outcome
  criteria that the issue text leaves implicit. Treat the numbers above as the
  motivating target, not a measured baseline on this worktree -- the loop must
  re-measure both the bf16 baseline and any aiter fp8 comparison on the same GPU.

## Ultimate Goal

Add a correct fp8 FlashAttention forward path to FlyDSL on gfx950, then optimize
it toward the aiter asm fp8 level (~`2000+T`), while preserving the existing
bf16/fp16 paths and all current coverage.

The headline target is performance parity with aiter asm fp8, but correctness
and non-regression of the existing dtype paths are hard gates. Do not weaken
tests, skip difficult cases, or replace PR683's harness with a toy benchmark. Do
not claim the fp8 win by silently loosening tolerances below what the plan fixes.

## Kernel Contract

K:

- Primary target files:
  - `kernels/flash_attn_gfx950.py`
  - `kernels/flash_attn_generic.py`
  - `python/flydsl/expr/rocdl.py` only if fp8-specific MFMA / conversion
    intrinsics require it
  - `tests/kernels/test_flash_attn_fwd.py` only for legitimate harness fixes or
    to add fp8 cases / measurement output
- Primary target architecture: gfx950 / CDNA4 / MI350X-MI355X.
- fp8 format on gfx950 is **e4m3fn** (OCP `float8_e4m3fn`), not the fnuz variant.
  State the exact dtype used for Q, K, V, and any accumulation explicitly.
- Define the fp8 numerics path: per-tensor or per-head scaling, where dequant /
  rescale happens, and how softmax accumulation precision is preserved (softmax
  and the running max/sum should stay in a safe precision, typically f32).
- Preserve gfx942 fallback behavior.
- Preserve PR683's dualwave SWP path, arbitrary sequence length support, packed
  varlen support, GQA/MQA semantics, split-K plumbing, and the existing
  bf16/fp16 dtype coverage. fp8 is additive: existing dtypes must not regress in
  correctness or measurable performance.

R:

- Correctness reference is PR683 `tests/kernels/test_flash_attn_fwd.py`.
- Reference output is PyTorch SDPA or chunked PyTorch SDPA from the same file,
  computed in a high-precision dtype (the fp8 candidate is compared against the
  same SDPA reference the bf16 path uses).
- fp8 correctness gate (fp8 is lossy; the threshold is fixed here, not relaxed
  mid-loop):
  - `max_err < 5e-2`
  - `min_cos > 0.98`
  - no FAIL or ERROR rows in the required fp8 sweep
- Existing bf16/fp16 gates are unchanged and must still pass:
  - `max_err < 1e-2`
  - `min_cos > 0.99`
- Do not relax the fp8 thresholds below the values above to make a candidate
  pass. If fp8 cannot meet them, that is evidence for a `NO-GO` or `BLOCKED`,
  not a reason to move the gate.
- Do not silently convert correctness failures into SKIP rows.

W:

- Required fp8 sweep (new):
  - PR683 `DEFAULT_CONFIGS` shapes, fp8 (e4m3fn)
  - causal and non-causal
  - MHA and GQA cases already present in PR683
- Required non-regression sweep (existing dtypes must still pass):
  - PR683 `DEFAULT_CONFIGS` + `VARLEN_CONFIGS`, bf16 and fp16, causal and
    non-causal
- Required split-K focus when split-K is touched:
  - `B=1 S=8192 H=2 Hkv=2 D=128 splits=4`
  - `B=1 S=4096 H=2 Hkv=2 D=128 splits=4`
  - `B=1 S=2048 H=4 Hkv=4 D=128 splits=4`
  - `B=1 S=8192 H=4 Hkv=4 D=128 splits=2`
- Headline comparison: FlyDSL fp8 vs **aiter asm fp8** on the same shapes, same
  GPU, same warmup/iters. Also report FlyDSL fp8 vs the FlyDSL bf16 baseline so
  the dtype speedup is explicit. aiter_ck fp8 is a secondary comparison when
  available.

## Required Knowledge Tools

Use these when they affect the next implementation or profiling decision:

- `ROCmKernelWiki`
  - Query fp8 attention, gfx950/CDNA4 fp8 MFMA (e4m3 throughput), aiter asm fp8
    flash attention pipeline, fp8 scaling/dequant patterns, direct-to-LDS,
    O-store, and split-K.
- `flyprof`
  - Use for instruction-level evidence when the fp8 path is slower than aiter asm
    or a candidate plateaus (is the gap MFMA issue rate, conversion overhead,
    VMEM/LDS wait, or O-store?).
- `rocm-report-skill`
  - Use when a rocprofv3 / ATT report should turn into one concrete fp8
    optimization hypothesis.

Do not run profiling as ritual. Profile only to answer a named question, such as:

- Is the fp8 candidate hitting the gfx950 fp8 MFMA throughput, or stalled on
  conversions / scaling?
- Where is the gap vs aiter asm: issue rate, VMEM wait, LDS wait, waitcnt
  dependency, O-store, or tail/grid underfill?
- Did a candidate actually move the top bubble?

## Baseline And Benchmark Commands

Set `GPU` to an idle gfx950 GPU id. Source the runenv first if preflight wrote
one: `source .humanize/kernel-agent/runenv.sh`.

bf16 baseline correctness/perf sweep (establish the non-fp8 reference number):

```bash
HIP_VISIBLE_DEVICES=$GPU python3 tests/kernels/test_flash_attn_fwd.py --warmup 10 --iters 20
```

fp8 correctness smoke (once the fp8 path exists):

```bash
HIP_VISIBLE_DEVICES=$GPU python3 tests/kernels/test_flash_attn_fwd.py --dtype fp8 --causal --warmup 3 --iters 5
```

fp8 promotion comparison sweep (vs aiter asm when available):

```bash
HIP_VISIBLE_DEVICES=$GPU python3 tests/kernels/test_flash_attn_fwd.py --dtype fp8 --compare --warmup 10 --iters 100
```

> Note: the exact `--dtype fp8` flag spelling and the aiter-asm comparison
> selector depend on the PR683 harness. The plan must confirm the real flag
> names from `tests/kernels/test_flash_attn_fwd.py` before relying on them, and
> add fp8 wiring to the harness if it is missing (that counts as a legitimate
> harness change under K).

Profiling examples:

```bash
flyprof doctor -f json
flyprof list --worktree "$PWD" -f json
flyprof run flash_attn_fwd --worktree "$PWD" --gpu "$GPU" --bundle "profile/flydsl-fa-fp8-gfx950-$(date +%Y%m%d-%H%M%S)/flyprof" -f json
```

## RLCR Loop Rules

- Do not implement kernel changes before RLCR is active.
- Keep `.humanize*` untracked.
- Keep raw rocprof, ATT, cache, build, and huge CSV artifacts untracked.
- Implement one candidate change at a time unless coupling is technically
  necessary and stated in the plan.
- Every performance claim must name:
  - GPU id and model
  - branch and commit
  - exact command
  - shape set
  - dtype (fp8 e4m3fn / bf16 / fp16) and causal mode
  - warmup / iteration count
  - CSV or profile artifact path
  - idle-GPU evidence if available
- Failed candidates must be recorded in `docs/attempts.jsonl` or
  `docs/optimization-ledger.md`.
- Do not claim a win from a single noisy near-threshold run.

## Lessons from the first fp8 loop (Loop 04)

The first fp8 loop landed a correct, additive, merge-ready forward but ran to its
iteration budget without reaching aiter-asm-fp8 parity. Bake these in next time
(see `results/loop-04-flashattn-fp8-gfx950.md` and the generic guardrails in
`docs/humanize_flow.md`):

- **Split the parity gate from the correctness gate explicitly.** The plan made
  asm-fp8 throughput parity a hard merge blocker (single-PR). Correctness landed
  early; parity did not, and the loop spent its remaining budget circling parity.
  State up front whether a correct-but-slower fp8 forward is independently
  mergeable, and put parity behind a binding-constraint escalation gate so the
  loop pauses for a human decision instead of exhausting rounds.
- **Capture a baseline ATT trace in round 0**, not at the end. For this kernel the
  trace shows it is stall-bound (cross-wave barrier + global-load waits dominate;
  MFMA issue is a small fraction). fp8 and bf16 `32x32x16` MMA have equal CDNA4
  throughput, so a packed-fp8 path that adds dequant/conversion is *slower* than a
  bf16-compute path — the win must come from removing round-trips/bandwidth, which
  only the trace makes obvious.
- **Falsify a precision NO-GO with host numerics before accepting it.** A
  "fundamental fp8-P precision wall" NO-GO was accepted, then overturned by a cheap
  host-side `fp8xfp8` PV numerics probe showing the real blocker was a layout
  defect. Run that probe first.
- **The packed-fp8 PV blocker is layout, not precision.** `ds_read_b64_tr_b8`
  permutes 8-bit lanes differently from the bf16 transpose+shuffle; match the fp8
  B-operand element order to the proven bf16 layout (use an operand-dump oracle),
  and keep a bf16-reuse correctness mode as the oracle while building the true-fp8
  no-round-trip path.

## Expected Plan Shape

The generated plan should include:

1. Context refresh from PR683, PR670, and the aiter asm fp8 pipeline.
2. Baseline lock procedure for PR683 (bf16) and a measured aiter-asm-fp8 number.
3. fp8 numerics design: format, scaling, accumulation precision, dequant points.
4. Correctness and benchmark commands for both fp8 and the non-regression sweep.
5. Initial optimization directions ranked by expected value and risk.
6. Profiling decision points (named questions only).
7. Candidate ledger format.
8. Promotion criteria, including the fp8-vs-aiter-asm parity target.
9. Formal outcome criteria using `IMPROVEMENT`, `NO-GO`, or `BLOCKED`.
10. Final report format.

## Final Deliverables

The completed loop should produce:

- Design summary, including the fp8 numerics decisions.
- Changed files.
- fp8 correctness table (vs SDPA reference).
- Non-regression correctness table for bf16/fp16.
- FlyDSL fp8 vs FlyDSL bf16 benchmark table (dtype speedup).
- FlyDSL fp8 vs aiter asm fp8 benchmark table (the #698 parity target).
- Split-K table if split-K changed.
- Profile evidence if profiling influenced the edit.
- Formal outcome: `IMPROVEMENT`, `NO-GO`, or `BLOCKED`.
- Known unsupported regimes or regressions.
- Exact reproduction commands.
