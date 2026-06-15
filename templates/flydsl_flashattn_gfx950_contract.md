# Humanize Gen-Plan Draft: FlyDSL FlashAttention Forward On gfx950

Use this draft to generate a Humanize RLCR implementation plan for optimizing
FlyDSL FlashAttention forward on AMD gfx950 / MI350X-MI355X.

This is not a generic attention task. The plan must be grounded in ROCm/FlyDSL
PR683's implementation and test harness.

## Source Refs

- FlyDSL upstream: https://github.com/ROCm/FlyDSL
- Working baseline: ROCm/FlyDSL PR683
- Historical related PR: ROCm/FlyDSL PR670
- Observed on: {{DATE}}
- FlyDSL main SHA: {{FLYDSL_MAIN_SHA}}
- PR670 head SHA: {{PR670_SHA}}
- PR683 head SHA: {{PR683_SHA}}
- Current worktree HEAD: {{WORKTREE_HEAD}}
- Current branch: {{WORKTREE_BRANCH}}

## Ultimate Goal

Optimize FlyDSL FlashAttention forward on gfx950 beyond the PR683 baseline while
preserving correctness and coverage.

The target is performance improvement, but correctness and coverage are hard
gates. Do not weaken tests, skip difficult cases, or replace PR683's harness
with a toy benchmark.

## Kernel Contract

K:

- Primary target files:
  - `kernels/flash_attn_gfx950.py`
  - `kernels/flash_attn_generic.py`
  - `python/flydsl/expr/rocdl.py` only if target-specific intrinsics require it
  - `tests/kernels/test_flash_attn_fwd.py` only for legitimate harness fixes or
    additional measurement output
- Primary target architecture: gfx950 / CDNA4 / MI350X-MI355X.
- Preserve gfx942 fallback behavior.
- Preserve PR683's dualwave SWP path, arbitrary sequence length support, packed
  varlen support, GQA/MQA semantics, split-K plumbing, and dtype coverage.

R:

- Correctness reference is PR683 `tests/kernels/test_flash_attn_fwd.py`.
- Reference output is PyTorch SDPA or chunked PyTorch SDPA from the same file.
- Preserve correctness gate:
  - `max_err < 1e-2`
  - `min_cos > 0.99`
  - no FAIL or ERROR rows in required sweeps
- Do not relax correctness thresholds.
- Do not silently convert correctness failures into SKIP rows.

W:

- Required default sweep:
  - PR683 `DEFAULT_CONFIGS`
  - bf16 and fp16
  - causal and non-causal
  - MHA and GQA cases already present in PR683
- Required varlen sweep:
  - PR683 `VARLEN_CONFIGS`
  - bf16 and fp16
  - causal and non-causal
- Required split-K focus when split-K is touched:
  - `B=1 S=8192 H=2 Hkv=2 D=128 splits=4`
  - `B=1 S=4096 H=2 Hkv=2 D=128 splits=4`
  - `B=1 S=2048 H=4 Hkv=4 D=128 splits=4`
  - `B=1 S=8192 H=4 Hkv=4 D=128 splits=2`
- Compare against PR683 baseline and aiter_ck / aiter_asm when available.

## Required Knowledge Tools

Use these when they affect the next implementation or profiling decision:

- `ROCmKernelWiki`
  - Query FlyDSL FlashAttention, gfx950 MFMA scheduling, CDNA4 waitcnt
    pipelining, LDS/buffer operations, direct-to-LDS, O-store patterns, split-K,
    and attention kernels in AITER / CK / FlyDSL.
- `flyprof`
  - Use for instruction-level evidence when performance is unclear or a
    candidate plateaus.
- `rocm-report-skill`
  - Use when a rocprofv3 / ATT report should turn into one concrete optimization
    hypothesis.

Do not run profiling as ritual. Profile only to answer a named question, such as:

- Is the candidate occupancy/register capped?
- Is the gap VMEM wait, LDS wait, waitcnt dependency, MFMA starvation, O-store,
  barrier/scheduler, or tail/grid underfill?
- Did a candidate actually reduce the top bubble?

## Baseline And Benchmark Commands

Set `GPU` to an idle gfx950 GPU id.

Quick correctness smoke:

```bash
HIP_VISIBLE_DEVICES=$GPU python3 tests/kernels/test_flash_attn_fwd.py --dtype bf16 --causal --warmup 3 --iters 5
```

Full PR683 correctness/perf sweep:

```bash
HIP_VISIBLE_DEVICES=$GPU python3 tests/kernels/test_flash_attn_fwd.py --warmup 10 --iters 20
```

Promotion comparison sweep:

```bash
HIP_VISIBLE_DEVICES=$GPU python3 tests/kernels/test_flash_attn_fwd.py --compare --warmup 10 --iters 100
```

Focused split-K examples:

```bash
HIP_VISIBLE_DEVICES=$GPU python3 tests/kernels/test_flash_attn_fwd.py --batch 1 --seq_len 8192 --num_heads 4 --num_kv_heads 4 --head_dim 128 --num_kv_splits 2 --dtype bf16 --warmup 10 --iters 100
```

Profiling examples:

```bash
flyprof doctor -f json
flyprof list --worktree "$PWD" -f json
flyprof run flash_attn_fwd --worktree "$PWD" --gpu "$GPU" --bundle "profile/flydsl-fa-gfx950-$(date +%Y%m%d-%H%M%S)/flyprof" -f json
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
  - dtype and causal mode
  - warmup / iteration count
  - CSV or profile artifact path
  - idle-GPU evidence if available
- Failed candidates must be recorded in `docs/attempts.jsonl` or
  `docs/optimization-ledger.md`.
- Do not claim a win from a single noisy near-threshold run.

## Expected Plan Shape

The generated plan should include:

1. Context refresh from PR683 and PR670.
2. Baseline lock procedure for PR683.
3. Correctness and benchmark commands.
4. Initial optimization directions ranked by expected value and risk.
5. Profiling decision points.
6. Candidate ledger format.
7. Promotion criteria.
8. Final report format.

## Final Deliverables

The completed loop should produce:

- Design summary.
- Changed files.
- Correctness table.
- Baseline vs candidate benchmark table.
- aiter_ck / aiter_asm comparison when available.
- Split-K table if split-K changed.
- Profile evidence if profiling influenced the edit.
- Known unsupported regimes or regressions.
- Exact reproduction commands.

