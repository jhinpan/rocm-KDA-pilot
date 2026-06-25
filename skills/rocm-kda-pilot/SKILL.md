---
name: rocm-kda-pilot
description: Orchestrator for ANY FlyDSL / AMD ROCm GPU kernel optimization loop -- FlashAttention, GEMM, MoE (incl. MXFP4/fp8), and other kernels, on gfx942 / gfx950 / gfx1250. Use whenever the task is tuning or optimizing a FlyDSL/ROCm kernel (raise MFU/TFLOPS, cut latency, fix a profiled bottleneck), not just FlashAttention. Combines ROCmKernelWiki, flyprof (flydsl-rocprof-cli), rocm-report-skill, benchmark discipline, and Humanize RLCR. Profiling MUST go through flyprof + rocm-report-skill -- never hand-roll a rocprofv3 collection script.
allowed-tools: Read Bash Grep Glob
---

# ROCm KDA Pilot

Use this skill for **any** FlyDSL / AMD ROCm GPU kernel optimization loop, not
only FlashAttention. It is the orchestrator that combines ROCmKernelWiki (prior
art), flyprof (capture/analyze/triage), rocm-report-skill (evidence -> one
hypothesis), benchmark discipline, and the Humanize RLCR loop. Trigger it
whenever the task is to tune/optimize a FlyDSL or ROCm kernel -- GEMM, MoE
(including MXFP4 per-1x32 fp4 and fp8 mixed), attention, or any other -- across
gfx942 / gfx950 / gfx1250.

## Kernel families (all in scope)

This orchestrator is kernel-family-agnostic. Pick the harness + contract for the
family, then run the same loop discipline:

- **FlashAttention** (bf16/fp16/fp8): harness `tests/kernels/test_flash_attn_fwd.py`;
  contracts `templates/flydsl_flashattn_gfx950{,_deep}_contract.md`,
  `templates/flydsl_flashattn_fp8_gfx950_contract.md`. Family specifics in the
  FlashAttention section below.
- **MoE 2-stage GEMM** (MXFP4 / fp8 mixed): harness `tests/kernels/test_moe_gemm.py`
  + aiter `op_tests/test_moe_2stage.py`; contract
  `templates/flydsl_mxfp4_moe_gfx950_contract.md` (ROCm/FlyDSL#708). Provision with
  `scripts/new_flydsl_task.sh --template mxfp4`.
- **GEMM / other**: use the closest harness in `tests/kernels/` and the
  `gemm-optimization` tile priors; the same mandatory-tooling + loop discipline
  apply.

The FlashAttention-specific guidance (PR683 baseline, fp8 parity targets) is one
family's specialization; the **mandatory tooling** and **loop discipline**
sections apply to every family.

## MANDATORY tooling -- do not hand-roll profiling

ROCm/FlyDSL kernel profiling goes through the verified user-level skill chain,
never a hand-written `rocprofv3` collection script:

1. **`flyprof-usage`** -- first, if unsure of the CLI surface.
2. **`flyprof-capture`** -- record the kernel's ATT trace (recon -> size -> capture).
3. **`flyprof-analyze`** / **`flyprof-triage`** -- bucket the stall taxonomy and
   name the top bubble class.
4. **`rocm-report-skill`** -- convert rocprofv3 / ATT evidence into ONE concrete
   optimization hypothesis.
5. **`ROCmKernelWiki`** -- prior art, only when it changes the next decision.

Rules (binding):

- **Never write a bespoke `rocprofv3 -i ... --output-format csv` collection script
  or a hand-rolled counter `summarize.py`.** If you start to, stop and use
  `flyprof-capture` + `rocm-report-skill` -- they are verified and produce
  auditable, replayable bundles.
- A bottleneck claim or a deeper-lever decision ("stage2 is LDS-wait bound", "do a
  pipeline/LDS rewrite") MUST cite a flyprof / rocm-report artifact. An unprofiled
  bottleneck assertion is not evidence and a review should reject it.
- "Profile only to answer a named question" means *use this chain to answer it* --
  it is NOT a license to skip profiling. A round drawing a profiling conclusion
  without a fresh-enough flyprof artifact should be sent back.

If these skills are missing, run `scripts/bootstrap.sh` (it symlinks them into
`~/.claude/skills` and installs flyprof); `scripts/preflight.sh` verifies they are
available before the loop starts.

For the **prep phase** (issue -> worktree -> draft -> plan), use the companion
`flydsl-task-setup` skill, which provisions a fresh `/sgl-workspace/FlyDSL-<slug>`
worktree on `rlcr/<slug>` via `scripts/new_flydsl_task.sh` and stops at the human
plan-review gate. This skill covers the loop-execution side.

Two steps always stay human-gated and must not be auto-advanced:

- **Plan review/refine** of `.humanize/kernel-agent/refined-plan.md` before the
  loop starts.
- **Loop start** (`/humanize:start-rlcr-loop`) with the exact locked
  `--base-branch`.

### FlashAttention family specifics (gfx950, bf16/fp16)

1. Use PR683 as the working baseline unless the user gives a newer ref.
2. Treat `tests/kernels/test_flash_attn_fwd.py` as the canonical correctness and
   benchmark harness.
3. Generate the Humanize plan from `.humanize/kernel-agent/draft.md`.
4. Start RLCR from `.humanize/kernel-agent/refined-plan.md`.
5. Profile via the mandatory tooling chain above (`flyprof-capture` ->
   `flyprof-analyze`/`triage` -> `rocm-report-skill`; `ROCmKernelWiki` for prior
   art) to answer each named question -- do not hand-roll rocprofv3.
6. Preserve bf16/fp16, causal/non-causal, MHA/GQA, varlen, arbitrary seq_len,
   split-K, and gfx942 fallback coverage.
7. Never weaken correctness thresholds to make a candidate pass.

For FlyDSL **fp8** FlashAttention (ROCm/FlyDSL#698):

- Provision with `scripts/new_flydsl_task.sh --template fp8`.
- fp8 on gfx950 is `e4m3fn` (not fnuz). Keep softmax max/sum accumulation in f32.
- Headline target is parity with the aiter **asm** fp8 pipeline (~2000+T) vs the
  current bf16 baseline (~1300+T).
- fp8 is additive: the existing bf16/fp16 paths must not regress.
- fp8 correctness gate is looser than bf16 (`max_err < 5e-2`, `min_cos > 0.98`)
  but fixed in the plan -- never relax it mid-loop to force a pass.
- Before starting, read the first-fp8-loop lessons in
  `templates/flydsl_flashattn_fp8_gfx950_contract.md` ("Lessons from the first fp8
  loop") and `results/loop-04-flashattn-fp8-gfx950.md`: split the parity gate from
  the correctness gate, capture a round-0 ATT trace (the kernel is stall-bound, not
  compute-bound), falsify any precision NO-GO with a host `fp8xfp8` numerics probe
  first, and treat the packed-fp8 PV blocker as a transpose-load layout problem.

The loop should produce correctness evidence, benchmark evidence, candidate
lineage, failed-attempt notes, and exact reproduction commands.

