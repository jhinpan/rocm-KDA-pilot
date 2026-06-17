---
name: rocm-kda-pilot
description: Use for running a Humanize/KDA-style AMD ROCm kernel optimization loop, especially FlyDSL FlashAttention on gfx950. Combines ROCmKernelWiki, flyprof, rocm-report-skill, PR683 benchmark discipline, and Humanize RLCR.
allowed-tools: Read Bash Grep Glob
---

# ROCm KDA Pilot

Use this skill when the user wants a BBuf/KDA-Pilot-style loop for AMD ROCm
kernel optimization.

For the **prep phase** (issue -> worktree -> draft -> plan), use the companion
`flydsl-task-setup` skill, which provisions a fresh `/sgl-workspace/FlyDSL-<slug>`
worktree on `rlcr/<slug>` via `scripts/new_flydsl_task.sh` and stops at the human
plan-review gate. This skill covers the loop-execution side.

Two steps always stay human-gated and must not be auto-advanced:

- **Plan review/refine** of `.humanize/kernel-agent/refined-plan.md` before the
  loop starts.
- **Loop start** (`/humanize:start-rlcr-loop`) with the exact locked
  `--base-branch`.

For FlyDSL FlashAttention on gfx950 (bf16/fp16):

1. Use PR683 as the working baseline unless the user gives a newer ref.
2. Treat `tests/kernels/test_flash_attn_fwd.py` as the canonical correctness and
   benchmark harness.
3. Generate the Humanize plan from `.humanize/kernel-agent/draft.md`.
4. Start RLCR from `.humanize/kernel-agent/refined-plan.md`.
5. Use ROCmKernelWiki for prior art only when it changes the next decision.
6. Use `flyprof` or `rocm-report-skill` only to answer a concrete profiling
   question.
7. Preserve bf16/fp16, causal/non-causal, MHA/GQA, varlen, arbitrary seq_len,
   split-K, and gfx942 fallback coverage.
8. Never weaken correctness thresholds to make a candidate pass.

For FlyDSL **fp8** FlashAttention (ROCm/FlyDSL#698):

- Provision with `scripts/new_flydsl_task.sh --template fp8`.
- fp8 on gfx950 is `e4m3fn` (not fnuz). Keep softmax max/sum accumulation in f32.
- Headline target is parity with the aiter **asm** fp8 pipeline (~2000+T) vs the
  current bf16 baseline (~1300+T).
- fp8 is additive: the existing bf16/fp16 paths must not regress.
- fp8 correctness gate is looser than bf16 (`max_err < 5e-2`, `min_cos > 0.98`)
  but fixed in the plan -- never relax it mid-loop to force a pass.

The loop should produce correctness evidence, benchmark evidence, candidate
lineage, failed-attempt notes, and exact reproduction commands.

