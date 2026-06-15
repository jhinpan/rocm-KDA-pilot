---
name: rocm-kda-pilot
description: Use for running a Humanize/KDA-style AMD ROCm kernel optimization loop, especially FlyDSL FlashAttention on gfx950. Combines ROCmKernelWiki, flyprof, rocm-report-skill, PR683 benchmark discipline, and Humanize RLCR.
allowed-tools: Read Bash Grep Glob
---

# ROCm KDA Pilot

Use this skill when the user wants a BBuf/KDA-Pilot-style loop for AMD ROCm
kernel optimization.

For FlyDSL FlashAttention on gfx950:

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

The loop should produce correctness evidence, benchmark evidence, candidate
lineage, failed-attempt notes, and exact reproduction commands.

