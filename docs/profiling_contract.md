# Profiling Contract

Profile only to answer a named question -- and answer it with the verified
flyprof + rocm-report-skill chain, **never** a hand-rolled rocprofv3 script.

## Mandatory tooling (binding)

All ROCm/FlyDSL kernel profiling uses the user-level skill chain:
`flyprof-usage` (if unsure) -> `flyprof-capture` -> `flyprof-analyze` /
`flyprof-triage` -> `rocm-report-skill`; `ROCmKernelWiki` for prior art.

- **Do NOT write a bespoke `rocprofv3 -i <pmc> --output-format csv` collection
  script or a hand-rolled counter `summarize.py`.** These reinvent a verified tool,
  are easy to misread (e.g. confusing work-items for workgroups, inverting a stall
  ratio), and produce non-replayable artifacts. Use `flyprof-capture` +
  `rocm-report-skill` instead.
- "Profile only to answer a named question" is NOT a license to skip profiling. It
  means: when a round needs a bottleneck/lever decision, *use this chain to produce
  the artifact that answers it*. A bottleneck claim ("LDS-wait bound") or a
  deeper-lever decision ("do a pipeline/LDS rewrite") with no flyprof/rocm-report
  artifact is unsupported and a review should reject it.
- A stale profile is not a fresh profile. If a conclusion is being reused several
  rounds later to justify new work, re-capture rather than re-cite.

Rationale (MXFP4 MoE loop): a 28-round loop profiled exactly once, early, with a
hand-written rocprofv3 collection script that the reviewer repeatedly faulted
(work-items misread as workgroups; the largest stall mislabeled); every later
"LDS-wait bound" conclusion leaned on that single stale hand-rolled capture, and
the verified flyprof / rocm-report / ROCmKernelWiki skills were never invoked once.
The chain below exists precisely to prevent that.

## Baseline trace is a round-0 deliverable, not a finalization step

Capture a baseline ATT thread trace of the target kernel **at the start of the
loop**, alongside the locked baseline and the external-reference (e.g. aiter)
comparison number. Every subsequent optimization hypothesis must cite it.

Rationale (fp8 FlashAttention loop): the loop ran to its iteration budget without
profiling once; the trace was only captured at PR time. It immediately showed the
kernel was **stall-bound, not compute-bound** (cross-wave barrier + global-load
waits dominated; MFMA issue was a small fraction of cycles). Had that been the
round-0 input, several rounds spent on operand-precision detours would instead
have targeted scheduling/synchronization from the first round.

Round-0 profiling checklist:

- One baseline ATT trace of the target kernel (bucketed stall taxonomy).
- The locked baseline number and the external-reference number on the same
  GPU/shape/warmup/iters (so the gap is quantified before any change).
- A one-line read of the top bubble class, which becomes the first hypothesis.

Use `flyprof` first for FlyDSL kernels. Discover the kernel name with
`flyprof list`, then run that kernel (it is `flash_attn_fwd` for FlashAttention,
a MoE GEMM target for the MoE family, a GEMM target for the GEMM family, etc.):

```bash
flyprof doctor -f json
flyprof list --worktree "$PWD" -f json          # pick <kernel> from this list
flyprof run <kernel> --worktree "$PWD" --gpu "$GPU" --bundle "profile/<run-id>/flyprof" -f json
```

Use `rocm-report-skill` when you need a report that converts rocprofv3 / ATT
evidence into one concrete optimization hypothesis.

Common questions:

- occupancy or register cap
- VMEM wait
- LDS wait or bank conflict
- waitcnt dependency
- MFMA starvation
- O-store bottleneck
- barrier or scheduler grouping
- tail or grid underfill
- source mapping failure

Do not publish raw ATT dumps. Keep raw artifacts local and write a small report
with exact paths and commands.

