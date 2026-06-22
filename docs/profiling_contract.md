# Profiling Contract

Profile only to answer a named question.

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

Use `flyprof` first for FlyDSL kernels:

```bash
flyprof doctor -f json
flyprof list --worktree "$PWD" -f json
flyprof run flash_attn_fwd --worktree "$PWD" --gpu "$GPU" --bundle "profile/<run-id>/flyprof" -f json
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

