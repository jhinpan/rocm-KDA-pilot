# Profiling Contract

Profile only to answer a named question.

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

