# Benchmark Contract

Use the same harness, input distribution, GPU, dtype, causal mode, warmup, and
iteration count for baseline and candidate.

Required commands:

```bash
HIP_VISIBLE_DEVICES=$GPU python3 tests/kernels/test_flash_attn_fwd.py --warmup 10 --iters 20
HIP_VISIBLE_DEVICES=$GPU python3 tests/kernels/test_flash_attn_fwd.py --compare --warmup 10 --iters 100
```

Report:

- per-shape time(us)
- per-shape TFLOPS
- grouped averages or geomean
- max error and min cosine
- GPU model/id
- branch/commit
- exact command
- CSV path

Near-threshold wins require a repeat run.

