# FlyDSL FlashAttention Rules

- Treat PR683 `tests/kernels/test_flash_attn_fwd.py` as the canonical harness.
- Preserve PyTorch SDPA / chunked SDPA reference behavior.
- Preserve bf16/fp16, causal/non-causal, MHA/GQA, varlen, arbitrary seq_len,
  split-K, and gfx942 fallback coverage.
- Do not weaken `max_err < 1e-2` or `min_cos > 0.99`.
- Do not turn unexpected failures into SKIP rows.
- Compare against PR683 baseline before claiming improvement.
- Keep raw profile artifacts local and summarize only the evidence needed for a
  reviewer to reproduce the claim.

