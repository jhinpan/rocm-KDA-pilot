# Loop 01 — FlyDSL FlashAttention forward, gfx950 (MI350X)

First end-to-end run of the ROCm KDA Pilot Humanize/RLCR workflow.

## At a glance

| | |
|---|---|
| Target | FlyDSL FlashAttention **forward**, gfx950 / MI350X |
| Baseline | ROCm/FlyDSL #683 (`DUALWAVE_SWP` forward kernel) |
| Worktree | `jhinpan/FlyDSL-lab` @ `kda/flydsl-flashattn-gfx950-pr683` |
| Reviewed against | `rocm-kda-base/flydsl-flashattn-gfx950-pr683` |
| Loop | Humanize RLCR, `--max 12`, `--codex-model gpt-5.5:xhigh` |
| Rounds used | 8 of 12 (build/review) + code-review phase (3 fix rounds) + finalize |
| Outcome | **IMPROVEMENT — 1 promoted optimization landed**; all acceptance criteria met; Codex review passed |
| Upstream PR | **[ROCm/FlyDSL#685](https://github.com/ROCm/FlyDSL/pull/685)** (draft) |

## What landed

A **dense short-sequence dispatch gate** in `kernels/flash_attn_generic.py`
(`_DUALWAVE_MIN_DENSE_SEQ = 256`): dense `seq_len < 256` routes to the generic
M128/M256 fallback (faster there); `seq_len >= 256` and all packed/varlen stay on
`DUALWAVE_SWP`; `FLYDSL_DISABLE_DUALWAVE_SWP=1` preserved. `flash_attn_gfx950.py`
unchanged. Net diff vs PR683: 1 file, +57/-18.

### Measured result (MI350X, gfx950)

| seq_len bucket | result |
|---|---|
| **128 (dense MHA)** | **1.37–1.77× faster** (mean ~1.56×); aiter gap closed/reversed for B=8, near parity B=1 causal |
| 256 | ~tie (left on dualwave by design) |
| ≥384 / GQA / varlen | unchanged (still dualwave; already beat aiter) |

Correctness: full `DEFAULT_CONFIGS` sweep 0 FAIL / 0 ERROR; `VARLEN_CONFIGS`
16/16 PASS; gate `max_err < 1e-2` AND `min_cos > 0.99` unchanged.

## Candidate ledger (12 candidates)

| candidate | disposition | note |
|---|---|---|
| baseline (PR683) | locked | 108 default + 16 varlen rows, 0 FAIL |
| env knobs: no_lazy / no_stagger | rejected | broad regression (geomean 0.90 / 0.87) |
| env knob: no_setprio | rejected | noise |
| env knob: waves_per_eu = 3 / 4 | rejected | **correctness FAIL (NaN)** |
| **dense short-seq dispatch gate** | **PROMOTED** | the landed win |
| kernel-body: lazy-rescale threshold 8→16 | rejected | neutral (geomean 0.9956), reverted |
| kernel-body: VMEM-prefetch slack | rejected | **correctness FAIL** (LDS double-buffer hazard), reverted |

## Per-bucket bottleneck profile (flyprof, MI350X)

- **short / mid dense buckets** (the aiter gap): **memory-bound** — `vmcnt`
  (VMEM wait) + `s_barrier` dominate; MFMA only ~3–5%; occupancy capped at
  **4 waves/CU (register-limited)**.
- **long MHA / GQA / split-K / varlen**: latency-bound (barrier + occupancy),
  already competitive with aiter.

## Honest assessment — why the win was narrow

The promoted change helps **only a small slice** (dense `S=128`). That is a direct
consequence of how Loop 01 was scoped, not a hardware limit:

1. The plan's **lower bound permitted a dispatch-only win** (AC: "≥1 promoted
   candidate"). The risk-averse loop took the cheapest satisfying path and stopped.
2. The **per-bucket "no regression" bar + "one isolated change at a time" + the
   hard correctness gate** made every deep kernel-body candidate look
   high-risk / low-reward; both attempted ones were neutral or
   correctness-failing and were correctly rejected.
3. The **real depth levers are inherently multi-step** (the LDS double-buffer
   prefetch depth is hardcoded; raising occupancy needs VGPR-footprint surgery),
   which the "isolated change" granularity rule pushes against.

To get #683-style **broad** speedups across many shapes, a deeper Loop 02 needs a
differently-structured draft/plan (see
[`templates/flydsl_flashattn_gfx950_deep_contract.md`](../templates/flydsl_flashattn_gfx950_deep_contract.md)
and the Loop 02 result report).

## Process cost (for workflow tracking)

- ~8 build/review rounds for ~3 rounds of conceptual work. Recoverable losses:
  - 3 serial code-review rounds each fixed one edge of the **same**
    dispatch-compatibility bug class (unexpected-kwarg → explicit-None →
    positional-stream). A routing-predicate unit test up front would have caught
    all three in the build phase.
  - 1 mis-stated artifact count ("216 rows") propagated into multiple ledgers and
    cost ~2 reconciliation rounds.
  - 1 infra loss: a `gpt5.5` vs `gpt-5.5` model-name typo forced a loop
    cancel+restart; the worktree also needed a manual `build-fly` / `_mlir`
    binding symlink.
- Methodology feedback filed upstream: commented on
  [PolyArch/humanize#193](https://github.com/PolyArch/humanize/issues/193) and
  filed [PolyArch/humanize#216](https://github.com/PolyArch/humanize/issues/216)
  (inherited-evidence revalidation, single-pass execute-and-capture, setup-time
  config preflight).

## Lessons learned (pilot-workflow)

1. **Reward shape determines depth.** A "≥1 candidate" lower bound yields a
   shallow win. If broad speedup is the goal, the draft must reward breadth and
   forbid a dispatch-only-only outcome.
2. **Pre-authorize structural candidates.** Deep levers (LDS prefetch depth,
   occupancy/VGPR) can't be "one isolated change"; the plan must allow them as
   milestones with sub-steps, not "last resort."
3. **Add a routing-predicate unit test whenever dispatch changes.** Cheap,
   no-GPU, catches the whole compatibility-contract class at build time.
4. **Generate ledger numbers from artifacts**, never hand-transcribe.
5. **Preflight the loop config** (model name, build tree) before round 0.
6. The pilot's run-environment (this-worktree Python + lab `build-fly` bindings)
   should be wired by the prepare script, not improvised mid-loop.
