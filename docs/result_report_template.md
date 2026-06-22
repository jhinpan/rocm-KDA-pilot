# Loop NN — <kernel>, <arch> (<gpu>) — <one-line outcome>

Canonical structure for a `results/loop-NN-*.md` report. Fill every section; an
empty section is a signal the loop is not finished. Outcome is one of
`IMPROVEMENT`, `NO-GO`, `BLOCKED` (a `NO-GO`/`BLOCKED` is a valid deliverable when
backed by correctness, benchmark, profiling, and ISA evidence).

## At a glance

- Outcome: `IMPROVEMENT` | `NO-GO` | `BLOCKED`
- Headline number vs baseline and vs the external reference (one line each).
- Rounds used / budget; exit reason (completed | maxiter | escalation).

## What landed

- Changed files (source only; artifacts stay untracked).
- The shipped path and any gated/opt-in modes.

## Measured result

- GPU id/model, branch/commit, exact command, shape set, dtype + causal mode,
  warmup/iters, artifact path. One row per shape; no single-noisy-run claims.
- Correctness vs the reference at the fixed gate (no relaxed thresholds, no
  SKIP-as-pass).

## Baseline + external-reference comparison

- Locked baseline number and the external-reference number (same
  GPU/shape/warmup/iters). State the gap as a percentage.

## Per-bucket bottleneck profile (round-0 trace + any deltas)

- The round-0 ATT trace stall taxonomy (barrier / vmcnt / valu / lds / mfma / …).
- Whether the kernel is stall-bound or compute-bound, and the top bubble class.
- For each promoted candidate: did the targeted bubble actually shrink?

## Candidate ledger

- One row per candidate (failed ones included): hypothesis, change, result,
  promoted/rejected, evidence path.

## What is NOT done / ranked next steps

- The remaining gap(s) and, for each, a concrete next hypothesis with the
  profiling signal that motivates it. This is what lets the next loop pick up
  where this one stopped.
- Any criterion left unmet, and whether it is `NO-GO`/`BLOCKED`/deferred (with the
  human decision that authorized it, if any — see the convergence guardrails in
  `humanize_flow.md`).

## Honest assessment

- Why the win was narrow / why the NO-GO is the right call. Name what would change
  the verdict (different geometry, a missing intrinsic, more occupancy, etc.).

## Process cost (for workflow tracking)

- Rounds, where time went (feature vs environment vs detours), and any preflight
  trap that should have been caught earlier.

## Lessons learned (pilot-workflow)

- Reusable lessons for the scaffold (feed back into preflight/contracts/templates).

## Artifacts

- Local paths to traces, CSVs, logs (kept untracked; list paths + commands).
