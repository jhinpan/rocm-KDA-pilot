---
name: Humanize kernel task
about: Source issue for a ROCm KDA Pilot draft, plan, and RLCR loop
title: "[Humanize] <target kernel / task>"
labels: humanize-task
---

## Goal

What should the loop improve, and why does it matter?

## Target Worktree

- Target repo:
- Working fork:
- Active branch:
- Locked baseline branch or commit:
- Upstream refs or PRs:

## Kernel Contract

K:

- Primary files:
- Target architecture / GPU:
- Behavior that must be preserved:

R:

- Correctness reference:
- Required thresholds:
- Cases that must not become SKIP:

W:

- Required benchmark command(s):
- Required profiling command(s), if any:
- Comparison backend(s), if any:

## Constraints

- What is allowed?
- What is out of scope?
- What must not be weakened?

## Evidence Required

- Correctness evidence:
- Performance evidence:
- Profiling or ISA evidence:
- Artifact paths to record:

## Expected Deliverables

- Source changes:
- Benchmark table:
- Profiling report:
- Expected outcome: `IMPROVEMENT`, `NO-GO`, or `BLOCKED`
- Known regressions or `NO-GO` criteria:
- Final result location under `results/`:

## Notes And Prior Art

Links, prior attempts, relevant PRs, papers, docs, or comments.
