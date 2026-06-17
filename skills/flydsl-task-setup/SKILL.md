---
name: flydsl-task-setup
description: Use to turn a ROCm/FlyDSL kernel-optimization issue into a ready-to-run Humanize task -- provision a fresh FlyDSL-lab worktree+branch, snapshot the issue into a draft, generate the plan, and stop at the human review gate. Fixed /sgl-workspace layout. Pairs with rocm-kda-pilot.
allowed-tools: Read Bash Grep Glob
---

# FlyDSL Task Setup

Use this skill for the **prep phase** of a ROCm KDA Pilot loop: everything from a
GitHub issue up to (but not past) the human plan-review gate. A Claude Code
instance can own this whole phase. It must NOT cross the two human gates.

## Fixed environment

This skill assumes the standard two-repo layout under `/sgl-workspace`. The paths
are fixed, so do not ask the user where things are -- use these:

| Path | Role |
|---|---|
| `/sgl-workspace/FlyDSL-lab` | The `jhinpan/FlyDSL-lab` checkout that hosts every FlyDSL worktree. `git worktree add` runs from here. Remotes: `origin` = jhinpan/FlyDSL-lab, `upstream` = ROCm/FlyDSL. |
| `/sgl-workspace/rocm-KDA-pilot` | This scaffold: templates, scripts, skills, results. |
| `/sgl-workspace/FlyDSL-<slug>` | The per-task worktree, created fresh per draft/plan. |

Keep `FlyDSL-lab` main clean. Never run a loop on `main` or on the shared
`FlyDSL-lab` working tree -- always on a dedicated `rlcr/<slug>` worktree.

## What this skill DOES (autonomous, no human needed)

1. Pick a task slug and template kind:
   - `default` -> bf16/fp16 first-pass FlashAttention contract
   - `deep` -> deeper kernel-body contract (Loop-02 style)
   - `fp8` -> fp8 FlashAttention contract (ROCm/FlyDSL#698)
2. Provision the worktree + branch + draft + bindings in one command:

   ```bash
   cd /sgl-workspace/rocm-KDA-pilot
   bash scripts/new_flydsl_task.sh --slug <slug> --template <default|deep|fp8> \
     --base <locked-base-ref> [--build-from /path/to/built-FlyDSL]
   ```

   This creates `/sgl-workspace/FlyDSL-<slug>` on branch `rlcr/<slug>` from the
   locked base, writes `.humanize/kernel-agent/draft.md`, wires `_mlir`
   bindings, and runs preflight.

3. If the task is backed by a live GitHub issue rather than a checked-in
   template, snapshot the issue into the draft instead of (or appended to) the
   template:

   ```bash
   cd /sgl-workspace/FlyDSL-<slug>
   ISSUE_URL="https://github.com/ROCm/FlyDSL/issues/<number>"
   {
     echo "# Humanize Draft From GitHub Issue"
     echo
     gh issue view "$ISSUE_URL" --comments
   } >> .humanize/kernel-agent/draft.md
   ```

4. Review the draft for sanity (refs resolve, K/R/W present, baseline named):

   ```bash
   cd /sgl-workspace/rocm-KDA-pilot
   bash scripts/review_humanize_artifact.sh /sgl-workspace/FlyDSL-<slug> draft --terminal
   ```

5. Generate the plan (this is automatable; reviewing it is not):

   ```text
   /humanize:gen-plan --input .humanize/kernel-agent/draft.md --output .humanize/kernel-agent/refined-plan.md --direct
   ```

Then STOP and hand off.

## What this skill must NOT do (human gates)

- **GATE 1 -- plan review/refine.** A human reads
  `.humanize/kernel-agent/refined-plan.md`, checks the K/R/W contract, the
  correctness gates, the promotion bar, and the formal-outcome criteria, and
  edits the plan if it is wrong. Do not start a loop on an unreviewed plan.
- **GATE 2 -- loop start.** A human runs `/humanize:start-rlcr-loop` with the
  exact locked `--base-branch`. The loop mutates code and burns real GPU/agent
  time; a human owns that go decision.

Present these two as explicit handoffs. Do not auto-advance past them even in
bypass-permissions mode.

## Naming conventions

- Worktree dir: `/sgl-workspace/FlyDSL-<slug>` (e.g. `FlyDSL-fa-fp8`).
- Branch: `rlcr/<slug>` (the loop work branch), created from the locked base.
- Slug should be short and task-describing: `fa-fp8`, `fa-splitk`, `fa-mfma-k32`.

## Cleanup

Old task worktrees accumulate. To list and assess them:

```bash
cd /sgl-workspace/FlyDSL-lab && git worktree list
```

Only remove a worktree after confirming it has no uncommitted changes and no
unpushed local-only commits (a branch with no upstream, or commits ahead of its
upstream, is local-only -- pushing or tagging it first preserves the work):

```bash
git -C /sgl-workspace/FlyDSL-lab worktree remove /sgl-workspace/FlyDSL-<slug>
```

When in doubt, leave it and ask the human. Losing an unpushed candidate branch
is worse than a stale directory.

See `skills/rocm-kda-pilot/SKILL.md` for the loop-execution side and
`docs/terminology.md` for loop/round/AC/DEC definitions.
