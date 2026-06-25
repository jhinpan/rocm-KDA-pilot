# ROCm KDA Pilot

ROCm KDA Pilot is a workflow scaffold for running Humanize/KDA-style AMD ROCm
kernel optimization loops.

The core idea is simple:

```text
GitHub issue -> Humanize draft -> Humanize plan -> RLCR loop -> result report
```

This repository does not contain the optimized kernels themselves. It holds the
task framing, skills, templates, scripts, benchmark rules, and result summaries.
Kernel code lands in the target worktree, currently
[`jhinpan/FlyDSL-lab`](https://github.com/jhinpan/FlyDSL-lab), reviewed against a
locked baseline branch.

## What This Repo Is For

- Capture kernel-optimization tasks as GitHub issues.
- Turn those issues into Humanize draft documents.
- Ask Humanize to generate executable plans from those drafts.
- Run RLCR loops where Claude implements and Codex reviews.
- Record `IMPROVEMENT`, `NO-GO`, and `BLOCKED` outcomes with reproducible
  benchmark evidence.

The first worked example is FlyDSL FlashAttention forward on gfx950 / MI350X,
using [`ROCm/FlyDSL`](https://github.com/ROCm/FlyDSL) as the target project.
The detailed write-ups live under [`results/`](results/).

## Issue As Draft

Yes: for our actual workflow, the GitHub issue should be treated as the canonical
draft source.

The issue should say what we want, why it matters, what baseline/review branch to
use, what correctness cannot be weakened, what benchmark or profiling evidence
counts, and what final deliverables are expected. Humanize then converts that
draft into a plan that the RLCR loop can execute and review.

Recommended shape:

```text
GitHub issue
  -> snapshot to .humanize/kernel-agent/draft.md
  -> /humanize:gen-plan
  -> review .humanize/kernel-agent/refined-plan.md
  -> /humanize:start-rlcr-loop
  -> record result under results/
```

Why snapshot the issue into `draft.md` instead of pointing the loop at a moving
issue page? Because the plan should be generated from a stable task contract.
Later comments can become a new draft, a refined plan, or a follow-up issue.

To materialize an issue as a draft from inside the target worktree:

```bash
mkdir -p .humanize/kernel-agent
ISSUE_URL="https://github.com/jhinpan/rocm-KDA-pilot/issues/<number>"
{
  echo "# Humanize Draft From GitHub Issue"
  echo
  gh issue view "$ISSUE_URL" --comments
} > .humanize/kernel-agent/draft.md
```

For new tasks, start from the issue template:
[`Humanize kernel task`](.github/ISSUE_TEMPLATE/humanize-task.md).

## Repository Layout

| Path | Purpose |
|---|---|
| [`templates/`](templates/) | Draft templates for known task families: the default and `--deep` FlashAttention contracts, the [`fp8`](templates/flydsl_flashattn_fp8_gfx950_contract.md) contract for [`ROCm/FlyDSL#698`](https://github.com/ROCm/FlyDSL/issues/698), and the [`mxfp4`](templates/flydsl_mxfp4_moe_gfx950_contract.md) MXFP4 MoE 2-stage tuning contract for [`ROCm/FlyDSL#708`](https://github.com/ROCm/FlyDSL/issues/708). |
| [`skills/rocm-kda-pilot/`](skills/rocm-kda-pilot/) | Project skill for running the loop (execution side). |
| [`skills/flydsl-task-setup/`](skills/flydsl-task-setup/) | Project skill for the prep phase: issue -> worktree -> draft -> plan, stopping at the human plan-review gate. |
| [`scripts/new_flydsl_task.sh`](scripts/new_flydsl_task.sh) | One command: provisions `/sgl-workspace/FlyDSL-<slug>` on `rlcr/<slug>`, writes the draft, wires bindings, runs preflight. |
| [`scripts/bootstrap.sh`](scripts/bootstrap.sh) | Links skills and installs local helper tooling. |
| [`scripts/prepare_flydsl_flashattn_task.sh`](scripts/prepare_flydsl_flashattn_task.sh) | Creates a draft (`--deep` / `--fp8` / `--mxfp4`) in a FlyDSL worktree. |
| [`scripts/preflight.sh`](scripts/preflight.sh) | Checks Codex model naming and FlyDSL runtime bindings before RLCR. |
| [`docs/`](docs/) | Short contracts for Humanize flow, benchmark rules, profiling rules, FlashAttention invariants, and terminology. |
| [`results/`](results/) | Completed loop result reports. |
| [`external/`](external/) | Submodules for ROCmKernelWiki, flyprof tooling, and ROCm report skill. |

Humanize itself is not vendored here. Install it from
[`PolyArch/humanize`](https://github.com/PolyArch/humanize).

## Quick Start

Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/jhinpan/rocm-KDA-pilot.git
cd rocm-KDA-pilot
```

If already cloned:

```bash
git submodule update --init --recursive
```

Bootstrap helper skills and `flyprof`:

```bash
bash scripts/bootstrap.sh
```

Install Humanize in Claude Code:

```text
/plugin marketplace add https://github.com/PolyArch/humanize.git
/plugin install humanize@PolyArch
/reload-plugins
```

Verify the commands exist:

```text
/humanize:gen-plan
/humanize:start-rlcr-loop
/humanize:ask-codex
```

Do not paste multiple Claude slash commands at once; some Claude Code versions
concatenate pasted slash-command lines.

## Using The Skills

After `bash scripts/bootstrap.sh`, Claude Code can see two project skills:

| Skill | Use it for | Stops before |
|---|---|---|
| `flydsl-task-setup` | Prep: issue -> FlyDSL worktree -> draft -> Humanize plan. | Human plan-review gate. |
| `rocm-kda-pilot` | Execution: reviewed plan -> RLCR loop -> optimization evidence -> result report. | Final result reporting / operator decisions. |

The usual operator flow is:

```text
1. Write or choose a GitHub issue
2. Ask Claude to use flydsl-task-setup
3. Review/refine the generated Humanize plan
4. Ask Claude to use rocm-kda-pilot, or run /humanize:start-rlcr-loop directly
5. Let RLCR run the auto loop
6. Record the final outcome under results/
```

Example request to Claude Code from `/sgl-workspace/rocm-KDA-pilot`:

```text
Use the flydsl-task-setup skill for ROCm/FlyDSL#698.
Slug: fa-fp8.
Template: fp8.
Base: upstream/main for now.
Create the FlyDSL worktree, write the draft, run preflight, generate the Humanize plan,
then stop before the human plan-review gate.
```

The skill should prepare `/sgl-workspace/FlyDSL-fa-fp8`, write
`.humanize/kernel-agent/draft.md`, run preflight, and generate
`.humanize/kernel-agent/refined-plan.md`. The live execution artifacts stay in the
FlyDSL worktree; this repo keeps the issue, workflow docs, templates, and final
result reports.

After reviewing the plan, start the loop from the FlyDSL task worktree:

```text
/humanize:start-rlcr-loop .humanize/kernel-agent/refined-plan.md --skip-quiz --claude-answer-codex --max 12 --codex-model gpt-5.5:xhigh --codex-timeout 5400 --base-branch <locked-baseline-branch>
```

`--skip-quiz --claude-answer-codex` is the normal autonomous mode: Claude keeps
working, Codex reviews at round boundaries, and Humanize continues until
`IMPROVEMENT`, evidence-backed `NO-GO`, `BLOCKED`, max rounds, or operator stop.
Do not use it until the plan and `<locked-baseline-branch>` have been reviewed by
a human.

## Running A Task

The fastest path on our fixed `/sgl-workspace` layout is the provisioning
script, which creates the worktree, branch, draft, and bindings in one step:

```bash
cd /sgl-workspace/rocm-KDA-pilot
bash scripts/new_flydsl_task.sh --slug <slug> --template <default|deep|fp8> --base <locked-base-ref>
```

See [Example: fp8 FlashAttention From Issue #698](#example-fp8-flashattention-from-issue-698)
for the full walkthrough. The manual steps below are the underlying mechanics.

Use a target kernel worktree for the actual code changes. For FlyDSL work, that
is usually a checkout of [`jhinpan/FlyDSL-lab`](https://github.com/jhinpan/FlyDSL-lab).

From the target worktree, create or copy the issue-backed draft:

```bash
mkdir -p .humanize/kernel-agent
gh issue view "https://github.com/jhinpan/rocm-KDA-pilot/issues/<number>" --comments \
  > .humanize/kernel-agent/draft.md
```

For the FlashAttention example, the helper script can still generate the draft
from a checked-in template:

```bash
# from this repo
bash scripts/prepare_flydsl_flashattn_task.sh /path/to/FlyDSL-worktree

# for the deeper follow-up contract
bash scripts/prepare_flydsl_flashattn_task.sh --deep /path/to/FlyDSL-worktree
```

Run preflight before starting the loop:

```bash
bash scripts/preflight.sh /path/to/FlyDSL-worktree --codex-model gpt-5.5:xhigh
```

Start Claude Code in the target worktree:

```bash
cd /path/to/FlyDSL-worktree
claude --permission-mode bypassPermissions
```

Generate the plan:

```text
/humanize:gen-plan --input .humanize/kernel-agent/draft.md --output .humanize/kernel-agent/refined-plan.md --direct
```

Review the plan before executing it:

```bash
less .humanize/kernel-agent/refined-plan.md
```

Start RLCR:

```text
/humanize:start-rlcr-loop .humanize/kernel-agent/refined-plan.md --skip-quiz --claude-answer-codex --max 12 --codex-model gpt-5.5:xhigh --codex-timeout 5400 --base-branch <locked-baseline-branch>
```

Use the exact baseline branch for the task. Do not guess it. Humanize/Codex
review quality depends on a clean, immutable comparison base.

## Example: fp8 FlashAttention From Issue #698

This is the canonical end-to-end walkthrough, from a GitHub issue to a running
loop, on our fixed `/sgl-workspace` layout. The task is
[`ROCm/FlyDSL#698`](https://github.com/ROCm/FlyDSL/issues/698): add an fp8
FlashAttention forward path on gfx950 and push it toward the aiter asm fp8 level
(~`2000+T`) without regressing the existing bf16 baseline (~`1300+T`).

The fixed layout means none of the paths below need to be guessed:

| Path | Role |
|---|---|
| `/sgl-workspace/FlyDSL-lab` | Host checkout of `jhinpan/FlyDSL-lab`; owns all FlyDSL worktrees. `origin` = jhinpan/FlyDSL-lab, `upstream` = ROCm/FlyDSL. |
| `/sgl-workspace/rocm-KDA-pilot` | This scaffold. |
| `/sgl-workspace/FlyDSL-fa-fp8` | The per-task worktree this example creates. |

### Step 0 -- the issue is the draft source

Issue #698 is sparse on its own ("port aiter asm fp8, 1300+T -> 2000+T"). The
[`fp8` contract template](templates/flydsl_flashattn_fp8_gfx950_contract.md)
supplies the K/R/W contract, fp8 numerics (gfx950 = `e4m3fn`), correctness gates,
and outcome criteria the issue text leaves implicit.

### Step 1 -- provision the task worktree (automatable)

One command creates the worktree, branch, draft, and binding wiring. A Claude
Code instance using the [`flydsl-task-setup`](skills/flydsl-task-setup/) skill
can own this:

```bash
cd /sgl-workspace/rocm-KDA-pilot
bash scripts/new_flydsl_task.sh --slug fa-fp8 --template fp8 \
  --base upstream/main \
  --build-from /sgl-workspace/FlyDSL-lab
```

This produces `/sgl-workspace/FlyDSL-fa-fp8` on branch `rlcr/fa-fp8`, writes
`.humanize/kernel-agent/draft.md`, wires `_mlir` bindings, and runs preflight.
Use the exact locked baseline for `--base` once you have one; `upstream/main` is
only the default.

Optionally append the live issue text to the draft:

```bash
cd /sgl-workspace/FlyDSL-fa-fp8
{ echo; echo "# Live Issue Snapshot"; echo; \
  gh issue view https://github.com/ROCm/FlyDSL/issues/698 --comments; \
} >> .humanize/kernel-agent/draft.md
```

### Step 2 -- generate the plan (automatable)

```bash
cd /sgl-workspace/FlyDSL-fa-fp8
claude --permission-mode bypassPermissions
```

```text
/humanize:gen-plan --input .humanize/kernel-agent/draft.md --output .humanize/kernel-agent/refined-plan.md --direct
```

### Step 3 -- review and refine the plan (HUMAN GATE)

Stop here. A human reads the plan, confirms the fp8 numerics, the correctness
gates, the aiter-asm parity bar, and the formal-outcome criteria, and edits it if
it is wrong. The loop should never start from an unreviewed plan.

```bash
cd /sgl-workspace/rocm-KDA-pilot
bash scripts/review_humanize_artifact.sh /sgl-workspace/FlyDSL-fa-fp8 refined --terminal
```

### Step 4 -- start the loop (HUMAN GATE)

A human starts the loop with the exact locked baseline:

```text
/humanize:start-rlcr-loop .humanize/kernel-agent/refined-plan.md --skip-quiz --claude-answer-codex --max 12 --codex-model gpt-5.5:xhigh --codex-timeout 5400 --base-branch <locked-baseline-branch>
```

### Why these two gates stay human

Prep -- snapshotting the issue, provisioning a worktree, generating a first plan
-- is mechanical and deterministic, so a skill does it well. Deciding whether the
plan is correct, and whether to spend real GPU/agent time running it, are
judgment calls; those stay with the operator. See
[`skills/flydsl-task-setup/`](skills/flydsl-task-setup/) for the automatable half
and [`skills/rocm-kda-pilot/`](skills/rocm-kda-pilot/) for the loop side.

### When the loop finishes

Record the outcome (`IMPROVEMENT`, `NO-GO`, or `BLOCKED`) under
[`results/`](results/), following the existing FlashAttention loop reports.

## FlashAttention Example And Results

FlashAttention forward on gfx950 is now an example/result set, not the whole
identity of this repo.

There are now **two example families**: Loops 01–03 *optimized* the existing
bf16/f16 FlashAttention; Loop 04 *adds a new dtype* (fp8 e4m3fn) from scratch and
chases aiter-asm-fp8 throughput parity.

The bf16 example was grounded in:

- [`ROCm/FlyDSL#683`](https://github.com/ROCm/FlyDSL/pull/683): the working
  FlashAttention baseline and canonical test/benchmark harness.
- [`ROCm/FlyDSL#670`](https://github.com/ROCm/FlyDSL/pull/670): historical
  optimization context for dwordx4 O-store and split-K direction.
- [`jhinpan/FlyDSL-lab`](https://github.com/jhinpan/FlyDSL-lab): the working fork
  where optimization branches were pushed.

The fp8 example (Loop 04) is grounded in the same harness plus:

- [`ROCm/FlyDSL#698`](https://github.com/ROCm/FlyDSL/issues/698): the *[Feature]
  fp8 flash attention* issue used as the draft.
- aiter **native ASM fp8** (`fmha_v3_fwd`, `how_v3_bf16_cvt=0`) as the parity
  target, with aiter ck fp8 as a secondary comparison.

Completed loops:

| Loop | Result | Takeaway |
|---|---|---|
| [`01`](results/loop-01-flashattn-gfx950.md) | `IMPROVEMENT`: promoted dispatch win, about `1.56x` mean speedup for dense `S=128`; correctness and coverage preserved; upstream draft PR [`ROCm/FlyDSL#685`](https://github.com/ROCm/FlyDSL/pull/685). | The first draft rewarded "at least one promoted candidate", so the loop found the cheapest safe win: dispatch routing. |
| [`02`](results/loop-02-flashattn-gfx950-deep.md) | `NO-GO`: no in-body optimization of the long dualwave kernel was landable. | The kernel body is at a barrier/occupancy co-optimum; credible in-body levers were load-bearing, neutral, or slower. |
| [`03`](results/loop-03-flashattn-gfx950-variant.md) | `NO-GO`: no short/mid specialized variant cleared the bar; best lever was about `1%`, below the promotion bar. | After the dispatch gate, FlyDSL is already competitive across most of the family; the remaining small-batch mid-sequence gap is structural. |
| [`04`](results/loop-04-flashattn-fp8-gfx950.md) | `IMPROVEMENT` (correct, additive, merge-ready **fp8 e4m3fn forward**, `min_cos ≈ 0.99999`); **AC-8 asm-fp8 throughput parity OPEN at ~66%**. Issue [`ROCm/FlyDSL#698`](https://github.com/ROCm/FlyDSL/issues/698). | A new dtype, not a tuning pass. Correctness is solved and a false "precision-wall" `NO-GO` was retracted via host numerics; parity is gated by a tractable true-fp8 V-staging layout, and profiling shows the gap is barrier/bandwidth-bound (MFMA only ~8%), not arithmetic. |

For a browser-friendly retrospective, open
[`results/loops-summary.html`](results/loops-summary.html).

The important process lesson is that `NO-GO` is a valid deliverable when it is
backed by correctness, benchmark, profiling, and ISA evidence — and, from Loop 04,
that a `NO-GO` claiming a *fundamental* limit must itself be verified (the
"precision wall" was retracted by a cheap idealized-numerics probe).

## Benchmark And Profiling Contracts

Keep the runtime contract short and explicit:

- Use the same harness, input distribution, GPU, dtype, causal mode, warmup, and
  iteration count for baseline and candidate.
- Preserve correctness thresholds and coverage.
- Report per-shape timings, grouped averages or geomean, exact command, commit,
  GPU, and artifact paths.
- Profile only to answer a named question.
- Keep raw profile/trace artifacts untracked; commit summaries and small reports.

References:

- [`docs/benchmark_contract.md`](docs/benchmark_contract.md)
- [`docs/profiling_contract.md`](docs/profiling_contract.md)
- [`docs/flydsl_flashattn_rules.md`](docs/flydsl_flashattn_rules.md)
- [`docs/humanize_flow.md`](docs/humanize_flow.md) (includes the convergence guardrails)
- [`docs/result_report_template.md`](docs/result_report_template.md)

## Terminology

The canonical outcome unit is a **loop**. We do not distinguish between
multiple names for that same unit in this repo.

- **Draft**: the task contract Humanize reads first. In our workflow, this is
  usually a snapshot of a GitHub issue.
- **Plan**: the executable implementation contract generated from the draft and
  reviewed by the human before RLCR starts.
- **Loop**: one outcome-bearing `/humanize:start-rlcr-loop` run. A loop contains
  multiple rounds and ends in a result: `IMPROVEMENT`, `NO-GO`, `BLOCKED`, or an
  explicitly cancelled/deferred follow-up.
- **Round**: one completion-attempt/review boundary inside a loop: Claude
  believes the plan is complete, writes a summary, and Codex reviews that
  summary and/or diff. A round is not one task, milestone, or candidate.
- **AC**: acceptance criterion. A numbered condition from the plan, such as
  "AC-4: no dispatch-only win counts for this loop."
- **DEC**: decision record. A numbered human/operator decision that changes or
  clarifies scope, such as accepting an evidence-backed `NO-GO` or lifting a
  constraint.
- **Candidate**: one proposed optimization tried inside a loop.
- **Promotion**: accepting a candidate because it passed correctness,
  performance, and evidence gates.
- **NO-GO**: a valid loop result where no candidate clears the bar, but the loop
  produces enough evidence to explain why. This is the formal outcome name; it
  replaces the looser phrase "negative result."

See [`docs/terminology.md`](docs/terminology.md) for the full glossary and the
Claude/Codex interaction model.

## Repository Hygiene

Keep these untracked:

- `.humanize*`
- raw rocprof / ATT artifacts
- caches
- build outputs
- large CSV or trace dumps

Commit only:

- source changes that are part of the candidate
- harness changes that are part of the task contract
- small summarized benchmark/profiling notes
- ledgers and final reports

## References And Credits

Directly used by this workflow:

| Project | Role |
|---|---|
| [`ROCm/FlyDSL`](https://github.com/ROCm/FlyDSL) | Target compiler/runtime/kernel repository. |
| [`jhinpan/FlyDSL-lab`](https://github.com/jhinpan/FlyDSL-lab) | Working fork where KDA optimization branches are pushed. |
| [`PolyArch/humanize`](https://github.com/PolyArch/humanize) | `gen-plan` and `start-rlcr-loop` provider. |
| [`jhinpan/ROCmKernelWiki`](https://github.com/jhinpan/ROCmKernelWiki) | ROCm kernel knowledge skill. |
| [`jhinpan/flydsl-rocprof-cli`](https://github.com/jhinpan/flydsl-rocprof-cli) | `flyprof` profiling CLI and companion skills. |
| [`jhinpan/rocm-report-skill`](https://github.com/jhinpan/rocm-report-skill) | Turns ROCm profiling artifacts into optimization hypotheses. |
| [`ROCm/aiter`](https://github.com/ROCm/aiter) | Optional comparison backend for FlashAttention benchmarks. |
| [`openai/codex`](https://github.com/openai/codex) | Independent review agent used by Humanize. |

Methodology references:

| Project | Relationship |
|---|---|
| [`BBuf/KDA-Pilot`](https://github.com/BBuf/KDA-Pilot) | Main inspiration for task-owned worktrees and evidence-led kernel optimization. |
| [`mit-han-lab/kernel-design-agents`](https://github.com/mit-han-lab/kernel-design-agents) | K/R/W task framing inspiration. |
| [`mit-han-lab/KernelWiki`](https://github.com/mit-han-lab/KernelWiki) | Knowledge-base pattern that inspired ROCmKernelWiki. |
| [`mit-han-lab/ncu-report-skill`](https://github.com/mit-han-lab/ncu-report-skill) | Report-skill pattern that inspired rocm-report-skill. |

Suggested citation:

```bibtex
@software{rocm_kda_pilot_2026,
  title        = {ROCm KDA Pilot: Humanize/KDA-style ROCm Kernel Optimization Workflow},
  author       = {Jhin Pan},
  year         = {2026},
  url          = {https://github.com/jhinpan/rocm-KDA-pilot},
  note         = {Workflow scaffold for ROCm kernel optimization with FlyDSL FlashAttention results}
}
```
