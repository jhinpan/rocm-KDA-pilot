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
| [`templates/`](templates/) | Draft templates for known task families. Useful when bootstrapping an issue or a first run. |
| [`skills/rocm-kda-pilot/`](skills/rocm-kda-pilot/) | Project skill that tells Claude how to run this workflow. |
| [`scripts/bootstrap.sh`](scripts/bootstrap.sh) | Links skills and installs local helper tooling. |
| [`scripts/prepare_flydsl_flashattn_task.sh`](scripts/prepare_flydsl_flashattn_task.sh) | Creates a FlashAttention draft in a FlyDSL worktree. |
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

## Running A Task

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

## FlashAttention Example And Results

FlashAttention forward on gfx950 is now an example/result set, not the whole
identity of this repo.

The original example was grounded in:

- [`ROCm/FlyDSL#683`](https://github.com/ROCm/FlyDSL/pull/683): the working
  FlashAttention baseline and canonical test/benchmark harness.
- [`ROCm/FlyDSL#670`](https://github.com/ROCm/FlyDSL/pull/670): historical
  optimization context for dwordx4 O-store and split-K direction.
- [`jhinpan/FlyDSL-lab`](https://github.com/jhinpan/FlyDSL-lab): the working fork
  where optimization branches were pushed.

Completed loops:

| Loop | Result | Takeaway |
|---|---|---|
| [`01`](results/loop-01-flashattn-gfx950.md) | `IMPROVEMENT`: promoted dispatch win, about `1.56x` mean speedup for dense `S=128`; correctness and coverage preserved; upstream draft PR [`ROCm/FlyDSL#685`](https://github.com/ROCm/FlyDSL/pull/685). | The first draft rewarded "at least one promoted candidate", so the loop found the cheapest safe win: dispatch routing. |
| [`02`](results/loop-02-flashattn-gfx950-deep.md) | `NO-GO`: no in-body optimization of the long dualwave kernel was landable. | The kernel body is at a barrier/occupancy co-optimum; credible in-body levers were load-bearing, neutral, or slower. |
| [`03`](results/loop-03-flashattn-gfx950-variant.md) | `NO-GO`: no short/mid specialized variant cleared the bar; best lever was about `1%`, below the promotion bar. | After the dispatch gate, FlyDSL is already competitive across most of the family; the remaining small-batch mid-sequence gap is structural. |

For a browser-friendly retrospective, open
[`results/loops-summary.html`](results/loops-summary.html).

The important process lesson is that `NO-GO` is a valid deliverable when it is
backed by correctness, benchmark, profiling, and ISA evidence.

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
- [`docs/humanize_flow.md`](docs/humanize_flow.md)

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
