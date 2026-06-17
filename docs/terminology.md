# Terminology

This repo uses one canonical outcome unit: **loop**.

Do not use multiple names for the same outcome unit. A loop is the thing we
start, monitor, stop, and record under `results/`.

## Workflow Units

| Term | Meaning |
|---|---|
| **Draft** | The task contract Humanize reads first. In this workflow it is usually a snapshot of a GitHub issue. It can contain motivation, constraints, prior art, baseline refs, benchmark requirements, and open questions. |
| **Plan** | The executable contract generated from the draft by `/humanize:gen-plan`. A human reviews this before RLCR starts. |
| **Loop** | One outcome-bearing `/humanize:start-rlcr-loop` run. It contains many rounds and ends in a formal outcome: `IMPROVEMENT`, `NO-GO`, `BLOCKED`, or an explicitly cancelled/deferred follow-up. |
| **Round** | One completion-attempt/review boundary inside a loop. Claude believes the plan is complete, writes the loop summary, and Codex reviews the summary and/or diff. A round is not one task, milestone, or candidate. |
| **Candidate** | One proposed optimization or design change tried inside a loop. A loop can reject many candidates before promoting one or concluding `NO-GO`. |
| **Promotion** | The point where a candidate is accepted because it passes correctness, performance, and evidence gates. |

## Outcome Types

These formal outcome names align with BBuf KDA-Pilot.

| Outcome | Meaning |
|---|---|
| `IMPROVEMENT` | A candidate is promoted because it clears correctness, performance, and evidence gates. |
| `NO-GO` | No candidate is promoted, but the loop has enough evidence to explain why the scoped path should not continue. A valid `NO-GO` needs baseline recovery, candidate attempts, correctness status, benchmark/profiling evidence, and a named active bound or blocker. |
| `BLOCKED` | The loop cannot make a valid improvement/no-go decision because required hardware, dependencies, baseline recovery, correctness reference, or benchmark evidence is missing. |

"Negative result" is allowed only as a research-facing synonym for
evidence-backed `NO-GO`; it is not the formal outcome name in this repo.

## Contract Terms

| Term | Meaning |
|---|---|
| **K / R / W** | Kernel / Reference / Workload framing. K names the code surface and invariants; R names the correctness reference; W names the benchmark/profiling workload. |
| **AC** | Acceptance Criterion. A numbered plan condition that defines success, failure, or required evidence. Example: `AC-4` may say dispatch-only wins do not count for a loop. |
| **DEC** | Decision Record. A numbered human/operator decision that changes or clarifies scope. Example: `DEC-3` can lift an earlier in-body-only constraint. |
| **Lower bound** | The minimum result the plan accepts. This can be `IMPROVEMENT`, or an evidence-backed `NO-GO` if the plan says so. |
| **Promotion bar** | The performance and evidence threshold a candidate must clear before it can be called a win. |
| **Locked baseline** | The immutable branch or commit used for review and benchmark comparison. Do not guess it. |

## Agent Roles

| Actor | Role |
|---|---|
| **Human / operator** | Owns the task, reviews the plan, answers questions, records DEC entries, and decides whether an evidence-backed `NO-GO` is acceptable. |
| **Claude Code** | Runs the Humanize commands, edits code, executes tests/benchmarks, writes round summaries, and responds to Codex feedback. |
| **Codex** | Independent reviewer invoked by Humanize. It checks plan compliance, summaries, diffs, unresolved questions, and final review quality. |
| **Humanize** | The control loop that connects the draft, plan, Claude implementation work, Codex review, and loop state. |

## RLCR Shape

RLCR means **Ralph Loop with Codex Review** in Humanize.

```text
GitHub issue
  -> draft.md
  -> /humanize:gen-plan
  -> refined-plan.md
  -> /humanize:start-rlcr-loop
  -> round 0: Claude work + summary -> Codex review
  -> round 1..N: Claude addresses review -> Codex reviews again
  -> final result under results/
```

The important part is that a loop is outcome-driven. If the plan says `NO-GO`
can satisfy the lower bound, the loop should be allowed to conclude with that
outcome once the required AC evidence exists. If the plan only rewards "one
promoted candidate", the loop will often find the cheapest safe win instead of
exploring deeper structural changes.
