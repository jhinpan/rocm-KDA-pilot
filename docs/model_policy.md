# ROCm KDA Pilot — Model Policy

This is the first-party model policy for ROCm KDA Pilot, scoped from
[issue #5](https://github.com/jhinpan/rocm-KDA-pilot/issues/5) ("KDA owns model
policy"). It is a **policy document, not a runtime change**: it states which models
the pilot defaults to and why, so loop configs, scripts, and final reports can be
checked against one canonical source instead of scattered script defaults.

## Context

ROCm KDA Pilot runs on a company API path with frontier models available by
default. Upstream Humanize carries defaults from a heterogeneous multi-agent setup
(Sonnet/Haiku-class specialist subagents, model-specific routes). For kernel
optimization on real GPUs we care about reproducible task state, correctness /
benchmark / profiling evidence, and clear human gates — not about preserving
upstream's subagent/model topology. So the pilot pins frontier models by default
and treats smaller models as an explicit, narrow opt-in.

## Policy

### Default routes

| Role | Model | Settings / effort | Used for |
|---|---|---|---|
| **Claude Code** (orchestration) | **Opus 4.8** (company API path) | `ultracode` settings (`--settings ultracode.json`, `--dangerously-skip-permissions`) | Draft/plan authoring, RLCR orchestration, the main loop, exploration, edits |
| **Codex** (implementation / review) | **gpt-5.5** (company API path) | effort **`xhigh`** (i.e. `gpt-5.5:xhigh`) | Independent implementation/verification and round review |

These are the **defaults for every loop** unless a deviation is explicitly chosen
and recorded (see *Deviations*).

### No small-model routes by default

Sonnet/Haiku-class (or other smaller) models are **not** used by default. They may
be used only for an **explicit, cheap, non-critical fan-out** (e.g. a bulk
mechanical scan where a frontier model is wasteful), and only when:

- the choice is stated up front in the plan or loop config, and
- nothing on the correctness / benchmark / decision path depends on it.

Anything that produces evidence, a candidate, a review verdict, or an outcome
decision uses the frontier defaults above.

### Auditability

Every final loop report (`results/loop-NN-*.md`) must record the models actually
used:

- Claude Code model + settings (e.g. `Opus 4.8 / ultracode`).
- Codex model + effort (e.g. `gpt-5.5:xhigh`).
- Any small-model fan-out that was used, with its scope and why it was safe.

If a loop deviated from the defaults, the report states the deviation and the
reason. "Model/tool usage is auditable in the final report" is the acceptance bar.

### Name discipline (preflight)

Model names are exact deployment names. A near-miss (e.g. `gpt5.5` vs `gpt-5.5`) is
accepted by the loop at start but fails at first review, costing a cancel/restart
— this happened in Loop 01. `scripts/preflight.sh` probes the Codex model name
before round 0; the policy default `gpt-5.5:xhigh` must pass that probe.

## Deviations

A loop may deviate from these defaults when there is a concrete reason (cost
ceiling on a throwaway scan, a model outage, an A/B of effort levels). When it
does:

1. State the deviation in the plan or loop start command.
2. Record it in the loop's final report under model auditability.

Renaming defaults here does **not** require a fork of upstream Humanize: the Codex
route is already config-driven (`codex_model` / `codex_effort`, `provider_mode`,
`agent_teams`), so this policy is realized via configuration/flags, not by patching
Humanize internals.

## Why frontier-first (evidence)

The pilot's experiment series supports pinning frontier models rather than relying
on upstream's smaller-model defaults:

- The value of the workflow came from issue→draft→plan, evidence-gated execution,
  and independent Codex review pressure — not from the upstream agent taxonomy
  (see the Loop 01–03 retrospective and
  [`results/loop-04-flashattn-fp8-gfx950.md`](../results/loop-04-flashattn-fp8-gfx950.md)).
- Loop 04 (fp8 FlashAttention, the RFC's suggested spike) was orchestrated by
  Claude Code (Opus 4.8 / ultracode) with Codex (`gpt-5.5:xhigh`) reviewing, and
  produced frontier-level results: a correct, additive fp8 forward
  (`min_cos ≈ 0.99999`), and a non-trivial methodology save — a falsely-recorded
  "fp8 precision-wall" `NO-GO` was retracted after a host-side numerics probe.
  That kind of self-correction is exactly the frontier-model behavior this policy
  defaults to, and it did not need smaller specialist subagents.

## Scope and relationship to issue #5

This document delivers only the **model policy** slice of issue #5. It does not
define the full KDA protocol, evidence schema, or the Humanize adapter layer —
those remain open items in the RFC. It is intentionally a small, low-risk,
documentation-only step that pins the model defaults the pilot already wants, so
later adapter/protocol work has one canonical model policy to reference.
