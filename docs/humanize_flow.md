# Humanize Flow For ROCm KDA Pilot

Run Humanize from the FlyDSL task worktree, not from this repo.

See [`terminology.md`](terminology.md) for the canonical loop/round/AC/DEC
definitions.

1. Generate a plan:

```text
/humanize:gen-plan --input .humanize/kernel-agent/draft.md --output .humanize/kernel-agent/refined-plan.md --direct
```

2. Review `.humanize/kernel-agent/refined-plan.md`.

From the ROCm KDA Pilot checkout, terminal review:

```bash
bash scripts/review_humanize_artifact.sh /path/to/FlyDSL-fa-kda refined --terminal
```

HTML review:

```bash
bash scripts/review_humanize_artifact.sh /path/to/FlyDSL-fa-kda refined --html
```

3. Start RLCR:

```text
/humanize:start-rlcr-loop .humanize/kernel-agent/refined-plan.md --skip-quiz --claude-answer-codex --max 12 --codex-model gpt-5.5:xhigh --codex-timeout 5400 --base-branch rocm-kda-base/flydsl-flashattn-gfx950-pr683
```

The base branch must point to the clean PR683 baseline. Do not guess it.

## Convergence guardrails (fp8 FlashAttention loop lessons)

These guardrails keep a loop from spending its whole iteration budget circling a
single hard or possibly-infeasible acceptance criterion. They are derived from a
maxiter loop whose review discipline was high but whose convergence was not: one
"match-an-external-reference" criterion stayed unmet from round 0 to the budget
exit, and its handling oscillated (declared NO-GO -> accepted -> reopened).

1. **Binding-constraint escalation gate.** When a round's review concludes the
   *only* remaining mainline gap is a single hard or possibly-infeasible
   criterion, pause for an explicit human go/no-go instead of spending more
   rounds. Offer three options:
   - amend the criterion (and write it back into the plan, see #3), or
   - accept a documented `NO-GO`/`BLOCKED` as the terminal outcome, or
   - authorize a bounded number of extra rounds with a stated kill-criterion.
   Treat the reviewer's "estimated remaining rounds" as a control signal: when it
   exceeds the remaining budget, that triggers this gate rather than being noted
   in prose and ignored.

2. **Falsify before you finalize a NO-GO.** Before any blocking `NO-GO` is
   accepted, run the cheapest experiment that directly tests its stated
   root-cause hypothesis, and record the result. A `NO-GO` is not eligible for
   acceptance until that experiment has run. (In the fp8 loop a structural
   "fundamental limit" NO-GO was accepted and triggered several finalization
   rounds; a cheap host-side numerics probe later falsified its premise and
   showed the real blocker was a tractable layout defect. The probe cost a
   fraction of one round; accepting the NO-GO cost several.)

3. **Write human scope decisions back into the plan immediately.** When a human
   authorizes a scope change (e.g. accepting a NO-GO, relaxing a target), amend
   the acceptance criteria in the same step and record it as the new baseline. A
   scope decision that is not written back is treated as not having happened. If
   user direction and the frozen plan conflict, the loop must stop and reconcile
   them, not silently pick one. (In the fp8 loop an unwritten approval to accept a
   NO-GO later collided with the still-immutable criterion and forced a reopen.)

4. **A within-round premise reversal closes the round.** If mid-round evidence
   overturns the round's own central premise, end the round and re-plan rather
   than continuing into a second implementation. Summaries report the final state
   only; superseded intermediate hypotheses go to the ledger, not inline.
