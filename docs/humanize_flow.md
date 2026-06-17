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
