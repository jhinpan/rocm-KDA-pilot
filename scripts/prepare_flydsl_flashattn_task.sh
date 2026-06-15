#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/prepare_flydsl_flashattn_task.sh /path/to/FlyDSL-worktree

Writes:
  .humanize/kernel-agent/draft.md

The FlyDSL worktree should usually be a jhinpan/FlyDSL-lab fork checkout on a
branch created from upstream ROCm/FlyDSL PR683.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLYDSL_ROOT="$(cd "$1" && pwd)"
TEMPLATE="$ROOT/templates/flydsl_flashattn_gfx950_contract.md"
DRAFT_DIR="$FLYDSL_ROOT/.humanize/kernel-agent"
DRAFT="$DRAFT_DIR/draft.md"

if [[ ! -f "$TEMPLATE" ]]; then
  echo "template not found: $TEMPLATE" >&2
  exit 1
fi

if [[ ! -d "$FLYDSL_ROOT/.git" ]]; then
  echo "not a git worktree: $FLYDSL_ROOT" >&2
  exit 1
fi

WORKTREE_HEAD="$(git -C "$FLYDSL_ROOT" rev-parse HEAD)"
WORKTREE_BRANCH="$(git -C "$FLYDSL_ROOT" symbolic-ref --quiet --short HEAD || echo detached)"
FLYDSL_MAIN_SHA="$(git ls-remote https://github.com/ROCm/FlyDSL.git refs/heads/main | awk '{print $1}')"
PR670_SHA="$(git ls-remote https://github.com/ROCm/FlyDSL.git refs/pull/670/head | awk '{print $1}')"
PR683_SHA="$(git ls-remote https://github.com/ROCm/FlyDSL.git refs/pull/683/head | awk '{print $1}')"
TODAY="$(date +%Y-%m-%d)"

mkdir -p "$DRAFT_DIR"

python3 - "$TEMPLATE" "$DRAFT" <<PY
from pathlib import Path
import sys

template = Path(sys.argv[1]).read_text()
replacements = {
    "{{DATE}}": "$TODAY",
    "{{FLYDSL_MAIN_SHA}}": "$FLYDSL_MAIN_SHA",
    "{{PR670_SHA}}": "$PR670_SHA",
    "{{PR683_SHA}}": "$PR683_SHA",
    "{{WORKTREE_HEAD}}": "$WORKTREE_HEAD",
    "{{WORKTREE_BRANCH}}": "$WORKTREE_BRANCH",
}
for old, new in replacements.items():
    template = template.replace(old, new)
Path(sys.argv[2]).write_text(template)
PY

if [[ ! -f "$FLYDSL_ROOT/.gitignore" ]] || ! grep -qxF ".humanize*" "$FLYDSL_ROOT/.gitignore"; then
  printf '\n.humanize*\n' >> "$FLYDSL_ROOT/.gitignore"
fi

cat <<EOF
Wrote:
  $DRAFT

Review the draft from this rocm-KDA-pilot checkout:
  bash scripts/review_humanize_artifact.sh "$FLYDSL_ROOT" draft --terminal
  bash scripts/review_humanize_artifact.sh "$FLYDSL_ROOT" draft --html

Now start Claude Code in:
  $FLYDSL_ROOT

Then run:
  /humanize:gen-plan --input .humanize/kernel-agent/draft.md --output .humanize/kernel-agent/refined-plan.md --direct

After reviewing the refined plan, run:
  /humanize:start-rlcr-loop .humanize/kernel-agent/refined-plan.md --skip-quiz --claude-answer-codex --max 12 --codex-model gpt-5.5:xhigh --codex-timeout 5400 --base-branch rocm-kda-base/flydsl-flashattn-gfx950-pr683
EOF
