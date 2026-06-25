#!/usr/bin/env bash
# Provision a fresh FlyDSL task worktree for a ROCm KDA Pilot loop.
#
# One command takes you from "I have an issue/task" to "a clean, binding-wired
# worktree with a draft ready for /humanize:gen-plan". It is deterministic and
# assumes the fixed two-repo layout under /sgl-workspace:
#
#   /sgl-workspace/FlyDSL-lab        -> the jhinpan/FlyDSL-lab checkout that owns
#                                       all FlyDSL worktrees (git worktree host)
#   /sgl-workspace/rocm-KDA-pilot    -> this scaffold (templates/scripts/skills)
#   /sgl-workspace/FlyDSL-<slug>     -> the per-task worktree this script creates
#
# It does NOT generate a plan and does NOT start a loop. Those stay behind the
# human review gates (see skills/flydsl-task-setup/SKILL.md). This script only
# does the mechanical prep that a Claude Code instance can safely own.
#
# Usage:
#   scripts/new_flydsl_task.sh --slug fa-fp8 [options]
#
# Required:
#   --slug <name>        Task slug. Worktree = /sgl-workspace/FlyDSL-<slug>,
#                        branch = rlcr/<slug> (override with --branch).
#
# Options:
#   --template <kind>    draft template: default | deep | fp8 | mxfp4  (default: default)
#   --base <ref>         Locked base ref to branch from. Default: upstream/main.
#                        Use the exact baseline you will review against (e.g. a
#                        PR683 baseline branch). Do not guess it for the loop.
#   --branch <name>      Override branch name (default: rlcr/<slug>).
#   --build-from <path>  Sibling built FlyDSL checkout to wire _mlir from
#                        (passed through to preflight --fix-bindings).
#   --codex-model <m>    Codex model for preflight probe (default gpt-5.5:xhigh).
#   --no-preflight       Skip the preflight step.
#   --force              Reuse the worktree dir/branch if they already exist.
#   -h, --help           Show this help.
set -euo pipefail

WORKSPACE="${KDA_WORKSPACE:-/sgl-workspace}"
FLYDSL_LAB="${KDA_FLYDSL_LAB:-$WORKSPACE/FlyDSL-lab}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SLUG=""
TEMPLATE_KIND="default"
BASE_REF="upstream/main"
BRANCH=""
BUILD_FROM=""
CODEX_MODEL="gpt-5.5:xhigh"
DO_PREFLIGHT=1
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --slug) SLUG="$2"; shift 2 ;;
    --template) TEMPLATE_KIND="$2"; shift 2 ;;
    --base) BASE_REF="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --build-from) BUILD_FROM="$2"; shift 2 ;;
    --codex-model) CODEX_MODEL="$2"; shift 2 ;;
    --no-preflight) DO_PREFLIGHT=0; shift ;;
    --force) FORCE=1; shift ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$SLUG" ]]; then
  echo "error: --slug is required" >&2
  exit 2
fi
case "$TEMPLATE_KIND" in
  default|deep|fp8|mxfp4) ;;
  *) echo "error: --template must be one of: default deep fp8 mxfp4" >&2; exit 2 ;;
esac

[[ -n "$BRANCH" ]] || BRANCH="rlcr/$SLUG"
WORKTREE="$WORKSPACE/FlyDSL-$SLUG"

if ! git -C "$FLYDSL_LAB" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "error: FlyDSL-lab host not found at $FLYDSL_LAB" >&2
  echo "       set KDA_FLYDSL_LAB to override." >&2
  exit 1
fi

echo "== new FlyDSL task =="
echo "  slug:      $SLUG"
echo "  worktree:  $WORKTREE"
echo "  branch:    $BRANCH"
echo "  base ref:  $BASE_REF"
echo "  template:  $TEMPLATE_KIND"

# Make sure the base ref is current (best-effort; never fail the run on fetch).
git -C "$FLYDSL_LAB" fetch upstream -q 2>/dev/null || true
git -C "$FLYDSL_LAB" fetch origin -q 2>/dev/null || true

if ! git -C "$FLYDSL_LAB" rev-parse --verify --quiet "$BASE_REF^{commit}" >/dev/null; then
  echo "error: base ref '$BASE_REF' does not resolve in $FLYDSL_LAB" >&2
  exit 1
fi

# --- create the worktree + branch ----------------------------------------------
if [[ -e "$WORKTREE" ]]; then
  if [[ $FORCE -eq 1 ]]; then
    echo "  [warn] $WORKTREE exists; reusing (--force)."
  else
    echo "error: $WORKTREE already exists. Use --force to reuse, or pick another --slug." >&2
    exit 1
  fi
else
  if git -C "$FLYDSL_LAB" show-ref --verify --quiet "refs/heads/$BRANCH"; then
    echo "  [info] branch $BRANCH already exists; checking it out into the worktree."
    git -C "$FLYDSL_LAB" worktree add "$WORKTREE" "$BRANCH"
  else
    git -C "$FLYDSL_LAB" worktree add -b "$BRANCH" "$WORKTREE" "$BASE_REF"
  fi
fi

# --- write the draft -----------------------------------------------------------
PREP_ARGS=()
case "$TEMPLATE_KIND" in
  deep)   PREP_ARGS+=(--deep) ;;
  fp8)    PREP_ARGS+=(--fp8) ;;
  mxfp4)  PREP_ARGS+=(--mxfp4) ;;
esac
bash "$ROOT/scripts/prepare_flydsl_flashattn_task.sh" "${PREP_ARGS[@]}" "$WORKTREE"

# --- preflight (bindings + codex model) ----------------------------------------
if [[ $DO_PREFLIGHT -eq 1 ]]; then
  PF_ARGS=("$WORKTREE" --codex-model "$CODEX_MODEL")
  if [[ -n "$BUILD_FROM" ]]; then
    PF_ARGS+=(--build-from "$BUILD_FROM" --fix-bindings)
  fi
  echo
  echo "-- preflight --"
  bash "$ROOT/scripts/preflight.sh" "${PF_ARGS[@]}" || \
    echo "  [warn] preflight reported issues; resolve them before starting the loop."
fi

cat <<EOF

== task worktree ready ==

  $WORKTREE  (branch $BRANCH, based on $BASE_REF)

Review the draft from this rocm-KDA-pilot checkout:
  bash scripts/review_humanize_artifact.sh "$WORKTREE" draft --terminal

NEXT STEPS (human-gated steps are marked):

  1. cd $WORKTREE && claude --permission-mode bypassPermissions
  2. /humanize:gen-plan --input .humanize/kernel-agent/draft.md --output .humanize/kernel-agent/refined-plan.md --direct
  3. [HUMAN REVIEW] read + refine .humanize/kernel-agent/refined-plan.md
     bash scripts/review_humanize_artifact.sh "$WORKTREE" refined --terminal
  4. [HUMAN STARTS LOOP]
     /humanize:start-rlcr-loop .humanize/kernel-agent/refined-plan.md --skip-quiz --claude-answer-codex --max 12 --codex-model $CODEX_MODEL --codex-timeout 5400 --base-branch $BASE_REF

Use the exact locked baseline for --base-branch. Do not guess it.
EOF
