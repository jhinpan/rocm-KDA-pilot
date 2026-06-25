#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/prepare_flydsl_flashattn_task.sh [--deep | --fp8 | --mxfp4] /path/to/FlyDSL-worktree

Writes:
  .humanize/kernel-agent/draft.md

Options:
  --deep   Use the deeper Loop-02 contract template
           (templates/flydsl_flashattn_gfx950_deep_contract.md) instead of the
           default first-pass contract. Use this for a second loop that must
           land a kernel-body change and is scored for breadth, not a single
           dispatch tweak. See README and docs/terminology.md.
  --fp8    Use the fp8 contract template
           (templates/flydsl_flashattn_fp8_gfx950_contract.md). This frames the
           ROCm/FlyDSL#698 task: add an fp8 (e4m3fn) FlashAttention forward path
           on gfx950 and optimize it toward the aiter asm fp8 level (~2000+T),
           without regressing the existing bf16/fp16 paths.
  --mxfp4  Use the MXFP4 MoE 2-stage contract template
           (templates/flydsl_mxfp4_moe_gfx950_contract.md). This frames the
           ROCm/FlyDSL#708 task: tune the MXFP4 (per-1x32 fp4) MoE stage1/stage2
           GEMM on gfx950 to raise MFU at large shapes and cut latency at small
           tokens across DeepSeek/Kimi/GPT-OSS, without breaking AITER layout
           compatibility or correctness. Harness: aiter op_tests/test_moe_2stage.py.

--deep, --fp8, and --mxfp4 are mutually exclusive.

The FlyDSL worktree should usually be a jhinpan/FlyDSL-lab fork checkout on a
branch created from upstream ROCm/FlyDSL (which now contains PR683).
EOF
}

DEEP=0
FP8=0
MXFP4=0
ARGS=()
for a in "$@"; do
  case "$a" in
    -h|--help) usage; exit 0 ;;
    --deep) DEEP=1 ;;
    --fp8) FP8=1 ;;
    --mxfp4) MXFP4=1 ;;
    *) ARGS+=("$a") ;;
  esac
done
set -- "${ARGS[@]}"

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 2
fi

if [[ $((DEEP + FP8 + MXFP4)) -gt 1 ]]; then
  echo "--deep, --fp8, and --mxfp4 are mutually exclusive" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLYDSL_ROOT="$(cd "$1" && pwd)"
if [[ $DEEP -eq 1 ]]; then
  TEMPLATE="$ROOT/templates/flydsl_flashattn_gfx950_deep_contract.md"
elif [[ $FP8 -eq 1 ]]; then
  TEMPLATE="$ROOT/templates/flydsl_flashattn_fp8_gfx950_contract.md"
elif [[ $MXFP4 -eq 1 ]]; then
  TEMPLATE="$ROOT/templates/flydsl_mxfp4_moe_gfx950_contract.md"
else
  TEMPLATE="$ROOT/templates/flydsl_flashattn_gfx950_contract.md"
fi
DRAFT_DIR="$FLYDSL_ROOT/.humanize/kernel-agent"
DRAFT="$DRAFT_DIR/draft.md"

if [[ ! -f "$TEMPLATE" ]]; then
  echo "template not found: $TEMPLATE" >&2
  exit 1
fi

if ! git -C "$FLYDSL_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "not a git worktree: $FLYDSL_ROOT" >&2
  exit 1
fi

WORKTREE_HEAD="$(git -C "$FLYDSL_ROOT" rev-parse HEAD)"
WORKTREE_BRANCH="$(git -C "$FLYDSL_ROOT" symbolic-ref --quiet --short HEAD || echo detached)"
FLYDSL_MAIN_SHA="$(git ls-remote https://github.com/ROCm/FlyDSL.git refs/heads/main | awk '{print $1}')"
PR670_SHA="$(git ls-remote https://github.com/ROCm/FlyDSL.git refs/pull/670/head | awk '{print $1}')"
PR683_SHA="$(git ls-remote https://github.com/ROCm/FlyDSL.git refs/pull/683/head | awk '{print $1}')"
TODAY="$(date +%Y-%m-%d)"
# Loop-02 (deep) drafts also reference the Loop-01 dispatch-gate commit, if
# it is present in this worktree's history (best-effort; left as a placeholder
# otherwise for the user to fill in).
ROUND1_GATE_SHA="$(git -C "$FLYDSL_ROOT" log --grep='short seq_len' --grep='_DUALWAVE_MIN_DENSE_SEQ' -i --format=%H -n 1 2>/dev/null || true)"
[[ -z "$ROUND1_GATE_SHA" ]] && ROUND1_GATE_SHA="{{ROUND1_GATE_SHA}}"

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
    "{{ROUND1_GATE_SHA}}": "$ROUND1_GATE_SHA",
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

Preflight the loop config BEFORE starting (validates the Codex model name and
wires/checks the FlyDSL build bindings -- avoids the loop-01 cancel/restart):
  bash scripts/preflight.sh "$FLYDSL_ROOT" --codex-model gpt-5.5:xhigh
  # add --build-from /path/to/built-FlyDSL --fix-bindings if _mlir is missing

Now start Claude Code in:
  $FLYDSL_ROOT

Then run:
  /humanize:gen-plan --input .humanize/kernel-agent/draft.md --output .humanize/kernel-agent/refined-plan.md --direct

After reviewing the refined plan, run:
  /humanize:start-rlcr-loop .humanize/kernel-agent/refined-plan.md --skip-quiz --claude-answer-codex --max 12 --codex-model gpt-5.5:xhigh --codex-timeout 5400 --base-branch <baseline-branch>
EOF
