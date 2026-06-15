#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLAUDE_SKILLS_DIR="${CLAUDE_SKILLS_DIR:-$HOME/.claude/skills}"

log() {
  printf '[rocm-kda-bootstrap] %s\n' "$*"
}

link_skill() {
  local name="$1"
  local target="$2"
  local link="$CLAUDE_SKILLS_DIR/$name"

  if [[ ! -f "$target/SKILL.md" ]]; then
    printf '[rocm-kda-bootstrap] missing SKILL.md for %s at %s\n' "$name" "$target" >&2
    exit 1
  fi

  mkdir -p "$CLAUDE_SKILLS_DIR"
  if [[ -L "$link" ]]; then
    rm "$link"
  elif [[ -e "$link" ]]; then
    printf '[rocm-kda-bootstrap] %s exists and is not a symlink; move it aside first\n' "$link" >&2
    exit 1
  fi
  ln -s "$target" "$link"
  log "linked skill $name -> $target"
}

log "repo root: $ROOT"
git -C "$ROOT" submodule update --init --recursive

link_skill "ROCmKernelWiki" "$ROOT/external/ROCmKernelWiki"
link_skill "rocm-report-skill" "$ROOT/external/rocm-report-skill"
link_skill "rocm-kda-pilot" "$ROOT/skills/rocm-kda-pilot"

for skill_dir in "$ROOT"/external/flydsl-rocprof-cli/skills/*; do
  [[ -d "$skill_dir" && -f "$skill_dir/SKILL.md" ]] || continue
  link_skill "$(basename "$skill_dir")" "$skill_dir"
done

if [[ -f "$ROOT/external/ROCmKernelWiki/requirements.txt" ]]; then
  log "installing ROCmKernelWiki requirements"
  python3 -m pip install -r "$ROOT/external/ROCmKernelWiki/requirements.txt"
fi

log "installing flyprof"
python3 -m pip install -e "$ROOT/external/flydsl-rocprof-cli"

cat <<'EOF'

Bootstrap complete.

Install Humanize inside Claude Code if you have not already:

  /plugin marketplace add https://github.com/PolyArch/humanize.git
  /plugin install humanize@PolyArch

Then verify:

  /humanize:gen-plan
  /humanize:start-rlcr-loop

EOF
