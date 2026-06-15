#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/review_humanize_artifact.sh /path/to/FlyDSL-worktree [draft|refined|path/to/file] [options]

Options:
  --terminal        Print the markdown path and open it with less if available.
  --print           Print markdown to stdout.
  --html            Generate an HTML preview next to the markdown file.
  --serve PORT      Generate HTML and serve its directory with python http.server.
  -h, --help        Show this help.

Examples:
  scripts/review_humanize_artifact.sh /sgl-workspace/FlyDSL-fa-kda draft --terminal
  scripts/review_humanize_artifact.sh /sgl-workspace/FlyDSL-fa-kda draft --html
  scripts/review_humanize_artifact.sh /sgl-workspace/FlyDSL-fa-kda refined --serve 8765
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

WORKTREE="$(cd "$1" && pwd)"
shift

ARTIFACT="${1:-draft}"
if [[ $# -gt 0 && "${1:-}" != --* ]]; then
  shift
fi

MODE_TERMINAL=false
MODE_PRINT=false
MODE_HTML=false
SERVE_PORT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --terminal)
      MODE_TERMINAL=true
      shift
      ;;
    --print)
      MODE_PRINT=true
      shift
      ;;
    --html)
      MODE_HTML=true
      shift
      ;;
    --serve)
      [[ -n "${2:-}" ]] || { echo "--serve requires a port" >&2; exit 2; }
      SERVE_PORT="$2"
      MODE_HTML=true
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$ARTIFACT" in
  draft)
    TARGET="$WORKTREE/.humanize/kernel-agent/draft.md"
    ;;
  refined|plan|refined-plan)
    TARGET="$WORKTREE/.humanize/kernel-agent/refined-plan.md"
    ;;
  *)
    if [[ "$ARTIFACT" = /* ]]; then
      TARGET="$ARTIFACT"
    else
      TARGET="$WORKTREE/$ARTIFACT"
    fi
    ;;
esac

if [[ ! -f "$TARGET" ]]; then
  echo "artifact not found: $TARGET" >&2
  exit 1
fi

if [[ "$MODE_TERMINAL" == false && "$MODE_PRINT" == false && "$MODE_HTML" == false ]]; then
  MODE_TERMINAL=true
fi

generate_html() {
  local md="$1"
  local html="${md%.md}.html"
  python3 - "$md" "$html" <<'PY'
from pathlib import Path
import datetime
import html
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
text = src.read_text(encoding="utf-8")
escaped = html.escape(text)
title = src.name
now = datetime.datetime.now().isoformat(timespec="seconds")
dst.write_text(f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{
      margin: 0;
      background: #f7f7f4;
      color: #151515;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      position: sticky;
      top: 0;
      border-bottom: 1px solid #d6d6cf;
      background: #ffffff;
      padding: 14px 24px;
      z-index: 1;
    }}
    h1 {{
      margin: 0 0 4px;
      font-size: 18px;
      letter-spacing: 0;
    }}
    .meta {{
      color: #5f5f59;
      font-size: 13px;
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 24px;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      overflow-wrap: anywhere;
      font: 14px/1.55 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      background: #ffffff;
      border: 1px solid #dfdfd8;
      border-radius: 8px;
      padding: 20px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <div class="meta">{html.escape(str(src))} - generated {html.escape(now)}</div>
  </header>
  <main>
    <pre>{escaped}</pre>
  </main>
</body>
</html>
""", encoding="utf-8")
print(dst)
PY
}

if [[ "$MODE_PRINT" == true ]]; then
  cat "$TARGET"
fi

if [[ "$MODE_TERMINAL" == true ]]; then
  echo "Markdown artifact:"
  echo "  $TARGET"
  echo
  if command -v less >/dev/null 2>&1 && [[ -t 1 ]]; then
    less -R "$TARGET"
  else
    sed -n '1,240p' "$TARGET"
    total_lines="$(wc -l < "$TARGET" | tr -d ' ')"
    if [[ "$total_lines" -gt 240 ]]; then
      echo
      echo "[truncated at 240 lines; run: less $TARGET]"
    fi
  fi
fi

if [[ "$MODE_HTML" == true ]]; then
  HTML_PATH="$(generate_html "$TARGET")"
  echo "HTML preview:"
  echo "  $HTML_PATH"
  if [[ -n "$SERVE_PORT" ]]; then
    echo
    echo "Serving:"
    echo "  http://127.0.0.1:$SERVE_PORT/$(basename "$HTML_PATH")"
    echo
    python3 -m http.server "$SERVE_PORT" --directory "$(dirname "$HTML_PATH")"
  fi
fi
