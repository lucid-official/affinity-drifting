#!/usr/bin/env bash
# Create .venv with the pinned stack and forbid bytecode caches (CLAUDE.md §7).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
python3 -m venv .venv
SITE=$(.venv/bin/python -c "import sysconfig; print(sysconfig.get_path('purelib'))")
# Executed at every interpreter start-up, before any import: no __pycache__ anywhere.
echo "import sys; sys.dont_write_bytecode = True" > "$SITE/00_affd_no_bytecode.pth"
.venv/bin/pip install --no-compile -q -e ".[dev]"
find . -name __pycache__ -type d -prune -exec rm -rf {} +
echo "ok: $(.venv/bin/python -c 'import sys, lucid; print(sys.dont_write_bytecode, lucid.__version__)')"
