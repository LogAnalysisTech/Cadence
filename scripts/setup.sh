#!/usr/bin/env bash
# Bootstrap the local tools used to submit both agents to Modal.
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo 'Usage: bash scripts/setup.sh'
    echo 'Optional: PYTHON_BIN=/path/to/python3.12 bash scripts/setup.sh'
    echo 'Creates .venv, installs local submission tools, preserves an existing .env.'
    exit 0
fi
if [[ $# -ne 0 ]]; then
    echo 'Unknown arguments. Use --help.' >&2
    exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    for candidate in python3.12 python3.11 python3.13 python3.10 python3 python; do
        if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(not ((3, 10) <= sys.version_info[:2] < (3, 14)))' 2>/dev/null; then
            PYTHON_BIN="$candidate"
            break
        fi
    done
fi
if [[ -z "${PYTHON_BIN:-}" ]]; then
    echo 'Python 3.10–3.13 is required (recommended: 3.12). Install it, then rerun.' >&2
    exit 1
fi
"$PYTHON_BIN" -c 'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 14), "Use Python 3.10–3.13"'

if [[ ! -d .venv ]]; then
    "$PYTHON_BIN" -m venv .venv
fi
.venv/bin/python -m pip install -r requirements-runner.txt
.venv/bin/python -m pip check
.venv/bin/python - <<'PY'
from pathlib import Path

target = Path('.env')
if not target.exists():
    with target.open('x', encoding='utf-8') as handle:
        handle.write(Path('.env.example').read_text(encoding='utf-8'))
    target.chmod(0o600)
    print('Created .env with blank API keys.')
else:
    print('Preserved existing .env.')
PY

for module in monitor_runtime.runners.mini_swe_agent monitor_runtime.runners.moatless monitor_runtime.evaluation.swebench monitor_runtime.evaluation.packager; do
    .venv/bin/modal run -m "$module" --help >/dev/null
done

echo 'Setup complete. Next:'
echo '  source .venv/bin/activate'
echo '  # Fill in .env, then log in to Modal:'
echo '  modal token new'
echo 'See README.md for method selection, evaluation and artifact downloads.'
