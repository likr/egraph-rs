#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/setup_venv.sh [--force] [venv_dir]

Create or verify an isolated Python 3.12 virtual environment for the current
git checkout / worktree with maturin, pytest, numpy, scipy, mypy, and ruff.
EOF
}

FORCE=0
VENV_ARG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ -n "$VENV_ARG" ]]; then
        echo "Unexpected extra argument: $1" >&2
        usage >&2
        exit 2
      fi
      VENV_ARG="$1"
      shift
      ;;
  esac
done

REPO_ROOT="$(git rev-parse --show-toplevel)"
VENV_DIR="${VENV_ARG:-$REPO_ROOT/.venv}"

is_venv_ready() {
  local py="$1/bin/python"
  [[ -x "$py" ]] && "$py" -c "import maturin, pytest, numpy, scipy, networkx, igraph, mypy, ruff" >/dev/null 2>&1
}

if [[ "$FORCE" -eq 0 ]] && is_venv_ready "$VENV_DIR"; then
  echo "[setup_venv] Ready: $VENV_DIR"
  "$VENV_DIR/bin/python" --version
  exit 0
fi

echo "[setup_venv] Provisioning isolated venv at $VENV_DIR..."
if command -v uv >/dev/null 2>&1; then
  if [[ "$FORCE" -eq 1 && -d "$VENV_DIR" ]]; then
    uv venv --clear --python 3.12 "$VENV_DIR"
  else
    uv venv --allow-existing --python 3.12 "$VENV_DIR"
  fi
  uv pip install --python "$VENV_DIR/bin/python" \
    "maturin>=1.7,<2.0" \
    "pytest>=8.0" \
    "numpy>=1.26" \
    "scipy>=1.11" \
    "networkx>=3.0" \
    "igraph>=0.11" \
    "mypy>=1.11" \
    "ruff>=0.6"
else
  python3 -m venv "$VENV_DIR"
  "$VENV_DIR/bin/python" -m pip install --upgrade pip
  "$VENV_DIR/bin/python" -m pip install \
    "maturin>=1.7,<2.0" \
    "pytest>=8.0" \
    "numpy>=1.26" \
    "scipy>=1.11" \
    "networkx>=3.0" \
    "igraph>=0.11" \
    "mypy>=1.11" \
    "ruff>=0.6"
fi

echo "[setup_venv] Installed toolchain in $VENV_DIR:"
"$VENV_DIR/bin/python" --version
"$VENV_DIR/bin/maturin" --version
"$VENV_DIR/bin/pytest" --version
"$VENV_DIR/bin/mypy" --version
"$VENV_DIR/bin/ruff" --version
