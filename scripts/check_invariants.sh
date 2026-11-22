#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/check_invariants.sh [--mode report|strict] [--crate <name>] [--check-wasm-size]

Verify single-threaded pure-Rust, WASM cross-compilation, wasm size budget,
deterministic RNG, and cross-target packaging invariants.
EOF
}

MODE="report"
TARGET_CRATE=""
CHECK_WASM_SIZE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --crate)
      TARGET_CRATE="${2:-}"
      shift 2
      ;;
    --check-wasm-size)
      CHECK_WASM_SIZE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$MODE" != "report" && "$MODE" != "strict" ]]; then
  echo "Invalid --mode '$MODE' (expected report or strict)" >&2
  exit 2
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
export RUST_FONTCONFIG_DLOPEN=1

HARD_FAILS=0
LEGACY_WARNS=0

pass() { echo "[PASS] $*"; }
warn_legacy() {
  LEGACY_WARNS=$((LEGACY_WARNS + 1))
  if [[ "$MODE" == "strict" ]]; then
    HARD_FAILS=$((HARD_FAILS + 1))
    echo "[FAIL] $*"
  else
    echo "[WARN-LEGACY] $*"
  fi
}
fail_hard() {
  HARD_FAILS=$((HARD_FAILS + 1))
  echo "[FAIL] $*"
}

cd "$REPO_ROOT"

# 1. Dependency tree gate (rayon / linfa / wgpu)
TREE_CRATES=()
if [[ -n "$TARGET_CRATE" ]]; then
  TREE_CRATES+=("$TARGET_CRATE")
else
  for candidate in egraph-wasm egraph-python egraph; do
    if cargo metadata --no-deps --format-version 1 2>/dev/null | python3 -c "import sys,json; pkgs={p['name'] for p in json.load(sys.stdin)['packages']}; sys.exit(0 if '$candidate' in pkgs else 1)"; then
      TREE_CRATES+=("$candidate")
    fi
  done
fi

for crate in ${TREE_CRATES[@]+"${TREE_CRATES[@]}"}; do
  TREE_OUT="$(cargo tree -p "$crate" -e features 2>&1 || true)"
  FORBIDDEN="$(printf '%s\n' "$TREE_OUT" | grep -E '(^|[^[:alnum:]_-])(rayon|linfa|linfa-nn|linfa-clustering|wgpu)([^[:alnum:]_-]|$)' || true)"
  if [[ -n "$FORBIDDEN" ]]; then
    warn_legacy "$crate dependency tree pulls forbidden crates:"
    printf '%s\n' "$FORBIDDEN" | sed 's/^/       /'
  else
    pass "$crate dependency tree is free of rayon/linfa/wgpu"
  fi
done

# 2. Static source scan for OS threads / GPU / rayon / non-deterministic RNG
if [[ -d "$REPO_ROOT/crates" ]]; then
  THREAD_HITS="$(grep -rnE 'std::thread::|rayon::|web_sys::Worker|SharedArrayBuffer|wgpu::' "$REPO_ROOT/crates" --include="*.rs" \
    --exclude-dir=target --exclude-dir=.worktrees | grep -vE '^[^:]+:[0-9]+:[[:space:]]*//' || true)"
  if [[ -n "$THREAD_HITS" ]]; then
    fail_hard "Forbidden threading or GPU references found in crates/:"
    printf '%s\n' "$THREAD_HITS" | sed 's/^/       /'
  else
    pass "Zero std::thread / rayon:: / Worker / SharedArrayBuffer / wgpu:: references in crates/"
  fi

  RNG_HITS="$(grep -rnE 'rand::thread_rng|thread_rng\(\)|from_entropy\(\)' "$REPO_ROOT/crates" --include="*.rs" \
    --exclude-dir=target --exclude-dir=.worktrees \
    | grep -E '/src/' \
    | grep -vE '^[^:]+:[0-9]+:[[:space:]]*//' \
    | grep -vE '#\[cfg\(test\)\]' || true)"
  if [[ -n "$RNG_HITS" ]]; then
    warn_legacy "Non-deterministic RNG (thread_rng/from_entropy) found in src/:"
    printf '%s\n' "$RNG_HITS" | head -n 20 | sed 's/^/       /'
  else
    pass "Zero thread_rng/from_entropy calls in crates/*/src/"
  fi
fi

# 3. wasm32-unknown-unknown cross-compilation check
WASM_CHECK_CRATE="${TARGET_CRATE:-egraph-wasm}"
if cargo metadata --no-deps --format-version 1 2>/dev/null | python3 -c "import sys,json; pkgs={p['name'] for p in json.load(sys.stdin)['packages']}; sys.exit(0 if '$WASM_CHECK_CRATE' in pkgs else 1)"; then
  if cargo check -p "$WASM_CHECK_CRATE" --target wasm32-unknown-unknown >/dev/null 2>&1; then
    pass "$WASM_CHECK_CRATE compiles cleanly for wasm32-unknown-unknown"
  else
    fail_hard "$WASM_CHECK_CRATE failed cargo check --target wasm32-unknown-unknown"
    cargo check -p "$WASM_CHECK_CRATE" --target wasm32-unknown-unknown || true
  fi
fi

# 4. WASM gzip size check (budget: 600 KiB = 614400 bytes)
WASM_FILE=""
for p in \
  "$REPO_ROOT/crates/wasm/pkg/egraph_wasm_bg.wasm" \
  "$REPO_ROOT/crates/wasm/dist/web/egraph_wasm_bg.wasm" \
  "$REPO_ROOT/target/wasm32-unknown-unknown/release/egraph_wasm.wasm"; do
  if [[ -f "$p" ]]; then
    WASM_FILE="$p"
    break
  fi
done

if [[ -n "$WASM_FILE" ]]; then
  RAW_BYTES="$(wc -c < "$WASM_FILE" | tr -d ' ')"
  GZIP_BYTES="$(gzip -9 -c "$WASM_FILE" | wc -c | tr -d ' ')"
  GZIP_KIB="$(python3 -c "print(f'{$GZIP_BYTES/1024:.1f}')")"
  if [[ "$GZIP_BYTES" -le 614400 ]]; then
    pass "WASM size OK: $WASM_FILE (raw=${RAW_BYTES} B, gzip=${GZIP_KIB} KiB <= 600.0 KiB)"
  else
    warn_legacy "WASM gzip size exceeds 600 KiB budget: $WASM_FILE (${GZIP_KIB} KiB)"
  fi
elif [[ "$CHECK_WASM_SIZE" -eq 1 ]]; then
  fail_hard "--check-wasm-size requested, but no built .wasm artifact found"
else
  echo "[INFO] No built .wasm artifact present yet; skipping gzip size check"
fi

# 5. Cross-target packaging metadata check
PKG_ISSUES="$(python3 - "$REPO_ROOT" <<'PY'
import pathlib, json, re, sys
root = pathlib.Path(sys.argv[1])
issues = []

py_toml = root / "crates" / "python" / "Cargo.toml"
py_proj = root / "crates" / "python" / "pyproject.toml"
wasm_toml = root / "crates" / "wasm" / "Cargo.toml"
wasm_pkg = root / "crates" / "wasm" / "package.json"

if py_toml.exists() and py_proj.exists():
    t_txt = py_toml.read_text()
    p_txt = py_proj.read_text()
    if "abi3-py37" in t_txt and '>=3.6' in p_txt:
        issues.append("Python abi3-py37 conflicts with requires-python >=3.6")
    if '"numpy"' not in p_txt and "'numpy'" not in p_txt and "numpy>=" not in p_txt:
        issues.append("crates/python/pyproject.toml omits runtime numpy dependency")

if wasm_toml.exists() and wasm_pkg.exists():
    wt = wasm_toml.read_text()
    wp = json.loads(wasm_pkg.read_text())
    m = re.search(r'^version\s*=\s*"([^"]+)"', wt, re.M)
    if m and m.group(1) != wp.get("version"):
        issues.append(f"WASM Cargo.toml version ({m.group(1)}) != package.json version ({wp.get('version')})")

for i in issues:
    print(i)
PY
)"

if [[ -n "$PKG_ISSUES" ]]; then
  while IFS= read -r line; do
    warn_legacy "Packaging metadata drift: $line"
  done <<< "$PKG_ISSUES"
else
  pass "Cross-target packaging metadata is aligned"
fi

echo "=== Invariant Summary (mode=$MODE): hard_fails=$HARD_FAILS legacy_warns=$LEGACY_WARNS ==="
if [[ "$HARD_FAILS" -gt 0 ]]; then
  exit 1
fi
exit 0
