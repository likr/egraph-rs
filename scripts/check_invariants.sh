#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/check_invariants.sh [--mode report|strict] [--crate <name>] [--check-wasm-size]

Verify single-threaded pure-Rust, WASM cross-compilation, optional .wasm size reporting,
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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$REPO_ROOT/scripts/verify_helpers.py" ]]; then
  VERIFY_HELPERS="$REPO_ROOT/scripts/verify_helpers.py"
else
  VERIFY_HELPERS="$SCRIPT_DIR/verify_helpers.py"
fi
export RUST_FONTCONFIG_DLOPEN=1

HARD_FAILS=0

pass() { echo "[PASS] $*"; }
fail_hard() {
  HARD_FAILS=$((HARD_FAILS + 1))
  echo "[FAIL] $*"
}

cd "$REPO_ROOT"

if [[ ! -f "$REPO_ROOT/Cargo.lock" ]]; then
  cargo generate-lockfile >/dev/null 2>&1 || cargo metadata --format-version 1 >/dev/null 2>&1 || true
fi

WORKSPACE_PKGS="$(python3 "$VERIFY_HELPERS" list-packages || true)"

has_workspace_pkg() {
  printf '%s\n' "$WORKSPACE_PKGS" | grep -qx "$1"
}

FORBIDDEN_RE='(^|[^[:alnum:]_-])(rayon|linfa|linfa-nn|linfa-clustering|getrandom|wgpu)([^[:alnum:]_-]|$)'

check_cargo_tree_crate() {
  local label="$1"
  local policy="$2"
  shift 2

  local tree_out tree_err err_file rc=0 forbidden
  err_file="$(mktemp)"
  if tree_out="$(cargo tree --quiet --edges normal -e features "$@" 2>"$err_file")"; then
    rc=0
  else
    rc=$?
  fi
  tree_err="$(cat "$err_file" 2>/dev/null || true)"
  rm -f "$err_file"

  if [[ "$rc" -ne 0 ]]; then
    fail_hard "$label cargo tree failed (exit $rc):"
    if [[ -n "$tree_err" ]]; then
      printf '%s\n' "$tree_err" | sed 's/^/       /'
    fi
    return 0
  fi

  forbidden="$(printf '%s\n' "$tree_out" | grep -E "$FORBIDDEN_RE" || true)"
  if [[ -n "$forbidden" ]]; then
    fail_hard "$label dependency tree pulls forbidden crates:"
    printf '%s\n' "$forbidden" | sed 's/^/       /'
  else
    pass "$label dependency tree is free of rayon/linfa/getrandom/wgpu"
  fi
}

# 1. Dependency tree gate (rayon / linfa / getrandom / wgpu)
# All workspace crates (including `egraph-python` and `egraph-wasm` default/minimal)
# ALWAYS fail_hard on rayon/linfa/getrandom/wgpu.
if [[ -n "$TARGET_CRATE" ]]; then
  case "$TARGET_CRATE" in
    egraph-core|egraph-ffi|egraph-linalg|egraph-linalg-mds|egraph-linalg-amg|egraph-neighborhood|egraph-clustering|egraph-cli|egraph-spatial-trees|egraph-spatial-fft|egraph-spatial-crossings|egraph-metrics-geometry|egraph-edge-routing|egraph-layout-embedding|egraph-layout-neighbor|egraph-layout-sgd|egraph-python)
      check_cargo_tree_crate "$TARGET_CRATE" "hard" -p "$TARGET_CRATE"
      ;;
    egraph-metrics|egraph-metrics-cluster)
      check_cargo_tree_crate "$TARGET_CRATE (default)" "hard" -p "$TARGET_CRATE"
      check_cargo_tree_crate "$TARGET_CRATE (--all-features)" "hard" -p "$TARGET_CRATE" --all-features
      check_cargo_tree_crate "$TARGET_CRATE (--no-default-features)" "hard" -p "$TARGET_CRATE" --no-default-features
      ;;
    egraph-dataset)
      check_cargo_tree_crate "egraph-dataset (default)" "hard" -p egraph-dataset
      check_cargo_tree_crate "egraph-dataset (--all-features)" "hard" -p egraph-dataset --all-features
      check_cargo_tree_crate "egraph-dataset (--no-default-features)" "hard" -p egraph-dataset --no-default-features
      check_cargo_tree_crate "egraph-dataset (--features embedded)" "hard" -p egraph-dataset --features embedded
      ;;
    egraph)
      check_cargo_tree_crate "egraph (default)" "hard" -p egraph
      check_cargo_tree_crate "egraph (--all-features)" "hard" -p egraph --all-features
      check_cargo_tree_crate "egraph (--no-default-features)" "hard" -p egraph --no-default-features
      check_cargo_tree_crate "egraph (--no-default-features --features dataset)" "hard" -p egraph --no-default-features --features dataset
      check_cargo_tree_crate "egraph (--no-default-features --features edge-routing)" "hard" -p egraph --no-default-features --features edge-routing
      ;;
    egraph-layout-constraints|egraph-layout-multilevel|egraph-layout-stress|egraph-layout-force|egraph-layout-refine|egraph-layout-composite)
      check_cargo_tree_crate "$TARGET_CRATE (default)" "hard" -p "$TARGET_CRATE"
      check_cargo_tree_crate "$TARGET_CRATE (--all-features)" "hard" -p "$TARGET_CRATE" --all-features
      check_cargo_tree_crate "$TARGET_CRATE (--no-default-features)" "hard" -p "$TARGET_CRATE" --no-default-features
      check_cargo_tree_crate "$TARGET_CRATE (--no-default-features --features baselines)" "hard" -p "$TARGET_CRATE" --no-default-features --features baselines
      ;;
    egraph-wasm)
      check_cargo_tree_crate "egraph-wasm" "hard" -p egraph-wasm
      ;;
    *)
      check_cargo_tree_crate "$TARGET_CRATE" "hard" -p "$TARGET_CRATE"
      ;;
  esac
else
  for strict_crate in egraph-core egraph-ffi egraph-dataset egraph-spatial-trees egraph-spatial-fft egraph-spatial-crossings egraph-edge-routing egraph-metrics egraph-metrics-geometry egraph-metrics-cluster egraph-linalg egraph-linalg-mds egraph-linalg-amg egraph-neighborhood egraph-clustering egraph-layout-constraints egraph-layout-stress egraph-layout-force egraph-layout-neighbor egraph-layout-sgd egraph-layout-embedding egraph-layout-multilevel egraph-layout-refine egraph-layout-composite egraph egraph-cli egraph-python; do
    if has_workspace_pkg "$strict_crate"; then
      check_cargo_tree_crate "$strict_crate" "hard" -p "$strict_crate"
    fi
  done
  for crate in egraph-dataset egraph-metrics egraph-metrics-cluster egraph egraph-edge-routing egraph-layout-constraints egraph-layout-stress egraph-layout-force egraph-layout-sgd egraph-layout-embedding egraph-layout-multilevel egraph-layout-refine egraph-layout-composite; do
    if has_workspace_pkg "$crate"; then
      check_cargo_tree_crate "$crate (--all-features)" "hard" -p "$crate" --all-features
    fi
  done
  for crate in egraph-metrics egraph-metrics-cluster egraph-dataset egraph egraph-edge-routing egraph-layout-constraints egraph-layout-stress egraph-layout-force egraph-layout-sgd egraph-layout-embedding egraph-layout-multilevel egraph-layout-refine egraph-layout-composite; do
    if has_workspace_pkg "$crate"; then
      check_cargo_tree_crate "$crate (--no-default-features)" "hard" -p "$crate" --no-default-features
    fi
  done
  if has_workspace_pkg "egraph-dataset"; then
    check_cargo_tree_crate "egraph-dataset (--features embedded)" "hard" -p egraph-dataset --features embedded
  fi
  if has_workspace_pkg "egraph"; then
    check_cargo_tree_crate "egraph (--no-default-features --features dataset)" "hard" -p egraph --no-default-features --features dataset
    check_cargo_tree_crate "egraph (--no-default-features --features edge-routing)" "hard" -p egraph --no-default-features --features edge-routing
  fi
  for crate in egraph-layout-constraints egraph-layout-stress egraph-layout-force egraph-layout-multilevel egraph-layout-refine egraph-layout-composite; do
    if has_workspace_pkg "$crate"; then
      check_cargo_tree_crate "$crate (--no-default-features --features baselines)" "hard" -p "$crate" --no-default-features --features baselines
    fi
  done
  if has_workspace_pkg "egraph-wasm"; then
    check_cargo_tree_crate "egraph-wasm" "hard" -p egraph-wasm
  fi
fi

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
    fail_hard "Non-deterministic RNG (thread_rng/from_entropy) found in src/:"
    printf '%s\n' "$RNG_HITS" | head -n 20 | sed 's/^/       /'
  else
    pass "Zero thread_rng/from_entropy calls in crates/*/src/"
  fi
fi

# 3. wasm32-unknown-unknown cross-compilation check
check_wasm32_variant() {
  local crate="$1"
  local label="$2"
  shift 2
  if cargo check -p "$crate" "$@" --target wasm32-unknown-unknown >/dev/null 2>&1; then
    pass "$crate$label compiles cleanly for wasm32-unknown-unknown"
  else
    fail_hard "$crate$label failed cargo check --target wasm32-unknown-unknown"
    cargo check -p "$crate" "$@" --target wasm32-unknown-unknown || true
  fi
}

check_wasm32_target() {
  local crate="$1"
  if ! has_workspace_pkg "$crate"; then
    return 0
  fi
  check_wasm32_variant "$crate" ""
  case "$crate" in
    egraph|egraph-dataset|egraph-metrics-cluster|egraph-edge-routing|egraph-layout-constraints|egraph-layout-stress|egraph-layout-force|egraph-layout-sgd|egraph-layout-embedding|egraph-layout-multilevel|egraph-layout-refine|egraph-layout-composite)
      check_wasm32_variant "$crate" " (--no-default-features)" --no-default-features
      ;;
  esac
  case "$crate" in
    egraph|egraph-dataset|egraph-metrics|egraph-metrics-cluster|egraph-edge-routing|egraph-layout-constraints|egraph-layout-stress|egraph-layout-force|egraph-layout-sgd|egraph-layout-embedding|egraph-layout-multilevel|egraph-layout-refine|egraph-layout-composite)
      check_wasm32_variant "$crate" " (--all-features)" --all-features
      ;;
  esac
  if [[ "$crate" == "egraph-dataset" ]]; then
    check_wasm32_variant "$crate" " (--features embedded)" --features embedded
  fi
  if [[ "$crate" == "egraph" ]]; then
    check_wasm32_variant "$crate" " (--no-default-features --features dataset)" --no-default-features --features dataset
    check_wasm32_variant "$crate" " (--no-default-features --features edge-routing)" --no-default-features --features edge-routing
  fi
  case "$crate" in
    egraph-layout-constraints|egraph-layout-stress|egraph-layout-force|egraph-layout-multilevel|egraph-layout-refine|egraph-layout-composite)
      check_wasm32_variant "$crate" " (--no-default-features --features baselines)" --no-default-features --features baselines
      ;;
  esac
}

check_wasm32_batch() {
  local label="$1"
  shift
  local flags=() crates=() arg in_crates=0
  for arg in "$@"; do
    if [[ "$arg" == "--" ]]; then
      in_crates=1
      continue
    fi
    if [[ "$in_crates" -eq 1 ]]; then
      if has_workspace_pkg "$arg"; then
        crates+=(-p "$arg")
      fi
    else
      flags+=("$arg")
    fi
  done
  if [[ "${#crates[@]}" -eq 0 ]]; then
    return 0
  fi
  if cargo check "${crates[@]}" ${flags[@]+"${flags[@]}"} --target wasm32-unknown-unknown >/dev/null 2>&1; then
    pass "wasm32 batch ($label: ${#crates[@]} crates/flags) compiles cleanly"
  else
    fail_hard "wasm32 batch ($label) failed cargo check --target wasm32-unknown-unknown"
    cargo check "${crates[@]}" ${flags[@]+"${flags[@]}"} --target wasm32-unknown-unknown || true
  fi
}

if [[ -n "$TARGET_CRATE" ]]; then
  case "$TARGET_CRATE" in
    egraph-python|egraph-cli)
      echo "[INFO] Skipping wasm32-unknown-unknown check for host-only crate '$TARGET_CRATE'"
      ;;
    *)
      check_wasm32_target "$TARGET_CRATE"
      ;;
  esac
else
  check_wasm32_batch "default" -- \
    egraph-core \
    egraph-ffi \
    egraph-dataset \
    egraph-spatial-trees \
    egraph-spatial-fft \
    egraph-spatial-crossings \
    egraph-edge-routing \
    egraph-metrics \
    egraph-metrics-geometry \
    egraph-metrics-cluster \
    egraph-linalg \
    egraph-linalg-mds \
    egraph-linalg-amg \
    egraph-neighborhood \
    egraph-clustering \
    egraph-layout-constraints \
    egraph-layout-stress \
    egraph-layout-force \
    egraph-layout-neighbor \
    egraph-layout-sgd \
    egraph-layout-embedding \
    egraph-layout-multilevel \
    egraph-layout-refine \
    egraph-layout-composite \
    egraph \
    egraph-wasm
  check_wasm32_batch "--all-features" --all-features -- \
    egraph \
    egraph-dataset \
    egraph-metrics \
    egraph-metrics-cluster \
    egraph-edge-routing \
    egraph-layout-constraints \
    egraph-layout-stress \
    egraph-layout-force \
    egraph-layout-sgd \
    egraph-layout-embedding \
    egraph-layout-multilevel \
    egraph-layout-refine \
    egraph-layout-composite
  check_wasm32_batch "--no-default-features" --no-default-features -- \
    egraph \
    egraph-dataset \
    egraph-metrics-cluster \
    egraph-edge-routing \
    egraph-layout-constraints \
    egraph-layout-stress \
    egraph-layout-force \
    egraph-layout-sgd \
    egraph-layout-embedding \
    egraph-layout-multilevel \
    egraph-layout-refine \
    egraph-layout-composite
  check_wasm32_batch "--no-default-features --features baselines" --no-default-features --features baselines -- \
    egraph-layout-constraints \
    egraph-layout-stress \
    egraph-layout-force \
    egraph-layout-multilevel \
    egraph-layout-refine \
    egraph-layout-composite
  check_wasm32_batch "egraph-dataset --features embedded" --features embedded -- egraph-dataset
  check_wasm32_batch "egraph --no-default-features --features dataset" --no-default-features --features dataset -- egraph
  check_wasm32_batch "egraph --no-default-features --features edge-routing" --no-default-features --features edge-routing -- egraph
fi

# 4. WASM artifact size report (informational; no gzip upper-bound enforced)
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
  pass "WASM artifact size: $WASM_FILE (raw=${RAW_BYTES} B, gzip=${GZIP_KIB} KiB)"
elif [[ "$CHECK_WASM_SIZE" -eq 1 ]]; then
  fail_hard "--check-wasm-size requested, but no built .wasm artifact found"
else
  echo "[INFO] No built .wasm artifact present yet; skipping size report"
fi

# 5. Cross-target packaging metadata check
if [[ "$MODE" == "strict" ]]; then
  PKG_ISSUES="$(python3 "$VERIFY_HELPERS" check-packaging "$REPO_ROOT" --strict-metadata)"
else
  PKG_ISSUES="$(python3 "$VERIFY_HELPERS" check-packaging "$REPO_ROOT")"
fi

if [[ -n "$PKG_ISSUES" ]]; then
  while IFS= read -r line; do
    fail_hard "Packaging metadata drift: $line"
  done <<< "$PKG_ISSUES"
else
  pass "Cross-target packaging metadata is aligned"
fi

echo "=== Invariant Summary (mode=$MODE): hard_fails=$HARD_FAILS ==="
if [[ "$HARD_FAILS" -gt 0 ]]; then
  exit 1
fi
exit 0
