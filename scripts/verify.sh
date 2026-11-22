#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/verify.sh [options]

Tiered, worktree-aware verification runner for Native Rust, Python, and WASM.
Automatically configures single-thread guards, sccache, CARGO_BUILD_JOBS from
EGRAPH_PARALLEL_WORKERS, and worktree-local .venv.

Modes (composable):
  --fast                   Tier 0: fmt check + clippy/nextest on modified or --crate targets + thread scan
  --rust                   Full workspace fmt check, clippy (excl. legacy petgraph-linalg-kernel unless --strict), release nextest
  --python                 Build crates/python into worktree .venv via maturin develop --release & run tests/mypy
  --wasm                   Check wasm32 + run wasm-pack test --node crates/wasm
  --invariants             Run scripts/check_invariants.sh
  --parity                 Run scripts/parity_harness.py
  --docs                   Build Sphinx HTML + doctest, TypeDoc, and js/examples Vite bundle
  --all                    Run --rust --python --wasm --invariants --parity

Options:
  --crate <name>           Target crate (repeatable or comma-separated) for --fast / --invariants
  --test <name>            Scoped test module/name for --python or --wasm
  --release                Build Python bindings with maturin develop --release (default)
  --dev-py                 Build Python bindings in debug mode (maturin develop without --release)
  --debug-rust             Run --rust workspace tests in debug mode instead of --release
  --build-wasm             Also run wasm-pack build --target web --profile wasm-release + assemble_pkg.mjs + wasm-opt -Oz + size check
  --strict                 Strict mode for invariants & workspace-wide clippy -D warnings
  --json-summary <path>    Write machine-readable JSON summary to <path>
EOF
}

RUN_FAST=0
RUN_RUST=0
RUN_PYTHON=0
RUN_WASM=0
RUN_INVARIANTS=0
RUN_PARITY=0
RUN_DOCS=0
STRICT=0
RELEASE_PY=1
DEBUG_RUST=0
BUILD_WASM=0
TEST_FILTER=""
JSON_SUMMARY=""
CRATES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fast) RUN_FAST=1; shift ;;
    --rust) RUN_RUST=1; shift ;;
    --python) RUN_PYTHON=1; shift ;;
    --wasm) RUN_WASM=1; shift ;;
    --invariants) RUN_INVARIANTS=1; shift ;;
    --parity) RUN_PARITY=1; shift ;;
    --docs) RUN_DOCS=1; shift ;;
    --all) RUN_RUST=1; RUN_PYTHON=1; RUN_WASM=1; RUN_INVARIANTS=1; RUN_PARITY=1; shift ;;
    --crate)
      IFS=',' read -ra _parts <<< "${2:-}"
      for p in "${_parts[@]}"; do
        [[ -n "$p" ]] && CRATES+=("$p")
      done
      shift 2
      ;;
    --test) TEST_FILTER="${2:-}"; shift 2 ;;
    --release) RELEASE_PY=1; shift ;;
    --dev-py) RELEASE_PY=0; shift ;;
    --debug-rust) DEBUG_RUST=1; shift ;;
    --build-wasm) BUILD_WASM=1; shift ;;
    --strict) STRICT=1; shift ;;
    --json-summary) JSON_SUMMARY="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ $((RUN_FAST + RUN_RUST + RUN_PYTHON + RUN_WASM + RUN_INVARIANTS + RUN_PARITY + RUN_DOCS)) -eq 0 ]]; then
  RUN_FAST=1
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

helper_path() {
  local name="$1"
  if [[ -x "$REPO_ROOT/scripts/$name" || -f "$REPO_ROOT/scripts/$name" ]]; then
    echo "$REPO_ROOT/scripts/$name"
  else
    echo "$SCRIPT_DIR/$name"
  fi
}

# Single-thread & build environment hygiene
export RUST_FONTCONFIG_DLOPEN=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if command -v sccache >/dev/null 2>&1 && [[ -z "${RUSTC_WRAPPER:-}" ]]; then
  export RUSTC_WRAPPER=sccache
  export CARGO_INCREMENTAL="${CARGO_INCREMENTAL:-0}"
fi

PARALLEL_WORKERS="${EGRAPH_PARALLEL_WORKERS:-1}"
if [[ -z "${CARGO_BUILD_JOBS:-}" ]]; then
  if [[ "$PARALLEL_WORKERS" -le 1 ]]; then
    export CARGO_BUILD_JOBS=8
  elif [[ "$PARALLEL_WORKERS" -eq 2 ]]; then
    export CARGO_BUILD_JOBS=5
  elif [[ "$PARALLEL_WORKERS" -eq 3 ]]; then
    export CARGO_BUILD_JOBS=4
  else
    export CARGO_BUILD_JOBS=2
  fi
fi

export VIRTUAL_ENV="$REPO_ROOT/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"

cd "$REPO_ROOT"

if command -v cargo-nextest >/dev/null 2>&1; then
  TEST_RUNNER="nextest"
else
  TEST_RUNNER="cargo-test"
fi

run_cargo_test_args() {
  if [[ "$TEST_RUNNER" == "nextest" ]]; then
    cargo nextest run --show-progress=none --no-input-handler --status-level=pass --final-status-level=fail "$@"
  else
    cargo test "$@"
  fi
}

OVERALL_START="$(python3 -c 'import time; print(f"{time.time():.3f}")')"
STEP_RECORDS_FILE="$(mktemp "${TMPDIR:-/tmp}/egraph_verify_steps.XXXXXX")"
DOCS_TMP_DIR=""
FINAL_STATUS="pass"
REACHED_END=0

write_json_summary() {
  local exit_code=$?
  if [[ "$exit_code" -eq 0 && ( "$FINAL_STATUS" != "pass" || "$REACHED_END" -ne 1 ) ]]; then
    exit_code=1
    FINAL_STATUS="fail"
  elif [[ "$exit_code" -ne 0 ]]; then
    FINAL_STATUS="fail"
  fi
  if [[ -n "$JSON_SUMMARY" ]]; then
    mkdir -p "$(dirname "$JSON_SUMMARY")"
    local sccache_info=""
    if command -v sccache >/dev/null 2>&1; then
      sccache_info="$(sccache --show-stats 2>/dev/null | head -n 15 || true)"
    fi
    python3 - "$JSON_SUMMARY" "$FINAL_STATUS" "$REPO_ROOT" "$PARALLEL_WORKERS" "$CARGO_BUILD_JOBS" "${RUSTC_WRAPPER:-none}" "$OVERALL_START" "$STEP_RECORDS_FILE" "$sccache_info" <<'PY'
import json, pathlib, subprocess, sys, time
out_path = pathlib.Path(sys.argv[1])
status = sys.argv[2]
repo_root = sys.argv[3]
workers = int(sys.argv[4])
jobs = int(sys.argv[5])
wrapper = sys.argv[6]
start_ts = float(sys.argv[7])
records_file = pathlib.Path(sys.argv[8])
sccache_info = sys.argv[9]

steps = []
if records_file.exists():
    for line in records_file.read_text().splitlines():
        if line.strip():
            steps.append(json.loads(line))

branch = subprocess.check_output(["git", "-C", repo_root, "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
sha = subprocess.check_output(["git", "-C", repo_root, "rev-parse", "--short", "HEAD"], text=True).strip()

payload = {
    "status": status,
    "worktree": repo_root,
    "branch": branch,
    "head_sha": sha,
    "parallel_workers": workers,
    "cargo_build_jobs": jobs,
    "rustc_wrapper": wrapper,
    "total_elapsed_sec": round(time.time() - start_ts, 3),
    "steps": steps,
    "sccache_stats": sccache_info,
}
out_path.write_text(json.dumps(payload, indent=2) + "\n")
PY
  fi
  rm -f "$STEP_RECORDS_FILE"
  if [[ -n "${DOCS_TMP_DIR:-}" ]]; then
    rm -rf "$DOCS_TMP_DIR"
  fi
  exit "$exit_code"
}
trap write_json_summary EXIT

run_step() {
  local name="$1"
  shift
  local t0 t1 elapsed rc=0
  t0="$(python3 -c 'import time; print(f"{time.time():.3f}")')"
  echo "=== [verify] $name ==="
  if "$@"; then
    rc=0
  else
    rc=$?
    FINAL_STATUS="fail"
  fi
  t1="$(python3 -c 'import time; print(f"{time.time():.3f}")')"
  elapsed="$(python3 -c "print(round($t1 - $t0, 3))")"
  python3 - "$STEP_RECORDS_FILE" "$name" "$rc" "$elapsed" <<'PY'
import json, sys
with open(sys.argv[1], "a") as f:
    f.write(json.dumps({"name": sys.argv[2], "status": "pass" if int(sys.argv[3]) == 0 else "fail", "elapsed_sec": float(sys.argv[4])}) + "\n")
PY
  if [[ "$rc" -ne 0 ]]; then
    return "$rc"
  fi
}

detect_changed_crates() {
  python3 - "$REPO_ROOT" <<'PY'
import pathlib, re, subprocess, sys

root = pathlib.Path(sys.argv[1])
changed = set()
for cmd in (["git", "-C", str(root), "diff", "--name-only", "HEAD"],
            ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard"]):
    out = subprocess.check_output(cmd, text=True).splitlines()
    changed.update(out)

pkgs = []
for toml in root.glob("crates/**/Cargo.toml"):
    crate_dir = toml.parent.relative_to(root).as_posix() + "/"
    if any(f.startswith(crate_dir) for f in changed):
        txt = toml.read_text()
        m = re.search(r'^name\s*=\s*"([^"]+)"', txt, re.M)
        if m and m.group(1) not in ("egraph-python", "egraph-wasm"):
            pkgs.append(m.group(1))

for p in sorted(set(pkgs)):
    print(p)
PY
}

ensure_venv() {
  local setup_bin
  setup_bin="$(helper_path setup_venv.sh)"
  if [[ ! -x "$VIRTUAL_ENV/bin/python" ]] || ! "$VIRTUAL_ENV/bin/python" -c "import maturin, pytest, numpy, scipy, networkx" >/dev/null 2>&1; then
    "$setup_bin" "$VIRTUAL_ENV"
  fi
}

# --- Tier 0: --fast ---
if [[ "$RUN_FAST" -eq 1 ]]; then
  run_step "rust_fmt_check" cargo fmt --all -- --check

  FAST_CRATES=()
  if [[ "${#CRATES[@]}" -gt 0 ]]; then
    FAST_CRATES=(${CRATES[@]+"${CRATES[@]}"})
  else
    while IFS= read -r line; do
      [[ -n "$line" ]] && FAST_CRATES+=("$line")
    done < <(detect_changed_crates)
  fi

  if [[ "${#FAST_CRATES[@]}" -eq 0 ]]; then
    echo "[verify] No changed library crates detected; running baseline smoke crate check."
    for fallback in egraph-core petgraph-algorithm-connected-components; do
      if cargo metadata --no-deps --format-version 1 2>/dev/null | python3 -c "import sys,json; pkgs={p['name'] for p in json.load(sys.stdin)['packages']}; sys.exit(0 if '$fallback' in pkgs else 1)"; then
        FAST_CRATES=("$fallback")
        break
      fi
    done
  fi

  for c in ${FAST_CRATES[@]+"${FAST_CRATES[@]}"}; do
    case "$c" in
      egraph-python)
        run_step "clippy_$c" cargo clippy --no-deps --all-targets -p egraph-python -- -D warnings
        echo "[verify] Skipping standalone Rust test run for egraph-python (cdylib extension-module); run ./scripts/verify.sh --python"
        ;;
      egraph-wasm)
        run_step "wasm32_check_$c" cargo check -p egraph-wasm --target wasm32-unknown-unknown
        run_step "clippy_$c" cargo clippy --no-deps --all-targets -p egraph-wasm -- -D warnings
        echo "[verify] Skipping standalone host test run for egraph-wasm (cdylib); run ./scripts/verify.sh --wasm"
        ;;
      *)
        if [[ "$c" != "petgraph-linalg-kernel" || "$STRICT" -eq 1 ]]; then
          run_step "clippy_$c" cargo clippy --no-deps --all-targets -p "$c" -- -D warnings
        fi
        run_step "test_$c" run_cargo_test_args -p "$c"
        ;;
    esac
  done

  run_step "thread_scan" bash -c "
    HITS=\$(grep -rnE 'std::thread::|rayon::|web_sys::Worker|SharedArrayBuffer|wgpu::' '$REPO_ROOT/crates' --include='*.rs' --exclude-dir=target --exclude-dir=.worktrees | grep -vE '^[^:]+:[0-9]+:[[:space:]]*//' || true)
    if [[ -n \"\$HITS\" ]]; then echo \"\$HITS\"; exit 1; fi
  "
fi

# --- Full Rust workspace ---
if [[ "$RUN_RUST" -eq 1 ]]; then
  run_step "rust_fmt_check" cargo fmt --all -- --check
  if [[ "$STRICT" -eq 1 ]]; then
    run_step "clippy_workspace_strict" cargo clippy --workspace --all-targets --all-features -- -D warnings
  else
    run_step "clippy_workspace_non_legacy" bash -c '
      set -euo pipefail
      PKGS=$(python3 -c "import json,subprocess; d=json.loads(subprocess.check_output([\"cargo\",\"metadata\",\"--no-deps\",\"--format-version\",\"1\"],text=True)); print(\" \".join(\"-p \"+p[\"name\"] for p in d[\"packages\"] if p[\"name\"]!=\"petgraph-linalg-kernel\"))")
      eval "cargo clippy --no-deps --all-targets $PKGS -- -D warnings"
    '
  fi
  if cargo metadata --no-deps --format-version 1 2>/dev/null | python3 -c "import sys,json; pkgs={p['name'] for p in json.load(sys.stdin)['packages']}; sys.exit(0 if 'egraph-layout' in pkgs else 1)"; then
    run_step "clippy_layout_no_default_features" cargo clippy --no-deps --all-targets --no-default-features -p egraph-layout -- -D warnings
  fi
  if cargo metadata --no-deps --format-version 1 2>/dev/null | python3 -c "import sys,json; pkgs={p['name'] for p in json.load(sys.stdin)['packages']}; sys.exit(0 if 'egraph' in pkgs else 1)"; then
    run_step "clippy_egraph_no_default_features" cargo clippy --no-deps --all-targets --no-default-features -p egraph -- -D warnings
  fi
  RUST_TEST_ARGS=()
  if [[ "$DEBUG_RUST" -eq 0 ]]; then
    RUST_TEST_ARGS+=(--release)
  fi
  run_step "rust_workspace_tests" run_cargo_test_args ${RUST_TEST_ARGS[@]+"${RUST_TEST_ARGS[@]}"} --workspace
  if cargo metadata --no-deps --format-version 1 2>/dev/null | python3 -c "import sys,json; pkgs={p['name'] for p in json.load(sys.stdin)['packages']}; sys.exit(0 if 'egraph-dataset' in pkgs else 1)"; then
    run_step "rust_dataset_all_features" run_cargo_test_args ${RUST_TEST_ARGS[@]+"${RUST_TEST_ARGS[@]}"} --all-features -p egraph-dataset
    run_step "rust_dataset_no_default_features" run_cargo_test_args ${RUST_TEST_ARGS[@]+"${RUST_TEST_ARGS[@]}"} --no-default-features -p egraph-dataset
  fi
  if cargo metadata --no-deps --format-version 1 2>/dev/null | python3 -c "import sys,json; pkgs={p['name'] for p in json.load(sys.stdin)['packages']}; sys.exit(0 if 'egraph-spatial' in pkgs else 1)"; then
    run_step "rust_spatial_all_features" run_cargo_test_args ${RUST_TEST_ARGS[@]+"${RUST_TEST_ARGS[@]}"} --all-features -p egraph-spatial
  fi
  if cargo metadata --no-deps --format-version 1 2>/dev/null | python3 -c "import sys,json; pkgs={p['name'] for p in json.load(sys.stdin)['packages']}; sys.exit(0 if 'egraph-layout' in pkgs else 1)"; then
    run_step "rust_layout_no_default_features" run_cargo_test_args ${RUST_TEST_ARGS[@]+"${RUST_TEST_ARGS[@]}"} --no-default-features -p egraph-layout
  fi
  if cargo metadata --no-deps --format-version 1 2>/dev/null | python3 -c "import sys,json; pkgs={p['name'] for p in json.load(sys.stdin)['packages']}; sys.exit(0 if 'egraph' in pkgs else 1)"; then
    run_step "rust_egraph_all_features" run_cargo_test_args ${RUST_TEST_ARGS[@]+"${RUST_TEST_ARGS[@]}"} --all-features -p egraph
    run_step "rust_egraph_no_default_features" run_cargo_test_args ${RUST_TEST_ARGS[@]+"${RUST_TEST_ARGS[@]}"} --no-default-features -p egraph
  fi
  if cargo metadata --no-deps --format-version 1 2>/dev/null | python3 -c "import sys,json; pkgs={p['name'] for p in json.load(sys.stdin)['packages']}; sys.exit(0 if 'egraph-cli' in pkgs else 1)"; then
    run_step "rust_cli_tests" run_cargo_test_args ${RUST_TEST_ARGS[@]+"${RUST_TEST_ARGS[@]}"} -p egraph-cli
  fi
fi

# --- Python ---
if [[ "$RUN_PYTHON" -eq 1 ]]; then
  run_step "python_venv" ensure_venv
  MATURIN_FLAGS=()
  if [[ "$RELEASE_PY" -eq 1 ]]; then
    MATURIN_FLAGS+=(--release)
  fi
  run_step "python_maturin_develop" bash -c "cd '$REPO_ROOT/crates/python' && '$VIRTUAL_ENV/bin/maturin' develop ${MATURIN_FLAGS[*]:-}"

  if [[ -n "$TEST_FILTER" ]]; then
    if [[ "$TEST_FILTER" == tests.* ]]; then
      PY_TARGET="$TEST_FILTER"
    elif [[ "$TEST_FILTER" == test_* || "$TEST_FILTER" == *.* ]]; then
      PY_TARGET="tests.$TEST_FILTER"
    else
      PY_TARGET="tests.test_$TEST_FILTER"
    fi
    run_step "python_test_scoped" bash -c "cd '$REPO_ROOT/crates/python' && '$VIRTUAL_ENV/bin/python' -m unittest '$PY_TARGET'"
  else
    run_step "python_unittest_all" bash -c "cd '$REPO_ROOT/crates/python' && '$VIRTUAL_ENV/bin/python' -m unittest discover -s tests"
  fi

  if [[ -f "$REPO_ROOT/crates/python/python/egraph/_core.pyi" ]]; then
    V6_PY_TESTS=()
    while IFS= read -r py_test; do
      [[ -n "$py_test" ]] && V6_PY_TESTS+=("$py_test")
    done < <(find "$REPO_ROOT/crates/python/tests" -maxdepth 1 -name 'test_v6_*.py' | sort)
    run_step "python_mypy" "$VIRTUAL_ENV/bin/mypy" --strict "$REPO_ROOT/crates/python/python/egraph" ${V6_PY_TESTS[@]+"${V6_PY_TESTS[@]}"}
    run_step "python_stubtest" "$VIRTUAL_ENV/bin/stubtest" egraph
  fi
fi

# --- WASM ---
if [[ "$RUN_WASM" -eq 1 ]]; then
  run_step "wasm32_check" cargo check -p egraph-wasm --target wasm32-unknown-unknown
  if [[ -n "$TEST_FILTER" ]]; then
    run_step "wasm_test_$TEST_FILTER" wasm-pack test --node "$REPO_ROOT/crates/wasm" --test "$TEST_FILTER"
  else
    run_step "wasm_test_all" wasm-pack test --node "$REPO_ROOT/crates/wasm"
    run_step "wasm_test_minimal" wasm-pack test --node "$REPO_ROOT/crates/wasm" --no-default-features --test v6_core --test v6_metrics --test v6_linalg_neighborhood --test v6_layout_clustering --test v6_m4 --test v6_m5
    run_step "wasm_test_legacy" wasm-pack test --node "$REPO_ROOT/crates/wasm" --features legacy
  fi

  if [[ "$BUILD_WASM" -eq 1 ]]; then
    run_step "wasm_build_release" bash -c "
      set -euo pipefail
      wasm-pack build --target web --no-opt --profile wasm-release -d pkg '$REPO_ROOT/crates/wasm'
      if [[ -f '$REPO_ROOT/crates/wasm/scripts/assemble_pkg.mjs' ]]; then
        node '$REPO_ROOT/crates/wasm/scripts/assemble_pkg.mjs'
      fi
      if command -v wasm-opt >/dev/null 2>&1 && [[ -f '$REPO_ROOT/crates/wasm/pkg/egraph_wasm_bg.wasm' ]]; then
        wasm-opt -Oz --enable-bulk-memory --enable-nontrapping-float-to-int \
          '$REPO_ROOT/crates/wasm/pkg/egraph_wasm_bg.wasm' \
          -o '$REPO_ROOT/crates/wasm/pkg/egraph_wasm_bg.wasm'
      fi
    "
    if [[ -f "$REPO_ROOT/crates/wasm/tests/node_esm_smoke.mjs" ]]; then
      run_step "wasm_node_esm_smoke" node "$REPO_ROOT/crates/wasm/tests/node_esm_smoke.mjs"
    fi
  fi
fi

# --- Invariants ---
if [[ "$RUN_INVARIANTS" -eq 1 ]]; then
  INV_BIN="$(helper_path check_invariants.sh)"
  INV_ARGS=(--mode "$([[ "$STRICT" -eq 1 ]] && echo strict || echo report)")
  if [[ "${#CRATES[@]}" -gt 0 ]]; then
    INV_ARGS+=(--crate "${CRATES[0]}")
  fi
  if [[ "$BUILD_WASM" -eq 1 ]]; then
    INV_ARGS+=(--check-wasm-size)
  fi
  run_step "invariants" "$INV_BIN" ${INV_ARGS[@]+"${INV_ARGS[@]}"}
fi

# --- Cross-target Parity ---
if [[ "$RUN_PARITY" -eq 1 ]]; then
  run_step "python_venv_for_parity" ensure_venv
  PARITY_BIN="$(helper_path parity_harness.py)"
  PARITY_ARGS=()
  if [[ -d "$REPO_ROOT/crates/wasm/pkg" ]]; then
    PARITY_ARGS+=(--wasm-pkg "$REPO_ROOT/crates/wasm/pkg")
  fi
  run_step "parity_harness" "$VIRTUAL_ENV/bin/python" "$PARITY_BIN" ${PARITY_ARGS[@]+"${PARITY_ARGS[@]}"}
fi

# --- Docs & Examples ---
if [[ "$RUN_DOCS" -eq 1 ]]; then
  run_step "python_venv_for_docs" ensure_venv
  if ! "$VIRTUAL_ENV/bin/python" -c "import egraph" >/dev/null 2>&1; then
    run_step "python_maturin_develop_for_docs" bash -c "cd '$REPO_ROOT/crates/python' && '$VIRTUAL_ENV/bin/maturin' develop --release"
  fi
  if [[ ! -f "$REPO_ROOT/crates/wasm/pkg/index.d.ts" ]]; then
    run_step "wasm_build_release_for_docs" bash -c "
      set -euo pipefail
      wasm-pack build --target web --no-opt --profile wasm-release -d pkg '$REPO_ROOT/crates/wasm'
      if [[ -f '$REPO_ROOT/crates/wasm/scripts/assemble_pkg.mjs' ]]; then
        node '$REPO_ROOT/crates/wasm/scripts/assemble_pkg.mjs'
      fi
      if command -v wasm-opt >/dev/null 2>&1 && [[ -f '$REPO_ROOT/crates/wasm/pkg/egraph_wasm_bg.wasm' ]]; then
        wasm-opt -Oz --enable-bulk-memory --enable-nontrapping-float-to-int \
          '$REPO_ROOT/crates/wasm/pkg/egraph_wasm_bg.wasm' \
          -o '$REPO_ROOT/crates/wasm/pkg/egraph_wasm_bg.wasm'
      fi
    "
  fi
  DOCS_TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/egraph_verify_docs.XXXXXX")"
  run_step "docs_sphinx_html" "$VIRTUAL_ENV/bin/python" -m sphinx -b html "$REPO_ROOT/crates/python/docs/source" "$DOCS_TMP_DIR/sphinx_html"
  run_step "docs_sphinx_doctest" "$VIRTUAL_ENV/bin/python" -m sphinx -b doctest "$REPO_ROOT/crates/python/docs/source" "$DOCS_TMP_DIR/sphinx_doctest"
  run_step "docs_typedoc" bash -c "
    set -euo pipefail
    if [[ ! -x '$REPO_ROOT/crates/wasm/node_modules/.bin/typedoc' ]]; then
      npm ci --prefix '$REPO_ROOT/crates/wasm'
    fi
    npx --prefix '$REPO_ROOT/crates/wasm' typedoc --options '$REPO_ROOT/crates/wasm/typedoc.json' --out '$DOCS_TMP_DIR/typedoc'
  "
  run_step "docs_js_examples_build" bash -c "
    set -euo pipefail
    if [[ ! -d '$REPO_ROOT/js/examples/node_modules' ]]; then
      npm --prefix '$REPO_ROOT/js/examples' ci || npm --prefix '$REPO_ROOT/js/examples' install --no-audit --no-fund
    fi
    npm --prefix '$REPO_ROOT/js/examples' run build
  "
  rm -rf "$DOCS_TMP_DIR"
  DOCS_TMP_DIR=""
fi

REACHED_END=1
echo "=== [verify] ALL REQUESTED CHECKS PASSED ==="
