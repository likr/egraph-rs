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
  --rust                   Full workspace fmt check, clippy, release nextest
  --python                 Build crates/python into worktree .venv via maturin develop --release & run tests/mypy
  --wasm                   Check wasm32 + run wasm-pack test --node crates/wasm
  --invariants             Run scripts/check_invariants.sh
  --parity                 Run scripts/parity_harness.py
  --docs                   Build Sphinx HTML + doctest, TypeDoc, and examples/web Vite bundle
  --all                    Run --rust --python --wasm --invariants --parity

Options:
  --crate <name>           Target crate (repeatable or comma-separated) for --fast / --invariants
  --test <name>            Scoped test module/name for --python or --wasm
  --release                Build Python bindings with maturin develop --release (default)
  --dev-py                 Build Python bindings in debug mode (maturin develop without --release)
  --debug-rust             Run --rust workspace tests in debug mode instead of --release
  --build-wasm             Also run wasm-pack build --target web --profile wasm-release + assemble_pkg.mjs + wasm-opt -Oz + size report
  --strict                 Strict mode for invariants & workspace-wide clippy -D warnings
  --json-summary <path>    Write machine-readable JSON summary to <path>
EOF
}

RUN_FAST=0; RUN_RUST=0; RUN_PYTHON=0; RUN_WASM=0; RUN_INVARIANTS=0; RUN_PARITY=0; RUN_DOCS=0
STRICT=0; RELEASE_PY=1; DEBUG_RUST=0; BUILD_WASM=0
TEST_FILTER=""; JSON_SUMMARY=""; CRATES=()

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
      for p in "${_parts[@]}"; do [[ -n "$p" ]] && CRATES+=("$p"); done
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
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ $((RUN_FAST + RUN_RUST + RUN_PYTHON + RUN_WASM + RUN_INVARIANTS + RUN_PARITY + RUN_DOCS)) -eq 0 ]] && RUN_FAST=1

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

VERIFY_HELPERS="$(helper_path verify_helpers.py)"

# Single-thread & build environment hygiene
export RUST_FONTCONFIG_DLOPEN=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

if command -v sccache >/dev/null 2>&1 && [[ -z "${RUSTC_WRAPPER:-}" ]]; then
  export RUSTC_WRAPPER=sccache
  export CARGO_INCREMENTAL="${CARGO_INCREMENTAL:-0}"
fi

PARALLEL_WORKERS="${EGRAPH_PARALLEL_WORKERS:-1}"
if [[ -z "${CARGO_BUILD_JOBS:-}" ]]; then
  if [[ "$PARALLEL_WORKERS" -le 1 ]]; then export CARGO_BUILD_JOBS=8
  elif [[ "$PARALLEL_WORKERS" -eq 2 ]]; then export CARGO_BUILD_JOBS=5
  elif [[ "$PARALLEL_WORKERS" -eq 3 ]]; then export CARGO_BUILD_JOBS=4
  else export CARGO_BUILD_JOBS=2
  fi
fi

export VIRTUAL_ENV="$REPO_ROOT/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
cd "$REPO_ROOT"

if command -v cargo-nextest >/dev/null 2>&1; then TEST_RUNNER="nextest"; else TEST_RUNNER="cargo-test"; fi

run_cargo_test_args() {
  if [[ "$TEST_RUNNER" == "nextest" ]]; then
    cargo nextest run --show-progress=none --no-input-handler --status-level=pass --final-status-level=fail "$@"
  else
    cargo test "$@"
  fi
}

OVERALL_START="$(python3 "$VERIFY_HELPERS" now)"
STEP_RECORDS_FILE="$(mktemp "${TMPDIR:-/tmp}/egraph_verify_steps.XXXXXX")"
DOCS_TMP_DIR=""; FINAL_STATUS="pass"; REACHED_END=0

write_json_summary() {
  local exit_code=$?
  if [[ "$exit_code" -eq 0 && ( "$FINAL_STATUS" != "pass" || "$REACHED_END" -ne 1 ) ]]; then
    exit_code=1; FINAL_STATUS="fail"
  elif [[ "$exit_code" -ne 0 ]]; then
    FINAL_STATUS="fail"
  fi
  if [[ -n "$JSON_SUMMARY" ]]; then
    mkdir -p "$(dirname "$JSON_SUMMARY")"
    local sccache_info=""
    command -v sccache >/dev/null 2>&1 && sccache_info="$(sccache --show-stats 2>/dev/null | head -n 15 || true)"
    python3 "$VERIFY_HELPERS" write-summary \
      "$JSON_SUMMARY" "$FINAL_STATUS" "$REPO_ROOT" "$PARALLEL_WORKERS" \
      "$CARGO_BUILD_JOBS" "${RUSTC_WRAPPER:-none}" "$OVERALL_START" \
      "$STEP_RECORDS_FILE" "$sccache_info"
  fi
  rm -f "$STEP_RECORDS_FILE"
  [[ -n "${DOCS_TMP_DIR:-}" ]] && rm -rf "$DOCS_TMP_DIR"
  exit "$exit_code"
}
trap write_json_summary EXIT

run_step() {
  local name="$1" t0 rc=0
  shift
  t0="$(python3 "$VERIFY_HELPERS" now)"
  echo "=== [verify] $name ==="
  if "$@"; then rc=0; else rc=$?; FINAL_STATUS="fail"; fi
  python3 "$VERIFY_HELPERS" record-step "$STEP_RECORDS_FILE" "$name" "$rc" "$t0"
  [[ "$rc" -eq 0 ]] || return "$rc"
}

ensure_venv() {
  local setup_bin
  setup_bin="$(helper_path setup_venv.sh)"
  if [[ ! -x "$VIRTUAL_ENV/bin/python" ]] || ! "$VIRTUAL_ENV/bin/python" -c "import maturin, pytest, numpy, scipy, networkx" >/dev/null 2>&1; then
    "$setup_bin" "$VIRTUAL_ENV"
  fi
}

ensure_docs_venv() {
  ensure_venv
  if ! "$VIRTUAL_ENV/bin/python" -c "import sphinx, sphinx_rtd_theme, matplotlib" >/dev/null 2>&1; then
    local req_file="$REPO_ROOT/crates/python/docs/requirements.txt"
    if command -v uv >/dev/null 2>&1; then uv pip install --python "$VIRTUAL_ENV/bin/python" -r "$req_file"
    else "$VIRTUAL_ENV/bin/python" -m pip install -r "$req_file"; fi
  fi
}

build_wasm_release_bundle() {
  (cd "$REPO_ROOT/crates/wasm" && wasm-pack build --target web --no-opt --profile wasm-release -d pkg .)
  [[ -f "$REPO_ROOT/crates/wasm/scripts/assemble_pkg.mjs" ]] && node "$REPO_ROOT/crates/wasm/scripts/assemble_pkg.mjs"
  if command -v wasm-opt >/dev/null 2>&1 && [[ -f "$REPO_ROOT/crates/wasm/pkg/egraph_wasm_bg.wasm" ]]; then
    wasm-opt -Oz --enable-bulk-memory --enable-nontrapping-float-to-int \
      "$REPO_ROOT/crates/wasm/pkg/egraph_wasm_bg.wasm" -o "$REPO_ROOT/crates/wasm/pkg/egraph_wasm_bg.wasm"
  fi
}

check_disk_space_warn() {
  local avail_kib avail_gib
  avail_kib="$(df -Pk "$REPO_ROOT" 2>/dev/null | awk 'NR==2 {print $4}' || true)"
  if [[ "$avail_kib" =~ ^[0-9]+$ ]] && (( avail_kib < 26214400 )); then
    avail_gib="$(awk -v k="$avail_kib" 'BEGIN {printf "%.1f", k/1048576}')"
    echo "[verify][WARN] Low free disk space on $REPO_ROOT (${avail_gib} GiB < 25 GiB recommended for parallel/release/WASM builds)." >&2
  fi
}

if [[ $((RUN_RUST + RUN_PYTHON + RUN_WASM + BUILD_WASM)) -gt 0 ]]; then
  check_disk_space_warn
fi

# --- Tier 0: --fast ---
if [[ "$RUN_FAST" -eq 1 ]]; then
  run_step "rust_fmt_check" cargo fmt --all -- --check
  FAST_CRATES=()
  if [[ "${#CRATES[@]}" -gt 0 ]]; then
    FAST_CRATES=(${CRATES[@]+"${CRATES[@]}"})
  else
    while IFS= read -r line; do [[ -n "$line" ]] && FAST_CRATES+=("$line"); done < <(python3 "$VERIFY_HELPERS" detect-changed-crates "$REPO_ROOT")
  fi
  if [[ "${#FAST_CRATES[@]}" -eq 0 ]]; then
    echo "[verify] No changed library crates detected; running baseline smoke crate check."
    for fallback in egraph-core; do
      if python3 "$VERIFY_HELPERS" has-package "$fallback"; then FAST_CRATES=("$fallback"); break; fi
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
        run_step "clippy_$c" cargo clippy --no-deps --all-targets -p "$c" -- -D warnings
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
    run_step "clippy_workspace" cargo clippy --workspace --all-targets -- -D warnings
  fi
  RUST_META_PKGS="$(python3 "$VERIFY_HELPERS" list-packages || true)"
  has_rust_pkg() { printf '%s\n' "$RUST_META_PKGS" | grep -qx "$1"; }
  run_batched_matrix() {
    local runner="$1" label="$2" flag_str="$3" pkg
    shift 3
    local flags=() pkg_args=()
    if [[ -n "$flag_str" ]]; then read -ra flags <<< "$flag_str"; fi
    for pkg in "$@"; do
      if has_rust_pkg "$pkg"; then pkg_args+=(-p "$pkg"); fi
    done
    if [[ "${#pkg_args[@]}" -eq 0 ]]; then return 0; fi
    if [[ "$runner" == "clippy" ]]; then
      run_step "clippy_${label}" cargo clippy --no-deps --all-targets ${flags[@]+"${flags[@]}"} "${pkg_args[@]}" -- -D warnings
    else
      run_step "rust_${label}" run_cargo_test_args ${flags[@]+"${flags[@]}"} "${pkg_args[@]}"
    fi
  }

  ALL_FEAT_CRATES=(egraph-dataset egraph-metrics egraph-metrics-cluster egraph-edge-routing egraph-layout-constraints egraph-layout-stress egraph-layout-force egraph-layout-sgd egraph-layout-embedding egraph-layout-multilevel egraph-layout-refine egraph-layout-composite egraph)
  NO_DEF_LAYOUT_CRATES=(egraph-edge-routing egraph-layout-constraints egraph-layout-stress egraph-layout-force egraph-layout-sgd egraph-layout-embedding egraph-layout-multilevel egraph-layout-refine egraph-layout-composite)
  NO_DEF_METRICS_CRATES=(egraph-metrics egraph-metrics-cluster)
  BASELINES_CRATES=(egraph-layout-constraints egraph-layout-stress egraph-layout-force egraph-layout-multilevel egraph-layout-refine egraph-layout-composite)

  run_batched_matrix clippy "all_features" "--all-features" "${ALL_FEAT_CRATES[@]}"
  run_batched_matrix clippy "no_default_layout" "--no-default-features" "${NO_DEF_LAYOUT_CRATES[@]}"
  run_batched_matrix clippy "no_default_metrics" "--no-default-features" "${NO_DEF_METRICS_CRATES[@]}"
  run_batched_matrix clippy "no_default_egraph" "--no-default-features" egraph
  run_batched_matrix clippy "no_default_dataset" "--no-default-features" egraph-dataset
  run_batched_matrix clippy "embedded_dataset" "--features embedded" egraph-dataset
  run_batched_matrix clippy "no_default_baselines" "--no-default-features --features baselines" "${BASELINES_CRATES[@]}"

  RUST_REL_FLAG=""
  [[ "$DEBUG_RUST" -eq 0 ]] && RUST_REL_FLAG="--release"
  run_step "rust_workspace_tests" run_cargo_test_args ${RUST_REL_FLAG:+$RUST_REL_FLAG} --workspace
  run_batched_matrix test "all_features" "${RUST_REL_FLAG:+$RUST_REL_FLAG }--all-features" "${ALL_FEAT_CRATES[@]}"
  run_batched_matrix test "no_default_layout" "${RUST_REL_FLAG:+$RUST_REL_FLAG }--no-default-features" "${NO_DEF_LAYOUT_CRATES[@]}"
  run_batched_matrix test "no_default_metrics" "${RUST_REL_FLAG:+$RUST_REL_FLAG }--no-default-features" "${NO_DEF_METRICS_CRATES[@]}"
  run_batched_matrix test "no_default_egraph" "${RUST_REL_FLAG:+$RUST_REL_FLAG }--no-default-features" egraph
  run_batched_matrix test "no_default_dataset" "${RUST_REL_FLAG:+$RUST_REL_FLAG }--no-default-features" egraph-dataset
  run_batched_matrix test "embedded_dataset" "${RUST_REL_FLAG:+$RUST_REL_FLAG }--features embedded" egraph-dataset
  run_batched_matrix test "no_default_baselines" "${RUST_REL_FLAG:+$RUST_REL_FLAG }--no-default-features --features baselines" "${BASELINES_CRATES[@]}"
fi

# --- Python ---
if [[ "$RUN_PYTHON" -eq 1 ]]; then
  run_step "python_venv" ensure_venv
  MATURIN_FLAGS=()
  [[ "$RELEASE_PY" -eq 1 ]] && MATURIN_FLAGS+=(--release)
  run_step "python_maturin_develop" bash -c "cd '$REPO_ROOT/crates/python' && '$VIRTUAL_ENV/bin/maturin' develop ${MATURIN_FLAGS[*]:-}"

  if [[ -n "$TEST_FILTER" ]]; then
    if [[ "$TEST_FILTER" == tests.* ]]; then PY_TARGET="$TEST_FILTER"
    elif [[ "$TEST_FILTER" == test_* || "$TEST_FILTER" == *.* ]]; then PY_TARGET="tests.$TEST_FILTER"
    else PY_TARGET="tests.test_$TEST_FILTER"; fi
    run_step "python_test_scoped" bash -c "cd '$REPO_ROOT/crates/python' && '$VIRTUAL_ENV/bin/python' -m unittest '$PY_TARGET'"
  else
    run_step "python_unittest_all" bash -c "cd '$REPO_ROOT/crates/python' && '$VIRTUAL_ENV/bin/python' -m unittest discover -s tests"
  fi

  if [[ -f "$REPO_ROOT/crates/python/python/egraph/_core.pyi" ]]; then
    TYPED_PY_TEST_NAMES=(
      test_amg_and_effective_resistance.py test_api_options_parity.py test_chebyshev_mds_and_declutter.py
      test_cluster_area_faithfulness.py test_cluster_area_faithfulness_weights.py
      test_cluster_baselines_and_metrics_dispatch.py test_clustering_and_manifold.py test_coarse_backbone.py
      test_community_and_clustered_layouts.py test_constraints_and_layering.py test_csr_and_layout2d.py
      test_docstrings.py test_force_and_overlap_solvers.py test_force_bundling_and_validation.py
      test_geometry_and_stress_metrics.py test_grip_gem_sel_pfs_sfc.py
      test_gumap_refiner_and_stress_solvers.py test_hde_and_sgd_presets.py test_interop.py
      test_layout3d_polylines_and_errors.py test_manifold_crossings_and_metrics_report.py
      test_maxent_stress_solver.py
      test_nd_and_manifold_solvers.py test_neighbor_and_sgd_solvers.py
      test_neighborhood_knn_and_landmarks.py test_rta_crosslessness_and_metric_errors.py
      test_screened_poisson.py test_spatial_trees_dijkstra_and_datasets.py
      test_spectral_and_chebyshev_linalg.py test_tsnet_and_sgd_solvers.py
    )
    TYPED_PY_TESTS=()
    for py_test in "${TYPED_PY_TEST_NAMES[@]}"; do TYPED_PY_TESTS+=("$REPO_ROOT/crates/python/tests/$py_test"); done
    run_step "python_mypy" "$VIRTUAL_ENV/bin/mypy" --strict "$REPO_ROOT/crates/python/python/egraph" "${TYPED_PY_TESTS[@]}"
    run_step "python_stubtest" "$VIRTUAL_ENV/bin/stubtest" egraph
  fi
fi

# --- WASM ---
if [[ "$RUN_WASM" -eq 1 ]]; then
  run_step "wasm32_check" cargo check -p egraph-wasm --target wasm32-unknown-unknown
  if [[ -n "$TEST_FILTER" ]]; then
    if [[ "$TEST_FILTER" == "wasm_suite" ]]; then
      run_step "wasm_test_wasm_suite" wasm-pack test --node "$REPO_ROOT/crates/wasm" --test wasm_suite
    else
      run_step "wasm_test_$TEST_FILTER" wasm-pack test --node "$REPO_ROOT/crates/wasm" --test wasm_suite -- "$TEST_FILTER"
    fi
  else
    run_step "wasm_test_all" wasm-pack test --node "$REPO_ROOT/crates/wasm" --test wasm_suite
  fi

  if [[ "$BUILD_WASM" -eq 1 ]]; then
    run_step "wasm_build_release" build_wasm_release_bundle
    if [[ -f "$REPO_ROOT/crates/wasm/tests/node_esm_smoke.mjs" ]]; then
      [[ -x "$REPO_ROOT/crates/wasm/node_modules/.bin/tsc" ]] || npm ci --prefix "$REPO_ROOT/crates/wasm"
      run_step "wasm_node_esm_smoke" node "$REPO_ROOT/crates/wasm/tests/node_esm_smoke.mjs"
    fi
  fi
fi

# --- Invariants ---
if [[ "$RUN_INVARIANTS" -eq 1 ]]; then
  INV_BIN="$(helper_path check_invariants.sh)"
  INV_MODE="$([[ "$STRICT" -eq 1 ]] && echo strict || echo report)"
  if [[ "${#CRATES[@]}" -gt 1 ]]; then
    for idx in "${!CRATES[@]}"; do
      INV_ARGS=(--mode "$INV_MODE" --crate "${CRATES[$idx]}")
      [[ "$BUILD_WASM" -eq 1 && "$idx" -eq 0 ]] && INV_ARGS+=(--check-wasm-size)
      run_step "invariants_${CRATES[$idx]}" "$INV_BIN" "${INV_ARGS[@]}"
    done
  else
    INV_ARGS=(--mode "$INV_MODE")
    [[ "${#CRATES[@]}" -eq 1 ]] && INV_ARGS+=(--crate "${CRATES[0]}")
    [[ "$BUILD_WASM" -eq 1 ]] && INV_ARGS+=(--check-wasm-size)
    run_step "invariants" "$INV_BIN" ${INV_ARGS[@]+"${INV_ARGS[@]}"}
  fi
fi

# --- Cross-target Parity ---
if [[ "$RUN_PARITY" -eq 1 ]]; then
  run_step "python_venv_for_parity" ensure_venv
  run_step "parity_mypy" "$VIRTUAL_ENV/bin/mypy" --strict --ignore-missing-imports "$REPO_ROOT/scripts/verify_helpers.py" "$REPO_ROOT/scripts/parity_harness.py" "$REPO_ROOT/scripts/parity/"
  PARITY_BIN="$(helper_path parity_harness.py)"
  PARITY_ARGS=()
  [[ -d "$REPO_ROOT/crates/wasm/pkg" ]] && PARITY_ARGS+=(--wasm-pkg "$REPO_ROOT/crates/wasm/pkg")
  run_step "parity_harness" "$VIRTUAL_ENV/bin/python" "$PARITY_BIN" ${PARITY_ARGS[@]+"${PARITY_ARGS[@]}"}
fi

# --- Docs & Examples ---
if [[ "$RUN_DOCS" -eq 1 ]]; then
  run_step "python_venv_for_docs" ensure_docs_venv
  if ! "$VIRTUAL_ENV/bin/python" -c "import egraph" >/dev/null 2>&1; then
    run_step "python_maturin_develop_for_docs" bash -c "cd '$REPO_ROOT/crates/python' && '$VIRTUAL_ENV/bin/maturin' develop --release"
  fi
  [[ ! -f "$REPO_ROOT/crates/wasm/pkg/index.d.ts" ]] && run_step "wasm_build_release_for_docs" build_wasm_release_bundle
  DOCS_TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/egraph_verify_docs.XXXXXX")"
  run_step "docs_sphinx_html" "$VIRTUAL_ENV/bin/python" -m sphinx -b html "$REPO_ROOT/crates/python/docs/source" "$DOCS_TMP_DIR/sphinx_html"
  run_step "docs_sphinx_doctest" "$VIRTUAL_ENV/bin/python" -m sphinx -b doctest "$REPO_ROOT/crates/python/docs/source" "$DOCS_TMP_DIR/sphinx_doctest"
  run_step "docs_typedoc" bash -c "
    set -euo pipefail
    [[ -x '$REPO_ROOT/crates/wasm/node_modules/.bin/typedoc' ]] || npm ci --prefix '$REPO_ROOT/crates/wasm'
    npx --prefix '$REPO_ROOT/crates/wasm' typedoc --options '$REPO_ROOT/crates/wasm/typedoc.json' --out '$DOCS_TMP_DIR/typedoc'
  "
  run_step "docs_examples_web_build" bash -c "
    set -euo pipefail
    if [[ ! -d '$REPO_ROOT/examples/web/node_modules' && ! -d '$REPO_ROOT/node_modules/vite' ]]; then
      if [[ -f '$REPO_ROOT/examples/web/package-lock.json' ]]; then npm --prefix '$REPO_ROOT/examples/web' ci
      else npm --prefix '$REPO_ROOT/examples/web' install --no-audit --no-fund --no-package-lock; fi
    fi
    npm --prefix '$REPO_ROOT/examples/web' run build
  "
  rm -rf "$DOCS_TMP_DIR"; DOCS_TMP_DIR=""
fi

REACHED_END=1
echo "=== [verify] ALL REQUESTED CHECKS PASSED ==="
