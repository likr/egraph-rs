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

WORKSPACE_PKGS="$(cargo metadata --no-deps --format-version 1 2>/dev/null | python3 -c 'import json,sys; print("\n".join(p["name"] for p in json.load(sys.stdin)["packages"]))' || true)"

has_workspace_pkg() {
  printf '%s\n' "$WORKSPACE_PKGS" | grep -qx "$1"
}

FORBIDDEN_RE='(^|[^[:alnum:]_-])(rayon|linfa|linfa-nn|linfa-clustering|getrandom|wgpu)([^[:alnum:]_-]|$)'

check_cargo_tree_crate() {
  local label="$1"
  local policy="$2"
  shift 2

  local tree_out forbidden
  tree_out="$(cargo tree --edges normal -e features "$@" 2>&1 || true)"
  forbidden="$(printf '%s\n' "$tree_out" | grep -E "$FORBIDDEN_RE" || true)"
  if [[ -n "$forbidden" ]]; then
    if [[ "$policy" == "hard" ]]; then
      fail_hard "$label dependency tree pulls forbidden crates:"
    else
      warn_legacy "$label dependency tree pulls forbidden crates:"
    fi
    printf '%s\n' "$forbidden" | sed 's/^/       /'
  else
    pass "$label dependency tree is free of rayon/linfa/getrandom/wgpu"
  fi
}

# 1. Dependency tree gate (rayon / linfa / getrandom / wgpu)
# Pure-Rust v6 core/spatial/metrics/adapter crates and `egraph-wasm --no-default-features` ALWAYS fail_hard.
# Full/legacy `egraph-wasm` and `egraph-python` stay on warn_legacy until M1/M3 removes `linfa`.
if [[ -n "$TARGET_CRATE" ]]; then
  case "$TARGET_CRATE" in
    egraph-core|egraph-ffi|egraph-petgraph|egraph-dataset|egraph-metrics|egraph-linalg|egraph-neighborhood)
      check_cargo_tree_crate "$TARGET_CRATE" "hard" -p "$TARGET_CRATE"
      ;;
    egraph-spatial)
      check_cargo_tree_crate "egraph-spatial (default)" "hard" -p egraph-spatial
      check_cargo_tree_crate "egraph-spatial (--features delaunay)" "hard" -p egraph-spatial --features delaunay
      ;;
    egraph-wasm)
      check_cargo_tree_crate "egraph-wasm (--no-default-features)" "hard" -p egraph-wasm --no-default-features
      check_cargo_tree_crate "egraph-wasm (full)" "legacy" -p egraph-wasm
      ;;
    *)
      check_cargo_tree_crate "$TARGET_CRATE" "legacy" -p "$TARGET_CRATE"
      ;;
  esac
else
  for strict_crate in egraph-core egraph-ffi egraph-petgraph egraph-metrics egraph-linalg egraph-neighborhood; do
    if has_workspace_pkg "$strict_crate"; then
      check_cargo_tree_crate "$strict_crate" "hard" -p "$strict_crate"
    fi
  done
  if has_workspace_pkg "egraph-spatial"; then
    check_cargo_tree_crate "egraph-spatial (default)" "hard" -p egraph-spatial
    check_cargo_tree_crate "egraph-spatial (--features delaunay)" "hard" -p egraph-spatial --features delaunay
  fi
  if has_workspace_pkg "egraph-wasm"; then
    check_cargo_tree_crate "egraph-wasm (--no-default-features)" "hard" -p egraph-wasm --no-default-features
    check_cargo_tree_crate "egraph-wasm (full)" "legacy" -p egraph-wasm
  fi
  for legacy_crate in egraph-python egraph; do
    if has_workspace_pkg "$legacy_crate"; then
      check_cargo_tree_crate "$legacy_crate" "legacy" -p "$legacy_crate"
    fi
  done
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
    warn_legacy "Non-deterministic RNG (thread_rng/from_entropy) found in src/:"
    printf '%s\n' "$RNG_HITS" | head -n 20 | sed 's/^/       /'
  else
    pass "Zero thread_rng/from_entropy calls in crates/*/src/"
  fi
fi

# 3. wasm32-unknown-unknown cross-compilation check
WASM_CHECK_CRATE="${TARGET_CRATE:-egraph-wasm}"
case "$WASM_CHECK_CRATE" in
  egraph-python|egraph-cli)
    echo "[INFO] Skipping wasm32-unknown-unknown check for host-only crate '$WASM_CHECK_CRATE'"
    ;;
  *)
    if has_workspace_pkg "$WASM_CHECK_CRATE"; then
      if cargo check -p "$WASM_CHECK_CRATE" --target wasm32-unknown-unknown >/dev/null 2>&1; then
        pass "$WASM_CHECK_CRATE compiles cleanly for wasm32-unknown-unknown"
      else
        fail_hard "$WASM_CHECK_CRATE failed cargo check --target wasm32-unknown-unknown"
        cargo check -p "$WASM_CHECK_CRATE" --target wasm32-unknown-unknown || true
      fi
      if [[ "$WASM_CHECK_CRATE" == "egraph-spatial" ]]; then
        if cargo check -p egraph-spatial --features delaunay --target wasm32-unknown-unknown >/dev/null 2>&1; then
          pass "egraph-spatial (--features delaunay) compiles cleanly for wasm32-unknown-unknown"
        else
          fail_hard "egraph-spatial (--features delaunay) failed cargo check --target wasm32-unknown-unknown"
          cargo check -p egraph-spatial --features delaunay --target wasm32-unknown-unknown || true
        fi
      fi
      if [[ "$WASM_CHECK_CRATE" == "egraph-wasm" ]]; then
        if cargo check -p egraph-wasm --no-default-features --target wasm32-unknown-unknown >/dev/null 2>&1; then
          pass "egraph-wasm (--no-default-features) compiles cleanly for wasm32-unknown-unknown"
        else
          fail_hard "egraph-wasm (--no-default-features) failed cargo check --target wasm32-unknown-unknown"
          cargo check -p egraph-wasm --no-default-features --target wasm32-unknown-unknown || true
        fi
      fi
    fi
    ;;
esac

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
import json
import pathlib
import re
import sys

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None

root = pathlib.Path(sys.argv[1])
issues = []


def load_toml(path: pathlib.Path) -> dict:
    txt = path.read_text(encoding="utf-8")
    if tomllib is not None:
        return tomllib.loads(txt)
    # Minimal fallback for Python < 3.11 without tomllib
    data = {"_raw": txt}
    ws_ver = re.search(r'^\[workspace\.package\][^\[]*?^version\s*=\s*"([^"]+)"', txt, re.M | re.S)
    if ws_ver:
        data.setdefault("workspace", {}).setdefault("package", {})["version"] = ws_ver.group(1)
    pkg_sec = re.search(r'^\[package\][^\[]*', txt, re.M | re.S)
    if pkg_sec:
        sec = pkg_sec.group(0)
        if re.search(r'^version\.workspace\s*=\s*true', sec, re.M) or re.search(r'^version\s*=\s*\{\s*workspace\s*=\s*true', sec, re.M):
            data.setdefault("package", {})["version"] = {"workspace": True}
        else:
            m = re.search(r'^version\s*=\s*"([^"]+)"', sec, re.M)
            if m:
                data.setdefault("package", {})["version"] = m.group(1)
    return data


root_cargo_path = root / "Cargo.toml"
ws_version = None
ws_deps = {}
if root_cargo_path.exists():
    root_data = load_toml(root_cargo_path)
    ws_version = root_data.get("workspace", {}).get("package", {}).get("version")
    ws_deps = root_data.get("workspace", {}).get("dependencies", {})


def resolve_cargo_version(cargo_path: pathlib.Path) -> str | None:
    data = load_toml(cargo_path)
    ver = data.get("package", {}).get("version")
    if isinstance(ver, dict) and ver.get("workspace") is True:
        return ws_version
    if isinstance(ver, str):
        return ver
    return None


for sub in ("core", "ffi", "petgraph", "dataset", "spatial", "metrics", "linalg", "neighborhood"):
    sub_toml = root / "crates" / sub / "Cargo.toml"
    if sub_toml.exists() and ws_version:
        sub_ver = resolve_cargo_version(sub_toml)
        if sub_ver != ws_version:
            issues.append(f"crates/{sub}/Cargo.toml version ({sub_ver}) != workspace version ({ws_version})")

py_toml = root / "crates" / "python" / "Cargo.toml"
py_proj = root / "crates" / "python" / "pyproject.toml"
if py_toml.exists() and py_proj.exists():
    py_cargo_data = load_toml(py_toml)
    py_cargo_ver = resolve_cargo_version(py_toml)
    if ws_version and py_cargo_ver != ws_version:
        issues.append(f"crates/python/Cargo.toml version ({py_cargo_ver}) != workspace version ({ws_version})")

    if tomllib is not None:
        pyo3_spec = py_cargo_data.get("dependencies", {}).get("pyo3")
        pyo3_features = []
        if isinstance(pyo3_spec, dict):
            pyo3_features.extend(pyo3_spec.get("features", []))
            if pyo3_spec.get("workspace") is True and isinstance(ws_deps.get("pyo3"), dict):
                pyo3_features.extend(ws_deps["pyo3"].get("features", []))
        pyproj_data = load_toml(py_proj)
        req_py = str(pyproj_data.get("project", {}).get("requires-python", ""))
        proj_deps = pyproj_data.get("project", {}).get("dependencies", [])
    else:
        t_txt = py_toml.read_text(encoding="utf-8")
        r_txt = root_cargo_path.read_text(encoding="utf-8") if root_cargo_path.exists() else ""
        p_txt = py_proj.read_text(encoding="utf-8")
        pyo3_features = re.findall(r'abi3-py\d+', t_txt + ("\n" + r_txt if "pyo3.workspace" in t_txt else ""))
        m_req = re.search(r'^requires-python\s*=\s*"([^"]+)"', p_txt, re.M)
        req_py = m_req.group(1) if m_req else ""
        m_deps = re.search(r'^dependencies\s*=\s*\[([^\]]*)\]', p_txt, re.M | re.S)
        proj_deps = re.findall(r'"([^"]+)"', m_deps.group(1)) if m_deps else []

    if any(f in pyo3_features for f in ("abi3-py37", "abi3-py38", "abi3-py39")) or "abi3-py310" not in pyo3_features:
        issues.append(f"Python pyo3 abi3 features ({pyo3_features}) must enable abi3-py310")
    if not req_py.startswith(">=3.10"):
        issues.append(f"crates/python/pyproject.toml requires-python ({req_py!r}) must be '>=3.10'")
    if not any(re.match(r"^\s*numpy(\b|[<>=!~])", str(d)) for d in proj_deps):
        issues.append("crates/python/pyproject.toml omits runtime numpy dependency")

wasm_toml = root / "crates" / "wasm" / "Cargo.toml"
wasm_pkg = root / "crates" / "wasm" / "package.json"
if wasm_toml.exists() and wasm_pkg.exists():
    wasm_cargo_ver = resolve_cargo_version(wasm_toml)
    wasm_pkg_ver = json.loads(wasm_pkg.read_text(encoding="utf-8")).get("version")
    if ws_version and wasm_cargo_ver != ws_version:
        issues.append(f"crates/wasm/Cargo.toml version ({wasm_cargo_ver}) != workspace version ({ws_version})")
    if ws_version and wasm_pkg_ver != ws_version:
        issues.append(f"crates/wasm/package.json version ({wasm_pkg_ver}) != workspace version ({ws_version})")
    if wasm_cargo_ver != wasm_pkg_ver:
        issues.append(f"WASM Cargo.toml version ({wasm_cargo_ver}) != package.json version ({wasm_pkg_ver})")

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
