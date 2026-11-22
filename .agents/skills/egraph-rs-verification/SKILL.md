---
name: egraph-rs-verification
description: Commands and procedures to format, lint, check single-thread/WASM invariants, and run tiered tests for Rust, WebAssembly, Python, and JS/TS in the egraph-rs project.
---

# Verification and Testing Guide for `egraph-rs`

This skill defines the tiered verification harness across Native Rust, Python (PyO3), and WebAssembly (`wasm-bindgen` / Node / Browser) in `egraph-rs`.

## 1. Tiered Verification Runner (`scripts/verify.sh`)

Always prefer `./scripts/verify.sh` over raw `cargo`/`maturin`/`unittest` invocations. It automatically sets:
- `RUST_FONTCONFIG_DLOPEN=1`
- Single-thread guards (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `RAYON_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`)
- `RUSTC_WRAPPER=sccache` (shared compilation cache across git worktrees)
- `CARGO_BUILD_JOBS` throttled by `EGRAPH_PARALLEL_WORKERS` (`1->8`, `2->5`, `3->4`, `>=4->2`)
- Isolated worktree-local `$REPO_ROOT/.venv` (`uv`-managed Python 3.12)
- `cargo nextest run` when installed

| Tier | Command | When to Run |
|---|---|---|
| **Tier 0 (`--fast`)** | `./scripts/verify.sh --fast [--crate <name>] [--json-summary <path>]` | After every local edit in a worktree (stays debug for quick crate loops; checks `rustfmt`, `clippy -D warnings` + `clippy.toml`, `nextest` on changed/`--crate` packages, `std::thread` scan). For binding crates: `--crate egraph-python` runs `clippy` and directs to `--python`; `--crate egraph-wasm` runs `wasm32` check + `clippy` and directs to `--wasm`. |
| **Tier 1 (`--rust`)** | `./scripts/verify.sh --rust [--debug-rust] [--strict]` | Before merging a Rust crate branch (full workspace `fmt --check`, `clippy` + `egraph-layout`/`egraph` `--no-default-features` clippy, release `nextest --workspace` + `egraph-dataset`/`egraph-spatial`/`egraph` `--all-features` + `egraph-layout`/`egraph`/`egraph-dataset` `--no-default-features` + `egraph-cli` by default; `--debug-rust` opts into debug tests; `--strict` includes legacy `petgraph-linalg-kernel`). |
| **Tier 1 (`--python`)** | `./scripts/verify.sh --python [--test <module>] [--dev-py]` | After editing `crates/python` or Rust code exposed to Python (rebuilds with `maturin develop --release` by default into local `.venv`; `--dev-py` opts into debug builds and `--release` remains accepted; runs `unittest`, `mypy --strict` across `crates/python/python/egraph` and `crates/python/tests/test_v6_*.py`, and `stubtest egraph` if `_core.pyi` exists). |
| **Tier 1 (`--wasm`)** | `./scripts/verify.sh --wasm [--test <name>] [--build-wasm]` | After editing `crates/wasm` or core WASM-facing crates (`wasm32` check + `wasm-pack test --node` across default `full = ["delaunay"]`, `wasm_test_minimal` with `--no-default-features` including `--test v6_m4` and `--test v6_m5`, and `--features legacy`; `--build-wasm` runs `(cd crates/wasm && wasm-pack build --target web --no-opt --profile wasm-release -d pkg .)` requiring `wasm-pack >= 0.14` / `v0.15.1` + `assemble_pkg.mjs` + explicit `wasm-opt -Oz --enable-bulk-memory --enable-nontrapping-float-to-int` + `node crates/wasm/tests/node_esm_smoke.mjs` + size check). |
| **Tier 2 (`--invariants`)** | `./scripts/verify.sh --invariants [--strict]` | Checks `cargo tree --edges normal -e features` (`rayon`/`linfa`/`wgpu` absence, plus `getrandom`-free enforcement on all v6 crates, `egraph`, `egraph-cli`, `egraph-dataset`, and `egraph-wasm` default & `--no-default-features`), `std::thread`/`thread_rng` scans, `wasm32` compile (`egraph`, `egraph-wasm`), `.wasm <= 600 KiB` gzip budget, and cross-target packaging metadata (`Cargo.toml`, `pyproject.toml`, `crates/wasm/package.json` `version` + npm provenance `repository`). (`--mode report` tolerates legacy `v6-alpha` bloat; `--mode strict` is required for v6 crates.) |
| **Tier 2 (`--parity`)** | `./scripts/verify.sh --parity` | Runs `scripts/parity_harness.py` (deterministic fixtures, adversarial `NaN`/error checks, optional Node.js WASM cross-check). |
| **Tier 2 (`--docs`)** | `./scripts/verify.sh --docs` | Auto-provisions `crates/python/docs/requirements.txt` (`sphinx`, `sphinx-rtd-theme`, `matplotlib`) in `$REPO_ROOT/.venv` if missing, builds Sphinx HTML + `sphinx -b doctest` (into cleaned temp dirs), TypeDoc from `crates/wasm/typedoc.json`, and `npm --prefix js/examples run build`. |
| **Tier 3 (`--all`)** | `./scripts/verify.sh --all [--build-wasm] [--strict]` | Milestone gate / Verifier sign-off / Pre-PR integration check (`--rust --python --wasm --invariants --parity`; combine with `--docs` and `node crates/wasm/tests/playwright_smoke.mjs` for full M5+ release gates). |

## 2. Quick Reference Commands (Makefile & Direct)

| Target / Action | Command |
|---|---|
| Provision local `.venv` | `make venv-setup` (or `./scripts/setup_venv.sh`) |
| Tier 0 fast gate | `make verify-fast` or `make verify-fast CRATE=<crate>` |
| Full tiered suite | `make verify-all` |
| Invariant report / strict | `make invariants` / `make invariants-strict` |
| Cross-target parity | `make parity` |
| Worktree create / list / clean | `make worktree-create NAME=<slug>` / `make worktree-list` / `make worktree-clean NAME=<slug>` |
| `sccache` hit stats | `make sccache-stats` |
| Rust format / lint / test | `make fmt` / `make lint` / `make test` / `make test-crate CRATE=<name>` |
| WASM check / build / test / size | `make wasm-check` / `make wasm-build` / `make wasm-test` / `make wasm-size` |
| Python build / test / module | `make python-build` / `make python-test` / `make python-test-module MODULE=<mod>` |
| Python docs / doctest / TypeDoc / examples | `make python-docs` / `make python-doctest` / `./scripts/verify.sh --docs` |
| WASM browser Playwright smoke | `node crates/wasm/tests/playwright_smoke.mjs` |
| JS/TS format / examples | `npx prettier --write .` / `npm --prefix js/examples run build` |

## 3. Mandatory Steps Before Completion

Whenever a task is ready to be completed:
1. Run `cargo fmt --all` and verify clean formatting (`cargo fmt --all -- --check`).
2. Run `./scripts/verify.sh --fast` (or `--rust` / `--python` / `--wasm` / `--all` / `--docs` matching scope; note legacy `petgraph-linalg-kernel` clippy warnings require `--strict` to surface, `--rust` and `--python` default to release mode with `--debug-rust` / `--dev-py` available for debug opt-in, and `--fast` stays debug).
3. Verify `clippy.toml` (`std::thread` forbidden) and `./scripts/check_invariants.sh` produce zero new violations; always pass `--edges normal` when inspecting dependencies manually with `cargo tree --edges normal -e features` so dev-dependencies do not trigger false positives. Run both the workspace-wide banned-dependency scan (`cargo tree --workspace --edges normal -e features | grep -E '(^|[^[:alnum:]_-])(rayon|linfa|linfa-nn|linfa-clustering|wgpu)([^[:alnum:]_-]|$)'` returning zero matches) and strict per-crate `getrandom` checks across all v6 crates (`egraph-core`, `egraph-ffi`, `egraph-petgraph`, `egraph-dataset` default & `--no-default-features`, `egraph-spatial` default & `--features delaunay`, `egraph-metrics`, `egraph-linalg`, `egraph-neighborhood`, `egraph-clustering`, `egraph-layout` default & `--no-default-features`, `egraph` default/`--all-features`/`--no-default-features`, `egraph-cli`, and `egraph-wasm` default `full = ["delaunay"]` & `--no-default-features`).
4. **PyO3 `extension-module` unit-test rule**: Pure-Rust helpers tested under `#[cfg(test)]` inside `crates/python` MUST return plain Rust types (`Result<_, String>` or `Option<_>`), never `PyErr` / `PyValueError`, so `cargo test --workspace` and `cargo nextest run --workspace` link cleanly on macOS without `libpython` symbols.
5. For Python binding changes, verify inside the worktree's isolated `.venv` (never pollute `/Users/likr/diffusion-kernel-experiments/.venv` from an unmerged worktree), and confirm `.venv/bin/mypy --strict crates/python/python/egraph crates/python/tests/test_v6_*.py` and `.venv/bin/stubtest egraph` (both executed automatically by `verify.sh --python` and CI, with `scipy`, `networkx`, and `igraph` installed for `egraph.interop`) pass cleanly.
6. **WASM `--no-default-features` flag placement**: `wasm-pack test --node` forwards trailing options directly to `cargo test`, so pass `--no-default-features` directly WITHOUT `--` (for example: `wasm-pack test --node crates/wasm --no-default-features --test v6_core --test v6_metrics --test v6_linalg_neighborhood --test v6_layout_clustering --test v6_m4 --test v6_m5`), which `verify.sh --wasm` runs automatically as `wasm_test_minimal` alongside default `wasm_test_all` and `wasm_test_legacy` (`--features legacy`), and `.github/workflows/ci.yml` runs in `wasm-pipeline`. By contrast, `wasm-pack build` requires `-- --no-default-features` after `--`. Keep `crates/wasm/Cargo.toml` `[[test]]` entries, `wasm_test_minimal`, and CI synchronized whenever new feature-ungated `v6_*.rs` WASM integration tests are added. After `--build-wasm` and `npm --prefix js/examples run build`, run `node crates/wasm/tests/playwright_smoke.mjs` to verify Chromium `memory.grow` detach recovery, `N=5,000` `runProgressive` `<= 10 ms`/frame budget, and all 13 `js/examples` routes.
7. **Single-thread SciPy / DKE micro-benchmarks**: Run Wave D comparative benchmarks from `egraph-rs/.venv` under explicit single-thread guards (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1`) with DKE kept strictly read-only (stub `sys.modules["utils"]` if needed to avoid `matplotlib` imports). Report pure-Python/SciPy-sparse graph preprocessing (`1+2hop`, `Track-A`) separately from C++ `_sparsetools`-backed SciPy SpMM kernels (`DoG-Var`).
8. **Local Release Packaging Clean-Env Smoke Recipes (M6)**:
   - **Python wheel & `sdist`**: Build into a scratch output dir under `<appDataDir>/scratch/teamwork_tmp/<id>` (`maturin build --manifest-path crates/python/Cargo.toml --release -o "$TMP/dist"` and `maturin sdist --manifest-path crates/python/Cargo.toml --out "$TMP/dist"`), then install each into a fresh `python3 -m venv` (without `egraph-rs` editable checkout on `sys.path`) and verify `import egraph; assert egraph.__version__` + `CsrGraph` / `TsNetSolver` / `Layout2d` smoke execution.
   - **npm ESM tarball**: Run `(cd crates/wasm && npm pack --dry-run && npm pack --pack-destination "$TMP")` after `--build-wasm`, verify `package/package.json` retains `"repository": {"type": "git", "url": "git+https://github.com/likr/egraph-rs.git", "directory": "crates/wasm"}` for `--provenance`, and smoke-import the unpacked tarball in Node.js ESM (`npm install --prefix "$TMP/consumer" "$TMP"/egraph-*.tgz`).
9. Always pass `--no-pager` on `git diff`, `git log`, and `git show` commands.
10. In swarm sessions, post a brief `@coordinator [interim]` status update on the board before starting any command expected to take `>60s` to prevent false monitor `[stall]` alerts.
