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
| **Tier 1 (`--rust`)** | `./scripts/verify.sh --rust [--debug-rust] [--strict]` | Before merging a Rust crate branch (full workspace `fmt --check`, `clippy`, release `nextest --workspace` + `egraph-dataset`/`egraph-spatial` `--all-features` by default; `--debug-rust` opts into debug tests; `--strict` includes legacy `petgraph-linalg-kernel`). |
| **Tier 1 (`--python`)** | `./scripts/verify.sh --python [--test <module>] [--dev-py]` | After editing `crates/python` or Rust code exposed to Python (rebuilds with `maturin develop --release` by default into local `.venv`; `--dev-py` opts into debug builds and `--release` remains accepted; runs `unittest` and `mypy` if `_core.pyi` exists; also validate stubs/v6 modules with `.venv/bin/mypy --strict crates/python/python/egraph crates/python/tests/test_v6_*.py`). |
| **Tier 1 (`--wasm`)** | `./scripts/verify.sh --wasm [--test <name>] [--build-wasm]` | After editing `crates/wasm` or core WASM-facing crates (`wasm32` check + `wasm-pack test --node`; `--build-wasm` runs `wasm-pack build --target web --no-opt --profile wasm-release` + `assemble_pkg.mjs` + explicit `wasm-opt -Oz --enable-bulk-memory --enable-nontrapping-float-to-int` + gzip size check because `[profile.wasm-release]` sets `strip = "symbols"`). |
| **Tier 2 (`--invariants`)** | `./scripts/verify.sh --invariants [--strict]` | Checks `cargo tree --edges normal -e features` (`rayon`/`linfa`/`wgpu` absence), `std::thread`/`thread_rng` scans, `wasm32` compile, `.wasm <= 600 KiB` gzip budget, and packaging metadata. (`--mode report` tolerates legacy `v6-alpha` bloat; `--mode strict` is required for v6 crates.) |
| **Tier 2 (`--parity`)** | `./scripts/verify.sh --parity` | Runs `scripts/parity_harness.py` (deterministic fixtures, adversarial `NaN`/error checks, optional Node.js WASM cross-check). |
| **Tier 3 (`--all`)** | `./scripts/verify.sh --all [--build-wasm] [--strict]` | Milestone gate / Verifier sign-off / Pre-PR integration check. |

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
| Python docs / doctest | `make python-docs` / `make python-doctest` |
| JS/TS format / examples | `npx prettier --write .` / `npm start` |

## 3. Mandatory Steps Before Completion

Whenever a task is ready to be completed:
1. Run `cargo fmt --all` and verify clean formatting (`cargo fmt --all -- --check`).
2. Run `./scripts/verify.sh --fast` (or `--rust` / `--python` / `--wasm` / `--all` matching scope; note legacy `petgraph-linalg-kernel` clippy warnings require `--strict` to surface, `--rust` and `--python` default to release mode with `--debug-rust` / `--dev-py` available for debug opt-in, and `--fast` stays debug).
3. Verify `clippy.toml` (`std::thread` forbidden) and `./scripts/check_invariants.sh` produce zero new violations; always pass `--edges normal` when inspecting dependencies manually with `cargo tree --edges normal -e features` so dev-dependencies do not trigger false positives.
4. **PyO3 `extension-module` unit-test rule**: Pure-Rust helpers tested under `#[cfg(test)]` inside `crates/python` MUST return plain Rust types (`Result<_, String>` or `Option<_>`), never `PyErr` / `PyValueError`, so `cargo test --workspace` and `cargo nextest run --workspace` link cleanly on macOS without `libpython` symbols.
5. For Python binding changes, verify inside the worktree's isolated `.venv` (never pollute `/Users/likr/diffusion-kernel-experiments/.venv` from an unmerged worktree).
6. Always pass `--no-pager` on `git diff`, `git log`, and `git show` commands.
7. In swarm sessions, post a brief interim status update on the board before starting any command expected to take `>60s` to prevent false monitor `[stall]` alerts.
