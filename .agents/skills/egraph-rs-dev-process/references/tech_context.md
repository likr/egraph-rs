# Technical Context: `egraph-rs`

This reference document outlines the technical environment, folder architecture, toolchain, and constraints for `egraph-rs` (active roadmap: `docs/V6_RELEASE_PLAN.md`, latest v6 release `v6.0.0-alpha.8`).

## Technologies Used

### Core Technologies
- **Rust (`edition = "2024"` workspace-wide)**: Primary implementation language.
- **Cargo + `sccache` + `cargo-nextest`**: Dependency management, cached parallel compilation across worktrees, and fast test runner.
- **PyO3 + `maturin` + `uv` (Python `>=3.10`, `abi3-py310`)**: Isolated per-worktree `.venv` Python bindings with `numpy`, `scipy`, `pytest`, `mypy`, `ruff`.
- **WebAssembly (`wasm32-unknown-unknown`, `wasm-bindgen`, `wasm-pack`, `binaryen`/`wasm-opt`) + Node.js / ESM**: Browser and Node.js WASM integration targeting a single-binary ESM bundle.

### Key Directories
- `crates/` — 45 workspace crates: 27 modern `egraph*` crates (`egraph-core`, `egraph-ffi`, `egraph-petgraph`, `egraph-dataset`, `egraph-spatial-{trees,fft,crossings}` + `egraph-spatial` umbrella, `egraph-linalg` + `egraph-linalg-{mds,amg}`, `egraph-neighborhood`, `egraph-clustering`, `egraph-metrics-{geometry,cluster}` + `egraph-metrics` umbrella, `egraph-layout-{constraints,stress,force,neighbor,sgd,multilevel}` + `egraph-layout` umbrella, unified `egraph` facade, `egraph-cli`, `egraph-python`, `egraph-wasm`) plus 18 legacy v5 `petgraph-*` crates.
- `scripts/` — Worktree, venv, invariant-check, tiered verification (`verify.sh`, `verify_helpers.py`), and cross-target parity harness (`parity_harness.py`, `parity/*.py` 30 modules) scripts.
- `docs/` — Release plan and migration guide (`docs/V6_RELEASE_PLAN.md`, `docs/MIGRATION_V6.md`).
- `js/dataset`, `js/examples` — JavaScript/ESM datasets and Vite interactive examples.
- `.agents/` — Workspace knowledge index (`AGENTS.md`) and agent skills (`skills/*`).

## Hard Architectural Constraints

1. **Strictly Single-Threaded Plain CPU Execution**:
   - No `rayon`, no `std::thread`, no Web Workers / `SharedArrayBuffer` requirement, no GPU (`CUDA` / `WebGPU` / `wgpu`), and no C/Fortran FFI dependencies in core crates.
   - `ndarray` must use `default-features = false` so `rayon` is never activated transitively.
   - Runtime thread-pool caps (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `RAYON_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`) are enforced by `scripts/verify.sh`.
2. **Equal First-Class Native / Python / WASM Parity**:
   - Deterministic `u64`-seeded RNGs everywhere (`thread_rng` / `from_entropy` forbidden in core `src/`).
   - Cooperative `step()` / `run_steps(k)` execution for non-blocking single-thread browser rendering.
   - Fallible boundary validation (`Result` / `PyValueError` / `JsError` — never panicking or poisoning WASM instances).
   - System dependency guard: `RUST_FONTCONFIG_DLOPEN=1` required when building/testing crates that transitively reference `fontconfig`.
