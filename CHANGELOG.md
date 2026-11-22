# Changelog

All notable changes to `egraph-rs` (`egraph` on PyPI and npm) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [6.0.0-alpha.4] - 2026-11-23

`v6.0.0-alpha.4` (`6.0.0a4` on PyPI, `6.0.0-alpha.4` with `--tag next` on npm) delivers a clean architectural redesign across Native Rust, Python (PyO3 `abi3-py310`), and WebAssembly (`wasm-bindgen` pure ESM), achieving 100% cross-target feature parity under strictly single-threaded, deterministic pure CPU Rust execution.

### Added

- **Workspace & Core Architecture (`egraph-core`, `egraph-ffi`, `egraph-petgraph`, `egraph-dataset`, `egraph`, `egraph-cli`)**:
  - Zero-copy `u32`-indexed `CsrGraph` and Structure-of-Arrays `Layout2d`, `LayoutNd`, and `ManifoldLayout2d` (Poincare hyperbolic disk, spherical, 2D torus).
  - Cooperative `IterativeSolver` execution (`step()`, `run_steps(k)` / `runSteps(k)`, `run()`) for non-blocking `requestAnimationFrame` browser rendering via `runProgressive`.
  - Explicit deterministic `u64` seeded RNG (`DeterministicRng::from_seed_u64`), `CoreError` structured error handling, and unified feature-gated `egraph` facade + `clap`-based `egraph-cli`.
- **Spatial & Quality Metrics (`egraph-spatial`, `egraph-metrics`)**:
  - `IbFft2d` ($p=1$ default, `G_dual` non-periodic convolution) and fused `FitField2d` (halving particle forward 2D FFT calls), `ArenaQuadtree2d`, `StaticKdTree`, `SweepLineCrossings`, `RectangleOverlapSweep`, and `spade`-backed `Delaunay2d` (`delaunay` feature).
  - Selective `evaluate_metrics` / `evaluateMetrics`: closed-form and pivot-sampled Scale-Normalized Stress (`SNS`), sweep-line crossing numbers (`crossing_edges`, `crossing_edges_torus`), `StaticKdTree`-backed `NP1` / `NP2` / Gabriel / Node Resolution, `prism_distortion`, `drng_preservation`, and `dgg_preservation`.
- **Linear Algebra & Neighborhoods (`egraph-linalg`, `egraph-neighborhood`)**:
  - Zero-allocation 3-vector `ChebyshevWorkspace` with adaptive polynomial degree and PSD-projected regularized DoG (`R1`/`R2`) + JL embeddings, `SubspaceEigen`, built-in `std = 1.0` normalized `PivotMds`, Koren degree-normalized spectral initialization, and core-first `MaxMinSp`.
  - Pure 1+2-hop non-duplicated CSR extraction (`extract_1hop_2hop_csr`), node-wise adaptive perplexity, 2-stage spectral reranked $k$-NN & Hybrid Adaptive $k$-NN, `lambda`-rescaling, `fuzzy_simplicial_set`, and `GuMap` 1+2-hop priority CSR.
- **Track A, Track B & Unified Force Layouts (`egraph-layout`, `egraph-clustering`)**:
  - Track A: Unified `TsNetSolver` (`pure_2hop`, `from_knn`, `from_probabilities`, `from_graph_partial_bfs`) with `ibFFT`, Barnes-Hut, and Exact backends.
  - Track B: Regularized Chebyshev DoG + 3-tier `SparseBarrier` `SgdSolver`, `GuMapSolver`, and 2D `StaticKdTree` `SpatialIntruderRefiner` (`hinge_fp=True`, `w_fn=0.0` defaults).
  - Unified force & multilevel solvers: `TFdpSolver` (`ibFFT`), `IbUmapSolver`, `SgdSolver.snap_t_fdp` (edge-incident $k=3$ negative sampling), `SgdSolver.pacmap_graph`, `SgdSolver.sgd2`, `ForceAtlas2Solver`, `YifanHuSolver` (`EC+MIVS`), `StressMajorizationSolver`, `KamadaKawaiSolver`, `FdebSolver`, $O(N \log N)$ `PrismOverlapRemoval`, `SeparationConstraint`, and DAG layering (`cycle_edges`, `remove_cycles`, `assign_layers`, `generate_layered_constraints`).
  - Pure Rust community detection and coarsening: `kmeans_plus_plus`, `spectral_clustering`, `louvain_communities`, `label_propagation`, `coarsen_graph`, and `project_clustered_layout`.
- **Python & WebAssembly Tooling / Docs / Demos**:
  - Python `egraph.interop` (`from_scipy_csr`, `to_scipy_csr`, `from_networkx`, `to_networkx`, `from_igraph`), complete PEP 484/561 `_core.pyi` + `py.typed` (`mypy --strict` and `stubtest` verified), and Sphinx documentation + `doctest`.
  - WASM pure ESM single-`.wasm` package (`pkg/index.js`, `pkg/egraph_wasm.node.js`, `pkg/index.d.ts`) with `Layout2d.coordsView()` (`memory.grow` detach-safe) and `runProgressive`, TypeDoc API reference, and updated `js/examples` Vite + React demos.
  - Migration guide at `docs/MIGRATION_V6.md`.

### Changed

- **Strictly Single-Threaded Pure Rust**: Removed `rayon`, `linfa`, `serde-wasm-bindgen` (default WASM build), and host-object (`Py<PyAny>` / `JsValue`) storage from core solvers; aligned `ndarray` (`0.17`, `default-features = false`) across the workspace.
- **Python Packaging**: Upgraded to PyO3 `0.28` (`abi3-py310`, `requires-python = ">=3.10"`) with runtime `numpy >= 1.26` dependency.
- **WASM Packaging**: Consolidated duplicate `dist/*` / UMD / Rollup outputs into a single `wasm-opt -Oz` optimized ESM `.wasm` bundle (`<= 600 KiB` gzipped) shared across browsers, bundlers, and Node.js 18+.
- **Breaking API Renames**: `OverwrapRemoval` -> `PrismOverlapRemoval`; `TsNet` / `BhTsNet` / `FitTsNet` / `LTsNet` -> `TsNetSolver`; unscaled `stress` -> Scale-Normalized Stress (`SNS`).

### Removed

- Removed legacy `InfoMap` clustering placeholder and legacy dead workflow/crate scaffolding (`crates/layout/random`).
