# Changelog

All notable changes to `egraph-rs` (`egraph` on PyPI and npm) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [6.0.0-alpha.6] - 2026-11-23

### Added

- **Row-Normalized 1+2-Hop tsNET (`TsNetSolver.from_graph_row_normalized` / `TsNetSolver.fromGraphRowNormalized`)**: Added `graph_12hop_row_normalized_probabilities` and `TsNetSolver` constructor supporting `w2` weighting and optional `max_2hop_per_node` capping across Rust, Python, and WASM.
- **Sparse 1+2-Hop Pair Extraction (`extract_sparse_1hop_2hop_pairs` / `extractSparse1hop2hopPairs`)**: Exposed `Sparse12HopPairs` and `extract_sparse_1hop_2hop_pairs(graph, max_2hop_per_node=32, seed=42)` in `egraph::prelude`, Python, and WASM.
- **Precomputed-Landmark Graph UMAP Constructors (`GuMapSolver.from_landmarks` / `IbUmapSolver.from_landmarks`)**: Added landmark-reusing constructors (`from_landmarks` / `fromLandmarks`, default `max_2hop_per_node=32`) across Rust, Python, and WASM.
- **PaCMAP-Graph 1-Hop Full Retention (`keep_all_1hop` / `keepAll1hop`)**: Added `keep_all_1hop` option on `PacmapGraphConfig` and `SgdSolver.pacmap_graph` / `SgdSolver.pacmapGraph` to preserve all 1-hop edges beyond `n_neighbors`.
- **Sampled Neighborhood Preservation (`sampled_np1` / `sampled_np2`, `sampledNp1` / `sampledNp2`)**: Added deterministic vertex-sampled 1-hop and 2-hop neighborhood preservation metrics to `egraph-metrics`, `evaluate_metrics` / `evaluateMetrics`, `egraph-cli` (`sampled-np1`, `sampled-np2`), Python, and WASM.
- **Spatial Refiner 2-Hop Anchoring & Dynamic `k_cap`**: Added `w_anchor2`, `k_cap_decay`, `k_cap_min`, `current_k_cap` / `currentKCap`, and `set_k_cap` / `setKCap` on `SpatialIntruderRefiner`.

### Changed

- **Automatic $O(N)$ tsNET `learning_rate` Scaling**: Switched `TsNetConfig.learning_rate` and Python `TsNetSolver` constructor defaults to `None` (resolving to $\max(200.0, N / 3.0)$ with a Stage 3 stability cap of $2N$ for $N < 100$).
- **Streaming `SweepLineCrossings` & Halved `SpatialIntruderRefiner` Memory**: Eliminated global `PairSet` materialization in `SweepLineCrossings::{count, count_edges, summarize, summarize_edges}` and eliminated the double-buffered pair vectors in `SpatialIntruderRefiner`.

### Fixed

- **Row-Normalized Perplexity Symmetrization**: Fixed `compute_knn_probabilities(..., row_normalize=True)` (`SymmetrizationMode::RowNormalized`) so row normalization applies to the symmetric support matrix rather than collapsing back to `StandardTsne`.
- **`pacmap_graph` FP 2-Hop Exclusion**: Excluded 2-hop neighbors (`has_common_neighbor`) during `SgdSolver.pacmap_graph` further-pair (`FP`) sampling.
- **WASM `sampledCrossingNumber` Docstring**: Corrected default `numSamples` documentation from `10000` to `4096`.

### Removed

- **Python `InfoMap` Binding**: Removed the legacy `InfoMap` PyO3 placeholder class (`crates/python/src/clustering/infomap.rs`) and its Python exports/stubs, aligning Python with WASM and `v6.0.0-alpha.5` release notes.

## [6.0.0-alpha.5] - 2026-11-23

`v6.0.0-alpha.5` (`6.0.0a5` on PyPI, `6.0.0-alpha.5` with `--tag next` on npm) delivers a clean architectural redesign across Native Rust, Python (PyO3 `abi3-py310`), and WebAssembly (`wasm-bindgen` pure ESM), achieving 100% cross-target feature parity under strictly single-threaded, deterministic pure CPU Rust execution. Prior tag `v6.0.0-alpha.4` remained unpublished after a `check_invariants.sh` lockfile-stderr bug was fixed prior to `v6.0.0-alpha.5`.

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

### Notes

- **Track A Interpolation Default**: `TsNetSolver.pure_2hop` defaults to `interpolation_order=1` (bilinear `ibFFT` for fast interactive/browser execution), while `interpolation_order=3` reproduces DKE Phase 4 cubic interpolation (`suite30` NP2 `0.6301`).
- **4M-Node Streaming Memory Profile**: `SgdSolver.snap_t_fdp` provides streaming `< 1.3 GiB` Peak RSS execution on 4M-node graphs (`com-LiveJournal`).
