---
name: egraph-rs-system-patterns
description: Core architectural and implementation patterns of layouts, clustering, layering, and triangulation in egraph-rs.
---

# System and Architectural Patterns for `egraph-rs`

This skill details the v6 architecture and algorithmic patterns used across `egraph-rs`. Use this skill when modifying, maintaining, or implementing core algorithms, quality metrics, clustering, triangulation, cooperative solvers, or cross-language bindings.

## v6 Architecture Overview

- **Target Parity**: Native Rust, Python (PyO3 `abi3-py310`), and WebAssembly (`wasm-bindgen` single-`.wasm` pure ESM) operate over shared, strictly single-threaded pure CPU Rust crates.
- **Zero-Copy & Cooperative Design**: Flat `u32`-indexed `CsrGraph` and Structure-of-Arrays `Layout2d<S>` / `LayoutNd<S>` / `ManifoldLayout2d<S>` buffers avoid `HashMap` allocations in inner loops, exclude host objects (`Py<PyAny>`, `JsValue`) from Rust cores, and expose bit-exact `IterativeSolver<S>::step()` / `run_steps(k)` / `run()` loops alongside explicit `DeterministicRng::from_seed_u64(seed)`.
- **Legacy Reference**: Historical `petgraph`-based v5 crate patterns (`Sgd`, `PivotMds`, `RdMds`, `Omega`, `TsNet`/`BhTsNet`/`FitTsNet`/`LTsNet`, `GuMap`, `DiffusionKernel`, `petgraph-clustering`, `petgraph-algorithm-layering`) are archived in [Legacy v5 Patterns](./references/legacy_v5_patterns.md).

## Community Detection

- **V6 Pure-Rust Clustering (`egraph-clustering`, `crates/cluster/`)**:
  - Operates directly on flat `CsrGraph` and `egraph-linalg` primitives without `HashMap` in inner loops or aggregation; excludes `InfoMap` per `V6_RELEASE_PLAN.md`.
  - **`kmeans_plus_plus` (`KMeansConfig`, `KMeansResult`)**: Deterministic $D^2$ weighted centroid seeding via `DeterministicRng`, multi-start `n_init` keeping lowest inertia, and canonical `0..k` labels.
  - **`spectral_clustering` (`SpectralClusteringConfig`)**: Ng-Jordan-Weiss spectral clustering via `smallest_sym_laplacian_eigenpairs`, two-pass Modified Gram-Schmidt orthogonalization, row normalization, and `kmeans_plus_plus`.
  - **`louvain` (`LouvainConfig`, `LouvainResult`)**: Two-phase multi-level modularity optimization using `VisitStamp` accumulators and compact `WeightedLevelGraph` super-graph aggregation.
  - **`label_propagation` (`LabelPropagationConfig`)**: Asynchronous Fisher-Yates shuffled sweeps (`DeterministicRng::shuffle`) with `VisitStamp`-guarded neighbor vote counting.
  - **`coarsen_graph` / `project_clustered_layout` (`CoarsenedGraph`)**: Hash-free label compaction into super-graphs and deterministic multilevel layout projection.

## Layering Algorithms

- **V6 `CsrGraph` Layering (`egraph-layout`, `crates/layout/src/layering.rs`)**:
  - Uses iterative 3-color DFS (`cycle_edges`, `remove_cycles`) to detect and reverse back-edges in $O(V + E)$ time without recursion stack overflow on deep graphs, and `assign_layers(graph, method)` supporting `LayeringMethod::{LongestPath, NetworkSimplex}` without `petgraph` or `HashMap`.

## Triangulation

- **Delaunay Triangulation (`egraph-spatial`, feature `delaunay`)**:
  - Uses `spade::DelaunayTriangulation::bulk_load_stable` on `(x, y)`-sorted deduplicated sites via `Delaunay2d::triangulate` and `Delaunay2d::triangulate_edges_into` (emitting sorted canonical undirected `(u, v)` with `u < v` without constructing faces).

## Spatial Structures & Quality Metrics

- **`ArenaQuadtree2d` Weighted Coulomb Repulsion (`egraph-spatial`)**:
  - Stores `total_mass: f64` per `QuadNode` and exposes `build_weighted` / `build_weighted_xy` and `compute_coulomb_repulsion(theta, eps_sq, out_x, out_y)` for degree/mass-weighted Barnes-Hut repulsion in `ForceAtlas2Solver` and `YifanHuSolver` (accumulating geometry and moments in `f64` even when `S = f32`).
- **`SweepLineCrossings` vs `sampled_crossing_number` (`egraph-spatial` / `egraph-metrics`)**:
  - `SweepLineCrossings` runs in exact $O((|E| + K)\log |E|)$ time using a compact `FastU64Hasher`-backed `PairSet` (`pack_pair`) and zero `Vec<CrossingPair>` allocation on count/angle paths, remaining fast up to $K \sim 10^6\text{--}10^7$ crossings.
  - On dense hairball graphs with $M > 15{,}000$ edges where $K \sim 10^8$ or larger (e.g., `GION_3`, `bcsstk31`), exact enumeration is lower-bounded by $\Omega(K)$; use $O(M)$ `sampled_crossing_number` (deterministic `DeterministicRng`) instead of exact sweep-line counting.

## Spectral Linear Algebra & Neighborhood Primitives (`egraph-linalg` & `egraph-neighborhood`)

- **`ChebyshevWorkspace` 16-Column Tiled Recurrence & Tail Truncation (`egraph-linalg`)**:
  - Evaluates $\sum_k c_k T_k(\tilde{L}) V$ using cache-line-aligned `CHEBYSHEV_TILE_COLS = 16` (`{16, 8, 4, 1, tail}`) with `as_chunks::<W>()` and pre-validated CSR indices, eliminating per-nonzero `Option` bounds branches without `unsafe`.
  - `effective_chebyshev_len(coeffs, 1e-12)` trims trailing coefficients whose $\ell_1$ sum is $\le 10^{-12}$, provably bounding spectral operator-norm truncation error by $\le 10^{-12}$ while cutting coarse heat/JL passes significantly.
- **Shifted $\tilde{L}$ Reuse & Single-Pass DoG-Varadhan (`egraph-linalg` / `egraph-neighborhood`)**:
  - `ChebyshevWorkspace::apply_batch_dual`, `compute_kernel_jl_embedding_shifted`, and `compute_dual_jl_squared_norms_shifted` operate directly on the zero-diagonal shifted Laplacian $\tilde{L} = \frac{2}{\lambda_{\max}} L_{\mathrm{sym}} - I$ and generate Rademacher probes per tile without allocating $(N, d_{\mathrm{JL}})$ matrices when only row squared norms are needed.
  - `compute_regularized_dog_varadhan_landmarks` constructs $L_{\mathrm{sym}}$, $\tilde{L}$, and MaxMin landmarks once and fuses fine DoG + coarse scaffold (`deg_c = max(d, 75)` for explicit `degree`, `max(deg_fine, adaptive(t_coarse))` otherwise) across shared degrees `0..=deg_fine`.
- **Unweighted `CsrGraph<W>` Scalar Convention**:
  - `egraph-linalg` functions require `W: Scalar` on `CsrGraph<W>`, whereas `egraph-neighborhood::extract_1hop_2hop_csr` emits canonical unweighted `CsrGraph<f64>` (`weights: None`) graphs compatible across both `egraph-linalg` and Python/WASM bindings.

## V6 Cooperative Layout Solvers (`egraph-layout`, `crates/layout/`)

All v6 layout solvers implement `egraph_core::IterativeSolver<S>` with cooperative `step(&mut self, layout: &mut Layout2d<S>)`, `run_steps(layout, k)`, and `run(layout)` producing bit-exact results between `step()` loops and `run()`, zero non-test `.unwrap()`/`panic!`/`unsafe`, and explicit `DeterministicRng::from_seed_u64`.

- **`TsNetSolver<S>` (`crates/layout/src/ts_net.rs`)**:
  - Consolidates `TsNetBackend::{Exact, BarnesHut, FftInterpolation, LinearFft}` into an allocation-free cooperative solver backed by `FitField2d` (`interpolation_order=1` bilinear `ibFFT` fast default vs `interpolation_order=3` cubic interpolation matching DKE Phase 4 `suite30` NP2 `0.6301`) and `ArenaQuadtree2d` (`BarnesHut`).
  - `TsNetConfig::pure_2hop()` provides the validated Track A Pure 1+2-Hop adaptive perplexity preset (`perplexity_c=0.9`, `min_perplexity=5.0`, `max_perplexity=1200.0`, `power=2.0`, 3-stage optimization `(250, 250, 500)` with `exaggeration=4.0`, `lambda_c=(0.1, 0.01, 0.0)`).
- **`SgdSolver<S>` (`crates/layout/src/sgd.rs`)**:
  - Pure-Rust `u32` Structure-of-Arrays (SoA) SGD engine storing packed `PackedSymPair<S>` (`i`, `j_hinge` with `HINGE_BIT = 1u32 << 31` bit-packed hinge flag, `d`, `w`) or `PackedGenPair<S>` for asymmetric weights/distances.
  - **Cache-Blocked Shuffle**: For pair counts `> BLOCK_SHUFFLE_THRESHOLD` (`65_536`), `try_for_each_shuffled_pair` shuffles `BLOCK_SIZE` (`16_384`) block indices and intra-block slices rather than full-array Fisher-Yates, preserving cache locality while keeping `step()` and `run()` RNG consumption identical.
  - **3-Tier Sparse Barrier & Streaming Negatives**: `SparseBarrierConfig` builds Tier 1 (1-hop), Tier 2 (2-hop), and Tier 3 (non-2-hop DoG-Varadhan / MaxMin pivot pairs with optional one-sided hinge repulsion and Horvitz-Thompson $\gamma_{\mathrm{eff}}$ scaling) via `SgdSolver::sparse_barrier_from_landmarks` / `sparse_barrier_dog`; `StreamingNegativeConfig` supports minimal-RAM on-the-fly non-edge negative sampling (`step()`/`run()`); `FullSgd`, `RandomPairSgd`, and `Omega` constructors share the same solver core.
  - **M4 Compact & Scheduled Extensions**:
    - `SnapTFdpConfig` (`SgdSolver::snap_t_fdp`): Streams canonical undirected 1-hop edges directly from CSR into flat `u32` `edges_u`, `edges_v`, and `edge_order` arrays (12 bytes/edge base footprint without temporary tuple vectors or materialized negatives) and samples `negatives_per_edge` ($k=3$) uniform negatives on-the-fly (`< 512 MB` at 1M nodes, `< 1.3 GiB` Peak RSS on 4M-node `com-LiveJournal` in M6 benchmarks).
    - `PacmapGraphConfig` (`SgdSolver::pacmap_graph`): 3-tier near/mid-near/further graph loss across 3 optimization phases.
    - `Sgd2Config` (`SgdSolver::sgd2`) & `SgdScheduleKind::Squared`: Doubly stochastic SGD with squared progress decay $\eta(t) = \max(\eta_{\max}(1 - t)^2, \eta_{\min})$.
  - **Non-Euclidean Manifolds (`ManifoldLayout2d`)**: `SgdSolver::step_manifold` / `run_manifold` perform exponential/geodesic updates across `ManifoldKind::{HyperbolicPoincare, Spherical, Torus2d}` with finite-norm and pole/boundary clamping.
- **`FieldLayoutSolver<S>` (`crates/layout/src/field_layout.rs`)**:
  - Cooperative `t-FDP` (`TFdpConfig`, `RhoSchedule`) and `ibUMAP` (`IbUmapConfig` with $c_K = 4$, $\tau_t = 4\alpha_t$, hub damping $\eta_i$, and constructors from `CsrGraph`, `KnnGraph`, or `SparseProbabilities`) supporting `FieldRepulsionBackend::{IbFft, BarnesHut, Exact}` with zero per-step heap allocations.
- **`ContinuousFdp<S>` (`crates/layout/src/continuous_fdp.rs`)**:
  - `ForceAtlas2Solver<S>` (`ForceAtlas2Config`): Degree-weighted Barnes-Hut `ArenaQuadtree2d::build_weighted` / `compute_coulomb_repulsion` (or exact $O(N^2)$), LinLog attraction, strong gravity, outbound attraction distribution, and adaptive swing dampening.
  - `YifanHuSolver<S>` (`YifanHuConfig`): Multilevel `EC + MIVS` coarsening aggregating undirected coarse CSR edges once in canonical `c < cv` order and mirroring `(c, cv)` / `(cv, c)` for bit-identical `f64` symmetry without 1-ULP drift, paired with Barnes-Hut Coulomb repulsion and adaptive step cooling.
- **`PrismOverlapRemoval<S>` (`crates/layout/src/prism.rs`)**:
  - Proximity-stress node overlap removal using `Delaunay2d::triangulate_edges_into` (`bulk_load_stable` under `feature = "delaunay"`) or `StaticKdTree` $k$-NN fallback (under `--no-default-features`), Sweep-and-Prune (SAP) overlap detection, fused 2-RHS split `L+D+U` sparse CSR Symmetric SSOR-PCG (`SparsePcgWorkspace`), coordinate-magnitude ULP clearance margins (`coord_ulp = |coord| * S::EPSILON`) for `f32` precision at large coordinate offsets, and deterministic 2D scanline QPSC stall fallback (`STALL_NO_IMPROVE_STREAK = 8` or at `max_iterations`).
- **`ConstraintProjector<S>` & 2D Overlap Projection (`crates/layout/src/constraints.rs`)**:
  - Flat `Vec`-backed arena 1D QPSC projector using intrusive linked lists, flat CSR adjacency, and `VisitStamp` (zero `HashMap`/`HashSet`, zero recursion), plus `generate_layered_constraints`, `project_rectangle_no_overlap_2d` (with `pair_ulp` coordinate-scaled margins), and `project_clustered_rectangle_no_overlap_2d`.
- **`FdebSolver<S>` (`crates/layout/src/fdeb.rs`)**:
  - Reference-passing `FdebSolver::new(&graph, &layout, config)` (eliminating JS/WASM by-value layout drops), clamping `acos` inputs to `[-1.0, 1.0]` to prevent collinear `NaN`s, and outputting contiguous flat `(edge_offsets, points_x, points_y)` / `to_interleaved_points()` buffers across cooperative `step()` calls.
- **`GuMapSolver<S>` (`crates/layout/src/gumap.rs`)**:
  - Cooperative UMAP-for-graphs solver emitting canonical undirected `(u, v)`/`(v, u)` (`u < v`) pair order (eliminating directional sweep bias) and evaluating attractive/repulsive gradients via the `dist_sq.powf(b)` identity without per-pair `sqrt`.
  - **2-Stage Warm-Start Pipeline**: Pair `SgdSolver.sparse_barrier_dog` (global DoG-Varadhan scaffold) -> `GuMapSolver` (local neighborhood structure) -> `SpatialIntruderRefiner` (`SparseBarrierDoG -> GuMap -> Refiner`) for high NP2 on graphs with weak initial spectral layouts.
- **`SpatialIntruderRefiner<S>` (`crates/layout/src/refiner.rs`)**:
  - $O(N \log N)$ 2D active-set spatial refiner using `StaticKdTree` and reusable `KdQueryScratch` to detect strictly $>2$-hop spatial false positives inside each node's $k_u$-NN ball, protect 1-hop anchors, and run bounded `SgdSolver` passes (`SpatialRefinerConfig` defaults: `rounds=4`, `k_cap=32`, `margin_scale=1.2`, `w_anchor=1.0`, `w_fn=0.0`, `w_fp=0.2`, `hinge_fp=true`, `seed=42`). Skips rounds with zero active violations for early convergence on clean layouts.
- **`StressMajorizationSolver<S>` (`crates/layout/src/stress_majorization.rs`)**:
  - Gansner-Koren-North (2004) Stress Majorization using `BfsWorkspace`/`DijkstraWorkspace` APSP and a zero-per-iteration-allocation Jacobi Preconditioned Conjugate Gradient workspace (`PcgWorkspace`) with singular/disconnected component exit guards.
- **`KamadaKawaiSolver<S>` (`crates/layout/src/kamada_kawai.rs`)**:
  - Cooperative Kamada-Kawai (1989) solver selecting the max-gradient vertex per `step()` with `|det| <= 1e-12` Hessian singularity guard falling back to bounded steepest descent.

## Unified Facade, CLI & Cross-Target Integration (M5)

- **`egraph` Facade Crate (`crates/egraph`)**:
  - Unconditionally re-exports `egraph-core` and feature-gates `spatial`, `delaunay`, `linalg`, `neighborhood`, `layout`, `clustering`, `metrics`, `dataset`, and optional `petgraph` (excluded from `default` so pure-Rust and WASM consumers remain `petgraph`-free and `getrandom`-free), exposing `egraph::prelude::*`.
- **`egraph-dataset` Optional `petgraph` Feature (`crates/dataset`)**:
  - Embedded SuiteSparse/benchmark graphs (`karate`, `lesmis`, `1138_bus`, `3elt`, `dwt_1005`, `dwt_2680`, `poli`, `qh882`, `USpowerGrid`) and `parse_csr_csv` operate in pure-CSR mode under `--no-default-features`, while `petgraph` conversion helpers are isolated behind `feature = "petgraph"`.
- **`egraph-cli` (`crates/cli`)**:
  - `clap` v4 derive binary (`egraph`) providing `layout`, `metrics`, `cluster`, and `bench` subcommands over JSON edge-list/CSR, MatrixMarket (`.mtx`), CSR CSV, and embedded datasets with JSON output.
- **Python Ecosystem Interop (`crates/python/python/egraph/interop.py`)**:
  - `egraph.interop` provides `from_scipy_csr`, `to_scipy_csr`, `from_networkx`, `to_networkx`, and `from_igraph` with lazy optional imports (`egraph[interop]`), sanitized CSR construction, and strict stub/runtime alignment enforced via `mypy --strict` and `stubtest egraph`.
- **WASM Pure ESM, `LayoutNd.coordsView()`, & `runProgressive` (`crates/wasm`)**:
  - Default features (`full = ["delaunay"]`) exclude `legacy` (`petgraph`/`getrandom`/`serde`) to stay compact and `getrandom`-free while retaining opt-in `--features legacy`.
  - `scripts/assemble_pkg.mjs` emits pure ESM (`pkg/index.js`, `pkg/egraph_wasm.node.js`, `pkg/index.d.ts`), attaches `memory.grow`-resilient `coordsView()` to both `Layout2d` and `LayoutNd`, and exports `runProgressive(solver, layout, onFrame, options)` for bounded-frame `requestAnimationFrame` execution (verified by `js/examples` Vite demos and `tests/playwright_smoke.mjs`).
