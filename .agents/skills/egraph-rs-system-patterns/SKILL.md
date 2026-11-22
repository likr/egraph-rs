---
name: egraph-rs-system-patterns
description: Core architectural and implementation patterns of layouts, clustering, layering, and triangulation in egraph-rs.
---

# System and Architectural Patterns for egraph-rs

This skill details the architectural design and algorithms used across the `egraph-rs` library. Use this skill when modifying, maintaining, or implementing core algorithms (SGD, MDS, community detection, layering, triangulation, etc.) or cross-language bindings.

## Architecture & Design

- **Modular Crate Structure**
  - algorithm (connected-components, shortest-path, triangulation, layering)
  - clustering (community detection)
  - dataset (graph dataset loaders)
  - drawing (Euclidean, Spherical, Hyperbolic, Torus)
  - edge-bundling (force-directed edge bundling)
  - layout (SGD, MDS, Stress-Majorization, Kamada-Kawai, Omega, overlap-removal, random, separation-constraints)
  - linalg (RdMds - Resistance-distance MDS for spectral embeddings)
  - quality-metrics (layout evaluation)
  - language bindings (Python, WebAssembly)
  - Hardened edge-case validation across `quality-metrics` (NaN/collinear `crossing_number`, degenerate `aspect_ratio`, edgeless `neighborhood_preservation`), `WeightedEdgeLength` (indexed by edge ID with fallible lookup), and PyO3 bindings (`PyResult`/`PyValueError` instead of panics).

- **Key Patterns**
  - Builder, Strategy, Adapter, Visitor, Factory Methods
  - Trait-based Interfaces (`CommunityDetection`, `LayeringAlgorithm`)
  - Composition over inheritance

## Layout Algorithms

### SGD (Stochastic Gradient Descent)
Force-directed layout with a unified concrete implementation.
- **Architectural Evolution**: Uses a concrete `Sgd<S>` struct rather than a trait-based design for efficiency.
- **Unified Framework**: Supports Full, Sparse, Distance-Adjusted, Omega, and Kernel-SGD variants through different node pair strategies.
- **Core Structure**:
  ```rust
  pub struct Sgd<S> {
      node_pairs: Vec<(usize, usize, S, S, S, S)>, // (i, j, dij, dji, wij, wji)
      epsilon: S,    // Stability parameter
      eta_min: S,    // Minimum learning rate (calculated from weights)
      eta_max: S,    // Maximum learning rate (calculated from weights)
  }
  ```
- **Learning Rate**: Automatic calculations based on weight distribution.
- **Scheduler Integration**: Custom scheduler trait:
  ```rust
  pub trait Scheduler<S> {
      fn run<F: FnMut(S)>(&mut self, callback: &mut F);
      fn step<F: FnMut(S)>(&mut self, callback: &mut F);
      fn is_finished(&self) -> bool;
  }
  ```
- **One-Sided Hinge Repulsion & Barrier Sparse SGD**:
  - `Sgd::new_with_hinge_pairs(springs, hinges)` pairs standard two-sided spring pairs with one-sided hinge repulsion pairs that exert force strictly when $0 < \|x_i - x_j\| < d_{\mathrm{barrier}}$.
  - PyO3 NumPy constructors: `Sgd.from_numpy_pairs`, `Sgd.from_numpy_hinge_pairs`, and `Sgd.from_numpy_pairs_with_hinges`.
  - `SparseSgd::barrier()` builder configures 3-tier barrier construction (`calibrate_one_hop`, `max_2hop_per_node`, `hinge_repulsion`, `delta_rel`, `barrier_sym`, `omega_1`, `omega_2`, `gamma_non2`, `min_ratio`, `eps_dist`, `seed`) and constructs pairs via `build_barrier_from_landmark_tables`.
- **Fallible Validation**: `FullSgd` and `StressMajorization` provide fallible square-matrix validation and `try_*` APIs with $n \le 1$ and NaN/Inf guards.

### MDS (Multidimensional Scaling) & Pivot MDS
Dimension reduction layout algorithms supporting unified distance matrix inputs.
- **Location**: `crates/layout/mds/`
- **Distance Trait**: `PivotMds` and `ClassicalMds` accept any matrix implementing `petgraph_linalg_kernel::Distance<N2, S> + ?Sized` (including `KernelDistance`, `NegLogDistance`, `NegLogSimDistance`, `PivotedNegLogDistance`, `FullDistanceMatrix`, `PivotedDistanceMatrix`).
- **Constructor Patterns**:
  - `new_with_distance_matrix`: Automatically uses default node indices (`petgraph::graph::node_index`).
  - `new_with_distance_matrix_and_indices`: Accepts explicit node index slice `&[N2]` for custom indexing.
- **Scaling & Validation**:
  - `PivotMds` computes $(n/k)^{1/4} C V \mathrm{diag}(e^{-1/4}) \approx U \Sigma^{1/2}$ with zero-eigenvalue and disconnected-graph finite-coordinate guards (preventing historical $U \Sigma^2$ scale explosion).
  - `ClassicalMds` provides fallible square-matrix validation.

### RdMds (Resistance-distance MDS)
Computes spectral embeddings using the graph Laplacian.
- **Location**: `crates/linalg/rdmds/`
- **Core Structure**:
  ```rust
  pub struct RdMds<S> {
      pub d: usize,
      pub shift: S,
      pub eigenvalue_max_iterations: usize,
      pub cg_max_iterations: usize,
      pub eigenvalue_tolerance: S,
      pub cg_tolerance: S,
  }
  ```

### Omega
Generates node pairs for SGD from spectral embeddings.
- **Location**: `crates/layout/omega/`
- **Core Structure**:
  ```rust
  pub struct Omega<S> {
      pub k: usize,        // Random pairs per node
      pub min_dist: S,     // Min distance
  }
  ```

### tsNET, BH-tsNET, FIt-tsNET & L-tsNET
t-SNE based graph layout algorithms optimizing KL-divergence, compression, and node repulsion.
- **Location**: `crates/layout/ts-net/`
- **Shared Acceleration & Validation**:
  - `partial_bfs` reuses `visited_stamp` and layer buffers across nodes to maintain true $O(N \cdot k \cdot \bar{d})$ complexity without per-node $O(N)$ allocations.
  - Per-node adaptive perplexity and variable-length CSR $k$-NN inputs are supported via `try_run_with_csr_knn_adaptive` in Rust and exposed on `BhTsNet`, `FitTsNet`, and `LTsNet` in PyO3 via `run_with_knn` and `run_with_knn_adaptive(drawing, indptr, indices, distances, perplexities)`.
  - Builders and runners reject non-finite parameters, invalid $k$-NN entries, and non-finite or mismatched drawings.
- **tsNET**:
  - Distance Trait: Generic over `petgraph_linalg_kernel::Distance<N, S> + ?Sized`.
  - Builder Pattern: Instantiated via `TsNetBuilder<S>` returning `Result<TsNet<S>, String>`.
  - 3-Stage Dynamic Optimization: Early exaggeration, early compression ($\lambda_c = 1.2$), and final refinement ($(\lambda_{KL}, \lambda_c, \lambda_r) = (1.0, 0.01, 0.6)$).
- **BH-tsNET**:
  - Barnes-Hut accelerated $O(N \log N)$ algorithm for large graphs.
  - C0: $O(N)$ Partial BFS for $k$-nearest neighbors with random tie-breaking.
  - C1 & C2: 2D Quadtree spatial approximation for KL divergence repulsion and entropy gradients.
  - Builder Pattern: Instantiated via `BhTsNetBuilder<S>` returning `Result<BhTsNet<S>, String>`.
- **FIt-tsNET**:
  - Fast Interpolation accelerated $O(N \log N)$ algorithm for large graphs.
  - C0: $O(N)$ Partial BFS for $k$-nearest neighbors with random tie-breaking.
  - C1: $O(N)$ 2D FFT-accelerated polynomial grid interpolation for low-dimensional Student-$t$ KL repulsion.
  - C2: $O(N \log N)$ 2D Quadtree spatial approximation for entropy gradient.
  - Builder Pattern: Instantiated via `FitTsNetBuilder<S>` returning `Result<FitTsNet<S>, String>`.
- **L-tsNET**:
  - Linear-time $O(N)$ and space $O(N + M)$ algorithm for large graphs.
  - C0: $O(N)$ Partial BFS for $k$-nearest neighbors with random tie-breaking.
  - C1: $O(N)$ 2D FFT-accelerated polynomial grid interpolation for Student-$t$ KL repulsion.
  - C2: $O(N)$ 2D FFT-accelerated polynomial grid interpolation for entropy gradient using reformulated $h_1, h_2, h_3$ (Stage 3 sign matches `BhTsNet` / `FitTsNet` repulsion without double negation).
  - Builder Pattern: Instantiated via `LTsNetBuilder<S>` returning `Result<LTsNet<S>, String>`.

### GUMAP (Graph UMAP)
Graph layout algorithm adapting Uniform Manifold Approximation and Projection (UMAP) to networks.
- **Location**: `crates/layout/gumap/`
- **Distance Trait**: Generic over `petgraph_linalg_kernel::Distance<N, S> + ?Sized`.
- **Sparse $k$-NN Input**: Supports `run_with_knn` and variable-length CSR `run_with_knn_csr` (`try_run_with_csr_knn`) with deterministic edge ordering (eliminating $O(N^2)$ full pairwise search) and strict non-finite/mismatched-drawing validation.
- **Fuzzy Simplicial Set**:
  - High-dimensional $k$-nearest neighbor search, local distance scale $\rho_i = \min_{j \in kNN(i)} d_{ij}$, and binary search for $\sigma_i$ targeting $\log_2(k)$.
  - Fuzzy set union symmetrization: $p_{ij} = p_{j|i} + p_{i|j} - p_{j|i} p_{i|j}$.
- **Curve Fitting**: Low-dimensional parameter estimation for $(a, b)$ via Levenberg-Marquardt non-linear least squares fit of $\psi(d)$ from `min_dist` and `spread`.
- **SGD Optimization**: Probabilistic edge sampling for attractive forces and negative sampling for repulsive forces with linear learning rate decay.
- **Builder Pattern**: Instantiated via `GuMapBuilder<S>` returning `Result<GuMap<S>, String>`.


### Kernel-SGD & Diffusion Kernels
Diffusion kernel-based computations for graph distances and embeddings.
- **Location**: `crates/linalg/kernel/`
- **BandPass & Multiscale Kernels**:
  - `BandPassDiffusionKernel` and `PivotedBandPassDiffusionKernel` use a uniform rank-1 `off_diagonal_floor` instead of zero-time `(beta_floor, 0.0) = I`, ensuring strictly positive off-diagonal similarity support.
  - Supports positive-coefficient multi-scale heat combinations via `multiscale_heat` / `multiscale_heat_terms`.
- **Core Architectural Rules**:
  1. **Builder Pattern**: All kernel and distance structs must be instantiated using the Builder pattern (e.g., `DiffusionKernelBuilder`). Builders must return `Result<T, String>` to ensure explicit error handling instead of silent fallbacks.
  2. **Explicit RNG**: Never instantiate internal `thread_rng()` within library functions. Random Number Generators (`&mut R` where `R: rand::Rng`) must be explicitly passed into builders/methods to guarantee determinism.
  3. **No Trace Estimation**: The Hutchinson method for trace estimation has been strictly removed. Avoid re-introducing randomized diagonal estimators.
  4. **Separation of Concerns (Kernel vs. Distance)**:
     - The `Kernel` trait must remain lightweight, requiring only `get(i, j)` and `n()`. No distance calculation logic should exist within the kernel implementations themselves.
     - Distance conversion is strictly the responsibility of unified distance matrices.
  5. **Unified Distance Matrices**: 
     - **`NegLogSimDistance`**: Replaces previous matrix variants (like `HeatGeodesicDistanceMatrix`). Computes standard pairwise geodesic distances using the formula: $\alpha (\log(K_{ii} + \beta) - 2\log(K_{ij} + \beta) + \log(K_{jj} + \beta))$.
     - **`NegLogDistance`**: Used with the `PivotedKernel` trait for single-source unit vector calculations. Computes $-\alpha \log(K_{ij} + \beta)$.
     - Default parameters are $\alpha = 1.0, \beta = 0.0$.

## Community Detection

- **V6 Pure-Rust Clustering (`egraph-clustering`, `crates/cluster/`)**:
  - Operates directly on flat `CsrGraph` and `egraph-linalg` primitives without `HashMap` in inner loops or aggregation; excludes `InfoMap` per `V6_RELEASE_PLAN.md`.
  - **`kmeans_plus_plus` (`KMeansConfig`, `KMeansResult`)**: Deterministic $D^2$ weighted centroid seeding via `DeterministicRng`, multi-start `n_init` keeping lowest inertia, and canonical `0..k` labels.
  - **`spectral_clustering` (`SpectralClusteringConfig`)**: Ng-Jordan-Weiss spectral clustering via `smallest_sym_laplacian_eigenpairs`, two-pass Modified Gram-Schmidt orthogonalization, row normalization, and `kmeans_plus_plus`.
  - **`louvain` (`LouvainConfig`, `LouvainResult`)**: Two-phase multi-level modularity optimization using `VisitStamp` accumulators and compact `WeightedLevelGraph` super-graph aggregation.
  - **`label_propagation` (`LabelPropagationConfig`)**: Asynchronous Fisher-Yates shuffled sweeps (`DeterministicRng::shuffle`) with `VisitStamp`-guarded neighbor vote counting.
  - **`coarsen_graph` / `project_clustered_layout` (`CoarsenedGraph`)**: Hash-free label compaction into super-graphs and deterministic multilevel layout projection.
- **Legacy v5 (`petgraph-clustering`)**:
  - Trait-based `CommunityDetection<G>` over `petgraph` graphs (Louvain, Label Propagation, Spectral Clustering, InfoMap) and custom coarsening merge functions.

## Layering Algorithms

- **V6 `CsrGraph` Layering (`egraph-layout`, `crates/layout/src/layering.rs`)**:
  - Uses iterative 3-color DFS (`cycle_edges`, `remove_cycles`) to detect and reverse back-edges in $O(V + E)$ time without recursion stack overflow on deep graphs, and `assign_layers(graph, method)` supporting `LayeringMethod::{LongestPath, NetworkSimplex}` without `petgraph` or `HashMap`.
- **Legacy v5 (`petgraph-algorithm-layering`)**:
  - Trait-based interface `LayeringAlgorithm<N, E, Ix: IndexType>` over `petgraph` directed graphs.

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
