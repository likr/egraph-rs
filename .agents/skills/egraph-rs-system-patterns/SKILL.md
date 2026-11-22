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

- **Unified Trait-Based Interface**:
  ```rust
  trait CommunityDetection<G> {
      fn detect_communities(&self, graph: G) -> HashMap<G::NodeId, usize>;
  }
  ```
- **Algorithms**: Louvain, Label Propagation, Spectral Clustering, InfoMap.
- **Graph Coarsening**: Custom merge functions for nodes and edges.

## Layering Algorithms

- **Unified Trait-Based Interface**:
  ```rust
  trait LayeringAlgorithm<N, E, Ix: IndexType> {
      fn assign_layers(&self, graph: &Graph<N, E, Directed, Ix>) -> HashMap<NodeIndex<Ix>, usize>;
  }
  ```
- **Cycle Handling**: Detection and removal functions to ensure directed graphs are acyclic before layering.

## Triangulation

- **Delaunay Triangulation**:
  - Uses the `spade` library to calculate 2D Euclidean triangulation.

## Spatial Structures & Quality Metrics

- **`SweepLineCrossings` vs `sampled_crossing_number` (`egraph-spatial` / `egraph-metrics`)**:
  - `SweepLineCrossings` runs in exact $O((|E| + K)\log |E|)$ time using a compact `FastU64Hasher`-backed `PairSet` (`pack_pair`) and zero `Vec<CrossingPair>` allocation on count/angle paths, remaining fast up to $K \sim 10^6\text{--}10^7$ crossings.
  - On ultra-dense hairball layouts where $K \sim 10^8$ or larger (e.g., `GION_3`, `bcsstk31`), exact enumeration is lower-bounded by $\Omega(K)$; use $O(M)$ `sampled_crossing_number` (deterministic `DeterministicRng`) instead of exact sweep-line counting.

## Spectral Linear Algebra & Neighborhood Primitives (`egraph-linalg` & `egraph-neighborhood`)

- **`ChebyshevWorkspace` 16-Column Tiled Recurrence & Tail Truncation (`egraph-linalg`)**:
  - Evaluates $\sum_k c_k T_k(\tilde{L}) V$ using cache-line-aligned `CHEBYSHEV_TILE_COLS = 16` (`{16, 8, 4, 1, tail}`) with `as_chunks::<W>()` and pre-validated CSR indices, eliminating per-nonzero `Option` bounds branches without `unsafe`.
  - `effective_chebyshev_len(coeffs, 1e-12)` trims trailing coefficients whose $\ell_1$ sum is $\le 10^{-12}$, provably bounding spectral operator-norm truncation error by $\le 10^{-12}$ while cutting coarse heat/JL passes significantly.
- **Shifted $\tilde{L}$ Reuse & Single-Pass DoG-Varadhan (`egraph-linalg` / `egraph-neighborhood`)**:
  - `ChebyshevWorkspace::apply_batch_dual`, `compute_kernel_jl_embedding_shifted`, and `compute_dual_jl_squared_norms_shifted` operate directly on the zero-diagonal shifted Laplacian $\tilde{L} = \frac{2}{\lambda_{\max}} L_{\mathrm{sym}} - I$ and generate Rademacher probes per tile without allocating $(N, d_{\mathrm{JL}})$ matrices when only row squared norms are needed.
  - `compute_regularized_dog_varadhan_landmarks` constructs $L_{\mathrm{sym}}$, $\tilde{L}$, and MaxMin landmarks once and fuses fine DoG + coarse scaffold (`deg_c = max(d, 75)` for explicit `degree`, `max(deg_fine, adaptive(t_coarse))` otherwise) across shared degrees `0..=deg_fine`.
- **Unweighted `CsrGraph<W>` Scalar Convention**:
  - `egraph-linalg` functions require `W: Scalar` on `CsrGraph<W>`, whereas `egraph-neighborhood::extract_1hop_2hop_csr` emits canonical unweighted `CsrGraph<f64>` (`weights: None`) graphs compatible across both `egraph-linalg` and Python/WASM bindings.
