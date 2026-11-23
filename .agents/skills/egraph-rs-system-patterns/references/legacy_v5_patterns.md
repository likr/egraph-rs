# Legacy v5 (`petgraph`-Based) Patterns Reference

This reference preserves legacy v5 (`petgraph-*`) architecture and solver details moved out of [`SKILL.md`](../SKILL.md) to keep the main v6 skill concise. Legacy crates remain opt-in compatibility targets (`petgraph-*` workspace crates, `egraph-wasm --features legacy`, and legacy Python wrappers).

## Legacy Architecture & Design

- **Modular Legacy Crate Structure**:
  - `algorithm` (`connected-components`, `shortest-path`, `triangulation`, `layering`)
  - `clustering` (`petgraph-clustering`, now backed by `egraph-clustering` primitives)
  - `dataset` (graph dataset loaders)
  - `drawing` (Euclidean, Spherical, Hyperbolic, Torus)
  - `edge-bundling` (force-directed edge bundling)
  - `layout` (SGD, MDS, Stress-Majorization, Kamada-Kawai, Omega, overwrap-removal, separation-constraints, ts-net, gumap)
  - `linalg` (`RdMds` - Resistance-distance MDS for spectral embeddings; `petgraph-linalg-kernel`)
  - `quality-metrics` (layout evaluation)
  - Hardened edge-case validation across `quality-metrics` (`NaN`/collinear `crossing_number`, degenerate `aspect_ratio`, edgeless `neighborhood_preservation`), `WeightedEdgeLength` (indexed by edge ID with fallible lookup), and PyO3 bindings (`PyResult`/`PyValueError` instead of panics).

- **Key Patterns**:
  - Builder, Strategy, Adapter, Visitor, Factory Methods
  - Trait-based Interfaces (`CommunityDetection<G>`, `LayeringAlgorithm<N, E, Ix: IndexType>`)
  - Composition over inheritance

## Legacy Layout Algorithms

### SGD (Stochastic Gradient Descent)
Force-directed layout with a concrete `Sgd<S>` struct rather than a trait-based design for efficiency.
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
- **Scheduler Integration**: Custom `Scheduler<S>` trait (`run`, `step`, `is_finished`) and automatic learning-rate bounds derived from weight distributions.
- **One-Sided Hinge Repulsion & Barrier Sparse SGD**:
  - `Sgd::new_with_hinge_pairs(springs, hinges)` pairs standard two-sided spring pairs with one-sided hinge repulsion pairs that exert force strictly when $0 < \|x_i - x_j\| < d_{\mathrm{barrier}}$.
  - PyO3 NumPy constructors: `Sgd.from_numpy_pairs`, `Sgd.from_numpy_hinge_pairs`, and `Sgd.from_numpy_pairs_with_hinges`.
  - `SparseSgd::barrier()` builder configures 3-tier barrier construction (`calibrate_one_hop`, `max_2hop_per_node`, `hinge_repulsion`, `delta_rel`, `barrier_sym`, `omega_1`, `omega_2`, `gamma_non2`, `min_ratio`, `eps_dist`, `seed`) via `build_barrier_from_landmark_tables`.
- **Fallible Validation**: `FullSgd` and `StressMajorization` provide fallible square-matrix validation and `try_*` APIs with $n \le 1$ and `NaN`/`Inf` guards.

### MDS (Multidimensional Scaling) & Pivot MDS
- **Location**: `crates/layout/mds/`
- **Distance Trait**: `PivotMds` and `ClassicalMds` accept `petgraph_linalg_kernel::Distance<N2, S> + ?Sized` (`KernelDistance`, `NegLogDistance`, `NegLogSimDistance`, `PivotedNegLogDistance`, `FullDistanceMatrix`, `PivotedDistanceMatrix`).
- **Constructor Patterns**: `new_with_distance_matrix` (default node indices) and `new_with_distance_matrix_and_indices`.
- **Scaling & Validation**: `PivotMds` computes $(n/k)^{1/4} C V \mathrm{diag}(e^{-1/4}) \approx U \Sigma^{1/2}$ with zero-eigenvalue and disconnected-graph finite-coordinate guards; `ClassicalMds` validates square matrices fallibly.

### RdMds (Resistance-distance MDS)
- Computes spectral embeddings using the graph Laplacian (`crates/linalg/rdmds/`, configuring `d`, `shift`, `eigenvalue_max_iterations`, `cg_max_iterations`, `eigenvalue_tolerance`, `cg_tolerance`).

### Omega
- **Location**: `crates/layout/omega/` (`k` random pairs per node, `min_dist`).

### tsNET, BH-tsNET, FIt-tsNET & L-tsNET (`crates/layout/ts-net/`)
- **Shared Acceleration & Validation**: `partial_bfs` reuses `visited_stamp` and layer buffers (`O(N * k * avg_deg)`). Supports per-node adaptive perplexity and variable-length CSR $k$-NN (`try_run_with_csr_knn_adaptive`, PyO3 `run_with_knn` and `run_with_knn_adaptive`).
- **tsNET**: Generic over `petgraph_linalg_kernel::Distance<N, S> + ?Sized`; `TsNetBuilder<S>` returning `Result<TsNet<S>, String>`; 3-stage dynamic optimization.
- **BH-tsNET**: Quadtree Barnes-Hut repulsion + entropy gradients.
- **FIt-tsNET**: 2D FFT polynomial grid interpolation for repulsion + Quadtree entropy gradient.
- **L-tsNET**: $O(N)$ 2D FFT for both repulsion and entropy gradient (`h_1, h_2, h_3` matching `BhTsNet`/`FitTsNet` sign convention).

### GUMAP (`crates/layout/gumap/`)
- Generic over `petgraph_linalg_kernel::Distance<N, S> + ?Sized`; supports `run_with_knn` and CSR `run_with_knn_csr` (`try_run_with_csr_knn`) with deterministic edge ordering; fuzzy set union symmetrization ($p_{ij} = p_{j|i} + p_{i|j} - p_{j|i} p_{i|j}$); Levenberg-Marquardt $(a, b)$ fit; SGD optimization via `GuMapBuilder<S>`.

### Kernel-SGD & Diffusion Kernels (`crates/linalg/kernel/`)
- **BandPass & Multiscale Kernels**: `BandPassDiffusionKernel` and `PivotedBandPassDiffusionKernel` use a uniform rank-1 `off_diagonal_floor` instead of zero-time identity, and support `multiscale_heat` / `multiscale_heat_terms`.
- **Core Rules**: Builder pattern returning `Result<T, String>`, explicit `&mut R` RNG, no Hutchinson trace estimation, lightweight `Kernel` trait separated from `NegLogSimDistance` / `NegLogDistance` ($\alpha = 1.0, \beta = 0.0$).

### Legacy Community Detection & Layering
- **Legacy v5 (`petgraph-clustering`)**: Trait-based `CommunityDetection<G>` over `petgraph` graphs (Louvain, Label Propagation, Spectral Clustering, InfoMap) and custom coarsening merge functions.
- **Legacy v5 (`petgraph-algorithm-layering`)**: Trait-based interface `LayeringAlgorithm<N, E, Ix: IndexType>` over `petgraph` directed graphs.
