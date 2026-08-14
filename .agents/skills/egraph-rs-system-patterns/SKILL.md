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

### MDS (Multidimensional Scaling) & Pivot MDS
Dimension reduction layout algorithms supporting unified distance matrix inputs.
- **Location**: `crates/layout/mds/`
- **Distance Trait**: `PivotMds` and `ClassicalMds` accept any matrix implementing `petgraph_linalg_kernel::Distance<N2, S> + ?Sized` (including `KernelDistance`, `NegLogDistance`, `NegLogSimDistance`, `PivotedNegLogDistance`, `FullDistanceMatrix`, `PivotedDistanceMatrix`).
- **Constructor Patterns**:
  - `new_with_distance_matrix`: Automatically uses default node indices (`petgraph::graph::node_index`).
  - `new_with_distance_matrix_and_indices`: Accepts explicit node index slice `&[N2]` for custom indexing.

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

### tsNET & BH-tsNET
t-SNE based graph layout algorithms optimizing KL-divergence, compression, and node repulsion.
- **Location**: `crates/layout/ts-net/`
- **tsNET**:
  - Distance Trait: Generic over `petgraph_linalg_kernel::Distance<N, S> + ?Sized`.
  - Builder Pattern: Instantiated via `TsNetBuilder<S>` returning `Result<TsNet<S>, String>`.
  - 3-Stage Dynamic Optimization: Early exaggeration, early compression ($\lambda_c = 1.2$), and final refinement ($(\lambda_{KL}, \lambda_c, \lambda_r) = (1.0, 0.01, 0.6)$).
- **BH-tsNET**:
  - Barnes-Hut accelerated $O(N \log N)$ algorithm for large graphs.
  - C0: $O(N)$ Partial BFS for $k$-nearest neighbors with random tie-breaking.
  - C1 & C2: 2D Quadtree spatial approximation for KL divergence repulsion and entropy gradients.
  - Builder Pattern: Instantiated via `BhTsNetBuilder<S>` returning `Result<BhTsNet<S>, String>`.


### Kernel-SGD & Diffusion Kernels
Diffusion kernel-based computations for graph distances and embeddings.
- **Location**: `crates/linalg/diffusion-kernel/` and `crates/linalg/distance/`
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
